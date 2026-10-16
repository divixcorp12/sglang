"""The gather-time lane head: which of a layer's miss lanes the GPU should take, scored by how likely each routed
expert is to be routed again by the next verify.

The fused route plan sorts a layer's miss lanes by an int64 key per expert, highest first, and the CPU experts take
the tail (expert_route_plan.cuh, lease_device.cuh type_lanes). Today the key is the residency ranking
(GpuResidencyUpdater.miss_keys). With ``SGLANG_OPT_LANE_HEAD_PATH`` set, this head writes the key instead, from the
layer's gate input: a per-layer PCA of the token routing the expert, the dense features of the offline study
(analysis/dsv41-drive/dspark/next_verify_head.py, variant B) and a per-expert bias and direction. The lanes the GPU
copies are inserted into the VRAM hot cache; the lanes the CPU computes are not, so the key decides what the cache
keeps.

The head serves what it was trained on. Training standardized the features (``feature_mean``/``feature_scale``,
``z_scale``) and the file keeps them unfolded, so serving computes ``(f - mean) / scale`` exactly as
``next_verify_head.predict`` does. A feature constant in training (the draft odds: no B0 block captured draft
confidence) has scale 1e9 and is fed the value training saw (``LaneHeadMeta.constant_features``), which makes its
term exactly zero. The score and age features are the training extraction's, not the residency updater's: a
per-request insert score that counts verify routes only, decays by ``score_decay ** live`` per verify and starts at
zero, and the verifies since the expert was last routed in this request. The head keeps that state itself
(:class:`LaneHead`), updates it in the key kernel after the keys are written, and is reset by every prefill.

Key per routed expert e: ``P = sigmoid(logit)``, ``key = bits(P) * 2^16 + (experts - 1 - e)`` (P's fp32 bit
pattern, the shape of residency_rank_keys). :class:`LaneHeadReference` is the Python definition the kernel
(kernels/ops/moe/lane_head.py) is tested against.

File (safetensors, written by analysis/dsv41-drive/dspark/export_lane_head.py): the tensors of
:data:`TENSOR_DTYPES` and one JSON metadata entry (:class:`LaneHeadMeta`). Serving reads it with
:func:`read_lane_head_file` only (no pickle).
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LANE_HEAD_FORMAT_VERSION = 1
_META_KEY = "lane_head"
# The dense features, in the kernel's order (next_verify_head.DENSE).
FEATURES = (
    "cnt1", "cnt2", "cnt3p", "pos0", "pos1", "pos2", "pos3", "pos4", "pos5", "gate_w", "log_score", "score0",
    "log_age", "first_touch", "resident", "live", "odds",
)
ODDS = FEATURES.index("odds")
# next_verify_head.constant_safe: a feature whose training deviation was ~0 is scaled by this, zeroing its term.
CONSTANT_SCALE = 1e9
# Gather widths the kernel takes: at most this many tokens, and 64 routes (the fused plan's dedup bound).
MAX_TOKENS = 16
MAX_ROUTES = 64
# The PCA kernel splits the hidden dimension over this many blocks per (token, 16 ranks).
PCA_SPLITS = 8
PROJ_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TENSOR_DTYPES = {
    "pca_mean": (torch.float32,),  # [layers, hidden]
    "pca_proj": tuple(PROJ_DTYPES.values()),  # [layers, hidden, rank], LaneHeadMeta.pca_proj_dtype
    "feature_mean": (torch.float32,),  # [layers, features]
    "feature_scale": (torch.float32,),  # [layers, features], constant_safe applied
    "z_scale": (torch.float32,),  # [layers, rank]
    "expert_bias": (torch.float32,),  # [layers, experts]
    "dense_w": (torch.float32,),  # [layers, features]
    "dense_b": (torch.float32,),  # [layers]
    "z_w": (torch.float32,),  # [layers, rank]
    "expert_u": (torch.float32,),  # [layers, experts, rank]
}


class LaneHeadMeta(msgspec.Struct, frozen=True):
    format_version: int
    layer_ids: tuple[int, ...]
    hidden: int
    rank: int
    experts: int
    features: tuple[str, ...]
    # "bf16" (26 MB at 40 layers, rank 64) or "fp32" (the training's own projection, bit for bit).
    pca_proj_dtype: str
    # The score feature's per-token decay (next_verify_head.DECAY), also the training captures' hot-cache insert decay;
    # a launch with another SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY is refused (check_serving_matches_training).
    score_decay: float
    # Features constant in training, with the value training saw; serving feeds exactly that value.
    constant_features: dict[str, float]
    training: dict[str, Any]

    @property
    def odds(self) -> float:
        return float(self.constant_features["odds"])


class LaneHeadFile(msgspec.Struct, frozen=True):
    path: str
    tensors: dict[str, torch.Tensor]
    meta: LaneHeadMeta

    @property
    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())


def write_lane_head_file(path: str, *, tensors: dict[str, torch.Tensor], meta: LaneHeadMeta) -> None:
    contiguous = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(contiguous, path, metadata={_META_KEY: msgspec.json.encode(meta).decode()})


def read_lane_head_meta(path: str) -> LaneHeadMeta:
    """The metadata alone, for the hot-cache budget before the tensors are loaded."""
    try:
        with safe_open(path, framework="pt") as f:
            raw = (f.metadata() or {}).get(_META_KEY)
    except Exception as e:
        raise ValueError(f"lane head file {path!r} is missing or unreadable: {e}") from e
    if raw is None:
        raise ValueError(f"lane head file {path!r} has no {_META_KEY!r} metadata entry")
    try:
        meta = msgspec.json.decode(raw, type=LaneHeadMeta)
    except msgspec.DecodeError as e:
        raise ValueError(f"lane head file {path!r} has malformed metadata: {e}") from e
    if meta.format_version != LANE_HEAD_FORMAT_VERSION:
        raise ValueError(
            f"lane head file {path!r} has format_version {meta.format_version}, this build reads "
            f"{LANE_HEAD_FORMAT_VERSION}"
        )
    if tuple(meta.features) != FEATURES:
        raise ValueError(f"lane head file {path!r} has features {meta.features}, this build computes {FEATURES}")
    if "odds" not in meta.constant_features:
        # The gather has no draft confidence; a head that learned from it cannot be served here.
        raise ValueError(f"lane head file {path!r}: the draft odds were not constant in training")
    if meta.pca_proj_dtype not in PROJ_DTYPES:
        raise ValueError(f"lane head file {path!r}: pca_proj_dtype {meta.pca_proj_dtype!r} is not one of {list(PROJ_DTYPES)}")
    return meta


def lane_head_device_bytes(meta: LaneHeadMeta) -> int:
    """The bytes :class:`LaneHead` puts on the device for a file with this metadata (weights plus the keys, the
    per-request score/age state and the scratch), so the hot-cache budget can be reduced by it before sizing."""
    layers, hidden, rank, experts, features = len(meta.layer_ids), meta.hidden, meta.rank, meta.experts, len(FEATURES)
    proj_bytes = PROJ_DTYPES[meta.pca_proj_dtype].itemsize
    weights = 4 * (
        layers * hidden  # pca_mean
        + 3 * layers * features  # feature_mean, feature_scale, dense_w
        + 2 * layers * rank  # z_scale, z_w
        + layers * experts  # expert_bias
        + layers  # dense_b
        + layers * experts * rank  # expert_u
    ) + proj_bytes * layers * hidden * rank
    state = (
        layers * experts * (8 + 4 + 4)  # keys, score, last
        + layers * (4 + 8)  # verify_ct, gather_ct
        + PCA_SPLITS * MAX_TOKENS * rank * 4  # z partials
        + (MAX_TOKENS + 1) * 4  # decay powers
    )
    return weights + state


def read_lane_head_file(
    path: str,
    *,
    layer_ids: Optional[list[int]] = None,
    hidden: Optional[int] = None,
    experts: Optional[int] = None,
) -> LaneHeadFile:
    """Load and validate; ValueError names the file and the first defect. ``layer_ids``/``hidden``/``experts`` are
    the launch's, checked against the file's when given."""
    meta = read_lane_head_meta(path)
    try:
        tensors = load_file(path)
    except Exception as e:
        raise ValueError(f"lane head file {path!r} is unreadable: {e}") from e
    layers, hidden_f, rank, experts_f, features = len(meta.layer_ids), meta.hidden, meta.rank, meta.experts, len(FEATURES)
    shapes = {
        "pca_mean": (layers, hidden_f),
        "pca_proj": (layers, hidden_f, rank),
        "feature_mean": (layers, features),
        "feature_scale": (layers, features),
        "z_scale": (layers, rank),
        "expert_bias": (layers, experts_f),
        "dense_w": (layers, features),
        "dense_b": (layers,),
        "z_w": (layers, rank),
        "expert_u": (layers, experts_f, rank),
    }
    extra = sorted(set(tensors) - set(TENSOR_DTYPES))
    if extra:
        raise ValueError(f"lane head file {path!r} has unexpected tensors {extra}")
    for name, dtypes in TENSOR_DTYPES.items():
        if name not in tensors:
            raise ValueError(f"lane head file {path!r} lacks tensor {name!r}")
        t = tensors[name]
        if t.dtype not in dtypes or tuple(t.shape) != shapes[name]:
            raise ValueError(
                f"lane head file {path!r}: {name} is {t.dtype} {tuple(t.shape)}, expected one of {dtypes} "
                f"{shapes[name]}"
            )
        if not torch.isfinite(t.float()).all():
            raise ValueError(f"lane head file {path!r}: {name} has a non-finite value")
    if tensors["pca_proj"].dtype != PROJ_DTYPES[meta.pca_proj_dtype]:
        raise ValueError(
            f"lane head file {path!r}: pca_proj is {tensors['pca_proj'].dtype}, the metadata says {meta.pca_proj_dtype}"
        )
    if rank % 16 != 0 or not 16 <= rank <= 128:
        raise ValueError(f"lane head file {path!r}: rank {rank} must be a multiple of 16 in [16, 128]")
    if experts_f > 512:
        raise ValueError(f"lane head file {path!r}: {experts_f} experts, the key kernel takes at most 512")
    if not (tensors["feature_scale"] > 0).all() or not (tensors["z_scale"] > 0).all():
        raise ValueError(f"lane head file {path!r}: a feature or PCA scale is not positive")
    if layer_ids is not None and tuple(layer_ids) != tuple(meta.layer_ids):
        raise ValueError(
            f"lane head file {path!r} covers layers {list(meta.layer_ids)}, the launch streams {list(layer_ids)}"
        )
    if hidden is not None and hidden != hidden_f:
        raise ValueError(f"lane head file {path!r} has hidden {hidden_f}, the router input is {hidden}")
    if experts is not None and experts != experts_f:
        raise ValueError(f"lane head file {path!r} has {experts_f} experts, the layers have {experts}")
    for name, value in meta.constant_features.items():
        if name not in FEATURES:
            raise ValueError(f"lane head file {path!r}: constant feature {name!r} is not a feature")
        col = FEATURES.index(name)
        if not torch.all(tensors["feature_scale"][:, col] >= CONSTANT_SCALE * 0.5):
            raise ValueError(f"lane head file {path!r}: constant feature {name!r} is not zeroed by its scale")
        if not torch.all(tensors["feature_mean"][:, col] == torch.tensor(value, dtype=torch.float32)):
            raise ValueError(
                f"lane head file {path!r}: constant feature {name!r} is fed {value}, but training's mean differs"
            )
    return LaneHeadFile(path=path, tensors=tensors, meta=meta)


def check_serving_matches_training(meta: LaneHeadMeta, *, insert_decay: float, top_prob_gate: float) -> None:
    """Refuse a launch whose features would come from a distribution training never saw.

    The training corpus served every verify at its full width (no DSpark top-probability gate: ``live`` is 6 on all
    but 26 of 45k rows, so a gated verify's ``live`` lands ~100 deviations off) and with the hot cache's insert decay
    equal to the score decay the features use (``score_decay``), which shaped the ``resident`` feature."""
    if top_prob_gate:
        raise ValueError(
            f"SGLANG_OPT_LANE_HEAD_PATH: the head was trained on ungated verifies; SGLANG_DSPARK_TOP_PROB_GATE="
            f"{top_prob_gate} cuts them (set it to 0 or retrain on gated captures)"
        )
    if abs(float(insert_decay) - meta.score_decay) > 1e-12:
        raise ValueError(
            f"SGLANG_OPT_LANE_HEAD_PATH: the head was trained with insert decay {meta.score_decay}; "
            f"SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY is {insert_decay}"
        )


# ---- the Python definition -----------------------------------------------------------------------------------------


def dense_features(
    cnt: int, first_pos: int, gate_w: float, score: float, age: int, resident: bool, live: int, odds: float
) -> np.ndarray:
    """One routed expert's raw dense features in :data:`FEATURES` order, computed as next_verify_head.extract_source
    does (numpy float32, the transcendental ones in double first). ``age`` is verifies since the expert was last
    routed in this request, negative for never."""
    f = np.zeros(len(FEATURES), dtype=np.float32)
    f[0 if cnt == 1 else 1 if cnt == 2 else 2] = 1.0
    f[3 + min(first_pos, 5)] = 1.0
    f[9] = gate_w
    f[10] = np.log1p(score)
    f[11] = 1.0 if score <= 0 else 0.0
    f[12] = np.log1p(age) if age >= 0 else 0.0
    f[13] = 1.0 if age < 0 else 0.0
    f[14] = 1.0 if resident else 0.0
    f[15] = live / 6.0
    f[16] = odds
    return f


def head_logits(tensors: dict[str, torch.Tensor], row: int, z: torch.Tensor, f: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
    """fp32 logits of rows ``(z, f, e)`` (raw PCA coordinates and raw features), in next_verify_head's op order:
    ``Head.forward`` on ``(z / zs, (f - fm) / fs, e)``."""
    zs = z / tensors["z_scale"][row]
    fs = (f - tensors["feature_mean"][row]) / tensors["feature_scale"][row]
    out = (
        tensors["expert_bias"][row][e]
        + torch.nn.functional.linear(fs, tensors["dense_w"][row].unsqueeze(0), tensors["dense_b"][row : row + 1]).squeeze(-1)
        + torch.nn.functional.linear(zs, tensors["z_w"][row].unsqueeze(0)).squeeze(-1)
    )
    return out + (tensors["expert_u"][row][e] * zs).sum(-1)


def plan_keys(p: torch.Tensor, e: torch.Tensor, experts: int) -> torch.Tensor:
    """int64 plan keys of probabilities ``p`` (fp32, >= 0) for experts ``e``: ``bits(P) << 16 | (experts-1-e)``, as
    residency_rank_keys. Nonnegative floats order like their bit patterns, so the keys order exactly as ``P``, down to
    the tiny probabilities a fixed-point scale would round to zero."""
    bits = (p.float() + 0.0).view(torch.int32).to(torch.int64)
    return bits * (1 << 16) + (experts - 1 - e.long())


class LaneHeadReference:
    """The fp32 Python definition of the head and its per-request state, on CPU. ``gather`` is one layer's call at
    one verify; ``reset`` is a prefill. The score is kept in double, as next_verify_head's extraction kept it."""

    def __init__(self, tensors: dict[str, torch.Tensor], meta: LaneHeadMeta):
        self.tensors = {name: t.cpu() for name, t in tensors.items()}
        self.meta = meta
        layers, experts = len(meta.layer_ids), meta.experts
        self.score = np.zeros((layers, experts), dtype=np.float64)
        self.last = np.full((layers, experts), -1, dtype=np.int64)
        self.verify_ct = np.zeros(layers, dtype=np.int64)

    def reset(self) -> None:
        self.score.fill(0.0)
        self.last.fill(-1)
        self.verify_ct.fill(0)

    def features(
        self,
        row: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x: torch.Tensor,
        resident: torch.Tensor,
        dead: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """The model inputs of one gather: experts ``e`` (first-appearance order), raw ``z`` [n, rank], raw ``f``
        [n, features], the per-expert route counts and the live token count. Reads the state; does not advance it."""
        ids = topk_ids.cpu().long().numpy()
        w = topk_weights.cpu().float().numpy()
        dead_rows = (
            np.zeros(ids.shape[0], dtype=bool) if dead is None else dead.cpu().reshape(-1)[: ids.shape[0]].numpy()
        )
        live_rows = [tok for tok in range(ids.shape[0]) if not dead_rows[tok] and (ids[tok] >= 0).any()]
        live = len(live_rows)
        cnt: dict[int, int] = {}
        best_w: dict[int, float] = {}
        best_tok: dict[int, int] = {}
        fpos: dict[int, int] = {}
        for tok in live_rows:
            for e, wt in zip(ids[tok].tolist(), w[tok].tolist()):
                if e < 0:
                    continue
                cnt[e] = cnt.get(e, 0) + 1
                if e not in best_w or wt > best_w[e]:
                    best_w[e], best_tok[e] = wt, tok
                if e not in fpos:
                    fpos[e] = tok
        mean = self.tensors["pca_mean"][row]
        proj = self.tensors["pca_proj"][row].float()
        z_all = (x.cpu().float() - mean) @ proj
        resident = resident.cpu()
        t = int(self.verify_ct[row])
        experts = list(cnt)
        f = np.stack(
            [
                dense_features(
                    cnt[e],
                    fpos[e],
                    best_w[e],
                    float(self.score[row, e]),
                    t - int(self.last[row, e]) if self.last[row, e] >= 0 else -1,
                    bool(resident[e]),
                    live,
                    self.meta.odds,
                )
                for e in experts
            ]
        ) if experts else np.zeros((0, len(FEATURES)), dtype=np.float32)
        z = z_all[[best_tok[e] for e in experts]] if experts else torch.zeros((0, self.meta.rank))
        return {"e": experts, "z": z, "f": torch.from_numpy(f), "cnt": cnt, "live": live}

    def gather(
        self,
        row: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x: torch.Tensor,
        resident: torch.Tensor,
        dead: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(keys, p)``, int64/float32 ``[experts]``, for one layer's gather (unrouted experts: key 0, p 0), then
        the state update the key kernel makes: score decays by ``score_decay ** live`` and adds the route counts,
        the routed experts' last verify becomes this one."""
        experts = self.meta.experts
        inp = self.features(row, topk_ids, topk_weights, x, resident, dead)
        keys = torch.zeros(experts, dtype=torch.int64)
        p_out = torch.zeros(experts, dtype=torch.float32)
        if inp["e"]:
            e = torch.tensor(inp["e"], dtype=torch.int64)
            with torch.no_grad():
                p = torch.sigmoid(head_logits(self.tensors, row, inp["z"], inp["f"], e))
            p_out[e] = p
            keys[e] = plan_keys(p, e, experts)
        d = self.meta.score_decay ** inp["live"]
        self.score[row] *= d
        t = int(self.verify_ct[row])
        for e, c in inp["cnt"].items():
            self.score[row, e] += c
            self.last[row, e] = t
        self.verify_ct[row] += 1
        return keys, p_out


# ---- device state ------------------------------------------------------------------------------------------------


class LaneHead:
    """One launch's head on the device: the weights, the per-layer keys the route plan reads, and the per-request
    state of the score and age features (the key kernel advances it; :meth:`reset` starts a request)."""

    def __init__(self, head: LaneHeadFile, device: torch.device | str):
        self.file = head
        self.meta = head.meta
        self.device = torch.device(device)
        t = head.tensors
        self.weights = {name: t[name].to(self.device).contiguous() for name in TENSOR_DTYPES}
        layers, experts, rank = len(self.meta.layer_ids), self.meta.experts, self.meta.rank
        self.keys = torch.zeros((layers, experts), dtype=torch.int64, device=self.device)
        self.score = torch.zeros((layers, experts), dtype=torch.float32, device=self.device)
        self.last = torch.full((layers, experts), -1, dtype=torch.int32, device=self.device)
        self.verify_ct = torch.zeros(layers, dtype=torch.int32, device=self.device)
        # Gathers that wrote keys, per layer, over the launch (never reset): the metrics trace's proof the head ran.
        self.gather_ct = torch.zeros(layers, dtype=torch.int64, device=self.device)
        # One scratch for every layer: the gathers run in order on one stream.
        self.z_partial = torch.zeros((PCA_SPLITS, MAX_TOKENS, rank), dtype=torch.float32, device=self.device)
        self.decay_pow = torch.tensor(
            [self.meta.score_decay**live for live in range(MAX_TOKENS + 1)], dtype=torch.float32, device=self.device
        )
        self.device_bytes = lane_head_device_bytes(self.meta)

    def describe(self) -> str:
        return (
            f"lane head active: file={self.file.path} layers={len(self.meta.layer_ids)} rank={self.meta.rank} "
            f"pca_proj={self.weights['pca_proj'].dtype} device_bytes={self.device_bytes} odds={self.meta.odds} "
            f"training={self.meta.training}"
        )

    def reset(self) -> None:
        """A request starts: every expert's score is zero and none has been routed."""
        self.score.zero_()
        self.last.fill_(-1)
        self.verify_ct.zero_()

    def launch(
        self,
        row: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x: torch.Tensor,
        expert_to_slot: torch.Tensor,
        dead: Optional[torch.Tensor],
    ) -> None:
        """Write ``keys[row]`` for this gather's routed experts and advance the row's state, on the current stream
        (graph-capturable)."""
        from sglang.kernels.ops.moe.lane_head import lane_head_keys

        w = self.weights
        lane_head_keys(
            topk_ids,
            topk_weights,
            x,
            dead,
            w["pca_mean"][row],
            w["pca_proj"][row],
            w["feature_mean"][row],
            w["feature_scale"][row],
            w["z_scale"][row],
            w["expert_bias"][row],
            w["dense_w"][row],
            w["dense_b"][row : row + 1],
            w["z_w"][row],
            w["expert_u"][row],
            expert_to_slot,
            self.score[row],
            self.last[row],
            self.verify_ct[row : row + 1],
            self.gather_ct[row : row + 1],
            self.z_partial,
            self.decay_pow,
            self.keys[row],
            self.meta.odds,
        )


class LaneHeadGather:
    """The per-layer hook ExpertStreamer._gather_graph calls before the fused plan: launches the head on the gate
    input Exl3MoEMethod._apply_graph staged for the CPU experts (``row_backend.cpu_input``, and the verify's dead
    rows in ``row_backend.verify_dead``)."""

    def __init__(self, head: LaneHead, updater, row: int, streamer):
        self.head = head
        self.updater = updater
        self.row = row
        self.streamer = streamer
        self.experts = head.meta.experts

    def __call__(self, topk_ids: torch.Tensor) -> None:
        backend = self.streamer.row_backend
        cpu_input = getattr(backend, "cpu_input", None)
        if cpu_input is None:
            raise RuntimeError(
                f"lane head: layer row {self.row} gathered without its input; Exl3MoEMethod._apply_graph stages "
                "cpu_input for the CPU experts, which the head needs"
            )
        x, topk_weights = cpu_input
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.head.meta.hidden:
            raise RuntimeError(
                f"lane head: the gate input is {x.dtype} {tuple(x.shape)}, the head was trained on bf16 "
                f"[tokens, {self.head.meta.hidden}]"
            )
        self.head.launch(
            self.row,
            topk_ids,
            topk_weights.reshape(topk_ids.shape),
            x.reshape(topk_ids.shape[0], -1),
            self.updater.mapping[self.row, : self.experts],
            getattr(backend, "verify_dead", None),
        )
