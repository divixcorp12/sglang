"""The gather-time lane head: which of a layer's miss lanes the GPU should take, scored by how likely each routed
expert is to be routed again by the next verify.

The fused route plan sorts a layer's miss lanes by an int64 key per expert, highest first, and the CPU experts take
the tail (expert_route_plan.cuh, lease_device.cuh type_lanes). Today the key is the residency ranking
(GpuResidencyUpdater.miss_keys). With ``SGLANG_OPT_LANE_HEAD_PATH`` set, this head writes the key instead, from the
layer's gate input: a per-layer PCA of the token routing the expert, the dense features of the offline study
(analysis/dsv41-drive/dspark/next_verify_head.py, variant B) and a per-expert bias and direction. The lanes the GPU
copies are inserted into the VRAM hot cache; the lanes the CPU computes are not, so the key decides what the cache
keeps.

File (safetensors, written by analysis/dsv41-drive/dspark/export_lane_head.py): per layer the PCA mean and
projection, the expert biases and directions, the dense weights with their standardization folded in, and one JSON
metadata entry (:class:`LaneHeadMeta`). Serving reads it with :func:`read_lane_head_file` only (no pickle).

Key per routed expert e: ``P = sigmoid(expert_bias[e] + bias + dense_w . f + z_w . z + expert_u[e] . z)`` with
``z = (x[i*] - pca_mean) @ pca_proj`` for the token ``i*`` routing e with the largest gate weight, and
``key = int64(P * 2^31) * 2^16 + (experts - 1 - e)``, the tie-break shape of residency_rank_keys.
:func:`reference_keys` is the fp32 Python definition the kernel (kernels/ops/moe/lane_head.py) is tested against.
"""

from __future__ import annotations

import math
from typing import Any

import msgspec
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
KEY_SCALE = float(1 << 31)
TENSOR_DTYPES = {
    "pca_mean": torch.float32,  # [layers, hidden]
    "pca_proj": torch.bfloat16,  # [layers, hidden, rank]
    "expert_bias": torch.float32,  # [layers, experts]
    "expert_u": torch.float32,  # [layers, experts, rank]
    "bias": torch.float32,  # [layers]
    "dense_w": torch.float32,  # [layers, len(FEATURES)]
    "z_w": torch.float32,  # [layers, rank]
}


class LaneHeadMeta(msgspec.Struct, frozen=True):
    format_version: int
    layer_ids: tuple[int, ...]
    hidden: int
    rank: int
    experts: int
    features: tuple[str, ...]
    # The insert score the features read is GpuResidencyUpdater.insert_scores, decayed by this per token.
    score_decay: float
    # Features that were constant in training, with the value training saw; serving feeds exactly that value (their
    # folded weight is zero either way). The draft odds are one: no capture with draft confidence trained the head.
    constant_features: dict[str, float]
    training: dict[str, Any]

    @property
    def odds(self) -> float:
        return float(self.constant_features.get("odds", 1.0))


class LaneHeadFile(msgspec.Struct, frozen=True):
    path: str
    tensors: dict[str, torch.Tensor]
    meta: LaneHeadMeta

    @property
    def device_bytes(self) -> int:
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
            f"lane head file {path!r} has format_version {meta.format_version}, this build reads {LANE_HEAD_FORMAT_VERSION}"
        )
    if tuple(meta.features) != FEATURES:
        raise ValueError(f"lane head file {path!r} has features {meta.features}, this build computes {FEATURES}")
    return meta


def lane_head_device_bytes(meta: LaneHeadMeta) -> int:
    """The bytes :func:`LaneHead.load` puts on the device for a file with this metadata (weights plus the per-layer
    keys and last-routed buffers), so the hot-cache budget can be reduced by it before the cache is sized."""
    layers, hidden, rank, experts = len(meta.layer_ids), meta.hidden, meta.rank, meta.experts
    weights = (
        layers * hidden * 4  # pca_mean
        + layers * hidden * rank * 2  # pca_proj bf16
        + layers * experts * 4  # expert_bias
        + layers * experts * rank * 4  # expert_u
        + layers * 4  # bias
        + layers * len(FEATURES) * 4  # dense_w
        + layers * rank * 4  # z_w
    )
    state = layers * experts * 8 + layers * experts * 4 + layers * 8 * rank * 4  # keys, last_routed, z scratch
    return weights + state


def read_lane_head_file(path: str, *, layer_ids: list[int] | None = None, hidden: int | None = None,
                        experts: int | None = None) -> LaneHeadFile:
    """Load and validate; ValueError names the file and the first defect. ``layer_ids``/``hidden``/``experts`` are
    the launch's, checked against the file's when given."""
    meta = read_lane_head_meta(path)
    try:
        tensors = load_file(path)
    except Exception as e:
        raise ValueError(f"lane head file {path!r} is unreadable: {e}") from e
    layers, hidden_f, rank, experts_f = len(meta.layer_ids), meta.hidden, meta.rank, meta.experts
    shapes = {
        "pca_mean": (layers, hidden_f), "pca_proj": (layers, hidden_f, rank), "expert_bias": (layers, experts_f),
        "expert_u": (layers, experts_f, rank), "bias": (layers,), "dense_w": (layers, len(FEATURES)), "z_w": (layers, rank),
    }
    for name, dtype in TENSOR_DTYPES.items():
        if name not in tensors:
            raise ValueError(f"lane head file {path!r} lacks tensor {name!r}")
        t = tensors[name]
        if t.dtype != dtype or tuple(t.shape) != shapes[name]:
            raise ValueError(f"lane head file {path!r}: {name} is {t.dtype} {tuple(t.shape)}, expected {dtype} {shapes[name]}")
    for name, t in tensors.items():
        if not torch.isfinite(t.float()).all():
            raise ValueError(f"lane head file {path!r}: {name} has a non-finite value")
    if rank % 16 != 0 or rank < 16:
        raise ValueError(f"lane head file {path!r}: rank {rank} must be a positive multiple of 16")
    if layer_ids is not None and tuple(layer_ids) != tuple(meta.layer_ids):
        raise ValueError(f"lane head file {path!r} covers layers {list(meta.layer_ids)}, the launch streams {list(layer_ids)}")
    if hidden is not None and hidden != hidden_f:
        raise ValueError(f"lane head file {path!r} has hidden {hidden_f}, the router input is {hidden}")
    if experts is not None and experts != experts_f:
        raise ValueError(f"lane head file {path!r} has {experts_f} experts, the layers have {experts}")
    for name in meta.constant_features:
        if name not in FEATURES:
            raise ValueError(f"lane head file {path!r}: constant feature {name!r} is not a feature")
        col = FEATURES.index(name)
        if not torch.all(tensors["dense_w"][:, col] == 0):
            raise ValueError(f"lane head file {path!r}: constant feature {name!r} has a nonzero folded weight")
    return LaneHeadFile(path=path, tensors=tensors, meta=meta)


# ---- the fp32 definition -------------------------------------------------------------------------------------------

def dense_features(cnt: int, first_pos: int, gate_w: float, score: float, age: int, resident: bool, live: int,
                   odds: float = 1.0) -> torch.Tensor:
    """The dense features of one routed expert, in :data:`FEATURES` order. ``age`` is forwards since the expert was
    last routed in this layer, negative for never (``first_touch``). ``odds`` was constant in training and is 1.0 in
    serving (its weight folds to zero)."""
    f = torch.zeros(len(FEATURES), dtype=torch.float32)
    f[0 if cnt == 1 else 1 if cnt == 2 else 2] = 1.0
    f[3 + min(first_pos, 5)] = 1.0
    f[9] = gate_w
    f[10] = math.log1p(max(score, 0.0))
    f[11] = 1.0 if score <= 0 else 0.0
    f[12] = math.log1p(age) if age >= 0 else 0.0
    f[13] = 1.0 if age < 0 else 0.0
    f[14] = 1.0 if resident else 0.0
    f[15] = live / 6.0
    f[16] = odds
    return f


def reference_keys(
    tensors: dict[str, torch.Tensor],
    row: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x: torch.Tensor,
    insert_scores: torch.Tensor,
    last_routed: torch.Tensor,
    expert_to_slot: torch.Tensor,
    forward: int,
    odds: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(keys, p)`` int64/float32 ``[experts]`` for one layer's gather, in fp32 on the tensors' device.

    ``topk_ids``/``topk_weights`` are ``[tokens, top_k]`` (ids < 0 are dead), ``x`` ``[tokens, hidden]`` the gate
    input, ``insert_scores`` the updater's decayed score before this forward's counts, ``last_routed`` the forward
    each expert was last routed in (-1 never), ``expert_to_slot`` the hot cache's int64 ``[experts]`` map (-1
    nonresident). Experts not routed keep key 0, p 0.
    """
    experts = insert_scores.numel()
    device = x.device
    keys = torch.zeros(experts, dtype=torch.int64, device=device)
    p_out = torch.zeros(experts, dtype=torch.float32, device=device)
    ids = topk_ids.long().cpu()
    w = topk_weights.float().cpu()
    live = int((ids >= 0).any(dim=1).sum())
    xf = x.float()
    z_all = (xf - tensors["pca_mean"][row].to(device)) @ tensors["pca_proj"][row].to(device).float()
    seen: dict[int, list] = {}
    for tok in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[tok, k])
            if e < 0:
                continue
            seen.setdefault(e, []).append((tok, float(w[tok, k])))
    scores = insert_scores.float().cpu()
    last = last_routed.cpu()
    slots = expert_to_slot.cpu()
    for e, hits in seen.items():
        best_tok = max(range(len(hits)), key=lambda i: (hits[i][1], -hits[i][0]))
        tok_star, gate_w = hits[best_tok]
        f = dense_features(len(hits), hits[0][0], gate_w, float(scores[e]), int(forward - last[e]) if int(last[e]) >= 0 else -1,
                           bool(slots[e] >= 0), live, odds).to(device)
        z = z_all[tok_star]
        logit = (tensors["expert_bias"][row, e].to(device) + tensors["bias"][row].to(device)
                 + (tensors["dense_w"][row].to(device) * f).sum() + (tensors["z_w"][row].to(device) * z).sum()
                 + (tensors["expert_u"][row, e].to(device) * z).sum())
        p = torch.sigmoid(logit)
        p_out[e] = p
        keys[e] = int(p.item() * KEY_SCALE) * (1 << 16) + (experts - 1 - e)
    return keys, p_out


# ---- device state ------------------------------------------------------------------------------------------------

class LaneHead:
    """One launch's head on the device: the weights, the per-layer keys the route plan reads, and the last-routed
    forward per (layer, expert) for the age feature."""

    def __init__(self, head: LaneHeadFile, device: torch.device | str, max_tokens: int = 8):
        self.file = head
        self.meta = head.meta
        self.device = torch.device(device)
        t = head.tensors
        self.pca_mean = t["pca_mean"].to(self.device).contiguous()
        self.pca_proj = t["pca_proj"].to(self.device).contiguous()
        self.expert_bias = t["expert_bias"].to(self.device).contiguous()
        self.expert_u = t["expert_u"].to(self.device).contiguous()
        self.bias = t["bias"].to(self.device).contiguous()
        self.dense_w = t["dense_w"].to(self.device).contiguous()
        self.z_w = t["z_w"].to(self.device).contiguous()
        layers, experts, rank = len(self.meta.layer_ids), self.meta.experts, self.meta.rank
        self.keys = torch.zeros((layers, experts), dtype=torch.int64, device=self.device)
        self.last_routed = torch.full((layers, experts), -1, dtype=torch.int32, device=self.device)
        self.z = torch.zeros((layers, max_tokens, rank), dtype=torch.float32, device=self.device)
        self.max_tokens = max_tokens
        self.launches = torch.zeros(layers, dtype=torch.int64, device=self.device)

    def describe(self) -> str:
        return (f"lane head active: file={self.file.path} layers={len(self.meta.layer_ids)} rank={self.meta.rank} "
                f"device_bytes={self.file.device_bytes} training={self.meta.training}")

    def launch(self, row: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor, x: torch.Tensor,
               insert_scores: torch.Tensor, expert_to_slot: torch.Tensor, forward: torch.Tensor) -> None:
        """Write ``keys[row]`` for this gather's routed experts, on the current stream (graph-capturable)."""
        from sglang.kernels.ops.moe.lane_head import lane_head_keys

        lane_head_keys(
            topk_ids, topk_weights, x, self.pca_mean[row], self.pca_proj[row], self.expert_bias[row], self.expert_u[row],
            self.bias[row : row + 1], self.dense_w[row], self.z_w[row], insert_scores, self.last_routed[row],
            expert_to_slot, forward, self.z[row], self.keys[row], self.launches[row : row + 1], self.meta.odds,
        )

    def record_routed(self, gate: torch.Tensor, routed: torch.Tensor, forwards: torch.Tensor, prefill: bool) -> None:
        """At a residency boundary: remember the forward each routed expert was last routed in, per layer, for the
        age feature; a prefill boundary starts a request, so every age resets to never."""
        if prefill:
            self.last_routed.fill_(-1)
            return
        self.last_routed.copy_(
            torch.where(gate.unsqueeze(1) & routed, forwards.to(torch.int32).expand_as(self.last_routed), self.last_routed)
        )


class LaneHeadGather:
    """The per-layer hook ExpertStreamer._gather_graph calls before the fused plan: launches the head on the gate
    input Exl3MoEMethod._apply_graph staged for the CPU experts (``row_backend.cpu_input``)."""

    def __init__(self, head: LaneHead, updater, row: int, streamer):
        self.head = head
        self.updater = updater
        self.row = row
        self.streamer = streamer
        self.experts = head.meta.experts

    def __call__(self, topk_ids: torch.Tensor) -> None:
        cpu_input = getattr(self.streamer.row_backend, "cpu_input", None)
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
            self.updater.insert_scores[self.row],
            self.updater.mapping[self.row, : self.experts],
            self.updater.forwards,
        )
