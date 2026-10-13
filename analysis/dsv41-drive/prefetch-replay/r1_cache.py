"""R1 group cache: whole groups collated to [G, 6, ...] tensors, held on the device, with seeded group batching.

The E5 features are b1_head.make_batch's rows reshaped per group (padding slots are zeros with live=False and never
enter a pool, loss or normaliser, R7); the B1 token labels are kept for T0. A batch is always whole groups of one
forward, so group-order shuffling is the only randomness. T5/G0 can also keep raw x and the native gate scores
s_native (the full 384-score target gate); x is bounded by an explicit byte budget.

Inputs: r1_groups.R1Group streams, an x_of(R1Group) callable, a train-fitted PCA and the Vocab the model uses.
Usage: cache = r1_cache.build_cache(groups, x_of=x_of, pca=pca, cfg=cfg, vocab=vocab, device=device)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
from r1_groups import MAX_WIDTH, NUM_EXPERTS, GroupBatch, GroupKey, ModelInput, R1Group  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

# Arbitrary default bound on raw x held on the device (f32, 5120 wide, 6 slots per group).
_X_BUDGET_BYTES = 8 << 30
_HIDDEN = 5120


class GroupCache(msgspec.Struct):
    inp: ModelInput  # every group, in input order
    keys: list[GroupKey]
    groups: int


def _slots(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group index and slot index for each token row of the concatenated batch."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    gi = np.repeat(np.arange(len(counts)), counts)
    return gi, np.arange(int(counts.sum())) - starts[gi]


def _pad(values: torch.Tensor, *, gi: torch.Tensor, slot: torch.Tensor, groups: int) -> torch.Tensor:
    out = torch.zeros((groups, MAX_WIDTH, *values.shape[1:]), dtype=values.dtype, device=values.device)
    out[gi, slot] = values
    return out


def _group_batch(groups: list[R1Group], *, b: b1_head.HeadBatch, counts: np.ndarray, gi: np.ndarray,
                 slot: np.ndarray) -> GroupBatch:
    g = len(groups)
    live = np.zeros((g, MAX_WIDTH), dtype=bool)
    position = np.zeros((g, MAX_WIDTH), dtype=np.int64)
    route = np.zeros((g, MAX_WIDTH, NUM_EXPERTS), dtype=np.float32)
    for i, r in enumerate(groups):
        n = int(counts[i])
        live[i, :n] = True
        position[i, :n] = r.positions
        route[i, :n] = r.labels.route
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    per_group = {name: getattr(b, name)[torch.from_numpy(first).to(b.u.device)].cpu().numpy()
                 for name in ("source_layer", "target_layer", "stratum")}
    gi_t, slot_t = torch.from_numpy(gi).to(b.u.device), torch.from_numpy(slot).to(b.u.device)

    def padded(t: torch.Tensor) -> np.ndarray:
        return _pad(t.float(), gi=gi_t, slot=slot_t, groups=g).cpu().numpy()

    return GroupBatch(
        live=live, position=position, width=counts.astype(np.float32), **per_group,
        miss=np.stack([r.labels.miss for r in groups]).astype(np.float32),
        available=np.stack([r.labels.available for r in groups]).astype(np.float32),
        route=route, token_y=padded(b.y), hard_neg=padded(b.hard_neg))


def _native_scores(groups: list[R1Group], *, x: torch.Tensor, live: torch.Tensor,
                   gates: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Biased full target gate scores f32 [G, 6, 384]; zeros on padding and for targets with no biased gate."""
    out = torch.zeros((x.shape[0], MAX_WIDTH, NUM_EXPERTS), dtype=torch.float32, device=x.device)
    for layer in sorted({r.g.target_layer for r in groups if r.g.gate_has_bias}):
        idx = [i for i, r in enumerate(groups) if r.g.target_layer == layer and r.g.gate_has_bias]
        W, bias = gates[layer]
        rows = torch.tensor(idx, device=x.device)
        flat = x[rows].reshape(-1, x.shape[-1])
        s = r1_gate.full_gate_scores(flat, W=W.to(x.device), bias=bias.to(x.device))
        out[rows] = s.reshape(len(idx), MAX_WIDTH, NUM_EXPERTS)
    return out * live.to(x.device)[:, :, None]


def collate(groups: list[R1Group], *, x_of: Callable[[R1Group], np.ndarray], pca: PCAStats, cfg: b1_head.HeadConfig,
            vocab: b1_head.Vocab, device, with_x: bool = False,
            gates: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
            x_dtype: torch.dtype = torch.float32) -> ModelInput:
    """Groups to one padded [G, 6, ...] ModelInput on ``device``; s_native needs ``gates`` and implies x.

    x is stored in ``x_dtype``; s_native is always computed from fp32 x and stays fp32.
    """
    counts = np.array([len(r.g.tokens) for r in groups], dtype=np.int64)
    if len(groups) == 0 or counts.max() > MAX_WIDTH:
        raise ValueError(f"need 1..{MAX_WIDTH} tokens per group and at least one group")
    xs = [np.asarray(x_of(r), dtype=np.float32) for r in groups]
    by_id = {id(r.g): x for r, x in zip(groups, xs, strict=True)}
    b = b1_head.make_batch([r.g for r in groups], x_of=lambda g: by_id[id(g)], pca=pca, cfg=cfg, vocab=vocab)
    gi, slot = _slots(counts)
    batch = _group_batch(groups, b=b, counts=counts, gi=gi, slot=slot)
    dev = torch.device(device)
    gi_t, slot_t = torch.from_numpy(gi).to(b.u.device), torch.from_numpy(slot).to(b.u.device)
    u = _pad(b.u, gi=gi_t, slot=slot_t, groups=len(groups)).to(dev)
    x = s_native = None
    if with_x or gates is not None:
        flat = torch.from_numpy(np.concatenate(xs)).to(dev)
        x = _pad(flat, gi=gi_t.to(dev), slot=slot_t.to(dev), groups=len(groups))
    if gates is not None:
        s_native = _native_scores(groups, x=x, live=torch.from_numpy(batch.live).float(), gates=gates)
    return ModelInput(batch=batch, u=u, x=x.to(x_dtype) if with_x else None, s_native=s_native)


_DEVICE_FIELDS = ("u", "x", "s_native")


def _empty_like_rows(part: ModelInput, *, n_groups: int) -> dict[str, torch.Tensor | None]:
    return {n: None if (t := getattr(part, n)) is None else t.new_empty((n_groups, *t.shape[1:]))
            for n in _DEVICE_FIELDS}


def _cat_batches(batches: list[GroupBatch]) -> GroupBatch:
    return GroupBatch(**{n: np.concatenate([getattr(b, n) for b in batches]) for n in GroupBatch.__struct_fields__})


def x_cache_bytes(n_groups: int, *, x_dtype: torch.dtype) -> int:
    """Device bytes of raw x for ``n_groups`` padded groups held in ``x_dtype``."""
    return n_groups * MAX_WIDTH * _HIDDEN * torch.empty((), dtype=x_dtype).element_size()


def cache_bytes(n_groups: int, *, cfg: b1_head.HeadConfig, x_dtype: torch.dtype, with_s_native: bool,
                hidden: int = _HIDDEN) -> int:
    """Device bytes of a with-x cache of ``n_groups``: fp32 u, raw x in ``x_dtype`` and fp32 s_native."""
    per_slot = 4 * b1_head._feature_dim(cfg) + hidden * torch.empty((), dtype=x_dtype).element_size()
    per_slot += 4 * NUM_EXPERTS if with_s_native else 0
    return n_groups * MAX_WIDTH * per_slot


def build_cache(groups, *, x_of: Callable[[R1Group], np.ndarray], pca: PCAStats, cfg: b1_head.HeadConfig,
                vocab: b1_head.Vocab, device, with_x: bool = False,
                gates: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None, chunk: int = 4096,
                x_budget_bytes: int = _X_BUDGET_BYTES, x_dtype: torch.dtype = torch.float32) -> GroupCache:
    """Collate groups ``chunk`` at a time into one cache; x is refused past ``x_budget_bytes``.

    Chunks are copied into tensors sized for every group, so the device peak is the cache plus one chunk.
    """
    groups = list(groups)
    if not groups:
        raise ValueError("no groups to cache")
    if with_x and x_cache_bytes(len(groups), x_dtype=x_dtype) > x_budget_bytes:
        raise MemoryError(f"raw x cache needs more than {x_budget_bytes} bytes; stream it instead")
    out: dict[str, torch.Tensor | None] = {}
    batches: list[GroupBatch] = []
    for a in range(0, len(groups), chunk):
        part = collate(groups[a:a + chunk], x_of=x_of, pca=pca, cfg=cfg, vocab=vocab, device=device, with_x=with_x,
                       gates=gates, x_dtype=x_dtype)
        out = out or _empty_like_rows(part, n_groups=len(groups))
        for name, t in out.items():
            if t is not None:
                t[a:a + part.u.shape[0]] = getattr(part, name)
        batches.append(part.batch)
    keys = [r.key for r in groups]
    return GroupCache(inp=ModelInput(batch=_cat_batches(batches), **out), keys=keys, groups=len(keys))


def take(cache: GroupCache, idx: torch.Tensor) -> ModelInput:
    """Groups ``idx`` (int64 tensor of group indices) as one ModelInput; x always comes out fp32."""
    sel = idx.cpu().numpy()
    inp = cache.inp
    batch = GroupBatch(**{n: getattr(inp.batch, n)[sel] for n in GroupBatch.__struct_fields__})
    dev = inp.u.device
    idx = idx.to(dev)
    # .float() is a no-op on fp32 storage, so R1 caches return the indexed tensor unchanged.
    return ModelInput(batch=batch, u=inp.u[idx], x=None if inp.x is None else inp.x[idx].float(),
                      s_native=None if inp.s_native is None else inp.s_native[idx])


def epoch_batches(cache: GroupCache, *, groups_per_step: int, seed: int, epoch: int) -> Iterator[torch.Tensor]:
    """Whole-group index batches of one epoch; the order depends only on (seed, epoch)."""
    order = np.random.default_rng([seed, epoch]).permutation(cache.groups)
    device = cache.inp.u.device
    for a in range(0, cache.groups, groups_per_step):
        yield torch.from_numpy(order[a:a + groups_per_step]).to(device)


def make_x_of(dataset_dir, *, split: str) -> Callable[[R1Group], np.ndarray]:
    """Reads each group's kept rows from its session's memmapped x shard; refuses locked_test (R4)."""
    if split == b1_rows.LOCKED_SPLIT:
        raise PermissionError("R1 never reads locked_test (R4); there is no override")
    if split not in r1_groups.R1_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {r1_groups.R1_SPLITS}")
    maps: dict[str, object] = {}

    def x_of(r: R1Group) -> np.ndarray:
        if r.g.split != split:
            raise ValueError(f"group of split {r.g.split!r} asked of the {split!r} reader")
        if r.g.session not in maps:
            maps[r.g.session] = b1_rows.read_x(dataset_dir, split=split, session=r.g.session)
        return r1_groups.x_rows(maps[r.g.session], r)

    return x_of
