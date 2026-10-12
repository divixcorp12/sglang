"""R1 group reader: whole B1 groups with the D14 exclusions applied, their D4 block labels, and the shared contracts.

Wraps b1_rows.read_groups for the train/dev/calib splits only; locked_test is refused at the call with no override
(R4). Tokens named in the exclusion manifest are dropped from every group whose source or target layer is the cell's
layer; a group left with no tokens is dropped. Each R1Group carries its labels: per-token routes, the block miss
union over live tokens (MISS_GPU or MISS_CPU), availability (not VRAM-hot, not in the RAM snapshot), and both recall
denominators: all true misses (B1's historical eligible rows) and the addressable subset (miss & available).

The offline in-flight set is empty: with one horizon each (forward, target layer) group has exactly one source, so no
earlier speculative read of the same target is pending when a group is scored.

Inputs: a B1 dataset dir (b1_rows layout) and an exclusion manifest (r1_audit.build_exclusions JSON).
Output: R1Group records; GroupScorer, GroupBatch and ModelInput are the contracts later R1 stages code against.

Usage: for r in r1_groups.read_r1_groups(D, split="dev", horizon=1, exclusions=r1_groups.load_exclusions(P)): ...
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Protocol

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402
import b1_score  # noqa: E402
import router_score  # noqa: E402

from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

if TYPE_CHECKING:
    from torch import Tensor

R1_SPLITS = ("train", "dev", "calib")
MAX_WIDTH = 6  # live tokens per verify forward
NUM_EXPERTS = b1_rows.NUM_EXPERTS


class Exclusions(msgspec.Struct, frozen=True):
    cells: frozenset[tuple[str, int, int, int]]  # (block, seq, layer_id, token)

    @classmethod
    def empty(cls) -> Exclusions:
        return cls(cells=frozenset())


class _Cell(msgspec.Struct, frozen=True):
    block: str
    seq: int
    layer_id: int
    token: int
    reason: str
    source_groups: list
    target_groups: list


class _Manifest(msgspec.Struct, frozen=True):
    cells: list[_Cell]


def load_exclusions(path) -> Exclusions:
    with open(path, "rb") as f:
        manifest = msgspec.json.decode(f.read(), type=_Manifest)
    return Exclusions(cells=frozenset((c.block, c.seq, c.layer_id, c.token) for c in manifest.cells))


class GroupKey(msgspec.Struct, frozen=True):
    session: str
    block: str
    seq: int
    source_layer: int
    target_layer: int
    horizon: int


class GroupLabels(msgspec.Struct, frozen=True):
    miss: np.ndarray  # bool [384], some live token routed e as MISS_GPU or MISS_CPU; not masked by available
    available: np.ndarray  # bool [384], not VRAM-hot and not in the target's RAM snapshot
    route: np.ndarray  # bool [n, 384], token m routed e at the target
    miss_gpu: np.ndarray  # bool [384]
    miss_cpu: np.ndarray  # bool [384]
    eligible_all: np.ndarray  # int16 [k], every true miss (B1's denominator)
    eligible_addr: np.ndarray  # int16 [k], miss & available


class R1Group(msgspec.Struct, frozen=True):
    key: GroupKey
    g: b1_rows.Group  # excluded tokens removed; g.x_offset is the unfiltered group's, so read x through x_index
    positions: np.ndarray  # int64 [n], token index in the forward, 0 = anchor
    labels: GroupLabels
    x_index: np.ndarray  # int64 [n], each kept token's row in the session's x shard


def skip_mask(g: b1_rows.Group) -> np.ndarray:
    """Rows a read never targets: the target's RAM snapshot or VRAM-hot set, exactly as b1_score scores them."""
    return b1_score._skip_mask(g, num_experts=NUM_EXPERTS)


def x_rows(xmap: np.ndarray, r: R1Group) -> np.ndarray:
    """The group's kept source inputs as f32 [n, hidden]; use this, not b1_rows.x_rows, once tokens may be dropped."""
    return router_score.bf16_to_f32(np.asarray(xmap[r.x_index]))


class GroupScorer(Protocol):
    name: str

    # f32 [384] block score per group; -inf = not a candidate. Consumed as is by the policy, never pooled again.
    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]: ...


class GroupBatch(msgspec.Struct):
    live: np.ndarray  # bool [G, 6]; padding rows are zeros with live=False
    position: np.ndarray  # int64 [G, 6]
    width: np.ndarray  # f32 [G], live tokens per group
    source_layer: np.ndarray  # int64 [G]
    target_layer: np.ndarray  # int64 [G]
    stratum: np.ndarray  # int64 [G]
    miss: np.ndarray  # f32 [G, 384]
    available: np.ndarray  # f32 [G, 384]
    route: np.ndarray  # f32 [G, 6, 384]
    token_y: np.ndarray  # f32 [G, 6, 384], B1's original token labels (T0)
    hard_neg: np.ndarray  # f32 [G, 6, 384], B1's original hard negatives (T0)


class ModelInput(msgspec.Struct):
    batch: GroupBatch
    u: Tensor  # f32 [G, 6, F], E5 features
    x: Tensor | None  # f32 [G, 6, 5120]
    s_native: Tensor | None  # f32 [G, 6, 384]


def _labels(g: b1_rows.Group) -> GroupLabels:
    n = len(g.tokens)
    ids = np.asarray(g.target_ids, dtype=np.int64)
    routed = ids >= 0
    route = np.zeros((n, NUM_EXPERTS), dtype=bool)
    route[np.nonzero(routed)[0], ids[routed]] = True
    miss_gpu = np.zeros(NUM_EXPERTS, dtype=bool)
    miss_gpu[ids[routed & (g.target_kinds == LaneKind.MISS_GPU)]] = True
    miss_cpu = np.zeros(NUM_EXPERTS, dtype=bool)
    miss_cpu[ids[routed & (g.target_kinds == LaneKind.MISS_CPU)]] = True
    available = ~skip_mask(g)
    eligible_all = b1_rows.eligible(g)
    return GroupLabels(miss=miss_gpu | miss_cpu, available=available, route=route, miss_gpu=miss_gpu,
                       miss_cpu=miss_cpu, eligible_all=eligible_all,
                       eligible_addr=eligible_all[available[eligible_all]].astype(np.int16))


def _keep(g: b1_rows.Group, *, cells: frozenset) -> np.ndarray:
    tokens = np.asarray(g.tokens, dtype=np.int64)
    layers = (g.source_layer, g.target_layer)
    return np.array([not any((g.block, g.seq, layer, int(t)) in cells for layer in layers) for t in tokens],
                    dtype=bool)


def _r1_group(g: b1_rows.Group, *, keep: np.ndarray | None) -> R1Group | None:
    x_index = g.x_offset + np.arange(len(g.tokens), dtype=np.int64)
    if keep is not None:
        if not keep.any():
            return None
        g = msgspec.structs.replace(
            g, tokens=g.tokens[keep], source_ids=g.source_ids[keep], source_weights=g.source_weights[keep],
            target_ids=g.target_ids[keep], target_kinds=g.target_kinds[keep], gate_ids=g.gate_ids[keep],
            gate_scores=g.gate_scores[keep])
        x_index = x_index[keep]
    key = GroupKey(session=g.session, block=g.block, seq=g.seq, source_layer=g.source_layer,
                   target_layer=g.target_layer, horizon=g.horizon)
    return R1Group(key=key, g=g, positions=np.asarray(g.tokens, dtype=np.int64), labels=_labels(g), x_index=x_index)


def read_r1_groups(dataset_dir, *, split: str, horizon: int, exclusions: Exclusions,
                   sessions: list[str] | None = None) -> Iterator[R1Group]:
    """Stream the split's groups in b1_rows order with exclusions applied and labels attached."""
    if split == b1_rows.LOCKED_SPLIT:
        raise PermissionError("R1 never reads locked_test (R4); there is no override")
    if split not in R1_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {R1_SPLITS}")
    groups = b1_rows.read_groups(dataset_dir, split=split, horizon=horizon, sessions=sessions)
    # Not a generator itself, so the refusals above raise at the call.
    return _stream(groups, cells=exclusions.cells)


def _stream(groups: Iterable[b1_rows.Group], *, cells: frozenset) -> Iterator[R1Group]:
    forwards = {(block, seq) for block, seq, _, _ in cells}
    for g in groups:
        keep = _keep(g, cells=cells) if (g.block, g.seq) in forwards else None
        if keep is not None and keep.all():
            keep = None
        r = _r1_group(g, keep=keep)
        if r is not None:
            yield r
