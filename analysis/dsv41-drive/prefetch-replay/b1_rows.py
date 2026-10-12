"""B1 row contract: the Group record every B1 stage codes against, its label helpers and the shard I/O.

A Group is one (verify forward, source row T, horizon h): the kept live tokens of the forward, their routes at T and
at the target row T+h, the target's RAM snapshot and VRAM-hot set, and the target gate's top-12 on x[T]. The source
input x[T] is stored once per split and session; groups point into it by ``x_offset``.

Layout under a dataset dir D:
  ``D/<split>/x/<session>.bin``   uint16 bf16 words [N, hidden], row-major; ``<session>.json`` holds rows/hidden.
  ``D/<split>/h<h>/<session>.npz`` the session's groups for horizon h, per-token arrays concatenated.

Locked-test shards are refused unless the caller passes the frozen selection file's sha256 (R1).

Usage: import b1_rows; for g in b1_rows.read_groups(D, split="dev", horizon=1): ...
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from typing import Literal, Protocol

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import router_score  # noqa: E402

from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

NUM_EXPERTS = 384
GATE_DEPTH = 12
TOPK = 6
HIDDEN = 5120
MASK_WORDS = 6  # 384 bits: expert e is bit e % 64 of word e // 64
TRAINED_HORIZONS = (1, 2, 4)
SPLITS = ("train", "dev", "calib", "locked_test")
LOCKED_SPLIT = "locked_test"
_MISS_KINDS = np.array([LaneKind.MISS_GPU, LaneKind.MISS_CPU], dtype=np.uint8)

_SCALARS = ("session", "split", "block", "seq", "rid", "warm_start", "flagged", "horizon", "source_row",
            "source_layer", "source_stratum", "gate_has_bias", "target_row", "target_layer", "target_stratum",
            "x_offset")
_STRS = frozenset({"session", "split", "block", "rid", "source_stratum", "target_stratum"})
_BOOLS = frozenset({"warm_start", "flagged", "gate_has_bias"})
# Per-token arrays, concatenated over the session's groups in file order.
_PER_TOKEN = {"tokens": np.int16, "source_ids": np.int16, "source_weights": np.float32, "target_ids": np.int16,
              "target_kinds": np.uint8, "gate_ids": np.int16, "gate_scores": np.float32}


class Group(msgspec.Struct, frozen=True):
    session: str
    split: str
    block: str
    seq: int
    rid: str
    warm_start: bool
    flagged: bool
    horizon: int
    source_row: int
    source_layer: int
    source_stratum: str
    gate_has_bias: bool  # of the target row
    target_row: int
    target_layer: int
    target_stratum: str
    tokens: np.ndarray  # int16 [n], kept live token indices
    x_offset: int  # first of the group's n rows in the session's x shard
    source_ids: np.ndarray  # int16 [n, 6]
    source_weights: np.ndarray  # f32 [n, 6]
    target_ids: np.ndarray  # int16 [n, 6]
    target_kinds: np.ndarray  # uint8 [n, 6], LaneKind
    ram_mask: np.ndarray  # uint64 [6], the target's snapshot
    hot: np.ndarray  # int16 [k], the target layer's VRAM-hot set
    gate_ids: np.ndarray  # int16 [n, 12], best first; -1 when the target has no biased gate
    gate_scores: np.ndarray  # f32 [n, 12], biased; -inf when the target has no biased gate


class TokenScores(msgspec.Struct, frozen=True):
    scores: np.ndarray  # f32 [n, E]; -inf = not a candidate
    rule: Literal["margin", "noisy_or", "count"]
    floor: float | None = None


class Scorer(Protocol):
    name: str

    def __call__(self, g: Group) -> TokenScores: ...


def eligible(g: Group) -> np.ndarray:
    """Unique experts the kept tokens routed at the target as an NVMe miss (MISS_GPU or MISS_CPU), int16."""
    miss = np.isin(g.target_kinds, _MISS_KINDS) & (g.target_ids >= 0)
    return np.unique(g.target_ids[miss]).astype(np.int16)


def never_candidates(g: Group) -> np.ndarray:
    """Eligible experts outside the target gate's top-12 for every kept token (D10)."""
    seen = g.gate_ids[g.gate_ids >= 0]
    elig = eligible(g)
    return elig[~np.isin(elig, seen)]


def in_ram(g: Group, experts: np.ndarray) -> np.ndarray:
    """Whether the target's snapshot showed each of ``experts`` mapped in RAM."""
    ids = np.asarray(experts, dtype=np.int64)
    word = g.ram_mask[ids // 64]
    return ((word >> (ids % 64).astype(np.uint64)) & np.uint64(1)).astype(bool)


def _split_dir(root, *, split: str, frozen_sha256: str | None) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == LOCKED_SPLIT and frozen_sha256 is None:
        raise PermissionError("locked_test is read only after selection is frozen: pass the selection's sha256")
    return os.path.join(os.fspath(root), split)


def write_x(out_dir, *, split: str, session: str, x: np.ndarray) -> str:
    """Write the session's x rows (uint16 bf16 words [N, hidden]) and their shape sidecar; returns the .bin path."""
    if x.dtype != np.uint16 or x.ndim != 2:
        raise ValueError(f"x must be uint16 [N, hidden], got {x.dtype} {x.shape}")
    d = os.path.join(os.fspath(out_dir), split, "x")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{session}.bin")
    np.ascontiguousarray(x).tofile(path)
    with open(os.path.join(d, f"{session}.json"), "w") as f:
        json.dump({"rows": int(x.shape[0]), "hidden": int(x.shape[1])}, f, indent=1)
    return path


def read_x(dataset_dir, *, split: str, session: str, frozen_sha256: str | None = None) -> np.memmap:
    d = os.path.join(_split_dir(dataset_dir, split=split, frozen_sha256=frozen_sha256), "x")
    with open(os.path.join(d, f"{session}.json")) as f:
        shape = json.load(f)
    return np.memmap(os.path.join(d, f"{session}.bin"), dtype=np.uint16, mode="r",
                     shape=(shape["rows"], shape["hidden"]))


def x_rows(xmap: np.ndarray, g: Group) -> np.ndarray:
    """The group's source inputs as f32 [n, hidden]."""
    n = len(g.tokens)
    return router_score.bf16_to_f32(np.asarray(xmap[g.x_offset:g.x_offset + n]))


def _pack(groups: list[Group]) -> dict[str, np.ndarray]:
    rows = [msgspec.structs.asdict(g) for g in groups]
    arrays: dict[str, np.ndarray] = {}
    for name in _SCALARS:
        dtype = str if name in _STRS else (bool if name in _BOOLS else np.int64)
        arrays[name] = np.array([r[name] for r in rows], dtype=dtype)
    arrays["n"] = np.array([len(g.tokens) for g in groups], dtype=np.int64)
    for name, dtype in _PER_TOKEN.items():
        width = {"tokens": None, "gate_ids": GATE_DEPTH, "gate_scores": GATE_DEPTH}.get(name, TOPK)
        empty = np.zeros((0,) if width is None else (0, width), dtype=dtype)
        arrays[name] = np.concatenate([empty] + [np.asarray(r[name], dtype=dtype) for r in rows])
    arrays["ram_mask"] = np.array([g.ram_mask for g in groups], dtype=np.uint64).reshape(len(groups), MASK_WORDS)
    arrays["hot_n"] = np.array([len(g.hot) for g in groups], dtype=np.int64)
    arrays["hot"] = np.concatenate([np.zeros(0, np.int16)] + [np.asarray(g.hot, dtype=np.int16) for g in groups])
    return arrays


def write_groups(out_dir, *, split: str, horizon: int, session: str, groups: list[Group]) -> str:
    """Write one session's groups for one horizon; returns the .npz path."""
    for g in groups:
        if g.session != session or g.split != split or g.horizon != horizon:
            raise ValueError(f"group ({g.session}, {g.split}, h{g.horizon}) in shard ({session}, {split}, h{horizon})")
    d = os.path.join(os.fspath(out_dir), split, f"h{horizon}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{session}.npz")
    np.savez(path, **_pack(groups))
    return path


def _unpack(z) -> Iterator[Group]:
    n, hot_n = z["n"], z["hot_n"]
    starts = np.concatenate([[0], np.cumsum(n)])
    hot_starts = np.concatenate([[0], np.cumsum(hot_n)])
    per_token = {name: z[name] for name in _PER_TOKEN}
    scalars = {name: z[name] for name in _SCALARS}
    ram_mask, hot = z["ram_mask"], z["hot"]
    order = np.lexsort((scalars["source_row"], scalars["seq"]))
    for i in order:
        a, b = int(starts[i]), int(starts[i + 1])
        fields = {name: _scalar(name, scalars[name][i]) for name in _SCALARS}
        fields.update({name: arr[a:b] for name, arr in per_token.items()})
        yield Group(**fields, ram_mask=ram_mask[i], hot=hot[int(hot_starts[i]):int(hot_starts[i + 1])])


def _scalar(name: str, value):
    if name in _STRS:
        return str(value)
    return bool(value) if name in _BOOLS else int(value)


def read_groups(dataset_dir, *, split: str, horizon: int, sessions: list[str] | None = None,
                frozen_sha256: str | None = None) -> Iterator[Group]:
    """Stream groups session by session (sorted names unless ``sessions`` gives the order), then by seq."""
    d = os.path.join(_split_dir(dataset_dir, split=split, frozen_sha256=frozen_sha256), f"h{horizon}")
    if sessions is None:
        sessions = sorted(name[:-len(".npz")] for name in os.listdir(d) if name.endswith(".npz"))
    # Not a generator itself, so the locked-test refusal raises at the call, not at the first next().
    return _stream(d, sessions=list(sessions))


def _stream(d: str, *, sessions: list[str]) -> Iterator[Group]:
    for session in sessions:
        with np.load(os.path.join(d, f"{session}.npz"), allow_pickle=False) as z:
            yield from _unpack(z)
