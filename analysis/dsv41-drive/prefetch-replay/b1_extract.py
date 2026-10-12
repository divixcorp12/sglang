"""B1 extraction: one B0 block directory's joined verify forwards as b1_rows Groups and their x rows.

Inputs are a block dir (``trace.jsonl``, ``router.*``, ``router.res.*``, ``rids.json``), the gates in residency row
order, the corpus manifest's per-session meta and the integrity report's non-tie mismatch cells. The record join is
b0_integrity's, so warm-up verifies (empty ``rids``) are skipped and counted. Per joined forward and source row T,
each horizon h in ``horizons`` whose target row T+h exists gives one Group over the forward's live tokens; x[T] of
those tokens is stored once and shared by the horizons. A mismatch cell (seq, layer_id, token) is the captured x and
ids of that layer disagreeing with the gate recompute, so its token is dropped from the groups whose source is that
layer. The target gate's fp32 top-12 is computed on x[T].

Usage: for session, groups, x in b1_extract.extract_block(BLOCK, W=W, bias=b, session_meta=m, drop=d): ...
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import Any

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b0_integrity as b0  # noqa: E402
import router_score  # noqa: E402
import tier_sim  # noqa: E402
from b1_rows import GATE_DEPTH, NUM_EXPERTS, TRAINED_HORIZONS, Group  # noqa: E402

from sglang.srt.layers.moe.residency_capture import load_residency  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import horizon_targets  # noqa: E402


class ExtractCounts(msgspec.Struct):
    """Running counts over the blocks extracted into it; dropped_by_split is split -> horizon -> dropped cells."""

    warmup_forwards: int = 0
    dropped_cells: int = 0
    invalid_horizons: int = 0
    dropped_by_split: dict[str, dict[int, int]] = {}


class _Block(msgspec.Struct, frozen=True):
    name: str
    res: Any
    capture: Any
    loaded: dict
    by_seq: dict[int, dict]  # graph forwards of the route log by seq
    rids: dict
    layer_ids: list[int]
    targets: list[list[int]]  # [row][h - 1], -1 = no target
    W: Any  # torch f32 [rows, experts, hidden]
    bias: Any  # torch f32 [rows, experts]


class _Cell(msgspec.Struct, frozen=True):
    """One (joined forward, source row) with its kept live tokens and the (horizon, target row) pairs it predicts."""

    seq: int
    router: int
    residency: int
    row: int
    tokens: np.ndarray
    targets: list[tuple[int, int]]


def mismatch_cells(integrity_report: dict, *, block_dir) -> set[tuple[int, int, int]]:
    """The (seq, layer_id, token) non-tie mismatches the report lists for ``block_dir``."""
    want = os.path.realpath(os.fspath(block_dir))
    cells = set()
    for report in integrity_report["per_session"]:
        if os.path.realpath(report["out_dir"]) != want or "mismatches" not in report:
            continue
        cells.update((int(m["seq"]), int(m["layer_id"]), int(m["token"])) for m in report["mismatches"])
    return cells


def _load_block(block_dir, *, W, bias, num_experts: int) -> tuple[_Block, b0.Joined, int]:
    block_dir = os.fspath(block_dir)
    res = load_residency(os.path.join(block_dir, b0.RESIDENCY_PREFIX))
    capture = router_score.load_capture(os.path.join(block_dir, "router"))
    loaded = tier_sim.load_forwards(os.path.join(block_dir, "trace.jsonl"), allow_dropped=True)
    layer_ids = [row.layer_id for row in res.rows]
    if layer_ids != capture.header["layer_ids"] or layer_ids != loaded["layer_ids"]:
        raise ValueError(f"{block_dir}: residency, router capture and route log name different layers")
    W, bias = torch.as_tensor(W, dtype=torch.float32), torch.as_tensor(bias, dtype=torch.float32)
    if len(W) != len(layer_ids) or tuple(bias.shape) != (len(layer_ids), num_experts):
        raise ValueError(f"{block_dir}: gates {tuple(W.shape)} for {len(layer_ids)} rows of {num_experts} experts")
    rids = b0._load_rids(block_dir)
    if rids is None:
        raise ValueError(f"{block_dir}: no rids.json, so forwards cannot be assigned to sessions")
    joined, counts = b0._join(loaded, capture, res)
    targets, _ = horizon_targets(layer_ids)
    by_seq = {int(f["seq"]): f for f in loaded["forwards"] if f["kind"] == "graph"}
    block = _Block(name=os.path.basename(os.path.normpath(block_dir)), res=res, capture=capture, loaded=loaded,
                   by_seq=by_seq, rids=rids, layer_ids=layer_ids, targets=targets, W=W, bias=bias)
    return block, joined, int(counts["no_request_forwards"])


def _plan(block: _Block, part: b0.Joined, *, split: str, drop: set, horizons, counts: ExtractCounts) -> list[_Cell]:
    """The session's cells in (seq, row) order; drops and no-target horizons are counted here."""
    by_horizon = counts.dropped_by_split.setdefault(split, {})
    cells = []
    for seq, router, q, live in zip(part.seqs.tolist(), part.router.tolist(), part.residency.tolist(),
                                    part.live.tolist()):
        for row, layer in enumerate(block.layer_ids):
            valid = [(h, block.targets[row][h - 1]) for h in horizons
                     if block.targets[row][h - 1] >= 0 and bool(block.res.valid[q, row, h - 1])]
            counts.invalid_horizons += len(horizons) - len(valid)
            dropped = [m for m in range(live) if (seq, layer, m) in drop]
            counts.dropped_cells += len(dropped)
            for h, _ in valid:
                by_horizon[h] = by_horizon.get(h, 0) + len(dropped)
            tokens = np.array([m for m in range(live) if m not in dropped], dtype=np.int64)
            if valid and len(tokens):
                cells.append(_Cell(seq=seq, router=router, residency=q, row=row, tokens=tokens, targets=valid))
    return cells


def _gate_top(x: torch.Tensor, *, W: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """The biased gate's top GATE_DEPTH ids and scores per token, best first, fp32."""
    biased = torch.nn.functional.softplus(x @ W.T).sqrt() + bias
    top = torch.topk(biased, GATE_DEPTH, dim=-1)
    return top.indices.numpy().astype(np.int16), top.values.numpy().astype(np.float32)


def _session_rid(forward: dict, *, rids: dict, session: str) -> str:
    return next(r for r in forward["rids"] if r in rids and rids[r]["session_id"] == session)


def _group(block: _Block, cell: _Cell, *, horizon: int, target: int, x: torch.Tensor, x_offset: int,
           session: str, meta: dict) -> Group:
    res, cap, rows, tok = block.res, block.capture, block.res.rows, cell.tokens
    n = len(tok)
    if rows[target].gate_has_bias:
        gate_ids, gate_scores = _gate_top(x, W=block.W[target], bias=block.bias[target])
    else:
        gate_ids = np.full((n, GATE_DEPTH), -1, dtype=np.int16)
        gate_scores = np.full((n, GATE_DEPTH), -np.inf, dtype=np.float32)
    forward = block.by_seq[cell.seq]
    target_layer = block.layer_ids[target]
    hot = [] if forward["hot"] is None else forward["hot"].get(target_layer, [])
    return Group(
        session=session, split=meta["split"], block=block.name, seq=cell.seq,
        rid=_session_rid(forward, rids=block.rids, session=session),
        warm_start=bool(meta["warm_start"]), flagged=bool(meta["flagged"]), horizon=horizon,
        source_row=cell.row, source_layer=block.layer_ids[cell.row], source_stratum=rows[cell.row].stratum,
        gate_has_bias=bool(rows[target].gate_has_bias), target_row=target, target_layer=target_layer,
        target_stratum=rows[target].stratum, tokens=tok.astype(np.int16), x_offset=x_offset,
        source_ids=np.asarray(cap.ids[cell.router, cell.row, tok]).astype(np.int16),
        source_weights=np.asarray(cap.w[cell.router, cell.row, tok]).astype(np.float32),
        target_ids=np.asarray(cap.ids[cell.router, target, tok]).astype(np.int16),
        target_kinds=np.asarray(res.kinds[cell.residency, target, tok]).astype(np.uint8),
        ram_mask=np.asarray(res.masks[cell.residency, cell.row, horizon - 1]).astype(np.uint64),
        hot=np.asarray(hot, dtype=np.int16), gate_ids=gate_ids, gate_scores=gate_scores,
    )


def _session_groups(block: _Block, cells: list[_Cell], *, session: str, meta: dict) -> tuple[list[Group], np.ndarray]:
    """Groups and the x shard they index: each cell's x rows once, shared by its horizons."""
    hidden = int(block.capture.header["hidden"])
    x = np.empty((sum(len(c.tokens) for c in cells), hidden), dtype=np.uint16)
    groups, offset = [], 0
    for cell in cells:
        n = len(cell.tokens)
        x[offset:offset + n] = block.capture.x[cell.router, cell.row, cell.tokens]
        xf = torch.from_numpy(router_score.bf16_to_f32(x[offset:offset + n]))
        groups.extend(_group(block, cell, horizon=h, target=t, x=xf, x_offset=offset, session=session, meta=meta)
                      for h, t in cell.targets)
        offset += n
    return groups, x


def extract_block(block_dir, *, W, bias, session_meta: dict[str, dict], drop: set[tuple[int, int, int]],
                  horizons=TRAINED_HORIZONS, num_experts: int = NUM_EXPERTS,
                  counts: ExtractCounts | None = None) -> Iterator[tuple[str, list[Group], np.ndarray]]:
    """Per session of the block: its id, its Groups (x_offset into the yielded x) and x uint16 [N, hidden].
    ``W``/``bias`` are in residency row order; ``counts`` accumulates as the iterator is consumed."""
    counts = ExtractCounts() if counts is None else counts
    block, joined, warmup = _load_block(block_dir, W=W, bias=bias, num_experts=num_experts)
    counts.warmup_forwards += warmup
    parts = b0._by_session_groups(joined, loaded=block.loaded, rids=block.rids)
    for session, part in parts.items():
        if session is None:
            continue
        if session not in session_meta:
            raise ValueError(f"{block_dir}: session {session} is not in the corpus manifest")
        meta = session_meta[session]
        cells = _plan(block, part, split=meta["split"], drop=drop, horizons=tuple(horizons), counts=counts)
        groups, x = _session_groups(block, cells, session=session, meta=meta)
        yield session, groups, x
