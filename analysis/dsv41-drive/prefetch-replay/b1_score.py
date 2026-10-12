"""B1 scoring contract: the one function every scorer (gate, gate+floors, prev, pop, E2, E5) is evaluated through.

A scorer turns a Group into per-token expert scores (b1_rows.TokenScores). Here they are aggregated over the kept
tokens by the score's rule, rows the target layer holds VRAM-hot or the snapshot shows in RAM are skipped, and the top
1/2/3 rows (or every row at or above a cut) are read per (forward, target layer). Eligible rows (b1_rows.eligible) are
the recall denominator and are the same for every scorer; a read of a row routed VRAM-hot at demand is wasted.

Inputs: an iterable of b1_rows.Group (one split, one horizon) and one or more scorers.
Output: a Table of Cells per budget ('reads=r', 'cut=v') and key ('pooled', 'pooled_noflag', 'layer:<id>',
'stratum:<s>', 'session:<id>'), plus calibration bins for probability scorers.

Usage: tables = b1_score.evaluate_many(b1_rows.read_groups(D, split="dev", horizon=1), [gate, head], cuts=(0.5,))
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402
from b1_rows import Group, Scorer, TokenScores  # noqa: E402

from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

READS = (1, 2, 3)
# What a read that removes a layer's last miss saves, DSV41_REFERENCE section 33.24; deeper-queue reads save more,
# so pricing every useful read at this slope is a lower bound.
SAVING_MS_PER_LAST_MISS = (0.33, 0.52)
CALIBRATION_BINS = 10
_MARGIN_RANK = 5  # the margin is taken to the token's 6th score


class Cell(msgspec.Struct):
    forwards: int = 0  # (forward, target layer) groups
    eligible: int = 0
    read: int = 0
    useful: int = 0
    wasted_hot: int = 0
    never_candidates: int = 0
    never_recovered: int = 0
    saved_ms_lo: float = 0.0
    saved_ms_hi: float = 0.0

    @property
    def recall(self) -> float:
        return self.useful / self.eligible if self.eligible else 0.0

    @property
    def precision(self) -> float:
        return self.useful / self.read if self.read else 0.0

    @property
    def never_recovery(self) -> float:
        return self.never_recovered / self.never_candidates if self.never_candidates else 0.0


class Table(msgspec.Struct):
    scorer: str
    split: str
    horizon: int
    cells: dict[str, dict[str, Cell]] = {}
    calibration: list[tuple[float, float, int, int]] = []  # (p_lo, p_hi, candidates, eligible among them)


class _Labels(msgspec.Struct, frozen=True):
    eligible: frozenset
    never: frozenset
    hot_routed: frozenset
    keys: tuple[str, ...]


def _skip_mask(g: Group, *, num_experts: int) -> np.ndarray:
    skip = b1_rows.in_ram(g, np.arange(num_experts))
    hot = np.asarray(g.hot, dtype=np.int64)
    skip[hot[(hot >= 0) & (hot < num_experts)]] = True
    return skip


def _margin_values(scores: np.ndarray, *, skip: np.ndarray, k: int) -> np.ndarray:
    """Best margin per expert over tokens, keeping up to k unskipped finite scores per token; -inf = not kept."""
    n, num_experts = scores.shape
    sixth = -np.sort(-scores, axis=1)[:, _MARGIN_RANK] if num_experts > _MARGIN_RANK else np.full(n, -np.inf)
    # A token with fewer than 6 finite scores has no 6th; its raw scores stand in for margins.
    sixth = np.where(np.isfinite(sixth), sixth, 0.0).astype(np.float32)
    live = np.where(skip[None, :], -np.inf, scores)
    order = np.argsort(-live, axis=1, kind="stable")  # ties by expert id
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(num_experts)[None, :].repeat(n, 0), axis=1)
    keep = (rank < k) & np.isfinite(live)
    margin = np.where(keep, live - sixth[:, None], -np.inf)
    return margin.max(axis=0) if n else np.full(num_experts, -np.inf)


def _aggregate(ts: TokenScores, *, skip: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(ts.scores, dtype=np.float32)
    if ts.rule == "margin":
        return _margin_values(scores, skip=skip, k=k)
    finite = np.isfinite(scores)
    if ts.rule == "noisy_or":
        p = np.where(finite, np.clip(scores, 0.0, 1.0), 0.0)
        value = 1.0 - np.prod(1.0 - p.astype(np.float64), axis=0)
    else:
        value = np.where(finite, scores, 0.0).astype(np.float64).sum(axis=0)
    return np.where(finite.any(axis=0) & ~skip, value, -np.inf)


def candidates(g: Group, ts: TokenScores, *, k: int = b1_rows.GATE_DEPTH) -> list[tuple[int, float]]:
    """(expert, value) the forward could read at the target, by (-value, id); hot and RAM rows removed, floor applied.

    'margin' keeps up to k per token by margin to the token's 6th score; 'noisy_or' is 1 - prod(1 - p); 'count' sums.
    """
    num_experts = int(np.asarray(ts.scores).shape[1])
    value = _aggregate(ts, skip=_skip_mask(g, num_experts=num_experts), k=k)
    ok = np.isfinite(value)
    if ts.floor is not None:
        ok &= value >= ts.floor
    ids = np.nonzero(ok)[0]
    vals = value[ids]
    order = np.lexsort((ids, -vals))
    return [(int(ids[i]), float(vals[i])) for i in order]


def _labels(g: Group) -> _Labels:
    hot = (g.target_kinds == KIND_VRAM_HOT) & (g.target_ids >= 0)
    keys = ["pooled", f"layer:{g.target_layer}", f"stratum:{g.target_stratum}", f"session:{g.session}"]
    if not g.flagged:
        keys.insert(1, "pooled_noflag")
    return _Labels(eligible=frozenset(b1_rows.eligible(g).tolist()),
                   never=frozenset(b1_rows.never_candidates(g).tolist()),
                   hot_routed=frozenset(np.unique(g.target_ids[hot]).tolist()), keys=tuple(keys))


def _cut_key(cut: float) -> str:
    return f"cut={float(cut)!r}"


def _budgets(cand: list[tuple[int, float]], *, reads, cuts) -> dict[str, list[int]]:
    out = {f"reads={r}": [e for e, _ in cand[:r]] for r in reads}
    out.update({_cut_key(c): [e for e, v in cand if v >= c] for c in cuts})
    return out


def _add(cells: dict[str, Cell], *, labels: _Labels, read: list[int]) -> None:
    useful = sum(e in labels.eligible for e in read)
    lo, hi = SAVING_MS_PER_LAST_MISS
    for key in labels.keys:
        c = cells.setdefault(key, Cell())
        c.forwards += 1
        c.eligible += len(labels.eligible)
        c.read += len(read)
        c.useful += useful
        c.wasted_hot += sum(e in labels.hot_routed for e in read)
        c.never_candidates += len(labels.never)
        c.never_recovered += sum(e in labels.never for e in read)
        c.saved_ms_lo += lo * useful
        c.saved_ms_hi += hi * useful


def _calibrate(counts: np.ndarray, *, cand: list[tuple[int, float]], labels: _Labels) -> None:
    for e, v in cand:
        b = min(int(v * CALIBRATION_BINS), CALIBRATION_BINS - 1)
        counts[b, 0] += 1
        counts[b, 1] += e in labels.eligible


def _calibration_bins(counts: np.ndarray) -> list[tuple[float, float, int, int]]:
    width = 1.0 / CALIBRATION_BINS
    return [(i * width, (i + 1) * width, int(counts[i, 0]), int(counts[i, 1])) for i in range(CALIBRATION_BINS)]


def evaluate_many(groups: Iterable[Group], scorers: list[Scorer], *, reads=READS, cuts=(),
                  k: int = b1_rows.GATE_DEPTH) -> dict[str, Table]:
    """One streaming pass over ``groups``: every scorer sees the same groups, labels and denominators."""
    cells = {s.name: {} for s in scorers}
    calib = {s.name: np.zeros((CALIBRATION_BINS, 2), np.int64) for s in scorers}
    probabilistic: dict[str, bool] = {}
    split, horizon = "", 0
    for g in groups:
        split, horizon = g.split, g.horizon
        labels = _labels(g)
        for s in scorers:
            ts = s(g)
            cand = candidates(g, ts, k=k)
            for budget, read in _budgets(cand, reads=reads, cuts=cuts).items():
                _add(cells[s.name].setdefault(budget, {}), labels=labels, read=read)
            probabilistic[s.name] = ts.rule == "noisy_or"
            if probabilistic[s.name]:
                _calibrate(calib[s.name], cand=cand, labels=labels)
    return {s.name: Table(scorer=s.name, split=split, horizon=horizon, cells=cells[s.name],
                          calibration=_calibration_bins(calib[s.name]) if probabilistic.get(s.name) else [])
            for s in scorers}


def evaluate(groups: Iterable[Group], scorer: Scorer, *, reads=READS, cuts=(), k: int = b1_rows.GATE_DEPTH) -> Table:
    return evaluate_many(groups, [scorer], reads=reads, cuts=cuts, k=k)[scorer.name]


def _cut_sweep(table: Table) -> list[tuple[float, Cell]]:
    cuts = [(float(b[len("cut="):]), keyed["pooled"]) for b, keyed in table.cells.items()
            if b.startswith("cut=") and "pooled" in keyed]
    return sorted(cuts, key=lambda cc: cc[0])


def cut_for_precision(table: Table, precision: float) -> float | None:
    """The lowest swept cut whose pooled precision reaches ``precision`` (the most recall at that precision)."""
    for cut, cell in _cut_sweep(table):
        if cell.read and cell.precision >= precision:
            return cut
    return None


def recall_at_precision(table: Table, precision: float) -> float | None:
    """Pooled recall at ``cut_for_precision``; None when no swept cut reaches ``precision``."""
    cut = cut_for_precision(table, precision)
    return None if cut is None else table.cells[_cut_key(cut)]["pooled"].recall


def _cell_json(c: Cell) -> dict:
    return {**msgspec.structs.asdict(c), "recall": c.recall, "precision": c.precision,
            "never_recovery": c.never_recovery}


def table_to_json(t: Table) -> dict:
    return {"scorer": t.scorer, "split": t.split, "horizon": t.horizon,
            "cells": {b: {key: _cell_json(c) for key, c in keyed.items()} for b, keyed in t.cells.items()},
            "calibration": [list(row) for row in t.calibration]}
