"""R1 capped read policy: the one path every control and candidate is scored through (spec section 1.3).

Token-score controls are pooled exactly once here (max or noisy-or); a GroupScorer's block scores are consumed as they
are. Per group, the skip-mask rows (VRAM-hot or in the target's RAM snapshot) are removed, the rest are ranked by
(-score, expert id), a global threshold keeps scores at or above it and the read set is truncated to cap 1 or 2. The
exact sorted-threshold sweep, the D8 selection point (best recall at mean reads <= 1 per forward-layer) and the D9 calib
threshold rule are computed over those candidates.

Inputs: R1Groups of one split and horizon (r1_groups.read_r1_groups) and GroupScorers.
Output: Candidates per scorer and group, SweepPoints, a selected point and a calibrated threshold.

Usage: pts = r1_policy.sweep(r1_policy.score_split(groups, [s])[s.name], [r.labels for r in groups], cap=2)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Literal

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_groups  # noqa: E402
from r1_groups import GroupLabels, GroupScorer, R1Group  # noqa: E402

NUM_EXPERTS = r1_groups.NUM_EXPERTS
CAPS = (1, 2)
MEAN_READS_BUDGET = 1.0  # reads per (forward, target layer), D8/D9


def pool_tokens(token_scores: np.ndarray, *, rule: Literal["max", "noisy_or"]) -> np.ndarray:
    """f32 [384] block score from f32 [n, 384] token scores; -inf where no token has a finite score.

    'noisy_or' is 1 - prod(1 - clip(p, 0, 1)) over finite token scores, accumulated in fp64.
    """
    scores = np.asarray(token_scores, dtype=np.float32)
    finite = np.isfinite(scores)
    if rule == "max":
        value = np.where(finite, scores, -np.inf).max(axis=0, initial=-np.inf)
    elif rule == "noisy_or":
        p = np.where(finite, np.clip(scores, 0.0, 1.0), 0.0).astype(np.float64)
        value = 1.0 - np.prod(1.0 - p, axis=0)
    else:
        raise ValueError(f"unknown pooling rule {rule!r}")
    return np.where(finite.any(axis=0), value, -np.inf).astype(np.float32)


class Candidates(msgspec.Struct, frozen=True):
    ids: np.ndarray  # int64 [k], ordered by (-val, id)
    vals: np.ndarray  # f32 [k], finite; skip-mask rows removed


def candidates(scores: np.ndarray, *, skip: np.ndarray) -> Candidates:
    value = np.asarray(scores, dtype=np.float32)
    ids = np.nonzero(np.isfinite(value) & ~np.asarray(skip, dtype=bool))[0].astype(np.int64)
    vals = value[ids]
    order = np.lexsort((ids, -vals))
    return Candidates(ids=ids[order], vals=vals[order])


def select(c: Candidates, *, threshold: float, cap: int) -> np.ndarray:
    """Ids read: candidates scoring >= threshold, best first, at most cap."""
    return c.ids[c.vals >= threshold][:cap]


class SweepPoint(msgspec.Struct, frozen=True):
    threshold: float
    reads: int
    useful: int
    eligible: int
    forwards: int
    recall: float
    precision: float
    mean_reads: float


def _eligible(lab: GroupLabels, *, denominator: str) -> np.ndarray:
    if denominator == "all":
        return lab.eligible_all
    if denominator == "addressable":
        return lab.eligible_addr
    raise ValueError(f"unknown denominator {denominator!r}")


def _point(*, threshold: float, reads: int, useful: int, eligible: int, forwards: int) -> SweepPoint:
    return SweepPoint(threshold=threshold, reads=reads, useful=useful, eligible=eligible, forwards=forwards,
                      recall=useful / eligible if eligible else 0.0, precision=useful / reads if reads else 0.0,
                      mean_reads=reads / forwards if forwards else 0.0)


def sweep(cands: list[Candidates], labels: list[GroupLabels], *, cap: int,
          denominator: Literal["all", "addressable"] = "all") -> list[SweepPoint]:
    """One point per distinct score among each group's top-cap candidates, by descending threshold (rising reads).

    A score below a group's cap-th candidate never changes that group's reads, so these points are every distinct
    outcome of the capped policy over all global thresholds.
    """
    if len(cands) != len(labels):
        raise ValueError(f"{len(cands)} candidate lists for {len(labels)} groups")
    eligible = [_eligible(lab, denominator=denominator) for lab in labels]
    total_eligible = int(sum(len(e) for e in eligible))
    vals = [c.vals[:cap] for c in cands]
    hits = [np.isin(c.ids[:cap], e) for c, e in zip(cands, eligible)]
    vals_all = np.concatenate(vals) if vals else np.zeros(0, np.float32)
    hits_all = np.concatenate(hits) if hits else np.zeros(0, bool)
    order = np.argsort(-vals_all, kind="stable")
    v, useful = vals_all[order], np.cumsum(hits_all[order])
    # The last index of each run of equal values: everything at or above that value is read.
    ends = np.nonzero(np.append(v[1:] != v[:-1], True))[0] if len(v) else np.zeros(0, np.int64)
    return [_point(threshold=float(v[i]), reads=int(i) + 1, useful=int(useful[i]), eligible=total_eligible,
                   forwards=len(cands)) for i in ends]


def _feasible(points: list[SweepPoint], *, mean_reads: float) -> list[SweepPoint]:
    return [p for p in points if p.mean_reads <= mean_reads]


def _nothing_read(points: list[SweepPoint]) -> SweepPoint:
    # Every swept threshold is over budget (e.g. a wide tie at the top score); reading nothing is the only feasible
    # policy.
    p = points[0]
    return _point(threshold=math.inf, reads=0, useful=0, eligible=p.eligible, forwards=p.forwards)


def best_under_budget(points: list[SweepPoint], *, mean_reads: float = MEAN_READS_BUDGET) -> SweepPoint:
    """D8: the best recall with mean reads <= budget; equal recall goes to fewer reads (the higher threshold)."""
    if not points:
        raise ValueError("empty sweep")
    ok = _feasible(points, mean_reads=mean_reads)
    if not ok:
        return _nothing_read(points)
    return max(ok, key=lambda p: (p.recall, -p.reads, p.threshold))


def calibrate_threshold(points: list[SweepPoint], *, mean_reads: float = MEAN_READS_BUDGET) -> float:
    """D9: the lowest threshold with mean reads <= budget; among points with its read count, the highest threshold.

    math.inf when no swept threshold fits the budget.
    """
    if not points:
        raise ValueError("empty sweep")
    ok = _feasible(points, mean_reads=mean_reads)
    if not ok:
        return math.inf
    reads = min(ok, key=lambda p: p.threshold).reads
    return max(p.threshold for p in ok if p.reads == reads)


def _skips(groups: list[R1Group]) -> list[np.ndarray]:
    skips = [r1_groups.skip_mask(r.g) for r in groups]
    for r, skip in zip(groups, skips):
        if not np.array_equal(skip, ~r.labels.available):
            raise AssertionError(f"skip mask disagrees with the labels' availability for {r.key}")
    return skips


def score_split(groups: list[R1Group], scorers: list[GroupScorer]) -> dict[str, list[Candidates]]:
    """Every scorer scores the same groups; one skip mask per group is applied to all of them."""
    names = [s.name for s in scorers]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate scorer names {names}")
    skips = _skips(groups)
    out: dict[str, list[Candidates]] = {}
    for s in scorers:
        scores = s.score_groups(groups)
        if len(scores) != len(groups):
            raise AssertionError(f"{s.name} returned {len(scores)} block scores for {len(groups)} groups")
        cands = []
        for r, skip, block in zip(groups, skips, scores):
            block = np.asarray(block)
            if block.shape != (NUM_EXPERTS,) or np.isnan(block).any():
                raise AssertionError(f"{s.name}: block score for {r.key} is not a NaN-free [{NUM_EXPERTS}] array")
            cands.append(candidates(block, skip=skip))
        out[s.name] = cands
    return out
