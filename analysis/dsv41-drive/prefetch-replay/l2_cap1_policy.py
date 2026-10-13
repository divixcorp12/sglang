"""L+2 cap-1 operating points: the pre-registered mean-read budgets through R1's own policy rules.

Budget 0 is an explicit read-nothing point. Other budgets use r1_policy.best_under_budget on dev and
r1_policy.calibrate_threshold on calib's own sweep (D5, unchanged). Also exports a sweep curve as JSON and a
matched-read lookup.

Inputs: SweepPoints from r1_policy.sweep(..., cap=CAP).
Output: budget -> threshold maps, JSON-ready curve rows, matched points.

Usage: thr = l2_cap1_policy.dev_thresholds(r1_policy.sweep(cands, labels, cap=l2_cap1_policy.CAP))
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_policy  # noqa: E402
from r1_policy import SweepPoint  # noqa: E402

CAP = 1
BUDGETS = (0.0, 0.25, 0.50, 0.75, 1.00)  # mean reads per forward-layer, pre-registered


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def read_nothing(points: list[SweepPoint], *, eligible: int | None = None, forwards: int | None = None) -> SweepPoint:
    """The policy that reads no row; eligible and forwards come from the sweep unless it is empty."""
    if points:
        eligible, forwards = points[0].eligible, points[0].forwards
    elif eligible is None or forwards is None:
        raise ValueError("an empty sweep needs eligible and forwards")
    return r1_policy._point(threshold=math.inf, reads=0, useful=0, eligible=eligible, forwards=forwards)


def dev_thresholds(points: list[SweepPoint], *, budgets: tuple[float, ...] = BUDGETS) -> dict[float, float]:
    return {b: math.inf if b == 0.0 else r1_policy.best_under_budget(points, mean_reads=b).threshold
            for b in budgets}


def calib_thresholds(points: list[SweepPoint], *, budgets: tuple[float, ...] = BUDGETS) -> dict[float, float]:
    return {b: math.inf if b == 0.0 else r1_policy.calibrate_threshold(points, mean_reads=b) for b in budgets}


def curve_json(points: list[SweepPoint]) -> list[dict]:
    return [{"threshold": _finite(p.threshold), "reads": p.reads, "useful": p.useful, "wasted": p.reads - p.useful,
             "recall": p.recall, "precision": p.precision, "mean_reads": p.mean_reads} for p in points]


def matched_point(points: list[SweepPoint], *, target_reads: int) -> dict:
    """The curve point whose reads are nearest target_reads (ties to fewer reads); read-nothing for target 0."""
    if target_reads == 0:
        best = read_nothing(points)
    else:
        if not points:
            raise ValueError("empty sweep")
        best = min(points, key=lambda p: (abs(p.reads - target_reads), p.reads))
    return {"target_reads": target_reads, "achieved_reads": best.reads, "threshold": best.threshold, "point": best}
