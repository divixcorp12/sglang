"""l2_fast_metrics.py: pure parity tables between reference and candidate L+2 block scores (F123 C14-C17, C20).

Compares a candidate path's per-group fp32 [384] block scores with the reference's on the available experts only (the
rows R1GroupScorer keeps finite). Read decisions go through r1_policy.candidates/select, so skipped rows are never
read and ties break to the lowest expert id, as in F109.

Inputs: R1Groups (r1_groups.read_r1_groups), reference and candidate block scores (one [384] array per group), the
frozen threshold.
Output: plain dicts (parity_table, recall_change) ready for result.json.

Usage: t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=thr)
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_metrics  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import R1Group  # noqa: E402


def json_finite(x: float) -> float | None:
    """None for +-inf and NaN, so json.dumps stays strict."""
    value = float(x)
    return value if math.isfinite(value) else None


def _candidates(r: R1Group, scores: np.ndarray) -> r1_policy.Candidates:
    return r1_policy.candidates(scores, skip=~r.labels.available)


def _read(c: r1_policy.Candidates, *, threshold: float) -> int | None:
    ids = r1_policy.select(c, threshold=threshold, cap=1)
    return int(ids[0]) if len(ids) else None


def _top(c: r1_policy.Candidates) -> tuple[int | None, float]:
    if not len(c.ids):
        return None, -math.inf
    return int(c.ids[0]), float(c.vals[0])


def _available_diffs(groups: list[R1Group], *, ref: list[np.ndarray], got: list[np.ndarray]) -> np.ndarray:
    parts = []
    for r, a, b in zip(groups, ref, got):
        x = np.asarray(a, dtype=np.float64)[r.labels.available]
        y = np.asarray(b, dtype=np.float64)[r.labels.available]
        with np.errstate(invalid="ignore"):
            parts.append(np.where(x == y, 0.0, np.abs(x - y)))
    return np.concatenate(parts) if parts else np.zeros(0)


def _is_near(rc: r1_policy.Candidates, gc: r1_policy.Candidates, *, threshold: float, near: float) -> bool:
    """Either side's deciding (top-1) score lies within `near` of the threshold."""
    return any(abs(_top(c)[1] - threshold) <= near for c in (rc, gc))


def _disagreement(index: int, r: R1Group, *, rc, gc, threshold: float) -> dict:
    return {"group": index, "session": r.key.session, "ref_read": _read(rc, threshold=threshold),
            "got_read": _read(gc, threshold=threshold), "ref_score": json_finite(_top(rc)[1]),
            "got_score": json_finite(_top(gc)[1])}


def parity_table(groups: list[R1Group], *, ref: list[np.ndarray], got: list[np.ndarray], threshold: float,
                 near: float = 1e-3) -> dict:
    """Score diffs, top-1 and cap-1 read-set agreement; read disagreements are split by distance to threshold."""
    if not (len(groups) == len(ref) == len(got)):
        raise ValueError(f"{len(groups)} groups, {len(ref)} ref and {len(got)} got score arrays")
    diffs = _available_diffs(groups, ref=ref, got=got)
    top1_equal, read_equal, near_list = 0, 0, []
    for i, (r, a, b) in enumerate(zip(groups, ref, got)):
        rc, gc = _candidates(r, a), _candidates(r, b)
        top1_equal += int(_top(rc)[0] == _top(gc)[0])
        if _read(rc, threshold=threshold) == _read(gc, threshold=threshold):
            read_equal += 1
            continue
        if _is_near(rc, gc, threshold=threshold, near=near):
            near_list.append(_disagreement(i, r, rc=rc, gc=gc, threshold=threshold))
    n = len(groups)
    disagreements = n - read_equal
    return {"groups": n, "max_abs_diff": float(diffs.max()) if len(diffs) else 0.0,
            "p99_abs_diff": float(np.percentile(diffs, 99)) if len(diffs) else 0.0,
            "top1_agree": top1_equal / n if n else 1.0, "read_agree": read_equal / n if n else 1.0,
            "disagreements": disagreements, "near_threshold": near_list,
            "far_disagreements": disagreements - len(near_list)}


def _cap1_report(groups: list[R1Group], scores: list[np.ndarray], *, name: str, threshold: float):
    cands = [_candidates(r, s) for r, s in zip(groups, scores)]
    return r1_metrics.policy_report(groups, cands, scorer=name, cap=1, threshold=threshold).cells["pooled"]


def recall_change(groups: list[R1Group], *, ref: list[np.ndarray], got: list[np.ndarray], threshold: float) -> dict:
    """Cap-1 recall of both score sets at one threshold (F109's denominator: every eligible miss row)."""
    a = _cap1_report(groups, ref, name="ref", threshold=threshold)
    b = _cap1_report(groups, got, name="got", threshold=threshold)
    return {"ref_recall": a.recall, "got_recall": b.recall, "diff": b.recall - a.recall, "ref_useful": a.useful,
            "got_useful": b.useful, "eligible": a.eligible_all}
