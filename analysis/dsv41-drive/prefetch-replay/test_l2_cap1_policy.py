"""l2_cap1_policy.py on synthetic candidates: the read-nothing point, budget thresholds, the one-row cap, curve JSON
and matched-read lookup. CPU only."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l2_cap1_policy  # noqa: E402
import r1_groups  # noqa: E402
import r1_policy  # noqa: E402

E = r1_groups.NUM_EXPERTS


def _labels(eligible):
    z = np.zeros(E, bool)
    elig = np.asarray(eligible, dtype=np.int16)
    return r1_groups.GroupLabels(miss=z, available=~z, route=np.zeros((1, E), bool), miss_gpu=z, miss_cpu=z,
                                 eligible_all=elig, eligible_addr=elig)


def _synthetic(seed):
    rng = np.random.default_rng(seed)
    cands, labels = [], []
    for _ in range(60):
        scores = np.full(E, -np.inf, np.float32)
        k = int(rng.integers(0, 5))
        ids = rng.choice(E, size=k, replace=False)
        scores[ids] = rng.integers(0, 16, size=k) / 16.0
        cands.append(r1_policy.candidates(scores, skip=np.zeros(E, bool)))
        labels.append(_labels(np.unique(rng.choice(E, size=3, replace=False))))
    return cands, labels


def _pt(threshold, reads, useful=0, forwards=10, eligible=20):
    return r1_policy._point(threshold=threshold, reads=reads, useful=useful, eligible=eligible, forwards=forwards)


def test_constants():
    assert l2_cap1_policy.CAP == 1
    assert l2_cap1_policy.BUDGETS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_read_nothing_reads_nothing_and_handles_empty_sweeps():
    p = l2_cap1_policy.read_nothing([_pt(0.9, 4, 3)])
    assert (p.threshold, p.reads, p.useful, p.eligible, p.forwards) == (math.inf, 0, 0, 20, 10)
    assert p.recall == 0.0 and p.mean_reads == 0.0
    e = l2_cap1_policy.read_nothing([], eligible=7, forwards=3)
    assert (e.eligible, e.forwards, e.reads) == (7, 3, 0)
    with pytest.raises(ValueError):
        l2_cap1_policy.read_nothing([])


def test_dev_points_are_within_budget_monotone_and_never_read_two_rows_per_group():
    cands, labels = _synthetic(0)
    pts = r1_policy.sweep(cands, labels, cap=l2_cap1_policy.CAP)
    thr = l2_cap1_policy.dev_thresholds(pts)
    assert thr[0.0] == math.inf
    recalls, reads_seen = [], []
    for budget, t in thr.items():
        reads = useful = 0
        for c, lab in zip(cands, labels):
            sel = r1_policy.select(c, threshold=t, cap=l2_cap1_policy.CAP)
            assert len(sel) <= 1
            reads += len(sel)
            useful += int(np.isin(sel, lab.eligible_all).sum())
        assert reads / len(cands) <= budget
        recalls.append(useful)
        reads_seen.append(reads)
    assert reads_seen[0] == 0
    assert recalls == sorted(recalls) and reads_seen == sorted(reads_seen)


def test_calib_thresholds_match_r1_policy_on_the_calib_sweep():
    cands, labels = _synthetic(1)
    pts = r1_policy.sweep(cands, labels, cap=l2_cap1_policy.CAP)
    thr = l2_cap1_policy.calib_thresholds(pts)
    assert thr[0.0] == math.inf
    for b in (0.25, 0.5, 0.75, 1.0):
        assert thr[b] == r1_policy.calibrate_threshold(pts, mean_reads=b)


def test_matched_point_picks_nearest_and_ties_go_to_fewer_reads():
    pts = [_pt(0.9, 4), _pt(0.8, 8), _pt(0.7, 12)]
    m = l2_cap1_policy.matched_point(pts, target_reads=9)
    assert (m["target_reads"], m["achieved_reads"], m["threshold"]) == (9, 8, 0.8)
    assert m["point"] is pts[1]
    assert l2_cap1_policy.matched_point(pts, target_reads=6)["achieved_reads"] == 4
    assert l2_cap1_policy.matched_point(pts, target_reads=100)["achieved_reads"] == 12
    zero = l2_cap1_policy.matched_point(pts, target_reads=0)
    assert zero["achieved_reads"] == 0 and zero["threshold"] == math.inf


def test_curve_json_serialises_inf_as_none_and_derives_wasted():
    rows = l2_cap1_policy.curve_json([_pt(0.9, 4, 3), l2_cap1_policy.read_nothing([_pt(0.9, 4, 3)])])
    assert rows[0] == {"threshold": 0.9, "reads": 4, "useful": 3, "wasted": 1, "recall": 3 / 20,
                       "precision": 3 / 4, "mean_reads": 0.4}
    assert rows[1]["threshold"] is None and rows[1]["wasted"] == 0
