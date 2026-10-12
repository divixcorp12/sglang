"""r1_policy.py on synthetic scores: single pooling, skip removal and id tie-breaks, threshold + cap, the exact sweep
against brute force, the D8/D9 rules and the shared group/skip path of score_split. CPU only."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import r1_groups  # noqa: E402
import r1_policy  # noqa: E402
from r1_policy import Candidates, SweepPoint  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

E = r1_groups.NUM_EXPERTS
MISS_GPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.HIT_COPY)


def _labels(eligible, *, addressable=None):
    eligible = np.asarray(eligible, dtype=np.int16)
    addr = eligible if addressable is None else np.asarray(addressable, dtype=np.int16)
    z = np.zeros(E, bool)
    return r1_groups.GroupLabels(miss=z, available=~z, route=np.zeros((1, E), bool), miss_gpu=z, miss_cpu=z,
                                 eligible_all=eligible, eligible_addr=addr)


def _cands(ids, vals):
    return Candidates(ids=np.asarray(ids, np.int64), vals=np.asarray(vals, np.float32))


def test_pool_tokens_max_and_noisy_or_with_no_finite_token_as_minus_inf():
    s = np.full((2, E), -np.inf, np.float32)
    s[0, 5], s[1, 5], s[1, 9] = 0.5, 0.4, 2.0
    mx = r1_policy.pool_tokens(s, rule="max")
    assert mx.dtype == np.float32 and mx.shape == (E,)
    assert mx[5] == np.float32(0.5) and mx[9] == np.float32(2.0) and np.isneginf(mx[0])
    no = r1_policy.pool_tokens(s, rule="noisy_or")
    assert no[5] == pytest.approx(1 - 0.5 * 0.6) and no[9] == 1.0 and np.isneginf(no[0])
    with pytest.raises(ValueError):
        r1_policy.pool_tokens(s, rule="sum")


def test_candidates_drop_skip_and_non_finite_rows_and_break_ties_by_id():
    scores = np.full(E, -np.inf, np.float32)
    scores[[40, 7, 3, 12, 200]] = [0.9, 0.5, 0.5, 0.95, 0.5]
    skip = np.zeros(E, bool)
    skip[12] = True
    c = r1_policy.candidates(scores, skip=skip)
    np.testing.assert_array_equal(c.ids, [40, 3, 7, 200])
    assert c.ids.dtype == np.int64 and c.vals.dtype == np.float32
    np.testing.assert_array_equal(r1_policy.select(c, threshold=0.5, cap=2), [40, 3])
    np.testing.assert_array_equal(r1_policy.select(c, threshold=0.6, cap=2), [40])
    np.testing.assert_array_equal(r1_policy.select(c, threshold=0.5, cap=1), [40])
    assert len(r1_policy.select(c, threshold=0.99, cap=2)) == 0


def _brute(cands, labels, *, threshold, cap, denominator):
    reads = useful = 0
    for c, lab in zip(cands, labels):
        read = r1_policy.select(c, threshold=threshold, cap=cap)
        elig = lab.eligible_all if denominator == "all" else lab.eligible_addr
        reads += len(read)
        useful += int(np.isin(read, elig).sum())
    return reads, useful


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize("denominator", ["all", "addressable"])
def test_sweep_matches_brute_force_at_every_distinct_value(cap, denominator):
    rng = np.random.default_rng(0)
    cands, labels = [], []
    for _ in range(40):
        scores = np.full(E, -np.inf, np.float32)
        k = int(rng.integers(0, 6))
        ids = rng.choice(E, size=k, replace=False)
        scores[ids] = rng.integers(0, 8, size=k) / 8.0  # coarse values force ties within and across groups
        cands.append(r1_policy.candidates(scores, skip=np.zeros(E, bool)))
        elig = rng.choice(E, size=3, replace=False)
        elig[0] = ids[0] if k else elig[0]
        elig = np.unique(elig)
        labels.append(_labels(elig, addressable=elig[:2]))
    pts = r1_policy.sweep(cands, labels, cap=cap, denominator=denominator)
    thresholds = [p.threshold for p in pts]
    assert thresholds == sorted(set(thresholds), reverse=True)
    assert all(a.reads < b.reads for a, b in zip(pts, pts[1:]))
    total = sum(len(lab.eligible_all if denominator == "all" else lab.eligible_addr) for lab in labels)
    for p in pts:
        reads, useful = _brute(cands, labels, threshold=p.threshold, cap=cap, denominator=denominator)
        assert (p.reads, p.useful, p.eligible, p.forwards) == (reads, useful, total, 40)
        assert p.mean_reads == reads / 40 and p.recall == useful / total
    # Every distinct value of any candidate gives one of the swept outcomes.
    outcomes = {(p.reads, p.useful) for p in pts}
    for v in np.unique(np.concatenate([c.vals for c in cands])):
        assert _brute(cands, labels, threshold=float(v), cap=cap, denominator=denominator) in outcomes


def _pt(threshold, reads, useful, forwards=10, eligible=20):
    return r1_policy._point(threshold=threshold, reads=reads, useful=useful, eligible=eligible, forwards=forwards)


def test_best_under_budget_is_the_best_recall_within_mean_reads():
    pts = [_pt(0.9, 4, 3), _pt(0.8, 8, 6), _pt(0.7, 10, 6), _pt(0.6, 12, 9)]
    best = r1_policy.best_under_budget(pts)
    assert best.threshold == 0.8 and best.recall == 6 / 20
    assert r1_policy.best_under_budget(pts, mean_reads=2.0).threshold == 0.6
    none = r1_policy.best_under_budget([_pt(0.9, 11, 5)])
    assert none.reads == 0 and none.threshold == math.inf and none.eligible == 20
    with pytest.raises(ValueError):
        r1_policy.best_under_budget([])


def test_calibrate_threshold_is_the_lowest_within_budget_with_equal_reads_to_the_higher():
    pts = [_pt(0.9, 4, 3), _pt(0.8, 8, 6), _pt(0.75, 10, 6), _pt(0.7, 10, 6), _pt(0.6, 12, 9)]
    assert r1_policy.calibrate_threshold(pts) == 0.75
    assert r1_policy.calibrate_threshold([_pt(0.9, 11, 5)]) == math.inf


def _group(*, seq):
    n = 2
    target_ids = np.array([[7, 3, 9, 1, 2, 0], [10, 11, 12, 13, 14, -1]], dtype=np.int16)
    kinds = np.array([[MISS_GPU, MISS_GPU, 0, 0, 0, 0], [MISS_GPU, HIT, HIT, HIT, HIT, 0]], dtype=np.uint8)
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 3)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=True, target_row=1,
        target_layer=4, target_stratum="b", tokens=np.array([0, 1], np.int16), x_offset=n * seq,
        source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6), source_weights=np.full((n, 6), 0.25, np.float32),
        target_ids=target_ids, target_kinds=kinds, ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


class _Fixed:
    def __init__(self, name, *, n=None):
        self.name, self.n, self.seen = name, n, []

    def score_groups(self, groups):
        self.seen.append([r.key for r in groups])
        out = []
        for r in groups:
            s = np.full(E, -np.inf, np.float32)
            s[[0, 3, 7, 10, 20]] = [0.99, 0.98, 0.5 + r.key.seq, 0.4, 0.4]
            out.append(s)
        return out if self.n is None else out[: self.n]


def _read(tmp_path):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=[_group(seq=0), _group(seq=1)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def test_score_split_scores_identical_groups_with_one_skip_mask(tmp_path):
    groups = _read(tmp_path)
    a, b = _Fixed("a"), _Fixed("b")
    out = r1_policy.score_split(groups, [a, b])
    assert a.seen == b.seen == [[r.key for r in groups]]
    for name in ("a", "b"):
        assert len(out[name]) == len(groups)
        # 0 is VRAM-hot and 3 is in the RAM snapshot: removed for every scorer; 10 and 20 tie, broken by id.
        np.testing.assert_array_equal(out[name][0].ids, [7, 10, 20])
    pts = r1_policy.sweep(out["a"], [r.labels for r in groups], cap=2)
    assert pts[-1].reads == 4 and pts[-1].useful == 4 and pts[-1].eligible == sum(
        len(r.labels.eligible_all) for r in groups)


def test_score_split_refuses_short_outputs_and_duplicate_names(tmp_path):
    groups = _read(tmp_path)
    with pytest.raises(AssertionError):
        r1_policy.score_split(groups, [_Fixed("a"), _Fixed("short", n=1)])
    with pytest.raises(ValueError):
        r1_policy.score_split(groups, [_Fixed("a"), _Fixed("a")])
