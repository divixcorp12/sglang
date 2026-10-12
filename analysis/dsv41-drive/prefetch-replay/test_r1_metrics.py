"""r1_metrics.py on hand-built groups: per-key cells, cap-dropped and never-candidate counts, the MISS_GPU/CPU split,
whole-layer coverage, the pre-cap histogram and bursts, position and row-set attribution, reliability bins, agreement
with r1_policy.sweep, and the paired session bootstrap. CPU only."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import r1_groups  # noqa: E402
import r1_metrics  # noqa: E402
import r1_policy  # noqa: E402
from r1_metrics import PolicyCell, PolicyReport  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

E = r1_groups.NUM_EXPERTS
GPU, CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), int(KIND_VRAM_HOT)

# Token 0 (anchor) misses 10 (GPU) and 11 (CPU); token 1 misses 10, 20, 30 (GPU) and routes 2 VRAM-hot at demand.
# 30 is in the RAM snapshot (not addressable); 0 and 1 are hot. Gate top-12 is 0..11, so 20 and 30 are never-candidates.
_IDS = [[10, 11, 12, 0, -1, -1], [10, 20, 30, 2, -1, -1]]
_KINDS = [[GPU, CPU, HIT, HOT, 0, 0], [GPU, GPU, GPU, HOT, 0, 0]]
_SCORES = {10: 0.95, 20: 0.8, 11: 0.75, 12: 0.7, 2: 0.65, 5: 0.6, 0: 0.99, 30: 0.95}


def _group(*, session="s1", seq=0, target_layer=4, flagged=False, warm=False):
    n = 2
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 30)
    return b1_rows.Group(
        session=session, split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=warm, flagged=flagged,
        horizon=1, source_row=0, source_layer=target_layer - 1, source_stratum="a", gate_has_bias=True,
        target_row=1, target_layer=target_layer, target_stratum="b", tokens=np.array([0, 1], np.int16),
        x_offset=n * seq, source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=np.array(_IDS, np.int16),
        target_kinds=np.array(_KINDS, np.uint8), ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _read(tmp_path, by_session):
    for session, groups in by_session.items():
        b1_rows.write_groups(tmp_path, split="dev", horizon=1, session=session, groups=groups)
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _cands(r, scores=_SCORES):
    s = np.full(E, -np.inf, np.float32)
    for e, v in scores.items():
        s[e] = v
    return r1_policy.candidates(s, skip=~r.labels.available)


def test_cell_counts_cap_drops_never_and_miss_kind_split(tmp_path):
    groups = _read(tmp_path, {"s1": [_group()]})
    cands = [_cands(r) for r in groups]
    np.testing.assert_array_equal(cands[0].ids, [10, 20, 11, 12, 2, 5])
    rep = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62)
    assert rep.split == "dev" and set(rep.cells) == {"pooled", "pooled_noflag", "layer:4", "stratum:b", "session:s1"}
    c = rep.cells["pooled"]
    assert (c.forwards, c.eligible_all, c.eligible_addr, c.read, c.useful, c.wasted) == (1, 4, 3, 2, 2, 0)
    assert (c.never_candidates, c.never_recovered, c.cap_dropped_useful, c.wasted_hot) == (2, 1, 1, 0)
    assert (c.miss_gpu_eligible, c.miss_cpu_eligible, c.miss_gpu_useful, c.miss_cpu_useful) == (3, 1, 2, 0)
    assert (c.layers_with_miss, c.layer_covered) == (1, 0)
    assert c.recall == 0.5 and c.recall_addr == pytest.approx(2 / 3) and c.precision == 1.0


def test_uncapped_reads_count_wasted_hot_and_unaddressable_rows_block_layer_coverage(tmp_path):
    groups = _read(tmp_path, {"s1": [_group()]})
    rep = r1_metrics.policy_report(groups, [_cands(groups[0])], scorer="x", cap=6, threshold=0.62)
    c = rep.cells["pooled"]
    assert (c.read, c.useful, c.wasted, c.wasted_hot, c.cap_dropped_useful) == (5, 3, 2, 1, 0)
    assert c.layer_covered == 0  # 30 sits in RAM, so the layer's misses can never all be read


def test_histogram_is_pre_cap_with_percentiles_and_bursts(tmp_path):
    groups = _read(tmp_path, {"s1": [_group(target_layer=4), _group(target_layer=5), _group(target_layer=6)]})
    low = {10: 0.9, 20: 0.1}
    cands = [_cands(groups[0]), _cands(groups[1]), _cands(groups[2], low)]
    rep = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62)
    h = rep.hist["pooled"]
    assert h.bins == {"0": 0, "1": 1, "2": 0, "3": 0, "4+": 2}
    assert h.max == 5 and h.p50 == 5 and h.p99 == 5
    assert h.bursts == {"2": 1} and h.max_burst == 2
    assert rep.hist["session:s1"].bursts == {"2": 1} and rep.hist["layer:4"].bursts == {}
    assert rep.forward_totals["max"] == 5.0 and rep.forward_totals["forwards"] == 1.0
    assert r1_metrics._bursts([(7, 3), (4, 3), (6, 0), (5, 3)], cap=2) == [2, 1]


def test_position_and_row_set_attribution(tmp_path):
    groups = _read(tmp_path, {"s1": [_group()]})
    rep = r1_metrics.policy_report(groups, [_cands(groups[0])], scorer="x", cap=2, threshold=0.62)
    p0, p1 = rep.positions["0"], rep.positions["1"]
    assert (p0.miss_rows, p0.miss_rows_addr, p0.covered) == (2, 2, 1)
    assert (p1.miss_rows, p1.miss_rows_addr, p1.covered) == (3, 2, 2)
    rs = {k: (v.eligible_all, v.eligible_addr, v.useful) for k, v in rep.rowsets.items()}
    assert rs == {"anchor_only": (1, 1, 0), "draft_only": (2, 1, 1), "shared": (1, 1, 1)}
    assert sum(v.eligible_all for v in rep.rowsets.values()) == rep.cells["pooled"].eligible_all


def test_reliability_bins_only_for_unit_interval_scores(tmp_path):
    groups = _read(tmp_path, {"s1": [_group()]})
    rep = r1_metrics.policy_report(groups, [_cands(groups[0])], scorer="x", cap=2, threshold=0.62)
    assert len(rep.reliability) == r1_metrics.RELIABILITY_BINS
    top = rep.reliability[-1]
    assert (top.candidates, top.candidates_miss, top.read, top.read_miss) == (1, 1, 1, 1)
    assert sum(b.candidates for b in rep.reliability) == 6 and sum(b.candidates_miss for b in rep.reliability) == 3
    assert sum(b.read for b in rep.reliability) == 2
    margins = r1_metrics.policy_report(groups, [_cands(groups[0], {10: 2.5, 20: -0.3})], scorer="m", cap=2,
                                       threshold=0.0)
    assert margins.reliability == []


def test_subset_keys_and_infinite_threshold_json(tmp_path):
    groups = _read(tmp_path, {"s1": [_group(flagged=True, warm=True)]})
    rep = r1_metrics.policy_report(groups, [_cands(groups[0])], scorer="x", cap=2, threshold=math.inf)
    assert "flagged" in rep.cells and "warm" in rep.cells and "pooled_noflag" not in rep.cells
    assert rep.cells["pooled"].read == 0 and rep.hist["pooled"].bins["0"] == 1
    js = r1_metrics.report_to_json(rep)
    assert js["threshold"] is None and js["cells"]["pooled"]["recall"] == 0.0
    assert js["rowsets"]["shared"]["recall"] == 0.0 and "coverage" in js["positions"]["0"]


def test_pooled_cells_match_the_exact_sweep(tmp_path):
    groups = _read(tmp_path, {"s1": [_group(seq=0), _group(seq=1)], "s2": [_group(session="s2", seq=2)]})
    rng = np.random.default_rng(0)
    cands = [_cands(r, {e: float(v) for e, v in zip((10, 20, 11, 12, 2, 5), rng.random(6))}) for r in groups]
    for cap in (1, 2):
        for p in r1_policy.sweep(cands, [r.labels for r in groups], cap=cap):
            c = r1_metrics.policy_report(groups, cands, scorer="x", cap=cap, threshold=p.threshold).cells["pooled"]
            assert (c.read, c.useful, c.eligible_all) == (p.reads, p.useful, p.eligible)


def test_report_rejects_skip_rows_and_length_mismatch(tmp_path):
    groups = _read(tmp_path, {"s1": [_group()]})
    bad = r1_policy.Candidates(ids=np.array([30], np.int64), vals=np.array([0.9], np.float32))
    with pytest.raises(AssertionError):
        r1_metrics.policy_report(groups, [bad], scorer="x", cap=2, threshold=0.5)
    with pytest.raises(ValueError):
        r1_metrics.policy_report(groups, [], scorer="x", cap=2, threshold=0.5)


def _rep(sessions, *, cap=2):
    cells = {f"session:{s}": PolicyCell(useful=u, eligible_all=e, eligible_addr=e, read=r)
             for s, (u, e, r) in sessions.items()}
    return PolicyReport(scorer="x", split="dev", cap=cap, threshold=0.5, cells=cells, hist={}, positions={},
                        rowsets={}, reliability=[], forward_totals={})


def test_session_bootstrap_point_interval_and_determinism():
    cand = _rep({"s1": (5, 10, 8), "s2": (2, 10, 8)})
    ctrl = _rep({"s1": (3, 10, 8), "s2": (2, 10, 8)})
    out = r1_metrics.session_bootstrap(cand, ctrl, n=2000, seed=0)
    assert out["diff"] == pytest.approx(0.1) and out["ratio"] == pytest.approx(0.35 / 0.25)
    assert out["precision_diff"] == pytest.approx(2 / 16)
    assert 0.0 <= out["lo"] <= out["diff"] <= out["hi"] <= 0.2 + 1e-12
    assert out == r1_metrics.session_bootstrap(cand, ctrl, n=2000, seed=0)
    same = r1_metrics.session_bootstrap(cand, cand, n=500, seed=3)
    assert same["diff"] == same["lo"] == same["hi"] == 0.0


def test_session_bootstrap_matches_a_plain_resampling_loop():
    cand = _rep({"a": (4, 9, 6), "b": (1, 7, 3), "c": (6, 6, 9)})
    ctrl = _rep({"a": (2, 9, 5), "b": (3, 7, 4), "c": (5, 6, 8)})
    n = 50
    idx = np.random.default_rng(7).integers(0, 3, size=(n, 3))
    u_c, u_k, e = np.array([4, 1, 6.0]), np.array([2, 3, 5.0]), np.array([9, 7, 6.0])
    diffs = [(u_c[i].sum() - u_k[i].sum()) / e[i].sum() for i in idx]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = r1_metrics.session_bootstrap(cand, ctrl, n=n, seed=7)
    assert out["lo"] == pytest.approx(lo) and out["hi"] == pytest.approx(hi)


def test_session_bootstrap_refuses_unpaired_reports():
    a = _rep({"s1": (5, 10, 8)})
    with pytest.raises(ValueError):
        r1_metrics.session_bootstrap(a, _rep({"s2": (5, 10, 8)}))
    with pytest.raises(ValueError):
        r1_metrics.session_bootstrap(a, _rep({"s1": (5, 11, 8)}))
    with pytest.raises(ValueError):
        r1_metrics.session_bootstrap(a, _rep({"s1": (5, 10, 8)}, cap=1))
