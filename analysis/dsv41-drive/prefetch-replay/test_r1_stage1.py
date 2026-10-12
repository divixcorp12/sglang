"""r1_stage1.py on hand-built groups and fixed-score scorers: shared denominators, D8 vs D9 thresholds, caps, read
records, the r1_report layout, timing cases from the plan and the Stage-2 gate. CPU only."""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import r1_groups  # noqa: E402
import r1_plan  # noqa: E402
import r1_stage1  # noqa: E402
from r1_stamps import Window  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

E = r1_groups.NUM_EXPERTS
GPU, CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), int(KIND_VRAM_HOT)

# Misses: 10 (GPU), 11 (CPU), 20 and 30 (GPU); 30 is in the RAM snapshot, 0 and 1 are hot; 12 and 5 are not misses.
_IDS = [[10, 11, 12, 0, -1, -1], [10, 20, 30, 2, -1, -1]]
_KINDS = [[GPU, CPU, HIT, HOT, 0, 0], [GPU, GPU, GPU, HOT, 0, 0]]


def _group(*, seq: int) -> b1_rows.Group:
    n = 2
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 30)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=True,
        target_row=1, target_layer=4, target_stratum="b", tokens=np.array([0, 1], np.int16),
        x_offset=n * seq, source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=np.array(_IDS, np.int16),
        target_kinds=np.array(_KINDS, np.uint8), ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _groups(tmp_path, n=2):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=[_group(seq=s) for s in range(n)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


class _Fixed:
    """Block scores from one {expert: score} dict per group (or one dict for every group)."""

    def __init__(self, name, scores):
        self.name = name
        self._scores = scores

    def score_groups(self, groups):
        per = self._scores if isinstance(self._scores, list) else [self._scores] * len(groups)
        out = []
        for d in per:
            s = np.full(E, -np.inf, np.float32)
            for e, v in d.items():
                s[e] = v
            out.append(s)
        return out


def _scorers():
    return [_Fixed("F0_noisy_or", {10: 0.9, 20: 0.8, 11: 0.7}), _Fixed("gate", {5: 0.9, 10: 0.8}),
            _Fixed("gate_floors", {12: 0.9, 0: 0.95})]


def _plan():
    return r1_plan.default_plan(dataset="d", corpus="c", b1_head="h", out_dir="o", inputs={})


def _windows(groups, ms):
    return {(r.key.block, r.key.seq, r.key.source_layer, r.key.horizon): Window(snap_ns=0, demand_ns=int(ms * 1e6))
            for r in groups}


def test_every_scorer_shares_denominators_and_keeps_budget_and_cap(tmp_path):
    groups = _groups(tmp_path)
    ev = r1_stage1.evaluate_split(groups, _scorers())
    assert set(ev) == {"F0_noisy_or", "gate", "gate_floors"}
    for caps in ev.values():
        assert set(caps) == {1, 2}
        for cap, v in caps.items():
            p = v["point"]
            assert (p["eligible"], p["eligible_addr"], p["forwards"]) == (8, 6, 2)
            assert p["mean_reads"] <= 1.0 and p["feasible"]
            assert all(len(ids) <= cap for ids in v["reads"])
            assert 0 not in np.concatenate(v["reads"])  # hot row never read
    f0 = ev["F0_noisy_or"][2]
    assert f0["point"]["threshold"] == np.float32(0.9) and f0["point"]["recall"] == 0.25
    assert [ids.tolist() for ids in f0["reads"]] == [[10], [10]]
    assert ev["gate"][2]["point"]["useful"] == 0
    assert f0["report_json"]["cells"]["pooled"]["eligible_all"] == 8


def test_select_takes_fewer_reads_on_ties_and_calibrate_the_lowest_threshold(tmp_path):
    groups = _groups(tmp_path)
    s = [_Fixed("x", [{10: 0.9}, {5: 0.5}])]
    sel = r1_stage1.evaluate_split(groups, s, caps=(2,), rule="select")["x"][2]["point"]
    cal = r1_stage1.evaluate_split(groups, s, caps=(2,), rule="calibrate")["x"][2]["point"]
    assert (sel["threshold"], sel["reads"], sel["useful"]) == (np.float32(0.9), 1, 1)
    assert (cal["threshold"], cal["reads"], cal["useful"]) == (np.float32(0.5), 2, 1)


def test_given_thresholds_are_applied_and_infinity_reads_nothing(tmp_path):
    groups = _groups(tmp_path)
    ev = r1_stage1.evaluate_split(groups, _scorers()[:1], caps=(1, 2),
                                  thresholds={"F0_noisy_or": {1: 0.75, 2: math.inf}})
    assert ev["F0_noisy_or"][1]["point"]["reads"] == 2
    assert ev["F0_noisy_or"][2]["point"]["reads"] == 0 and ev["F0_noisy_or"][2]["point"]["threshold"] is None


def test_read_records_round_trip(tmp_path):
    groups = _groups(tmp_path / "data")
    reads = [np.array([10, 20]), np.zeros(0, np.int64)]
    path = tmp_path / "out" / "reads.npz"
    r1_stage1.write_read_records(str(path), groups, reads)
    z = np.load(path)
    assert z["offsets"].tolist() == [0, 2, 2] and z["ids"].tolist() == [10, 20]
    assert z["seq"].tolist() == [0, 1] and z["session"].tolist() == ["s1", "s1"]
    assert z["target_layer"].tolist() == [4, 4]
    assert not list(path.parent.glob("*.tmp.*"))


def test_summary_layout_timing_cases_and_gate_pass(tmp_path):
    groups = _groups(tmp_path)
    ev = {"dev": r1_stage1.evaluate_split(groups, _scorers())}
    out = r1_stage1.summarize(ev, groups={"dev": groups}, plan=_plan(), windows=_windows(groups, 100.0),
                              scorer_delay_ms=0.1)
    cell = out["stage1"]["dev"]["F0_noisy_or"]["cap2"]
    assert {"useful", "wasted", "eligible", "precision", "recall", "reads"} <= set(cell)
    hist = out["stage1"]["histograms"]["gate"]["cap2/dev/pooled"]
    assert {"0", "1", "2", "3", "4+", "p50", "p99", "max"} <= set(hist)
    assert set(out["timing"]["cases"]) == {"zero", "nominal", "adverse"}
    assert out["timing"]["assumptions"]["contention"] == {"zero": 1.0, "nominal": 2.0, "adverse": 3.0}
    assert out["timing"]["assumptions"]["scorer_delay_ms"] == 0.1
    assert out["timing"]["cases"]["nominal"]["F0_noisy_or"]["on_time_useful"] == 2
    gate = out["stage2_gate"]
    assert gate["passed"] is True and gate["values"]["best_control"] in r1_stage1.NATIVE_CONTROLS


def test_gate_fails_when_no_read_lands_in_time(tmp_path):
    groups = _groups(tmp_path)
    ev = {"dev": r1_stage1.evaluate_split(groups, _scorers())}
    out = r1_stage1.summarize(ev, groups={"dev": groups}, plan=_plan(), windows=_windows(groups, 1.0),
                              scorer_delay_ms=0.0)
    nominal = out["timing"]["cases"]["nominal"]["F0_noisy_or"]
    assert (nominal["on_time_useful"], nominal["late_useful"]) == (0, 2)
    assert out["stage2_gate"]["passed"] is False
    assert out["stage2_gate"]["values"]["recall"] is True and out["stage2_gate"]["values"]["timing"] is False
