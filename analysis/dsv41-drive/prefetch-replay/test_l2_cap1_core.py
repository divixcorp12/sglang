"""l2_cap1_core.evaluate_cap1 on hand-built dev and calib groups (two sessions, three target layers each) with a gate and
one learned scorer: cap-1 reads, the read-nothing budget, the paired comparisons and JSON output. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import l2_cap1_core  # noqa: E402
import l2_cap1_policy  # noqa: E402
import r1_groups  # noqa: E402
import r1_policy  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

E = r1_groups.NUM_EXPERTS
GPU, CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), int(KIND_VRAM_HOT)
SESSIONS = ("s1", "s2")

# Eligible rows 10, 11, 20 and 30; 30 is in the RAM snapshot, so no scorer may read it.
_IDS = [[10, 11, 12, 0, -1, -1], [10, 20, 30, 2, -1, -1]]
_KINDS = [[GPU, CPU, HIT, HOT, 0, 0], [GPU, GPU, GPU, HOT, 0, 0]]


def _group(*, split, session, seq):
    n = 2
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 30)
    target_layer = 4 + seq
    return b1_rows.Group(
        session=session, split=split, block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=target_layer - 1, source_stratum="a", gate_has_bias=True,
        target_row=1, target_layer=target_layer, target_stratum="b", tokens=np.array([0, 1], np.int16),
        x_offset=n * seq, source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=np.array(_IDS, np.int16),
        target_kinds=np.array(_KINDS, np.uint8), ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _split_groups(tmp_path, split):
    for s in SESSIONS:
        b1_rows.write_groups(tmp_path, split=split, horizon=1, session=s,
                             groups=[_group(split=split, session=s, seq=i) for i in range(3)])
    return list(r1_groups.read_r1_groups(tmp_path, split=split, horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _cands(r, scores):
    s = np.full(E, -np.inf, np.float32)
    for e, v in scores.items():
        s[e] = v
    return r1_policy.candidates(s, skip=~r.labels.available)


def _gate(groups):
    # Distinct per-group scores so the sweep has several points; the gate's best row is useful in half the groups.
    return [_cands(r, {10 if i % 2 else 5: 2.0 - 0.1 * i, 11: 1.0 - 0.05 * i}) for i, r in enumerate(groups)]


def _learned(groups):
    return [_cands(r, {20: 0.9 - 0.1 * i, 3: 0.3}) for i, r in enumerate(groups)]


def _inputs(tmp_path, *, learned=_learned):
    groups = {split: _split_groups(tmp_path, split) for split in ("dev", "calib")}
    cands = {split: {"gate_cap1": _gate(gs), "L2_T0": learned(gs)} for split, gs in groups.items()}
    return groups, cands


def test_budget_points_respect_cap_one_and_budget(tmp_path):
    groups, cands = _inputs(tmp_path)
    result, reads = l2_cap1_core.evaluate_cap1(groups, cands)
    for split, by_scorer in result.items():
        for name, out in by_scorer.items():
            assert out["curve"] == l2_cap1_policy.curve_json(
                r1_policy.sweep(cands[split][name], [r.labels for r in groups[split]], cap=1))
            for b, entry in out["budgets"].items():
                p = entry["point"]
                assert p["reads"] <= p["forwards"] == len(groups[split])
                assert p["mean_reads"] <= float(b) + 1e-12
                assert p["wasted"] == p["reads"] - p["useful"]
                assert entry["coverage"]["forward_layers"] == len(groups[split])
            for b, rs in reads[split][name].items():
                assert len(rs) == len(groups[split]) and all(len(x) <= 1 for x in rs)
                assert sum(len(x) for x in rs) == out["budgets"][str(b)]["point"]["reads"]


def test_budget_zero_reads_nothing(tmp_path):
    groups, cands = _inputs(tmp_path)
    result, reads = l2_cap1_core.evaluate_cap1(groups, cands)
    for split in groups:
        for name in cands[split]:
            entry = result[split][name]["budgets"]["0.0"]
            assert entry["point"]["reads"] == 0 and entry["point"]["threshold"] is None
            assert entry["coverage"]["abstention_rate"] == 1.0
            assert all(len(x) == 0 for x in reads[split][name][0.0])


def test_gate_against_itself_has_zero_difference(tmp_path):
    groups, cands = _inputs(tmp_path, learned=_gate)
    result, _ = l2_cap1_core.evaluate_cap1(groups, cands)
    for split in groups:
        for entry in result[split]["L2_T0"]["budgets"].values():
            for boot in (entry["same_budget"], entry["matched"]["bootstrap"]):
                assert boot["diff"] == 0.0 and boot["lo"] == 0.0 and boot["hi"] == 0.0
                assert boot["useful_diff"] == 0.0 and boot["n"] == 10_000 and boot["seed"] == 0
            assert entry["matched"]["achieved_reads"] == entry["matched"]["target_reads"]


def test_matched_entry_reports_target_and_achieved_reads(tmp_path):
    groups, cands = _inputs(tmp_path)
    result, _ = l2_cap1_core.evaluate_cap1(groups, cands)
    for split in groups:
        assert "same_budget" not in result[split]["gate_cap1"]["budgets"]["0.5"]
        for entry in result[split]["L2_T0"]["budgets"].values():
            m = entry["matched"]
            assert m["target_reads"] == m["learned_reads"] == entry["point"]["reads"]
            assert m["achieved_reads"] == m["gate_reads"] == m["gate_point"]["reads"]
            assert {"useful_lo", "useful_hi", "lo", "hi"} <= set(m["bootstrap"])
            assert set(m["gain_split"]) == {"inside", "outside"}


def test_output_is_json_serialisable(tmp_path):
    groups, cands = _inputs(tmp_path)
    result, _ = l2_cap1_core.evaluate_cap1(groups, cands)
    text = json.dumps(result, allow_nan=False)
    assert json.loads(text)["dev"]["L2_T0"]["budgets"]["1.0"]["dspark"]["widths"]


def test_assert_identical_refuses_misaligned_inputs(tmp_path):
    groups, cands = _inputs(tmp_path)
    l2_cap1_core.assert_identical(groups, cands)
    short = {**cands, "dev": {**cands["dev"], "L2_T0": cands["dev"]["L2_T0"][:-1]}}
    with pytest.raises(ValueError):
        l2_cap1_core.assert_identical(groups, short)
    with pytest.raises(ValueError):
        l2_cap1_core.assert_identical({"dev": groups["dev"]}, cands)
    with pytest.raises(ValueError):
        l2_cap1_core.evaluate_cap1(groups, cands, gate="gate")
