"""l2_cap1_timing.py: the cap-1 view, per-scorer delays, the zero-delay sensitivity and the frozen-candidate rule.
CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import l2_cap1_timing  # noqa: E402
import r1_groups  # noqa: E402
import r1_plan  # noqa: E402
from r1_stamps import Window  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

GPU, CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), int(KIND_VRAM_HOT)
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


def _groups(tmp_path):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=[_group(seq=s) for s in range(2)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _plan():
    return r1_plan.default_plan(dataset="d", corpus="c", b1_head="h", out_dir="o", inputs={})


def _costs(cap=1):
    return {n: {"cap": cap, "gate_ms": g, "gate_only_ms": go}
            for n, g, go in (("L2_T0", 0.4, 0.05), ("L2_T3", 0.6, 0.06), ("L2_T0_best_cap1", 0.5, 0.05))}


def test_timing_view_is_cap_one_with_the_plans_own_assumptions():
    plan = _plan()
    view = l2_cap1_timing.timing_view(plan)
    assert view.stage2_gate.cap == 1
    assert view.timing.contention == plan.timing.contention
    assert view.timing.row_read_ms == plan.timing.row_read_ms
    assert plan.stage2_gate.cap == 2


def test_delays_map_names_and_refuse_cap_two_or_missing_scorer():
    d = l2_cap1_timing.delays_from_costs(_costs())
    assert d == {"L2_T0": 0.4, "L2_T3": 0.6, "L2_T0_best_cap1": 0.5, "gate_cap1": 0.05}
    with pytest.raises(ValueError):
        l2_cap1_timing.delays_from_costs(_costs(cap=2))
    missing = _costs()
    del missing["L2_T3"]
    with pytest.raises(ValueError):
        l2_cap1_timing.delays_from_costs(missing)


def test_frozen_candidate_boundary_and_lower_budget_wins():
    on_time = {0.25: 50, 0.5: 95, 0.75: 99, 1.0: 100}
    out = l2_cap1_timing.frozen_candidate(on_time)
    assert out["budget"] == 0.5 and out["reference"] == 100
    on_time[0.5] = 94
    assert l2_cap1_timing.frozen_candidate(on_time)["budget"] == 0.75
    on_time[0.25] = 100
    assert l2_cap1_timing.frozen_candidate(on_time)["budget"] == 0.25
    assert l2_cap1_timing.frozen_candidate({0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0})["budget"] == 0.25


def test_zero_delay_on_time_useful_is_not_below_measured(tmp_path):
    groups = _groups(tmp_path)
    reads = [np.array([10]), np.array([10])]
    evals = {"dev": {"L2_T0": {1: {"reads": reads}}}}
    windows = {(r.key.block, r.key.seq, r.key.source_layer, r.key.horizon): Window(snap_ns=0, demand_ns=int(5e6))
               for r in groups}
    out = l2_cap1_timing.replay_budgets(evals, groups={"dev": groups}, plan=_plan(), windows=windows,
                                        delays={"L2_T0": 2.0})
    m = out["measured"]["cases"]["nominal"]["L2_T0"]["on_time_useful"]
    z = out["zero_delay"]["cases"]["nominal"]["L2_T0"]["on_time_useful"]
    assert z >= m
    assert out["measured"]["assumptions"]["cap"] == 1
    assert out["zero_delay"]["assumptions"]["scorer_delay_ms"] == {"L2_T0": 0.0}
