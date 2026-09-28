"""chain_report: stamp grouping, edge gaps, and the real-chain PDL savings against the gate and ship bar (CPU)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import chain_report as report  # noqa: E402


def _s(kernel, entry, waited, exit_, block=0):
    return [kernel | (block << 8), entry, waited, exit_]


def test_stamps_group_into_replays_and_one_launch_per_kernel_call():
    stamps = [
        _s(1, 100, 101, 110), _s(2, 105, 111, 150), _s(3, 200, 201, 210),
        _s(4, 205, 211, 300, 0), _s(4, 206, 212, 320, 1),  # S: two blocks, one launch
        _s(3, 318, 321, 330), _s(6, 331, 332, 340),
        _s(1, 400, 401, 410), _s(2, 402, 411, 420),
    ]
    reps = report.replays(stamps)
    assert [[l["kernel"] for l in r] for r in reps] == [[1, 2, 3, 4, 3, 6], [1, 2]]
    s = reps[0][3]
    assert (s["entry"], s["waited"], s["exit"], s["blocks"]) == (205, 212, 320, 2)


def test_edges_measure_entry_and_body_start_against_the_predecessors_exit():
    reps = report.replays([_s(1, 100, 101, 110), _s(2, 105, 111, 150), _s(3, 160, 161, 170)])
    e = report.edges(reps)
    assert e["post->W1"] == [{"entry_after_exit": -5, "body_after_exit": 1}]
    assert e["W1->A1"] == [{"entry_after_exit": 10, "body_after_exit": 11}]


def _chain(scenario, mode, us):
    return {"kind": "chain", "scenario": scenario, "mode": mode, "replay_us_p50": us}


def test_savings_are_per_scenario_against_off_with_the_gate_and_ship_bar():
    rows = report.savings([_chain("all_hit", "off", 50.0), _chain("all_hit", "pdl", 48.0),
                           _chain("all_hit", "pdl_early", 47.0), _chain("mixed", "off", 900.0)])
    by = {(r["scenario"], r["mode"]): r for r in rows}
    assert by[("all_hit", "pdl")]["per_layer_us"] == pytest.approx(2.0)
    assert by[("all_hit", "pdl")]["per_step_us"] == pytest.approx(80.0) and by[("all_hit", "pdl")]["gate"] is True
    assert by[("all_hit", "pdl_early")]["step_share"] == pytest.approx(120.0 / 66800)
    assert by[("all_hit", "pdl_early")]["ship"] is False  # 0.18% < 1%
    assert ("mixed", "pdl") not in by


def test_savings_over_rounds_use_the_median_replay_per_scenario_and_mode():
    recs = [_chain("mixed", "off", t) for t in (900.0, 440.0, 880.0)] + \
           [_chain("mixed", "pdl_early", t) for t in (870.0, 430.0, 420.0)]
    (row,) = report.savings(recs)
    assert row["rounds"] == 3
    assert row["per_layer_us"] == pytest.approx(880.0 - 430.0)  # medians, not the last record of each


def test_prologue_counts_only_launches_that_entered_after_their_predecessor_finished():
    reps = report.replays([_s(1, 100, 104, 110), _s(2, 105, 111, 150), _s(3, 160, 163, 170)])
    p = report.prologue(reps)
    assert p["A1"] == {"n": 1, "p50_ns": 3}  # entered at 160 > W1's exit 150: 163 - 160
    assert "W1" not in p  # entered at 105, before post's exit at 110: its wait time is not prologue
    assert p["post"] == {"n": 1, "p50_ns": 4}  # a replay's first launch has no stamped predecessor
