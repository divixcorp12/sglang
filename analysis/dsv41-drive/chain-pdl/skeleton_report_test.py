"""skeleton_report: PDL saving per layer and per step against mode 0, and the Task 8 gate (CPU, no torch)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import skeleton_report as report  # noqa: E402


def _rec(mode, work, pre, replay_us, layers=40):
    rec = {"kind": "skeleton", "mode": mode, "work_ns": work, "layers": layers, "replay_us_p50": replay_us}
    if pre is not None:
        rec["pre_ns"] = pre
    return rec


def test_savings_are_against_mode_0_of_the_same_work_and_pre_wait_spin():
    rows = report.savings([_rec(0, 2000, 200, 1000.0), _rec(1, 2000, 200, 960.0), _rec(2, 2000, 200, 900.0),
                           _rec(0, 2000, None, 953.54), _rec(2, 2000, None, 894.27)])
    by = {(r["work_ns"], r["pre_ns"], r["mode"]): r for r in rows}
    assert by[(2000, 200, 1)]["per_layer_us"] == pytest.approx(1.0)
    assert by[(2000, 200, 2)]["per_step_us"] == pytest.approx(100.0)
    assert by[(2000, 0, 2)]["per_layer_us"] == pytest.approx(1.482, abs=1e-3)  # records without pre_ns are pre 0
    assert by[(2000, 200, 1)]["gate"] is False and by[(2000, 200, 2)]["gate"] is True
    assert by[(2000, 200, 2)]["step_share"] == pytest.approx(100 / 66800)


def test_the_gate_is_two_microseconds_per_layer():
    rows = report.savings([_rec(0, 2000, 500, 1100.0), _rec(2, 2000, 500, 1010.0)])
    assert rows[0]["per_layer_us"] == pytest.approx(2.25) and rows[0]["gate"] is True
