"""r1_timing.py on hand-built groups and windows: admission, on-time vs late useful, partial stall, missing windows."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r1_timing as tm  # noqa: E402
from r1_groups import GroupKey, GroupLabels, R1Group  # noqa: E402
from r1_stamps import Window  # noqa: E402

E = 384


def _group(seq: int, *, miss: list[int]) -> R1Group:
    m = np.zeros(E, dtype=bool)
    m[miss] = True
    empty = np.zeros(0, dtype=np.int16)
    labels = GroupLabels(miss=m, available=np.ones(E, dtype=bool), route=np.zeros((1, E), dtype=bool),
                         miss_gpu=m, miss_cpu=np.zeros(E, dtype=bool), eligible_all=empty, eligible_addr=empty)
    key = GroupKey(session="s", block="b", seq=seq, source_layer=19, target_layer=20, horizon=1)
    return R1Group(key=key, g=None, positions=np.zeros(1, dtype=np.int64), labels=labels,
                   x_index=np.zeros(1, dtype=np.int64))


def _window(ms: float) -> Window:
    return Window(snap_ns=1_000, demand_ns=1_000 + int(ms * 1e6))


def _case(*, contention=1.0, delay=0.0) -> tm.TimingCase:
    return tm.TimingCase(name="c", contention=contention, row_read_ms=2.0, scorer_delay_ms=delay)


def _windows(groups, ms) -> dict:
    return {("b", g.key.seq, 19, 1): _window(m) for g, m in zip(groups, ms)}


def test_room_for_both_reads_is_on_time():
    g = [_group(0, miss=[5])]
    res = tm.replay(g, [np.array([5, 7])], windows=_windows(g, [4.0]), case=_case())
    assert (res.reads_admitted, res.reads_dropped_window) == (2, 0)
    assert (res.useful, res.on_time_useful, res.late_useful) == (1, 1, 0)
    assert res.label == "model"


def test_contention_and_delay_shrink_admission():
    g = [_group(0, miss=[5, 7])]
    w = _windows(g, [4.5])
    ids = [np.array([5, 7])]
    assert tm.replay(g, ids, windows=w, case=_case(contention=1.0)).on_time_useful == 2
    assert tm.replay(g, ids, windows=w, case=_case(contention=2.0)).on_time_useful == 1
    assert tm.replay(g, ids, windows=w, case=_case(contention=1.0, delay=1.0)).on_time_useful == 1


def test_partial_arrival_prices_only_served_part():
    g = [_group(0, miss=[7])]
    res = tm.replay(g, [np.array([5, 7])], windows=_windows(g, [3.0]), case=_case())
    assert (res.reads_admitted, res.reads_dropped_window) == (1, 1)
    assert (res.on_time_useful, res.late_useful) == (0, 1)
    assert res.partial_saved_ms == pytest.approx(1.0)


def test_missing_window_is_excluded():
    g = [_group(0, miss=[5]), _group(1, miss=[5])]
    w = _windows(g[:1], [10.0])
    res = tm.replay(g, [np.array([5]), np.array([5])], windows=w, case=_case())
    assert res.missing_windows == 1
    assert res.useful == 1


def test_empty_reads_and_nonpositive_room():
    g = [_group(0, miss=[5])]
    res = tm.replay(g, [np.zeros(0, dtype=np.int64)], windows=_windows(g, [4.0]), case=_case())
    assert res.useful == 0 and res.reads_admitted == 0
    res = tm.replay(g, [np.array([5])], windows=_windows(g, [1.0]), case=_case(delay=2.0))
    assert (res.reads_admitted, res.late_useful, res.partial_saved_ms) == (0, 1, 0.0)


def test_cases_from_plan_carry_frozen_values():
    plan = r1_plan.default_plan(dataset="d", corpus="c", b1_head="h", out_dir="o", inputs={})
    cases = tm.cases_from_plan(plan.timing, scorer_delay_ms=0.5)
    assert [c.name for c in cases] == ["zero", "nominal", "adverse"]
    assert [c.contention for c in cases] == [1.0, 2.0, 3.0]
    assert all(c.row_read_ms == plan.timing.row_read_ms and c.scorer_delay_ms == 0.5 for c in cases)
    res = tm.replay([], [], windows={}, case=cases[1])
    assert res.assumptions["contention_source"] == plan.timing.contention_source


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        tm.replay([_group(0, miss=[])], [], windows={}, case=_case())
