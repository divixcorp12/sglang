"""mech_report: link ceilings, knees, bandwidth-delay products and the refusals (CPU, no torch)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mech_report as report  # noqa: E402


def _meta(gen=3, width=16):
    return {"meta": True, "host": "h", "pcie_gen": gen, "pcie_width": width}


def _cell(method, in_flight, gbs, name=None):
    return {"kind": "cell", "method": method, "in_flight": in_flight, "gbs": gbs, "name": name}


def test_gen3_x16_is_15_75_and_gen5_x16_is_63():
    assert report.link_gbs(3, 16) == pytest.approx(15.754, abs=0.01)
    assert report.link_gbs(5, 16) == pytest.approx(63.015, abs=0.01)
    with pytest.raises(ValueError):
        report.link_gbs(2, 16)


def test_bdp_is_latency_times_rate():
    assert report.bdp_bytes(711, 50.0) == 35550  # the Gen5 figure the plan argues from
    assert report.bdp_bytes(711, 15.75) == 11198


def test_knee_is_the_fewest_bytes_within_five_percent_of_the_best():
    points = [(4096, 5.0), (8192, 11.8), (16384, 12.2), (32768, 12.3)]
    assert report.knee(points) == 8192  # 11.8 >= 0.95 * 12.3 = 11.685
    assert report.knee([]) is None


def test_above_ceiling_cells_are_reported():
    cells = [_cell("sm_cv16", 32768, 12.3), _cell("sm_cv16", 65536, 16.1)]
    assert report.above_ceiling(cells, 15.75) == [cells[1]]


def test_summarize_names_methods_knees_and_the_named_cells():
    records = [
        _meta(),
        _cell("sm_cv16", 8192, 11.8), _cell("sm_cv16", 32768, 12.3, name="s_pattern"),
        _cell("ce_each", 0, 13.6),
        {"kind": "latency", "serial_acquire_ns": 711.0, "device_acquire_ns": 117.0},
        {"kind": "pingpong", "rtt_ns_p50": 4000, "rtt_ns_min": 3500, "rounds": 64},
        {"kind": "fresh", "method": "sm_cv16", "fresh": True},
        {"kind": "fresh", "method": "nc_control", "fresh": False},
    ]
    s = report.summarize(records)
    assert s["theoretical_gbs"] == pytest.approx(15.75, abs=0.01)
    assert s["measured_ceiling_gbs"] == 13.6
    assert s["methods"]["sm_cv16"]["knee_bytes"] == 8192
    assert s["methods"]["ce_each"]["knee_bytes"] is None  # a copy-engine call has no in-flight knob
    assert s["named"] == {"s_pattern": 12.3}
    # Primary: the round trip (min), which is what bounds a copy with a fixed number of bytes in flight; the serial
    # acquire understates it (divix01: 16 KiB / 11.59 GB/s = 1.41 us = the 1408 ns ping-pong minimum).
    assert s["bdp_bytes"] == report.bdp_bytes(3500, 13.6)
    assert s["bdp_acquire_bytes"] == report.bdp_bytes(711.0, 13.6)
    assert s["unsafe"] == [] and s["control_blind"] is False


def test_a_method_that_failed_its_fresh_check_is_unsafe_and_a_passing_control_is_blind():
    records = [
        _meta(), _cell("tma", 65536, 12.0),
        {"kind": "fresh", "method": "tma", "fresh": False},
        {"kind": "fresh", "method": "nc_control", "fresh": True},
    ]
    s = report.summarize(records)
    assert s["unsafe"] == ["tma"]
    assert s["control_blind"] is True


def test_summarize_needs_exactly_one_meta():
    with pytest.raises(ValueError):
        report.summarize([_cell("sm_cv16", 1, 1.0)])


def test_bdp_falls_back_to_the_acquire_without_a_round_trip():
    s = report.summarize([_meta(), _cell("sm_cv16", 8192, 12.0),
                          {"kind": "latency", "serial_acquire_ns": 700.0, "device_acquire_ns": 100.0}])
    assert s["bdp_bytes"] == s["bdp_acquire_bytes"] == report.bdp_bytes(700.0, 12.0)


def test_curves_list_each_methods_points_by_bytes_in_flight():
    s = report.summarize([_meta(), _cell("sm_cv16", 32768, 12.3), _cell("sm_cv16", 8192, 11.8),
                          _cell("sm_cv16", 32768, 12.1), _cell("ce_each", 0, 13.6)])
    # best GB/s at each in-flight size, ascending; copy-engine methods have no in-flight knob
    assert s["curves"] == {"sm_cv16": [(8192, 11.8), (32768, 12.3)]}


def test_a_missing_control_is_blind_and_a_swept_method_without_a_fresh_record_is_unsafe():
    records = [
        _meta(), _cell("sm_cv16", 16384, 11.8), _cell("tma", 32768, 12.3), _cell("ce_each", 0, 13.6),
        _cell("cw_real", 4096, 3.8), _cell("sm_small", 32768, 6.7),
        {"kind": "fresh", "method": "sm_cv16", "fresh": True},
    ]
    s = report.summarize(records)
    assert s["control_blind"] is True  # no nc_control record: nothing shows the check can see staleness
    assert s["unsafe"] == ["tma"]  # swept, never fresh-checked; ce*, cw_real and sm_small have no fresh check
