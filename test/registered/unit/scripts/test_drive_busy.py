"""The expert drives' busy time and bytes during a served run's decode steps (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _module():
    spec = importlib.util.spec_from_file_location(
        "drive_busy", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "drive_busy.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_stat_line_gives_sectors_read_and_busy_ms():
    m = _module()
    # Fields: reads merged sectors ms writes merged sectors ms in_flight io_ticks time_in_queue ...
    line = "  100 0 2048 50 7 0 16 3 2 900 1200 0 0 0 0 0 0"
    assert m.parse_stat(line) == {"read_bytes": 2048 * 512, "busy_ms": 900, "in_flight": 2}


def _sample(t_ms, read_mb, busy_ms):
    return {"ns": int(t_ms * 1e6), "dev": {"a": {"read_bytes": int(read_mb * 1e6), "busy_ms": busy_ms, "in_flight": 0}}}


def test_a_window_takes_the_overlapping_share_of_each_sample_interval():
    m = _module()
    # 0-100 ms: 10 MB and 50 ms busy; 100-200 ms: 30 MB and 100 ms busy.
    samples = [_sample(0, 0, 0), _sample(100, 10, 50), _sample(200, 40, 150)]
    w = m.window(samples, int(50e6), int(150e6))
    assert w["a"]["read_bytes"] == pytest.approx(5e6 + 15e6)
    assert w["a"]["busy_ms"] == pytest.approx(25 + 50)


def test_all_idle_counts_only_the_time_no_drive_was_busy():
    m = _module()
    # Two drives, one 100 ms interval: a busy 60 ms, b busy 30 ms; at most 40 ms all idle, at least 10.
    s0 = {"ns": 0, "dev": {"a": {"read_bytes": 0, "busy_ms": 0, "in_flight": 0},
                           "b": {"read_bytes": 0, "busy_ms": 0, "in_flight": 0}}}
    s1 = {"ns": int(100e6), "dev": {"a": {"read_bytes": 0, "busy_ms": 60, "in_flight": 0},
                                    "b": {"read_bytes": 0, "busy_ms": 30, "in_flight": 0}}}
    b = m.all_idle_bounds([s0, s1], 0, int(100e6))
    assert b == pytest.approx((10.0, 40.0))
