"""drive_busy.py drive-load: per-root busy shares by kind and their overlap from a server log's last
``exl3 RAM miss drive load`` line (CPU)."""

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


def _root(demand_busy, spec_busy, overlap, demand_bytes, spec_bytes):
    return {"demand_reads": 0, "spec_reads": 0, "demand_inflight_bytes": 0, "spec_inflight_bytes": 0,
            "demand_bytes": demand_bytes, "spec_bytes": spec_bytes, "demand_busy_ns": demand_busy,
            "spec_busy_ns": spec_busy, "overlap_ns": overlap}


def _line(elapsed, roots):
    return "exl3 RAM miss drive load " + json.dumps({"elapsed_ns": elapsed, "clock_reads": 7, "roots": roots})


def test_the_last_line_gives_each_roots_shares_and_bytes():
    stale = _line(1_000, [_root(1, 1, 1, 1, 1)])
    log = "\n".join([
        "INFO starting", stale, "exl3 RAM miss thread counters {}",
        "[2026-10-09 12:00:00] " + _line(10_000_000_000, [
            _root(4_000_000_000, 1_000_000_000, 500_000_000, 3_000_000_000, 1_000_000_000),
            _root(2_000_000_000, 0, 0, 1_000_000_000, 0),
        ]),
    ])
    got = _module().drive_load_report(log)
    assert got["lines"] == 2 and got["elapsed_s"] == 10.0 and got["clock_reads"] == 7
    first, second = got["roots"]
    assert first == {"root": 0, "demand_busy_share": 0.4, "spec_busy_share": 0.1, "overlap_share": 0.05,
                     "overlap_of_demand": 0.125, "demand_gb": 3.0, "spec_gb": 1.0, "bytes_share": 0.8,
                     "in_flight_at_log": 0}
    assert second["demand_busy_share"] == 0.2 and second["overlap_of_demand"] == 0.0 and second["bytes_share"] == 0.2


def test_a_root_that_served_no_demand_has_no_overlap_ratio():
    got = _module().drive_load_report(_line(100, [_root(0, 50, 0, 0, 10)]))
    assert got["roots"][0]["overlap_of_demand"] is None and got["roots"][0]["spec_busy_share"] == 0.5


def test_a_log_without_the_line_is_refused():
    with pytest.raises(ValueError, match="no 'exl3 RAM miss drive load' line"):
        _module().drive_load_report("exl3 RAM miss thread counters {}\n")


def test_the_cli_writes_the_report(tmp_path, capsys):
    log = tmp_path / "server.log"
    log.write_text(_line(1_000, [_root(500, 0, 0, 10, 0)]) + "\n")
    out = tmp_path / "report.json"
    module = _module()
    import sys

    argv = sys.argv
    sys.argv = ["drive_busy.py", "drive-load", str(log), "--json", str(out)]
    try:
        assert module.main() == 0
    finally:
        sys.argv = argv
    assert json.loads(out.read_text())["roots"][0]["demand_busy_share"] == 0.5
