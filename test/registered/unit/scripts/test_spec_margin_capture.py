"""The prefetch margin capture's server env and its counters summary (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _capture():
    spec = importlib.util.spec_from_file_location(
        "spec_margin_capture", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin_capture.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_capture_scores_where_asked_and_otherwise_runs_the_prefetch_arm():
    cap = _capture()
    cpu, gpu = cap.server_env("/out", False), cap.server_env("/out", False, "gpu")
    assert (cpu["SGLANG_DSV41_RAM_PREFETCH_SCORER"], gpu["SGLANG_DSV41_RAM_PREFETCH_SCORER"]) == ("cpu", "gpu")
    assert {k: v for k, v in gpu.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"} == {
        k: v for k, v in cpu.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"
    }
    assert gpu["SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX"] == "/out/events"


def test_the_counter_summary_reads_the_last_counters_line(tmp_path):
    cap = _capture()
    log = tmp_path / "server.log"
    first = {"spec_scored": 1, "spec_score_ns": 1, "spec_used": 1, "spec_promoted": 1}
    last = {"spec_scored": 10, "spec_score_ns": 50_000, "spec_used": 40, "spec_promoted": 4, "spec_late": 2}
    log.write_text("".join("exl3 RAM miss thread counters " + json.dumps(c) + "\n" for c in (first, last)))
    s = cap.counter_summary(str(log))
    assert s["wait_or_score_us_per_record"] == pytest.approx(5.0)
    assert s["in_flight_at_use"] == pytest.approx(0.1) and s["spec_late"] == 2
    assert cap.counter_summary(str(tmp_path / "missing.log")) == {}
