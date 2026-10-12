"""Tests for r1_report: label, empty results, attribution statuses, rendered content."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_report  # noqa: E402


def _results() -> dict:
    return {
        "label": r1_report.LABEL,
        "plan_sha": "abc",
        "stage1": {
            "calib": {"gate_floors": {"cap2": {"useful": 10, "wasted": 5, "eligible": 40,
                                               "precision": 0.66, "recall": 0.25, "reads": 15}}},
            "histograms": {"gate": {"layer:3": {"0": 5, "1": 3, "p50": 1.0, "max": 6}}},
        },
        "timing": {"assumptions": {"contention": {"zero": 1.0, "nominal": 2.0}},
                   "cases": {"nominal": {"gate": {"on_time_useful": 3}}}},
        "arms": {"T0": {"parent": None, "seed": 0, "epochs": 60, "checkpoint_sha256": "c0",
                        "curve": [0.1, 0.2]}},
        "calibration": {"F0": {"recall_cap2": 0.30}, "T0": {"recall_cap2": 0.27},
                        "T1": {"recall_cap2": 0.20}},
        "cost": {"bf16": {"complete": {"median_ms": 0.08}, "head_only": {"median_ms": 0.03}},
                 "fp32": {"complete": {"median_ms": 0.12}, "head_only": {"median_ms": 0.05}},
                 "read_set_diff_fraction": 0.01},
        "stopped_after_stage1": False,
    }


def test_label_and_explanations_are_frozen():
    assert r1_report.LABEL == "exploratory on reused data; independent confirmation pending"
    assert r1_report.EXPLANATIONS == ("ranking", "abstention", "block_aggregation", "capacity",
                                      "training_length", "lead")


def test_attribution_statuses():
    attr = r1_report.attribution(_results())
    assert set(attr) == set(r1_report.EXPLANATIONS)
    assert attr["ranking"]["status"] == "supported"
    assert attr["block_aggregation"]["status"] == "not_supported"
    assert attr["abstention"]["status"] == "unresolved"
    assert attr["lead"]["status"] == "unresolved"
    assert attr["training_length"]["status"] == "not_supported"
    assert attr["capacity"]["status"] == "unresolved"


def test_render_contains_sections_and_label():
    text = r1_report.render(_results())
    assert r1_report.LABEL in text
    for needle in ("Stage 1", "Timing sensitivity", "Complete offline scorer cost",
                   "read_set_diff_fraction", "Continuation gate", "checkpoint sha256", "c0"):
        assert needle in text


def test_render_empty_results_is_pending():
    text = r1_report.render({})
    assert r1_report.LABEL in text
    assert "pending" in text


def test_stopped_after_stage1_is_stated():
    assert "Stopped after Stage 1" in r1_report.render({"stopped_after_stage1": True})


def test_main_writes_file(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(_results()))
    out = tmp_path / "REPORT.md"
    assert r1_report.main([str(src), "--out", str(out)]) == 0
    assert r1_report.LABEL in out.read_text()
