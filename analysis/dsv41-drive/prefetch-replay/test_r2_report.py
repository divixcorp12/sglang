"""Tests for r2_report: sections, pending results, the D12 sentence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r2_report  # noqa: E402

SECTIONS = ("Arms: per-seed", "Selection", "Dev comparison", "Calib points",
            "Timing sensitivity replay", "Scorer cost", "## L+2", "R2 continuation gate", "## Reading")


def _results(*, winner_median: float = 0.40, t3: float = 0.35) -> dict:
    cell = {"recall": 0.3, "precision": 0.5, "reads": 0.9}
    return {
        "label": r1_plan.LABEL,
        "plan_sha": "abc",
        "arms": {"T5": {"0": {"epochs": 12, "dev_recall_cap2": winner_median, "curve": [0.1, 0.2],
                              "checkpoint_sha256": "c5", "parent_sha256": "c3"}},
                 "G0": {"0": {"epochs": 20, "dev_recall_cap2": 0.2, "curve": [0.1],
                              "checkpoint_sha256": "cg", "parent_sha256": None}}},
        "selection": {"winner": "T5", "median_dev_recall": {"T5": winner_median, "G0": 0.2},
                      "t3_median": t3},
        "comparison": {"cap2": {"T5": cell, "G0": cell, "T3": cell, "F0": cell, "gate": cell},
                       "cap1": {"T5": cell}},
        "calib": {"T5": {"threshold": 0.4, "recall_cap2": 0.3, "precision": 0.5, "reads": 0.9}},
        "timing": {"assumptions": {"contention": {"zero": 1.0}},
                   "cases": {"nominal": {"T5": {"on_time_useful": 3}}}},
        "cost": {"G0": {"bf16": {"correction": {"median_ms": 0.02, "runs": 100},
                                 "complete": {"median_ms": 0.1}}}},
        "l2": {"T5": {"recall_cap2": 0.2, "precision": 0.4, "reads": 1.0}},
        "continuation_gate": {"passed": False, "reasons": ["recall below bar"],
                              "values": {"arms": {"T5": {"recall": 0.9, "cost": 0.02}}}},
        "adds_over_t3": winner_median > t3,
    }


def test_every_section_appears():
    text = r2_report.render(_results())
    assert r1_plan.LABEL in text
    for needle in SECTIONS + ("c5", "gate figure", "Curve T5 seed 0"):
        assert needle in text


def test_pending_results_render():
    text = r2_report.render({})
    assert r1_plan.LABEL in text
    assert "pending" in text
    for needle in SECTIONS:
        assert needle in text
    assert r2_report.D12_SENTENCE not in text


def test_d12_sentence_only_when_winner_does_not_beat_t3():
    assert r2_report.D12_SENTENCE not in r2_report.render(_results(winner_median=0.40, t3=0.35))
    assert r2_report.D12_SENTENCE in r2_report.render(_results(winner_median=0.35, t3=0.35))
    assert r2_report.D12_SENTENCE in r2_report.render(_results(winner_median=0.30, t3=0.35))


def test_main_writes_file(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(_results()))
    out = tmp_path / "REPORT.md"
    assert r2_report.main([str(src), "--out", str(out)]) == 0
    assert r1_plan.LABEL in out.read_text()
