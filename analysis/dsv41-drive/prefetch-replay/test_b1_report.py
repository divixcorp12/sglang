"""Tests for b1_report: render a small constructed results dict."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_report  # noqa: E402


def _cell(recall, precision, never):
    return {"recall": recall, "precision": precision, "never_recovery": never}


def _table(scale):
    keys = {"pooled": _cell(0.5 * scale, 0.4 * scale, 0.1 * scale),
            "pooled_noflag": _cell(0.6 * scale, 0.5 * scale, 0.2 * scale),
            "layer:19": _cell(0.3 * scale, 0.2 * scale, 0.0),
            "stratum:code": _cell(0.7 * scale, 0.6 * scale, 0.3 * scale)}
    return {"cells": {"reads=1": keys, "reads=2": keys, "cut=0.5": keys}}


def _results():
    return {
        "horizons": {"1": {"tables": {"gate": _table(1.0), "E2": _table(1.2)}}},
        "learning_curve": {"E2": {"15": 0.1, "30": 0.2, "60": 0.25}},
        "bootstrap": {"median": 0.1, "lo": 0.02, "hi": 0.2},
        "cost": {"bytes_bf16": 1000, "median_ms": 0.05, "within_0_1ms": True},
        "verdict": {"advance": False, "arms": {"precision_gain": 0.05}, "reasons": ["gain below +0.15"]},
        "frozen_sha256": "abc",
        "paths": {"dataset": "/mnt/nvme1/x"},
    }


def test_render_contains_sections_and_values():
    text = b1_report.render(_results())
    assert "### L+1" in text
    assert "| gate | reads=1 | 0.500 | 0.400 | 0.100 |" in text
    assert "| E2 | cut=0.5 |" in text
    assert "DO NOT ADVANCE" in text
    assert "gain below +0.15" in text
    assert "[0.020, 0.200]" in text
    assert "| 19 |" in text and "| code |" in text
    assert "abc" in text


def test_missing_values_render_pending():
    text = b1_report.render({})
    assert "pending" in text
    assert "Verdict: pending" in text
    assert "| E5 | pending | pending | pending |" in text


def test_main_writes_file(tmp_path):
    src = tmp_path / "r.json"
    out = tmp_path / "r.md"
    src.write_text(json.dumps(_results()))
    assert b1_report.main([str(src), "--out", str(out)]) == 0
    assert out.read_text() == b1_report.render(_results())
