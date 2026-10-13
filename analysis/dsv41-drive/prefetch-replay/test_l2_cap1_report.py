"""Tests for l2_cap1_report: labels on every section, the report's fixed statements, atomic JSON output."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_cap1_report  # noqa: E402

SHA_T0 = "e6f7661d" + "0" * 56
SHA_T3 = "e675e2bd" + "1" * 56
SCORERS = ("gate_cap1", "L2_T0", "L2_T3")


def _bootstrap() -> dict:
    return {"diff": 0.05, "lo": 0.01, "hi": 0.09, "ratio": 1.2, "precision_diff": -0.01, "useful_diff": 12.0,
            "useful_lo": 3.0, "useful_hi": 20.0}


def _budget(*, learned: bool) -> dict:
    point = {"budget": 0.25, "threshold": math.inf, "reads": 10, "forwards": 40, "mean_reads": 0.25, "useful": 4,
             "wasted": 6, "eligible": 20, "recall": 0.2, "precision": 0.4}
    out = {"point": point,
           "coverage": {"abstention_rate": 0.5, "fully_covered": 2, "partly_covered": 1},
           "dspark": {"rowsets": {"anchor_only": {"eligible_all": 5, "useful": 1, "recall": 0.2}},
                      "positions": {"0": {"miss_rows": 5, "covered": 1, "coverage": 0.2}},
                      "widths": {"6": {"useful": 4, "eligible_all": 20, "recall": 0.2}}},
           "candidate_split": {}}
    if learned:
        out["matched"] = {"target_reads": 10, "achieved_reads": 9, "bootstrap": _bootstrap(),
                          "gain_split": {"inside": {"cand": 3, "gate": 2, "diff": 1},
                                         "outside": {"cand": 1, "gate": 0, "diff": 1}}}
    return out


def _cap1() -> dict:
    return {split: {n: {"curve": [], "budgets": {"0.25": _budget(learned=n != "gate_cap1")}} for n in SCORERS}
            for split in ("dev", "calib")}


def _result(*, timing: dict | None = None) -> dict:
    cell = {"on_time_useful": None, "late_useful": 1, "useful": 3, "wasted": 2, "reads": 5, "recall": 0.1}
    default_timing = {"measured": {"assumptions": {"scorer_delay_ms": {"*": 0.0}},
                                   "cases": {"nominal": {"gate_cap1": cell}}, "calib": {}},
                      "zero_delay": None}
    return l2_cap1_report.build_result(
        run_dir="/mnt/nvme1/nvfp4-work/b1-r1-20261012", out="/mnt/nvme1/nvfp4-work/l2-cap1-x",
        commands=["python l2_cap1_eval.py RUN --dataset D"], plan_sha="ab" * 32,
        ckpts={"T0_seed0": {"path": "T0_seed0.pt", "sha": SHA_T0}, "T3_seed0": {"path": "T3_seed0.pt", "sha": SHA_T3}},
        sanity={"checked": 6, "status": "reproduced"}, cap1=_cap1(), uncovered={"dev": {"groups": 2}},
        timing=default_timing if timing is None else timing,
        costs={"L2_T0": {"cap": 1, "tokens": 6, "gate_ms": 0.1, "gate_only_ms": 0.05}},
        frozen={"L2_T0": {"budget": 0.5, "reference": 10, "tolerance": 0.05, "rule": "r",
                          "on_time_useful_by_budget": {0.25: 5, 0.5: 10}}},
        group_counts={"dev": {"covered": 38, "uncovered": 2}, "calib": {"covered": 38, "uncovered": 2}})


def test_every_section_is_labelled() -> None:
    result = _result()
    assert "frozen_candidate" in result
    for name, section in result.items():
        assert section["label"] in l2_cap1_report.LABELS, name
    assert result["timing"]["label"] == "replay estimate"
    assert result["serving_interface"]["label"] == "hypothesis"
    assert result["cap1"]["label"] == "observed offline"


def test_report_holds_fixed_statements_and_pending() -> None:
    text = l2_cap1_report.render_report(_result())
    assert "No ordinary-decode dataset exists" in text
    for name in SCORERS:
        assert name in text
    assert SHA_T0 in text and SHA_T3 in text
    assert "pending" in text
    assert "38 of 40" in text


def test_missing_timing_renders_pending() -> None:
    text = l2_cap1_report.render_report(_result(timing={}))
    assert "pending" in text


def test_budget_zero_read_nothing_row_shows_inf_not_pending() -> None:
    result = _result()
    for threshold in (None, math.inf):
        b0 = _budget(learned=False)
        b0["point"].update(budget=0.0, threshold=threshold, reads=0, mean_reads=0.0, useful=0, wasted=0, recall=0.0,
                           precision=0.0)
        result["cap1"]["splits"]["dev"]["gate_cap1"]["budgets"] = {"0.0": b0}
        rows = [line for line in l2_cap1_report.render_report(result).splitlines()
                if line.startswith("| gate_cap1 | 0.0 | inf (read nothing) |")]
        assert len(rows) == 1
        assert "pending" not in rows[0]



def test_write_outputs_is_valid_json_with_null_for_inf(tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    result_path, report_path = l2_cap1_report.write_outputs(out, _result())
    loaded = json.loads(Path(result_path).read_text())
    assert loaded["cap1"]["splits"]["dev"]["L2_T0"]["budgets"]["0.25"]["point"]["threshold"] is None
    assert Path(report_path).read_text().startswith("# ")
    assert sorted(os.listdir(out)) == ["REPORT.md", "result.json"]
