"""The T3 quick-check report: session pairing, counters, text status and the label (CPU)."""

import importlib.util
import json
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DSPARK = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark")
LABEL = "quick check: one order, 4 sessions, cross-day baseline; not a verdict"


def _report():
    spec = importlib.util.spec_from_file_location("l2_t3_quick_report", os.path.join(DSPARK, "l2_t3_quick_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(session_id, step_ms, ms_per_token):
    verifies = 10
    return {
        "session_id": session_id,
        "ttft": 1.0,
        "total": 1.0 + step_ms / 1000.0 * verifies,
        "decode_tokens_per_sec": 1000.0 / ms_per_token,
        "completion_tokens": verifies * 3,
        "spec_tokens_details": {"spec_verify_ct": verifies},
    }


# Real results.jsonl rows carry corpus ids, not the indices passed to --sessions.
IDS = tuple(f"cfq-train-Single_X/201{i}/page_{i}.pdf-1" for i in range(6))


def _write_dir(path, arms, text=None, sessions=6):
    """arms: {arm: (step_ms, ms_per_token, (issued, landed, used))}; the first `sessions` ids are written."""
    summary = {}
    for arm, (step_ms, ms_per_token, (issued, landed, used)) in arms.items():
        run = path / "servers" / arm / "run-a"
        run.mkdir(parents=True)
        rows = [_row(sid, step_ms, ms_per_token) for sid in IDS[:sessions]]
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[arm] = {
            "drive_flags": [],
            "ram": {"spec_issued": issued, "spec_landed": landed, "spec_used": used},
            "text_vs_reference": {"pass": True} if text is None else text,
        }
    (path / "summary.json").write_text(json.dumps(summary))
    return str(path)


def _dirs(tmp_path, r, text=None):
    counters = (100, 80, 50)
    f108 = _write_dir(
        tmp_path / "f108",
        {r.MISSCUT: (100.0, 50.0, (0, 0, 0)), r.L2: (80.0, 40.0, counters)},
    )
    quick = _write_dir(
        tmp_path / "quick",
        {r.L2: (82.0, 41.0, counters), r.T3: (60.0, 30.0, (120, 90, 70))},
        text=text,
        sessions=4,
    )
    return quick, f108


def test_ratios_pair_by_session_id_over_sessions_0_to_3_and_t3_is_read_against_the_fresh_l2(tmp_path):
    """Mutant: T3 compared against F108's L+2 instead of the fresh L+2, F108's extra sessions left in, or rows
    filtered by integer index (real ids are corpus strings, so every table came out empty) -- red."""
    r = _report()
    quick, f108 = _dirs(tmp_path, r)
    result = r.analyse(quick, f108)
    t3_vs_fresh = result["pairs"]["t3_vs_fresh_l2"]
    assert t3_vs_fresh["other"] == r.QUICK_L2
    assert [row["session_id"] for row in t3_vs_fresh["sessions"]] == sorted(IDS[:4])
    assert [row["session_id"] for row in result["pairs"]["t3_vs_f108_misscut"]["sessions"]] == sorted(IDS[:4])
    assert abs(t3_vs_fresh["step_ratio_median"] - 60.0 / 82.0) < 1e-9
    assert abs(result["pairs"]["t3_vs_f108_misscut"]["step_ratio_median"] - 0.6) < 1e-9
    assert abs(result["pairs"]["fresh_l2_vs_f108_l2_drift"]["step_ratio_median"] - 82.0 / 80.0) < 1e-9
    assert abs(result["pairs"]["l2_vs_f108_misscut"]["step_ratio_median"] - 0.82) < 1e-9


def test_counters_waste_is_landed_minus_used_and_text_failure_is_shown(tmp_path):
    """Mutant: wasted reported as issued - used, or a failed text compare shown as pass -- red."""
    r = _report()
    quick, f108 = _dirs(tmp_path, r, text={"pass": False})
    counters = r.analyse(quick, f108)["counters"]
    assert counters[r.QUICK_T3]["wasted"] == 20
    assert counters[r.QUICK_T3]["issued"] == 120
    assert counters[r.QUICK_T3]["text_status"] == "FAIL"
    assert counters[r.F108_L2]["text_status"] == "pass"


def test_the_label_heads_the_report_and_every_section_and_f108_is_not_written(tmp_path):
    """Mutant: a table without the label, or a file written under the F108 dir -- red."""
    r = _report()
    quick, f108 = _dirs(tmp_path, r)
    before = sorted(os.listdir(f108))
    out = tmp_path / "out"
    assert r.main([quick, "--f108", f108, "--out", str(out)]) == 0
    assert sorted(os.listdir(f108)) == before
    report = (out / "REPORT.md").read_text()
    assert LABEL in report.split("\n## ")[0]
    for section in report.split("\n## ")[1:] + report.split("\n### ")[1:]:
        assert LABEL in section
    assert len(report.split("\n### ")) - 1 == len(r.PAIRS)
    results = json.loads((out / "results.json").read_text())
    assert results["label"] == LABEL
