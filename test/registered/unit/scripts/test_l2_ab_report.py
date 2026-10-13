"""The served L+2 A/B report: the pre-registered verdict and the per-session step time (CPU)."""

import importlib.util
import json
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _report():
    spec = importlib.util.spec_from_file_location(
        "l2_ab_report", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "l2_ab_report.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(session_id, step_s, verifies=10, ms_per_token=50.0, ttft=1.0):
    return {
        "session_id": session_id,
        "ttft": ttft,
        "total": ttft + step_s * verifies,
        "decode_tokens_per_sec": 1000.0 / ms_per_token,
        "completion_tokens": verifies * 3,
        "spec_tokens_details": {"spec_verify_ct": verifies},
    }


def _write_order(tmp_path, name, arms, flags=None, in_flight=None, text=None):
    """arms: {arm: [row]}; flags: {arm: [session_id]}; in_flight: {arm: fraction}; text: text_vs_reference of each arm."""
    order = tmp_path / name
    summary = {}
    for arm, rows in arms.items():
        run = order / "servers" / arm / "run-b"
        run.mkdir(parents=True)
        (order / "servers" / arm / "run-a").mkdir()
        (order / "servers" / arm / "run-a" / "results.jsonl").write_text(json.dumps(_row(99, 9.0)) + "\n")
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[arm] = {
            "drive_flags": [{"session_id": s} for s in (flags or {}).get(arm, [])],
            "ram": {"spec_landed": 100, "spec_used": 60},
            "spec_in_flight_at_use": (in_flight or {}).get(arm),
            "text_vs_reference": {"pass": True} if text is None else text,
        }
    (order / "summary.json").write_text(json.dumps(summary))
    return str(order)


def test_verdict_wins_only_when_both_orders_pass_the_ratio_and_the_paired_median_and_loses_only_when_both_fail():
    """Mutant: win on one order, a 0.971 ratio, or a paired median against L+2 -- red; lose on one order -- red."""
    r = _report()
    assert r.verdict([0.96, 0.97], [3.0, 1.0], [0.96, 0.97]) == "win"
    assert r.verdict([0.96, 0.99], [3.0, 1.0], [0.96, 0.99]) == "tie"
    assert r.verdict([0.96, 0.971], [3.0, 1.0], [0.96, 0.971]) == "tie"
    assert r.verdict([0.90, 0.90], [3.0, -0.5], [0.90, 0.90]) == "tie"
    assert r.verdict([1.03, 1.10], [-3.0, -1.0], [1.03, 1.10]) == "lose"
    assert r.verdict([1.03, 1.02], [-3.0, -1.0], [1.03, 1.02]) == "tie"


def test_a_drive_flagged_session_is_left_out_of_the_median_and_the_step_uses_verifies_not_tokens(tmp_path):
    """Mutant: median over every session, or step time divided by completion tokens -- red."""
    r = _report()
    floors = [_row(0, 0.10), _row(1, 0.10), _row(2, 0.10)]
    # Session 1 of the L+2 arm is drive-flagged and 3x slower; sessions 0 and 2 are 0.8x.
    l2 = [_row(0, 0.08), _row(1, 0.30), _row(2, 0.08)]
    order = _write_order(
        tmp_path, "fwd", {r.FLOORS: floors, r.L2: l2}, flags={r.L2: [1]}, in_flight={r.FLOORS: 0.4, r.L2: 0.1}
    )
    result = r.analyse_order(order)
    pair = result["pairs"][f"{r.L2}__vs__{r.FLOORS}"]
    assert pair["sessions_used"] == 2
    assert abs(pair["step_ratio_median"] - 0.8) < 1e-9
    by_session = {row["session_id"]: row for row in pair["sessions"]}
    assert by_session[1]["excluded"] is True
    # (total - ttft) / spec_verify_ct = 0.08 s -> 80 ms; the earlier run-a rows are not read.
    assert abs(by_session[0][f"step_ms_{r.L2}"] - 80.0) < 1e-9


def test_the_report_files_carry_the_verdict_and_say_whether_the_in_flight_fraction_fell(tmp_path):
    """Mutant: the in-flight comparison inverted or left out of the report -- red."""
    r = _report()

    def arms(l2_step):
        return {
            r.MISSCUT: [_row(0, 0.12), _row(1, 0.12)],
            r.FLOORS: [_row(0, 0.10), _row(1, 0.10)],
            r.IDLE2: [_row(0, 0.11), _row(1, 0.11)],
            r.L2: [_row(0, l2_step, ms_per_token=45.0), _row(1, l2_step, ms_per_token=45.0)],
        }

    in_flight = {r.FLOORS: 0.38, r.IDLE2: 0.39, r.L2: 0.05}
    fwd = _write_order(tmp_path, "fwd", arms(0.09), in_flight=in_flight)
    rev = _write_order(tmp_path, "rev", arms(0.09), in_flight=in_flight)
    out = tmp_path / "out"
    assert r.main([fwd, rev, "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["combined"]["verdict"] == "win"
    report = (out / "REPORT.md").read_text()
    assert "FELL from L+1 to L+2" in report
    assert "## Verdict: win" in report

    flat = {r.FLOORS: 0.38, r.IDLE2: 0.39, r.L2: 0.37}
    rev2 = _write_order(tmp_path, "rev2", arms(0.09), in_flight=flat)
    assert r.in_flight_reading(
        {"forward": r.analyse_order(fwd), "reversed": r.analyse_order(rev2)}
    )["text"].startswith("the in-flight fraction did NOT fall")


def test_a_faster_l2_arm_whose_text_differs_from_the_reference_is_not_a_win(tmp_path):
    """Mutant: the verdict from step ratios alone, ignoring text_vs_reference -- red."""
    r = _report()
    arms = {r.FLOORS: [_row(0, 0.10), _row(1, 0.10)], r.L2: [_row(0, 0.08), _row(1, 0.08)]}
    rev = _write_order(tmp_path, "rev", arms)
    for i, text in enumerate(({"pass": False}, {"error": "no flips"}, "missing reference probe")):
        fwd = _write_order(tmp_path, f"fwd{i}", arms, text=text)
        combined = r.combine({"forward": r.analyse_order(fwd), "reversed": r.analyse_order(rev)})
        assert combined["verdict"] == "tie"
        assert "text" in combined["note"]
