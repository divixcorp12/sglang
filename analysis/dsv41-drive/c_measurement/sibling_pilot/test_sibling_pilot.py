"""CPU tests for the sibling pilot. Run: python3 test_sibling_pilot.py"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sibling_pilot_analysis as an

def test_selftest_verdicts():
    assert an.selftest() == 0

def test_visit_validity_rules():
    ok = {"arm": "B", "launch_foreign_pct": 1.0, "sib_foreign_pct": 2.0, "sib_spinner_pct": 97.0, "spinner_ticks": 40}
    assert an.visit_ok(ok)
    assert not an.visit_ok({**ok, "sib_spinner_pct": 60.0})                     # spinner descheduled: the B arm would be an A arm
    assert not an.visit_ok({**ok, "sib_foreign_pct": 12.0})                     # a foreign burst on the sibling
    assert not an.visit_ok({**ok, "launch_foreign_pct": 30.0})
    a = {**ok, "arm": "A", "sib_spinner_pct": 0.0, "spinner_ticks": 0}
    assert an.visit_ok(a) and not an.visit_ok({**a, "spinner_ticks": 3}) and not an.visit_ok({**a, "sib_foreign_pct": 15.0})   # foreign on the sibling in A makes A look like B

def test_dry_run_plumbing():
    cpus = sorted(os.sched_getaffinity(0)); L = cpus[0]
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["taskset", "-c", str(L), sys.executable, str(HERE / "sibling_pilot.py"), "--dry-run", "--out", d, "--repo", "/x", "--launch-cpu", str(L)], capture_output=True, text=True, timeout=200)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = [json.loads(x) for x in open(Path(d, "visits.jsonl"))]
        assert len(rows) == 20 * 2 * 4 and {x["arm"] for x in rows} == {"A", "B"} and all(len(x["T_ms"]) == 100 for x in rows)
        assert max(x["spinner_ticks"] for x in rows if x["arm"] == "A") == 0
        assert json.loads(Path(d, "meta.json").read_text())["sibling_cpu"] != L

if __name__ == "__main__":
    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]: t(); print("ok", t.__name__)
    print("ALL PASSED")
