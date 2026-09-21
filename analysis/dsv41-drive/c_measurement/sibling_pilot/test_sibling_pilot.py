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

def _visits_file(d):
    rows = [{"rep": r, "n": n, "arm": arm, "T_ms": [1.0] * 100, "sib_spinner_pct": 99.0 if arm == "B" else 0.0, "spinner_ticks": 50 if arm == "B" else 0,
             "sib_foreign_pct": 1.0, "launch_foreign_pct": 1.0, "window_s": 0.5} for n in (3, 6) for r in range(10) for arm in "ABBA"]
    f = Path(d, "visits.jsonl"); f.write_text("".join(json.dumps(x) + "\n" for x in rows)); return f

def test_analysis_refuses_beside_an_invalid_marker_and_computes_without_one():
    with tempfile.TemporaryDirectory() as d:
        f = _visits_file(d)
        ok = subprocess.run([sys.executable, str(HERE / "sibling_pilot_analysis.py"), str(f)], capture_output=True, text=True)
        assert ok.returncode == 0 and "VERDICT: INSENSITIVE" in ok.stdout                                  # no marker: computes as before
        Path(d, "INVALID").write_text("a lane of ours at the END: [(1, 'x')]\n")
        bad = subprocess.run([sys.executable, str(HERE / "sibling_pilot_analysis.py"), str(f)], capture_output=True, text=True)
        assert bad.returncode == 3 and bad.stdout.strip().splitlines()[0] == "VERDICT: INVALID" and "lane of ours" in bad.stdout
        assert "INSENSITIVE" not in bad.stdout and "SENSITIVE" not in bad.stdout and "shift" not in bad.stdout and "valid_reps" not in bad.stdout     # nothing else is printed

def test_analysis_diff_only_adds_a_refusal():
    """The rule is untouched: the constants and the verdict logic are byte-identical to the pre-refusal file (its hash is in the prereg, section 9 note)."""
    src = (HERE / "sibling_pilot_analysis.py").read_text()
    for frag in ("BAND = 0.005; MIN_VALID = 8; FOREIGN_MAX = 10.0; SPIN_MIN = 90.0", "if all(lo >= -BAND and hi <= BAND for lo, hi in cis): out[\"verdict\"] = \"INSENSITIVE\"",
                 "elif any(lo > BAND or hi < -BAND for lo, hi in cis): out[\"verdict\"] = \"SENSITIVE\"", "T975[len(shifts) - 1] * S.stdev(shifts) / len(shifts) ** 0.5"):
        assert frag in src

def test_dry_run_plumbing():
    cpus = sorted(os.sched_getaffinity(0)); L = cpus[0]
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["taskset", "-c", str(L), sys.executable, str(HERE / "sibling_pilot.py"), "--dry-run", "--out", d, "--repo", "/x", "--launch-cpu", str(L)], capture_output=True, text=True, timeout=200)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = [json.loads(x) for x in open(Path(d, "visits.jsonl"))]
        assert len(rows) == 20 * 2 * 4 and {x["arm"] for x in rows} == {"A", "B"} and all(len(x["T_ms"]) == 100 for x in rows)
        assert max(x["spinner_ticks"] for x in rows if x["arm"] == "A") == 0
        assert json.loads(Path(d, "meta.json").read_text())["sibling_cpu"] != L


def test_stdlib_attributes_used_by_the_scripts_exist():
    """The class of defect the first real launch found (os.sched_getcpu does not exist): every `module.attr` the scripts use on an imported stdlib module must exist. Catches typos and platform-only names on a CPU box."""
    import ast, importlib, sys as _s
    for path in [HERE / "sibling_pilot.py", HERE / "sibling_pilot_analysis.py"]:
        tree = ast.parse(Path(path).read_text()); mods = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in _s.stdlib_module_names: mods[(a.asname or a.name).split(".")[0]] = a.name
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in mods:
                m = importlib.import_module(mods[n.value.id])
                assert hasattr(m, n.attr), "%s: %s.%s does not exist" % (path, n.value.id, n.attr)


if __name__ == "__main__":
    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]: t(); print("ok", t.__name__)
    print("ALL PASSED")
