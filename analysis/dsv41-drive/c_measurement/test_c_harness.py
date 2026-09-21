"""CPU tests for c_harness.py and its hand-off to c_analysis.py. No CUDA. Run: python3 test_c_harness.py"""
import json, sys, tempfile, importlib, os, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_harness as h, c_analysis as a

def test_segments():
    assert sum(b for _, b in h.SEGMENTS) == h.ROW_BYTES == 13_315_584 and len(h.SEGMENTS) == 6

def test_ring_and_reuse():
    ring = h.ring_ids(150, 7); assert sorted(ring) == list(range(150))
    seq = list(itertools.chain.from_iterable(h.draw(ring, i * 4, 4) for i in range(200)))       # 800 consumptions, n = 4
    assert h.min_reuse_distance(seq) == 150                                                        # exactly the ring period
    assert h.min_reuse_distance([1, 2, 3]) == 10 ** 9
    assert h.min_reuse_distance([5, 6, 5]) == 2
    # the gate constant is at most the ring period, so a ring consumption can never trip it
    assert a.MIN_REUSE_ROWS <= h.ROWS_PER_NODE

def test_abba_covers_every_cell_twice():
    import random
    order = h.abba(h.cell_list(), random.Random(1)); n = len(h.cell_list())
    assert len(order) == 2 * n and [c for c, v in order[:n]] == [c for c, v in reversed(order[n:])]
    assert {c for c, _ in order} == set(h.cell_list())

def test_cells_match_the_registered_arms():
    cells = h.cell_list()
    cold = [c for c in cells if (c.engine, c.state, c.load, c.launch) == ("sm", "cold", "idle", "eager")]
    assert sorted({c.node for c in cold}) == [0, 1] and sorted({c.n for c in cold}) == [1, 2, 3, 4, 5, 6]
    assert {c.n for c in h.load_cell_list()} == {1, 3, 6} and {c.node for c in h.load_cell_list()} == {0, 1}
    assert any(c.launch == "graph" and c.n == 3 for c in cells) and any(c.engine == "ce" for c in cells) and any(c.state == "hot" for c in cells) and any(c.state == "repeat" for c in cells)

def _dry(tmp, with_nvme):
    argv = ["--out", str(tmp), "--dry-run"] + (["--with-nvme", "--reader-cpus", "0"] if with_nvme else [])
    assert h.main(argv) == 0

def test_dry_run_feeds_the_analysis():
    with tempfile.TemporaryDirectory() as d:
        _dry(d, True)
        lines = [json.loads(l) for l in open(Path(d) / "results.jsonl")]
        keys = {"process", "engine", "state", "node", "load", "launch", "n", "row_bytes", "T_ms", "distinct_rows", "min_reuse_distance_rows", "link_gen_start", "link_gen_end", "pstate_start", "other_gpu_procs", "foreign_max_core_pct"}
        assert all(keys <= set(x) for x in lines)
        assert {x["process"] for x in lines} == set(range(h.PASSES)) and all(len(x["T_ms"]) == h.LAUNCHES_PER_CELL for x in lines)
        a.simulate = lambda T, *_: (36.0, 4.9, 0.0)                     # the trace model needs the divix01 traces; the gates and fits do not
        r = a.analyse(lines)
        assert r["verdict"] != "INVALID", r["gates"]
        fit = r["fits"][("sm", "cold", 0, "idle", "eager")]
        assert abs(fit[1] - 1.08) < 0.01 and abs(fit[0] - 0.006) < 0.005, fit                # recovers the synthetic law
        loaded = r["T_ms"][("sm", "cold", 0, "nvme", "eager")]; idle = r["T_ms"][("sm", "cold", 0, "idle", "eager")]
        assert abs(loaded[3] / idle[3] - 1.08) < 0.02                                        # the load arm is seen and distinct
        meta = json.loads((Path(d) / "meta.json").read_text()); assert meta["dry_run"] is True and meta["load_windows"]

def test_each_analysis_gate_fires_on_bad_harness_output():
    with tempfile.TemporaryDirectory() as d:
        _dry(d, False)
        good = [json.loads(l) for l in open(Path(d) / "results.jsonl")]
    a.simulate = lambda T, *_: (36.0, 4.9, 0.0)
    for name, edit in (("row bytes", lambda x: x.update(row_bytes=2_764_808)), ("reuse", lambda x: x.update(min_reuse_distance_rows=3)),
                       ("link", lambda x: x.update(link_gen_start=1)), ("foreign", lambda x: x.update(foreign_max_core_pct=50.0)),
                       ("other gpu", lambda x: x.update(other_gpu_procs=1)), ("tail", lambda x: x.update(T_ms=x["T_ms"][:190] + [x["T_ms"][0] * 3] * 10))):
        bad = json.loads(json.dumps(good)); edit(bad[0]); assert a.analyse(bad)["verdict"] == "INVALID", name
    assert a.analyse(good)["verdict"] != "INVALID"

def test_foreign_load_sees_a_spinning_process():
    import subprocess, time
    p = subprocess.Popen([sys.executable, "-c", "while True: pass"]); time.sleep(0.2)
    f = h.ForeignLoad([os.getpid()]); f.start(); time.sleep(1.0); pct, name = f.stop(); p.kill()
    assert pct > 50.0, (pct, name)

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests: t(); print("ok", t.__name__)
    print("ALL PASSED", len(tests))
