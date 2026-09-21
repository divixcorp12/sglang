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

def _spin_on(cpu):
    import subprocess
    return subprocess.Popen(["taskset", "-c", str(cpu), sys.executable, "-c", "while True: pass"])

def test_foreign_load_counts_only_the_cores_we_use_and_not_ours():
    import time
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 3
    mine, other = cpus[0], cpus[1]
    on_mine, on_other = _spin_on(mine), _spin_on(other); time.sleep(0.3)
    try:
        f = h.ForeignLoad([os.getpid()], {mine}); f.start(); time.sleep(1.0); pct, where = f.stop()
        assert pct > 50.0 and ("cpu%d" % mine) in where, (pct, where)                    # a spinner on our core is seen
        f2 = h.ForeignLoad([os.getpid()], {cpus[2]}); f2.start(); time.sleep(1.0); f2.stop()
        assert f2.last["box_foreign_cores"] >= 1.5, f2.last                              # box-wide, both spinners are foreign (other load on the machine can only add)
        f3 = h.ForeignLoad([os.getpid(), on_mine.pid, on_other.pid], {mine, other}); f3.start(); time.sleep(1.0); pct3, _ = f3.stop()
        assert pct3 <= pct - 30.0, (pct, pct3)                                             # declaring the spinners ours removes them from the count
    finally:
        on_mine.kill(); on_other.kill()

def test_foreign_load_counts_cores_we_used_not_the_whole_mask():
    import time
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 2
    ours, spare = cpus[0], cpus[1]
    mine, foreign_spin = _spin_on(ours), _spin_on(spare); time.sleep(0.3)               # our spinner on `ours`; someone else's spinner on the spare core of our mask
    try:
        no_smt = lambda c: (c,)                                                           # isolate the used-core rule from the sibling rule (tested separately)
        f = h.ForeignLoad([os.getpid(), mine.pid], {ours, spare}, siblings=no_smt); f.start(); time.sleep(1.0); pct, where = f.stop()
        assert f.last["used_cores"] == [ours] and pct < 60.0, (f.last, pct, where)      # foreign CPU on the spare core is not counted: we did not run there
        f = h.ForeignLoad([os.getpid()], {spare}, siblings=no_smt); f.start(); time.sleep(1.0); pct2, _ = f.stop()
        assert pct2 > 50.0                                                                # with no presence of ours the whole set counts (a survey stays conservative)
    finally:
        mine.kill(); foreign_spin.kill()

def test_foreign_load_resolution_floor():
    import time
    for ticks, clamped in ((2, True), (3, False)):
        f = h.ForeignLoad([os.getpid()], {0}); vals = iter([{0: 100}, {0: 100 + ticks}]); f._cpu = lambda: next(vals); f._own = lambda: {}
        f.start(); time.sleep(0.05); pct, _ = f.stop()
        assert (pct <= 9.9) == clamped, (ticks, pct)            # 2 ticks in 50 ms is 40% but below MIN_TICKS: not evidence; 3 ticks counts

def test_top_scan_names_a_spinner():
    import time
    sp = _spin_on(sorted(os.sched_getaffinity(0))[0]); time.sleep(0.3)
    try:
        top = h.TopScan([os.getpid()]).sample(1.0)
        assert any(t["cpu_pct"] > 50 and "while True" in t["cmdline"] for t in top), top
    finally: sp.kill()

def test_visit_retry_is_environmental_and_logged():
    for seq, want_retries in (([], 0), ([25.0, 0.0], 1), ([30.0, 30.0, 30.0] + [0.0] * 5000, 0)):     # the last: every attempt of the first visit fails, then clean
        h.DRY_FOREIGN_SEQ[:] = list(seq)
        with tempfile.TemporaryDirectory() as d:
            assert h.main(["--out", d, "--dry-run"]) == 0
            r = [json.loads(l) for l in open(Path(d, "retries.jsonl"))]; m = json.loads(Path(d, "meta.json").read_text())
            if seq == []: assert r == [] and m["retried_visits"] == 0
            elif seq[:2] == [25.0, 0.0]: assert len(r) == 1 and "foreign 25.0%" in r[0]["why"][0] and m["retried_visits"] == 1
            else:
                assert len(r) == 2 and m["retried_visits"] == 2                       # attempts 1 and 2 logged; the third is kept and left to the frozen gate
                kept = [json.loads(l) for l in open(Path(d, "results.jsonl"))]
                assert any(x["foreign_max_core_pct"] == 30.0 for x in kept)
    h.DRY_FOREIGN_SEQ[:] = []

def test_drive_accounting_helpers():
    assert abs(h.drive_gb_per_s({"nvme0n1": 0}, {"nvme0n1": 2_000_000}, 1.0) - 1.024) < 1e-6         # 2e6 sectors x 512 B in 1 s
    assert h.drive_gb_per_s({}, {}, 1.0) == 0.0
    try: got = h.nvme_sectors_read()
    except OSError: return
    assert all(k.startswith("nvme") and "p" not in k.split("n", 1)[1] for k in got)               # whole devices only, no partitions

def test_cpu_list_parser():
    assert h._parse_cpus("18-20,54") == [18, 19, 20, 54] and h._parse_cpus("") == []

def test_box_foreign_gate():
    # the gate: foreign CPU that differs between the idle and the load arms by more than a core writes results.INVALID (exit 3)
    for shift, want_rc in ((0.0, 0), (2.5, 3)):
        h.DRY_BOX_SHIFT = shift
        with tempfile.TemporaryDirectory() as d:
            rc = h.main(["--out", d, "--dry-run", "--with-nvme", "--reader-cpus", "0"])
            assert rc == want_rc and (Path(d, "results.INVALID").exists() == (want_rc == 3)), (shift, rc)
            assert "box_foreign_cores_idle_vs_load_by_pass" in json.loads(Path(d, "meta.json").read_text())
    h.DRY_BOX_SHIFT = 0.0

def test_meta_records_not_measured_and_load_fields():
    with tempfile.TemporaryDirectory() as d:
        assert h.main(["--out", d, "--dry-run", "--skip-arms", "hot"]) == 0
        m = json.loads(Path(d, "meta.json").read_text())
        assert "hot" in m["NOT_MEASURED"] and "nvme" in m["NOT_MEASURED"]                    # nvme is reported not measured when it was not run
        line = json.loads(open(Path(d, "results.jsonl")).readline())
        assert {"box_foreign_cores", "loadavg_start", "loadavg_end", "attempts"} <= set(line)

def _sib_map():
    """This box's layout as measured 2026-09-21: 2 threads per core; cpu c and c+36 (mod 72) are siblings, so 28-31 pair with 64-67 and 35 with 71."""
    return lambda c: tuple(sorted({c % 36, c % 36 + 36}))

def test_quiet_check_candidates_use_the_interleaved_numa_layout_and_exclude_reserved_neighbours():
    import quiet_check as q
    n0, n1 = h._parse_cpus("0-17,36-53"), h._parse_cpus("18-35,54-71")
    hc, rc = q.candidates(n0, n1, _sib_map())
    assert hc == list(range(36, 54))                                                  # no node-0 candidate has a sibling in 64-71
    assert rc == list(range(18, 28)) and all(c not in rc for c in (28, 29, 30, 31, 35))   # 28-31 pair with 64-67, 35 with 71: excluded permanently
    assert q.reserved_neighbours(_sib_map()) == [28, 29, 30, 31, 32, 33, 34, 35]
    assert not any(64 <= c <= 71 for c in rc + hc) and all(c in n0 for c in hc) and all(c in n1 for c in rc)
    assert q.candidates(list(range(0, 36)), list(range(36, 72)), _sib_map())[0] == []        # a contiguous split would give a different answer: the lists come from /sys

def test_pairs_show_both_members_and_score_the_worst():
    import quiet_check as q
    got = q.pairs([42, 44], {42: 5.0, 6: 30.0, 44: 40.0, 8: 2.0}, lambda c: tuple(sorted({c % 36, c % 36 + 36})))
    assert got[0][0] == 42 and got[0][1] == [6] and got[0][2] == 30.0 and got[1][2] == 40.0

def test_sibling_foreign_cpu_is_counted_against_a_used_core():
    import time
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 2
    ours, sib = cpus[0], cpus[1]
    mine, other = _spin_on(ours), _spin_on(sib); time.sleep(0.3)
    try:
        f = h.ForeignLoad([os.getpid(), mine.pid], {ours}, siblings=lambda c: (ours, sib)); f.start(); time.sleep(1.0); pct, where = f.stop()
        assert pct > 50.0 and "siblings" in where, (pct, where, f.last)                  # a foreign thread on the SIBLING of a core we occupy is contention, and is scored
        f = h.ForeignLoad([os.getpid(), mine.pid], {ours}, siblings=lambda c: (c,)); f.start(); time.sleep(1.0); pct2, _ = f.stop()
        assert pct2 < 50.0, pct2                                                          # without the sibling rule the same situation scores as clean (the hole the lead measured)
    finally:
        mine.kill(); other.kill()

def test_rehearsal_measures_windows():
    import quiet_check as q
    cpus = sorted(os.sched_getaffinity(0)); res = q.rehearse(cpus[1:3], cpus[1:2], 2.0, window=0.5)
    assert len(res) == 4 and all(isinstance(x[0], float) for x in res)

def test_watched_nvme_resolves_through_st_dev_and_skips_the_unresolvable():
    assert h.watched_nvme(["/definitely/not/a/path"]) == set()
    got = h.watched_nvme(["/", "/tmp", os.path.expanduser("~")])
    assert all(n.startswith("nvme") and "p" not in n[4:] for n in got)               # whole devices only, never a partition or a mapper name

def test_node_mem_reader():
    try: m = h.node_mem_mib(0)
    except OSError: return
    assert "MemFree" in m and "FilePages" in m

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests: t(); print("ok", t.__name__)
    print("ALL PASSED", len(tests))
