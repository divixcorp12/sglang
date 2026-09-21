#!/usr/bin/env python3
"""T(n): the cost of the production per-layer gather at counts 1..6, and the delta of a count-1 launch.

Task 6 / PER_ROW_TRANSFER.md OPEN 1 / C_MEASUREMENT_PREREG.md section 0. The per-row mechanism launches
`copy_expert_row_segments_gpu` once per row, so what matters is the function T(n), not one bandwidth number.
delta = how much more a count-1 launch costs than the batched launches predict (T(1) minus the line through n = 2..6; the "T(1) - T(k)/k" wording of OPEN 1 is reported beside it); per-row pays it once per
read request (13.97 per decode step), so `delta = 80 us` is 1.1 ms/step, the whole of G* = 1.114 ms.

THIS IS A THIN DRIVER, NOT A SECOND INSTRUMENT. The instrument is c_measurement/c_harness.py's `RealDevice` (frozen by
hash in C_MEASUREMENT_PREREG.md section 9), imported and NOT modified: production gather kernel, the six real segments
(13,315,584 B per row), pinned slabs from the service's own allocator, a permutation ring with reuse distance 150 rows
(2.0 GB = 20.9 x the 96 MiB L2), 64 destination slots, a 600 us spin before every timed launch so host launch latency
stays outside T. What this file adds and c_harness.py does not do:
  * no gates: clocks, P-state, link generation, other GPU processes and load average are RECORDED per visit and
    reported, never used to refuse (run 1 of c_harness.py was INVALID on its P-state gate; see C_MEASUREMENT_PREREG.md
    section 20 item 1). Whatever this prints is therefore NOT a registered result;
  * the graph launch at every n = 1..6 (the production path is a graph node), not only n = 3;
  * the L2 control (`repeat`) and the copy-engine yardstick (`ce`) at every n;
  * delta reported directly, in us and in ms/step, with the sanity check against the link spec.

    gpu-run.sh python gather_tn.py run     --repo <tree under test> --out DIR [--nodes 0] [--passes 3]
    python gather_tn.py analyse DIR/results.jsonl.gz            (CPU only, stdlib only)

The interpreter trap: PYTHONPATH must point at <repo>/python; this script refuses if sglang resolves anywhere else.
"""
import argparse, collections, gzip, hashlib, json, os, random, statistics as S, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c_measurement"))

ROW_BYTES = 13_315_584
NS = (1, 2, 3, 4, 5, 6)
READS_PER_STEP = 13.97          # read requests per decode step (PER_ROW_TRANSFER.md; C_MEASUREMENT_PREREG.md section 0)
G_STAR_MS = 1.114               # the most extra per-step cost best-order per-row can carry (PER_ROW_TRANSFER.md)
GEN3_X16_GBS = 15.75            # the link spec; anything above it is impossible and means the working set was cached
CE_GBS = 13.79                  # copy engine, NC_VISIBILITY.md
SEED = 20260921


def cells():
    out = []
    for n in NS:
        out.append(("sm", "cold", "eager", n))
        out.append(("sm", "cold", "graph", n))
        out.append(("sm", "repeat", "eager", n))      # L2 control: the same rows every launch
        out.append(("ce", "cold", "eager", n))        # copy engine yardstick, not the production path
    return out


def box_state():
    d = {"loadavg": open("/proc/loadavg").read().split()[:3]}
    try:
        d["gpu_apps"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip().splitlines()
        d["gpu_mem_used_mib"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pstate,clocks.sm,clocks.max.sm",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        d["nvidia_smi_error"] = str(e)
    return d


def run(a):
    import c_harness as H
    assert H.ROW_BYTES == ROW_BYTES
    repo = Path(a.repo).resolve()
    dev = H.RealDevice(str(repo))
    meta = {"script": "gather_tn.py", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": str(repo),
            "repo_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "c_harness_sha256": hashlib.sha256((HERE.parent / "c_measurement" / "c_harness.py").read_bytes()).hexdigest(),
            "gather_tn_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "launches_per_visit": a.launches, "passes": a.passes, "nodes": a.nodes, "box_start": box_state()}
    setup_args = argparse.Namespace(nodes=tuple(int(x) for x in a.nodes.split(",")))
    meta.update(dev.setup(setup_args))
    import sglang
    meta["sglang_file"] = sglang.__file__
    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit("INTERPRETER TRAP: sglang imported from %s, not under %s; set PYTHONPATH=%s/python" % (sglang.__file__, repo, repo))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if a.check_only:
        print("check-only ok", meta["sglang_file"]); return 0
    smi = H.SmiSampler(os.getpid()); smi.start(); time.sleep(2.0)
    for node in setup_args.nodes:                        # untimed load so the link leaves Gen1
        for _ in range(3): dev.run_visit(H.Cell("sm", "cold", node, "idle", "eager", 6), 40, a)
    rng = random.Random(SEED)
    f = gzip.open(out / "results.jsonl.gz", "wt")
    for p in range(a.passes):
        for node in setup_args.nodes:
            base = cells()
            first = list(base); rng.shuffle(first)
            order = [(c, 0) for c in first] + [(c, 1) for c in reversed(first)]      # ABBA
            for (engine, state, launch, n), visit in order:
                cell = H.Cell(engine, state, node, "idle", launch, n)
                t0 = time.monotonic()
                T, extra = dev.run_visit(cell, a.launches, a)
                t1 = time.monotonic()
                cond = smi.window(t0, t1) or {"link_gen_start": None, "link_gen_end": None, "pstate_start": None, "sm_mhz_min": None,
                                              "sm_mhz_max": None, "other_gpu_procs": None}
                rec = H.record(cell, p, T, dev.reuse_distance(node), cond, (0.0, "not judged"), extra)
                rec.update({"visit": visit, "loadavg1": float(open("/proc/loadavg").read().split()[0]), "wall_s": t1 - t0})
                f.write(json.dumps(rec) + "\n"); f.flush()
        print("pass %d/%d done  load %s" % (p + 1, a.passes, open("/proc/loadavg").read().split()[:3]), flush=True)
    f.close(); smi.stop()
    meta["box_end"] = box_state(); meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return analyse_file(out / "results.jsonl.gz")


# ------------------------------------------------------------------------------------------------ analysis (stdlib)

def ols(pts):
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]; n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def stat(xs, kind):
    return {"p50": S.median(xs), "min": min(xs), "p90": pct(xs, 0.9)}[kind]


def deltas(Tn, kind):
    """Tn: {n: [ms samples]}. Returns the fit and the deltas in microseconds, from the `kind` statistic (p50 or min)."""
    t = {n: stat(v, kind) for n, v in Tn.items()}
    # The line is fitted on the BATCHED counts n >= 2 only and extrapolated to n = 1, so an excess at count 1 (a latency
    # limit, a fixed cost the batch amortises) shows up as a residual instead of being smeared into the fit. Per-row's gain over
    # two-phase on a request of m rows is T(m) - T(1); the model books it as (m-1)*c_m; delta is what T(1) carries beyond the
    # line, i.e. the shortfall per request. (If T is exactly linear, delta = 0 and f is a fixed launch cost that per-row's
    # hidden launches pay but its exposed one does not.)
    f, cm = ols([(n, t[n]) for n in sorted(t) if n >= 2])
    d = {"T_ms": t, "f_us": f * 1e3, "c_marginal_ms": cm,
         "delta_marginal_us": (t[1] - (f + cm)) * 1e3,                   # HEADLINE: T(1) minus the line through n = 2..6 at n = 1
         "delta_share_us": {k: (t[1] - t[k] / k) * 1e3 for k in sorted(t) if k > 1},   # T(1) - T(k)/k: the wording of PER_ROW_TRANSFER OPEN 1; includes f/k
         "linearity_at_1": abs(t[1] - (f + cm)) / t[1]}
    return d


def analyse(recs, boots=400):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    conds = collections.defaultdict(list)
    for r in recs:
        by[(r["engine"], r["state"], r["launch"], r["node"])][r["n"]].extend(r["T_ms"])
        conds[(r["engine"], r["state"], r["launch"], r["node"])].append(r)
    lines, rng = [], random.Random(SEED)
    allc = [r for r in recs]
    lines.append("CONDITIONS (recorded, not gated)")
    lines.append("  cells %d; loadavg1 min/median/max %.1f/%.1f/%.1f; pstate seen %s; SM MHz min/max over cells %s/%s" % (
        len(allc), min(r["loadavg1"] for r in allc), S.median(r["loadavg1"] for r in allc), max(r["loadavg1"] for r in allc),
        sorted({r["pstate_start"] for r in allc}), min((r["sm_mhz_min"] or 0) for r in allc), max((r["sm_mhz_max"] or 0) for r in allc)))
    lines.append("  link gen start/end seen %s/%s; other GPU processes seen %s; min reuse distance (rows) %s" % (
        sorted({r["link_gen_start"] for r in allc}, key=str), sorted({r["link_gen_end"] for r in allc}, key=str),
        sorted({r["other_gpu_procs"] for r in allc}, key=str), min(r["min_reuse_distance_rows"] for r in allc)))
    for key in sorted(by):
        engine, state, launch, node = key
        Tn = by[key]
        if set(Tn) != set(NS): continue
        lines.append("")
        lines.append("== %s / %s / %s / node %s  (%d samples per n)" % (engine, state, launch, node, len(Tn[1])))
        lines.append("   n   T p50 ms   T min ms   T p90 ms   GB/s@p50  GB/s@min   per-row p50 ms (T/n)")
        for n in NS:
            v = Tn[n]; p50, mn = S.median(v), min(v)
            gb = lambda t, n=n: n * ROW_BYTES / (t * 1e-3) / 1e9
            flag = "  IMPOSSIBLE (> %.2f GB/s link spec: cached or mis-timed)" % GEN3_X16_GBS if gb(mn) > GEN3_X16_GBS and engine == "sm" else ""
            lines.append("  %2d  %9.4f  %9.4f  %9.4f  %8.2f  %8.2f   %9.4f%s" % (n, p50, mn, pct(v, .9), gb(p50), gb(mn), p50 / n, flag))
        if engine != "sm" or state != "cold": continue
        for kind in ("p50", "min"):
            d = deltas(Tn, kind)
            lines.append("  fit on %s over n=2..6: T(n) = f + c_m*n   f = %.1f us   c_m = %.4f ms/row (%.2f GB/s)   |T(1)-(f+c_m)|/T(1) = %.1f%%" % (
                kind, d["f_us"], d["c_marginal_ms"], ROW_BYTES / (d["c_marginal_ms"] * 1e-3) / 1e9, 100 * d["linearity_at_1"]))
            dm = d["delta_marginal_us"]
            lines.append("  delta (%s) = T(1) - (f + c_m) = %+.1f us  ->  x %.2f reads/step = %+.3f ms/step   (G* = %.3f ms)" % (kind, dm, READS_PER_STEP, dm * READS_PER_STEP / 1e3, G_STAR_MS))
            for k, v in d["delta_share_us"].items():
                lines.append("      vs batched share T(%d)/%d: delta = %+.1f us -> %+.3f ms/step" % (k, k, v, v * READS_PER_STEP / 1e3))
        # bootstrap 95% interval over launches, p50 statistic, for the marginal delta and the delta against T(6)/6
        bm, b6 = [], []
        for _ in range(boots):
            res = {n: rng.choices(Tn[n], k=len(Tn[n])) for n in NS}
            d = deltas(res, "p50"); bm.append(d["delta_marginal_us"]); b6.append(d["delta_share_us"][6])
        bm.sort(); b6.sort()
        lines.append("  bootstrap 95%% over launches (p50): delta [%+.1f, %+.1f] us, delta_vs_T(6)/6 [%+.1f, %+.1f] us" % (
            bm[int(.025 * boots)], bm[int(.975 * boots)], b6[int(.025 * boots)], b6[int(.975 * boots)]))
        # per-pass spread (does one pass agree with the next?)
        per_pass = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in conds[key]: per_pass[r["process"]][r["n"]].extend(r["T_ms"])
        dp = [deltas(per_pass[p], "p50")["delta_marginal_us"] for p in sorted(per_pass) if set(per_pass[p]) == set(NS)]
        lines.append("  per-pass delta (p50), us: %s" % ", ".join("%+.1f" % x for x in dp))
    # L2 control and eager-vs-graph, side by side
    lines.append("")
    lines.append("CONTROLS (p50 ms): cold vs repeat (L2 control: repeat visibly faster means L2 holds host reads), eager vs graph")
    lines.append("   n   cold-eager  repeat-eager  cold-graph   repeat/cold   graph-eager (us)")
    for node in sorted({k[3] for k in by}):
        for n in NS:
            g = lambda e, s, l: S.median(by[(e, s, l, node)][n]) if (e, s, l, node) in by and n in by[(e, s, l, node)] else float("nan")
            ce, re, cg = g("sm", "cold", "eager"), g("sm", "repeat", "eager"), g("sm", "cold", "graph")
            lines.append("  %2d  %10.4f  %12.4f  %10.4f   %10.3f   %+9.1f" % (n, ce, re, cg, re / ce, (cg - ce) * 1e3))
    return "\n".join(lines)


def analyse_file(path):
    with gzip.open(path, "rt") as f: recs = [json.loads(l) for l in f]
    print(analyse(recs)); return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--repo", required=True); r.add_argument("--out", required=True)
    r.add_argument("--nodes", default="0", help="NUMA node(s) of the pinned source slabs, comma separated; 2.0 GB each")
    r.add_argument("--passes", type=int, default=3); r.add_argument("--launches", type=int, default=100)
    r.add_argument("--check-only", action="store_true")
    a = sub.add_parser("analyse"); a.add_argument("results")
    ns = p.parse_args()
    return run(ns) if ns.cmd == "run" else analyse_file(ns.results)


if __name__ == "__main__":
    sys.exit(main())
