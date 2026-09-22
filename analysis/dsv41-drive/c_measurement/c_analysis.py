#!/usr/bin/env python3
"""Frozen analysis for the c measurement (C_MEASUREMENT_PREREG.md). CPU only, stdlib only.

Input: JSONL, one line per (process, arm, n) cell:
  {"process": int, "engine": "sm"|"ce", "state": "cold"|"hot"|"repeat", "node": 0|1, "load": "idle"|"nvme", "launch": "eager"|"graph",
   "n": int, "row_bytes": int, "T_ms": [float, ...], "distinct_rows": int, "min_reuse_distance_rows": int,
   "link_gen_start": int, "link_gen_end": int, "pstate_start": int, "other_gpu_procs": int, "foreign_max_core_pct": float}
  T_ms are GPU-timeline times of ONE gather launch (or one cudaMemcpyAsync batch for engine "ce") moving n rows, event-timed on an idle stream.
Usage:  c_analysis.py results.jsonl [--traces NAME_TIMINGS NAME_LANES]     gates, T(n) tables, fits, the model recompute, the label
        c_analysis.py --selftest                                          synthetic T(n) with known c; asserts labels and gates (needs the divix01 traces)
"""
import json, sys, os, random, itertools, collections, statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "per_row_recompute"))

# ---- registered constants (frozen with this file's hash) ----
ROW_BYTES = 13_315_584                    # six streamed tensors per pinned row (DSV41_REFERENCE 16.2)
CE_CEILING_GBS, SPEC_GBS = 13.79, 15.75   # measured cudaMemcpyAsync (NC_VISIBILITY) and Gen3 x16 theoretical
NS = (1, 2, 3, 4, 5, 6)                   # production lanes never exceed 6 (schema-4 trace: max lanes 6)
MIN_REUSE_ROWS = 100                      # 100 rows = 1.33 GB = 13.9 x the 96 MiB L2 (about 7 rows fit): no row is re-read within the cell
TAIL_TOL = 1.25                           # p99 / p50 per cell
GRAPH_EAGER_TOL = 0.03
LIN_TOL = 0.03                            # |T(1) - (f + c_m)| / T(1)
FOREIGN_MAX_PCT = 10.0
E_TOTAL, E_C = 159.55, 20.86              # extra triples per step, and the hidden-if-fits class (g_exposure_counts.py)
BAR_MS = 0.015 * 254.4
G_LO_US, G_HI_US = 8.0, 14.0              # the plan's assumed range for g
STEPS = 511
BOOT, SEED = 1000, 20260921

def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def fit(T):                                # T: {n: ms} -> (f_ms, c_ms) least squares
    xs = sorted(T); mx = sum(xs) / len(xs); my = sum(T[x] for x in xs) / len(xs)
    c = sum((x - mx) * (T[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
    return my - c * mx, c

def tfun(T):                               # piecewise-linear T(n) for n >= 0 from the six medians
    f, c = fit(T)
    return lambda n: T[n] if n in T else (0.0 if n == 0 else f + c * n)

def chain(costs, readies, t0):
    e = t0
    for cost, rd in zip(costs, readies): e = max(e, rd) + cost
    return e

def open_reqs(name):
    D = "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-drive/task1-results/"
    out, steps = [], {}
    for line in open(D + name + ".trace"):
        d = json.loads(line)
        if d.get("kind") == "graph_step": steps[d["forward"]] = d
        elif d.get("kind") == "ram_miss_request": out.append(d)
    return steps, out

def simulate(T, timings="task1-2-new-on-T", lanes="task1f-0-new-on-T"):
    """Returns (V1 ceiling ms/step, gross ms/step, nonlinearity penalty ms/step, extra-launch fixed cost included in g ms/step)."""
    sA, rA = open_reqs(timings); sF, rF = open_reqs(lanes)
    assert len(rA) == len(rF) and all((a["layer"], a["request"]["type"], a["rows_asked"], a["status"]) == (b["layer"], b["request"]["type"], b["rows_asked"], b["status"]) for a, b in zip(rA, rF)), "streams differ"
    Tf = tfun(T); f, c = fit(T); t1 = Tf(1)
    gross = ceil = nl = 0.0
    for a, b in zip(rA, rF):
        if a["forward"] not in sA: continue
        k = b["request"]["lanes"]; m = a["rows_asked"]; h = max(k - m, 0)
        if m == 0:
            if k >= 2: nl += (k * t1 - Tf(k)) - (k - 1) * f      # per-row launches k singles where two-phase launches one batch of k
            continue
        sg = a["stages_ns"]
        if a["status"] != "served" or a["untraced"]["rows"] != 0 or len(a["row_pack_ns"]) != m: continue
        k = max(m, k); h = k - m
        t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
        R = sorted(r["end"] / 1e6 for r in a["row_pack_ns"])
        hit_end = (max(t0, res) + Tf(h)) if h else 0.0
        t_two = max(done, hit_end) + Tf(m)
        t_pr = chain([t1] * k, [res] * h + R, t0)               # hits first, then miss rows in arrival order (best order)
        gross += t_two - t_pr; ceil += Tf(h) if h else 0.0
    return ceil / STEPS, gross / STEPS, nl / STEPS

def label(gross, nl):
    S_ = gross - nl
    gx = (S_ - BAR_MS) / E_TOTAL * 1000.0; gh = (S_ - BAR_MS) / (E_TOTAL - E_C) * 1000.0
    if gh <= G_LO_US: lab = "STANDS-FOR-ALL-ASSUMED-g"
    elif gx >= G_HI_US: lab = "WITHDRAWN-FOR-ALL-ASSUMED-g"
    else: lab = "INTERMEDIATE"
    return S_, gx, gh, lab

def analyse(lines, timings=None, lanes=None):
    cells = collections.defaultdict(lambda: collections.defaultdict(dict))     # arm key -> process -> n -> list
    gates = []
    for d in lines:
        key = (d["engine"], d["state"], d["node"], d["load"], d["launch"])
        cells[key][d["process"]][d["n"]] = d["T_ms"]
        tag = "%s/%s/n%s/%s/%s p%s n=%s" % (key + (d["process"], d["n"]))
        if d["row_bytes"] != ROW_BYTES: gates.append("row bytes %s != %s (%s)" % (d["row_bytes"], ROW_BYTES, tag))
        if d["state"] != "repeat" and d["engine"] == "sm" and d["min_reuse_distance_rows"] < MIN_REUSE_ROWS: gates.append("rows re-read within %d rows: L2/cache-resident risk (%s)" % (d["min_reuse_distance_rows"], tag))
        if d["link_gen_start"] != 3 or d["link_gen_end"] != 3 or d["pstate_start"] != 0: gates.append("link gen/pstate not 3/P0 (%s)" % tag)
        if d["other_gpu_procs"] != 0: gates.append("another process on the GPU (%s)" % tag)
        if d["foreign_max_core_pct"] > FOREIGN_MAX_PCT: gates.append("foreign process above %.0f%% of a core (%s)" % (FOREIGN_MAX_PCT, tag))
        if pct(d["T_ms"], 0.99) / S.median(d["T_ms"]) > TAIL_TOL: gates.append("p99/p50 > %.2f (%s)" % (TAIL_TOL, tag))
    med = {k: {p: {n: S.median(v) for n, v in cs.items()} for p, cs in ps.items()} for k, ps in cells.items()}
    # sanity: no implied bandwidth above the measured copy engine (x1.03) or the spec; SM n>=2 at least 8 GB/s
    ce = {}
    for k, ps in med.items():
        if k[0] == "ce":
            for p, T in ps.items():
                for n, t in T.items(): ce.setdefault((k[1], k[2], p, n), n * ROW_BYTES / (t * 1e6) )
    for k, ps in med.items():
        for p, T in ps.items():
            for n, t in T.items():
                bw = n * ROW_BYTES / (t * 1e6)                                   # GB/s (bytes / (ms * 1e6))
                if bw > SPEC_GBS: gates.append("implied %.2f GB/s exceeds the Gen3 x16 spec: an L2 or timing artefact (%s p%s n=%s)" % (bw, k, p, n))
                if k[0] == "sm" and k[1] != "repeat" and bw > CE_CEILING_GBS * 1.03: gates.append("SM %.2f GB/s exceeds the measured copy-engine ceiling %.2f (%s p%s n=%s)" % (bw, CE_CEILING_GBS, k, p, n))
                if k[0] == "sm" and n >= 2 and bw < 8.0: gates.append("SM %.2f GB/s below 8: harness or box broken (%s p%s n=%s)" % (bw, k, p, n))
    need = [k for k in med if k[0] == "sm" and k[1] == "cold" and k[3] == "idle" and k[4] == "eager"]
    if len(need) < 2: gates.append("need sm/cold/idle/eager for both nodes")
    for k in need:
        if len(med[k]) < 5: gates.append("fewer than 5 interleaved passes for %s" % (k,))
        for p, T in med[k].items():
            if set(T) != set(NS): gates.append("missing n in %s p%s" % (k, p))
    # graph vs eager agreement at n = 3
    for k in list(med):
        if k[4] == "graph":
            ke = k[:4] + ("eager",)
            for p in med[k]:
                if ke in med and p in med[ke] and 3 in med[k][p] and 3 in med[ke][p]:
                    a, b = med[k][p][3], med[ke][p][3]
                    if abs(a - b) / max(a, b) > GRAPH_EAGER_TOL: gates.append("graph %.3f vs eager %.3f ms at n=3 differ by >%.0f%% (%s p%s)" % (a, b, 100 * GRAPH_EAGER_TOL, k, p))
    out = {"gates": sorted(set(gates)), "T_ms": {}, "fits": {}}
    for k in med:
        allp = med[k]; Tm = {n: S.median(allp[p][n] for p in allp if n in allp[p]) for n in NS if any(n in allp[p] for p in allp)}
        out["T_ms"][k] = Tm
        if len(Tm) == len(NS):
            f, c = fit(Tm); lin = abs(Tm[1] - (f + c)) / Tm[1]
            out["fits"][k] = (f, c, lin, [n * ROW_BYTES / (Tm[n] * 1e6) for n in NS])
            if k[0] == "sm" and k[1] == "cold" and lin > LIN_TOL: out.setdefault("flags", []).append("T(1) is %.1f%% off the line f + c*n for %s: count=1 is not the marginal cost" % (100 * lin, k))
    if out["gates"]:
        out["verdict"] = "INVALID"; return out
    out["model"] = {}
    for k in need:
        Tm = out["T_ms"][k]
        ceil, gross, nl = simulate(Tm, timings or "task1-2-new-on-T", lanes or "task1f-0-new-on-T")
        S_, gx, gh, lab = label(gross, nl)
        out["model"][k] = {"c_marginal_ms": out["fits"][k][1], "f_ms": out["fits"][k][0], "T1_ms": Tm[1], "v1_ceiling_ms_per_step": ceil, "gross_ms_per_step": gross, "nonlinearity_penalty_ms": nl,
                           "net_before_g_ms": S_, "g_star_all_exposed_us": gx, "g_star_C_hidden_us": gh, "label": lab}
    labs = {v["label"] for v in out["model"].values()}
    out["verdict"] = labs.pop() if len(labs) == 1 else "STRADDLES-NODES"
    return out

def selftest():
    rng = random.Random(3)
    def synth(c, f=0.006, bw_cap=1.0, gen=3, foreign=0.0, reuse=4000, rows=13_315_584, tail=0.004):
        L = []
        for p in range(5):
            for node in (0, 1):
                for eng in ("sm",):
                    for n in NS:
                        cc = c * (1.0 if node == 0 else 1.04)                       # node 1 4% slower, as a synthetic asymmetry
                        T = [(f + cc * n) * (1 + abs(rng.gauss(0, tail))) for _ in range(200)]
                        L.append({"process": p, "engine": eng, "state": "cold", "node": node, "load": "idle", "launch": "eager", "n": n, "row_bytes": rows, "T_ms": T,
                                  "distinct_rows": 300, "min_reuse_distance_rows": reuse, "link_gen_start": gen, "link_gen_end": gen, "pstate_start": 0, "other_gpu_procs": 0, "foreign_max_core_pct": foreign})
        return L
    cases = [("c=1.00 stands", synth(1.00), "STANDS-FOR-ALL-ASSUMED-g"), ("c=1.055 intermediate", synth(1.055), "INTERMEDIATE"),
             ("c=1.15 intermediate", synth(1.15), "INTERMEDIATE"), ("c=1.40 withdrawn", synth(1.40), "WITHDRAWN-FOR-ALL-ASSUMED-g"),
             ("impossible bandwidth", synth(0.80), "INVALID"), ("idle link", synth(1.055, gen=1), "INVALID"), ("row reuse", synth(1.055, reuse=5), "INVALID"),
             ("foreign load", synth(1.055, foreign=40.0), "INVALID"), ("wrong row size", synth(1.055, rows=2_764_808), "INVALID")]
    ok = True
    for name, lines, want in cases:
        r = analyse(lines); got = r["verdict"]; ex = ""
        if "model" in r:
            m = r["model"][next(iter(r["model"]))]; ex = " c_m %.3f gross %.2f g*_X %.2f g*_H %.2f v1 %.1f" % (m["c_marginal_ms"], m["gross_ms_per_step"], m["g_star_all_exposed_us"], m["g_star_C_hidden_us"], m["v1_ceiling_ms_per_step"])
        print("%-24s want %-28s got %-28s %s%s" % (name, want, got, "ok" if got == want else "MISMATCH", ex)); ok &= got == want
    print("SELFTEST", "PASSED" if ok else "FAILED"); return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest": sys.exit(selftest())
    args = sys.argv[1:]; tr = None
    if "--traces" in args: i = args.index("--traces"); tr = args[i + 1:i + 3]; args = args[:i]
    lines = [json.loads(l) for l in open(args[0])]
    r = analyse(lines, *(tr or [None, None]))
    print("VERDICT:", r["verdict"])
    for g in r["gates"]: print("  gate:", g)
    # amendment 9. A marker beside the input is printed here, after the verdict word and before anything else, so its
    # SCOPE is always in view: section 12 item 3(b) voids the `nvme` arm's rho, not the run, and an unconditional
    # refusal would permanently block a run whose only fault is rho.
    marker = os.path.join(os.path.dirname(os.path.abspath(args[0])), "results.INVALID")
    if os.path.exists(marker):
        print("  marker:", open(marker).read().strip())
    # amendment 9. On an INVALID verdict no fit, flag or model line is printed, whatever the caller does with the
    # output. Run 1 of `c` was collected with `tail`, which showed the fits before line 1 had been read, so fitted
    # values for a voided run were seen; the answer to "I saw something I should not have" is to make seeing it
    # impossible. This can only withhold numbers, never admit them.
    if r["verdict"] == "INVALID":
        print("  (INVALID: no fit, flag or model line is printed)")
        sys.exit(3)
    for k, f in r["fits"].items(): print("  fit %s: f %.4f ms, c_marginal %.4f ms, T(1) off-line %.3f, GB/s by n %s" % (k, f[0], f[1], f[2], ["%.2f" % x for x in f[3]]))
    for fl in r.get("flags", []): print("  flag:", fl)
    for k, m in r.get("model", {}).items(): print("  model", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in m.items()})
