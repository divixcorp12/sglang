#!/usr/bin/env python3
"""Frozen analysis for the g measurement (G_MEASUREMENT_PREREG.md). CPU only, stdlib only.

Input: a JSONL file, one line per (process, variant, N) cell:
  {"process": int, "variant": str, "N": int, "R": int, "batch_ms": [float, ...],
   "link_gen_start": int, "link_gen_end": int, "sm_mhz_min": int, "sm_mhz_max": int, "other_gpu_procs": int,
   "nodes": int}
  batch_ms[i] is the event-timed wall time of one batch of R back-to-back graph replays; per-replay time = batch_ms[i] / R.
Variants: empty | active_p1 | active_p4 | active_p6 | control20 | empty_base8k
  N is the number of stage triples in the captured graph (3 kernels each; "nodes" must equal 3*N + base).
Usage: g_analysis.py results.jsonl        -> prints the gates, the slopes, the verdict
       g_analysis.py --selftest           -> synthetic data with known g, asserts the verdict logic
Exit code 0 always prints a verdict word; INVALID means no number is to be quoted.
"""
import json, sys, random, statistics as S, collections

# ---- registered constants (frozen with this file's hash; see the pre-registration) ----
E_EMPTY, E_B, E_C, E_D = 85.10, 48.93, 20.86, 4.65      # extra triples per decode step, g_exposure_counts.py on task1f (511 steps)
GROSS_MS = 4.93                                          # per-row over two-phase at best order, measured k, 511 steps (k-free)
BAR_MS = 0.015 * 254.4                                   # 1.5% of the four-new:on-arm mean step time = 3.816 ms
G_STAR_MS = GROSS_MS - BAR_MS                            # 1.114 ms: the most extra stage cost per step best-order per-row can carry
P_REGISTERED = "active_p4"                               # the W_s as designed: fatal, shutdown, RowResult acquire, seqlock re-read
NS_REQUIRED = {0, 40, 80, 160, 320, 640}
NS_BASE8K = {0, 160, 640}                                # the same slope on top of 8,000 filler nodes (production graph size)
BASE_TOL = 0.15                                          # base8k slope must match the bare-chain slope within 15%
CONTROL_US, CONTROL_TOL_US = 20.0, 1.0                   # positive control: +20 us of spin per triple must be recovered
LINEARITY_TOL = 0.10                                     # slope on N<=160 vs N>=160 must agree within 10%
BOOT = 2000                                              # bootstrap resamples over batches
SEED = 20260921

def slope_us(cells, rng):
    """cells: {N: [per-replay ms per batch]}. OLS slope of the per-N median, in microseconds per triple."""
    def fit(med):
        xs = sorted(med); n = len(xs); mx = sum(xs) / n; my = sum(med[x] for x in xs) / n
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (med[x] - my) for x in xs)
        return 1000.0 * sxy / sxx
    med = {N: S.median(v) for N, v in cells.items()}
    point = fit(med); boots = []
    for _ in range(BOOT):
        m = {N: S.median(rng.choices(v, k=len(v))) for N, v in cells.items()}
        boots.append(fit(m))
    boots.sort()
    return point, boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]

def analyse(lines):
    rng = random.Random(SEED)
    gates, notes = [], []
    by = collections.defaultdict(lambda: collections.defaultdict(dict))     # variant -> process -> N -> list
    for d in lines:
        by[d["variant"]][d["process"]][d["N"]] = [b / d["R"] for b in d["batch_ms"]]
        if d["link_gen_start"] != 3 or d["link_gen_end"] != 3: gates.append("PCIe link gen != 3 (process %s, %s, N=%s): idle-link state" % (d["process"], d["variant"], d["N"]))
        if d["other_gpu_procs"] != 0: gates.append("another process used the GPU (process %s, %s, N=%s)" % (d["process"], d["variant"], d["N"]))
        if d["sm_mhz_max"] > 0 and d["sm_mhz_min"] < 0.9 * d["sm_mhz_max"]: gates.append("SM clock varied by >10%% within a cell (%s, N=%s)" % (d["variant"], d["N"]))
        if d["nodes"] != 3 * d["N"] + (8000 if d["variant"] == "empty_base8k" else 0): gates.append("node count %s != 3N (+8000 for the base variant) for %s N=%s: something was optimised away" % (d["nodes"], d["variant"], d["N"]))
    res = {}
    for v, procs in by.items():
        res[v] = {}
        for p, cells in procs.items():
            if set(cells) != (NS_BASE8K if v == "empty_base8k" else NS_REQUIRED): gates.append("variant %s process %s lacks the registered N set" % (v, p))
            pt, lo, hi = slope_us(cells, rng)
            lo_cells = {N: c for N, c in cells.items() if N <= 160}; hi_cells = {N: c for N, c in cells.items() if N >= 160}
            lin = None
            if len(lo_cells) >= 3 and len(hi_cells) >= 3:
                a = slope_us(lo_cells, rng)[0]; b = slope_us(hi_cells, rng)[0]; lin = abs(a - b) / max(abs(a), abs(b), 1e-9)
                if lin > LINEARITY_TOL: gates.append("non-linear: %s process %s slope N<=160 %.2f vs N>=160 %.2f us" % (v, p, a, b))
            res[v][p] = (pt, lo, hi, lin)
    need = ["empty", P_REGISTERED, "control20", "empty_base8k"]
    for v in need:
        if v not in res: gates.append("missing variant %s" % v)
    if len({len(res[v]) for v in need if v in res}) > 1 or any(len(res[v]) < 3 for v in need if v in res):
        gates.append("fewer than 3 processes for a registered variant")
    if "control20" in res and "empty" in res:
        for p in res["control20"]:
            if p in res["empty"]:
                rec = res["control20"][p][0] - res["empty"][p][0]
                if abs(rec - CONTROL_US) > CONTROL_TOL_US: gates.append("positive control not recovered (process %s: +%.2f us, expected +%.1f +- %.1f)" % (p, rec, CONTROL_US, CONTROL_TOL_US))
    if "empty_base8k" in res and "empty" in res:
        for p in res["empty_base8k"]:
            if p in res["empty"]:
                a, b = res["empty_base8k"][p][0], res["empty"][p][0]
                if abs(a - b) / max(abs(a), abs(b), 1e-9) > BASE_TOL: gates.append("slope on an 8,000-node base (%.2f us) differs from the bare chain (%.2f us), process %s" % (a, b, p))
    out = {"gates": sorted(set(gates)), "slopes": {v: {p: res[v][p] for p in res[v]} for v in res}}
    if out["gates"]:
        out["verdict"] = "INVALID"; return out
    ge_lo = min(res["empty"][p][1] for p in res["empty"]); ge_hi = max(res["empty"][p][2] for p in res["empty"])
    ga_lo = min(res[P_REGISTERED][p][1] for p in res[P_REGISTERED]); ga_hi = max(res[P_REGISTERED][p][2] for p in res[P_REGISTERED])
    ge_pt = S.median(res["empty"][p][0] for p in res["empty"]); ga_pt = S.median(res[P_REGISTERED][p][0] for p in res[P_REGISTERED])
    G = lambda ge, ga, exposed_c: (E_EMPTY * ge + (E_B + E_D + (E_C if exposed_c else 0.0)) * ga) / 1000.0     # ms per step
    out.update({"g_empty_us": (ge_pt, ge_lo, ge_hi), "g_active_us": (ga_pt, ga_lo, ga_hi),
                "G_star_ms": G_STAR_MS,
                "G_ms_C_hidden": (G(ge_pt, ga_pt, False), G(ge_lo, ga_lo, False), G(ge_hi, ga_hi, False)),
                "G_ms_C_exposed": (G(ge_pt, ga_pt, True), G(ge_lo, ga_lo, True), G(ge_hi, ga_hi, True))})
    Gx_hi = G(ge_hi, ga_hi, True); Gh_lo = G(ge_lo, ga_lo, False); Gx_lo = G(ge_lo, ga_lo, True); Gh_hi = G(ge_hi, ga_hi, False)
    if Gx_hi <= G_STAR_MS: out["verdict"] = "CLEARS"
    elif Gh_lo >= G_STAR_MS: out["verdict"] = "REJECTION-STANDS"
    elif Gh_hi < G_STAR_MS < Gx_lo: out["verdict"] = "EXPOSURE-DEPENDENT"
    else: out["verdict"] = "UNRESOLVED"
    # sensitivity to the p-variants (the verdict is registered at p4; report whether it would change)
    sens = {}
    for v in ("active_p1", "active_p6"):
        if v in res:
            g = S.median(res[v][p][0] for p in res[v]); sens[v] = (g, G(ge_pt, g, True), G(ge_pt, g, False))
    out["p_sensitivity_us_and_G_X_H"] = sens
    return out

def selftest():
    rng = random.Random(1)
    def synth(ge, ga, ctrl_extra=CONTROL_US, gen=3, other=0, clock=(2400, 2400), nl=0.0, base_factor=1.0):
        lines = []
        for p in range(3):
            for v, g in (("empty", ge), ("active_p1", ga * 0.6), ("active_p4", ga), ("active_p6", ga * 1.3), ("control20", ge + ctrl_extra), ("empty_base8k", ge * base_factor)):
                for N in sorted(NS_BASE8K if v == "empty_base8k" else NS_REQUIRED):
                    per = (30.0 + (8000 * 0.0 if v != "empty_base8k" else 0.0) + g * N + nl * N * N / 640.0) / 1000.0        # ms per replay
                    R = 200; batches = [(per * R) * (1 + rng.gauss(0, 0.004)) for _ in range(30)]
                    lines.append({"process": p, "variant": v, "N": N, "R": R, "batch_ms": batches, "link_gen_start": gen, "link_gen_end": gen,
                                  "sm_mhz_min": clock[0], "sm_mhz_max": clock[1], "other_gpu_procs": other, "nodes": 3 * N + (8000 if v == "empty_base8k" else 0)})
        return lines
    cases = [("clears", synth(4.0, 6.0), "CLEARS"), ("rejects", synth(8.0, 14.0), "REJECTION-STANDS"),
             ("exposure-dependent", synth(6.0, 9.5), "EXPOSURE-DEPENDENT"), ("unresolved (G_H on G*)", synth(6.5, 10.47), "UNRESOLVED"),
             ("bad control", synth(6.0, 9.0, ctrl_extra=5.0), "INVALID"), ("idle link", synth(6.0, 9.0, gen=1), "INVALID"),
             ("other gpu user", synth(6.0, 9.0, other=1), "INVALID"), ("non-linear", synth(6.0, 9.0, nl=40.0), "INVALID"), ("base8k slope differs", synth(6.0, 9.0, base_factor=1.4), "INVALID")]
    ok = True
    for name, lines, want in cases:
        r = analyse(lines); got = r["verdict"]
        extra = "" if got == "INVALID" else "  g_e %.2f g_a %.2f  G_H %.3f G_X %.3f (G* %.3f)" % (r["g_empty_us"][0], r["g_active_us"][0], r["G_ms_C_hidden"][0], r["G_ms_C_exposed"][0], G_STAR_MS)
        print("%-24s want %-19s got %-19s %s%s" % (name, want, got, "ok" if got == want else "MISMATCH", extra)); ok &= got == want
    print("SELFTEST", "PASSED" if ok else "FAILED"); return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest": sys.exit(selftest())
    lines = [json.loads(l) for l in open(sys.argv[1])]
    r = analyse(lines)
    print("VERDICT:", r["verdict"])
    for g in r["gates"]: print("  gate:", g)
    for k in ("g_empty_us", "g_active_us", "G_star_ms", "G_ms_C_hidden", "G_ms_C_exposed", "p_sensitivity_us_and_G_X_H"):
        if k in r: print(" ", k, r[k])
    for v, procs in r["slopes"].items():
        for p, x in procs.items(): print("  slope %-14s process %s: %.2f us/triple [%.2f, %.2f], linearity diff %s" % (v, p, x[0], x[1], x[2], "n/a" if x[3] is None else "%.3f" % x[3]))
