#!/usr/bin/env python3
"""Frozen analysis for the SMT-sibling pilot (C_MEASUREMENT_PREREG.md section 18). CPU only, stdlib only.

Question: does a busy SMT sibling of the launching CPU shift the GPU-side time T of one production gather launch (cold rows, node 0)?
Input: JSONL, one line per visit: {"rep": int, "n": int, "arm": "A"|"B", "T_ms": [..], "sib_spinner_pct": float, "sib_foreign_pct": float,
       "launch_foreign_pct": float, "spinner_ticks": int, "window_s": float}.   A = sibling idle (spinner stopped), B = spinner running on the sibling.
Rules (registered before any run):
  visit valid  <=>  launch_foreign_pct < 10 and sib_foreign_pct < 10 and (B: sib_spinner_pct >= 90; A: spinner_ticks == 0)
  rep-n valid  <=>  all four visits (A B B A) valid.   >= 8 valid reps per n, else INVALID.
  shift(rep, n) = median(T of the two B visits) / median(T of the two A visits) - 1
  per n: mean shift and a two-sided 95% t-interval over valid reps (df = reps - 1).
  verdict:  INSENSITIVE  if both n have the whole interval inside +-0.5%
            SENSITIVE    if any n has the whole interval outside +-0.5% (below -0.5% or above +0.5%)
            INCONCLUSIVE otherwise (an interval that straddles a band edge, or is wider than the band)
Usage: sibling_pilot_analysis.py visits.jsonl | --selftest
"""
import json, os, sys, statistics as S, random
BAND = 0.005; MIN_VALID = 8; FOREIGN_MAX = 10.0; SPIN_MIN = 90.0
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        # df 20-39 added for the reps-40 re-run (amendment 7): the table stopped at 19, so reps 40 raised
        # KeyError(38) before any shift was computed. Same source as 1-19, which scipy reproduces exactly.
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042, 31: 2.040, 32: 2.037, 33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026,
        38: 2.024, 39: 2.023}

def visit_ok(v):
    if v["launch_foreign_pct"] >= FOREIGN_MAX or v["sib_foreign_pct"] >= FOREIGN_MAX: return False
    return v["sib_spinner_pct"] >= SPIN_MIN if v["arm"] == "B" else v["spinner_ticks"] == 0

def analyse(visits):
    by = {}
    for v in visits: by.setdefault((v["n"], v["rep"]), []).append(v)
    out = {"per_n": {}, "excluded": []}
    for n in sorted({k[0] for k in by}):
        shifts = []
        for (nn, rep), vs in sorted(by.items()):
            if nn != n: continue
            a = [x for x in vs if x["arm"] == "A"]; b = [x for x in vs if x["arm"] == "B"]
            if len(a) != 2 or len(b) != 2 or not all(visit_ok(x) for x in vs): out["excluded"].append((n, rep)); continue
            ta = S.median([t for x in a for t in x["T_ms"]]); tb = S.median([t for x in b for t in x["T_ms"]]); shifts.append(tb / ta - 1.0)
        if len(shifts) < MIN_VALID: out["per_n"][n] = {"valid_reps": len(shifts), "verdict": "INVALID"}; continue
        m = S.mean(shifts); half = T975[len(shifts) - 1] * S.stdev(shifts) / len(shifts) ** 0.5
        out["per_n"][n] = {"valid_reps": len(shifts), "mean_shift": m, "ci": (m - half, m + half)}
    if any(d.get("verdict") == "INVALID" for d in out["per_n"].values()) or not out["per_n"]: out["verdict"] = "INVALID"; return out
    cis = [d["ci"] for d in out["per_n"].values()]
    if all(lo >= -BAND and hi <= BAND for lo, hi in cis): out["verdict"] = "INSENSITIVE"
    elif any(lo > BAND or hi < -BAND for lo, hi in cis): out["verdict"] = "SENSITIVE"
    else: out["verdict"] = "INCONCLUSIVE"
    return out

def selftest():
    rng = random.Random(5)
    def synth(shift, noise=0.002, bad_reps=0, spin=99.0, sib_foreign=1.0, reps=20):
        L = []
        for n, base in ((3, 3.4), (6, 6.6)):
            for rep in range(reps):
                for arm in "ABBA":
                    T = [base * (1 + (shift if arm == "B" else 0.0)) * (1 + rng.gauss(0, noise)) for _ in range(100)]
                    bad = rep < bad_reps
                    L.append({"rep": rep, "n": n, "arm": arm, "T_ms": T, "sib_spinner_pct": (spin if arm == "B" else 0.0), "spinner_ticks": (int(spin * 0.5) if arm == "B" else 0),
                              "sib_foreign_pct": 40.0 if (bad and arm == "B") else sib_foreign, "launch_foreign_pct": 1.0, "window_s": 0.5})
        return L
    cases = [("no effect", synth(0.0), "INSENSITIVE"), ("+2% effect", synth(0.02), "SENSITIVE"), ("-1.5% effect", synth(-0.015), "SENSITIVE"),
             ("noisy, no effect", synth(0.0, noise=0.2), "INCONCLUSIVE"), ("13 of 20 reps hit by a foreign burst on the sibling", synth(0.02, bad_reps=13), "INVALID"),
             ("5 of 20 reps hit (excluded, 15 left)", synth(0.02, bad_reps=5), "SENSITIVE"), ("spinner descheduled", synth(0.0, spin=50.0), "INVALID"),
             ("tight +0.45% (inside the band by the rule)", synth(0.0045, noise=0.0005), "INSENSITIVE"), ("tight +0.55% (outside the band)", synth(0.0055, noise=0.0005), "SENSITIVE"),
             ("interval straddling the edge", synth(0.005, noise=0.004), "INCONCLUSIVE")]
    ok = True
    for name, v, want in cases:
        r = analyse(v); got = r["verdict"]; print("%-46s want %-13s got %-13s %s %s" % (name, want, got, "ok" if got == want else "MISMATCH", {n: (round(d.get("mean_shift", 0) * 100, 3), d["valid_reps"]) for n, d in r["per_n"].items()})); ok &= got == want
    print("SELFTEST", "PASSED" if ok else "FAILED"); return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest": sys.exit(selftest())
    marker = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "INVALID")          # REFUSAL ONLY (added after run 2): a run marked INVALID is never analysed
    if os.path.exists(marker):
        print("VERDICT: INVALID"); print(open(marker).read().strip()); sys.exit(3)
    r = analyse([json.loads(l) for l in open(sys.argv[1])]); print("VERDICT:", r["verdict"])
    for n, d in r["per_n"].items(): print(" n=%s" % n, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()})
    print(" excluded (n, rep):", r["excluded"])
