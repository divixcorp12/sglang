#!/usr/bin/env python3
"""The fixed per-launch component of T(n), with uncertainty, from a REGISTERED c run. CPU only, stdlib only.

    python3 c_intercept_report.py <outdir>/results.jsonl

Task 6's V1 splits a reading layer's launch into count = h and count = k - h, and V2 launches one row per request, so both pay a fixed
cost per extra launch that PER_ROW_TRANSFER.md 1.2's `35.74 = 33.9 x c` does not model. c_analysis.py's fit gives an intercept `f` and a
slope `c_m` but no interval on either; this reports them, in those terms.

THIS SCRIPT CAN ONLY WITHHOLD. It runs the frozen gates first (c_analysis.analyse with the trace model stubbed: the gates need no
traces) and prints the verdict word on line 1; if a results.INVALID marker that voids the run sits beside the input (a marker scoped to the nvme arm's rho does not; it is printed and
ignored), or the verdict is INVALID, it prints that and nothing else and exits 3, exactly as c_analysis.py does. It reads the fitted values only after that.

Per node, for the registered arm sm/cold/idle/eager and, where present, graph launch:
  * fit A: T(n) = f + c_m n over n = 1..6 (what c_analysis.py fits);
  * fit B: the same over the BATCHED counts n = 2..6 only, extrapolated to n = 1, so an excess at count 1 is a residual and not smeared
    into the line; `delta1 = T(1) - (f_B + c_m_B)` is that residual;
  * the intervals: a bootstrap over launches (1,000 resamples, per-n resampling, p50 statistic) and the spread across the passes'
    own p50s (the passes are the independent replications; a bootstrap over launches within a pass understates between-pass variation).
"""
import collections, json, os, random, statistics as S, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "c_measurement"))
import c_analysis as A

NS = A.NS; ROW = A.ROW_BYTES


def ols(pts):
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]; n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def fits(t):                                     # t: {n: ms}
    fa, ca = ols([(n, t[n]) for n in NS]); fb, cb = ols([(n, t[n]) for n in NS if n >= 2])
    return {"f_A_us": fa * 1e3, "c_A_ms": ca, "f_B_us": fb * 1e3, "c_B_ms": cb, "delta1_us": (t[1] - (fb + cb)) * 1e3}


def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


def main(path):
    marker = os.path.join(os.path.dirname(os.path.abspath(path)), "results.INVALID")
    scope = None
    if os.path.exists(marker):
        text = open(marker).read().strip()
        if not A.marker_is_scoped(text):
            print("VERDICT: REFUSED (a results.INVALID marker that voids the run sits beside the input)"); print("  marker:", text); return 3
        scope = text                       # scoped to the nvme arm's rho (amendment 11): T(n) may be quoted, rho may not; this report never touches the nvme arm
    recs = [json.loads(l) for l in open(path)]
    A.simulate = lambda T, *_: (0.0, 0.0, 0.0)          # the gates and the verdict word do not need the divix01 traces; the model is not used here
    r = A.analyse(recs)
    print("VERDICT (frozen gates, trace model not evaluated):", r["verdict"])
    if scope: print("  SCOPED MARKER: %s | the nvme arm's rho is NOT to be quoted; nothing about the nvme arm is printed here" % scope)
    for g in r["gates"]: print("  gate:", g)
    if r["verdict"] == "INVALID": print("  (INVALID: no number is printed)"); return 3
    rng = random.Random(A.SEED)
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))     # (node, launch) -> pass -> n -> samples
    for d in recs:
        if d["engine"] == "sm" and d["state"] == "cold" and d["load"] == "idle": by[(d["node"], d["launch"])][d["process"]][d["n"]].extend(d["T_ms"])
    for key in sorted(by):
        passes = by[key]
        if any(set(v) != set(NS) for v in passes.values()): continue
        pooled = {n: [x for p in passes.values() for x in p[n]] for n in NS}
        pt = fits({n: S.median(pooled[n]) for n in NS})
        boot = collections.defaultdict(list)
        for _ in range(1000):
            f = fits({n: S.median(rng.choices(pooled[n], k=len(pooled[n]))) for n in NS})
            for k, v in f.items(): boot[k].append(v)
        pp = [fits({n: S.median(p[n]) for n in NS}) for _, p in sorted(passes.items())]
        print("\n== node %s, %s launch, %d passes, %d launches per n" % (key[0], key[1], len(passes), len(pooled[1])))
        print("   n   T p50 ms   T min ms   GB/s@p50  (link spec 15.75, copy engine 13.79)")
        for n in NS: print("  %2d  %9.4f  %9.4f  %8.2f" % (n, S.median(pooled[n]), min(pooled[n]), n * ROW / (S.median(pooled[n]) * 1e6)))
        for k, label in (("f_A_us", "intercept f, fit n=1..6 (us)"), ("c_A_ms", "slope c_m, fit n=1..6 (ms/row)"), ("f_B_us", "intercept f, fit n=2..6 (us)"),
                         ("c_B_ms", "slope c_m, fit n=2..6 (ms/row)"), ("delta1_us", "delta1 = T(1) - line(n=2..6)(1) (us)")):
            b = sorted(boot[k]); per = [x[k] for x in pp]
            print("  %-42s %9.4f   bootstrap 95%% [%9.4f, %9.4f]   per-pass %s" % (label, pt[k], q(b, .025), q(b, .975), ", ".join("%.4f" % x for x in per)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
