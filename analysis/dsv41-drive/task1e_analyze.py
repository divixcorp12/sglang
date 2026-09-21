"""Analysis for the task1e series, fixed before the series ran (see task1e-PREDICTIONS.txt).

    task1e_analyze.py <task1-results dir> <label-prefix>... [--manifest clean-reference.json]

Reads every <prefix>-<i>-<code>-on-U.json in the results dir with its .verdict.txt, orders the arms by the
UTC of their first boundary sample, classifies each as clean or INCONCLUSIVE by the pre-registered rules,
and prints the comparison. Standard library only; run it under taskset -c 0-63.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

EXPECT_GEN = {"old": "gen0", "new": "gen2"}
LOAD_SPREAD, LOAD_CELL_GAP, LOAD_TREND = 2.0, 1.0, 2.0          # load1 units
FOREIGN_SPREAD, FOREIGN_CELL_GAP, FOREIGN_TREND = 100.0, 100.0, 100.0   # percentage points of one core, summed over foreign processes on cores 32-63
SD_REFERENCE = 0.0209
UNRESOLVABLE_SD = 3 * SD_REFERENCE
ARM_CORES = range(32, 64)


def _betacf(a, b, x):
    tiny, m_max = 1e-300, 400
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, m_max + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln_front) * _betacf(a, b, x) / a
    return 1 - math.exp(ln_front) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2, 0.5, x)
    return 1 - p if t > 0 else p


def t_crit(df, level=0.95):
    target, lo, hi = 1 - (1 - level) / 2, 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < target else (lo, mid)
    return (lo + hi) / 2


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")


def load_arm(json_path, code):
    verdict_path = json_path[:-5] + ".verdict.txt"
    rep = json.load(open(json_path))
    text = open(verdict_path).read() if os.path.exists(verdict_path) else ""
    lines = text.splitlines()
    ps = rep.get("per_session") or []
    samples = rep.get("boundary_samples") or []
    gen = next((l[len("NOTE GENERATION "):] for l in lines if l.startswith("NOTE GENERATION ")), "missing")
    contended = next((l.split()[2] for l in lines if l.startswith("NOTE CONTENDED ")), "missing")
    problems = [l for l in lines if l.startswith("PROBLEM")]
    valid = bool(lines) and lines[-1].strip() == "VALID"
    outliers = [l for l in lines if l.startswith("NOTE OUTLIER")]
    cross = [l for l in lines if re.match(r"NOTE CROSS-ARM session_", l)]
    loads = [b["loadavg"][0] for b in samples if b.get("loadavg")]
    foreign, sightings = [], []
    for b in samples:
        tot = 0.0
        for p in b.get("top_other_cpu") or []:
            if p.get("cpu_num") in ARM_CORES:
                tot += p["cpu_pct"]
                if p["cpu_pct"] >= 50:
                    sightings.append(f"{p['name']}({p['cpu_pct']:.0f}%)@{b['label']}")
        foreign.append(tot)
    return {
        "name": os.path.basename(json_path)[:-5], "code": code, "t0": (samples[0].get("utc") if samples else ""),
        "sessions": [r["decode_tok_s"] for r in ps], "ttft": [r["ttft_s"] for r in ps],
        "mean": mean([r["decode_tok_s"] for r in ps]) if ps else float("nan"),
        "valid": valid, "gen": gen, "gen_ok": gen.startswith(EXPECT_GEN[code]), "contended": contended,
        "outliers": outliers, "cross": cross, "problems": problems,
        "L": mean(loads) if loads else float("nan"), "F": mean(foreign) if foreign else float("nan"),
        "sightings": sightings,
    }


def inconclusive_reasons(a, ignore_cross=False):
    r = []
    if not a["valid"]:
        r.append("not VALID")
    if a["outliers"]:
        r.append("OUTLIER note")
    if a["cross"] and not ignore_cross:
        r.append("CROSS-ARM flag")
    if a["contended"] != "not":
        r.append(f"CONTENDED {a['contended']}")
    if not a["gen_ok"]:
        r.append(f"generation {a['gen'][:40]!r}")
    return r


def welch(new, old):
    mn, mo, sn, so = mean(new), mean(old), sd(new), sd(old)
    vn, vo = sn ** 2 / len(new), so ** 2 / len(old)
    se = math.sqrt(vn + vo)
    df = (vn + vo) ** 2 / (vn ** 2 / (len(new) - 1) + vo ** 2 / (len(old) - 1)) if se > 0 else float("nan")
    tc = t_crit(df)
    diff = mn - mo
    lr_se = math.sqrt(vn / mn ** 2 + vo / mo ** 2)
    lr = math.log(mn / mo)
    return {"mn": mn, "mo": mo, "sn": sn, "so": so, "diff": diff, "se": se, "df": df, "tcrit": tc,
            "diff_ci": (diff - tc * se, diff + tc * se), "ratio": mn / mo,
            "ratio_ci": (math.exp(lr - tc * lr_se), math.exp(lr + tc * lr_se))}


def ols(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx else float("nan"), n


def position_fit(arms_with_pos):
    """tok/s = a + c*[new] + s*position by least squares (normal equations, 3 parameters)."""
    rows = [(1.0, 1.0 if a["code"] == "new" else 0.0, float(p), a["mean"]) for a, p in arms_with_pos]
    if len(rows) < 4 or len({r[1] for r in rows}) < 2:
        return None
    n = 3
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * r[3] for r in rows) for i in range(n)]
    for i in range(n):  # gauss-jordan
        piv = max(range(i, n), key=lambda k: abs(ata[k][i]))
        if abs(ata[piv][i]) < 1e-12:
            return None
        ata[i], ata[piv], atb[i], atb[piv] = ata[piv], ata[i], atb[piv], atb[i]
        for k in range(n):
            if k != i:
                f = ata[k][i] / ata[i][i]
                ata[k] = [x - f * y for x, y in zip(ata[k], ata[i])]
                atb[k] -= f * atb[i]
    return {"cell_new_minus_old": atb[1] / ata[1][1], "slope_per_slot": atb[2] / ata[2][2]}


def stationarity(arms):
    """Every arm run, in order. Returns (non_stationary, [reasons])."""
    reasons = []
    for key, spread_t, gap_t, trend_t in (("L", LOAD_SPREAD, LOAD_CELL_GAP, LOAD_TREND), ("F", FOREIGN_SPREAD, FOREIGN_CELL_GAP, FOREIGN_TREND)):
        vals = [a[key] for a in arms]
        if any(math.isnan(v) for v in vals):
            reasons.append(f"{key}: missing samples")
            continue
        spread = max(vals) - min(vals)
        gap = mean([a[key] for a in arms if a["code"] == "new"]) - mean([a[key] for a in arms if a["code"] == "old"])
        slope, _ = ols(list(range(len(vals))), vals)
        trend = slope * (len(vals) - 1)
        name = {"L": "load1", "F": "foreign CPU% on cores 32-63"}[key]
        if spread > spread_t:
            reasons.append(f"{name}: spread across arms {spread:.2f} > {spread_t}")
        if abs(gap) > gap_t:
            reasons.append(f"{name}: new-minus-old cell mean {gap:+.2f}, |.| > {gap_t}")
        if abs(trend) > trend_t:
            reasons.append(f"{name}: fitted change over the series {trend:+.2f}, |.| > {trend_t}")
    cn = sum(1 for a in arms if a["code"] == "new" and a["sightings"])
    co = sum(1 for a in arms if a["code"] == "old" and a["sightings"])
    if abs(cn - co) >= 2:
        reasons.append(f">=50% foreign process on an arm core seen in {cn} new arms vs {co} old arms")
    return bool(reasons), reasons


def report_set(title, arms):
    new = [a["mean"] for a in arms if a["code"] == "new"]
    old = [a["mean"] for a in arms if a["code"] == "old"]
    print(f"\n== {title}: n_old={len(old)} n_new={len(new)}")
    if len(new) < 2 or len(old) < 2:
        print("   not computable (a cell has fewer than 2 arms)")
        return None
    w = welch(new, old)
    print(f"   old mean {w['mo']:.4f} sd {w['so']:.4f} | new mean {w['mn']:.4f} sd {w['sn']:.4f}")
    print(f"   diff {w['diff']:+.4f} tok/s  95% CI [{w['diff_ci'][0]:+.4f}, {w['diff_ci'][1]:+.4f}]  (Welch df {w['df']:.2f}, t {w['tcrit']:.3f})")
    print(f"   RATIO new/old {w['ratio']:.4f}  95% CI [{w['ratio_ci'][0]:.4f}, {w['ratio_ci'][1]:.4f}]")
    big = [n for n, s in (("old", w["so"]), ("new", w["sn"])) if s > UNRESOLVABLE_SD]
    if big:
        print(f"   NOTE: within-cell sd of {big} exceeds {UNRESOLVABLE_SD:.4f} (3x the gen1 quiet sd): this design cannot resolve 3.2%")
    lo, hi = w["ratio_ci"]
    print(f"   vs pre-registered rules: lower bound > 1.0: {lo > 1.0}; contains 1.032: {lo <= 1.032 <= hi}; contains 1.0: {lo <= 1.0 <= hi}")
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("prefixes", nargs="+")
    a = p.parse_args()
    arms = []
    for prefix in a.prefixes:
        for path in sorted(glob.glob(os.path.join(a.results, f"{prefix}-*-*-on-U.json"))):
            if path.endswith((".cache.json", ".regime.json")):
                continue
            code = os.path.basename(path).split("-")[2]
            arms.append(load_arm(path, code))
    arms.sort(key=lambda x: x["t0"])
    pos = {x["name"]: i + 1 for i, x in enumerate(arms)}
    print("ARMS IN TIME ORDER (position, name, mean tok/s, per-session, ttft, valid, gen, contended, outliers, cross, load1, foreignCPU)")
    for x in arms:
        print(f" {pos[x['name']]} {x['name']}: mean {x['mean']:.4f} sess {[round(s, 3) for s in x['sessions']]} ttft {[round(s, 1) for s in x['ttft']]} "
              f"valid={x['valid']} gen={x['gen'][:6]} contended={x['contended']} outl={len(x['outliers'])} cross={len(x['cross'])} L={x['L']:.2f} F={x['F']:.0f}")
        why = inconclusive_reasons(x)
        print(f"    -> {'CLEAN' if not why else 'INCONCLUSIVE: ' + '; '.join(why)}" + (f"  sightings {x['sightings']}" if x["sightings"] else ""))
        for n in x["outliers"] + x["cross"]:
            print(f"       {n[:200]}")
    non_stat, why = stationarity(arms)
    print(f"\nSTATIONARITY over all {len(arms)} arms run: {'NON-STATIONARY' if non_stat else 'stationary'}")
    for r in why:
        print("   ", r)
    primary = [x for x in arms if not inconclusive_reasons(x)]
    sens1 = [x for x in arms if not inconclusive_reasons(x, ignore_cross=True)]
    sens2 = [x for x in arms if x["valid"] and x["gen_ok"]]
    print(f"\nARMS DROPPED: primary {len(arms) - len(primary)}, S1 (CROSS-ARM ignored) {len(arms) - len(sens1)}, S2 (all VALID, right generation) {len(arms) - len(sens2)}")
    w = None
    for title, s in (("PRIMARY (inherited INCONCLUSIVE rules)", primary), ("S1 sensitivity: VALID, OUTLIER-free, not contended, CROSS-ARM ignored", sens1),
                     ("S2 descriptive: every VALID right-generation arm", sens2)):
        r = report_set(title, s)
        if title.startswith("PRIMARY"):
            w = r
        fit = position_fit([(x, pos[x["name"]]) for x in s])
        print("   position fit:", (f"cell(new-old) {fit['cell_new_minus_old']:+.4f} tok/s at fixed position, slope {fit['slope_per_slot']:+.4f} tok/s per slot" if fit else "not computable"))
    n_new = sum(1 for x in primary if x["code"] == "new")
    n_old = sum(1 for x in primary if x["code"] == "old")
    print("\nPRE-REGISTERED VERDICT:")
    if n_new < 3 or n_old < 3:
        print(f"   UNRESOLVED: {n_old} clean old, {n_new} clean new; fewer than 3 in a cell.")
    elif non_stat:
        print("   UNRESOLVED: contention was non-stationary across the series; no ratio is reported as a result.")
    else:
        print("   Ratio and interval above are the result (contended absolute levels).")


if __name__ == "__main__":
    main()
