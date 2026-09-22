#!/usr/bin/env python3
"""Per-row transfer precheck (Task 6). Frozen by PER_ROW_PRECHECK_PREREG.txt; read that first.

Read-only over the stage traces of Task 1 arms. CPU only, one thread, streams each file line by line.
Usage: per_row_precheck.py <task1-results dir> [--selftest]
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys

C_PRIMARY = 1.055
C_SENS = (0.55, 1.055, 1.6)
LAYERS = 40
LANE_CAP = 6
PERMS = 64
LAUNCH_MS = 1.0
REJECT_FRAC = 0.015
SUPPORT_FRAC = 0.03
IRRELEVANT_RATIO = 0.25

ON_TRACED = ["task1-2-new-on-T", "task1b-0-new-on-T", "task1c-0-new-on-T"]
OFF_TRACED = ["t1-0-new-off-T", "task1-0-new-off-T", "task1-1-new-off-T", "task1c-2-new-off-T"]
ON_UNTRACED = ["task1-3-new-on-U", "task1-6-new-on-U", "task1c-3-new-on-U"]
OFF_UNTRACED = ["task1-4-new-off-U", "task1-5-new-off-U"]


def chain(order, ready, c, t0):
    e = t0
    for lane in order:
        r = ready[lane]
        e = (r if r > e else e) + c
    return e


def request_savings(t0, res, done, miss_ready, k, c, seed):
    """(best, random, random_h0, miss_only_best) savings in ms of one request. Times in ms."""
    m = len(miss_ready)
    h = k - m
    ready = [res] * h + list(miss_ready)  # hit lanes first (indices 0..h-1), then miss rows in R order
    order_best = sorted(range(k), key=lambda i: (ready[i], i))
    t_b = done + k * c
    best = t_b - chain(order_best, ready, c, t0)
    rng = random.Random(seed)
    lanes = list(range(k))
    acc = 0.0
    for _ in range(PERMS):
        rng.shuffle(lanes)
        acc += chain(lanes, ready, c, t0)
    rand = t_b - acc / PERMS
    # miss rows only (h = 0), random order
    mready = list(miss_ready)
    t_b0 = done + m * c
    rng = random.Random(seed ^ 0x5DEECE66D)
    lanes0 = list(range(m))
    acc0 = 0.0
    for _ in range(PERMS):
        rng.shuffle(lanes0)
        acc0 += chain(lanes0, mready, c, t0)
    rand0 = t_b0 - acc0 / PERMS
    return best, rand, rand0


def seed_of(name, seq):
    return int(hashlib.sha1(f"{name}:{seq}".encode()).hexdigest()[:12], 16)


def analyse_trace(path, name, cs):
    steps = {}
    reqs = []
    stats = dict(lines=0, demand=0, excluded_unserved=0, excluded_untraced=0, excluded_nostep=0, excluded_mismatch=0)
    schemas = set()
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            kind = d.get("kind")
            if kind == "graph_step":
                steps[d["forward"]] = d
            elif kind == "ram_miss_request":
                stats["lines"] += 1
                schemas.add(d.get("schema"))
                if d["request"]["type"] != "demand" or d["rows_asked"] < 1:
                    continue
                stats["demand"] += 1
                reqs.append(d)
    out = {c: dict(best=0.0, rand=0.0, rand0=0.0, gen=0.0) for c in cs}
    used_steps = set()
    n_used = 0
    spreads = []
    hide_ok = 0
    m_hist = {}
    for d in reqs:
        if d["status"] != "served":
            stats["excluded_unserved"] += 1
            continue
        if d["untraced"]["rows"] != 0:
            stats["excluded_untraced"] += 1
            continue
        step = steps.get(d["forward"])
        if step is None:
            stats["excluded_nostep"] += 1
            continue
        rows = d["row_pack_ns"]
        m = d["rows_asked"]
        if len(rows) != m:
            stats["excluded_mismatch"] += 1
            continue
        st = d["stages_ns"]
        t0 = st["observed"] / 1e6
        res = st["reserved"] / 1e6
        done = st["done"] / 1e6
        miss_ready = sorted(r["end"] / 1e6 for r in rows)
        k = min(LANE_CAP, max(m, int(round(step["vram_miss"] / LAYERS))))
        used_steps.add(d["forward"])
        n_used += 1
        m_hist[m] = m_hist.get(m, 0) + 1
        if m >= 2:
            spreads.append(miss_ready[-1] - miss_ready[0])
        h = k - m
        if done - t0 >= h * C_PRIMARY:
            hide_ok += 1
        seed = seed_of(name, d["request"]["seq"])
        for c in cs:
            best, rand, rand0 = request_savings(t0, res, done, miss_ready, k, c, seed)
            gen_k = LANE_CAP
            gen_best, _, _ = request_savings(t0, res, done, miss_ready, max(gen_k, m), c, seed)
            o = out[c]
            o["best"] += best
            o["rand"] += rand
            o["rand0"] += rand0
            o["gen"] += gen_best
    n_steps = max(1, len(used_steps))
    per_step = {c: {key: val / n_steps for key, val in o.items()} for c, o in out.items()}
    spreads.sort()

    def pct(p):
        return spreads[min(len(spreads) - 1, int(p * len(spreads)))] if spreads else None

    return dict(
        schemas=sorted(s for s in schemas if s is not None),
        stats=stats,
        requests_used=n_used,
        steps_used=n_steps,
        m_hist=dict(sorted(m_hist.items())),
        per_step_ms=per_step,
        spread_n=len(spreads),
        spread_ms=dict(p10=pct(0.10), p50=pct(0.50), p90=pct(0.90)),
        spread_ge_c=(sum(1 for s in spreads if s >= C_PRIMARY) / len(spreads)) if spreads else None,
        hide_ok_frac=hide_ok / n_used if n_used else None,
    )


def mean_tok(dirpath, names):
    vals, used, missing = [], [], []
    for n in names:
        p = os.path.join(dirpath, n + ".json")
        if not os.path.exists(p):
            missing.append(n)
            continue
        vals.append(json.load(open(p))["mean_decode_tok_s"])
        used.append(n)
    return (sum(vals) / len(vals) if vals else None), used, missing


def classify(results, t_step):
    def frac(name, key):
        r = results[name]["per_step_ms"][C_PRIMARY]
        return r[key] / t_step

    arms = list(results)
    rejects = [frac(a, "gen") < REJECT_FRAC for a in arms]
    supports = [(results[a]["per_step_ms"][C_PRIMARY]["rand"] - LAUNCH_MS) / t_step >= SUPPORT_FRAC for a in arms]
    irrelevant = [
        results[a]["per_step_ms"][C_PRIMARY]["rand"] > 0
        and results[a]["per_step_ms"][C_PRIMARY]["rand0"] / results[a]["per_step_ms"][C_PRIMARY]["rand"] < IRRELEVANT_RATIO
        for a in arms
    ]
    cls = "REJECT" if all(rejects) else "SUPPORT" if all(supports) else "INCONCLUSIVE"
    return cls, ("SPREAD-IRRELEVANT" if all(irrelevant) else "not spread-irrelevant"), rejects, supports, irrelevant


def selftest():
    # 3 miss rows, packed 2.7 ms apart, read done with the last; batched copies 3 after done.
    best, rand, rand0 = request_savings(0.0, 0.0, 10.0, [4.0, 6.7, 9.4], 3, 1.0, 1)
    assert abs(best - (10.0 + 3 - (9.4 + 1.0))) < 1e-9, best  # only the last row's copy is left after 9.4
    # slowest lane first in fixed order: per-row is no better than batched (saving 0 when that lane is first)
    t_b = 10.0 + 3 * 1.0
    assert chain([2, 0, 1], [0.0, 0.0, 10.0], 1.0, 0.0) == t_b
    assert rand <= best + 1e-9
    # one miss row, no hits: nothing to gain
    b, r, r0 = request_savings(0.0, 0.0, 5.0, [5.0], 1, 1.0, 1)
    assert abs(b) < 1e-9 and abs(r0) < 1e-9
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    d = sys.argv[1]
    res = {}
    for group, names in (("ON", ON_TRACED), ("OFF", OFF_TRACED)):
        for n in names:
            p = os.path.join(d, n + ".trace")
            if not os.path.exists(p):
                print(f"MISSING trace {n}")
                continue
            res[(group, n)] = analyse_trace(p, n, C_SENS)
    t_on, used_on, miss_on = mean_tok(d, ON_UNTRACED)
    t_off, used_off, miss_off = mean_tok(d, OFF_UNTRACED)
    print(f"untraced denominators: ON {used_on} missing {miss_on} mean tok/s {t_on}; OFF {used_off} missing {miss_off} mean tok/s {t_off}")
    for (group, n), r in res.items():
        print(f"\n== {group} {n}: schemas {r['schemas']} stats {r['stats']}")
        print(f"   requests used {r['requests_used']} over {r['steps_used']} steps; m histogram {r['m_hist']}")
        print(f"   miss-row spread (m>=2, n={r['spread_n']}): {r['spread_ms']} ms; fraction >= c({C_PRIMARY}): {r['spread_ge_c']}")
        print(f"   requests whose read-wait covers all hit copies: {r['hide_ok_frac']}")
        for c in C_SENS:
            o = r["per_step_ms"][c]
            print(f"   c={c}: per-step ms  BEST {o['best']:.2f}  RANDOM {o['rand']:.2f}  RANDOM-miss-only {o['rand0']:.2f}  GENEROUS {o['gen']:.2f}")
    for group, t_tok in (("ON", t_on), ("OFF", t_off)):
        if t_tok is None:
            continue
        t_step = 1000.0 / t_tok
        sub = {n: r for (g, n), r in res.items() if g == group}
        if not sub:
            continue
        cls, irr, rej, sup, irl = classify(sub, t_step)
        print(f"\n### {group}: T_step {t_step:.1f} ms  ->  {cls}; {irr}")
        for a in sub:
            o = sub[a]["per_step_ms"][C_PRIMARY]
            print(
                f"   {a}: GENEROUS/T {o['gen'] / t_step:.4f}  (RANDOM-L)/T {(o['rand'] - LAUNCH_MS) / t_step:.4f}  "
                f"RANDOM-miss-only/RANDOM {o['rand0'] / o['rand'] if o['rand'] > 0 else float('nan'):.3f}  "
                f"BEST/T {o['best'] / t_step:.4f}"
            )


if __name__ == "__main__":
    main()
