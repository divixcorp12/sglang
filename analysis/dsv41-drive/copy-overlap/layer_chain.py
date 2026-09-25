#!/usr/bin/env python3
"""Per-layer MoE chain and elementwise catalog from a node-mode decode trace (one graph per step, one stream).

Each layer is cut at its RAM-miss post kernel. Groups, in stream order:
  attn     -- from the previous layer's exl3_moe_gather to the shared expert (attention, mHC, norms)
  shared   -- the shared expert: the three exl3_gemv before the router GEMM and the elementwise around them
  route    -- router GEMM, top-k, route planning, up to the post kernel
  chain    -- post, W1, C1, A1, S, A2, F (the RAM-miss chain)
  book     -- after F up to exl3_moe_kernel (DIRECT residency and gather bookkeeping)
  moe      -- exl3_moe_kernel and exl3_moe_gather
Prints per-token ms per group, per-kernel-name ms and counts per group, and the tiny-kernel totals.

    python3 layer_chain.py <node trace .sqlite> [--json out.json]
"""
import argparse
import collections
import json
import sqlite3
import statistics

TINY_US = 5.0  # a kernel at or under this is counted as "tiny" (elementwise / scatter / index class)


def steps_of(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    rows = c.execute(
        "select start, end, correlationId, shortName, gridX from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0 order by start"
    ).fetchall()
    steps = collections.defaultdict(list)
    for s, e, corr, name, grid in rows:
        steps[corr].append((s, e, names[name], grid))
    return [v for v in steps.values()]


def cut_layers(step):
    posts = [i for i, k in enumerate(step) if k[2] == "exl3_ram_miss_post_kernel"]
    gathers = [i for i, k in enumerate(step) if k[2] == "exl3_moe_gather_kernel"]
    layers = []
    for p in posts:
        router = max(i for i in range(p) if step[i][2] == "tiny_n_gemm_kernel")
        gemvs = [i for i in range(router) if step[i][2] == "exl3_gemv_int8_sq_kernel"]
        shared0 = gemvs[-3] - 1
        prev_gather = max([g for g in gathers if g < shared0], default=-1)
        fin = min(i for i in range(p, len(step)) if step[i][2] == "exl3_ram_miss_lease_finalize_kernel")
        moe = min(i for i in range(fin, len(step)) if step[i][2] == "exl3_moe_kernel")
        layers.append(
            {
                "attn": (prev_gather + 1, shared0),
                "shared": (shared0, router),
                "route": (router, p),
                "chain": (p, fin + 1),
                "book": (fin + 1, moe),
                "moe": (moe, moe + 2),
            }
        )
    return layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json")
    a = ap.parse_args()
    steps = [s for s in steps_of(a.db) if len(s) > 1000]
    group_ms = collections.defaultdict(list)
    names = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    chain_names = collections.defaultdict(float)
    span = []
    tiny = []
    for step in steps:
        layers = cut_layers(step)
        per = collections.defaultdict(float)
        for layer in layers:
            for g, (lo, hi) in layer.items():
                for s, e, n, _ in step[lo:hi]:
                    per[g] += (e - s) / 1e6
                    names[g][n][0] += 1
                    names[g][n][1] += (e - s) / 1e6
                    if g == "chain":
                        chain_names[n] += (e - s) / 1e6
        for g, v in per.items():
            group_ms[g].append(v)
        span.append((step[-1][1] - step[0][0]) / 1e6)
        tiny.append(sum((e - s) / 1e6 for s, e, n, _ in step if (e - s) / 1e3 <= TINY_US))
    n = len(steps)
    out = {
        "steps": n,
        "kernels_per_step": statistics.median(len(s) for s in steps),
        "step_span_ms_p50": statistics.median(span),
        "tiny_kernel_ms_per_step_p50": statistics.median(tiny),
        "group_ms_per_step_mean": {g: statistics.mean(v) for g, v in group_ms.items()},
        "group_ms_per_step_p50": {g: statistics.median(v) for g, v in group_ms.items()},
        "chain_kernel_ms_per_step": {k: v / n for k, v in chain_names.items()},
        "by_group": {
            g: sorted(
                ({"kernel": k, "per_step": c / n, "ms_per_step": t / n} for k, (c, t) in d.items()),
                key=lambda r: -r["ms_per_step"],
            )
            for g, d in names.items()
        },
    }
    print(json.dumps({k: v for k, v in out.items() if k != "by_group"}, indent=1))
    for g, rows in out["by_group"].items():
        print(f"\n== {g}: {sum(r['per_step'] for r in rows):.0f} kernels/step, {sum(r['ms_per_step'] for r in rows):.2f} ms/step")
        for r in rows[:25]:
            print(f"  {r['per_step']:7.1f}  {r['ms_per_step']:8.3f} ms  {r['kernel'][:70]}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
