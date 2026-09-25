#!/usr/bin/env python3
"""Per-layer kernel count, kernel time and span of the three fused ranges, from a node-mode decode trace.

  A   gather_destinations: after plan_unique_routes_kernel up to exl3_ram_miss_post_kernel
  BC  commit_gather + route tables: after the go_total add (finalize + 1) up to exl3_moe_kernel
Also the kernels per step and per layer (gather to gather). Flag off: A 26, BC 63; flag on: A 1, BC 2. A stage-traced
run adds GraphRouteLog.record's two index_copy kernels per layer to A in both arms.

    taskset -c 0-63 python3 fusion_ranges.py <node-mode trace .sqlite>
"""
import collections, json, sqlite3, statistics, sys

db = sys.argv[1]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
names = dict(c.execute("select id, value from StringIds"))
rows = c.execute(
    "select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0 order by start"
).fetchall()
steps = collections.defaultdict(list)
for s, e, corr, n in rows:
    steps[corr].append((s, e, names[n]))
steps = [v for v in steps.values() if len(v) > 1000]
skipped = 0
stats = collections.defaultdict(list)
per_step = []
for step in steps:
    idx = lambda name: [i for i, k in enumerate(step) if k[2] == name]
    posts = idx("exl3_ram_miss_post_kernel")
    plans = idx("plan_unique_routes_kernel")
    fins = idx("exl3_ram_miss_lease_finalize_kernel")
    moes = idx("exl3_moe_kernel")
    gathers = idx("exl3_moe_gather_kernel")
    if len(moes) != len(posts) or len(gathers) != len(posts):
        skipped += 1
        continue  # a step the capture cut short
    per_step.append(len(step))
    stats["layers"].append(len(posts))
    prev = -1
    for L, p in enumerate(posts):
        plan = max(i for i in plans if i < p)
        fin = min(i for i in fins if i > p)
        moe = min(i for i in moes if i > fin)
        g = min(i for i in gathers if i > moe)
        for tag, lo, hi in (("A", plan + 1, p), ("BC", fin + 2, moe)):
            ks = step[lo:hi]
            stats[tag + "_n"].append(len(ks))
            stats[tag + "_us"].append(sum(e - s for s, e, _ in ks) / 1e3)
            stats[tag + "_span_us"].append((step[hi][0] - step[lo - 1][1]) / 1e3)
        stats["layer_kernels"].append(g - prev)
        prev = g
out = {"steps": len(steps) - skipped, "skipped_steps": skipped, "kernels_per_step_p50": statistics.median(per_step)}
for k, v in stats.items():
    out[k] = {"p50": statistics.median(v), "mean": statistics.mean(v), "min": min(v), "max": max(v)}
print(json.dumps(out, indent=1))
