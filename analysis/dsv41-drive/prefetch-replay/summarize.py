"""Markdown table of run_arms.py output, each arm against the ``none`` arm with the same timing constants.

Usage: summarize.py OUT_JSONL [...]
"""

import json
import sys

TIMING = ("nvme_row_ms", "nvme_lat_ms", "pieces", "link_row_ms", "compute_ms", "step_ms")


def key(r):
    return tuple(r["args"][k] for k in TIMING)


rows = [json.loads(line) for path in sys.argv[1:] for line in open(path) if line.strip()]
base = {key(r): r for r in rows if r["args"]["predictor"] == "none"}
print("| arm | prec (target) | prec (any use) | spec rows/tok | RAM misses/tok | saved/tok | harmful evict/tok "
      "| demand delayed/tok | mean delay ms | exposed ms/tok | saved ms/tok | late/tok | NVMe busy |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda r: r["name"]):
    b = base.get(key(r))
    if b is None:
        continue
    print(
        f"| {r['name']} | {r['precision_target']:.2f} | {r['precision_any_use']:.2f} | {r['spec_rows_per_token']:.2f} "
        f"| {r['ram_misses_per_token']:.2f} | {b['ram_misses_per_token'] - r['ram_misses_per_token']:+.2f} "
        f"| {r['harmful_evictions_per_token']:.2f} | {r['demand_rows_delayed_per_token']:.2f} "
        f"| {r['demand_delay_mean_ms']:.2f} | {r['exposed_ms_per_token']:.2f} "
        f"| {b['exposed_ms_per_token'] - r['exposed_ms_per_token']:+.2f} | {r['late_prefetch_per_token']:.2f} "
        f"| {r['nvme_busy_frac']:.2f} |"
    )
