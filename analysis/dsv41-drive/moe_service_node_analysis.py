#!/usr/bin/env python3
"""Join selected MoE stage rows to timed Nsight graph nodes by replay order.

Only accept a capture with exactly one post, wait, and pinned-row copy per
selected layer, stable graph-node IDs, and a validated host/GPU clock offset.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path


NAMES = {
    "exl3_ram_miss_post_kernel": "post",
    "exl3_ram_miss_wait_kernel": "wait",
    "copy_expert_row_segments_gpu_kernel": "copy",
}


def analyze(rows_path: Path, sqlite_path: Path, layers: int) -> dict:
    with rows_path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or len(rows) % layers:
        raise ValueError("selected stage rows do not form complete replays")
    with sqlite3.connect(sqlite_path) as db:
        base = db.execute("SELECT startTime FROM ANALYSIS_DETAILS").fetchone()[0]
        kernels = db.execute("""
            SELECT k.correlationId, k.graphNodeId, k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id = k.shortName
            WHERE k.graphNodeId IS NOT NULL AND s.value IN (?, ?, ?)
            ORDER BY k.start
        """, tuple(NAMES)).fetchall()
    groups: dict[int, dict[str, list[tuple]]] = {}
    for correlation, node, start, end, name in kernels:
        groups.setdefault(correlation, collections.defaultdict(list))[NAMES[name]].append((node, start, end))
    ordered = sorted(groups.values(), key=lambda g: min(item[1] for v in g.values() for item in v))
    if len(ordered) != len(rows) // layers:
        raise ValueError(f"Nsight replay groups {len(ordered)} != stage replays {len(rows) // layers}")
    node_ids: dict[str, list[int]] = {}
    joined = []
    for replay, group in enumerate(ordered):
        for name in NAMES.values():
            if len(group[name]) != layers:
                raise ValueError(f"replay {replay}: {name} nodes {len(group[name])} != {layers}")
            group[name].sort(key=lambda item: item[1])
            ids = [item[0] for item in group[name]]
            if name in node_ids and ids != node_ids[name]:
                raise ValueError(f"replay {replay}: {name} graph-node IDs changed")
            node_ids.setdefault(name, ids)
        for layer in range(layers):
            row = rows[replay * layers + layer]
            if row["layer"] != layer:
                raise ValueError(f"replay {replay}: expected layer {layer}, got {row['layer']}")
            post, wait, copy = (group[name][layer] for name in ("post", "wait", "copy"))
            if not (post[1] <= post[2] <= wait[1] <= wait[2] <= copy[1] <= copy[2]):
                raise ValueError(f"replay {replay} layer {layer}: invalid graph-node order")
            stage = row["stages_ns"]
            joined.append({
                "seq": row["seq"], "layer": layer, "type": row["type"],
                "post_us": (post[2] - post[1]) / 1e3,
                "wait_us": (wait[2] - wait[1]) / 1e3,
                "copy_us": (copy[2] - copy[1]) / 1e3,
                "cpu_service_us": (stage["done"] - stage["observed"]) / 1e3,
                "observed_minus_post_start_us": (stage["observed"] - (base + post[1])) / 1e3,
                "observed_minus_post_end_us": (stage["observed"] - (base + post[2])) / 1e3,
                "wait_end_minus_done_us": ((base + wait[2]) - stage["done"]) / 1e3,
                "observed_inside_post_wait": base + post[1] <= stage["observed"] <= base + wait[2],
                "done_before_wait_end": stage["done"] <= base + wait[2],
            })
    demand = [row for row in joined if row["type"] == "demand"]
    clock_checks = {
        "demand_observed_inside_post_wait": sum(row["observed_inside_post_wait"] for row in demand),
        "demand_done_before_wait_end": sum(row["done_before_wait_end"] for row in demand),
        "demand_count": len(demand),
    }
    clock_checks["all_demand_joins_plausible"] = (
        clock_checks["demand_observed_inside_post_wait"] == len(demand)
        and clock_checks["demand_done_before_wait_end"] == len(demand)
    )
    # These broad interval checks establish only millisecond-level plausibility.
    # An NVTX or device timer anchor is needed to interpret small doorbell gaps.
    exploratory_means = (
        {key: sum(row[key] for row in demand) / len(demand) for key in (
            "observed_minus_post_start_us", "observed_minus_post_end_us", "wait_end_minus_done_us")}
        if demand and clock_checks["all_demand_joins_plausible"] else None
    )
    if not clock_checks["all_demand_joins_plausible"]:
        for row in joined:
            for key in ("observed_minus_post_start_us", "observed_minus_post_end_us", "wait_end_minus_done_us"):
                row[key] = None
    return {
        "selected_rows": len(rows), "replays": len(ordered),
        "nsight_start_monotonic_ns_candidate": base,
        "clock_checks": clock_checks,
        "clock_resolution_limit": "Post-to-wait containment only; small cross-clock gaps require NVTX/device-timer calibration",
        "node_totals_ms": {key: sum(row[f"{key}_us"] for row in joined) / 1e3 for key in NAMES.values()},
        "demand_node_totals_ms": {key: sum(row[f"{key}_us"] for row in demand) / 1e3 for key in NAMES.values()},
        "demand_cpu_service_ms": sum(row["cpu_service_us"] for row in demand) / 1e3,
        "demand_wait_minus_cpu_service_ms": sum(row["wait_us"] - row["cpu_service_us"] for row in demand) / 1e3,
        "exploratory_cross_clock_means_us": exploratory_means,
        "joined": joined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected_rows", type=Path)
    parser.add_argument("nsys_sqlite", type=Path)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--rows-out", type=Path)
    args = parser.parse_args()
    report = analyze(args.selected_rows, args.nsys_sqlite, args.layers)
    if args.rows_out:
        with args.rows_out.open("w") as output:
            for row in report.pop("joined"):
                output.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
