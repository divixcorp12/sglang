#!/usr/bin/env python3
"""Copy-engine counters of copy-engine smoke arms, from their stage traces (stages.jsonl, graph_step lines).

Each graph_step line carries the RAM-miss service's cumulative counters (``thread``). Per arm this reports, over
the graph decode steps: copy jobs, lanes and bytes per step, the mean and max submit-to-observed-completion latency
per job (the grant to the copy thread's successful cuEventQuery), the API time per job, fallbacks and errors, and
the lane leases released by each path. CPU only.

    ce_stats.py NAME=DIR [NAME=DIR ...] [--json OUT]
"""

import argparse
import json
import os


def arm_stats(directory):
    steps, last, first = 0, None, None
    for line in open(os.path.join(directory, "stages.jsonl")):
        record = json.loads(line)
        if record.get("kind") != "graph_step" or not record.get("thread"):
            continue
        steps += int(record.get("steps", 1))
        first = first or record["thread"]
        last = record["thread"]
    if last is None:
        return {"graph_steps": 0}
    c = {k: last.get(k, 0) for k in last}
    jobs = c.get("copy_jobs", 0)
    out = {
        "graph_steps": steps,
        "copy_jobs": jobs,
        "copy_lanes": c.get("copy_lanes", 0),
        "copy_jobs_per_step": round(jobs / steps, 2) if steps else None,
        "copy_lanes_per_step": round(c.get("copy_lanes", 0) / steps, 2) if steps else None,
        "copy_mb_per_step": round(c.get("copy_bytes", 0) / steps / 1e6, 2) if steps else None,
        "copy_latency_us_per_job": round(c.get("copy_latency_ns", 0) / jobs / 1e3, 1) if jobs else None,
        "copy_latency_ms_per_step": round(c.get("copy_latency_ns", 0) / steps / 1e6, 2) if steps else None,
        "copy_latency_max_ms": round(c.get("copy_latency_max_ns", 0) / 1e6, 2),
        "copy_issue_us_per_job": round(c.get("copy_issue_ns", 0) / jobs / 1e3, 1) if jobs else None,
        "copy_gbps": round(c.get("copy_bytes", 0) / c["copy_latency_ns"], 2) if c.get("copy_latency_ns") else None,
        "copy_fallbacks": c.get("copy_fallbacks", 0),
        "copy_errors": c.get("copy_errors", 0),
        "copy_generation_mismatches": c.get("copy_generation_mismatches", 0),
        "leases_acked": c.get("leases_acked", 0),
        "leases_copied": c.get("leases_copied", 0),
        "leases_voided": c.get("leases_voided", 0),
        "hit_leases_granted": c.get("hit_leases_granted", 0),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    result = {}
    for spec in a.arms:
        name, directory = spec.split("=", 1)
        result[name] = arm_stats(directory)
        print(name, json.dumps(result[name]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
