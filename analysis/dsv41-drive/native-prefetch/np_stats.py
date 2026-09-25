#!/usr/bin/env python3
"""Copy-engine and native-prefetch counters of native-prefetch smoke arms, from their stage traces and server logs.

Per arm, over the graph decode steps (graph_step lines carry the service's cumulative counters, ``thread``):

- the copy engine: jobs, lanes and MB per step, fallbacks and errors (the proof that the arm ran with it on);
- the prefetch: requests, issued, copied, skipped (unarmed / not ready / invalid), used and wasted as the target
  layer's next request judged them, precision = used / (used + wasted), requests and copies per step, jobs held
  behind demand, and the mean request-to-completion latency;
- from server.log: the copy engine's arming line and the last device-counter line of the prefetch kernels.

    np_stats.py NAME=DIR [NAME=DIR ...] [--json OUT]
"""

import argparse
import ast
import json
import os
import re


def arm_stats(directory):
    steps, last = 0, None
    for line in open(os.path.join(directory, "stages.jsonl")):
        if '"graph_step"' not in line:
            continue
        record = json.loads(line)
        if record.get("kind") != "graph_step" or not record.get("thread"):
            continue
        steps += int(record.get("steps", 1))
        last = record["thread"]
    if last is None:
        return {"graph_steps": 0}
    c = last
    per = lambda key: round(c.get(key, 0) / steps, 2) if steps else None  # noqa: E731
    used, wasted = c.get("prefetch_used", 0), c.get("prefetch_wasted", 0)
    copied = c.get("prefetch_copied", 0)
    out = {
        "graph_steps": steps,
        "copy_jobs_per_step": per("copy_jobs"),
        "copy_lanes_per_step": per("copy_lanes"),
        "copy_mb_per_step": round(c.get("copy_bytes", 0) / steps / 1e6, 2) if steps else None,
        "copy_fallbacks": c.get("copy_fallbacks", 0),
        "copy_errors": c.get("copy_errors", 0),
        "copy_generation_mismatches": c.get("copy_generation_mismatches", 0),
        "prefetch_requests": c.get("prefetch_requests", 0),
        "prefetch_issued": c.get("prefetch_issued", 0),
        "prefetch_copied": copied,
        "prefetch_skipped_unarmed": c.get("prefetch_skipped_unarmed", 0),
        "prefetch_skipped_not_ready": c.get("prefetch_skipped_not_ready", 0),
        "prefetch_skipped_invalid": c.get("prefetch_skipped_invalid", 0),
        "prefetch_used": used,
        "prefetch_wasted": wasted,
        "prefetch_precision": round(used / (used + wasted), 3) if used + wasted else None,
        "prefetch_requests_per_step": per("prefetch_requests"),
        "prefetch_copied_per_step": per("prefetch_copied"),
        "prefetch_used_per_step": per("prefetch_used"),
        "prefetch_held": c.get("prefetch_held", 0),
        "prefetch_latency_us": round(c.get("prefetch_latency_ns", 0) / copied / 1e3, 1) if copied else None,
    }
    log = os.path.join(directory, "server.log")
    if os.path.exists(log):
        text = open(log, errors="replace").read()
        armed = re.findall(r"copy engine armed after (\d+) decode forwards", text)
        out["copy_engine_armed_after"] = int(armed[-1]) if armed else None
        device = re.findall(r"native prefetch device counters (\{.*?\})", text)
        out["prefetch_device_counters"] = ast.literal_eval(device[-1]) if device else None
        out["fatal_in_log"] = "fail-stop" in text
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
        print(name, json.dumps(result[name], indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
