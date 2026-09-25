#!/usr/bin/env python3
"""Report of one copy-engine soak (soak.sh's output directory). CPU only.

    soak_report.py OUT [--json FILE]

Reads OUT/requests.jsonl (soak_driver.py), OUT/stages.jsonl (the server's stage trace), OUT/server.log and
OUT/driver_summary.json, and reports:

  survival     whether the server lived to the end; every fail-stop line of the log (a request that timed out or
               failed) with its time, and the driver's request in flight at that moment.
  requests     per kind: count, HTTP 200s, errors (status and message), mean wall time.
  stalls       client inter-token gaps >= 0.5 s and >= 2 s (streaming requests, arm_metrics.py's thresholds), and the
               stage trace's multi-row demands (rows_asked >= 2) in decode forwards whose first->last completion
               span exceeds 10 ms (compare_arms.py's definition).
  counters     the RAM-miss service's cumulative counters on the last graph_step record: copy jobs, lanes,
               fallbacks, errors, generation mismatches, leases copied / acknowledged / voided.
  grammar      grammar-constrained requests whose output parses (JSON with the required keys) or fully matches.
  determinism  the control prompt's outputs, byte-identical to the first.
  trend        ms/token over the soak in 10-minute windows: median of the streaming requests' client ms/token
               (>= 16 chunks), and the median interval between consecutive graph_step records of one decode run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict

SLOW_SPAN_US = 10000.0
WINDOW_S = 600


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def stage_scan(path):
    """Two passes over a possibly multi-GB trace: graph_step forwards and counters, then decode demands."""
    decode_fwds, last_thread, steps = set(), None, 0
    step_ts = []  # (t, forward) of graph_step records, file order
    for line in open(path):
        if '"graph_step"' not in line:
            continue
        r = json.loads(line)
        if r.get("kind") != "graph_step":
            continue
        decode_fwds.add(r["forward"])
        steps += r.get("steps", 1)
        step_ts.append((r["t"], r["forward"]))
        if r.get("thread"):
            last_thread = r["thread"]
    multi = slow = demands = 0
    worst = []
    statuses = Counter()
    for line in open(path):
        if '"ram_miss_request"' not in line:
            continue
        r = json.loads(line)
        if r.get("kind") != "ram_miss_request" or r["forward"] not in decode_fwds:
            continue
        statuses[r["status"]] += 1
        if r["status"] != "served" or not r.get("rows_asked"):
            continue
        demands += 1
        if r["rows_asked"] >= 2:
            multi += 1
            s = r["stages_ns"]
            span = (s["last_cqe"] - s["first_cqe"]) / 1e3
            if span > SLOW_SPAN_US:
                slow += 1
                worst.append({"forward": r["forward"], "rows": r["rows_asked"], "span_ms": round(span / 1e3, 2)})
    # Decode cadence: consecutive graph_step records one forward apart are one decode run's steps.
    intervals = []
    for (t0, f0), (t1, f1) in zip(step_ts, step_ts[1:]):
        if f1 == f0 + 1:
            intervals.append((t1 - step_ts[0][0], 1000 * (t1 - t0)))
    return {
        "graph_steps": steps,
        "decode_demand_statuses": dict(statuses),
        "decode_demands_served": demands,
        "multi_row_demands": multi,
        "slow_multi_row_gt_10ms": slow,
        "slowest": sorted(worst, key=lambda x: -x["span_ms"])[:10],
        "thread": last_thread,
        "_intervals": intervals,
    }


def log_scan(path):
    out = {"armed": None, "failstops": [], "dropped_warnings": 0, "errors": []}
    for line in open(path, errors="replace"):
        if "copy engine armed" in line and out["armed"] is None:
            out["armed"] = line.strip()[:200]
        if "timed out or failed" in line or "fail-stop" in line:
            out["failstops"].append(line.strip()[:600])
        if "stage records dropped" in line:
            out["dropped_warnings"] += 1
        if re.search(r"\b(ERROR|Error|Exception)\b", line) and len(out["errors"]) < 40:
            out["errors"].append(line.strip()[:300])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--json")
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(os.path.join(a.out, "requests.jsonl")) if l.strip()]
    summary_path = os.path.join(a.out, "driver_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    stages = stage_scan(os.path.join(a.out, "stages.jsonl"))
    log = log_scan(os.path.join(a.out, "server.log"))

    by_kind = defaultdict(list)
    for r in reqs:
        by_kind[r["kind"]].append(r)
    kinds = {}
    for k, rs in sorted(by_kind.items()):
        errs = Counter(f"{r.get('status')}: {(r.get('error') or '')[:160]}" for r in rs if not r["ok"])
        kinds[k] = {"n": len(rs), "ok": sum(r["ok"] for r in rs), "wall_s_mean": round(statistics.mean(r["wall_s"] for r in rs), 2),
                    "errors": dict(errs)}

    streamed = [r for r in reqs if r.get("chunks")]
    stalls = {f"client_gaps_ge_{s}s": sum(r.get(f"gaps_ge_{s}s", 0) for r in streamed) for s in (0.5, 2.0)}
    stalls["requests_with_gap_ge_0.5s"] = [
        {"idx": r["idx"], "kind": r["kind"], "gap_ms_max": r["gap_ms_max"], "t_start": r["t_start"]}
        for r in streamed if r.get("gaps_ge_0.5s")
    ]
    stalls["client_gap_ms_max"] = max((r["gap_ms_max"] for r in streamed), default=None)
    stalls["stage_multi_row_demands"] = stages["multi_row_demands"]
    stalls["stage_slow_multi_row_gt_10ms"] = stages["slow_multi_row_gt_10ms"]
    stalls["stage_slowest"] = stages["slowest"]

    grammar = defaultdict(lambda: {"n": 0, "valid": 0, "invalid_samples": []})
    for r in reqs:
        if "validate" in r and r["ok"]:
            g = grammar[r["validate"]["type"] + ":" + r["kind"]]
            g["n"] += 1
            g["valid"] += bool(r.get("valid"))
            if not r.get("valid") and len(g["invalid_samples"]) < 3:
                g["invalid_samples"].append({"idx": r["idx"], "finish": r.get("finish_reason"), "text": r["text"][:200]})

    det = [r for r in reqs if r.get("determinism")]
    determinism = {"runs": len(det), "identical": sum(bool(r.get("determinism_identical")) for r in det),
                   "distinct_texts": len({r["text"] for r in det})}

    windows = defaultdict(list)
    for r in streamed:
        if r["chunks"] >= 16:
            windows[int(r["t_start"] // WINDOW_S)].append(r["ms_per_token"])
    swindows = defaultdict(list)
    for t, ms in stages["_intervals"]:
        swindows[int(t // WINDOW_S)].append(ms)
    trend = [{"window_min": f"{10 * w}-{10 * w + 10}", "client_ms_per_token_p50": med(windows.get(w, [])),
              "client_requests": len(windows.get(w, [])), "step_interval_ms_p50": med(swindows.get(w, [])),
              "steps": len(swindows.get(w, []))} for w in sorted(set(windows) | set(swindows))]

    t = stages["thread"] or {}
    counters = {k: t.get(k) for k in ("copy_jobs", "copy_lanes", "copy_bytes", "copy_fallbacks", "copy_errors",
                                      "copy_generation_mismatches", "leases_copied", "leases_acked", "leases_voided",
                                      "lease_double_signal", "late_after_fatal", "late_after_terminal", "read_errors",
                                      "slots_quarantined", "copy_latency_max_ns")}
    other = {k: kinds.get(k) for k in ("n2", "abort", "burst")}
    report = {
        "dir": a.out,
        "driver": summary,
        "survival": {"server_died": summary.get("server_died"), "armed": log["armed"], "failstops": log["failstops"],
                     "stage_dropped_warnings": log["dropped_warnings"]},
        "requests": {"total": len(reqs), "ok": sum(r["ok"] for r in reqs), "by_kind": kinds,
                     "prompt_tokens_max": max(((r.get("usage") or {}).get("prompt_tokens") or 0 for r in reqs), default=0),
                     "streamed": len(streamed), "aborted": sum(1 for r in reqs if "aborted_after_chunks" in r),
                     "max_tokens_hist": dict(Counter(("1" if (r["params"].get("max_tokens") or r["params"].get("sampling_params", {}).get("max_new_tokens") or 0) == 1 else ">1") for r in reqs))},
        "stalls": stalls,
        "counters": counters,
        "graph_steps": stages["graph_steps"],
        "decode_demand_statuses": stages["decode_demand_statuses"],
        "grammar": dict(grammar),
        "determinism": determinism,
        "trend": trend,
        "notable_kinds": other,
        "log_errors": log["errors"],
    }
    print(json.dumps(report, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
