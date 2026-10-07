"""Join the instrumented host's bounded CPU/copy event logs by NUMA group and sequence.

DMA completion is bracketed by host polls. Classification leaves overlapping intervals ambiguous.
Run after shutdown: python expert_job_trace.py '/path/events.*.jsonl' --output findings.json
"""

import argparse
import bisect
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import re


def summarize(pattern, start_ns=None, end_ns=None):
    cpus, copies, gates = {}, defaultdict(dict), {}
    clocks, events = [], []
    for filename in sorted(glob.glob(pattern)):
        records = [json.loads(line) for line in Path(filename).read_text().splitlines()]
        if not records or records[0].get("schema") != 1 or "dropped" not in records[-1]:
            raise ValueError(f"incomplete trace: {filename}")
        if records[-1]["dropped"]:
            raise ValueError(f"dropped events: {filename}: {records[-1]['dropped']}")
        clocks.append(records[0])
        match = re.search(r"cpu-exp(\d+)", filename)
        cpu_group = int(match[1]) if match else 0
        for event in records[1:-1]:
            e = dict(event)
            events.append(e)
            kind = e["event"]
            if kind.startswith("cpu_"):
                e["group"] = cpu_group
                job = cpus.setdefault((cpu_group, e["seq"], e["row"]), {})
                if kind in job:
                    raise ValueError("duplicate CPU identity; select one process/service per input pattern")
                job[kind] = e
            elif kind == "gate_open":
                gates[e["gen"]] = e
            elif kind.startswith("copy_") or kind == "group_done":
                copies[e["gen"], e["group"]][kind] = e
    cpu_index = defaultdict(list)
    for (group, seq, row), job in cpus.items():
        cpu_index[group, row].append((seq, job))
    for key, values in cpu_index.items():
        values.sort(key=lambda item: item[0])
        cpu_index[key] = ([item[0] for item in values], [item[1] for item in values])
    rows = []
    for (gen, group), record in copies.items():
        needed = {"copy_submit", "copy_issue", "group_done"}
        if not needed <= record.keys() or gen not in gates:
            raise ValueError(f"incomplete copy record: {gen}, {group}")
        if start_ns is not None and gates[gen]["ns"] < start_ns:
            continue
        if end_ns is not None and gates[gen]["ns"] > end_ns:
            continue
        submit, issue = (record[k] for k in ("copy_submit", "copy_issue"))
        # Older captures may retire a copy between the observer's poll and the retirement query.
        # Retirement proves completion by then; without a recorded pending poll the lower bound stays unknown.
        dma = record.get("copy_dma_observed", {"ns": record["group_done"]["ns"], "a": 0})
        last_seq = submit["a"] if submit["b"] else submit["seq"]
        sequences, candidates = cpu_index.get((group, submit["row"]), ([], []))
        jobs = candidates[bisect.bisect_left(sequences, submit["seq"]):bisect.bisect_right(sequences, last_seq)]
        has_cpu = bool(submit["b"] or submit["c"])
        if has_cpu and (not jobs or any(not {"cpu_submit", "cpu_start", "cpu_end", "cpu_shape"} <= j.keys() for j in jobs)):
            raise ValueError(f"incomplete CPU record: {gen}, {group}")
        if not has_cpu:
            jobs = []
        cpu_end = max((j["cpu_end"]["ns"] for j in jobs), default=0)
        # Ten microseconds protects classification from the small end-marker -> done-store interval.
        tolerance = 10_000
        if not has_cpu:
            last = "dma_only" if issue["a"] else "no_work"
        elif not issue["a"]:
            last = "cpu_only"
        elif cpu_end > dma["ns"] + tolerance:
            last = "cpu_last_confirmed"
        elif dma["a"] > cpu_end + tolerance:
            last = "dma_last_confirmed"
        else:
            last = "ambiguous"
        def ms(ns):
            return ns / 1e6
        rows.append({
            "gen": gen, "group": group, "row": submit["row"], "last": last,
            "submit_ns": submit["ns"], "gate_ns": gates[gen]["ns"],
            "host_to_gate_ms": ms(gates[gen]["ns"] - submit["ns"]),
            "dma_bound_source": "poll" if "copy_dma_observed" in record else "retirement_fallback",
            "dma_bytes": issue["a"], "dma_lanes": issue["b"], "forced_cpu_lanes": submit["b"],
            "cpu_lanes": sum(j["cpu_shape"]["b"] for j in jobs),
            "token_expert_routes": sum(j["cpu_shape"]["c"] for j in jobs),
            "cpu_parts": [j["cpu_submit"]["a"] for j in jobs],
            "cpu_ready_after_copy_submit_ms": [ms(j["cpu_submit"]["ns"] - submit["ns"]) for j in jobs],
            "cpu_queue_ms": [ms(j["cpu_start"]["ns"] - j["cpu_submit"]["ns"]) for j in jobs],
            "cpu_compute_ms": [ms(j["cpu_end"]["ns"] - j["cpu_start"]["ns"]) for j in jobs],
            "cpu_end_ns": cpu_end, "dma_pending_ns": dma["a"], "dma_done_observed_ns": dma["ns"],
            "dma_after_cpu_lower_ms": ms(max(0, dma["a"] - cpu_end)) if has_cpu else None,
            "cpu_after_dma_lower_ms": ms(max(0, cpu_end - dma["ns"])) if has_cpu else None,
        })
    if not rows:
        raise ValueError("no completed copy records")
    rows.sort(key=lambda r: (r["gen"], r["group"]))
    layers = []
    by_gen = defaultdict(list)
    for row in rows:
        by_gen[row["gen"]].append(row)
    for gen, groups in by_gen.items():
        required = gates[gen]["a"]
        if required and sum(1 << g["group"] for g in groups) != required:
            raise ValueError(f"missing NUMA group: {gen}")
        cpu_end = max(g["cpu_end_ns"] for g in groups)
        dma_groups = [g for g in groups if g["dma_bytes"]]
        dma_upper = max((g["dma_done_observed_ns"] for g in dma_groups), default=0)
        dma_lower = max((g["dma_pending_ns"] for g in dma_groups), default=0)
        if not cpu_end:
            last = "dma_only" if dma_groups else "no_work"
        elif not dma_groups:
            last = "cpu_only"
        elif cpu_end > dma_upper + 10_000:
            last = "cpu_last_confirmed"
        elif dma_lower > cpu_end + 10_000:
            last = "dma_last_confirmed"
        else:
            last = "ambiguous"
        layers.append({"gen": gen, "row": groups[0]["row"], "last": last,
                       "gate_ns": gates[gen]["ns"],
                       "host_to_gate_ms": (gates[gen]["ns"] - min(g["submit_ns"] for g in groups)) / 1e6,
                       "cpu_end_ns": cpu_end, "dma_pending_ns": dma_lower, "dma_done_observed_ns": dma_upper})
    return {"clock": "CLOCK_MONOTONIC", "clock_anchors": clocks,
            "classification_tolerance_ns": 10_000, "group_records": len(rows),
            "classifications": dict(Counter(r["last"] for r in rows)), "rows": rows,
            "layer_records": len(layers), "layer_classifications": dict(Counter(r["last"] for r in layers)),
            "layer_rows": layers}, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    args = parser.parse_args()
    result, events = summarize(args.pattern, args.start_ns, args.end_ns)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    # Chrome/Perfetto importable companion. Original event labels/fields remain available on each instant.
    origin = min(e["ns"] for e in events)
    chrome = [{"name": e["event"], "ph": "i", "s": "t", "pid": 0,
               "tid": e["group"], "ts": (e["ns"] - origin) / 1000,
               "args": e} for e in events]
    args.output.with_suffix(".timeline.json").write_text(json.dumps({"traceEvents": chrome}))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "layer_rows", "clock_anchors")}, indent=2))


if __name__ == "__main__":
    main()
