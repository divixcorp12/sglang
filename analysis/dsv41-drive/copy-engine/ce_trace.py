#!/usr/bin/env python3
"""Per-step RAM-miss chain kernels and copy-engine copies from a NODE-mode decode trace (sqlite export).

Per graph step (the kernels sharing one graph launch's correlation id): the summed device time of each chain kernel
(W1, C1, A1, S, A2, CW, F), the step's kernel span, and the copy-engine copies (memcpy records off the graph's
stream) that start inside the step: count, bytes, busy time, and how much of it overlaps S and CW. Also the driver
calls the copy thread made (cuMemcpyAsync, cuEventQuery), when the trace holds them. Node mode inflates small-kernel
cost: read kernel and copy times from it, never ms/token (CLAUDE.md). Bound memory: run under taskset on divix01.

    python3 ce_trace.py <trace.sqlite> [--json out.json] [--skip 20]
"""

import argparse
import collections
import json
import sqlite3
import statistics

CHAIN = {
    "exl3_ram_miss_post_kernel": "post",
    "exl3_ram_miss_lease_stream_hit_wait_kernel": "W1",
    "copy_expert_row_segments_gpu_kernel": "C1",
    "exl3_ram_miss_lease_stage_ack_kernel": "A",
    "exl3_ram_miss_lease_stream_kernel": "S",
    "exl3_ram_miss_lease_copy_wait_kernel": "CW",
    "exl3_ram_miss_lease_finalize_kernel": "F",
}


def overlap(a0, a1, spans):
    return sum(max(0, min(a1, e) - max(a0, s)) for s, e in spans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json")
    ap.add_argument("--skip", type=int, default=20, help="leading graph steps to drop (capture warm-up, unarmed)")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    kernels = c.execute(
        "select start, end, correlationId, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0 "
        "order by start"
    ).fetchall()
    steps = collections.defaultdict(list)
    graph_streams = set()
    for s, e, corr, name, stream in kernels:
        steps[corr].append((s, e, names[name]))
        graph_streams.add(stream)
    ordered = sorted(steps.values(), key=lambda k: k[0][0])[a.skip :]
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        # nsys 2026.3's export has no graphId on memcpy rows, only graphNodeId (NULL outside a graph).
        cols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        outside = "graphId = 0" if "graphId" in cols else "coalesce(graphNodeId, 0) = 0"
        copies = c.execute(
            f"select start, end, bytes, streamId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where {outside} order by start"
        ).fetchall()
    ce = [(s, e, b) for s, e, b, stream, kind in copies if stream not in graph_streams and kind == 1]
    per_step = []
    for step in ordered:
        t0, t1 = step[0][0], max(k[1] for k in step)
        sums = collections.Counter()
        s_spans, cw_spans = [], []
        for s, e, name in step:
            if name in CHAIN:
                sums[CHAIN[name]] += e - s
                if CHAIN[name] == "S":
                    s_spans.append((s, e))
                elif CHAIN[name] == "CW":
                    cw_spans.append((s, e))
        inside = [x for x in ce if t0 <= x[0] < t1]
        busy = sum(e - s for s, e, _ in inside)
        per_step.append(
            {
                "span_ms": (t1 - t0) / 1e6,
                **{f"{k}_ms": v / 1e6 for k, v in sums.items()},
                "ce_copies": len(inside),
                "ce_mb": sum(b for _, _, b in inside) / 1e6,
                "ce_busy_ms": busy / 1e6,
                "ce_under_s_ms": sum(overlap(s, e, s_spans) for s, e, _ in inside) / 1e6,
                "ce_under_cw_ms": sum(overlap(s, e, cw_spans) for s, e, _ in inside) / 1e6,
            }
        )
    keys = sorted({k for row in per_step for k in row})
    summary = {"steps": len(per_step)}
    for k in keys:
        values = [row.get(k, 0.0) for row in per_step]
        summary[k] = {"mean": round(statistics.fmean(values), 3), "p50": round(statistics.median(values), 3)}
    busy = sum(e - s for s, e, _ in ce)
    summary["ce_gbps"] = round(sum(b for _, _, b in ce) / busy, 2) if busy else None
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        calls = collections.defaultdict(list)
        for name, s, e in c.execute("select nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
            n = names.get(name, "")
            if n.startswith(("cuMemcpyAsync", "cuEventQuery", "cuEventRecord", "cudaGraphLaunch", "cuGraphLaunch")):
                calls[n].append((e - s) / 1e3)
        summary["api_us"] = {
            n: {"n": len(v), "p50": round(statistics.median(v), 2), "p99": round(sorted(v)[int(0.99 * (len(v) - 1))], 2),
                "max": round(max(v), 2)}
            for n, v in calls.items()
        }
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "steps": per_step}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
