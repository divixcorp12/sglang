#!/usr/bin/env python3
"""Copy-engine copies against the decode graph's kernels, from a NODE-mode trace (sqlite export).

ce_trace.py reports the copies' overlap with S and CW only. This adds, per graph step: the copies' busy time (their
union on the timeline), how much of it lies under any graph kernel (the union of kernel intervals), and how much under
each chain-and-layer group's kernels by name class, plus the copy-engine idle share of the step. Copies are memcpy
records outside the graph (graphId 0) on a stream no graph kernel uses, host to device (copyKind 1), as in
ce_trace.py. Node mode inflates small-kernel cost and the gaps between kernels: read overlap and bytes from it, never
ms/token (CLAUDE.md). Run on divix01 under taskset with bounded memory.

    python3 ce_overlap.py <trace.sqlite> [--json out.json]
"""

import argparse
import bisect
import collections
import json
import sqlite3
import statistics


def union(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def intersect_len(a, b):
    """Total length of the intersection of two sorted disjoint interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    steps = collections.defaultdict(list)
    graph_streams = set()
    for s, e, corr, name, stream in c.execute(
        "select start, end, correlationId, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0"
    ):
        steps[corr].append((s, e, names[name]))
        graph_streams.add(stream)
    steps = sorted((sorted(v) for v in steps.values() if len(v) > 1000), key=lambda k: k[0][0])
    # nsys 2026.3's export has no graphId on memcpy rows, only graphNodeId (NULL outside a graph).
    cols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
    outside = "graphId = 0" if "graphId" in cols else "coalesce(graphNodeId, 0) = 0"
    copies = [
        (s, e, b)
        for s, e, b, stream, kind in c.execute(
            f"select start, end, bytes, streamId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where {outside}"
        )
        if stream not in graph_streams and kind == 1
    ]
    copies.sort()
    rows = []
    under_kernel = collections.Counter()
    ci = 0
    for step in steps:
        t0, t1 = step[0][0], max(k[1] for k in step)
        while ci < len(copies) and copies[ci][0] < t0:
            ci += 1
        inside = []
        k = ci
        while k < len(copies) and copies[k][0] < t1:
            inside.append(copies[k])
            k += 1
        ce = union([(s, e) for s, e, _ in inside])
        kern = union([(s, e) for s, e, _ in step])
        busy = sum(e - s for s, e in ce)
        ends = [x[1] for x in ce]
        for s, e, name in step:
            i = bisect.bisect_right(ends, s)
            while i < len(ce) and ce[i][0] < e:
                under_kernel[name] += min(e, ce[i][1]) - max(s, ce[i][0])
                i += 1
        rows.append(
            {
                "span_ms": (t1 - t0) / 1e6,
                "copies": len(inside),
                "mb": sum(b for _, _, b in inside) / 1e6,
                "ce_busy_ms": busy / 1e6,
                "ce_under_any_kernel_ms": intersect_len(ce, kern) / 1e6,
                "ce_with_no_kernel_ms": (busy - intersect_len(ce, kern)) / 1e6,
            }
        )
    n = len(rows)
    summary = {"steps": n, "copies_total": sum(r["copies"] for r in rows), "bytes_total": sum(r["mb"] for r in rows) * 1e6}
    for key in rows[0] if rows else []:
        values = [r[key] for r in rows]
        summary[key] = {"mean": round(statistics.fmean(values), 3), "p50": round(statistics.median(values), 3),
                        "p90": round(sorted(values)[int(0.9 * (n - 1))], 3)}
    summary["ce_under_kernel_ms_per_step_top"] = {
        name: round(v / 1e6 / n, 3) for name, v in under_kernel.most_common(10)
    }
    busy_all = sum(r["ce_busy_ms"] for r in rows)
    summary["ce_overlap_fraction"] = round(sum(r["ce_under_any_kernel_ms"] for r in rows) / busy_all, 4) if busy_all else None
    summary["ce_gbps_while_busy"] = round(summary["bytes_total"] / (busy_all * 1e6), 2) if busy_all else None
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "steps": rows}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
