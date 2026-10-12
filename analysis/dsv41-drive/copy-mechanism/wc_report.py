#!/usr/bin/env python3
"""Write-combined vs ordinary pinned source slab (wc_bench.py JSONL), paired by cell shape. Pure Python.

    python3 wc_report.py <divix01-wc.jsonl>   # markdown; exit 1 if the WC fresh check read stale bytes

Slabs: `pinned` (allocate_host_slab, the sweep's), `hostalloc` (cudaHostAlloc Mapped: the allocation path without
WC, so a WC effect is not an allocator effect) and `wc` (cudaHostAlloc Mapped | WriteCombined). Each shape ran once
per slab per round, interleaved; the median over rounds is reported.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

SLABS = ("pinned", "hostalloc", "wc")


def pairs(records: list[dict]) -> list[dict]:
    by: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("kind") == "cell" and "slab" in r:
            by[(r["method"], r["grid"], r["a"], r["b"])][r["slab"]].append(r["gbs"])
    rows = []
    for (method, grid, a, b), slabs in sorted(by.items()):
        gbs = {s: round(statistics.median(slabs[s]), 3) for s in SLABS if s in slabs}
        base = gbs.get("pinned")
        rows.append({"method": method, "grid": grid, "a": a, "b": b, "gbs": gbs,
                     "wc_ratio": gbs["wc"] / base if base and "wc" in gbs else None,
                     "hostalloc_ratio": gbs["hostalloc"] / base if base and "hostalloc" in gbs else None})
    return rows


def main() -> int:
    records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    print("| method | grid | a | pinned GB/s | hostalloc GB/s | WC GB/s | WC / pinned | hostalloc / pinned |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    for r in pairs(records):
        g = r["gbs"]
        print(f"| {r['method']} | {r['grid']} | {r['a']} | {fmt(g.get('pinned'))} | {fmt(g.get('hostalloc'))} | "
              f"{fmt(g.get('wc'))} | {fmt(r['wc_ratio'])} | {fmt(r['hostalloc_ratio'])} |")
    fresh = [r for r in records if r.get("kind") == "fresh"]
    for r in fresh:
        print(f"fresh {r['method']}: {r['fresh']}")
    for r in records:
        if r.get("kind") == "host_time":
            print(f"host {r['slab']}: write {r['write_bytes'] / r['write_ns']:.2f} GB/s, "
                  f"read {r['read_bytes'] / r['read_ns']:.3f} GB/s")
    bad = any(r["method"] == "sm_cv16@wc" and not r["fresh"] for r in fresh)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
