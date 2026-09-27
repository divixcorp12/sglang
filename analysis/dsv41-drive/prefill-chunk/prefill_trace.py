"""Where a traced prefill's time goes: gathers, compute, copies and GPU idle, per chunk.

Usage: prefill_trace.py <run dir>   (reads <run dir>/trace.sqlite and stages.jsonl, from chunk_smoke.sh CHUNK_NSYS=1)

The window is the eager kernels before the first CUDA-graph kernel (decode). Chunks are split at the per-layer
`sparse_mla_prefill*` kernels: 40 per forward, one per layer.
"""

import json
import sqlite3
import sys
from collections import defaultdict

run = sys.argv[1]
db = sqlite3.connect(f"file:{run}/trace.sqlite?mode=ro", uri=True)
q = db.execute

names = dict(q("select id, value from StringIds"))
t_graph = q("select coalesce(min(start), 1e30) from CUPTI_ACTIVITY_KIND_KERNEL where graphId > 0").fetchone()[0]
kernels = q(
    "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start < ? order by start", (t_graph,)
).fetchall()
t0, t1 = kernels[0][0], max(k[1] for k in kernels)


def category(name: str) -> str:
    if name.startswith("_gather_host_rows"):
        return "gather (pinned RAM over PCIe)"
    if name.startswith("exl3_"):
        return "exl3 gemm/gemv"
    if "sparse_mla" in name or "flashmla" in name:
        return "attention"
    if "indexer" in name.lower() or "topk" in name.lower():
        return "indexer"
    return "other"


# Chunk boundaries: a forward's first attention-prefill kernel follows its predecessor's 40th.
mla = [k[0] for k in kernels if names[k[2]].startswith("sparse_mla_prefill")]
bounds = [mla[i] for i in range(0, len(mla), 40)] + [t1]
print(f"prefill window {(t1 - t0) / 1e9:.1f} s, {len(kernels)} kernels, {len(mla)} attention-prefill kernels "
      f"-> {len(mla) / 40:.2f} forwards")


def summarise(lo: int, hi: int, label: str) -> None:
    busy = defaultdict(int)
    gaps = defaultdict(lambda: [0, 0])
    prev_end, prev_name = None, None
    covered_to = lo
    union = 0
    for s, e, n in kernels:
        if s < lo or s >= hi:
            continue
        name = names[n]
        busy[category(name)] += e - s
        if prev_end is not None and s - prev_end > 1_000_000:
            g = gaps[f"{names[prev_name][:34]} -> {name[:30]}"]
            g[0] += 1
            g[1] += s - prev_end
        if e > covered_to:
            union += e - max(s, covered_to)
            covered_to = e
        prev_end, prev_name = max(e, prev_end or 0), n
    span = hi - lo
    print(f"\n== {label}: {span / 1e9:.1f} s; GPU busy (union) {union / 1e9:.1f} s, idle {(span - union) / 1e9:.1f} s")
    for c, v in sorted(busy.items(), key=lambda x: -x[1]):
        print(f"   {c:32s} {v / 1e9:6.2f} s")
    print("   idle gaps > 1 ms, by neighbours (count, total s):")
    for k, (n, v) in sorted(gaps.items(), key=lambda x: -x[1][1])[:6]:
        print(f"     {n:5d} {v / 1e9:6.2f}  {k}")
    for kind, label2 in ((1, "HtoD"), (8, "DtoD"), (2, "DtoH")):
        n, b, d = q(
            "select count(*), coalesce(sum(bytes),0), coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where copyKind=? and start>=? and start<?", (kind, lo, hi)
        ).fetchone()
        if n:
            print(f"   memcpy {label2}: {n} copies, {b / 1e9:.2f} GB, {d / 1e9:.2f} s")
    for api in ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaMemcpyAsync"):
        n, d = q(
            "select count(*), coalesce(sum(r.end-r.start),0) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
            "on s.id=r.nameId where s.value like ? and r.start>=? and r.start<?", (api + "%", lo, hi)
        ).fetchone()
        if n:
            print(f"   host {api}: {n} calls, {d / 1e9:.2f} s")


summarise(t0, t1, "whole prefill")
for i in range(len(bounds) - 1):
    summarise(bounds[i], bounds[i + 1], f"forward {i}")

# Host-side NVMe reads per forward, from the expert trace (its clock differs; forwards are matched by order).
fw = defaultdict(lambda: {"read": 0.0, "ram": 0, "vram": 0, "experts": 0, "layers": 0, "tokens": 0})
for line in open(f"{run}/stages.jsonl"):
    r = json.loads(line)
    if r.get("phase") != "extend" or "forward" not in r:
        continue
    f = fw[r["forward"]]
    f["read"] += r["read_ms"]
    f["ram"] += r["ram_miss"]
    f["vram"] += r["vram_miss"]
    f["experts"] += len(r["experts"])
    f["layers"] += 1
    f["tokens"] = max(f["tokens"], r["tokens"])
print("\nexpert trace, extend forwards: fwd tokens layers experts vram_miss ram_miss(NVMe rows) read_s")
for k in sorted(fw):
    f = fw[k]
    print(f"  {k:4d} {f['tokens']:5d} {f['layers']:3d} {f['experts']:6d} {f['vram']:6d} {f['ram']:6d} "
          f"{f['read'] / 1e3:6.1f}")
