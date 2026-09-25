"""Decode kernels per step of a node-mode Nsight export, by source group (plan 2026-09-25-dsv41-decode-fusion-2).

Usage: kernel_groups.py <report.sqlite> [--min-kernels N]

A decode step is one graph launch (one correlationId) with at least N graph kernels (default 1500). Node mode inflates
tiny kernels, so the us column ranks groups; only the counts are exact.
"""

import argparse
import collections
import sqlite3
import statistics

CASTS = ("direct_copy_kernel_cuda", "bfloat16_copy_kernel_cuda")


def group(name: str, prev: str, nxt: str) -> str:
    cast = "elementwise" in name and any(c in name for c in CASTS)
    if "exl3_gemv" in name:
        return "exl3 gemv"
    if cast and "exl3_gemv" in nxt:
        return "cast before gemv"
    if cast and "exl3_gemv" in prev:
        return "cast after gemv"
    if (cast and "exl3_moe_gather" in prev) or "scale_to_bf16" in name:
        return "MoE combine"
    if "AUnaryFunctor<c10::BFloat16" in name or "CUDAFunctor_add<c10::BFloat16>" in name:
        return "MoE combine"
    if "CatArrayBatchedCopy" in name or "silu_mul_clamp" in name:
        return "shared expert glue"
    if "_hc_" in name or "_mhc_" in name:
        return "mHC"
    if "exl3_ram_miss" in name or "copy_expert_row_segments" in name or "CUDAFunctor_add<int>" in name:
        return "RAM-miss chain"
    if any(k in name for k in ("direct_gather_destinations", "direct_commit_gather", "plan_unique_routes", "route_tables")):
        return "residency bookkeeping"
    if "exl3_moe_kernel" in name or "exl3_moe_gather" in name:
        return "routed MoE"
    if "tiny_n_gemm" in name or "_router_triton" in name:
        return "router"
    if "engram" in name:
        return "engram"
    attention = ("sparse_mla", "rope", "_page_", "FillFunctor<signed char>", "RMSNorm", "cutlass", "flash_c", "index_k",
                 "deep_gemm", "gemvx", "n128k512")
    if any(k in name for k in attention):
        return "attention"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--min-kernels", type=int, default=1500)
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    rows = db.execute(
        "select k.start, k.end, k.demangledName, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "where k.graphNodeId is not null order by k.start"
    ).fetchall()
    names = dict(db.execute("select id, value from StringIds").fetchall())
    per = collections.defaultdict(list)
    for r in rows:
        per[r[3]].append(r)
    steps = [ks for ks in per.values() if len(ks) >= args.min_kernels]
    if not steps:
        raise SystemExit("no decode steps found")
    counts = collections.Counter(len(ks) for ks in steps)
    print(f"decode steps {len(steps)}; kernels per step: {dict(sorted(counts.items()))}")
    cnt, us = collections.Counter(), collections.Counter()
    for ks in steps:
        seq = [names[r[2]] for r in ks]
        for i, r in enumerate(ks):
            g = group(seq[i], seq[i - 1] if i else "", seq[i + 1] if i + 1 < len(seq) else "")
            cnt[g] += 1
            us[g] += (r[1] - r[0]) / 1e3
    n = len(steps)
    for g, c in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"{c / n:8.1f} kernels {us[g] / n:10.1f} us  {g}")
    busy = [sum(r[1] - r[0] for r in ks) / 1e6 for ks in steps]
    print(f"median summed kernel time per step {statistics.median(busy):.2f} ms (node mode, waits included)")


if __name__ == "__main__":
    main()
