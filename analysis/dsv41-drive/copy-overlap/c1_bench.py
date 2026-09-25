#!/usr/bin/env python3
"""C1 copy bandwidth by launch shape, against the copy engine, alone and beside a compute load.

Pinned host rows (six real EXL3 segments, 13,315,584 B per row) on one NUMA node, copied to VRAM slots by:
  sm:G:U  -- the production kernel (U=0) or the bench's unrolled kernel (U loads in flight per thread) at grid G
  ce      -- cudaMemcpyAsync per segment (the copy engine)
Each cell reports GB/s alone and GB/s while a second stream runs an HBM-bound GEMV loop, plus how much that loop slows.

    PYTHONPATH=<repo>/python gpu-run.sh python c1_bench.py --repo <repo> --out results.jsonl
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SEGMENTS = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)
ROW_BYTES = sum(SEGMENTS)
ROWS = 160  # 2.1 GB of host rows, so no launch reuses a row within 40 launches
SLOTS = 16
NS = (1, 2, 4)


def load_bench(repo: Path):
    from sglang.kernels.jit.utils.compile.loader import load_jit

    here = Path(__file__).resolve().parent
    return load_jit(
        "c1_bench_copy_overlap",
        cuda_files=[str(here / "c1_bench.cuh")],
        cuda_wrappers=[("bench_copy_segments", "bench_copy_segments")],
        extra_include_paths=[str(repo / "python/sglang/kernels/jit/csrc")],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    dev = torch.device("cuda")
    mod = load_bench(repo)
    src = [
        allocate_host_slab(ROWS, (b,), torch.uint8, register=True, placement=((a.node, ROWS * b),))
        for b in SEGMENTS
    ]
    for s in src:
        for r in range(ROWS):
            s[r, :512].fill_(r % 251)
    dst = [torch.zeros((SLOTS, b), dtype=torch.uint8, device=dev) for b in SEGMENTS]
    table = expert_row_segments(list(zip(src, dst))).table
    slots = torch.arange(SLOTS, dtype=torch.int32, device=dev)
    cursor = [0]

    def next_rows(n: int) -> torch.Tensor:
        rows = [(cursor[0] + i) % ROWS for i in range(n)]
        cursor[0] = (cursor[0] + n) % ROWS
        return rows

    def sm(grid: int, unroll: int):
        def launch(rows: list[int]):
            r = torch.tensor(rows, dtype=torch.int64, device=dev)
            c = torch.tensor([len(rows)], dtype=torch.int32, device=dev)
            torch.cuda.current_stream().synchronize()
            return lambda: mod.bench_copy_segments(table, r, slots, c, grid, unroll)

        return launch

    def ce(rows: list[int]):
        def go():
            for i, row in enumerate(rows):
                for s, d in zip(src, dst):
                    d[i].copy_(s[row], non_blocking=True)

        return go

    variants = {"ce": ce, "sm:8:0": sm(8, 0)}
    for grid in (16, 32, 64):
        variants[f"sm:{grid}:0"] = sm(grid, 0)
    for unroll in (2, 4, 8):
        variants[f"sm:8:{unroll}"] = sm(8, unroll)
    variants["sm:16:4"] = sm(16, 4)

    # Correctness: every variant moves the intended rows.
    for name, make in variants.items():
        rows = next_rows(4)
        for d in dst:
            d.zero_()
        make(rows)()
        torch.cuda.synchronize()
        for i, row in enumerate(rows):
            for s, d in zip(src, dst):
                if not torch.equal(d[i].cpu(), s[row]):
                    raise SystemExit(f"{name} copied the wrong bytes for row {row}")

    mat = torch.randn((8192, 16384), dtype=torch.bfloat16, device=dev)  # 256 MiB, past the 96 MiB L2
    vec = torch.randn((16384,), dtype=torch.bfloat16, device=dev)
    side = torch.cuda.Stream()

    def gemv_loop(k: int):
        for _ in range(k):
            torch.mv(mat, vec)

    def timed(fn, stream) -> float:
        with torch.cuda.stream(stream):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
        return e0, e1

    main_stream = torch.cuda.current_stream()
    # GEMV loop cost alone
    gemv_k = 40
    for _ in range(3):
        gemv_loop(gemv_k)
    torch.cuda.synchronize()
    alone_gemv = []
    for _ in range(10):
        e = timed(lambda: gemv_loop(gemv_k), side)
        torch.cuda.synchronize()
        alone_gemv.append(e[0].elapsed_time(e[1]))
    gemv_ms = statistics.median(alone_gemv)
    out = open(a.out, "a")
    meta = {"meta": True, "node": a.node, "gemv_loop_ms": gemv_ms, "gemv_gbs": 256 * 1.048576e6 * gemv_k / (gemv_ms * 1e-3) / 1e9,
            "sglang": sglang.__file__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(meta), flush=True)
    out.write(json.dumps(meta) + "\n")
    for n in NS:
        for name, make in variants.items():
            for _ in range(3):
                make(next_rows(n))()
            torch.cuda.synchronize()
            alone, conc, slow = [], [], []
            for _ in range(a.reps):
                fn = make(next_rows(n))
                e = timed(fn, main_stream)
                torch.cuda.synchronize()
                alone.append(e[0].elapsed_time(e[1]))
                fn = make(next_rows(n))
                g = timed(lambda: gemv_loop(gemv_k), side)
                e = timed(fn, main_stream)
                torch.cuda.synchronize()
                conc.append(e[0].elapsed_time(e[1]))
                slow.append(g[0].elapsed_time(g[1]) / gemv_ms)
            gb = lambda ms: n * ROW_BYTES / (ms * 1e-3) / 1e9
            rec = {"variant": name, "n": n, "alone_ms_p50": statistics.median(alone), "alone_gbs": round(gb(statistics.median(alone)), 2),
                   "with_gemv_gbs": round(gb(statistics.median(conc)), 2), "gemv_slowdown_p50": round(statistics.median(slow), 3)}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
