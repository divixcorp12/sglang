"""DSpark draft-shaped experts through the CPU expert kernel: ms per stage call at 1-6 rows.

Plan: docs/superpowers/plans/2026-10-02-dsv41-dspark-cpu-draft.md, Task 1. Random EXL3 weights at DSV4.1's expert
shape (H 5120, I 2304) and BITS bits, 128 experts per layer as the draft's stages hold them. Each call routes ROWS rows
to top-3 experts:
  independent  every row draws its own 3 distinct experts (the largest union, up to 3*ROWS)
  shared       every row uses the same 3 experts (union 3)
Usage (divix01, repo root, PYTHONPATH=$PWD/python):
  EXL3_MOE_CPU_PIN=0 numactl --membind=1 taskset -c 18-29 python analysis/dsv41-drive/cpu-experts/draft_bench.py \
      BITS THREADS ROWS PATTERNS TAG
BITS, THREADS, ROWS and PATTERNS are comma lists. Results append to draft_bench_results.jsonl beside this file.
"""

import json
import os
import statistics
import sys
import time

import torch

os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

for key in ("SGLANG_EXL3_SRC", "SGLANG_EXL3_BUILD_DIR"):
    os.environ.setdefault(key, arm_env.base_env()[key])

H, I, E, TOPK = 5120, 2304, 128, 3
LIMIT = 10.0
WARMUP, CALLS = 5, 60
RESULTS = os.path.join(HERE, "draft_bench_results.jsonl")


def slabs(bits: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    def trellis(*shape):
        return torch.randint(-32768, 32767, shape, generator=gen, dtype=torch.int16)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).half()

    return {
        "w13_trellis": trellis(E, 2, H // 16, I // 16, 16 * bits),
        "w13_suh": signs(E, 2, H),
        "w13_svh": signs(E, 2, I),
        "w2_trellis": trellis(E, I // 16, H // 16, 16 * bits),
        "w2_suh": signs(E, I),
        "w2_svh": signs(E, H),
    }


def make_layer(ext, s) -> int:
    rows = range(E)
    return ext.exl3_moe_cpu_make_layer(
        [s["w13_trellis"][i, 0] for i in rows],
        [s["w13_suh"][i, 0] for i in rows],
        [s["w13_svh"][i, 0] for i in rows],
        [s["w13_trellis"][i, 1] for i in rows],
        [s["w13_suh"][i, 1] for i in rows],
        [s["w13_svh"][i, 1] for i in rows],
        [s["w2_trellis"][i] for i in rows],
        [s["w2_suh"][i] for i in rows],
        [s["w2_svh"][i] for i in rows],
        [],
        [],
        [],
        0,
        LIMIT,
        0,
    )


def routes(rows: int, pattern: str, gen: torch.Generator) -> torch.Tensor:
    if pattern == "shared":
        return torch.randperm(E, generator=gen)[:TOPK].repeat(rows, 1)
    if pattern == "independent":
        return torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(rows)])
    raise SystemExit(f"unknown pattern {pattern}")


def weight_passes(ids: torch.Tensor) -> tuple[int, int]:
    """(union, sum over the union of ceil(t/2)): the kernel reads an expert once per 2 rows that route to it."""
    counts = torch.bincount(ids.reshape(-1), minlength=E)
    used = counts[counts > 0]
    return int(used.numel()), int(((used + 1) // 2).sum())


def cell(ext, handle, bits, threads, rows, pattern, gen):
    x = (torch.randn(rows, H, generator=gen) * 0.5).half()
    w = torch.full((rows, TOPK), 1.0 / TOPK).half()
    out = torch.empty(rows, H, dtype=torch.float32)
    ms, unions, passes = [], [], []
    for i in range(WARMUP + CALLS):
        ids = routes(rows, pattern, gen)
        start = time.perf_counter()
        ext.exl3_moe_cpu_forward(handle, x, ids, w, out, threads)
        elapsed = (time.perf_counter() - start) * 1e3
        if i >= WARMUP:
            u, p = weight_passes(ids)
            ms.append(elapsed)
            unions.append(u)
            passes.append(p)
    median = statistics.median(ms)
    mean_passes = statistics.mean(passes)
    return {
        "bits": bits,
        "threads": threads,
        "rows": rows,
        "pattern": pattern,
        "ms_median": round(median, 3),
        "ms_p90": round(sorted(ms)[int(0.9 * len(ms))], 3),
        "union_mean": round(statistics.mean(unions), 2),
        "passes_mean": round(mean_passes, 2),
        "ms_per_pass": round(median / mean_passes, 3),
        "draft_step_ms": round(3 * median, 2),
    }


def main():
    bits_list, threads_list, rows_list = (
        [int(v) for v in arg.split(",")] for arg in sys.argv[1:4]
    )
    patterns, tag = sys.argv[4].split(","), sys.argv[5]
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    ext = exl3_ext()
    gen = torch.Generator().manual_seed(20261002)
    for bits in bits_list:
        handle = make_layer(ext, slabs(bits, gen))
        try:
            for threads in threads_list:
                for rows in rows_list:
                    for pattern in patterns:
                        rec = {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "tag": tag,
                            "affinity": sorted(os.sched_getaffinity(0)),
                            **cell(ext, handle, bits, threads, rows, pattern, gen),
                        }
                        line = json.dumps(rec)
                        print(line, flush=True)
                        with open(RESULTS, "a") as f:
                            f.write(line + "\n")
        finally:
            ext.exl3_moe_cpu_free_layer(handle)


if __name__ == "__main__":
    main()
