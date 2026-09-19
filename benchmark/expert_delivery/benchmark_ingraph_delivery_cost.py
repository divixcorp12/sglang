#!/usr/bin/env python3
"""Measure InGraphRowBackend.post() delivery cost at real per-layer geometry.

This is a defensible-upper-bound instrument for expert-row delivery cost,
deliberately built through ``InGraphRowBackend`` (python/sglang/srt/layers/moe/
expert_row_plan.py) rather than the doorbell copier: ``post`` issues
``copy_expert_row_segments_gpu`` synchronously on the current stream, so it
serializes with whatever compute shares that stream. The doorbell's own
DOORBELL_ROW_MS number is not used anywhere in this script or its arithmetic.

A single production ``post()`` call moves at most ``top_k`` rows for one
target layer (scratch is sized ``bs * top_k`` rows/layer, not sized for a
whole token), so this benchmark sweeps small per-call row counts (0..top_k)
and fits a linear model ``ms(n) = fixed_ms + per_row_ms * n``, the same shape
E28 fit from a live nsys trace of this exact kernel (0.2239 ms/row + 0.006 ms,
MOE_EXPERT_TRANSFER.md). Per-token cost is then 48 launches (one per layer)
plus per_row_ms times the token's total miss count -- linear in total misses,
independent of how they are spread across layers.

Row geometry (H=2560, I=640, the six NVFP4_STREAM_TENSORS) reproduces the
real per-expert-row byte layout used by benchmark/expert_doorbell/bench_doorbell.py
on divix01 (cc-pcie-bench Setup): total 2,764,808 B/row, 8 bytes more than the
2,764,800 planning-doc rounding (two float32 alpha scalars). This script
derives the actual byte total from the constructed segments table rather than
hardcoding either constant.

Usage:
    python benchmark/expert_delivery/benchmark_ingraph_delivery_cost.py \
        --rows 0 1 2 3 5 8 10 --warmup 50 --iterations 200 --cuda-graph
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence

import torch

# Real per-expert-row tensor shapes (H=2560, I=640), matching production and
# benchmark/expert_doorbell's cc-pcie-bench Setup (verified on divix01).
H, INTER = 2560, 640
TENSOR_SHAPES: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
    ("w13_weight", (2 * INTER, H // 2), torch.uint8),
    ("w2_weight", (H, INTER // 2), torch.uint8),
    ("w13_blockscale_swizzled", (2 * INTER, H // 16), torch.float8_e4m3fn),
    ("w2_blockscale_swizzled", (H, INTER // 16), torch.float8_e4m3fn),
    ("g1_alphas", (), torch.float32),
    ("g2_alphas", (), torch.float32),
)

TAG = 0
GIB = 1024**3


def _positive_int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 5, 8, 10],
        help="per-call row counts to sweep (production top_k=10 caps a single call)",
    )
    parser.add_argument("--num-experts", type=int, default=512, help="source rows per tensor")
    parser.add_argument("--cache-slots", type=int, default=16, help="destination rows per tensor")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--out", default=None, help="append JSONL records here too")
    args = parser.parse_args()
    if max(args.rows) > args.cache_slots or max(args.rows) > args.num_experts:
        parser.error("--cache-slots/--num-experts must be >= max(--rows)")
    return args


def _build_segments(num_experts: int, cache_slots: int, device: torch.device):
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments

    pairs = []
    for _name, shape, dtype in TENSOR_SHAPES:
        source = torch.zeros((num_experts, *shape), dtype=dtype, pin_memory=True)
        destination = torch.zeros((cache_slots, *shape), dtype=dtype, device=device)
        pairs.append((source, destination))
    segments = expert_row_segments(pairs)
    actual_row_bytes = sum(
        source.numel() // source.shape[0] * source.element_size() for source, _ in pairs
    )
    return segments, actual_row_bytes


def _make_plan(rows: int, num_experts: int, cache_slots: int, device: torch.device):
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    capacity = max(rows, 1)
    expert_ids = torch.zeros(capacity, dtype=torch.int64, device=device)
    if rows > 0:
        expert_ids[:rows] = torch.arange(num_experts - rows, num_experts, dtype=torch.int64, device=device)
    slots = torch.zeros(capacity, dtype=torch.int32, device=device)
    if rows > 0:
        slots[:rows] = torch.arange(cache_slots - rows, cache_slots, dtype=torch.int32, device=device)
    count = torch.tensor([rows], dtype=torch.int32, device=device)
    return ExpertRowPlan(expert_ids=expert_ids, slots=slots, count=count)


def _time_submissions(
    submit,
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Return per-iteration device elapsed time in microseconds."""
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            submit()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            start.record(stream)
            submit()
            end.record(stream)
    stream.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _fit_linear(rows: Sequence[int], median_us: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit of ms(n) = per_row_ms * n + fixed_ms, matching E28's model shape."""
    n = len(rows)
    xs = [float(r) for r in rows]
    ys = [v / 1000.0 for v in median_us]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return 0.0, mean_y
    slope = cov / var
    intercept = mean_y - slope * mean_x
    return slope, intercept


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from sglang.srt.layers.moe.expert_row_plan import InGraphRowBackend

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    segments, actual_row_bytes = _build_segments(args.num_experts, args.cache_slots, device)
    backend = InGraphRowBackend({TAG: segments})
    stream = torch.cuda.Stream(device=device)

    records: list[dict[str, object]] = []
    per_row_medians_us: list[float] = []
    for rows in args.rows:
        plan = _make_plan(rows, args.num_experts, args.cache_slots, device)

        def submit(plan=plan) -> None:
            backend.post(TAG, plan)

        samples_us = _time_submissions(
            submit, stream=stream, warmup=args.warmup, iterations=args.iterations
        )
        median_us = statistics.median(samples_us)
        record = {
            "mode": "eager",
            "rows": rows,
            "row_bytes": actual_row_bytes,
            "bytes_moved": rows * actual_row_bytes,
            "median_latency_us": median_us,
            "p90_latency_us": sorted(samples_us)[int(0.9 * (len(samples_us) - 1))],
            "min_latency_us": min(samples_us),
            "effective_gib_per_s": (
                (rows * actual_row_bytes) / (median_us / 1e6) / GIB if rows > 0 and median_us > 0 else None
            ),
            "warmup": args.warmup,
            "iterations": args.iterations,
        }
        records.append(record)
        per_row_medians_us.append(median_us)

        if args.cuda_graph:
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                submit()
            stream.synchronize()
            graph_samples_us = _time_submissions(
                graph.replay, stream=stream, warmup=args.warmup, iterations=args.iterations
            )
            graph_median_us = statistics.median(graph_samples_us)
            records.append(
                {
                    "mode": "cuda_graph_replay",
                    "rows": rows,
                    "row_bytes": actual_row_bytes,
                    "bytes_moved": rows * actual_row_bytes,
                    "median_latency_us": graph_median_us,
                    "p90_latency_us": sorted(graph_samples_us)[
                        int(0.9 * (len(graph_samples_us) - 1))
                    ],
                    "min_latency_us": min(graph_samples_us),
                    "effective_gib_per_s": (
                        (rows * actual_row_bytes) / (graph_median_us / 1e6) / GIB
                        if rows > 0 and graph_median_us > 0
                        else None
                    ),
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                }
            )

    per_row_ms, fixed_ms = _fit_linear(args.rows, per_row_medians_us)
    summary = {
        "mode": "fit_eager",
        "row_bytes": actual_row_bytes,
        "per_row_ms": per_row_ms,
        "fixed_ms_per_launch": fixed_ms,
        "implied_gib_per_s": (actual_row_bytes / (per_row_ms / 1000.0) / GIB if per_row_ms > 0 else None),
        "rows_swept": args.rows,
    }
    records.append(summary)

    out_lines = [json.dumps(r, sort_keys=True) for r in records]
    for line in out_lines:
        print(line)
    if args.out:
        with open(args.out, "a") as fh:
            for line in out_lines:
                fh.write(line + "\n")


if __name__ == "__main__":
    main()
