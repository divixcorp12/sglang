#!/usr/bin/env python3
"""Measure captured side-stream one-row expert pull request latency.

Each reported JSON object describes one ``post_target``/``join_target`` pair
against a synthetic capacity-1 plan, run either eagerly or as a captured CUDA
graph replay, at a given offered-row count. For example:

    python benchmark/kernels/moe/benchmark_expert_request_latency.py \
        --row-bytes 65536 --counts 0,1 --cuda-graph
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _counts(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(","))
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("counts must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device (default: cuda)")
    parser.add_argument(
        "--source-rows", type=_positive, default=32, help="pinned source rows"
    )
    parser.add_argument(
        "--cache-slots", type=_positive, default=8, help="CUDA destination rows"
    )
    parser.add_argument(
        "--row-bytes", type=_positive, default=65536, help="bytes in the pulled row"
    )
    parser.add_argument(
        "--tensor-count", type=_positive, default=1, help="tensor pairs pulled per target"
    )
    parser.add_argument(
        "--counts", type=_counts, default=(0, 1), help="offered-row counts to measure"
    )
    parser.add_argument("--warmup", type=int, default=25, help="untimed replays")
    parser.add_argument("--iterations", type=_positive, default=200, help="timed replays")
    parser.add_argument(
        "--cuda-graph", action="store_true", help="also measure captured graph replay"
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.cache_slots < 1:
        parser.error("--cache-slots must be positive")
    return args


def _time_calls(
    call: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> float:
    """Return median device elapsed time in microseconds for one call."""
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            start.record(stream)
            call()
            end.record(stream)
    stream.synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    )


def _result(
    *,
    mode: str,
    count: int,
    median_latency_us: float,
    row_bytes: int,
    tensor_count: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        "count": count,
        "device": str(device),
        "iterations": iterations,
        "median_latency_us": median_latency_us,
        "mode": mode,
        "row_bytes": row_bytes,
        "tensor_count": tensor_count,
        "warmup_iterations": warmup,
    }


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the expert request latency benchmark")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    sources = tuple(
        torch.zeros(
            (args.source_rows, args.row_bytes), dtype=torch.uint8, pin_memory=True
        )
        for _ in range(args.tensor_count)
    )
    destinations = tuple(
        torch.zeros(
            (args.cache_slots, args.row_bytes), dtype=torch.uint8, device=device
        )
        for _ in range(args.tensor_count)
    )
    slot = args.cache_slots - 1

    results: list[dict[str, object]] = []
    for count in args.counts:
        pipeline = ExpertGpuPullPipeline(device)
        plan = ExpertRowPlan(
            expert_ids=torch.zeros(1, dtype=torch.int64, device=device),
            slots=torch.tensor([slot], dtype=torch.int32, device=device),
            count=torch.tensor([count], dtype=torch.int32, device=device),
        )
        segments = expert_row_segments(list(zip(sources, destinations)))
        target = pipeline.create_target(f"count-{count}", segments, plan, slot=slot)

        def submit_eager() -> None:
            pipeline.post_target(target)
            pipeline.join_target(target)

        eager_latency_us = _time_calls(
            submit_eager,
            stream=torch.cuda.current_stream(device),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(
            _result(
                mode="eager",
                count=count,
                median_latency_us=eager_latency_us,
                row_bytes=args.row_bytes,
                tensor_count=args.tensor_count,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
        )

        if args.cuda_graph:
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                submit_eager()
            torch.cuda.synchronize(device)
            graph_latency_us = _time_calls(
                graph.replay,
                stream=torch.cuda.current_stream(device),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.append(
                _result(
                    mode="cuda_graph_replay",
                    count=count,
                    median_latency_us=graph_latency_us,
                    row_bytes=args.row_bytes,
                    tensor_count=args.tensor_count,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
            )
            del graph
    return results


def main() -> None:
    args = _parse_args()
    for result in _run(args):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
