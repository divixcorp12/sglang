#!/usr/bin/env python3
"""Compare GPU-pull and CUDA-copy-engine NVFP4 expert-row transfers.

Each reported JSON object describes one fixed-plan submission that moves the
same selected rows for all six NVFP4-style tensors. Sources are pinned CPU
allocations; destinations, the GPU plan, and timing events share one CUDA
stream. For example:

    python benchmark/kernels/moe/benchmark_expert_cache_transfer.py \
        --rows 32 --row-bytes 65536 --cuda-graph
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence

import torch


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("all", "gpu", "dma"),
        default="all",
        help="transfer engine or engines to measure (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda", help="CUDA device (default: cuda)"
    )
    parser.add_argument(
        "--rows", type=_positive, default=32, help="selected rows per submission"
    )
    parser.add_argument(
        "--source-experts",
        type=_positive,
        default=128,
        help="number of pinned source rows per tensor",
    )
    parser.add_argument(
        "--cache-slots",
        type=_positive,
        default=128,
        help="number of CUDA destination rows per tensor",
    )
    parser.add_argument(
        "--row-bytes",
        type=_positive,
        default=65536,
        help="bytes in each packed NVFP4 tensor row",
    )
    parser.add_argument(
        "--tensor-count",
        type=_positive,
        default=6,
        help="NVFP4 tensors copied per submission (default: 6)",
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="untimed submissions per path"
    )
    parser.add_argument(
        "--iterations", type=_positive, default=100, help="timed submissions"
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="also measure CUDA-graph replay for the GPU pull path",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.rows > args.source_experts:
        parser.error("--rows cannot exceed --source-experts")
    if args.rows > args.cache_slots:
        parser.error("--rows cannot exceed --cache-slots")
    if args.cuda_graph and args.backend == "dma":
        parser.error("--cuda-graph requires --backend gpu or all")
    return args


def _time_submissions(
    submit: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> float:
    """Return median device elapsed time in microseconds for one submission."""
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
    return statistics.median(
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    )


def _result(
    *,
    requested_backend: str,
    actual_backend: str,
    mode: str,
    median_latency_us: float,
    rows: int,
    bytes_per_submission: int,
    warmup: int,
    iterations: int,
    tensor_count: int,
    device: torch.device,
) -> dict[str, object]:
    median_seconds = median_latency_us / 1_000_000.0
    return {
        "actual_backend": actual_backend,
        "bytes_per_submission": bytes_per_submission,
        "device": str(device),
        "effective_gib_per_s": bytes_per_submission / median_seconds / (1024**3),
        "iterations": iterations,
        "median_latency_us": median_latency_us,
        "mode": mode,
        "requested_backend": requested_backend,
        "rows_per_submission": rows,
        "tensor_count": tensor_count,
        "warmup_iterations": warmup,
    }


def _make_row_plan(
    rows: int, source_experts: int, cache_slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic, unique row/slot selections shared by both engines."""
    selected_rows = torch.arange(
        source_experts - 1, source_experts - rows - 1, -1, dtype=torch.int64
    )
    selected_slots = torch.arange(
        cache_slots - 1, cache_slots - rows - 1, -1, dtype=torch.int32
    )
    return selected_rows, selected_slots


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the expert transfer benchmark")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu
    from sglang.srt.layers.moe.expert_dma import ExpertDMABackend
    from sglang.srt.layers.moe.expert_transfer import FixedRowTransferPlan

    source_rows, destination_slots = _make_row_plan(
        args.rows, args.source_experts, args.cache_slots
    )
    gpu_plan = FixedRowTransferPlan(max_rows=args.rows, device=device)
    gpu_plan.set_rows(
        source_rows,
        destination_slots,
        torch.zeros(args.rows, dtype=torch.int64),
    )
    dma_source_rows: Sequence[int] = tuple(source_rows.tolist())
    dma_destination_slots: Sequence[int] = tuple(destination_slots.tolist())

    sources = tuple(
        torch.empty(
            (args.source_experts, args.row_bytes), dtype=torch.uint8, pin_memory=True
        )
        for _ in range(args.tensor_count)
    )
    destinations = tuple(
        torch.empty(
            (args.cache_slots, args.row_bytes), dtype=torch.uint8, device=device
        )
        for _ in range(args.tensor_count)
    )
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    bytes_per_submission = args.rows * args.row_bytes * args.tensor_count

    def submit_gpu() -> None:
        for source, destination in zip(sources, destinations):
            copy_expert_rows_gpu(
                source,
                destination,
                gpu_plan.source_rows,
                gpu_plan.destination_slots,
                gpu_plan.count,
            )

    dma_backend = ExpertDMABackend()

    def submit_dma() -> None:
        for source, destination in zip(sources, destinations):
            dma_backend.copy_rows(
                source, destination, dma_source_rows, dma_destination_slots
            )

    results: list[dict[str, object]] = []
    if args.backend in ("all", "gpu"):
        gpu_latency_us = _time_submissions(
            submit_gpu,
            stream=stream,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(
            _result(
                requested_backend="gpu",
                actual_backend="gpu",
                mode="eager",
                median_latency_us=gpu_latency_us,
                rows=args.rows,
                bytes_per_submission=bytes_per_submission,
                warmup=args.warmup,
                iterations=args.iterations,
                tensor_count=args.tensor_count,
                device=device,
            )
        )

        if args.cuda_graph:
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                submit_gpu()
            stream.synchronize()
            graph_latency_us = _time_submissions(
                graph.replay,
                stream=stream,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.append(
                _result(
                    requested_backend="gpu",
                    actual_backend="gpu",
                    mode="cuda_graph_replay",
                    median_latency_us=graph_latency_us,
                    rows=args.rows,
                    bytes_per_submission=bytes_per_submission,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    tensor_count=args.tensor_count,
                    device=device,
                )
            )

    if args.backend in ("all", "dma"):
        dma_latency_us = _time_submissions(
            submit_dma,
            stream=stream,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(
            _result(
                requested_backend="dma",
                actual_backend=dma_backend.actual_backend or "unknown",
                mode="eager",
                median_latency_us=dma_latency_us,
                rows=args.rows,
                bytes_per_submission=bytes_per_submission,
                warmup=args.warmup,
                iterations=args.iterations,
                tensor_count=args.tensor_count,
                device=device,
            )
        )
    return results


def main() -> None:
    args = _parse_args()
    for result in _run(args):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
