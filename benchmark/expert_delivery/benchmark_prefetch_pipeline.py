#!/usr/bin/env python3
"""CUDA-event decomposition of the captured one-row prefetch pull pipeline.

Uses the production NVFP4 row geometry and ``ExpertGpuPullPipeline``.  It
prints count-zero and count-one records only after byte validation; the four
intervals are ready-to-copy-start, copy, join exposure, and consumer-ready.
"""

import argparse
import json
import statistics

import torch


H, INTER = 2560, 640
GEOMETRY = (
    ((2 * INTER, H // 2), torch.uint8), ((H, INTER // 2), torch.uint8),
    ((2 * INTER, H // 16), torch.float8_e4m3fn), ((H, INTER // 16), torch.float8_e4m3fn),
    ((), torch.float32), ((), torch.float32),
)


def _segments(device):
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    pairs = []
    for shape, dtype in GEOMETRY:
        source = torch.randint(0, 255, (2, *shape), dtype=torch.uint8).pin_memory() if dtype is torch.uint8 else torch.ones((2, *shape), dtype=dtype).pin_memory()
        destination = torch.zeros((2, *shape), dtype=dtype, device=device)
        pairs.append((source, destination))
    return expert_row_segments(pairs), pairs


def _measure(device, count, iterations):
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan
    segments, pairs = _segments(device)
    plan = ExpertRowPlan(torch.tensor([0], device=device), torch.tensor([1], dtype=torch.int32, device=device), torch.tensor([count], dtype=torch.int32, device=device))
    pipeline = ExpertGpuPullPipeline(device)
    target = pipeline.create_target(f"count-{count}", segments, plan, 1)
    samples = []
    for _ in range(iterations):
        ready, copy_start, copy_end, joined, residual_start, residual_end, consumed = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        ready.record()
        # Inline the production post sequence so copy_start brackets the real
        # copy kernel rather than its preceding cross-stream dependency.
        origin = torch.cuda.current_stream(device)
        target.ready.record(origin)
        with torch.cuda.stream(pipeline.side_stream):
            pipeline.side_stream.wait_event(target.ready)
            copy_start.record(pipeline.side_stream)
            copy_expert_row_segments_gpu(target.segments, target.plan.expert_ids, target.plan.slots, target.plan.count)
            copy_end.record(pipeline.side_stream)
            target.done.record(pipeline.side_stream)
        pipeline.join_target(target)
        joined.record()
        # Use the same registered source/destination geometry for ordinary
        # residual demand, deliberately distinct from the side-pull slot.
        residual_start.record()
        for source, destination in pairs:
            destination[0].copy_(source[0], non_blocking=True)
        residual_end.record()
        # The consumer-ready event is after a real load of the target row.
        sum(destination[1].float().sum() for _, destination in pairs).item()
        consumed.record(); consumed.synchronize()
        samples.append({
            "ready_to_copy_start_ms": ready.elapsed_time(copy_start),
            "copy_ms": copy_start.elapsed_time(copy_end),
            "join_exposure_ms": copy_end.elapsed_time(joined),
            "residual_demand_ms": residual_start.elapsed_time(residual_end),
            "consumer_ready_ms": ready.elapsed_time(consumed),
        })
    if count:
        for source, destination in pairs:
            torch.testing.assert_close(destination[1].cpu(), source[0])
    row_bytes = sum(source[0].numel() * source.element_size() for source, _ in pairs)
    return {"count": count, "payload_bytes": count * row_bytes, "physical_rows": count, **{key: statistics.median([sample[key] for sample in samples]) for key in samples[0]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    for count in (0, 1):
        print(json.dumps(_measure(device, count, args.iterations), sort_keys=True))


if __name__ == "__main__":
    main()
