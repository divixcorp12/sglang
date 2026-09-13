"""Focused tests for the fixed-plan NVFP4 expert-row GPU transfer kernel."""

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Expert cache transfer tests require a CUDA GPU.",
)


def _make_plan(source_rows: list[int], destination_slots: list[int]):
    capacity = 8
    source_plan = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    destination_plan = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    source_plan[: len(source_rows)] = torch.tensor(
        source_rows, dtype=torch.int64, device="cuda"
    )
    destination_plan[: len(destination_slots)] = torch.tensor(
        destination_slots, dtype=torch.int32, device="cuda"
    )
    count = torch.tensor([len(source_rows)], dtype=torch.int32, device="cuda")
    return source_plan, destination_plan, count


def test_copy_expert_rows_gpu_moves_selected_pinned_rows():
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu

    source = torch.arange(8 * 3 * 7, dtype=torch.uint8).reshape(8, 3, 7).pin_memory()
    destination = torch.full((6, 3, 7), 255, dtype=torch.uint8, device="cuda")
    source_rows, destination_slots, count = _make_plan([5, 1, 7, 3], [2, 4, 0, 5])

    copy_expert_rows_gpu(source, destination, source_rows, destination_slots, count)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        destination[torch.tensor([2, 4, 0, 5], device="cuda")],
        source[torch.tensor([5, 1, 7, 3])].to("cuda"),
    )
    assert torch.equal(destination[1], torch.full_like(destination[1], 255))


def test_copy_expert_rows_gpu_rejects_invalid_launch_inputs():
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu

    source = torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4)
    destination = torch.empty((8, 4), dtype=torch.uint8, device="cuda")
    source_rows, destination_slots, count = _make_plan([0], [0])

    with pytest.raises(ValueError, match="pinned or CUDA-registered"):
        copy_expert_rows_gpu(source, destination, source_rows, destination_slots, count)

    with pytest.raises(ValueError, match="matching dtype"):
        copy_expert_rows_gpu(
            source.pin_memory(),
            destination.to(torch.int16),
            source_rows,
            destination_slots,
            count,
        )

    with pytest.raises(ValueError, match="CUDA"):
        copy_expert_rows_gpu(
            source.pin_memory(),
            destination,
            source_rows.cpu(),
            destination_slots,
            count,
        )


def test_copy_expert_rows_gpu_replays_changed_fixed_plan():
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu

    source = torch.arange(8 * 16, dtype=torch.uint8).reshape(8, 16).pin_memory()
    destination = torch.full((8, 16), 255, dtype=torch.uint8, device="cuda")
    source_rows, destination_slots, count = _make_plan([0, 1], [1, 3])

    copy_expert_rows_gpu(source, destination, source_rows, destination_slots, count)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_expert_rows_gpu(source, destination, source_rows, destination_slots, count)

    destination.fill_(255)
    source_rows[:3].copy_(torch.tensor([6, 2, 4], dtype=torch.int64, device="cuda"))
    destination_slots[:3].copy_(
        torch.tensor([0, 5, 7], dtype=torch.int32, device="cuda")
    )
    count.fill_(3)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        destination[torch.tensor([0, 5, 7], device="cuda")],
        source[torch.tensor([6, 2, 4])].to("cuda"),
    )
    assert torch.equal(destination[1], torch.full_like(destination[1], 255))
