"""JIT serving selector for a single-row, non-resident expert prefetch offer."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_expert_prefetch_top1_module():
    return load_jit(
        "expert_prefetch_top1",
        cuda_files=["moe/expert_prefetch_top1.cuh"],
        cuda_wrappers=[("select_prefetch_top1_gpu", "select_prefetch_top1_gpu")],
    )


def supports_prefetch_top1_cuda(scores: torch.Tensor, expert_to_slot: torch.Tensor) -> bool:
    """Whether the JIT path preserves the reference candidate semantics.

    The kernel deliberately specializes only the one-row fp32 scorer output.
    For that shape summation is an identity operation, so it exactly preserves
    the reference dtype and reduction. Wider scorer batches use the reference
    stable-sort candidate bank instead.
    """
    return (
        scores.device.type == "cuda"
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.shape[0] == 1
        and scores.is_contiguous()
        and expert_to_slot.device == scores.device
        and expert_to_slot.dtype == torch.int64
        and expert_to_slot.ndim == 1
        and expert_to_slot.numel() == scores.shape[1]
        and expert_to_slot.is_contiguous()
    )


def select_prefetch_top1_cuda(
    scores: torch.Tensor,
    expert_to_slot: torch.Tensor,
    expert_id_out: torch.Tensor,
    valid_out: torch.Tensor,
    count_out: torch.Tensor,
) -> None:
    """Write persistent ``id``, ``valid`` and ``count`` for one serving offer."""
    if not supports_prefetch_top1_cuda(scores, expert_to_slot):
        raise ValueError("unsupported expert-prefetch top-1 JIT inputs")
    for name, tensor, dtype in (
        ("expert_id_out", expert_id_out, torch.int64),
        ("valid_out", valid_out, torch.bool),
        ("count_out", count_out, torch.int32),
    ):
        if (
            tensor.device != scores.device
            or tensor.dtype != dtype
            or tensor.numel() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous CUDA {dtype} scalar")
    _jit_expert_prefetch_top1_module().select_prefetch_top1_gpu(
        scores.reshape(-1), expert_to_slot, expert_id_out, valid_out, count_out
    )
