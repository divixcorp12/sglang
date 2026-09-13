"""CUDA-graph-safe pull transfers from pinned expert rows into an HBM cache."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_expert_cache_transfer_module() -> Module:
    return load_jit(
        "expert_cache_transfer",
        cuda_files=["moe/expert_cache_transfer.cuh"],
        cuda_wrappers=[("copy_expert_rows_gpu", "copy_expert_rows_gpu")],
    )


def _validate_copy_expert_rows_gpu_inputs(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    count: torch.Tensor,
) -> None:
    if source.device.type not in ("cpu", "cuda"):
        raise ValueError("source must be a pinned CPU or CUDA tensor.")
    if source.device.type == "cpu" and not source.is_pinned():
        raise ValueError("source must use pinned or CUDA-registered CPU storage.")
    if destination.device.type != "cuda":
        raise ValueError("destination must be a CUDA tensor.")
    if source.device.type == "cuda" and source.device != destination.device:
        raise ValueError("source and destination must share one CUDA device.")
    if any(
        tensor.device.type != "cuda"
        for tensor in (source_rows, destination_slots, count)
    ):
        raise ValueError("destination and plan tensors must be CUDA tensors.")
    if any(
        tensor.device != destination.device
        for tensor in (source_rows, destination_slots, count)
    ):
        raise ValueError("destination and plan tensors must share one CUDA device.")
    if source.ndim < 1 or destination.ndim != source.ndim:
        raise ValueError("source and destination must have matching row dimensions.")
    if source.shape[1:] != destination.shape[1:]:
        raise ValueError("source and destination must have matching row width.")
    if source.dtype != destination.dtype:
        raise ValueError("source and destination must have matching dtype.")
    if not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError("source and destination must be contiguous.")
    if source_rows.dtype != torch.int64 or destination_slots.dtype != torch.int32:
        raise ValueError(
            "source_rows must be int64 and destination_slots must be int32."
        )
    if count.dtype != torch.int32:
        raise ValueError("count must be int32.")
    if source_rows.ndim != 1 or destination_slots.ndim != 1:
        raise ValueError("source_rows and destination_slots must be one-dimensional.")
    if source_rows.numel() != destination_slots.numel():
        raise ValueError(
            "source_rows and destination_slots must have matching capacity."
        )
    if count.numel() != 1:
        raise ValueError("count must contain exactly one int32 value.")
    if not all(
        tensor.is_contiguous() for tensor in (source_rows, destination_slots, count)
    ):
        raise ValueError("plan tensors must be contiguous.")


def copy_expert_rows_gpu(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Copy device-plan-selected pinned-host or CUDA rows into CUDA cache slots.

    ``count`` is only loaded by the kernel, so callers can update its fixed CUDA
    allocation between CUDA-graph replays without a host synchronization.
    """
    _validate_copy_expert_rows_gpu_inputs(
        source, destination, source_rows, destination_slots, count
    )
    _jit_expert_cache_transfer_module().copy_expert_rows_gpu(
        source, destination, source_rows, destination_slots, count
    )
