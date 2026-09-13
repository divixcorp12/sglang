"""CUDA-graph-safe pull transfers from pinned expert rows into an HBM cache."""

from __future__ import annotations

import functools
from collections.abc import Sequence
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
        cuda_wrappers=[
            ("copy_expert_rows_gpu", "copy_expert_rows_gpu"),
            ("copy_expert_row_segments_gpu", "copy_expert_row_segments_gpu"),
        ],
    )


def _validate_row_pair(source: torch.Tensor, destination: torch.Tensor) -> None:
    from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor

    if source.device.type not in ("cpu", "cuda"):
        raise ValueError("source must be a pinned CPU or CUDA tensor.")
    if source.device.type == "cpu" and not is_gpu_readable_host_tensor(source):
        raise ValueError("source must use pinned or CUDA-registered CPU storage.")
    if destination.device.type != "cuda":
        raise ValueError("destination must be a CUDA tensor.")
    if source.device.type == "cuda" and source.device != destination.device:
        raise ValueError("source and destination must share one CUDA device.")
    if source.ndim < 1 or destination.ndim != source.ndim:
        raise ValueError("source and destination must have matching row dimensions.")
    if source.shape[1:] != destination.shape[1:]:
        raise ValueError("source and destination must have matching row width.")
    if source.dtype != destination.dtype:
        raise ValueError("source and destination must have matching dtype.")
    if not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError("source and destination must be contiguous.")


def _validate_plan(
    device: torch.device,
    source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    count: torch.Tensor,
) -> None:
    if any(
        tensor.device.type != "cuda"
        for tensor in (source_rows, destination_slots, count)
    ):
        raise ValueError("destination and plan tensors must be CUDA tensors.")
    if any(
        tensor.device != device for tensor in (source_rows, destination_slots, count)
    ):
        raise ValueError("destination and plan tensors must share one CUDA device.")
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
    _validate_row_pair(source, destination)
    _validate_plan(destination.device, source_rows, destination_slots, count)
    _jit_expert_cache_transfer_module().copy_expert_rows_gpu(
        source, destination, source_rows, destination_slots, count
    )


def expert_row_segments(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Pack ``(source, destination)`` addresses and row widths into a CUDA plan.

    The returned ``[pairs, 3]`` int64 tensor stores raw data pointers, so every
    source and destination must stay allocated at its current address for as
    long as the plan is used.
    """
    if not pairs:
        raise ValueError("expert row segments need at least one tensor pair.")
    for source, destination in pairs:
        _validate_row_pair(source, destination)
    devices = {destination.device for _, destination in pairs}
    if len(devices) != 1:
        raise ValueError("expert row segment destinations must share one CUDA device.")
    return torch.tensor(
        [
            [
                source.data_ptr(),
                destination.data_ptr(),
                source.stride(0) * source.element_size(),
            ]
            for source, destination in pairs
        ],
        dtype=torch.int64,
        device=devices.pop(),
    )


def copy_expert_row_segments_gpu(
    segments: torch.Tensor,
    source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Copy the planned rows of every tensor pair in ``segments`` in one launch.

    ``segments`` comes from :func:`expert_row_segments`. Each launched thread
    copies its lane of the selected row in every segment, so a layer's tensors
    cost one kernel launch instead of one per tensor.
    """
    if segments.device.type != "cuda" or segments.dtype != torch.int64:
        raise ValueError("segments must be a CUDA int64 tensor.")
    if segments.ndim != 2 or segments.shape[0] < 1 or segments.shape[1] != 3:
        raise ValueError("segments must have shape [pairs, 3].")
    if not segments.is_contiguous():
        raise ValueError("segments must be contiguous.")
    _validate_plan(segments.device, source_rows, destination_slots, count)
    _jit_expert_cache_transfer_module().copy_expert_row_segments_gpu(
        segments, source_rows, destination_slots, count
    )
