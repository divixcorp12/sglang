"""JIT wrapper for the fused BS1, top_k<=32 demand route planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_ROUTES = 32
_ROUTING_DTYPES = (torch.int32, torch.int64)


@cache_once
def _jit_expert_route_plan_module(topk_dtype: torch.dtype, remap_dtype: torch.dtype) -> Module:
    args = make_cpp_args(topk_dtype, remap_dtype)
    return load_jit(
        "expert_route_plan",
        *args,
        cuda_files=["moe/expert_route_plan.cuh"],
        cuda_wrappers=[("plan_unique_routes_gpu", f"plan_unique_routes_gpu<{args}>")],
    )


def _validate_device_tensor(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must share the planner's CUDA device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_scalar(name: str, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> None:
    _validate_device_tensor(name, tensor, device)
    if tensor.dtype != dtype or tensor.numel() != 1:
        raise ValueError(f"{name} must hold exactly one {dtype} value.")


def _validate_counters(name: str, tensor: Optional[torch.Tensor], device: torch.device) -> None:
    if tensor is None:
        return
    _validate_device_tensor(name, tensor, device)
    if tensor.dtype != torch.int64 or tensor.numel() != 2:
        raise ValueError(f"{name} must be an int64 CUDA tensor of two elements.")


def _validate_route_plan_inputs(
    topk_ids: torch.Tensor,
    expert_to_slot: torch.Tensor,
    source_rows_out: torch.Tensor,
    slots_out: torch.Tensor,
    count_out: torch.Tensor,
    remap_out: torch.Tensor,
    graph_counters: Optional[torch.Tensor],
    graph_unique_counters: Optional[torch.Tensor],
    route_counts: Optional[torch.Tensor],
    prefetch_expert: torch.Tensor,
    prefetch_count: torch.Tensor,
) -> None:
    if topk_ids.device.type != "cuda":
        raise ValueError("topk_ids must be a CUDA tensor.")
    device = topk_ids.device
    if topk_ids.dtype not in _ROUTING_DTYPES:
        raise ValueError("topk_ids must be int32 or int64.")
    if topk_ids.ndim != 1:
        raise ValueError("topk_ids must be one-dimensional.")
    if not 0 < topk_ids.numel() <= MAX_ROUTES:
        raise ValueError(f"topk_ids must hold 1-{MAX_ROUTES} routes.")
    _validate_device_tensor("topk_ids", topk_ids, device)
    if expert_to_slot.dtype != torch.int64 or expert_to_slot.ndim != 1:
        raise ValueError("expert_to_slot must be int64 [num_experts].")
    _validate_device_tensor("expert_to_slot", expert_to_slot, device)
    if source_rows_out.dtype != torch.int64 or source_rows_out.shape != topk_ids.shape:
        raise ValueError("source_rows_out must be int64, matching topk_ids' shape.")
    _validate_device_tensor("source_rows_out", source_rows_out, device)
    if slots_out.dtype != torch.int32 or slots_out.shape != topk_ids.shape:
        raise ValueError("slots_out must be int32, matching topk_ids' shape.")
    _validate_device_tensor("slots_out", slots_out, device)
    if remap_out.dtype not in _ROUTING_DTYPES or remap_out.shape != topk_ids.shape:
        raise ValueError("remap_out must be int32 or int64, matching topk_ids' shape.")
    _validate_device_tensor("remap_out", remap_out, device)
    _validate_scalar("count_out", count_out, device, torch.int32)
    _validate_scalar("prefetch_expert", prefetch_expert, device, torch.int64)
    _validate_scalar("prefetch_count", prefetch_count, device, torch.int32)
    _validate_counters("graph_counters", graph_counters, device)
    _validate_counters("graph_unique_counters", graph_unique_counters, device)
    if route_counts is not None:
        _validate_device_tensor("route_counts", route_counts, device)
        if route_counts.dtype != torch.float32:
            raise ValueError("route_counts must be float32.")
        if route_counts.numel() != expert_to_slot.numel():
            raise ValueError("route_counts must have one entry per expert.")


def plan_unique_routes_cuda(
    topk_ids: torch.Tensor,
    expert_to_slot: torch.Tensor,
    scratch_base: int,
    source_rows_out: torch.Tensor,
    slots_out: torch.Tensor,
    count_out: torch.Tensor,
    remap_out: torch.Tensor,
    graph_counters: Optional[torch.Tensor],
    graph_unique_counters: Optional[torch.Tensor],
    route_counts: Optional[torch.Tensor],
    prefetch_expert: torch.Tensor,
    prefetch_count: torch.Tensor,
    prefetch_slot: int,
) -> None:
    """Write a BS1, K<=32 plan into supplied stable CUDA buffers.

    route_counts is an optional per-expert float32 counter tensor.
    prefetch_expert is int64[1], prefetch_count is int32[1].
    Zero prefetch_count disables coverage; prefetch_slot is a fixed integer.
    Caller has joined prefetch completion before invoking this function.
    """
    _validate_route_plan_inputs(
        topk_ids,
        expert_to_slot,
        source_rows_out,
        slots_out,
        count_out,
        remap_out,
        graph_counters,
        graph_unique_counters,
        route_counts,
        prefetch_expert,
        prefetch_count,
    )
    module = _jit_expert_route_plan_module(topk_ids.dtype, remap_out.dtype)
    module.plan_unique_routes_gpu(
        topk_ids,
        expert_to_slot,
        int(scratch_base),
        source_rows_out,
        slots_out,
        count_out,
        remap_out,
        graph_counters,
        graph_unique_counters,
        route_counts,
        prefetch_expert,
        prefetch_count,
        int(prefetch_slot),
    )
