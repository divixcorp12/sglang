"""JIT wrappers for the DSV4.1 EXL3 decode layer fusions (SGLANG_DSV41_ENABLE_LAYER_FUSION).

Each wrapper launches one kernel in place of a per-layer chain of small torch ops, with bit-identical results:

* ``direct_gather_destinations`` -- ``GpuResidencyUpdater.gather_destinations`` (26 kernels per layer), with the
  route-slot lookup ``expert_to_slot.index_select(0, flat.long())`` folded in;
* ``direct_commit_gather`` -- ``GpuResidencyUpdater.commit_gather`` (41 kernels per layer);
* ``exl3_moe_route_tables`` -- ``exl3_fused_moe.route_tables`` and the copies around it in ``Exl3FusedMoE.run``
  (22 kernels per layer).

The torch chains stay the reference: the flag-off path runs them, and the parity tests compare against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@cache_once
def _gather_module(
    id_dtype: torch.dtype, remap_in: torch.dtype, remap_out: torch.dtype
) -> Module:
    args = make_cpp_args(id_dtype, remap_in, remap_out)
    return load_jit(
        "dsv41_direct_gather_destinations",
        *args,
        cuda_files=["moe/dsv41_layer_fusion.cuh"],
        cuda_wrappers=[("run", f"direct_gather_destinations_gpu<{args}>")],
    )


@cache_once
def _commit_module() -> Module:
    return load_jit(
        "dsv41_direct_commit_gather",
        cuda_files=["moe/dsv41_layer_fusion.cuh"],
        cuda_wrappers=[("run", "direct_commit_gather_gpu")],
    )


@cache_once
def _route_tables_module(
    remap: torch.dtype, weight: torch.dtype, x: torch.dtype
) -> Module:
    args = make_cpp_args(remap, weight, x)
    return load_jit(
        "dsv41_exl3_moe_route_tables",
        *args,
        cuda_files=["moe/dsv41_layer_fusion.cuh"],
        cuda_wrappers=[("run", f"exl3_moe_route_tables_gpu<{args}>")],
    )


def direct_gather_destinations(
    topk_ids: torch.Tensor,
    expert_to_slot: torch.Tensor,
    victims: torch.Tensor,
    victim_valid: torch.Tensor,
    miss_count: torch.Tensor,
    remap: torch.Tensor,
    scratch_base: int,
    destination_slots_out: torch.Tensor,
    destinations_out: torch.Tensor,
    live_out: torch.Tensor,
    remap_out: torch.Tensor,
) -> None:
    """One layer's DIRECT gather destinations; see ``GpuResidencyUpdater.gather_destinations``.

    ``topk_ids`` and ``remap`` are this forward's flat routes and the planner's remap (int32 or int64);
    ``remap_out`` may be either dtype. ``victims``/``victim_valid`` are the layer's shortlist row.

    The launcher checks every tensor; this refuses only a dtype that has no instantiation.
    """
    for name, tensor in (
        ("topk_ids", topk_ids),
        ("remap", remap),
        ("remap_out", remap_out),
    ):
        if tensor.dtype not in _ID_DTYPES:
            raise ValueError(f"{name} must be {_ID_DTYPES}, got {tensor.dtype}")
    _gather_module(topk_ids.dtype, remap.dtype, remap_out.dtype).run(
        topk_ids,
        expert_to_slot,
        victims,
        victim_valid,
        miss_count,
        remap,
        int(scratch_base),
        destination_slots_out,
        destinations_out,
        live_out,
        remap_out,
    )


def direct_commit_gather(
    destinations: torch.Tensor,
    live: torch.Tensor,
    new_experts: torch.Tensor,
    mapping: torch.Tensor,
    slot_to_expert: torch.Tensor,
    slot_state: torch.Tensor,
    slot_generations: torch.Tensor,
    gather_insertions: torch.Tensor,
    gather_evictions: torch.Tensor,
    insertion_truncated: torch.Tensor,
    delivered: Optional[torch.Tensor],
    keep: Optional[torch.Tensor],
    miss_count: torch.Tensor,
    *,
    ready: int,
    free_state: int,
) -> None:
    """One layer's DIRECT residency commit; see ``GpuResidencyUpdater.commit_gather``.

    ``mapping`` is the layer's ``[experts + 1]`` row (last column the dump), ``slot_*`` its ``[slots + 1]`` rows (last
    column the dump). ``delivered`` and ``keep`` are the leased backend's delivered count and keep flag, both or
    neither; without them the truncation tripwire compares ``miss_count``.
    """
    _commit_module().run(
        destinations,
        live,
        new_experts,
        mapping.numel() - 1,
        slot_to_expert.numel() - 1,
        mapping,
        slot_to_expert,
        slot_state,
        slot_generations,
        gather_insertions,
        gather_evictions,
        insertion_truncated,
        delivered,
        keep,
        miss_count,
        int(ready),
        int(free_state),
    )


def exl3_moe_route_tables(
    remap: torch.Tensor,
    weights: torch.Tensor,
    keep: torch.Tensor,
    x: torch.Tensor,
    remap64_out: torch.Tensor,
    x16_out: torch.Tensor,
    out_zero: torch.Tensor,
    expert_count: torch.Tensor,
    inv_order: torch.Tensor,
    weight_sorted: torch.Tensor,
    det: torch.Tensor,
) -> None:
    """The fused MoE's route tables and input staging; see ``exl3_fused_moe.route_tables``.

    Writes ``remap64_out`` (``remap`` as int64), ``x16_out`` (``x`` as fp16), zeroes ``out_zero``, and fills
    ``expert_count`` [slots + 1], ``inv_order``, ``weight_sorted`` (fp16) and ``det`` [3, slots + 1].

    The launcher checks every tensor; this refuses only a dtype that has no instantiation.
    """
    for name, tensor, dtypes in (
        ("remap", remap, _ID_DTYPES),
        ("weights", weights, _FLOAT_DTYPES),
        ("x", x, _FLOAT_DTYPES),
    ):
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} must be {dtypes}, got {tensor.dtype}")
    _route_tables_module(remap.dtype, weights.dtype, x.dtype).run(
        remap,
        weights,
        keep,
        x,
        remap64_out,
        x16_out,
        out_zero,
        expert_count,
        inv_order,
        weight_sorted,
        det,
    )
