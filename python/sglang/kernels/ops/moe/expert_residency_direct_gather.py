"""JIT wrappers for the DIRECT residency kernels of ``GpuResidencyUpdater`` (SGLANG_DSV41_ENABLE_LAYER_FUSION).

Each wrapper launches one kernel in place of a per-layer chain of small torch ops, with bit-identical results:

* ``direct_gather_destinations`` -- ``GpuResidencyUpdater.gather_destinations`` (26 kernels per layer), with the
  route-slot lookup ``expert_to_slot.index_select(0, flat.long())`` folded in;
* ``direct_commit_gather`` -- ``GpuResidencyUpdater.commit_gather`` (41 kernels per layer).

The torch chains stay the reference: the flag-off path runs them, and the parity tests compare against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)


@cache_once
def _gather_module(
    id_dtype: torch.dtype, remap_in: torch.dtype, remap_out: torch.dtype
) -> Module:
    args = make_cpp_args(id_dtype, remap_in, remap_out)
    return load_jit(
        "expert_residency_direct_gather_destinations",
        *args,
        cuda_files=["moe/expert_residency/direct_gather.cuh"],
        cuda_wrappers=[
            ("run", f"expert_residency::direct_gather_destinations_gpu<{args}>")
        ],
    )


@cache_once
def _commit_module() -> Module:
    return load_jit(
        "expert_residency_direct_commit_gather",
        cuda_files=["moe/expert_residency/direct_gather.cuh"],
        cuda_wrappers=[("run", "expert_residency::direct_commit_gather_gpu")],
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
    idle_destination: int = 0,
    idle_slot: int = 0,
) -> None:
    """One layer's DIRECT gather destinations; see ``GpuResidencyUpdater.gather_destinations``.

    A lane that is not live gets ``idle_destination`` and ``idle_slot``; nonzero values always run the wide kernel.

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
        int(idle_destination),
        int(idle_slot),
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
    cpu_lanes: Optional[torch.Tensor] = None,
) -> None:
    """One layer's DIRECT residency commit; see ``GpuResidencyUpdater.commit_gather``.

    ``mapping`` is the layer's ``[experts + 1]`` row (last column the dump), ``slot_*`` its ``[slots + 1]`` rows (last
    column the dump). ``delivered`` and ``keep`` are the leased backend's delivered count and keep flag, both or
    neither; without them the truncation tripwire compares ``miss_count``. ``cpu_lanes`` (int32 ``[2]`` (``[3]`` on a wide wire): CPU lanes, then the part bits, then the lanes' high half; CPU experts) is
    the mask of lanes the CPU computed: never inserted, and not counted as truncated.
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
        cpu_lanes,
        int(ready),
        int(free_state),
    )
