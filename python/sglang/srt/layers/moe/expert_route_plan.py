"""Route planning for streamed expert gathers: deduplication and graph remaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

NO_DEDUP_LIMIT = 64
FUSED_MAX_ROUTES = 32


def should_dedup(topk_ids: torch.Tensor) -> bool:
    """Whether an eager gather should read one source row per distinct expert.

    One token's top-k IDs are distinct, so single-token decode gathers its
    routes as they are. A forward with more than one token row (prefill,
    speculative verify) shares one row between duplicate routes.
    """
    return topk_ids.numel() > NO_DEDUP_LIMIT or (
        topk_ids.ndim > 1 and topk_ids.shape[0] > 1
    )


@dataclass(frozen=True)
class GraphRoutePlan:
    """Device tensors describing one sync-free graph gather.

    ``remap`` sends each route to its expert's hot slot or scratch row.
    ``source_rows`` holds expert IDs for the first ``min(routes, scratch_rows)``
    plan rows, distinct misses first in order of first appearance, so the pull
    kernel's first ``miss_plan_rows`` rows are exactly the distinct misses.
    ``unique_hit_rows`` and ``unique_miss_rows`` count distinct experts;
    ``routed_miss_rows`` counts missed routes with multiplicity.
    """

    remap: torch.Tensor
    source_rows: torch.Tensor
    miss_plan_rows: torch.Tensor
    unique_hit_rows: torch.Tensor
    unique_miss_rows: torch.Tensor
    routed_miss_rows: torch.Tensor


def plan_graph_routes(
    flat: torch.Tensor,
    expert_to_slot: torch.Tensor,
    scratch_rows: int,
    scratch_base: int,
) -> GraphRoutePlan:
    """Plan a gather that gives each distinct missed expert one scratch row.

    ``flat`` holds int64 expert IDs, ``expert_to_slot`` maps experts to hot
    slots (-1 for nonresident), and scratch rows start at ``scratch_base``.
    The caller guarantees ``flat.numel() <= scratch_rows``, so the distinct
    misses always fit; a scratch smaller than the routes would need an
    overflow flag and an eager re-verify, which the graph verify phase of the
    NEXTN offload plan adds. Only fixed-shape device operations run and no
    device value is read on the host, so a CUDA graph can capture the plan
    and replay it for any routes.
    """
    count = flat.numel()
    positions = torch.arange(count, dtype=torch.long, device=flat.device)
    slots = expert_to_slot.index_select(0, flat)
    hit = slots >= 0
    sorted_ids, order = torch.sort(flat, stable=True)
    group_starts = torch.ones_like(hit)
    group_starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
    start_index = torch.cummax(torch.where(group_starts, positions, 0), dim=0).values
    first_position = torch.empty_like(order)
    first_position.scatter_(0, order, order.index_select(0, start_index))
    first_route = first_position == positions
    new_miss = first_route & ~hit
    rank = (torch.cumsum(new_miss, dim=0) - 1).index_select(0, first_position)
    unique_misses = new_miss.sum()
    plan_order = torch.argsort((~new_miss).to(torch.uint8), stable=True)
    return GraphRoutePlan(
        remap=torch.where(hit, slots, rank + scratch_base),
        source_rows=flat.index_select(0, plan_order[: min(count, scratch_rows)]),
        miss_plan_rows=unique_misses.to(torch.int32),
        unique_hit_rows=(first_route & hit).sum(),
        unique_miss_rows=unique_misses,
        routed_miss_rows=count - hit.sum(),
    )


def supports_fused_graph_routes(
    flat: torch.Tensor, expert_to_slot: torch.Tensor, scratch_rows: int
) -> bool:
    """Whether `plan_graph_routes_fused` may serve this gather.

    The fused kernel is BS1 and unique-ID only: one warp, one route per lane,
    no duplicate-ID handling. A caller whose routes may repeat (prefill,
    speculative verify batches) must keep using `plan_graph_routes`.
    """
    return (
        flat.is_cuda
        and flat.ndim == 1
        and 0 < flat.numel() <= FUSED_MAX_ROUTES
        and flat.numel() <= scratch_rows
        and expert_to_slot.dtype == torch.int64
        and expert_to_slot.device == flat.device
    )


_ZERO_PREFETCH_STATE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _zero_prefetch_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A permanent zero-count prefetch state: no route is ever a covered miss."""
    state = _ZERO_PREFETCH_STATE.get(device)
    if state is None:
        state = (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _ZERO_PREFETCH_STATE[device] = state
    return state


def plan_graph_routes_fused(
    flat: torch.Tensor,
    expert_to_slot: torch.Tensor,
    scratch_base: int,
    remap_dtype: torch.dtype,
    source_rows_out: torch.Tensor,
    slots_out: torch.Tensor,
    count_out: torch.Tensor,
    graph_counters: Optional[torch.Tensor] = None,
    graph_unique_counters: Optional[torch.Tensor] = None,
    route_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`plan_graph_routes`'s BS1, unique-ID fast path: one fused kernel launch.

    Reproduces its exact semantics for unique-ID ``flat``. Prediction inputs
    are wired into the call but permanently disabled: prefetch count is
    always zero, so every nonresident route is a residual scratch row, never
    a covered miss. ``source_rows_out`` and ``slots_out`` are written for
    every one of ``flat``'s routes (residual routes first), matching
    ``GraphRoutePlan.source_rows``' full-vector contract; ``count_out``
    receives the residual row count. Writes ``graph_counters``,
    ``graph_unique_counters`` and ``route_counts`` in place when given,
    instead of returning fresh per-call tensors for them.

    Returns the per-route remap in ``remap_dtype``, shaped like ``flat``.
    """
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    prefetch_expert, prefetch_count = _zero_prefetch_state(flat.device)
    remap_out = torch.empty(flat.shape, dtype=remap_dtype, device=flat.device)
    plan_unique_routes_cuda(
        flat,
        expert_to_slot,
        scratch_base,
        source_rows_out,
        slots_out,
        count_out,
        remap_out,
        graph_counters,
        graph_unique_counters,
        route_counts,
        prefetch_expert,
        prefetch_count,
        0,
    )
    return remap_out
