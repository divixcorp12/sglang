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
    prefetch_expert: Optional[torch.Tensor] = None,
    prefetch_slot: int = -1,
    prefetch_count: Optional[torch.Tensor] = None,
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

    ``prefetch_expert`` is an optional int64 scalar (or one-element) tensor
    holding a side-stream pull's predicted expert id for this forward.
    ``prefetch_count`` is its real int32 posted-row count when supplied;
    only exactly one posted row enables coverage. Omitting the count retains
    the historical compatibility behavior: a supplied expert enables the
    check. Omitting the expert, or using a negative id, disables the check
    (no real topk id is ever negative). Every covered nonresident route --
    every duplicate of it, too -- is excluded from the demand-scratch plan
    and remapped straight to ``prefetch_slot`` instead of a scratch row,
    mirroring ``plan_unique_routes_kernel``'s ``prefetched`` lane. This never
    changes ``unique_miss_rows``/``routed_miss_rows``, which stay logical
    counts of every nonresident route whether or not prefetch covers it;
    only ``miss_plan_rows`` and the plan rows a copy backend actually reads
    for this forward shrink.
    """
    count = flat.numel()
    positions = torch.arange(count, dtype=torch.long, device=flat.device)
    slots = expert_to_slot.index_select(0, flat)
    hit = slots >= 0
    if prefetch_expert is None:
        prefetched = torch.zeros_like(hit)
    elif prefetch_count is None:
        prefetched = ~hit & (flat == prefetch_expert.reshape(()))
    else:
        posted = prefetch_count.reshape(()) == 1
        prefetched = ~hit & posted & (flat == prefetch_expert.reshape(()))
    residual = ~hit & ~prefetched
    sorted_ids, order = torch.sort(flat, stable=True)
    group_starts = torch.ones_like(hit)
    group_starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
    start_index = torch.cummax(torch.where(group_starts, positions, 0), dim=0).values
    first_position = torch.empty_like(order)
    first_position.scatter_(0, order, order.index_select(0, start_index))
    first_route = first_position == positions
    new_miss = first_route & ~hit
    first_residual = first_route & residual
    rank = (torch.cumsum(first_residual, dim=0) - 1).index_select(0, first_position)
    unique_misses = new_miss.sum()
    miss_plan_rows = first_residual.sum()
    plan_order = torch.argsort((~first_residual).to(torch.uint8), stable=True)
    remap = torch.where(
        hit,
        slots,
        torch.where(prefetched, torch.full_like(slots, prefetch_slot), rank + scratch_base),
    )
    return GraphRoutePlan(
        remap=remap,
        source_rows=flat.index_select(0, plan_order[: min(count, scratch_rows)]),
        miss_plan_rows=miss_plan_rows.to(torch.int32),
        unique_hit_rows=(first_route & hit).sum(),
        unique_miss_rows=unique_misses,
        routed_miss_rows=count - hit.sum(),
    )


def supports_fused_graph_routes(
    topk_ids: torch.Tensor, expert_to_slot: torch.Tensor, scratch_rows: int
) -> bool:
    """Whether `plan_graph_routes_fused` may serve this gather.

    The fused kernel is BS1 and unique-ID only: one warp, one route per lane,
    no duplicate-ID handling. ``graph_gather_rows`` is sized ``tokens *
    top_k``, so a multi-token call can pass the route-count and scratch
    checks below while carrying several requests' routes, which may repeat
    an expert across tokens (``plan_graph_routes`` dedups that; this kernel
    does not). Requiring exactly one token row (``topk_ids.shape[0] == 1``)
    is what actually guarantees unique IDs here.

    Within one token row, uniqueness holds structurally rather than by
    construction of this predicate: `torch.topk` returns distinct indices,
    and the logical-to-physical expert remap applied before this gather
    (`topk_ids_logical_to_physical`, `eplb/expert_location_dispatch.py`) maps
    distinct logical experts to distinct physical replicas, because the
    physical-to-logical direction (`phy2log`, `eplb/lplb_solver.py`) is a
    function -- every physical expert belongs to exactly one logical expert,
    so two different logical experts can never land on the same physical ID.
    A within-row duplicate therefore cannot arise from real routing; this
    function does not itself check for one.
    """
    return (
        topk_ids.is_cuda
        and topk_ids.ndim >= 1
        and topk_ids.shape[0] == 1
        and 0 < topk_ids.numel() <= FUSED_MAX_ROUTES
        and topk_ids.numel() <= scratch_rows
        and expert_to_slot.dtype == torch.int64
        and expert_to_slot.device == topk_ids.device
    )


_ZERO_PREFETCH_STATE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_ONE_PREFETCH_COUNT: dict[torch.device, torch.Tensor] = {}


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


def _one_prefetch_count(device: torch.device) -> torch.Tensor:
    """A permanent count of one: enables the ``prefetch_expert`` check every call."""
    state = _ONE_PREFETCH_COUNT.get(device)
    if state is None:
        state = torch.ones(1, dtype=torch.int32, device=device)
        _ONE_PREFETCH_COUNT[device] = state
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
    prefetch_expert: Optional[torch.Tensor] = None,
    prefetch_slot: int = -1,
    prefetch_count: Optional[torch.Tensor] = None,
    outcome_counters: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`plan_graph_routes`'s BS1, unique-ID fast path: one fused kernel launch.

    Reproduces its exact semantics for unique-ID ``flat``, prefetch coverage
    included. ``prefetch_expert`` is an optional int64 CUDA scalar (or
    one-element) tensor holding a side-stream pull's predicted expert id for
    this forward. ``prefetch_count`` can supply the real persistent posted
    count; only a value of one enables coverage. Its omission preserves
    compatibility for existing callers by using a permanent count of one
    when an expert is supplied. Without an expert, a permanent zero count
    disables coverage. ``source_rows_out`` and
    ``slots_out`` are written for every one of ``flat``'s routes (residual
    routes first), matching ``GraphRoutePlan.source_rows``' full-vector
    contract; ``count_out`` receives the residual row count, which shrinks
    by one exactly when a route is prefetch-covered. Writes
    ``graph_counters``, ``graph_unique_counters`` and ``route_counts`` in
    place when given, instead of returning fresh per-call tensors for them;
    these stay logical (prefetch-covered routes still count as misses there),
    matching ``plan_graph_routes``. ``outcome_counters`` optionally holds a
    persistent int64[4] accumulator ordered [covered, residual, wasted,
    posted]. The first two are logical routes; the last two are physical
    speculative rows.

    Returns the per-route remap in ``remap_dtype``, shaped like ``flat``.
    """
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    if prefetch_expert is None:
        prefetch_expert, prefetch_count = _zero_prefetch_state(flat.device)
        prefetch_slot = 0
    elif prefetch_count is None:
        prefetch_count = _one_prefetch_count(flat.device)
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
        prefetch_slot,
        outcome_counters,
    )
    return remap_out
