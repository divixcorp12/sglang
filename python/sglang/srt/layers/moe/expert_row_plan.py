"""Static-capacity expert-row copy plans, their planner, and the backends that serve them.

Interface contract, shared by the current-layer miss path and a later
next-layer (L+1) prediction mode:

* **Plan** (``ExpertRowPlan``): per target layer, device tensors of a fixed
  capacity ``C``: ``expert_ids`` int64 ``[C]`` (host source rows, one per
  expert), ``slots`` int32 ``[C]`` (destination rows in that target layer's
  hot-cache tensors) and ``count`` int32 ``[1]`` (rows in use, a prefix). Any
  producer may fill a plan; nothing derives it from the current layer's
  routing inside a copy kernel.
* **Planner** (``ExpertRowPlanner``): ``plan_routes(flat, plan)`` is the
  router-miss producer (today's distinct misses in first-appearance order, the
  order the route remap expects). ``plan_candidates(candidates, plan,
  priority=None)`` takes int64 expert ids ``[N]`` (``-1`` pads) or a bool mask
  ``[num_experts]``, with an optional float priority aligned to them; it drops
  experts already resident in the hot cache, dedupes, and clamps to
  ``min(C, scratch_rows)`` in priority order (eligible experts always rank
  above ineligible ones; without a priority earlier id positions come first,
  and equal priorities break by lower expert id). Both run fixed-shape device ops only, so a CUDA
  graph can capture them.
* **Destinations**: a planner addresses its own target layer's scratch rows,
  ``scratch_base + r`` for plan row ``r``. Scratch holds ``bs x top_k`` rows
  per layer (``graph_gather_scratch_rows``) unless extended; this build does
  not change that. An L+1 plan of 32-64 candidates therefore clamps to scratch
  capacity by priority; targeting evictable hot slots instead would need the
  reservation rule below and is not built.
* **Tags**: one outstanding request per target-layer tag. ``post(tag, plan)``
  and ``resolve(tag, plan)`` are separate in-graph steps; resolve matches that
  tag's own latest post, so it never reports another layer's delivery. The
  copier's ring holds at least two outstanding requests.
* **Delivery** (``ExpertRowDelivery``): ``resolve`` returns a device-side
  result; ``mask()`` is bool ``[C]`` of plan rows valid in their slots.
  Residual rows (planned or actually needed but not delivered) go through the
  in-graph copy kernel: ``copy_residual`` for the plan itself, and
  ``plan_residual_routes`` for actual routes minus a delivered (possibly
  mispredicted) plan.
* **Reservation rule**: a slot named by an outstanding plan must not be read
  before its resolve returns, nor written by anyone else until then, and plan
  tensors must not change between a post and its resolve. A post for layer
  L+1 writes only layer L+1's rows, never rows layer L reads.
* **Backends**: ``in_graph`` (the existing ``copy_expert_row_segments_gpu``
  serves the plan at ``post``; resolve reports everything delivered) and
  ``doorbell`` (the copier thread serves it; resolve reports the thread's
  delivered flag, and timeouts or a disabled copier leave rows to the
  residual). ``SGLANG_MOE_EXPERT_DOORBELL`` selects ``doorbell``;
  ``SGLANG_MOE_EXPERT_DOORBELL_MODE`` is ``current`` (post(L) then resolve(L)
  inside layer L); ``next_layer`` is reserved and not implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import (
    ExpertRowSegments,
    copy_expert_row_segments_gpu,
)
from sglang.srt.layers.moe.expert_route_plan import GraphRoutePlan, plan_graph_routes

BACKEND_IN_GRAPH = "in_graph"
BACKEND_DOORBELL = "doorbell"
MODE_CURRENT = "current"
MODE_NEXT_LAYER = "next_layer"


@dataclass(frozen=True)
class ExpertRowPlan:
    """One target layer's device copy plan of static capacity.

    ``expert_ids`` int64 ``[C]`` names host source rows, ``slots`` int32 ``[C]``
    the destination rows in the target layer's hot-cache tensors, and ``count``
    int32 ``[1]`` how many leading rows are in use.
    """

    expert_ids: torch.Tensor
    slots: torch.Tensor
    count: torch.Tensor

    def __post_init__(self) -> None:
        if self.expert_ids.dtype != torch.int64 or self.expert_ids.ndim != 1:
            raise ValueError("plan expert_ids must be int64 [C].")
        if self.slots.dtype != torch.int32 or self.slots.shape != self.expert_ids.shape:
            raise ValueError("plan slots must be int32 [C].")
        if self.count.dtype != torch.int32 or self.count.numel() != 1:
            raise ValueError("plan count must be int32 [1].")

    @property
    def capacity(self) -> int:
        return self.expert_ids.numel()

    @classmethod
    def for_scratch(
        cls, capacity: int, scratch_base: int, scratch_rows: int, device: torch.device
    ) -> ExpertRowPlan:
        """A plan whose row ``r`` targets scratch row ``scratch_base + r``.

        Rows at or past ``scratch_rows`` keep the first scratch row as their slot;
        the planners never count them.
        """
        rows = torch.arange(capacity, dtype=torch.int32, device=device)
        rows = torch.where(rows < scratch_rows, rows, 0)
        return cls(
            expert_ids=torch.zeros(capacity, dtype=torch.int64, device=device),
            slots=rows + scratch_base,
            count=torch.zeros(1, dtype=torch.int32, device=device),
        )


def live_slot_map(slot_source) -> Callable[[], torch.Tensor]:
    """A zero-argument lookup of a target layer's live int64 expert-to-slot map.

    ``slot_source`` is a hot cache (any object with an ``expert_to_slot``
    attribute, read on every lookup) or a callable returning the map. A bare
    map tensor is refused: the cache's owner may rebind the attribute to a
    different tensor (the GPU residency update does), and a held tensor would
    silently route against the stale map.
    """
    if isinstance(slot_source, torch.Tensor):
        raise TypeError(
            "pass the hot cache (or a callable returning its expert_to_slot), not an "
            "expert_to_slot tensor: the GPU residency update rebinds the cache's map"
        )
    if hasattr(slot_source, "expert_to_slot"):
        return lambda: slot_source.expert_to_slot
    if callable(slot_source):
        return slot_source
    raise TypeError(
        "slot_source must be a hot cache or a callable returning expert_to_slot."
    )


class ExpertRowPlanner:
    """Turns routes or candidate experts into one target layer's scratch plan."""

    def __init__(self, slot_source, scratch_base: int, scratch_rows: int) -> None:
        """``slot_source`` is the target layer's hot cache, or a callable returning its live
        expert-to-slot map; the map is looked up on every call (see ``live_slot_map``)."""
        self._lookup = live_slot_map(slot_source)
        self.scratch_base = scratch_base
        self.scratch_rows = scratch_rows

    @property
    def expert_to_slot(self) -> torch.Tensor:
        return self._lookup()

    @property
    def num_experts(self) -> int:
        return self.expert_to_slot.numel()

    def route_plan(self, flat: torch.Tensor) -> GraphRoutePlan:
        """Plan int64 routes ``flat`` against the live expert-to-slot mapping.

        Distinct misses take scratch rows in first-appearance order, matching
        the returned ``remap``. ``flat.numel()`` must not exceed the scratch
        rows or the plan capacity.
        """
        return plan_graph_routes(
            flat, self.expert_to_slot, self.scratch_rows, self.scratch_base
        )

    @staticmethod
    def fill_routes(route: GraphRoutePlan, plan: ExpertRowPlan) -> None:
        """Write a route plan's distinct misses and their count into ``plan``."""
        plan.expert_ids[: route.source_rows.numel()].copy_(route.source_rows)
        plan.count.copy_(route.miss_plan_rows.reshape(1))

    def plan_routes(self, flat: torch.Tensor, plan: ExpertRowPlan) -> GraphRoutePlan:
        """``route_plan`` followed by ``fill_routes``: the router-miss producer."""
        route = self.route_plan(flat)
        self.fill_routes(route, plan)
        return route

    def plan_candidates(
        self,
        candidates: torch.Tensor,
        plan: ExpertRowPlan,
        priority: torch.Tensor | None = None,
    ) -> None:
        """Plan nonresident candidate experts into ``plan``, highest priority first.

        ``candidates`` is int64 expert ids ``[N]`` (negative ids are padding;
        duplicates keep their highest priority) or a bool mask
        ``[num_experts]``. ``priority`` is an optional float tensor of the same
        shape; without it earlier ids (or lower expert ids for a mask) come
        first, and equal priorities break by lower expert id. Eligible experts
        always rank above ineligible ones, whatever their priority. At most
        ``min(plan.capacity, scratch_rows)`` rows are counted.
        """
        experts = self.num_experts
        device = self.expert_to_slot.device
        if candidates.dtype == torch.bool:
            if candidates.shape != (experts,):
                raise ValueError("a candidate mask must be bool [num_experts].")
            present = candidates
            score = (
                priority.to(torch.float32)
                if priority is not None
                else torch.zeros(experts, dtype=torch.float32, device=device)
            )
        else:
            if candidates.dtype != torch.int64 or candidates.ndim != 1:
                raise ValueError("candidate ids must be int64 [N].")
            valid = (candidates >= 0) & (candidates < experts)
            ids = torch.where(valid, candidates, 0)
            order_score = (
                priority.to(torch.float32)
                if priority is not None
                else -torch.arange(candidates.numel(), dtype=torch.float32, device=device)
            )
            present = (
                torch.zeros(experts, dtype=torch.uint8, device=device)
                .scatter_reduce(0, ids, valid.to(torch.uint8), "amax")
                .bool()
            )
            score = torch.full(
                (experts,), float("-inf"), dtype=torch.float32, device=device
            ).scatter_reduce(
                0, ids, torch.where(valid, order_score, float("-inf")), "amax"
            )
        if priority is not None and priority.shape != candidates.shape:
            raise ValueError("priority must align with candidates.")
        eligible = present & (self.expert_to_slot < 0)
        lowest = torch.finfo(torch.float32).min
        ranked = torch.argsort(
            torch.where(eligible, score.nan_to_num(nan=lowest).clamp(min=lowest), float("-inf")),
            descending=True,
            stable=True,
        )
        rows = min(plan.capacity, experts)
        plan.expert_ids[:rows].copy_(ranked[:rows])
        limit = min(plan.capacity, self.scratch_rows)
        plan.count.copy_(eligible.sum().clamp(max=limit).to(torch.int32).reshape(1))


class ExpertRowDelivery:
    """A resolved plan: which of its rows are valid in their slots, on the device."""

    def __init__(self, plan: ExpertRowPlan, delivered: torch.Tensor | None) -> None:
        self.plan = plan
        self.delivered = delivered

    def mask(self) -> torch.Tensor:
        """Bool ``[C]``: planned rows valid in their slots."""
        rows = torch.arange(
            self.plan.capacity, dtype=torch.int32, device=self.plan.count.device
        )
        planned = rows < self.plan.count
        if self.delivered is None:
            return planned
        return planned & (self.delivered != 0)


class InGraphRowBackend:
    """Serves a plan with the in-graph copy kernel at ``post``; everything is delivered."""

    name = BACKEND_IN_GRAPH

    def __init__(self, segments: Mapping[int, ExpertRowSegments]) -> None:
        self.segments = dict(segments)

    def post(self, tag: int, plan: ExpertRowPlan) -> None:
        copy_expert_row_segments_gpu(
            self.segments[tag], plan.expert_ids, plan.slots, plan.count
        )

    def resolve(self, tag: int, plan: ExpertRowPlan) -> ExpertRowDelivery:
        return ExpertRowDelivery(plan, None)

    def copy_residual(self, tag: int, delivery: ExpertRowDelivery) -> None:
        """Nothing is ever left undelivered."""


class DoorbellRowBackend:
    """Serves plans through an ``ExpertDoorbellCopier`` thread, one tag per target layer."""

    name = BACKEND_DOORBELL

    def __init__(self, copier, segments: Mapping[int, ExpertRowSegments]) -> None:
        self.copier = copier
        self.segments = dict(segments)

    def post(self, tag: int, plan: ExpertRowPlan) -> None:
        self.copier.post(plan.expert_ids, plan.slots, plan.count, tag=tag)

    def resolve(self, tag: int, plan: ExpertRowPlan) -> ExpertRowDelivery:
        return ExpertRowDelivery(plan, self.copier.resolve(tag))

    def copy_residual(self, tag: int, delivery: ExpertRowDelivery) -> None:
        """Copy in-graph the plan rows the thread did not deliver."""
        plan = delivery.plan
        copy_expert_row_segments_gpu(
            self.segments[tag],
            plan.expert_ids,
            plan.slots,
            self.copier.undelivered_count(tag, plan.count),
        )


def plan_residual_routes(
    flat: torch.Tensor,
    slot_source,
    scratch_base: int,
    delivery: ExpertRowDelivery,
    residual: ExpertRowPlan,
) -> torch.Tensor:
    """Plan actual routes against a delivered scratch plan; return the route remap.

    ``slot_source`` is the target layer's hot cache or a callable returning its
    live expert-to-slot map (never the map tensor itself, see
    ``live_slot_map``). ``flat`` holds int64 expert ids of the routes. A route whose expert is
    resident maps to its hot slot, one whose expert the plan delivered maps to
    that plan row's scratch row, and every other distinct expert (the residual)
    takes a scratch row no needed delivered expert occupies, in first-appearance
    order, written into ``residual`` for the in-graph copy kernel. The delivered
    plan must address scratch rows ``scratch_base + r``, and the distinct
    nonresident routes must fit ``residual.capacity`` scratch rows.
    """
    plan = delivery.plan
    if residual is plan or any(
        mine.data_ptr() == theirs.data_ptr()
        for mine, theirs in (
            (residual.expert_ids, plan.expert_ids),
            (residual.slots, plan.slots),
            (residual.count, plan.count),
        )
    ):
        raise ValueError("the residual plan must not share tensors with the delivered plan.")
    expert_to_slot = live_slot_map(slot_source)()
    experts = expert_to_slot.numel()
    device = flat.device
    scratch_rows = residual.capacity
    plan_rows = torch.arange(plan.capacity, dtype=torch.long, device=device)
    delivered = delivery.mask()
    row_of_expert = torch.full((experts + 1,), -1, dtype=torch.long, device=device)
    row_of_expert.scatter_(
        0,
        torch.where(delivered, plan.expert_ids, experts),
        torch.where(delivered, plan_rows, -1),
    )
    row_of_expert = row_of_expert[:experts]
    covering = torch.where(
        (row_of_expert >= 0) & (expert_to_slot < 0),
        row_of_expert + scratch_base,
        expert_to_slot,
    )
    route = plan_graph_routes(flat, covering, scratch_rows, scratch_base)
    route_rows = row_of_expert.index_select(0, flat)
    covered = (expert_to_slot.index_select(0, flat) < 0) & (route_rows >= 0)
    needed = torch.zeros(scratch_rows + 1, dtype=torch.bool, device=device)
    needed.scatter_(0, torch.where(covered, route_rows, scratch_rows), True)
    free_rows = torch.argsort(needed[:scratch_rows].to(torch.uint8), stable=True)
    missing = covering.index_select(0, flat) < 0
    rank = (route.remap - scratch_base).clamp(0, scratch_rows - 1)
    remap = torch.where(
        missing, free_rows.index_select(0, rank) + scratch_base, route.remap
    )
    distinct = route.source_rows.numel()
    residual.expert_ids[:distinct].copy_(route.source_rows)
    residual.slots[:distinct].copy_(
        (free_rows[:distinct] + scratch_base).to(torch.int32)
    )
    residual.count.copy_(route.miss_plan_rows.reshape(1))
    return remap
