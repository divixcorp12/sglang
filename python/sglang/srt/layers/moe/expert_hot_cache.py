"""Fixed CUDA slots for frequently selected host-resident expert rows."""

from __future__ import annotations

import atexit
import copy
import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import IntEnum
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from sglang.srt.environ import InsertOnMissStage, envs
from sglang.srt.layers.moe.async_telemetry import AsyncTelemetry, TorchTelemetryBackend
from sglang.srt.layers.moe.expert_prefetch import (
    ExpertPrefetchCoordinator,
    SparseNextLayerPolicy,
)
from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyPolicy,
    advance_residency_policies,
    decide_residency_policies,
    decide_residency_policies_from_host_scores,
)
from sglang.srt.layers.moe.expert_residency_clock import (
    ForwardKind,
    ResidencyBoundaryClock,
    classify_forward,
)
from sglang.srt.layers.moe.expert_format import (
    inclusive_hot_slot_limit,
    iter_expert_streamers,
    require_graph_gather_support,
)
from sglang.srt.layers.moe.expert_stream import ExpertStreamer

from sglang.srt.layers.moe.expert_transfer import (
    NVFP4_TRANSFER_TENSOR_COUNT,
    AsyncExpertTransferExecutor,
    ExpertCopySubmission,
    ExpertRowCopyRequest,
    ExpertRowCopyRoutes,
    ExpertTransferTicket,
    FixedRowTransferPlan,
    submit_expert_row_copy_batch,
)
if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def graph_gather_scratch_rows(tokens: int, top_k: int, max_rows: int = 0) -> int:
    """Scratch rows one layer's graph gather reserves.

    One row per route of a ``tokens``-token forward (decode batch size, times
    verify tokens per request under speculation), capped at ``max_rows`` when
    positive. The graph gather then serves only forwards with at most that
    many routes.
    """
    tokens, top_k, max_rows = index(tokens), index(top_k), index(max_rows)
    if tokens < 0 or top_k < 0 or max_rows < 0:
        raise ValueError("graph gather scratch sizes cannot be negative")
    rows = tokens * top_k
    return min(rows, max_rows) if max_rows else rows


@dataclass(frozen=True)
class HotCacheUpdateStats:
    promoted_experts: int
    evicted_experts: int
    migration_bytes: int


class HotCacheSlotState(IntEnum):
    """Lifecycle states for a fixed expert-cache destination slot."""

    FREE = 0
    RESERVED = 1
    LOADING = 2
    READY = 3


@dataclass(frozen=True)
class HotCacheSlotTicket:
    """Generation-qualified authority to load or retire one cache slot."""

    slot: int
    expert_id: int
    generation: int


class ExpertHotCache:
    """Own stable slot tensors; reassign only between serialized gather calls.

    Slots and lookups live on the streamer's CUDA source device, or the current
    CUDA device when all backing sources are on the host. Like ExpertStreamer,
    updates and gathers must run on the same CUDA stream. Sources must remain
    unchanged while their rows are resident in the cache.
    """

    def __init__(self, streamer: ExpertStreamer, capacity: int, scratch_rows: int = 0):
        """``scratch_rows`` extra rows after the slots receive graph-gather misses.

        With a non-off ``SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE``, one further trailing row is
        appended for ``DedicatedPrefetchSlot`` (plan section 7.1): its index is
        ``capacity + scratch_rows``, so the allocation must be exactly one row
        larger than the flag-off shape for ``PrefetchPuller``'s setup-time
        ``assert_within_allocation`` to accept it.
        """
        capacity = index(capacity)
        scratch_rows = index(scratch_rows)
        if not 0 <= capacity <= streamer.num_experts:
            raise ValueError("hot cache capacity must be within the expert count")
        if scratch_rows < 0:
            raise ValueError("hot cache scratch rows cannot be negative")
        self.streamer = streamer
        self.capacity = capacity
        self.scratch_rows = scratch_rows
        self.reserves_prefetch_pull_row = (
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off"
        )
        allocation_rows = capacity + scratch_rows + int(self.reserves_prefetch_pull_row)
        self.bytes_per_expert = streamer.bytes_per_expert
        self.capacity_bytes = capacity * self.bytes_per_expert
        self.scratch_bytes = scratch_rows * self.bytes_per_expert
        self.prefetch_pull_bytes = (
            self.bytes_per_expert if self.reserves_prefetch_pull_row else 0
        )
        self.allocation_bytes = allocation_rows * self.bytes_per_expert
        devices = {
            streamer.source(spec.name).device
            for spec in streamer.specs
            if spec.residence == "device"
        }
        if len(devices) > 1:
            raise ValueError("hot cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            spec.name: torch.empty(
                (allocation_rows,) + spec.row_shape,
                dtype=spec.dtype,
                device=self.device,
            )
            for spec in streamer.specs
        }
        self.expert_to_slot = torch.full(
            (streamer.num_experts,), -1, dtype=torch.long, device=self.device
        )
        self.slot_to_expert = [-1] * capacity
        self.slot_states = [HotCacheSlotState.FREE] * capacity
        self.slot_state = torch.full(
            (capacity,),
            int(HotCacheSlotState.FREE),
            dtype=torch.uint8,
            device=self.device,
        )
        self.slot_generations = torch.zeros(
            (capacity,), dtype=torch.long, device=self.device
        )
        self._slot_generations = [0] * capacity
        streamer.hot_cache = self
        self.copy_backend = getattr(streamer, "expert_copy_backend", "gpu")
        self._transfer_executor = (
            AsyncExpertTransferExecutor.for_device(self.device) if capacity else None
        )
        self._transfer_plan = (
            FixedRowTransferPlan(max_rows=capacity, device=self.device)
            if capacity
            else None
        )
        self.last_copy_submission: ExpertCopySubmission | None = None
        self.promotion_in_flight: HotCachePromotion | None = None
        self._copy_routes: dict[tuple, ExpertRowCopyRoutes] = {}
        self._slot_batch_depth = 0
        self._slots_dirty = False
        self._slot_upload_recorded = False
        if capacity:
            self._slot_upload_host = torch.zeros(
                (3, capacity), dtype=torch.int64
            ).pin_memory()
            self._slot_upload_rows = self._slot_upload_host.numpy()
            self._slot_upload_device = torch.zeros(
                (3, capacity), dtype=torch.int64, device=self.device
            )
            self._slot_upload_event = torch.cuda.Event()
            self._slot_ids = torch.arange(
                capacity, dtype=torch.long, device=self.device
            )
            self._slot_dump_targets = np.arange(
                streamer.num_experts, streamer.num_experts + capacity, dtype=np.int64
            )
            self._mapping_scratch = torch.full(
                (streamer.num_experts + capacity,),
                -1,
                dtype=torch.long,
                device=self.device,
            )

    @staticmethod
    def capacity_for_budget(streamer: ExpertStreamer, budget_bytes: int) -> int:
        """Round a runtime tensor byte budget down to complete expert slots."""
        budget_bytes = index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("hot cache byte budget cannot be negative")
        return min(streamer.num_experts, budget_bytes // streamer.bytes_per_expert)

    def _publish_slots(self) -> None:
        """Push host slot states, generations and the expert-to-slot mapping to the device.

        The host lists are authoritative. One non-blocking copy of a pinned
        ``[3, capacity]`` buffer carries each slot's state, generation and
        mapping target: its expert when READY, otherwise a dump index past the
        experts unique to the slot. The mapping is rebuilt on the device with
        ``index_copy_`` and copied into ``expert_to_slot`` in place. The pinned
        buffer is rewritten only after its previous upload has run.
        """
        self._slots_dirty = False
        if getattr(self, "device_residency", None) is not None:
            raise RuntimeError("hot cache slots are owned by the GPU residency update")
        if not self.capacity:
            return
        if self._slot_upload_recorded:
            self._slot_upload_event.synchronize()
        rows = self._slot_upload_rows
        rows[0] = self.slot_states
        rows[1] = self._slot_generations
        ready = rows[0] == int(HotCacheSlotState.READY)
        rows[2] = np.where(
            ready, np.asarray(self.slot_to_expert, dtype=np.int64), self._slot_dump_targets
        )
        upload = self._slot_upload_device
        upload.copy_(self._slot_upload_host, non_blocking=True)
        self._slot_upload_event.record(torch.cuda.current_stream(self.device))
        self._slot_upload_recorded = True
        self.slot_state.copy_(upload[0])
        self.slot_generations.copy_(upload[1])
        self._mapping_scratch.fill_(-1)
        self._mapping_scratch.index_copy_(0, upload[2], self._slot_ids)
        self.expert_to_slot.copy_(self._mapping_scratch[: self.streamer.num_experts])

    def wait_for_slot_publication(self) -> None:
        """Block the host until the last slot publication has run on the device."""
        if self._slot_upload_recorded:
            self._slot_upload_event.synchronize()

    @contextmanager
    def _slot_batch(self, publish: bool = True) -> Iterator[None]:
        """Collect lifecycle changes and publish them once as the outermost batch exits.

        ``publish=False`` leaves the collected changes for an explicit
        :meth:`_publish_slots`, including when the batch exits by an exception.
        """
        self._slot_batch_depth += 1
        try:
            yield
        finally:
            self._slot_batch_depth -= 1
            if publish and not self._slot_batch_depth and self._slots_dirty:
                self._publish_slots()

    def _set_slot_state(self, slot: int, state: HotCacheSlotState) -> None:
        self.slot_states[slot] = state
        self._slots_dirty = True

    def resident_experts(self) -> frozenset[int]:
        """Return only mappings that are ready for a gather consumer."""
        if getattr(self, "device_residency", None) is not None:
            raise RuntimeError("hot cache slots are owned by the GPU residency update")
        return frozenset(
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is HotCacheSlotState.READY
        )

    def _ticket_matches(
        self, ticket: HotCacheSlotTicket, *states: HotCacheSlotState
    ) -> bool:
        return (
            0 <= ticket.slot < self.capacity
            and self._slot_generations[ticket.slot] == ticket.generation
            and self.slot_to_expert[ticket.slot] == ticket.expert_id
            and self.slot_states[ticket.slot] in states
        )

    def reserve(
        self,
        placements: Sequence[tuple[int, int]],
        *,
        consumer_complete: bool = False,
    ) -> tuple[HotCacheSlotTicket, ...]:
        """Reserve slots without making their destinations visible to gathers.

        A transfer executor may call this before issuing its six row copies, then
        use ``begin_loading`` and ``publish_ready`` after its completion event.
        A READY victim can only be reused when the caller confirms its last
        consumer has completed.
        """
        with self._slot_batch():
            return self._reserve(placements, consumer_complete)

    def _reserve(
        self, placements: Sequence[tuple[int, int]], consumer_complete: bool
    ) -> tuple[HotCacheSlotTicket, ...]:
        assignments = tuple((index(expert), index(slot)) for expert, slot in placements)
        if len({expert for expert, _ in assignments}) != len(assignments):
            raise ValueError("hot cache expert IDs must be unique")
        if len({slot for _, slot in assignments}) != len(assignments):
            raise ValueError("hot cache destination slots must be unique")
        if any(
            expert < 0
            or expert >= self.streamer.num_experts
            or slot < 0
            or slot >= self.capacity
            for expert, slot in assignments
        ):
            raise ValueError("hot cache prefetch placement is outside capacity")
        active_experts = {
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is not HotCacheSlotState.FREE
        }
        if any(expert in active_experts for expert, _ in assignments):
            raise ValueError("hot cache reservation requires a nonresident expert")
        ready_victims = [
            slot
            for _, slot in assignments
            if self.slot_states[slot] is HotCacheSlotState.READY
        ]
        if ready_victims and not consumer_complete:
            raise RuntimeError("hot cache victim still has an active consumer")
        for slot in ready_victims:
            if not self.retire(self.ticket_for_slot(slot), consumer_complete=True):
                raise RuntimeError("hot cache victim could not be retired")
        if any(
            self.slot_states[slot] is not HotCacheSlotState.FREE
            for _, slot in assignments
        ):
            raise RuntimeError("hot cache slot is not available for reservation")
        tickets = []
        for expert, slot in assignments:
            generation = self._slot_generations[slot] + 1
            self._slot_generations[slot] = generation
            self.slot_to_expert[slot] = expert
            self._set_slot_state(slot, HotCacheSlotState.RESERVED)
            tickets.append(HotCacheSlotTicket(slot, expert, generation))
        return tuple(tickets)

    def begin_loading(self, ticket: HotCacheSlotTicket) -> bool:
        """Mark a valid reservation as loading; reject stale tickets."""
        if not self._ticket_matches(ticket, HotCacheSlotState.RESERVED):
            return False
        with self._slot_batch():
            self._set_slot_state(ticket.slot, HotCacheSlotState.LOADING)
        return True

    def publish_ready(self, ticket: HotCacheSlotTicket) -> bool:
        """Publish a fully copied six-tensor slot after its ticket completes."""
        if not self._ticket_matches(ticket, HotCacheSlotState.LOADING):
            return False
        with self._slot_batch():
            self._set_slot_state(ticket.slot, HotCacheSlotState.READY)
        return True

    def cancel(self, ticket: HotCacheSlotTicket) -> bool:
        """Release an unconsumed reservation; stale tickets cannot alter slots."""
        if not self._ticket_matches(
            ticket, HotCacheSlotState.RESERVED, HotCacheSlotState.LOADING
        ):
            return False
        with self._slot_batch():
            self.slot_to_expert[ticket.slot] = -1
            self._set_slot_state(ticket.slot, HotCacheSlotState.FREE)
        return True

    def ticket_for_slot(self, slot: int) -> HotCacheSlotTicket:
        """Return the current generation-qualified ticket for a nonfree slot."""
        slot = index(slot)
        if not 0 <= slot < self.capacity:
            raise ValueError("hot cache slot is outside capacity")
        if self.slot_states[slot] is HotCacheSlotState.FREE:
            raise ValueError("hot cache free slots have no active ticket")
        return HotCacheSlotTicket(
            slot, self.slot_to_expert[slot], self._slot_generations[slot]
        )

    def retire(self, ticket: HotCacheSlotTicket, *, consumer_complete: bool) -> bool:
        """Unpublish a READY slot only after its final consumer has completed."""
        if not consumer_complete or not self._ticket_matches(
            ticket, HotCacheSlotState.READY
        ):
            return False
        with self._slot_batch():
            self.slot_to_expert[ticket.slot] = -1
            self._set_slot_state(ticket.slot, HotCacheSlotState.FREE)
        return True

    def _load_reserved(self, tickets: Sequence[HotCacheSlotTicket]) -> None:
        """Copy one reserved placement bundle before publishing any slot mapping.

        Tensors without a dense source are read by the streamer's row source:
        through the pinned host tier in chunks of its capacity when the layer
        has six tensors and a pinned tier, else one row at a time into staging.
        """
        if not tickets:
            return
        assert self._transfer_executor is not None
        spec_only = self.streamer.has_spec_only_tensors
        six_tensors = len(self.streamer.tensor_names) == NVFP4_TRANSFER_TENSOR_COUNT
        pinned_cache = self.streamer.pinned_host_cache
        if spec_only and six_tensors and pinned_cache is not None and pinned_cache.capacity:
            self._load_reserved_in_chunks(tickets, pinned_cache)
            return
        if not six_tensors or spec_only:
            for ticket in tickets:
                if not self.begin_loading(ticket):
                    raise RuntimeError("hot cache reservation became stale")
                source_ids = torch.tensor(
                    [ticket.expert_id], dtype=torch.long, device=self.device
                )
                outputs = {
                    name: tensor[ticket.slot : ticket.slot + 1]
                    for name, tensor in self.tensors.items()
                }
                self.streamer._copy_source_rows(source_ids, outputs)
                if not self.publish_ready(ticket):
                    raise RuntimeError("hot cache completion ticket became stale")
            return
        promotion = self._prepare_promotion(tickets)
        current_stream = torch.cuda.current_stream(self.device)
        ticket = submit_hot_cache_promotions([promotion], producer_stream=current_stream)
        self._transfer_executor.wait(ticket, current_stream)
        self.complete_promotion(promotion)
        self.wait_for_slot_publication()

    def _load_reserved_in_chunks(
        self, tickets: Sequence[HotCacheSlotTicket], pinned_cache
    ) -> None:
        """Promote tickets through the pinned tier, one chunk per transfer.

        Each chunk holds at most ``pinned_cache.evictable_rows()`` tickets,
        read when the chunk starts: an inclusive ``is_pinned`` protects the
        rows the hot cache has reserved, so every admitted chunk can shrink the
        room for the next. Each chunk's rows are admitted to the pinned tier,
        copied from its slabs, and waited for on the host before the next chunk
        may evict them. When no slot is evictable, the remaining tickets are
        cancelled and the call raises.

        If a chunk fails, the tickets after it are cancelled, and so is its own
        promotion if it is still in flight. A failed ``_prepare_promotion`` or
        submission has already cancelled its own tickets. Once copies were
        submitted, their slots are freed only after the device has drained
        them. If the device cannot drain (a sticky CUDA error), the promotion
        stays in flight with its slots LOADING. ``stage_reassign`` then refuses
        further updates, so no later reservation can reuse a slot a copy may
        still write.
        """
        tickets = tuple(tickets)
        start = 0
        while start < len(tickets):
            # One host use from sizing the chunk to its completed copy: a slot
            # table with an owner (a native reader thread) must not move the
            # chunk's pinned rows while the promotion reads them.
            with pinned_cache.host_use():
                chunk_rows = pinned_cache.evictable_rows()
                if chunk_rows < 1:
                    self._cancel_tickets(tickets[start:])
                    raise RuntimeError(
                        "the pinned host tier has no evictable slots for hot cache "
                        "promotions"
                    )
                chunk = tickets[start : start + chunk_rows]
                promotion = None
                submitted = False
                try:
                    promotion = self._prepare_promotion(chunk)
                    current_stream = torch.cuda.current_stream(self.device)
                    ticket = submit_hot_cache_promotions(
                        [promotion], producer_stream=current_stream
                    )
                    submitted = True
                    self._transfer_executor.wait(ticket, current_stream)
                    current_stream.synchronize()
                    self.complete_promotion(promotion)
                except BaseException:
                    if promotion is not None and self.promotion_in_flight is promotion:
                        if not submitted or self._drain_device():
                            self.abort_promotion(promotion)
                    rest = tickets[start + len(chunk) :]
                    if rest:
                        self._cancel_tickets(rest)
                    raise
            start += len(chunk)
        self.wait_for_slot_publication()

    def _drain_device(self) -> bool:
        """Wait until every queued copy on the cache's device has run; False if it cannot."""
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            logger.warning(
                "hot cache promotion copies could not be drained; their slots stay "
                "LOADING and the cache refuses further updates"
            )
            return False
        return True

    def _copy_routes_for(
        self, sources: Mapping[str, torch.Tensor], use_secondary: Sequence[bool]
    ) -> ExpertRowCopyRoutes:
        """Routes validated once per backend, secondary selection and source storage."""
        key = (
            self.copy_backend,
            tuple(use_secondary),
            tuple(
                (source.data_ptr(), source.dtype, tuple(source.shape), source.device)
                for source in sources.values()
            ),
        )
        routes = self._copy_routes.get(key)
        if routes is None:
            routes = ExpertRowCopyRoutes(
                [(sources[name], self.tensors[name]) for name in self.streamer.tensor_names],
                backend=self.copy_backend,
                use_secondary_source_rows=use_secondary,
            )
            if len(self._copy_routes) >= 4:
                self._copy_routes.clear()
            self._copy_routes[key] = routes
        return routes

    def _prepare_promotion(
        self, tickets: Sequence[HotCacheSlotTicket]
    ) -> HotCachePromotion:
        """Stage reserved tickets' rows in the transfer plan and mark them loading.

        Nothing is copied and nothing is published. On failure every ticket is
        cancelled.
        """
        assert self._transfer_plan is not None
        try:
            expert_rows = [ticket.expert_id for ticket in tickets]
            destination_slots = [ticket.slot for ticket in tickets]
            sources = {
                name: self.streamer.source(name)
                for name in self.streamer.tensor_names
            }
            secondary_source_rows = None
            use_secondary_source_rows = [False] * len(self.streamer.tensor_names)
            pinned_cache = self.streamer.pinned_host_cache
            if pinned_cache is not None and pinned_cache.cached_names:
                # Admission and the slot read share one host use, so the slots
                # read are the ones ensure_rows filled.
                with pinned_cache.host_use():
                    pinned_cache.ensure_rows(
                        torch.tensor(expert_rows, device=self.device)
                    )
                    cached_rows = [
                        pinned_cache._expert_to_slot.get(expert_id, -1)
                        for expert_id in expert_rows
                    ]
                if all(slot >= 0 for slot in cached_rows):
                    secondary_source_rows = cached_rows
                    for position, name in enumerate(self.streamer.tensor_names):
                        if name in pinned_cache.tensors:
                            sources[name] = pinned_cache.tensors[name]
                            use_secondary_source_rows[position] = True
            if any(source is None for source in sources.values()):
                raise RuntimeError(
                    "hot cache promotion of expert tensors without a dense source "
                    "needs every row in the pinned host tier; promote at most its "
                    "capacity at once"
                )
            self._transfer_plan.set_rows(
                expert_rows,
                destination_slots,
                [ticket.generation for ticket in tickets],
                secondary_source_rows=secondary_source_rows,
            )
            routes = self._copy_routes_for(sources, use_secondary_source_rows)
            with self._slot_batch(publish=False):
                if not all(self.begin_loading(ticket) for ticket in tickets):
                    raise RuntimeError("hot cache reservation became stale")
        except BaseException:
            self._cancel_tickets(tickets)
            raise
        promotion = HotCachePromotion(
            self,
            tuple(tickets),
            routes,
            expert_rows,
            destination_slots,
            secondary_source_rows,
        )
        self.promotion_in_flight = promotion
        return promotion

    def _cancel_tickets(self, tickets: Sequence[HotCacheSlotTicket]) -> None:
        with self._slot_batch():
            for ticket in tickets:
                self.cancel(ticket)

    def abort_promotion(self, promotion: HotCachePromotion) -> None:
        """Cancel a promotion whose copies were never submitted and publish the slots."""
        if self.promotion_in_flight is promotion:
            self.promotion_in_flight = None
        self._cancel_tickets(promotion.tickets)

    def complete_promotion(self, promotion: HotCachePromotion) -> None:
        """Publish a promotion's slots after its copies completed, in one device push.

        A promotion completes at most once: completing one that is not in
        flight raises instead of publishing its tickets again.
        """
        if self.promotion_in_flight is not promotion:
            raise RuntimeError("hot cache promotion is not in flight")
        self.promotion_in_flight = None
        try:
            with self._slot_batch():
                if not all(self.publish_ready(ticket) for ticket in promotion.tickets):
                    raise RuntimeError("hot cache completion ticket became stale")
        except BaseException:
            self._cancel_tickets(promotion.tickets)
            raise

    def stage_reassign(
        self, expert_ids: Sequence[int], *, publish: bool
    ) -> tuple[HotCacheUpdateStats, HotCachePromotion | None]:
        """Retire evicted slots and reserve promoted ones without copying rows.

        Returns the update's statistics and, when rows must be copied, the
        staged promotion to submit with :func:`submit_hot_cache_promotions`
        and finish with :meth:`complete_promotion`. With ``publish`` the
        retired and loading slots reach the device before this returns, as a
        caller must require when gathers may run before the copies complete;
        otherwise they reach it with the completed promotion. An update with
        nothing to copy is always published.
        """
        desired = [index(expert_id) for expert_id in expert_ids]
        if len(desired) > self.capacity:
            raise ValueError("expert selection exceeds hot cache capacity")
        if len(set(desired)) != len(desired):
            raise ValueError("hot cache expert IDs must be unique")
        if any(
            expert_id < 0 or expert_id >= self.streamer.num_experts
            for expert_id in desired
        ):
            raise ValueError("hot cache expert ID is outside the expert range")
        if self.promotion_in_flight is not None:
            raise RuntimeError("hot cache promotion is still in flight")
        wanted = set(desired)
        existing = {
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is HotCacheSlotState.READY
        }
        promoted = [expert_id for expert_id in desired if expert_id not in existing]
        evicted = existing - wanted
        if not promoted and not evicted:
            return HotCacheUpdateStats(0, 0, 0), None
        promotion = None
        try:
            with self._slot_batch(publish=False):
                for slot, expert_id in enumerate(self.slot_to_expert):
                    if expert_id in evicted:
                        self.retire(self.ticket_for_slot(slot), consumer_complete=True)
                free_slots = [
                    slot
                    for slot, state in enumerate(self.slot_states)
                    if state is HotCacheSlotState.FREE
                ]
                if len(free_slots) < len(promoted):
                    raise RuntimeError("hot cache has no free slots for reassignment")
                tickets = self.reserve(
                    tuple(zip(promoted, free_slots)), consumer_complete=True
                )
                if (
                    len(self.streamer.tensor_names) != NVFP4_TRANSFER_TENSOR_COUNT
                    or self.streamer.has_spec_only_tensors
                ):
                    self._load_reserved(tickets)
                elif tickets:
                    promotion = self._prepare_promotion(tickets)
        except BaseException:
            self._publish_slots()
            raise
        if promotion is None or publish:
            self._publish_slots()
        return (
            HotCacheUpdateStats(
                len(promoted), len(evicted), len(promoted) * self.bytes_per_expert
            ),
            promotion,
        )

    def reassign(self, expert_ids: Sequence[int]) -> HotCacheUpdateStats:
        """Synchronously replace slots through the generation-safe lifecycle."""
        stats, promotion = self.stage_reassign(expert_ids, publish=False)
        if promotion is not None:
            current_stream = torch.cuda.current_stream(self.device)
            ticket = submit_hot_cache_promotions(
                [promotion], producer_stream=current_stream
            )
            self._transfer_executor.wait(ticket, current_stream)
            self.complete_promotion(promotion)
        self.wait_for_slot_publication()
        return stats

    def prefetch_destinations(
        self, candidates: Sequence[int], protected_slots: Sequence[int]
    ) -> tuple[tuple[int, int], ...]:
        """Select stable unprotected victim slots for speculative admissions."""
        protected = {index(slot) for slot in protected_slots}
        if any(slot < 0 or slot >= self.capacity for slot in protected):
            raise ValueError("protected hot cache slot is outside capacity")
        existing = {
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is not HotCacheSlotState.FREE
        }
        pending = []
        for expert_id in candidates:
            expert_id = index(expert_id)
            if expert_id in existing or expert_id in pending:
                continue
            if expert_id < 0 or expert_id >= self.streamer.num_experts:
                raise ValueError("hot cache expert ID is outside the expert range")
            pending.append(expert_id)
        writable = [
            slot
            for slot, state in enumerate(self.slot_states)
            if slot not in protected
            and state in (HotCacheSlotState.FREE, HotCacheSlotState.READY)
        ]
        writable.sort(
            key=lambda slot: (self.slot_states[slot] is HotCacheSlotState.READY, slot)
        )
        return tuple(zip(pending, writable))

    def assign_prefetch(
        self, placements: Sequence[tuple[int, int]]
    ) -> HotCacheUpdateStats:
        """Copy speculative rows into destinations selected before stream launch."""
        assignments = tuple((index(expert), index(slot)) for expert, slot in placements)
        if any(
            expert < 0
            or expert >= self.streamer.num_experts
            or slot < 0
            or slot >= self.capacity
            for expert, slot in assignments
        ):
            raise ValueError("hot cache prefetch placement is outside capacity")
        existing = {
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is not HotCacheSlotState.FREE
        }
        if any(expert in existing for expert, _ in assignments):
            raise ValueError("hot cache prefetch must not replace a resident expert")
        evictions = sum(
            self.slot_states[slot] is HotCacheSlotState.READY for _, slot in assignments
        )
        tickets = self.reserve(assignments, consumer_complete=True)
        self._load_reserved(tickets)
        return HotCacheUpdateStats(
            len(assignments), evictions, len(assignments) * self.bytes_per_expert
        )

    def lookup(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slot IDs (-1 for misses) and a hit mask for valid expert IDs."""
        if source_ids.device != self.device:
            raise ValueError("selected expert IDs must use the hot cache CUDA device")
        slots = self.expert_to_slot[source_ids.long()]
        return slots, slots >= 0

    def data_ptrs(self) -> tuple[int, ...]:
        """Expose slot and mapping pointers for stable-allocation verification."""
        return tuple(tensor.data_ptr() for tensor in self.tensors.values()) + (
            self.expert_to_slot.data_ptr(),
            self.slot_state.data_ptr(),
            self.slot_generations.data_ptr(),
        )


@dataclass(eq=False)
class HotCachePromotion:
    """One cache's reserved slots whose rows are staged but not yet published."""

    cache: ExpertHotCache
    tickets: tuple[HotCacheSlotTicket, ...]
    routes: ExpertRowCopyRoutes
    source_rows: list[int]
    destination_slots: list[int]
    secondary_source_rows: list[int] | None
    submission: ExpertCopySubmission | None = None


def submit_hot_cache_promotions(
    promotions: Sequence[HotCachePromotion], *, producer_stream
) -> ExpertTransferTicket:
    """Submit staged promotions' row copies behind one transfer ticket.

    The copies wait for ``producer_stream``, so device work queued before this
    call (slot publication, plan uploads, a forward still running) precedes
    them. Each cache's ``last_copy_submission`` reports its own rows. If the
    submission fails, every promotion is cancelled and published.
    """
    promotions = tuple(promotions)
    executor = promotions[0].cache._transfer_executor
    try:
        if any(
            promotion.cache._transfer_executor is not executor
            for promotion in promotions
        ):
            raise ValueError("hot cache promotions must share one transfer executor")
        submissions = submit_expert_row_copy_batch(
            executor,
            [
                ExpertRowCopyRequest(
                    promotion.routes,
                    promotion.cache._transfer_plan,
                    promotion.source_rows,
                    promotion.destination_slots,
                    promotion.secondary_source_rows,
                )
                for promotion in promotions
            ],
            producer_stream=producer_stream,
        )
    except BaseException:
        for promotion in promotions:
            promotion.cache.abort_promotion(promotion)
        raise
    for promotion, submission in zip(promotions, submissions):
        promotion.submission = submission
        promotion.cache.last_copy_submission = submission
    return submissions[0].ticket


def normalize_expert_frequency_seed(data: Mapping[str, Any]) -> torch.Tensor:
    """Normalize recorder steps or routing artifacts to CPU [layer, expert] counts."""
    if not isinstance(data, Mapping):
        raise ValueError("expert frequency seed must be a mapping")
    key = next(
        (name for name in ("logical_count", "count", "mass") if name in data), None
    )
    if key is None:
        raise ValueError("expert frequency seed requires logical_count, count, or mass")
    counts = torch.as_tensor(data[key], dtype=torch.float64, device="cpu")
    expected_ndim = 3 if key == "logical_count" else 2
    if counts.ndim != expected_ndim or counts.numel() == 0:
        raise ValueError("expert frequency seed has invalid dimensions")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("expert frequency seed must contain finite nonnegative counts")
    return counts.sum(dim=0) if key == "logical_count" else counts


@dataclass
class _OperationalCounters:
    """One phase/layer's cumulative telemetry.

    ``miss_rows`` is the total count of physical host-copy rows: every row
    actually copied to serve this layer, whether through the ordinary demand
    path (``unique_missed`` routes with no resident slot) or delivered ahead
    of routing by the one-row side-stream pull (``side_pull_rows``), useful or
    wasted. ``unique_miss_rows`` and ``routed_miss_rows`` stay logical counts
    of distinct and routed actual misses and never include a side-pull row
    that a miss did not itself require crossing the demand path for.
    """

    requested_rows: int = 0
    miss_rows: int = 0
    hot_hits: int = 0
    file_misses: int | None = None
    d2d_bytes: int = 0
    h2d_bytes: int = 0
    backing_source_bytes: int = 0
    requested_copy_backend: str | None = None
    actual_copy_backend: str | None = None
    copy_rows: int = 0
    copy_bytes: int = 0
    copy_submissions: int = 0
    copy_fallbacks: int = 0
    file_source_bytes: int | None = None
    requested_unique_experts: int = 0
    promotions: int = 0
    evictions: int = 0
    migration_bytes: int = 0
    residency_bytes: int = 0
    pinned_hits: int = 0
    pinned_misses: int = 0
    pinned_admissions: int = 0
    pinned_evictions: int = 0
    file_fallbacks: int | None = None
    transfer_wait_ns: int = 0
    gather_fallbacks: int = 0
    gather_copy_engine_bytes: int = 0
    routed_rows: int = 0
    routed_miss_rows: int = 0
    unique_miss_rows: int = 0
    gathers: int = 0
    side_pull_rows: int = 0
    side_pull_bytes: int = 0
    side_pull_posted_rows: int = 0
    side_pull_useful_posts: int = 0
    side_pull_wasted_rows: int = 0
    side_pull_covered_routes: int = 0
    side_pull_residual_routes: int = 0
    side_pull_useful_precision: float = 0.0
    host_read_rows: int = 0
    host_read_file_bytes: int = 0
    host_read_split_bytes: int = 0
    host_read_ns: int = 0
    host_split_ns: int = 0


def _add_host_read_counters(counters: _OperationalCounters, stats: Any) -> None:
    """Sum one gather's row-source reads into a phase/layer's counters."""
    counters.host_read_rows += getattr(stats, "host_read_rows", 0)
    counters.host_read_file_bytes += getattr(stats, "host_read_file_bytes", 0)
    counters.host_read_split_bytes += getattr(stats, "host_read_split_bytes", 0)
    counters.host_read_ns += getattr(stats, "host_read_ns", 0)
    counters.host_split_ns += getattr(stats, "host_split_ns", 0)


_PHASES = {
    ForwardKind.PREFILL: "prefill",
    ForwardKind.DECODE: "decode",
    ForwardKind.IDLE: "decode",
    ForwardKind.VERIFY: "speculative",
}


_INSERTION_TRACE_NAMES = (
    "gpu_residency:insertions",
    "gpu_residency:insertion_evictions",
    "gpu_residency:insertion_truncated",
)


def _add_insertions(entry: dict[str, Any], device: Mapping[str, list], row: int) -> None:
    """Report a layer's insert-on-miss copies with its decode counters; they are device rows, not migrations."""
    entry["insertions"] = entry.get("insertions", 0) + device["insertions"][row]
    entry["insertion_evictions"] = (
        entry.get("insertion_evictions", 0) + device["insertion_evictions"][row]
    )


class ExpertHotCacheManager:
    """Allocate a global slot budget and observe the recorder after each forward.

    Routing counts are borrowed synchronously, never retained or counted again.
    Placement and gathers must share the streamer's serialized CUDA stream.
    Startup may set each layer's `_nvfp4_file_source_bytes_per_expert` to enable
    exact file attribution; absent metadata is reported as unknown, not guessed.
    """

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        budget_bytes: int,
        seed_path: str | None,
        dynamic: bool,
        update_prefill_tokens: int,
        min_residence_forwards: int,
        benefit_ratio: float,
        log_interval: int = 100,
        metrics_path: str | os.PathLike[str] | None = None,
        route_history_limit: int = 32,
        copy_backend: str = "gpu",
        graph_gather_batch_size: int = 0,
        update_decode_forwards: int = 0,
        decay_tokens: int = 0,
        promotion_sigmas: float = 0.0,
        graph_gather_max_rows: int = 0,
        async_promotions: bool = False,
        async_residency_scores: bool = False,
        gpu_residency_update: bool = False,
        gpu_residency_max_promotions: int = 64,
        expert_doorbell: bool = False,
        doorbell_cpu_core: int = 71,
        doorbell_timeout_polls: int = 0,
        doorbell_degraded_polls: int = 0,
        doorbell_drain_polls: int = 0,
        doorbell_plan_capacity: int = 0,
        doorbell_fatal_wait_s: float = 30.0,
        insert_on_miss: bool | int | None = None,
        insert_on_miss_decay: float | None = None,
        fused_insert: bool | None = None,
    ) -> ExpertHotCacheManager | None:
        """Build the per-layer hot caches.

        ``async_promotions`` returns from a residency boundary once the
        promotion copies are submitted; their slots become hits on the first
        forward after the copies complete, and a layer with copies in flight
        skips later boundaries until they land.

        ``graph_gather_batch_size`` > 0 is the tokens of the largest graph
        forward; it reserves ``graph_gather_scratch_rows(tokens, top_k,
        graph_gather_max_rows)`` scratch rows per layer from the budget and
        enables each streamer's sync-free graph gather for routes of at most
        that many rows.
        ``update_decode_forwards`` > 0 also updates dynamic residency after every
        that many decode or speculative verify forwards, so a long decode is not
        served by the set the last long prefill chose.
        ``decay_tokens`` > 0 decays scores once per that many routed tokens
        instead of once per boundary, and ``promotion_sigmas`` adds that many
        standard deviations of count noise to the lead a promotion needs; see
        :class:`ExpertResidencyPolicy`.
        ``insert_on_miss`` is an :class:`InsertOnMissStage` and defaults with
        ``insert_on_miss_decay`` to ``SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE`` and its
        decay; see :class:`GpuResidencyUpdater`. Stage DIRECT reserves no
        graph-gather scratch at all, because its gathers copy into victim slots.
        ``fused_insert`` defaults to ``SGLANG_MOE_HOT_FUSED_INSERT`` and runs
        stage SCRATCH's boundary copies through the fused masked kernel; it is
        byte-identical to the index copy it replaces and only changes its cost.
        """
        if fused_insert is None:
            fused_insert = envs.SGLANG_MOE_HOT_FUSED_INSERT.get()
        fused_insert = bool(fused_insert)
        if insert_on_miss is None:
            insert_on_miss = envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get()
        insert_on_miss = int(insert_on_miss)
        if insert_on_miss not in tuple(InsertOnMissStage):
            raise ValueError(
                f"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE must be one of "
                f"{[int(stage) for stage in InsertOnMissStage]}, not {insert_on_miss}"
            )
        if insert_on_miss_decay is None:
            insert_on_miss_decay = envs.SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY.get()
        if insert_on_miss:
            if not gpu_residency_update:
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE requires SGLANG_MOE_GPU_RESIDENCY_UPDATE"
                )
            if index(update_decode_forwards) != 1:
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE requires SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1"
                )
            if not 0.0 < insert_on_miss_decay <= 1.0:
                raise ValueError(
                    "insert-on-miss decay (SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY) must be in (0, 1]"
                )
        if fused_insert and insert_on_miss != InsertOnMissStage.SCRATCH:
            # The fused kernel replaces stage SCRATCH's boundary copy loop and nothing else:
            # stage OFF has no such loop, and stage DIRECT lands its copies in the gather. A
            # run that sets this flag anywhere else would report a fused arm having measured
            # the unfused path, so refuse instead of accepting it as a no-op.
            raise ValueError(
                "SGLANG_MOE_HOT_FUSED_INSERT only applies to "
                f"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE={int(InsertOnMissStage.SCRATCH)} "
                f"({InsertOnMissStage.SCRATCH.name}), not stage {insert_on_miss}"
            )
        budget_bytes = index(budget_bytes)
        if budget_bytes == 0:
            return None
        update_prefill_tokens = index(update_prefill_tokens)
        update_decode_forwards = index(update_decode_forwards)
        decay_tokens = index(decay_tokens)
        min_residence_forwards = index(min_residence_forwards)
        log_interval = index(log_interval)
        route_history_limit = index(route_history_limit)
        if (
            budget_bytes < 0
            or update_prefill_tokens < 1
            or update_decode_forwards < 0
            or decay_tokens < 0
            or min_residence_forwards < 0
            or log_interval < 1
            or route_history_limit < 1
        ):
            raise ValueError("invalid expert hot cache budget or update interval")
        if not math.isfinite(benefit_ratio) or benefit_ratio < 0:
            raise ValueError(
                "expert hot cache benefit ratio must be finite and nonnegative"
            )
        if not math.isfinite(promotion_sigmas) or promotion_sigmas < 0:
            raise ValueError(
                "expert hot cache promotion sigmas must be finite and nonnegative"
            )
        streamers = {}
        if copy_backend not in ("gpu", "dma"):
            raise ValueError("expert copy backend must be gpu or dma")
        for streamer in iter_expert_streamers(model):
            layer_id = index(streamer.layer_id)
            if layer_id < 0 or layer_id in streamers:
                raise ValueError(
                    "expert hot cache requires unique nonnegative layer IDs"
                )
            streamers[layer_id] = streamer
        if not streamers:
            return None
        if index(graph_gather_batch_size) or gpu_residency_update or expert_doorbell:
            require_graph_gather_support(
                streamers.values(),
                pinned_tier_ok=not (gpu_residency_update or expert_doorbell),
            )
        seed = None
        if seed_path is not None:
            path = Path(seed_path)
            if path.suffix == ".json":
                with path.open() as source:
                    payload = json.load(source)
            else:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            seed = normalize_expert_frequency_seed(payload)
            if any(
                layer_id >= seed.shape[0] or seed.shape[1] != streamer.num_experts
                for layer_id, streamer in streamers.items()
            ):
                raise ValueError(
                    "expert frequency seed does not match model layers and experts"
                )
        candidates = sorted(
            (
                -(float(seed[layer_id, expert_id]) if seed is not None else 1.0)
                * streamer.bytes_per_expert,
                expert_id,
                layer_id,
            )
            for layer_id, streamer in streamers.items()
            for expert_id in range(streamer.num_experts)
        )
        graph_gather_batch_size = index(graph_gather_batch_size)
        if graph_gather_batch_size < 0:
            raise ValueError("graph gather batch size cannot be negative")
        gather_rows = {}
        for layer_id, streamer in streamers.items():
            top_k = getattr(streamer.layer, "top_k", None)
            if graph_gather_batch_size and top_k is None:
                raise ValueError("graph gather needs each streamed layer's top_k")
            gather_rows[layer_id] = (
                graph_gather_scratch_rows(
                    graph_gather_batch_size, top_k, graph_gather_max_rows
                )
                if graph_gather_batch_size
                else 0
            )
        # A DIRECT gather lands its misses in victim slots, so it needs the same route width
        # but none of the rows: those rows go back to the budget as residency.
        direct = insert_on_miss == InsertOnMissStage.DIRECT
        scratch_rows = {
            layer_id: 0 if direct else rows for layer_id, rows in gather_rows.items()
        }
        selected = {layer_id: [] for layer_id in streamers}
        pull_row_enabled = envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off"
        allocated_layers = {layer_id for layer_id, rows in gather_rows.items() if rows and not direct}
        remaining = budget_bytes - sum(
            rows * streamers[layer_id].bytes_per_expert
            for layer_id, rows in scratch_rows.items()
        )
        # Graph-gather scratch makes a cache allocation unconditional. Reserve its
        # dedicated side-pull row before considering any resident slots. For a
        # layer without scratch, its first resident slot pays for the same row
        # atomically below, so a selection can never overcommit the physical
        # cache allocation by one expert row.
        if pull_row_enabled:
            remaining -= sum(
                streamers[layer_id].bytes_per_expert for layer_id in allocated_layers
            )
        if remaining < 0:
            raise ValueError(
                "expert hot cache budget cannot hold the graph-gather scratch and pull rows"
            )
        # DIRECT refuses any layer holding fewer than twice its gather rows (see
        # `_init_insert_direct`), so a seed that scores one layer low would refuse the
        # whole budget. Give every layer that floor from its own best experts first; the
        # scores then spend the rest. When every layer clears the floor anyway, the
        # selection is unchanged: each layer's picks are its top-scored experts either way.
        floors = {
            layer_id: min(2 * rows, streamers[layer_id].num_experts) if direct else 0
            for layer_id, rows in gather_rows.items()
        }
        chosen = {layer_id: set() for layer_id in streamers}
        # A format with an inclusive pinned tier keeps every hot expert in host memory
        # too, so its layers hold at most `inclusive_hot_slot_limit` slots and the
        # budget they cannot use goes to other layers. Other formats have no limit.
        slot_limits = {
            layer_id: inclusive_hot_slot_limit(streamer)
            for layer_id, streamer in streamers.items()
        }
        clamped = set()
        for floor_pass in (True, False):
            for _, expert_id, layer_id in candidates:
                if expert_id in chosen[layer_id] or (
                    floor_pass and len(chosen[layer_id]) >= floors[layer_id]
                ):
                    continue
                limit = slot_limits[layer_id]
                if limit is not None and len(chosen[layer_id]) >= limit:
                    # This clamp is reachable during the floor pass only if a
                    # layer both has a floor (DIRECT, graph_gather_batch_size > 0)
                    # and an inclusive pinned tier's slot limit. That never
                    # happens today: DIRECT requires gpu_residency_update, under
                    # which require_graph_gather_support refuses pinned_tier
                    # formats (pinned_tier_ok=False), and no dense format sets
                    # inclusive_pinned_tier. A pinned_tier format (EXL3's
                    # inclusive tier) passes the support check only for the plain
                    # graph gather, which has no floor. Assert this instead of
                    # relying on it silently, since the clamp would otherwise
                    # cut into a layer's floor and DIRECT would refuse the budget.
                    assert not floor_pass or floors[layer_id] == 0
                    clamped.add(layer_id)
                    continue
                slot_bytes = streamers[layer_id].bytes_per_expert
                pull_row_bytes = (
                    slot_bytes if pull_row_enabled and layer_id not in allocated_layers else 0
                )
                if slot_bytes + pull_row_bytes <= remaining:
                    chosen[layer_id].add(expert_id)
                    remaining -= slot_bytes + pull_row_bytes
                    allocated_layers.add(layer_id)
        for _, expert_id, layer_id in candidates:
            if expert_id in chosen[layer_id]:
                selected[layer_id].append(expert_id)
        for layer_id in sorted(clamped):
            streamer = streamers[layer_id]
            logger.info(
                "Expert hot cache clamps layer %d to %d slots: its inclusive pinned "
                "tier holds %d rows and an eager gather stages up to %d more",
                layer_id,
                slot_limits[layer_id],
                streamer.pinned_host_cache.capacity,
                streamer.format.max_gather_rows or 0,
            )
        if not any(selected.values()) and not any(gather_rows.values()):
            return None
        manager = cls()
        manager.streamers = streamers
        manager.caches = {}
        manager.residency_policies = {}
        manager.dynamic = dynamic
        manager.update_prefill_tokens = update_prefill_tokens
        manager.update_decode_forwards = update_decode_forwards
        manager._boundary_clock = ResidencyBoundaryClock(
            update_prefill_tokens, update_decode_forwards, enabled=dynamic
        )
        manager.min_residence_forwards = min_residence_forwards
        manager.async_promotions = bool(async_promotions)
        manager._async_residency_scores = bool(async_residency_scores)
        manager._async_residency_backend = None
        manager._async_residency_buffers = None
        manager._async_residency_event = None
        manager._async_residency_pending = None
        manager._async_residency_refresh = None
        manager._inflight_promotions = []
        manager.deferred_residency_updates = 0
        manager.benefit_ratio = benefit_ratio
        manager.log_interval = log_interval
        manager.metrics_path = Path(metrics_path) if metrics_path else None
        manager.route_history_limit = route_history_limit
        manager._last_update = {layer_id: 0 for layer_id in streamers}
        manager._last_gather = {
            layer_id: streamer.last_gather_stats
            for layer_id, streamer in streamers.items()
        }
        manager._last_pinned_cache_stats = {
            layer_id: manager._pinned_cache_stats(streamer)
            for layer_id, streamer in streamers.items()
        }
        manager._counters = {
            mode: {layer_id: _OperationalCounters() for layer_id in streamers}
            for mode in ("prefill", "decode")
        }
        manager._counters["speculative"] = {
            layer_id: _OperationalCounters() for layer_id in streamers
        }
        manager._layer_ids = sorted(streamers)
        manager._layer_positions = {
            layer_id: position for position, layer_id in enumerate(manager._layer_ids)
        }
        manager._layer_index = {}
        manager._registers = {}
        # Route history and its dense affinity matrix are observer work, not
        # residency input.  Start them only when an on-disk trace consumes
        # them; next-layer prefetch enables both before its first forward.
        manager._collect_route_history = manager.metrics_path is not None
        manager._collect_affinity = manager.metrics_path is not None
        manager._gathered_zero_masks = {}
        manager._graph_counters = None
        manager._graph_unique_counters = None
        manager._side_pull_snapshots = {}
        manager._last_side_pull_totals = {}
        manager._last_observer_mode = None
        # Unlike `residency_policies`, a `PrefetchPuller` cannot be populated here:
        # it is built from this manager's own `caches` by `PrefetchScoring`, which
        # runs after `from_model` returns. See `register_prefetch_puller`.
        manager._prefetch_puller = None
        # The profiling-only calibration observer is also constructed after the
        # caches. It deliberately has a separate registration: profiling runs
        # use pull mode off and therefore have no PrefetchPuller to piggyback on.
        manager._prefetch_calibration = None
        # Trace schemas grow only when a new serving phase first creates its
        # device registers.  Keep an independent fixed-slot pool per schema so
        # adding that optional phase never waits for an older trace to drain.
        manager._trace_telemetry: dict[tuple[tuple[str, tuple[int, ...], str], ...], AsyncTelemetry] = {}
        manager._trace_write_condition = threading.Condition()
        manager._trace_sequence = 0
        manager._trace_next_write = 0
        # Inclusive sorted ranges keep a stalled earliest write from retaining
        # one Python object per later dropped trace.
        manager._trace_skipped: list[tuple[int, int]] = []
        manager._trace_ready: dict[int, tuple[Path, str]] = {}
        # Completed snapshots may arrive out of order across the fixed telemetry
        # pools.  Bound their host staging too: optional records drop rather
        # than accumulating behind one delayed D2H event or a slow disk.
        manager._trace_reorder_limit = 4
        manager._trace_flush_active = False
        manager._trace_drain_thread: threading.Thread | None = None
        if manager.metrics_path is not None:
            atexit.register(manager.close_telemetry)
        for layer_id, expert_ids in selected.items():
            if expert_ids or gather_rows[layer_id]:
                layer_streamer = streamers[layer_id]
                layer_streamer.expert_copy_backend = copy_backend
                cache = ExpertHotCache(
                    layer_streamer, len(expert_ids), scratch_rows[layer_id]
                )
                cache._owner_manager = manager
                manager.caches[layer_id] = cache
                manager._record_update(layer_id, cache.reassign(expert_ids))
                if dynamic:
                    policy = ExpertResidencyPolicy(
                        layer_streamer.num_experts,
                        cache.capacity,
                        device=cache.device,
                        promotion_margin=benefit_ratio,
                        decay_tokens=decay_tokens or None,
                        promotion_sigmas=promotion_sigmas,
                        initial_scores=(
                            seed[layer_id] if seed is not None else None
                        ),
                    )
                    layer_streamer.residency_policy = policy
                    manager.residency_policies[layer_id] = policy
        for layer_id, rows in gather_rows.items():
            if rows:
                streamers[layer_id].enable_graph_gather(
                    rows, scratch_destinations=not direct
                )
        manager._share_graph_counters()
        manager.gpu_residency = None
        if gpu_residency_update:
            if not dynamic:
                raise ValueError("GPU residency update requires dynamic residency")
            if async_residency_scores:
                raise ValueError(
                    "async CPU residency scores cannot run with GPU residency update"
                )
            from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

            manager.gpu_residency = GpuResidencyUpdater(
                manager,
                max_promotions=gpu_residency_max_promotions,
                insert_on_miss=int(insert_on_miss),
                insert_on_miss_decay=float(insert_on_miss_decay),
                fused_insert=fused_insert,
            )
        manager.doorbell = (
            manager._start_doorbell(
                doorbell_cpu_core,
                doorbell_timeout_polls,
                doorbell_degraded_polls,
                doorbell_drain_polls,
                doorbell_plan_capacity,
                doorbell_fatal_wait_s,
            )
            if expert_doorbell
            else None
        )
        if manager.gpu_residency is not None and manager.gpu_residency.insert_on_miss:
            manager.gpu_residency.check_miss_plans()
        devices = {cache.device for cache in manager.caches.values()}
        logger.info(
            "Expert hot cache startup %s",
            json.dumps(
                {
                    # The stage the process actually resolved, not the one a launcher meant to
                    # ask for: a matrix verifies this line, so an intent/behaviour mismatch
                    # (a retired alias, a typo) fails verification instead of a later number.
                    "insert_on_miss_stage": InsertOnMissStage(insert_on_miss).name,
                    "fused_insert": fused_insert,
                    "requested_bytes": budget_bytes,
                    "residency_bytes": manager.residency_bytes,
                    "allocation_bytes": manager.allocation_bytes,
                    "slots": sum(cache.capacity for cache in manager.caches.values()),
                    "scratch_bytes": sum(
                        cache.scratch_bytes for cache in manager.caches.values()
                    ),
                    "prefetch_pull_bytes": manager.prefetch_pull_bytes,
                    "layers": len(manager.caches),
                    "cuda_allocated_bytes": sum(
                        torch.cuda.memory_allocated(device) for device in devices
                    ),
                    "cuda_reserved_bytes": sum(
                        torch.cuda.memory_reserved(device) for device in devices
                    ),
                },
                sort_keys=True,
            ),
        )
        manager._attach_formats()
        return manager

    def enable_next_layer_prefetch(self, max_candidates: int) -> None:
        """Attach default-off sparse prefetch callbacks between adjacent layers."""
        max_candidates = index(max_candidates)
        if max_candidates <= 0:
            return
        # The prefetch policy consumes the route tables, including the dense
        # adjacent-layer affinity relation.  This happens during setup, before
        # any observer forward can allocate registers.
        self._collect_route_history = True
        self._collect_affinity = True
        policy = SparseNextLayerPolicy(max_candidates)
        self.prefetch_coordinators = {}
        layer_ids = sorted(self.streamers)
        for current_layer, next_layer in zip(layer_ids, layer_ids[1:]):
            cache = self.caches.get(next_layer)
            if cache is None:
                continue
            next_streamer = self.streamers[next_layer]
            if next_streamer.pinned_host_cache is None:
                continue
            coordinator = ExpertPrefetchCoordinator(enabled=True, device=cache.device)
            next_streamer.prefetch_coordinator = coordinator
            self.prefetch_coordinators[next_layer] = coordinator

            def schedule(
                source_ids,
                current_layer=current_layer,
                next_layer=next_layer,
                cache=cache,
                coordinator=coordinator,
            ):
                popularity, affinity = self._route_tables("decode", next_layer)
                resident = cache.resident_experts()
                candidates = policy.predict(
                    source_ids.tolist(), popularity, affinity, resident
                )
                placements = cache.prefetch_destinations(
                    candidates, coordinator.protected_slots
                )
                if not placements:
                    return
                predicted = tuple(expert for expert, _ in placements)
                destination_slots = tuple(slot for _, slot in placements)

                def submit(submitted):
                    if submitted != predicted:
                        raise ValueError("prefetch candidates changed after placement")
                    update = cache.assign_prefetch(placements)
                    self._record_update(next_layer, update)
                    coordinator.record_placement(
                        cache_pollution_bytes=update.migration_bytes,
                        evictions=update.evicted_experts,
                    )
                    return update.migration_bytes

                coordinator.launch(
                    predicted, protected_slots=destination_slots, submit=submit
                )

            self.streamers[current_layer].next_layer_prefetch = schedule
        if not self.prefetch_coordinators:
            raise ValueError("NVFP4 expert prefetch has no eligible adjacent layers")

    @property
    def residency_bytes(self) -> int:
        return sum(cache.capacity_bytes for cache in self.caches.values())

    @property
    def allocation_bytes(self) -> int:
        """Physical cache bytes, including graph scratch and the pull row."""
        return sum(cache.allocation_bytes for cache in self.caches.values())

    @property
    def prefetch_pull_bytes(self) -> int:
        return sum(cache.prefetch_pull_bytes for cache in self.caches.values())

    def _record_update(
        self, layer_id: int, update: HotCacheUpdateStats, phase: str = "prefill"
    ) -> None:
        counters = self._counters[phase][layer_id]
        counters.promotions += update.promoted_experts
        counters.evictions += update.evicted_experts
        counters.migration_bytes += update.migration_bytes
        submission = self.caches[layer_id].last_copy_submission
        if submission is None or update.promoted_experts == 0:
            return
        counters.requested_copy_backend = submission.requested_backend
        if counters.actual_copy_backend in (None, submission.actual_backend):
            counters.actual_copy_backend = submission.actual_backend
        else:
            counters.actual_copy_backend = "mixed"
        counters.copy_rows += submission.rows
        counters.copy_bytes += submission.bytes
        counters.copy_submissions += submission.submissions
        counters.copy_fallbacks += submission.fallbacks

    def _share_graph_counters(self) -> None:
        """Point every graph gather's device counters into one manager buffer.

        Captured gathers add to these fixed rows during replay; the manager
        moves them into phase registers each forward without reading them.
        """
        graph_streamers = [
            (position, self.streamers[layer_id])
            for position, layer_id in enumerate(self._layer_ids)
            if self.streamers[layer_id].graph_counters is not None
        ]
        if not graph_streamers:
            return
        device = graph_streamers[0][1].graph_counters.device
        shared = torch.zeros(
            (len(self._layer_ids), 2), dtype=torch.int64, device=device
        )
        unique_counters = torch.zeros_like(shared)
        for position, streamer in graph_streamers:
            shared[position].copy_(streamer.graph_counters)
            streamer.graph_counters = shared[position]
            unique_counters[position].copy_(streamer.graph_unique_counters)
            streamer.graph_unique_counters = unique_counters[position]
        self._graph_counters = shared
        self._graph_unique_counters = unique_counters

    def register_prefetch_puller(self, puller: Any) -> None:
        """Register the side-stream ``PrefetchPuller`` built from this manager's caches.

        Built later than this manager (``PrefetchScoring`` needs ``self.caches``
        first), so it cannot be populated internally at construction the way
        ``residency_policies`` is; call this once it exists, before the first
        ``discard_graph_capture_routes`` or metrics read that should see it.
        """
        self._prefetch_puller = puller

    def register_prefetch_calibration(self, calibration: Any) -> None:
        """Register the profiling observer so graph-capture warmup can be purged."""
        self._prefetch_calibration = calibration

    def discard_graph_capture_routes(self) -> None:
        """Drop routes and counters that CUDA-graph warmup and capture recorded.

        Captured gathers execute once while recording, so their dummy routes
        land in the residency counts, graph counters, and route registers like
        a real forward. A registered ``PrefetchPuller``'s ``PullDeliveryStats``
        gets the same treatment: its warmup replay posts and joins a dummy pull
        exactly like a real forward's, so it is purged here too.
        """
        if getattr(self, "_async_residency_pending", None) is not None:
            # Capture reset is outside serving. Keep the pinned buffer alive
            # until its queued D2H copy finishes, then discard both the captured
            # score decision and any newer boundary waiting to be resampled.
            self._async_residency_backend.synchronize(self._async_residency_event)
            self._async_residency_pending = None
        self._async_residency_refresh = None
        if self._graph_counters is not None:
            self._graph_counters.zero_()
            self._graph_unique_counters.zero_()
        for registers in self._registers.values():
            for register in registers.values():
                register.zero_()
        for policy in self.residency_policies.values():
            policy.pending_counts.zero_()
        puller = getattr(self, "_prefetch_puller", None)
        if puller is not None:
            if hasattr(puller, "discard_delivery_stats"):
                puller.discard_delivery_stats()
            else:
                for stats in puller.stats.values():
                    stats.counts.zero_()
        calibration = getattr(self, "_prefetch_calibration", None)
        if calibration is not None:
            calibration.reset()
        self._side_pull_snapshots.clear()
        self._last_side_pull_totals.clear()
        self.finish_promotions()
        if getattr(self, "gpu_residency", None) is not None:
            self.gpu_residency.reset_after_capture(self._boundary_clock)
        # Warmup and capture forwards replaced each layer's last gather stats and
        # admitted pinned rows. Start the eager counters from here, so the first
        # real forward counts only its own.
        for layer_id, streamer in self.streamers.items():
            self._last_gather[layer_id] = streamer.last_gather_stats
            self._last_pinned_cache_stats[layer_id] = self._pinned_cache_stats(streamer)

    def _start_doorbell(
        self,
        cpu_core: int,
        timeout_polls: int,
        degraded_polls: int,
        drain_polls: int,
        plan_capacity: int = 0,
        fatal_wait_s: float = 30.0,
    ) -> "ExpertDoorbellCopier":
        """Serve every graph-gather layer's host miss plans through one doorbell thread.

        Each layer becomes one target-layer tag of a shared ``DoorbellRowBackend``
        (``expert_row_plan``), posting and resolving its own static-capacity
        plan; rows the thread does not deliver are copied in-graph as the
        residual. The thread copies on a torch-created stream (a stream the
        thread creates itself is held behind CUDA-graph replays, E32) with one
        batched copy in stream order per request. ``plan_capacity`` 0 uses the
        scratch rows; a larger capacity is allowed, but plans still count at most
        the scratch rows. A zero resolve budget is sized to four times the
        largest per-layer miss copy at 8 GiB/s, at least 20 ms, and a zero
        degraded budget to twice that copy, at least 4096 polls, converting at
        256 ns per poll, the lower end of the 256-270 ns measured on the RTX 5090
        (E34), so each budget lasts at least its wall-time target. A zero drain
        budget is 524,288 polls (134-142 ms), one launch: a never-launched
        kernel's launch blocks until all queued device work finishes (E34f), so
        only the total queued wait bounds a stall. When a drain runs out the
        copier is disabled for the rest of the process and the resolve returns
        undelivered, served by the residual copy; ``doorbell_fail_stop_check``
        must run before the next forward and holds until the committed copies
        landed, and the copier's watchdog aborts the process with an ERROR if
        they have not landed ``fatal_wait_s`` later.

        The copier is primed before serving (``ExpertDoorbellCopier._prime``)
        into the first layer's first scratch row, which nothing reads before
        capture; the prime is empirically required for availability (E34 3g),
        not for correctness.
        """
        from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier
        from sglang.srt.layers.moe.expert_row_plan import (
            DoorbellRowBackend,
            ExpertRowPlan,
        )

        link_bytes_per_s = 8 * 1024**3
        poll_s = 256e-9
        served = [
            self.streamers[layer_id]
            for layer_id in self._layer_ids
            if self.streamers[layer_id].graph_gather_rows > 0
            and self.streamers[layer_id]._graph_row_segments is not None
        ]
        if not served:
            raise ValueError(
                "SGLANG_MOE_EXPERT_DOORBELL needs graph-gather layers with host rows"
            )
        capacities = {streamer.graph_gather_rows for streamer in served}
        if len(capacities) != 1:
            raise ValueError("doorbell layers must share one graph-gather row count")
        scratch_rows = capacities.pop()
        if plan_capacity < 0:
            raise ValueError("SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY cannot be negative")
        capacity = max(plan_capacity, scratch_rows)
        largest_copy_s = (
            scratch_rows
            * max(streamer.host_bytes_per_expert for streamer in served)
            / link_bytes_per_s
        )
        timeout_polls = timeout_polls or int(max(0.02, 4 * largest_copy_s) / poll_s) + 1
        degraded_polls = degraded_polls or max(4096, int(2 * largest_copy_s / poll_s) + 1)
        drain_polls = drain_polls or 524_288
        device = served[0].hot_cache.device
        copier = ExpertDoorbellCopier(
            [streamer._graph_row_segments for streamer in served],
            capacity,
            ring=max(64, 2 * len(served)),
            max_tags=len(served),
            cpu_core=cpu_core,
            timeout_polls=timeout_polls,
            degraded_polls=degraded_polls,
            drain_polls=drain_polls,
            fatal_wait_s=fatal_wait_s,
            copy_api="batch",
            src_access_order="stream",
            stream=torch.cuda.Stream(device),
            prime_slot=served[0].hot_cache.capacity,
        )
        backend = DoorbellRowBackend(
            copier,
            {tag: streamer._graph_row_segments for tag, streamer in enumerate(served)},
        )
        for tag, streamer in enumerate(served):
            streamer.row_tag = tag
            streamer.row_backend = backend
            if capacity != streamer.row_plan.capacity:
                cache = streamer.hot_cache
                streamer.row_plan = ExpertRowPlan.for_scratch(
                    capacity, cache.capacity, scratch_rows, device
                )
        self._doorbell_disabled_logged = False
        spin_cpu = copier.stats()["spin_cpu"]
        if cpu_core >= 0 and spin_cpu != cpu_core:
            logger.warning(
                "Expert doorbell thread asked for CPU %d but runs on CPU %d; "
                "check the process CPU affinity",
                cpu_core,
                spin_cpu,
            )
        logger.info(
            "Expert doorbell startup %s",
            json.dumps(
                {
                    "layers": len(served),
                    "backend": backend.name,
                    "mode": "current",
                    "plan_capacity_rows": capacity,
                    "scratch_rows": scratch_rows,
                    "cpu_core": cpu_core,
                    "timeout_polls": timeout_polls,
                    "degraded_polls": degraded_polls,
                    "drain_polls": drain_polls,
                    "fatal_wait_s": fatal_wait_s,
                    "largest_copy_bytes": int(largest_copy_s * link_bytes_per_s),
                    "prime_s": round(copier.prime_s, 4),
                    "spin_cpu": spin_cpu,
                },
                sort_keys=True,
            ),
        )
        return copier

    def quiesce_doorbell(self) -> None:
        """Drain and pause the doorbell thread before a CUDA graph capture."""
        if getattr(self, "doorbell", None) is not None:
            self.doorbell.quiesce()

    def resume_doorbell(self) -> None:
        """Clear capture-time wait state and let the doorbell thread service again."""
        if getattr(self, "doorbell", None) is not None:
            self.doorbell.reset_wait_state()
            self.doorbell.resume()

    def stop_doorbell(self) -> None:
        """Drain and join the doorbell threads; idempotent.

        Called from the scheduler's graceful shutdown before host resources are
        released, so the process never reaches interpreter teardown with a running
        copier thread (that ended in std::terminate and a scheduler stuck in the
        driver). The copier module's atexit hook covers other exit paths.
        """
        if getattr(self, "doorbell", None) is not None:
            self.doorbell.stop()

    def register_fail_stop_check(self, check: Callable[[], None]) -> None:
        """Run ``check`` after every batch result, before the doorbell's own check.

        A check raises to stop the process; it must not synchronize the device.
        """
        if not hasattr(self, "fail_stop_checks"):
            self.fail_stop_checks = []
        self.fail_stop_checks.append(check)

    def add_residency_listener(
        self, listener: Callable[[int, list[int]], None]
    ) -> None:
        """Call ``listener(layer_id, slot_to_expert)`` after startup and every residency update.

        The update is committed before listeners run. A listener must not raise
        for a recoverable condition: a raise skips the remaining listeners and
        layers and propagates out of the forward observer, stopping the process.
        Listeners are not called by the GPU residency updater, so
        ``_attach_formats`` refuses them when it is on.
        """
        if not hasattr(self, "residency_listeners"):
            self.residency_listeners = []
        self.residency_listeners.append(listener)

    def _notify_residency_listeners(self) -> None:
        """Push every layer's hot slots to each listener; a listener's raise propagates."""
        for listener in getattr(self, "residency_listeners", ()):
            for layer_id, cache in self.caches.items():
                listener(layer_id, list(cache.slot_to_expert))

    def _attach_formats(self) -> None:
        """Offer the finished manager to each streamed format that asks for it
        (``attach_hot_cache_manager(manager, streamer)``), then push residency once.

        Runs once per manager; later calls return at once, so hooks never
        register their checks and listeners twice.
        """
        if getattr(self, "_formats_attached", False):
            return
        self._formats_attached = True
        for streamer in self.streamers.values():
            hook = getattr(streamer.format, "attach_hot_cache_manager", None)
            if hook is not None:
                hook(self, streamer)
        if (
            getattr(self, "gpu_residency", None) is not None
            and getattr(self, "residency_listeners", None)
        ):
            raise ValueError(
                "a format registered a residency listener, but the GPU residency "
                "updater changes residency without notifying listeners"
            )
        self._notify_residency_listeners()

    def doorbell_fail_stop_check(self, synchronize: bool = False) -> float:
        """Hold until every committed doorbell copy a drain gave up on has landed.

        The scheduler calls this after each forward's results are processed and
        before the next forward, in every forward mode, since any forward with a
        1-token gather posts to the doorbell, and it passes ``synchronize=True``.
        That synchronizes the current stream unconditionally: the fatal word is
        written by device kernels, and a replayed CUDA graph posts from the device
        without entering Python, so a "did a post run" flag would read False on
        exactly the steps that posted. With no exhausted drain it then reads two
        host words and returns 0.0 without a device-wide synchronize; see
        ``ExpertDoorbellCopier.fail_stop_check``.
        """
        for check in getattr(self, "fail_stop_checks", ()):
            check()
        doorbell = getattr(self, "doorbell", None)
        if doorbell is None:
            return 0.0
        return doorbell.fail_stop_check(synchronize=synchronize)

    def _log_doorbell(self) -> None:
        doorbell = getattr(self, "doorbell", None)
        if doorbell is None:
            return
        stats = doorbell.stats()
        if stats.get("disabled") and not getattr(self, "_doorbell_disabled_logged", False):
            self._doorbell_disabled_logged = True
            logger.warning(
                "Expert doorbell disabled after a drain ran out (%d drain timeouts): "
                "every later miss copy runs in-graph for the rest of the process",
                stats.get("drain_timeouts", 0),
            )
        logger.info(
            "Expert doorbell %s",
            json.dumps(
                {
                    key: stats.get(key)
                    for key in (
                        "running",
                        "disabled",
                        "disabled_posts",
                        "discarded_disabled",
                        "fatal_seq",
                        "completed_seq",
                        "posted",
                        "waits",
                        "timeouts",
                        "drains",
                        "drain_timeouts",
                        "degraded",
                        "record_mismatches",
                        "serviced",
                        "skipped_abandoned",
                        "skipped_overrun",
                        "invalid_records",
                        "copy_errors",
                        "last_copy_error",
                        "rows_copied",
                        "late_completions",
                    )
                },
                sort_keys=True,
            ),
        )

    def finish_promotions(self) -> None:
        """Publish every in-flight promotion, ordering the current stream behind its copies.

        The host does not block; slots are published on the device after the
        copies. Call before shutdown or anything that must see a settled cache.
        """
        self._publish_completed_promotions(wait=True)

    def _publish_completed_promotions(self, wait: bool) -> None:
        """Publish in-flight promotions whose copies landed, or all of them with ``wait``."""
        pending = []
        for ticket, promotions in self._inflight_promotions:
            cache = promotions[0].cache
            executor = cache._transfer_executor
            if not executor.has_completed(ticket):
                if not wait:
                    pending.append((ticket, promotions))
                    continue
                executor.wait(ticket, torch.cuda.current_stream(cache.device))
            for promotion in promotions:
                promotion.cache.complete_promotion(promotion)
        self._inflight_promotions = pending

    def _update_residency(self, boundary_tokens: int | None, mode: str) -> None:
        """Advance every layer's scores, decide together and copy all promotions at once.

        Scores advance by fused launches. The ordinary path reads them on the
        host at this boundary; the async path queues a pinned snapshot and
        applies its decision after a later event query. Promotion copies are
        ordered behind one transfer ticket in either path.
        """
        clock = self._boundary_clock
        layers = [
            (layer_id, self.caches[layer_id], self.residency_policies[layer_id])
            for layer_id in self._layer_ids
            if layer_id in self.caches and layer_id in self.residency_policies
        ]
        advance_residency_policies([policy for _, _, policy in layers], boundary_tokens)
        if getattr(self, "_async_residency_scores", False):
            if self._async_residency_pending is None:
                self._enqueue_residency_scores(mode, clock.forwards)
            else:
                # Every boundary still advances device scores. A single owned
                # pinned buffer stays with the earlier copy until it is used;
                # the later boundary is then resampled from the latest scores.
                self._async_residency_refresh = (mode, clock.forwards)
            return
        self._apply_residency_decisions(layers, mode, clock.forwards)

    def _enqueue_residency_scores(self, mode: str, boundary_forward: int) -> None:
        policies = [
            self.residency_policies[layer_id]
            for layer_id in self._layer_ids
            if layer_id in self.caches and layer_id in self.residency_policies
        ]
        if not policies:
            return
        scores = torch.stack([policy._scores for policy in policies])
        backend = self._async_residency_backend
        if backend is None:
            backend = TorchTelemetryBackend({"scores": scores})
            self._async_residency_backend = backend
            self._async_residency_buffers = backend.allocate({"scores": scores})
            self._async_residency_event = backend.event()
        backend.enqueue(
            self._async_residency_buffers, {"scores": scores}, self._async_residency_event
        )
        self._async_residency_pending = (mode, boundary_forward)

    def _poll_async_residency_scores(self) -> None:
        pending = self._async_residency_pending
        if pending is None or not self._async_residency_backend.ready(
            self._async_residency_event
        ):
            return
        mode, boundary_forward = pending
        layers = [
            (layer_id, self.caches[layer_id], self.residency_policies[layer_id])
            for layer_id in self._layer_ids
            if layer_id in self.caches and layer_id in self.residency_policies
        ]
        score_rows = list(self._async_residency_buffers["scores"].unbind(0))
        self._async_residency_pending = None
        self._apply_residency_decisions(layers, mode, boundary_forward, score_rows)
        self._notify_residency_listeners()
        refresh = self._async_residency_refresh
        self._async_residency_refresh = None
        if refresh is not None:
            self._enqueue_residency_scores(*refresh)

    def _apply_residency_decisions(
        self,
        layers: list[tuple[int, ExpertHotCache, ExpertResidencyPolicy]],
        mode: str,
        boundary_forward: int,
        score_rows: list[torch.Tensor] | None = None,
    ) -> None:
        deciding = []
        deciding_scores = []
        for position, layer in enumerate(layers):
            layer_id, cache, _ = layer
            if boundary_forward - self._last_update[layer_id] < self.min_residence_forwards:
                continue
            if cache.promotion_in_flight is not None:
                self.deferred_residency_updates += 1
                continue
            deciding.append(layer)
            if score_rows is not None:
                deciding_scores.append(score_rows[position])
        policies = [policy for _, _, policy in deciding]
        residents = [cache.resident_experts() for _, cache, _ in deciding]
        decisions = (
            decide_residency_policies(policies, residents)
            if score_rows is None
            else decide_residency_policies_from_host_scores(
                policies, residents, deciding_scores
            )
        )
        staged = []
        try:
            for (layer_id, cache, policy), decision in zip(deciding, decisions):
                if not decision.promotions and not decision.evictions:
                    continue
                policy.schedule_transfers(exact_demand=(), decision=decision)
                update, promotion = cache.stage_reassign(
                    decision.desired_experts, publish=self.async_promotions
                )
                self._last_update[layer_id] = boundary_forward
                if promotion is None:
                    self._record_update(layer_id, update, mode)
                else:
                    staged.append((layer_id, update, promotion))
        except BaseException:
            for _, _, promotion in staged:
                promotion.cache.abort_promotion(promotion)
            raise
        if not staged:
            return
        promotions = [promotion for _, _, promotion in staged]
        current_stream = torch.cuda.current_stream(promotions[0].cache.device)
        ticket = submit_hot_cache_promotions(promotions, producer_stream=current_stream)
        for layer_id, update, _ in staged:
            self._record_update(layer_id, update, mode)
        if self.async_promotions:
            self._inflight_promotions.append((ticket, promotions))
            return
        promotions[0].cache._transfer_executor.wait(ticket, current_stream)
        for promotion in promotions:
            promotion.cache.complete_promotion(promotion)

    @staticmethod
    def _pinned_cache_stats(streamer: ExpertStreamer) -> tuple[int, int]:
        cache = streamer.pinned_host_cache
        if cache is None:
            return (0, 0)
        return (cache.stats.populated_rows, cache.stats.evictions)

    def _registers_for(
        self, phase: str, device: torch.device, experts: int
    ) -> dict[str, torch.Tensor]:
        registers = self._registers.get(phase)
        if registers is None:
            layers = len(self._layer_ids)
            registers = {
                "popularity": torch.zeros(
                    (layers if self._collect_route_history else 0, experts),
                    dtype=torch.float64,
                    device=device,
                ),
                "affinity": torch.zeros(
                    (max(layers - 1, 0) if self._collect_affinity else 0, experts, experts),
                    dtype=torch.float32,
                    device=device,
                ),
                "unique_experts": torch.zeros(layers, dtype=torch.int64, device=device),
                "graph_rows": torch.zeros((layers, 2), dtype=torch.int64, device=device),
                "graph_unique_rows": torch.zeros(
                    (layers, 2), dtype=torch.int64, device=device
                ),
                "gathers": torch.zeros(layers, dtype=torch.int64, device=device),
            }
            self._registers[phase] = registers
        elif self._collect_route_history and not registers["popularity"].shape[0]:
            layers = len(self._layer_ids)
            registers["popularity"] = torch.zeros(
                (layers, experts), dtype=torch.float64, device=device
            )
            if self._collect_affinity:
                registers["affinity"] = torch.zeros(
                    (max(layers - 1, 0), experts, experts),
                    dtype=torch.float32,
                    device=device,
                )
        elif self._collect_affinity and not registers["affinity"].shape[0]:
            layers = len(self._layer_ids)
            registers["affinity"] = torch.zeros(
                (max(layers - 1, 0), experts, experts), dtype=torch.float32, device=device
            )
        return registers

    def _accumulate_registers(
        self, phase: str, counts: torch.Tensor, eager_gathered: list[bool]
    ) -> None:
        """Add this forward's routes and graph gather counts to device registers.

        Nothing here reads a device value on the host, so decode forwards pay no
        synchronization for metrics; registers are read only by snapshots.
        """
        device = (
            self._graph_counters.device
            if self._graph_counters is not None
            else counts.device
        )
        registers = self._registers_for(phase, device, counts.shape[1])
        index = self._layer_index.get(device)
        if index is None:
            index = torch.tensor(self._layer_ids, dtype=torch.long, device=device)
            self._layer_index[device] = index
        rows = counts.detach().to(device=device, non_blocking=True).index_select(0, index)
        if self._collect_route_history:
            registers["popularity"].add_(rows)
        if self._collect_affinity and registers["affinity"].shape[0]:
            routed = rows.to(torch.float32)
            registers["affinity"].baddbmm_(
                routed[:-1].unsqueeze(2), routed[1:].unsqueeze(1)
            )
        if any(eager_gathered):
            # Eager fallback owns a host list, so retain its one-shot pinned
            # transfer. Reusing that host storage without an ownership event
            # would race the preceding asynchronous H2D copy.
            gathered = torch.tensor(eager_gathered, dtype=torch.bool)
            if device.type == "cuda":
                gathered = gathered.pin_memory().to(device, non_blocking=True)
        elif self._graph_counters is not None:
            # Graph replay already maintains this device-visible truth. This
            # common path avoids the per-forward pinned mask and device temp.
            gathered = self._gathered_zero_masks.get(device)
            if gathered is None:
                gathered = torch.empty(len(self._layer_ids), dtype=torch.bool, device=device)
                self._gathered_zero_masks[device] = gathered
            torch.gt(self._graph_counters[:, 0], 0, out=gathered)
        else:
            gathered = self._gathered_zero_masks.get(device)
            if gathered is None:
                gathered = torch.zeros(len(self._layer_ids), dtype=torch.bool, device=device)
                self._gathered_zero_masks[device] = gathered
            else:
                gathered.zero_()
        if self._graph_counters is not None:
            if any(eager_gathered):
                gathered.logical_or_(self._graph_counters[:, 0] > 0)
            registers["graph_rows"].add_(self._graph_counters)
            registers["graph_unique_rows"].add_(self._graph_unique_counters)
            self._graph_counters.zero_()
            self._graph_unique_counters.zero_()
        registers["unique_experts"].add_(rows.ne(0).sum(dim=1) * gathered)
        registers["gathers"].add_(gathered)

    def _top_entries(self, matrix: torch.Tensor) -> list[list[tuple[int, float]]]:
        """Nonzero top ``route_history_limit`` (index, value) pairs of each row."""
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return [[] for _ in range(matrix.shape[0])]
        values, indices = matrix.topk(min(self.route_history_limit, matrix.shape[1]))
        entries = []
        for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
            row = [
                (index, value)
                for index, value in zip(row_indices, row_values)
                if value > 0
            ]
            row.sort(key=lambda item: (-item[1], item[0]))
            entries.append(row)
        return entries

    def _route_tables(
        self, phase: str, next_layer: int
    ) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        """Popularity of ``next_layer`` and affinity from the layer before it."""
        registers = self._registers.get(phase)
        position = self._layer_positions[next_layer]
        if registers is None or position == 0:
            return {}, {}
        experts = registers["popularity"].shape[1]
        (popularity,) = self._top_entries(
            registers["popularity"][position : position + 1]
        )
        (affinity,) = self._top_entries(
            registers["affinity"][position - 1 : position].flatten(1)
        )
        return dict(popularity), {
            (flat // experts, flat % experts): value for flat, value in affinity
        }

    def snapshot_route_statistics(self) -> dict[str, dict[str, dict[str, list]]]:
        """Top ``route_history_limit`` popularity and affinity entries per phase."""
        result = {}
        pairs = list(zip(self._layer_ids, self._layer_ids[1:]))
        for phase in self._counters:
            registers = self._registers.get(phase)
            if registers is None or not registers["popularity"].shape[0]:
                result[phase] = {
                    "popularity": {str(layer_id): [] for layer_id in self._layer_ids},
                    "affinity": {},
                }
                continue
            experts = registers["popularity"].shape[1]
            popularity = {
                str(layer_id): [[index, value] for index, value in entries]
                for layer_id, entries in zip(
                    self._layer_ids, self._top_entries(registers["popularity"])
                )
            }
            affinity = {
                f"{source}->{target}": [
                    [flat // experts, flat % experts, value] for flat, value in entries
                ]
                for (source, target), entries in zip(
                    pairs, self._top_entries(registers["affinity"].flatten(1))
                )
            }
            result[phase] = {"popularity": popularity, "affinity": affinity}
        return result

    def _trace_sources(self) -> dict[str, torch.Tensor]:
        """Live device buffers copied as one stream-ordered trace snapshot."""
        sources: dict[str, torch.Tensor] = {}
        for phase, registers in self._registers.items():
            for name, tensor in registers.items():
                sources[f"register:{phase}:{name}"] = tensor
        updater = getattr(self, "gpu_residency", None)
        if updater is not None:
            sources.update(
                {
                    "gpu_residency:promotions": updater.promotions,
                    "gpu_residency:evictions": updater.evictions,
                    "gpu_residency:boundary_updates": updater.boundary_updates,
                    "gpu_residency:truncated_layers": updater.truncated,
                }
            )
            # Through the same accessor snapshot() uses: this is the path that fills
            # hot-cache.metrics.jsonl, and the stages hold these counters in different tensors.
            sources.update(
                {
                    f"gpu_residency:{name}": counter
                    for name, counter in updater.insertion_counters().items()
                }
            )
        return sources

    def _trace_metadata(
        self, phase: str, telemetry: AsyncTelemetry, sequence: int
    ) -> dict[str, Any]:
        """Immutable host state paired with a trace's immutable device buffers."""
        counters = {
            mode: {
                str(layer_id): asdict(values)
                for layer_id, values in layers.items()
            }
            for mode, layers in self._counters.items()
        }
        coordinators = getattr(self, "prefetch_coordinators", {})
        policies = getattr(self, "residency_policies", {})
        metadata: dict[str, Any] = {
            "path": self.metrics_path,
            "timestamp_ns": time.time_ns(),
            "phase": phase,
            "sequence": sequence,
            "counters": counters,
            "side_pull_snapshots": copy.deepcopy(self._side_pull_snapshots),
            "prefetch": {
                str(layer_id): coordinator.snapshot_stats()
                for layer_id, coordinator in coordinators.items()
            },
            "residency_policy": {
                str(layer_id): policy.snapshot_metrics()
                for layer_id, policy in policies.items()
            },
            "telemetry": telemetry.stats(),
        }
        if self.async_promotions:
            metadata["residency_async"] = {
                "deferred_updates": self.deferred_residency_updates,
                "inflight_submissions": len(self._inflight_promotions),
            }
        doorbell = getattr(self, "doorbell", None)
        if doorbell is not None:
            metadata["doorbell"] = doorbell.stats()
        updater = getattr(self, "gpu_residency", None)
        if updater is not None:
            metadata["gpu_residency_layers"] = tuple(updater.layer_ids)
            metadata["gpu_residency_insert_on_miss"] = updater.insert_on_miss
        return metadata

    def _schedule_trace(self, phase: str) -> None:
        """Queue optional trace telemetry without waiting or performing file I/O."""
        if self.metrics_path is None:
            return
        self._refresh_side_pull_delivery()
        sources = self._trace_sources()
        if not sources:
            return
        key = tuple(
            sorted(
                (name, tuple(tensor.shape), str(tensor.dtype))
                for name, tensor in sources.items()
            )
        )
        telemetry = self._trace_telemetry.get(key)
        if telemetry is None:
            telemetry = AsyncTelemetry(
                sources=sources,
                backend=TorchTelemetryBackend(sources),
                writer=self._write_trace_snapshot,
                thread_name="moe-hot-cache-trace",
            )
            self._trace_telemetry[key] = telemetry
        sequence = self._reserve_trace_sequence()
        if sequence is None:
            return
        if not telemetry.schedule(sources, self._trace_metadata(phase, telemetry, sequence)):
            self._skip_trace_sequence(sequence)

    def close_telemetry(self) -> None:
        """Flush optional traces after serving has stopped; never call on a forward."""
        telemetry = tuple(getattr(self, "_trace_telemetry", {}).values())
        for writer in telemetry:
            writer.prepare_close()
        for writer in telemetry:
            writer.finish_close()
        with self._trace_write_condition:
            drain_thread = self._trace_drain_thread
        if drain_thread is not None and drain_thread is not threading.current_thread():
            drain_thread.join()

    def _skip_trace_sequence(self, sequence: int) -> None:
        with self._trace_write_condition:
            self._mark_trace_sequence_skipped(sequence)
            self._advance_trace_sequence()
            if self._trace_ready and not self._trace_flush_active:
                self._trace_flush_active = True
                self._trace_drain_thread = threading.Thread(
                    target=self._drain_trace_records,
                    name="moe-hot-cache-trace-drain",
                    daemon=True,
                )
                self._trace_drain_thread.start()
            self._trace_write_condition.notify_all()

    def _reserve_trace_sequence(self) -> int | None:
        """Reserve one bounded ordered-trace slot, or drop optional telemetry.

        The window includes every accepted sequence after ``_trace_next_write``:
        device copies still pending, records staged for the writer, and skipped
        sequence ranges.  Capping it here prevents a blocked earliest write
        from turning later dropped samples into unbounded skip bookkeeping.
        """
        with self._trace_write_condition:
            self._advance_trace_sequence()
            if (
                self._trace_sequence - self._trace_next_write
                >= self._trace_reorder_limit
            ):
                return None
            sequence = self._trace_sequence
            self._trace_sequence += 1
            return sequence

    def _advance_trace_sequence(self) -> None:
        while (
            self._trace_skipped
            and self._trace_skipped[0][0] <= self._trace_next_write
        ):
            _start, end = self._trace_skipped.pop(0)
            if end >= self._trace_next_write:
                self._trace_next_write = end + 1

    def _mark_trace_sequence_skipped(self, sequence: int) -> None:
        """Add one skipped sequence to the sorted, inclusive skip ranges.

        Callers hold ``_trace_write_condition``.  A completed snapshot can be
        dropped out of order, so merge both preceding and following ranges.
        """
        ranges = self._trace_skipped
        position = 0
        while position < len(ranges) and ranges[position][1] + 1 < sequence:
            position += 1
        start = end = sequence
        if position and ranges[position - 1][1] + 1 >= start:
            start = ranges[position - 1][0]
            end = max(end, ranges[position - 1][1])
            position -= 1
            del ranges[position]
        while position < len(ranges) and ranges[position][0] <= end + 1:
            start = min(start, ranges[position][0])
            end = max(end, ranges[position][1])
            del ranges[position]
        ranges.insert(position, (start, end))

    def _write_trace_snapshot(
        self, buffers: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
    ) -> None:
        """Build and append a trace from writer-owned CPU tensors only."""
        sequence = metadata["sequence"]
        try:
            trace = {
                "timestamp_ns": metadata["timestamp_ns"],
                "phase": metadata["phase"],
                "counters": self._trace_counters_from_host(buffers, metadata),
                "route_statistics": self._trace_routes_from_host(buffers),
                "telemetry": metadata["telemetry"],
            }
            encoded = json.dumps(trace, sort_keys=True) + "\n"
        except Exception:
            # Formatting is optional work too.  Release this sequence so a
            # later completed snapshot never waits behind a bad record.
            logger.exception("Could not format optional telemetry trace")
            self._skip_trace_sequence(sequence)
            return

        with self._trace_write_condition:
            self._advance_trace_sequence()
            if len(self._trace_ready) >= self._trace_reorder_limit:
                if sequence != self._trace_next_write:
                    self._mark_trace_sequence_skipped(sequence)
                    self._advance_trace_sequence()
                    self._trace_write_condition.notify_all()
                    return
                # Prefer the earliest sequence needed to make progress over
                # the farthest completed record when the reorder buffer fills.
                evicted = max(self._trace_ready)
                del self._trace_ready[evicted]
                self._mark_trace_sequence_skipped(evicted)
            self._trace_ready[sequence] = (metadata["path"], encoded)
            if self._trace_flush_active:
                return
            self._trace_flush_active = True

        # Exactly one background worker drains ready records.  The condition
        # protects only sequence state; JSON construction and file I/O happen
        # outside it, so an inference-thread drop never waits on a slow disk.
        self._drain_trace_records()

    def _drain_trace_records(self) -> None:
        """Write staged trace records in sequence order without owning state locks."""
        while True:
            with self._trace_write_condition:
                self._advance_trace_sequence()
                record = self._trace_ready.pop(self._trace_next_write, None)
                if record is None:
                    self._trace_flush_active = False
                    if self._trace_drain_thread is threading.current_thread():
                        self._trace_drain_thread = None
                    return
            path, encoded = record
            try:
                with path.open("a", encoding="utf-8") as destination:
                    destination.write(encoded)
            except Exception:
                logger.exception("Could not write optional telemetry trace")
            finally:
                with self._trace_write_condition:
                    self._trace_next_write += 1
                    self._advance_trace_sequence()
                    self._trace_write_condition.notify_all()

    def _trace_counters_from_host(
        self, buffers: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Apply completed device counter snapshots to their accepted host bases."""
        result = copy.deepcopy(metadata["counters"])
        for mode, layers in result.items():
            prefix = f"register:{mode}:"
            graph_rows = buffers.get(prefix + "graph_rows")
            graph_unique = (
                buffers[prefix + "graph_unique_rows"].tolist()
                if graph_rows is not None
                else None
            )
            requested_unique = (
                buffers[prefix + "unique_experts"].tolist()
                if graph_rows is not None
                else None
            )
            gathers = buffers[prefix + "gathers"].tolist() if graph_rows is not None else None
            rows = graph_rows.tolist() if graph_rows is not None else None
            for position, layer_id in enumerate(self._layer_ids):
                row = layers[str(layer_id)]
                streamer = self.streamers[layer_id]
                if graph_rows is not None:
                    routed, routed_missed = rows[position]
                    unique_hit, unique_missed = graph_unique[position]
                    row["requested_rows"] += unique_hit + unique_missed
                    row["routed_rows"] += routed
                    row["miss_rows"] += unique_missed
                    row["routed_miss_rows"] += routed_missed
                    row["unique_miss_rows"] += unique_missed
                    row["gathers"] += gathers[position]
                    row["hot_hits"] += unique_hit
                    row["d2d_bytes"] += routed * (
                        streamer.bytes_per_expert - streamer.host_bytes_per_expert
                    )
                    row["h2d_bytes"] += unique_missed * streamer.host_bytes_per_expert
                    row["backing_source_bytes"] += unique_missed * streamer.bytes_per_expert
                    row["requested_unique_experts"] += requested_unique[position]
                    side_pull = metadata["side_pull_snapshots"].get((mode, layer_id))
                    if side_pull is not None:
                        covered, residual, wasted, posted = side_pull
                        useful_posts = posted - wasted
                        covered_demand_rows = min(useful_posts, unique_missed)
                        row["miss_rows"] -= covered_demand_rows
                        row["h2d_bytes"] -= covered_demand_rows * streamer.host_bytes_per_expert
                        row["backing_source_bytes"] -= covered_demand_rows * streamer.bytes_per_expert
                        row["miss_rows"] += posted
                        row["side_pull_rows"] += posted
                        row["side_pull_bytes"] += posted * streamer.bytes_per_expert
                        row["side_pull_h2d_bytes"] = posted * streamer.host_bytes_per_expert
                        row["side_pull_d2d_bytes"] = posted * (
                            streamer.bytes_per_expert - streamer.host_bytes_per_expert
                        )
                        row["residual_demand_rows"] = unique_missed - covered_demand_rows
                        row["residual_demand_h2d_bytes"] = (
                            row["residual_demand_rows"] * streamer.host_bytes_per_expert
                        )
                        row["side_pull_posted_rows"] = posted
                        row["side_pull_useful_posts"] = useful_posts
                        row["side_pull_wasted_rows"] = wasted
                        row["side_pull_covered_routes"] = covered
                        row["side_pull_residual_routes"] = residual
                        row["side_pull_useful_precision"] = (
                            (posted - wasted) / posted if posted else 0.0
                        )
                cache = self.caches.get(layer_id)
                row["residency_bytes"] = cache.capacity_bytes if cache else 0
                row["allocation_bytes"] = cache.allocation_bytes if cache else 0
                row["scratch_bytes"] = cache.scratch_bytes if cache else 0
                row["prefetch_pull_bytes"] = cache.prefetch_pull_bytes if cache else 0
        if metadata["prefetch"]:
            result["prefetch"] = metadata["prefetch"]
        if metadata["residency_policy"]:
            result["residency_policy"] = metadata["residency_policy"]
        if "residency_async" in metadata:
            result["residency_async"] = metadata["residency_async"]
        if "doorbell" in metadata:
            result["doorbell"] = metadata["doorbell"]
        if "gpu_residency_layers" in metadata:
            device = {}
            for name in (
                "gpu_residency:promotions",
                "gpu_residency:evictions",
                "gpu_residency:boundary_updates",
                "gpu_residency:truncated_layers",
            ) + (_INSERTION_TRACE_NAMES if metadata["gpu_residency_insert_on_miss"] else ()):
                device[name.rsplit(":", 1)[-1]] = buffers[name].tolist()
            result["residency_gpu"] = device
            for row, layer_id in enumerate(metadata["gpu_residency_layers"]):
                bytes_per_expert = self.caches[layer_id].bytes_per_expert
                for phase, mode in enumerate(("decode", "prefill")):
                    entry = result[mode][str(layer_id)]
                    entry["promotions"] += device["promotions"][phase][row]
                    entry["evictions"] += device["evictions"][phase][row]
                    entry["migration_bytes"] += device["promotions"][phase][row] * bytes_per_expert
                if "insertions" in device:
                    _add_insertions(result["decode"][str(layer_id)], device, row)
                policy = result.get("residency_policy", {}).get(str(layer_id))
                if policy is not None:
                    policy["boundary_updates"] += device["boundary_updates"][row]
                    policy["promotions"] += device["promotions"][0][row] + device["promotions"][1][row]
                    policy["evictions"] += device["evictions"][0][row] + device["evictions"][1][row]
        return result

    def _trace_routes_from_host(self, buffers: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        result = {}
        pairs = list(zip(self._layer_ids, self._layer_ids[1:]))
        for phase in self._counters:
            popularity = buffers.get(f"register:{phase}:popularity")
            affinity = buffers.get(f"register:{phase}:affinity")
            if popularity is None or popularity.shape[0] == 0:
                result[phase] = {
                    "popularity": {str(layer_id): [] for layer_id in self._layer_ids},
                    "affinity": {},
                }
                continue
            experts = popularity.shape[1]
            popular_entries = self._top_entries(popularity)
            affinity_entries = self._top_entries(affinity.flatten(1))
            result[phase] = {
                "popularity": {
                    str(layer_id): [[index, value] for index, value in entries]
                    for layer_id, entries in zip(self._layer_ids, popular_entries)
                },
                "affinity": {
                    f"{source}->{target}": [
                        [flat // experts, flat % experts, value] for flat, value in entries
                    ]
                    for (source, target), entries in zip(pairs, affinity_entries)
                },
            }
        return result

    def snapshot_counters(self) -> dict[str, dict[str, dict[str, int | float | None]]]:
        """Return JSON-compatible cumulative totals and current allocation gauges."""
        self._refresh_side_pull_delivery()
        result = {}
        register_totals = {
            phase: torch.cat(
                [
                    registers["graph_rows"],
                    registers["unique_experts"].unsqueeze(1),
                    registers["graph_unique_rows"],
                    registers["gathers"].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .tolist()
            for phase, registers in self._registers.items()
        }
        for mode, layers in self._counters.items():
            result[mode] = {}
            totals = register_totals.get(mode)
            for layer_id, counters in layers.items():
                row = asdict(counters)
                graph_logical_unique_misses = 0
                if totals is not None:
                    (
                        routed,
                        routed_missed,
                        unique,
                        unique_hit,
                        unique_missed,
                        gathers,
                    ) = totals[self._layer_positions[layer_id]]
                    streamer = self.streamers[layer_id]
                    row["requested_rows"] += unique_hit + unique_missed
                    row["routed_rows"] += routed
                    row["miss_rows"] += unique_missed
                    row["routed_miss_rows"] += routed_missed
                    row["unique_miss_rows"] += unique_missed
                    row["gathers"] += gathers
                    row["hot_hits"] += unique_hit
                    row["d2d_bytes"] += routed * (
                        streamer.bytes_per_expert - streamer.host_bytes_per_expert
                    )
                    row["h2d_bytes"] += unique_missed * streamer.host_bytes_per_expert
                    row["backing_source_bytes"] += (
                        unique_missed * streamer.bytes_per_expert
                    )
                    row["requested_unique_experts"] += unique
                    graph_logical_unique_misses = unique_missed
                side_pull = self._side_pull_snapshots.get((mode, layer_id))
                if side_pull is not None:
                    covered, residual, wasted, posted = side_pull
                    delivered = posted
                    # One useful capacity-one post replaces one distinct
                    # demand expert, regardless of how many logical routes
                    # covered that expert. The planner's unique-miss counter
                    # intentionally retains the logical demand total, so
                    # remove that one would-be demand copy before adding the
                    # side-stream's actual physical copy.
                    useful_posts = posted - wasted
                    covered_demand_rows = min(useful_posts, graph_logical_unique_misses)
                    streamer = self.streamers[layer_id]
                    row["miss_rows"] -= covered_demand_rows
                    row["h2d_bytes"] -= covered_demand_rows * streamer.host_bytes_per_expert
                    row["backing_source_bytes"] -= covered_demand_rows * streamer.bytes_per_expert
                    row["miss_rows"] += delivered
                    row["side_pull_rows"] += delivered
                    row["side_pull_bytes"] += (
                        delivered * streamer.bytes_per_expert
                    )
                    row["side_pull_h2d_bytes"] = delivered * streamer.host_bytes_per_expert
                    row["side_pull_d2d_bytes"] = delivered * (
                        streamer.bytes_per_expert - streamer.host_bytes_per_expert
                    )
                    row["residual_demand_rows"] = (
                        graph_logical_unique_misses - covered_demand_rows
                    )
                    row["residual_demand_h2d_bytes"] = (
                        row["residual_demand_rows"] * streamer.host_bytes_per_expert
                    )
                    row["side_pull_posted_rows"] = posted
                    row["side_pull_useful_posts"] = useful_posts
                    row["side_pull_wasted_rows"] = wasted
                    row["side_pull_covered_routes"] = covered
                    row["side_pull_residual_routes"] = residual
                    row["side_pull_useful_precision"] = (posted - wasted) / posted if posted else 0.0
                cache = self.caches.get(layer_id)
                row["residency_bytes"] = cache.capacity_bytes if cache else 0
                row["allocation_bytes"] = cache.allocation_bytes if cache else 0
                row["scratch_bytes"] = cache.scratch_bytes if cache else 0
                row["prefetch_pull_bytes"] = (
                    cache.prefetch_pull_bytes if cache else 0
                )
                result[mode][str(layer_id)] = row
        coordinators = getattr(self, "prefetch_coordinators", {})
        if coordinators:
            result["prefetch"] = {
                str(layer_id): coordinator.snapshot_stats()
                for layer_id, coordinator in coordinators.items()
            }
        policies = getattr(self, "residency_policies", {})
        if policies:
            result["residency_policy"] = {
                str(layer_id): policy.snapshot_metrics()
                for layer_id, policy in policies.items()
            }
        if self.async_promotions:
            result["residency_async"] = {
                "deferred_updates": self.deferred_residency_updates,
                "inflight_submissions": len(self._inflight_promotions),
            }
        updater = getattr(self, "gpu_residency", None)
        if updater is not None:
            device = updater.snapshot()
            result["residency_gpu"] = device
            for row, layer_id in enumerate(updater.layer_ids):
                bytes_per_expert = self.caches[layer_id].bytes_per_expert
                for phase, mode in enumerate(("decode", "prefill")):
                    entry = result[mode][str(layer_id)]
                    entry["promotions"] += device["promotions"][phase][row]
                    entry["evictions"] += device["evictions"][phase][row]
                    entry["migration_bytes"] += device["promotions"][phase][row] * bytes_per_expert
                if "insertions" in device:
                    _add_insertions(result["decode"][str(layer_id)], device, row)
                metrics = result.get("residency_policy", {}).get(str(layer_id))
                if metrics is not None:
                    metrics["boundary_updates"] += device["boundary_updates"][row]
                    metrics["promotions"] += device["promotions"][0][row] + device["promotions"][1][row]
                    metrics["evictions"] += device["evictions"][0][row] + device["evictions"][1][row]
        doorbell = getattr(self, "doorbell", None)
        if doorbell is not None:
            result["doorbell"] = doorbell.stats()
        return result

    def _refresh_side_pull_delivery(self) -> None:
        """Read pull counters only when a caller actually asks for a snapshot."""
        mode = self._last_observer_mode
        puller = getattr(self, "_prefetch_puller", None)
        if mode is None or puller is None:
            return
        if hasattr(puller, "poll_delivery_stats"):
            snapshots = puller.poll_delivery_stats()
            if snapshots is None:
                return
        elif hasattr(puller, "snapshot_delivery_stats"):
            snapshots = puller.snapshot_delivery_stats()
        else:
            snapshots = {
                layer_id: stats.snapshot() for layer_id, stats in puller.stats.items()
            }
        for layer_id, values in snapshots.items():
            self.record_side_pull_delivery(layer_id, mode, *values)

    def on_expert_distribution(
        self, forward_batch: ForwardBatch, single_pass_data: Mapping[str, Any]
    ) -> None:
        """Account for fresh gathers and change slots at residency boundaries.

        Draft-worker forwards are ignored before any counter is touched. A
        boundary is a prefill of at least ``update_prefill_tokens`` tokens or,
        when ``update_decode_forwards`` is set, that many decode or verify
        forwards after the previous boundary; see
        :class:`ResidencyBoundaryClock`. Every layer's scores advance at each
        boundary; ``min_residence_forwards`` only holds back that layer's slot
        changes.
        """
        # A query-only poll gives completed trace buffers to their writer while
        # every later forward continues without waiting for a D2H copy.
        for telemetry in tuple(self._trace_telemetry.values()):
            telemetry.poll()
        kind, tokens = classify_forward(forward_batch)
        if kind is ForwardKind.DRAFT:
            return
        if self._inflight_promotions:
            self._publish_completed_promotions(wait=False)
        if getattr(self, "_async_residency_scores", False):
            self._poll_async_residency_scores()
        counts = single_pass_data.get("global_physical_count")
        if counts is None:
            return
        if counts.ndim != 2 or any(
            layer_id >= counts.shape[0] or counts.shape[1] != streamer.num_experts
            for layer_id, streamer in self.streamers.items()
        ):
            raise ValueError(
                "recorder counts do not match expert cache layers and experts"
            )
        mode = _PHASES[kind]
        self._last_observer_mode = mode
        clock = self._boundary_clock
        boundary_tokens = clock.observe(kind, tokens)
        qualifying = boundary_tokens is not None
        eager_gathered = []
        for layer_id in self._layer_ids:
            streamer = self.streamers[layer_id]
            stats = streamer.last_gather_stats
            gathered = stats is not self._last_gather[layer_id]
            eager_gathered.append(gathered)
            if not gathered:
                continue
            self._last_gather[layer_id] = stats
            counters = self._counters[mode][layer_id]
            counters.requested_rows += stats.requested_rows
            counters.miss_rows += stats.miss_rows
            counters.routed_rows += stats.routed_rows
            counters.routed_miss_rows += stats.routed_miss_rows
            counters.unique_miss_rows += stats.unique_miss_rows
            counters.hot_hits += stats.hot_hit_rows
            counters.d2d_bytes += stats.d2d_bytes
            counters.h2d_bytes += stats.h2d_bytes
            counters.backing_source_bytes += stats.source_bytes
            counters.pinned_hits += stats.pinned_host_hit_rows
            counters.pinned_misses += stats.pinned_host_miss_rows
            counters.transfer_wait_ns += getattr(stats, "transfer_wait_ns", 0)
            counters.gather_fallbacks += int(
                getattr(stats, "gather_fallback_used", False)
            )
            counters.gather_copy_engine_bytes += getattr(stats, "copy_engine_bytes", 0)
            _add_host_read_counters(counters, stats)
            file_bytes = streamer.file_source_bytes_per_expert
            if file_bytes is not None:
                counters.file_source_bytes = (
                    counters.file_source_bytes or 0
                ) + stats.miss_rows * file_bytes
                counters.file_misses = (counters.file_misses or 0) + (
                    stats.miss_rows if file_bytes else 0
                )
                counters.file_fallbacks = (counters.file_fallbacks or 0) + (
                    stats.pinned_host_miss_rows
                    if streamer.pinned_host_cache is not None
                    else stats.miss_rows
                )
            previous_admissions, previous_evictions = self._last_pinned_cache_stats[
                layer_id
            ]
            admissions, evictions = self._pinned_cache_stats(streamer)
            counters.pinned_admissions += admissions - previous_admissions
            counters.pinned_evictions += evictions - previous_evictions
            self._last_pinned_cache_stats[layer_id] = (admissions, evictions)
        self._accumulate_registers(mode, counts, eager_gathered)
        updater = getattr(self, "gpu_residency", None)
        if updater is not None:
            first = self._layer_positions[updater.layer_ids[0]]
            updater.observe_forward(
                kind,
                tokens,
                qualifying,
                graph_served=kind is not ForwardKind.IDLE and not eager_gathered[first],
            )
        elif qualifying:
            self._update_residency(boundary_tokens, mode)
            if not getattr(self, "_async_residency_scores", False):
                self._notify_residency_listeners()
        if clock.forwards % self.log_interval == 0:
            self._schedule_trace(mode)
            self._log_doorbell()

    def on_speculative_commit(self, accepted_tokens: int) -> None:
        """Correct the oldest outstanding verify's drafted tokens to those it committed.

        The speculative worker calls this once accept lengths are on the host;
        the next boundary advances scores by the corrected token count.
        """
        self._boundary_clock.commit(accepted_tokens)

    def record_side_pull_delivery(
        self, layer_id: int, mode: str, covered: int, residual: int, wasted: int, posted: int
    ) -> None:
        """Attribute new rows from one cumulative side-pull delivery snapshot.

        ``covered``/``residual``/``wasted``/``posted`` are the FULL cumulative
        totals a producer's ``PullDeliveryStats.snapshot()`` returns at this
        call. The manager differences them from the prior producer read, then
        adds only the delta to this ``(mode, layer_id)`` ledger. This prevents
        a decode snapshot from relabeling prefill's already-reported pulls.
        ``posted`` is the
        producer's PHYSICAL row count (1 per forward a real prediction was
        posted, regardless of how many routes it covered or none), and is
        what this class adds to ``miss_rows``/``side_pull_rows``. ``covered``
        sums matched ROUTES rather than delivered rows, so it can exceed 1 on
        a multi-token forward with overlap on the predicted expert; ``covered
        + wasted`` would overcount physical rows in that case, which is why
        ``posted`` -- not that sum -- is the delivered-row count used here.
        Never touches ``unique_miss_rows`` or ``routed_miss_rows``, which stay
        logical counts of distinct and routed actual misses read from the
        ordinary routing path.

        ``discard_graph_capture_routes`` clears both producer and manager
        baselines after graph warmup. A smaller source total also defensively
        starts a fresh telemetry epoch, so a reset cannot produce a negative
        phase delta or retain warmup rows.
        """
        current = tuple(index(value) for value in (covered, residual, wasted, posted))
        previous = self._last_side_pull_totals.get(layer_id)
        if previous is None:
            delta = current
        elif any(value < prior for value, prior in zip(current, previous)):
            for key in tuple(self._side_pull_snapshots):
                if key[1] == layer_id:
                    del self._side_pull_snapshots[key]
            delta = current
        else:
            delta = tuple(value - prior for value, prior in zip(current, previous))
        self._last_side_pull_totals[layer_id] = current
        accumulated = self._side_pull_snapshots.get((mode, layer_id), (0, 0, 0, 0))
        self._side_pull_snapshots[(mode, layer_id)] = tuple(
            prior + value for prior, value in zip(accumulated, delta)
        )
