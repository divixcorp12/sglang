"""Fixed CUDA slots for frequently selected host-resident expert rows."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from sglang.srt.layers.moe.expert_prefetch import (
    ExpertPrefetchCoordinator,
    SparseNextLayerPolicy,
)
from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy
from sglang.srt.layers.moe.expert_stream import ExpertStreamer, _tensor_data

from sglang.srt.layers.moe.expert_transfer import (
    AsyncExpertTransferExecutor,
    ExpertCopySubmission,
    FixedRowTransferPlan,
    submit_expert_row_copies,
)
if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


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
        """``scratch_rows`` extra rows after the slots receive graph-gather misses."""
        capacity = index(capacity)
        scratch_rows = index(scratch_rows)
        if not 0 <= capacity <= streamer.num_experts:
            raise ValueError("hot cache capacity must be within the expert count")
        if scratch_rows < 0:
            raise ValueError("hot cache scratch rows cannot be negative")
        self.streamer = streamer
        self.capacity = capacity
        self.scratch_rows = scratch_rows
        self.bytes_per_expert = streamer.bytes_per_expert
        self.capacity_bytes = capacity * self.bytes_per_expert
        self.scratch_bytes = scratch_rows * self.bytes_per_expert
        devices = {
            _tensor_data(getattr(streamer.layer, name)).device
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cuda"
        }
        if len(devices) > 1:
            raise ValueError("hot cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity + scratch_rows,) + tuple(source.shape[1:]),
                dtype=source.dtype,
                device=self.device,
            )
            for name in streamer.tensor_names
            for source in [_tensor_data(getattr(streamer.layer, name))]
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

    @staticmethod
    def capacity_for_budget(streamer: ExpertStreamer, budget_bytes: int) -> int:
        """Round a runtime tensor byte budget down to complete expert slots."""
        budget_bytes = index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("hot cache byte budget cannot be negative")
        return min(streamer.num_experts, budget_bytes // streamer.bytes_per_expert)

    def _publish_mapping(self) -> None:
        mapping = [-1] * self.streamer.num_experts
        for slot, expert_id in enumerate(self.slot_to_expert):
            if self.slot_states[slot] is HotCacheSlotState.READY:
                mapping[expert_id] = slot
        self.expert_to_slot.copy_(
            torch.tensor(mapping, dtype=torch.long, device=self.device)
        )

    def _set_slot_state(self, slot: int, state: HotCacheSlotState) -> None:
        self.slot_states[slot] = state
        self.slot_state[slot] = int(state)

    def resident_experts(self) -> frozenset[int]:
        """Return only mappings that are ready for a gather consumer."""
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
            self.slot_generations[slot] = generation
            self.slot_to_expert[slot] = expert
            self._set_slot_state(slot, HotCacheSlotState.RESERVED)
            tickets.append(HotCacheSlotTicket(slot, expert, generation))
        return tuple(tickets)

    def begin_loading(self, ticket: HotCacheSlotTicket) -> bool:
        """Mark a valid reservation as loading; reject stale tickets."""
        if not self._ticket_matches(ticket, HotCacheSlotState.RESERVED):
            return False
        self._set_slot_state(ticket.slot, HotCacheSlotState.LOADING)
        return True

    def publish_ready(self, ticket: HotCacheSlotTicket) -> bool:
        """Publish a fully copied six-tensor slot after its ticket completes."""
        if not self._ticket_matches(ticket, HotCacheSlotState.LOADING):
            return False
        self._set_slot_state(ticket.slot, HotCacheSlotState.READY)
        self._publish_mapping()
        return True

    def cancel(self, ticket: HotCacheSlotTicket) -> bool:
        """Release an unconsumed reservation; stale tickets cannot alter slots."""
        if not self._ticket_matches(
            ticket, HotCacheSlotState.RESERVED, HotCacheSlotState.LOADING
        ):
            return False
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
        self.slot_to_expert[ticket.slot] = -1
        self._set_slot_state(ticket.slot, HotCacheSlotState.FREE)
        self._publish_mapping()
        return True

    def _load_reserved(self, tickets: Sequence[HotCacheSlotTicket]) -> None:
        """Copy one reserved placement bundle before publishing any slot mapping."""
        if not tickets:
            return
        assert self._transfer_executor is not None
        if len(self.streamer.tensor_names) != 6:
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
        assert self._transfer_plan is not None
        expert_rows = [ticket.expert_id for ticket in tickets]
        destination_slots = [ticket.slot for ticket in tickets]
        sources = {
            name: _tensor_data(getattr(self.streamer.layer, name))
            for name in self.streamer.tensor_names
        }
        source_rows = expert_rows
        secondary_source_rows = None
        use_secondary_source_rows = [False] * len(self.streamer.tensor_names)
        pinned_cache = self.streamer.pinned_host_cache
        if pinned_cache is not None and pinned_cache.cached_names:
            pinned_cache.ensure_rows(torch.tensor(expert_rows, device=self.device))
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
        self._transfer_plan.set_rows(
            source_rows,
            destination_slots,
            [ticket.generation for ticket in tickets],
        )
        if secondary_source_rows is not None:
            self._transfer_plan.secondary_source_rows.zero_()
            self._transfer_plan.secondary_source_rows[: len(secondary_source_rows)].copy_(
                torch.as_tensor(
                    secondary_source_rows,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
        try:
            if not all(self.begin_loading(ticket) for ticket in tickets):
                raise RuntimeError("hot cache reservation became stale")
            submission = submit_expert_row_copies(
                self._transfer_executor,
                self._transfer_plan,
                [(sources[name], self.tensors[name]) for name in self.streamer.tensor_names],
                backend=self.copy_backend,
                source_rows_cpu=source_rows,
                destination_slots_cpu=destination_slots,
                secondary_source_rows_cpu=secondary_source_rows,
                use_secondary_source_rows=use_secondary_source_rows,
                producer_stream=torch.cuda.current_stream(self.device),
            )
            self._transfer_executor.wait(
                submission.ticket, torch.cuda.current_stream(self.device)
            )
            if not all(self.publish_ready(ticket) for ticket in tickets):
                raise RuntimeError("hot cache completion ticket became stale")
            self.last_copy_submission = submission
        except Exception:
            for ticket in tickets:
                self.cancel(ticket)
            raise

    def reassign(self, expert_ids: Sequence[int]) -> HotCacheUpdateStats:
        """Synchronously replace slots through the generation-safe lifecycle."""
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
        wanted = set(desired)
        existing = {
            expert
            for expert, state in zip(self.slot_to_expert, self.slot_states)
            if state is HotCacheSlotState.READY
        }
        promoted = [expert_id for expert_id in desired if expert_id not in existing]
        evicted = existing - wanted
        if not promoted and not evicted:
            return HotCacheUpdateStats(0, 0, 0)
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
        tickets = self.reserve(tuple(zip(promoted, free_slots)), consumer_complete=True)
        self._load_reserved(tickets)
        return HotCacheUpdateStats(
            len(promoted), len(evicted), len(promoted) * self.bytes_per_expert
        )

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
    ) -> ExpertHotCacheManager | None:
        """Build the per-layer hot caches.

        ``graph_gather_batch_size`` > 0 reserves ``batch_size * top_k`` scratch
        rows per layer from the budget and enables each streamer's sync-free
        graph gather for routes of at most that many rows.
        ``update_decode_forwards`` > 0 also updates dynamic residency after every
        that many decode forwards, so a long decode is not served by the set the
        last long prefill chose.
        ``decay_tokens`` > 0 decays scores once per that many routed tokens
        instead of once per boundary, and ``promotion_sigmas`` adds that many
        standard deviations of count noise to the lead a promotion needs; see
        :class:`ExpertResidencyPolicy`.
        """
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
        for module in model.modules():
            streamer = getattr(module, "_nvfp4_expert_streamer", None)
            if streamer is None:
                continue
            layer_id = index(streamer.layer_id)
            if layer_id < 0 or layer_id in streamers:
                raise ValueError(
                    "expert hot cache requires unique nonnegative layer IDs"
                )
            streamers[layer_id] = streamer
        if not streamers:
            return None
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
        scratch_rows = {}
        for layer_id, streamer in streamers.items():
            top_k = getattr(streamer.layer, "top_k", None)
            if graph_gather_batch_size and top_k is None:
                raise ValueError("graph gather needs each streamed layer's top_k")
            scratch_rows[layer_id] = (
                graph_gather_batch_size * index(top_k) if graph_gather_batch_size else 0
            )
        selected = {layer_id: [] for layer_id in streamers}
        remaining = budget_bytes - sum(
            rows * streamers[layer_id].bytes_per_expert
            for layer_id, rows in scratch_rows.items()
        )
        if remaining < 0:
            raise ValueError(
                "expert hot cache budget cannot hold the graph-gather scratch rows"
            )
        for _, expert_id, layer_id in candidates:
            slot_bytes = streamers[layer_id].bytes_per_expert
            if slot_bytes <= remaining:
                selected[layer_id].append(expert_id)
                remaining -= slot_bytes
        if not any(selected.values()) and not any(scratch_rows.values()):
            return None
        manager = cls()
        manager.streamers = streamers
        manager.caches = {}
        manager.residency_policies = {}
        manager.dynamic = dynamic
        manager.update_prefill_tokens = update_prefill_tokens
        manager.update_decode_forwards = update_decode_forwards
        manager._decode_forwards_since_boundary = 0
        manager._tokens_since_boundary = 0
        manager.min_residence_forwards = min_residence_forwards
        manager.benefit_ratio = benefit_ratio
        manager.log_interval = log_interval
        manager.metrics_path = Path(metrics_path) if metrics_path else None
        manager.route_history_limit = route_history_limit
        manager._forward_count = 0
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
        manager._graph_counters = None
        for layer_id, expert_ids in selected.items():
            if expert_ids or scratch_rows[layer_id]:
                layer_streamer = streamers[layer_id]
                layer_streamer.expert_copy_backend = copy_backend
                cache = ExpertHotCache(
                    layer_streamer, len(expert_ids), scratch_rows[layer_id]
                )
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
        for layer_id, rows in scratch_rows.items():
            if rows:
                streamers[layer_id].enable_graph_gather(rows)
        manager._share_graph_counters()
        devices = {cache.device for cache in manager.caches.values()}
        logger.info(
            "Expert hot cache startup %s",
            json.dumps(
                {
                    "requested_bytes": budget_bytes,
                    "residency_bytes": manager.residency_bytes,
                    "slots": sum(cache.capacity for cache in manager.caches.values()),
                    "scratch_bytes": sum(
                        cache.scratch_bytes for cache in manager.caches.values()
                    ),
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
        return manager

    def enable_next_layer_prefetch(self, max_candidates: int) -> None:
        """Attach default-off sparse prefetch callbacks between adjacent layers."""
        max_candidates = index(max_candidates)
        if max_candidates <= 0:
            return
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
        for position, streamer in graph_streamers:
            shared[position].copy_(streamer.graph_counters)
            streamer.graph_counters = shared[position]
        self._graph_counters = shared

    def discard_graph_capture_routes(self) -> None:
        """Drop routes and counters that CUDA-graph warmup and capture recorded.

        Captured gathers execute once while recording, so their dummy routes
        land in the residency counts, graph counters, and route registers like
        a real forward.
        """
        if self._graph_counters is not None:
            self._graph_counters.zero_()
        for registers in self._registers.values():
            for register in registers.values():
                register.zero_()
        for policy in self.residency_policies.values():
            policy.pending_counts.zero_()

    @staticmethod
    def _pinned_cache_stats(streamer: ExpertStreamer) -> tuple[int, int]:
        cache = streamer.pinned_host_cache
        if cache is None:
            return (0, 0)
        return (cache.stats.populated_rows, cache.stats.evictions)

    @staticmethod
    def _phase(forward_batch: ForwardBatch) -> str:
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return "speculative"
        if mode.is_extend_without_speculative():
            return "prefill"
        return "decode"

    def _registers_for(
        self, phase: str, device: torch.device, experts: int
    ) -> dict[str, torch.Tensor]:
        registers = self._registers.get(phase)
        if registers is None:
            layers = len(self._layer_ids)
            registers = {
                "popularity": torch.zeros(
                    (layers, experts), dtype=torch.float64, device=device
                ),
                "affinity": torch.zeros(
                    (max(layers - 1, 0), experts, experts),
                    dtype=torch.float32,
                    device=device,
                ),
                "unique_experts": torch.zeros(layers, dtype=torch.int64, device=device),
                "graph_rows": torch.zeros((layers, 2), dtype=torch.int64, device=device),
            }
            self._registers[phase] = registers
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
        registers["popularity"].add_(rows)
        if registers["affinity"].shape[0]:
            routed = rows.to(torch.float32)
            registers["affinity"].baddbmm_(
                routed[:-1].unsqueeze(2), routed[1:].unsqueeze(1)
            )
        gathered = torch.tensor(eager_gathered, dtype=torch.bool)
        if device.type == "cuda":
            gathered = gathered.pin_memory().to(device, non_blocking=True)
        if self._graph_counters is not None:
            gathered = gathered | (self._graph_counters[:, 0] > 0)
            registers["graph_rows"].add_(self._graph_counters)
            self._graph_counters.zero_()
        registers["unique_experts"].add_(rows.ne(0).sum(dim=1) * gathered)

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
            if registers is None:
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

    def _write_trace(self, phase: str) -> None:
        if self.metrics_path is None:
            return
        trace = {
            "timestamp_ns": time.time_ns(),
            "phase": phase,
            "counters": self.snapshot_counters(),
            "route_statistics": self.snapshot_route_statistics(),
        }
        with self.metrics_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(trace, sort_keys=True) + "\n")

    def snapshot_counters(self) -> dict[str, dict[str, dict[str, int | None]]]:
        """Return JSON-compatible cumulative totals and current allocation gauges."""
        result = {}
        register_totals = {
            phase: torch.cat(
                [registers["graph_rows"], registers["unique_experts"].unsqueeze(1)],
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
                if totals is not None:
                    requested, missed, unique = totals[self._layer_positions[layer_id]]
                    streamer = self.streamers[layer_id]
                    row["requested_rows"] += requested
                    row["miss_rows"] += missed
                    row["hot_hits"] += requested - missed
                    row["d2d_bytes"] += requested * (
                        streamer.bytes_per_expert - streamer.host_bytes_per_expert
                    )
                    row["h2d_bytes"] += missed * streamer.host_bytes_per_expert
                    row["backing_source_bytes"] += missed * streamer.bytes_per_expert
                    row["requested_unique_experts"] += unique
                cache = self.caches.get(layer_id)
                row["residency_bytes"] = cache.capacity_bytes if cache else 0
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
        return result

    def on_expert_distribution(
        self, forward_batch: ForwardBatch, single_pass_data: Mapping[str, Any]
    ) -> None:
        """Account for fresh gathers and change slots at residency boundaries.

        A boundary is a prefill of at least ``update_prefill_tokens`` tokens or,
        when ``update_decode_forwards`` is set, that many decode forwards after
        the previous boundary. Every layer's scores advance at each boundary;
        ``min_residence_forwards`` only holds back that layer's slot changes.
        """
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
        self._forward_count += 1
        mode = self._phase(forward_batch)
        extend_tokens = forward_batch.extend_num_tokens or 0
        self._tokens_since_boundary += (
            extend_tokens
            if mode == "prefill"
            else getattr(forward_batch, "batch_size", 1)
        )
        if mode == "decode" and not forward_batch.forward_mode.is_idle():
            self._decode_forwards_since_boundary += 1
        qualifying = self.dynamic and (
            (mode == "prefill" and extend_tokens >= self.update_prefill_tokens)
            or (
                mode == "decode"
                and self.update_decode_forwards > 0
                and self._decode_forwards_since_boundary >= self.update_decode_forwards
            )
        )
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
            file_bytes = getattr(
                streamer.layer, "_nvfp4_file_source_bytes_per_expert", None
            )
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
        for layer_id in self._layer_ids if qualifying else ():
            cache = self.caches.get(layer_id)
            policy = self.residency_policies.get(layer_id)
            if cache is None or policy is None:
                continue
            policy.advance(self._tokens_since_boundary)
            if (
                self._forward_count - self._last_update[layer_id]
                < self.min_residence_forwards
            ):
                continue
            decision = policy.decide(cache.resident_experts())
            if not decision.promotions and not decision.evictions:
                continue
            policy.schedule_transfers(exact_demand=(), decision=decision)
            self._record_update(
                layer_id, cache.reassign(decision.desired_experts), mode
            )
            self._last_update[layer_id] = self._forward_count
        if qualifying:
            self._tokens_since_boundary = 0
            self._decode_forwards_since_boundary = 0
        if self._forward_count % self.log_interval == 0:
            self._write_trace(mode)
