"""In-graph residency update for dynamic expert hot caches.

The decode forward's residency boundary runs as fixed-shape device operations
inside the first streamed layer's graph gather, so a captured decode graph
decays scores, decides promotions, rewrites every layer's expert-to-slot
mapping and slot state, and copies promoted rows without the host.
"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.expert_residency import (
    decide_residency_on_device,
    residency_rank_keys,
)
from sglang.srt.layers.moe.expert_residency_clock import ForwardKind

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

_FREE = 0
_READY = 3
_DECODE_PHASE = 0
_PREFILL_PHASE = 1


class GpuResidencyUpdater:
    """Device-owned residency state and the boundary update over all streamed layers.

    Construction moves every layer's scores, route counts, expert-to-slot
    mapping, slot states and generations into ``[layers, ...]`` device banks
    and rebinds the caches and policies to row views, so graph gathers and
    route recording keep their addresses. A dump column past each row absorbs
    the writes of unused fixed-shape entries.

    Boundary timing mirrors :class:`ResidencyBoundaryClock`. A decode or
    verify forward that completes ``update_decode_forwards`` sets a device
    pending flag; the first streamed layer's gather of the next forward
    applies it before any gather reads a mapping, which is when the host path
    would have applied it after the forward. An eager forward applies a
    pending boundary before its first gather through :meth:`flush`, and a
    qualifying prefill applies its boundary eagerly at the end of the forward.
    Decode boundaries promote at most ``max_promotions`` experts per layer,
    the best in rank order; prefill boundaries are uncapped. Promotions are
    copied from the registered host rows straight into the evicted or free
    slots on the current stream, ahead of the gathers that read them.

    With ``insert_on_miss`` (every decode forward is a boundary), a decode
    boundary promotes no host rows. Instead each layer's missed experts of the
    previous forward, still in its scratch rows, are copied device to device
    into free slots, then into the slots of the residents with the lowest
    per-token decayed route score among residents that forward did not route.
    Prefill boundaries keep the promotion decision above.
    """

    def __init__(
        self,
        manager: ExpertHotCacheManager,
        *,
        max_promotions: int,
        decay_table_tokens: int = 1 << 14,
        insert_on_miss: bool = False,
        insert_on_miss_decay: float = 0.98,
    ) -> None:
        layer_ids = [
            layer_id
            for layer_id in manager._layer_ids
            if layer_id in manager.caches and layer_id in manager.residency_policies
        ]
        if not layer_ids or len(layer_ids) != len(manager.caches):
            raise ValueError("GPU residency update needs a dynamic policy for every hot cache")
        caches = [manager.caches[layer_id] for layer_id in layer_ids]
        policies = [manager.residency_policies[layer_id] for layer_id in layer_ids]
        streamers = [manager.streamers[layer_id] for layer_id in layer_ids]
        if any(streamer.graph_gather_rows < 1 for streamer in streamers):
            raise ValueError("GPU residency update needs the graph gather on every hot-cache layer")
        num_experts = {policy.num_experts for policy in policies}
        devices = {cache.device for cache in caches}
        configs = {
            (policy.decay, policy.decay_tokens, policy.promotion_margin, policy.promotion_sigmas)
            for policy in policies
        }
        if len(num_experts) != 1 or len(devices) != 1 or len(configs) != 1:
            raise ValueError("GPU residency update needs one expert count, device and policy configuration")
        self.manager = manager
        self.layer_ids = layer_ids
        self.caches = caches
        self.policies = policies
        self.streamers = streamers
        self.num_experts = num_experts.pop()
        self.device = devices.pop()
        self.num_layers = len(layer_ids)
        self.update_decode_forwards = manager.update_decode_forwards
        self.min_residence_forwards = manager.min_residence_forwards
        self.promotion_margin = policies[0].promotion_margin
        self.promotion_sigmas = policies[0].promotion_sigmas
        capacities = [cache.capacity for cache in caches]
        self.max_capacity = max(capacities)
        self.max_promotions = max(1, min(index(max_promotions), max(self.max_capacity, 1)))
        self.prefill_promotions = max(self.max_capacity, 1)
        device = self.device
        layers, experts, slots = self.num_layers, self.num_experts, self.max_capacity

        self.scores = torch.stack([policy._scores for policy in policies]).contiguous()
        self.route_counts = torch.stack([policy._pending_counts for policy in policies]).contiguous()
        for row, policy in enumerate(policies):
            policy._scores = self.scores[row]
            policy._pending_counts = self.route_counts[row]
        self.mapping = torch.full((layers, experts + 1), -1, dtype=torch.long, device=device)
        self.slot_state = torch.zeros((layers, slots + 1), dtype=torch.uint8, device=device)
        self.slot_generations = torch.zeros((layers, slots + 1), dtype=torch.long, device=device)
        self.slot_to_expert = torch.full((layers, slots + 1), -1, dtype=torch.long, device=device)
        for row, cache in enumerate(caches):
            capacity = cache.capacity
            self.mapping[row, :experts].copy_(cache.expert_to_slot)
            self.slot_state[row, :capacity].copy_(cache.slot_state)
            self.slot_generations[row, :capacity].copy_(cache.slot_generations)
            self.slot_to_expert[row, :capacity].copy_(
                torch.tensor(cache.slot_to_expert, dtype=torch.long)
            )
            cache.expert_to_slot = self.mapping[row, :experts]
            cache.slot_state = self.slot_state[row, :capacity]
            cache.slot_generations = self.slot_generations[row, :capacity]
            cache.device_residency = self
        self.capacity = torch.tensor(capacities, dtype=torch.long, device=device)
        self.slot_ids = torch.arange(slots + 1, dtype=torch.long, device=device)
        self.slot_valid = self.slot_ids.unsqueeze(0) < self.capacity.unsqueeze(1)
        self.columns = torch.arange(self.prefill_promotions, dtype=torch.long, device=device)
        self.scratch_rows = self.capacity.unsqueeze(1)
        self.source_rows = torch.zeros((layers, self.prefill_promotions), dtype=torch.int64, device=device)
        self.destination_rows = torch.zeros_like(self.source_rows)
        self.destination_slots = torch.zeros((layers, self.prefill_promotions), dtype=torch.int32, device=device)
        self.copy_counts = torch.zeros((layers, 1), dtype=torch.int32, device=device)
        self.segments = [getattr(streamer, "_graph_row_segments", None) for streamer in streamers]
        self.device_pairs = [tuple(getattr(streamer, "_graph_device_pairs", ())) for streamer in streamers]

        self.forwards = torch.zeros(1, dtype=torch.long, device=device)
        self.tokens = torch.zeros(1, dtype=torch.long, device=device)
        self.decode_forwards = torch.zeros(1, dtype=torch.long, device=device)
        self.boundary_pending = torch.zeros(1, dtype=torch.bool, device=device)
        self.enabled = torch.zeros(1, dtype=torch.bool, device=device)
        self.last_update = torch.zeros(layers, dtype=torch.long, device=device)
        self.host_pending = False
        decay_values = self._decay_values(policies[0], index(decay_table_tokens))
        self.decay_table_tokens = len(decay_values) - 1
        self.decay_table = torch.tensor(decay_values, dtype=torch.float32, device=device)

        self.promotions = torch.zeros((2, layers), dtype=torch.long, device=device)
        self.evictions = torch.zeros((2, layers), dtype=torch.long, device=device)
        self.boundary_updates = torch.zeros(layers, dtype=torch.long, device=device)
        self.truncated = torch.zeros(layers, dtype=torch.long, device=device)

        self.insert_on_miss = bool(insert_on_miss)
        self.insert_on_miss_decay = float(insert_on_miss_decay)
        self.insert_scores = None
        self.insert_segments = None
        if self.insert_on_miss:
            self._init_insert_on_miss()

        streamers[0].residency_update = self
        streamers[0].before_eager_gather = self.flush

    @staticmethod
    def _decay_values(policy, limit: int) -> list[float]:
        """Boundary decays by token count until they reach float32 zero, which every longer window keeps.

        Clamping the token index to the last entry is then exact. A decay that
        stays nonzero through ``limit`` tokens would be approximated past it, so
        it is refused.
        """
        if policy.decay_tokens is None or policy.decay >= 1.0:
            return [policy.boundary_decay(None)] * 2
        values = []
        for tokens in range(limit + 1):
            value = policy.boundary_decay(tokens)
            values.append(value)
            if float(torch.tensor(value, dtype=torch.float32)) == 0.0:
                return values
        raise ValueError(
            "GPU residency update cannot tabulate this decay exactly; "
            "lower SGLANG_MOE_HOT_DECAY_TOKENS"
        )

    def _init_insert_on_miss(self) -> None:
        from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments

        if self.update_decode_forwards != 1:
            raise ValueError("insert-on-miss needs a residency boundary after every decode forward")
        if not 0.0 < self.insert_on_miss_decay <= 1.0:
            raise ValueError("insert-on-miss decay must be in (0, 1]")
        rows = {streamer.graph_gather_rows for streamer in self.streamers}
        if len(rows) != 1:
            raise ValueError("insert-on-miss needs one graph-gather row count on every layer")
        device, layers = self.device, self.num_layers
        self.miss_rows = rows.pop()
        # Victims are ranked over every slot column, padded to at least one column per miss row.
        self.victim_columns = max(self.max_capacity + 1, self.miss_rows)
        self.miss_columns = torch.arange(self.miss_rows, dtype=torch.long, device=device)
        self.insert_scores = self.scores.clone()
        values = (
            [1.0, 1.0]
            if self.insert_on_miss_decay == 1.0
            else self._tabulate_decay(lambda tokens: self.insert_on_miss_decay**tokens, 1 << 16)
        )
        self.insert_decay_table_tokens = len(values) - 1
        self.insert_decay_table = torch.tensor(values, dtype=torch.float32, device=device)
        # A layer's plan is fresh from its graph gather until a boundary inserts it.
        self.plans_fresh = torch.zeros(layers, dtype=torch.bool, device=device)
        self.insert_sources = torch.zeros((layers, self.miss_rows), dtype=torch.int64, device=device)
        self.insert_slots = torch.zeros((layers, self.miss_rows), dtype=torch.int32, device=device)
        self.insert_counts = torch.zeros((layers, 1), dtype=torch.int32, device=device)
        self.insertions = torch.zeros(layers, dtype=torch.long, device=device)
        self.insertion_evictions = torch.zeros(layers, dtype=torch.long, device=device)
        self.insertion_truncated = torch.zeros(layers, dtype=torch.long, device=device)
        # Scratch rows and slots of every tensor, host-backed or not, live in the cache's own tensors.
        self.insert_segments = [
            expert_row_segments([(tensor, tensor) for tensor in cache.tensors.values()])
            for cache in self.caches
        ]

    def check_miss_plans(self) -> None:
        """Refuse a layer whose copy backend posts a plan other than the gather's own miss buffers.

        Insertions read the scratch row of each missed expert from those buffers.
        """
        for streamer in self.streamers:
            plan = streamer.row_plan
            if (
                plan.expert_ids.data_ptr() != streamer._graph_source_rows.data_ptr()
                or plan.count.data_ptr() != streamer._graph_miss_count.data_ptr()
            ):
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS needs each graph gather's own miss plan; "
                    "leave SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY at 0"
                )

    @staticmethod
    def _tabulate_decay(decay_of_tokens, limit: int) -> list[float]:
        values = []
        for tokens in range(limit + 1):
            value = decay_of_tokens(tokens)
            values.append(value)
            if float(torch.tensor(value, dtype=torch.float32)) == 0.0:
                return values
        raise ValueError("insert-on-miss decay is too close to one to tabulate exactly")

    def on_graph_forward(self, tokens: int) -> None:
        """At the first streamed layer's graph gather: apply a pending boundary, then count this forward.

        Without decode boundaries the forward is only counted.
        """
        if self.update_decode_forwards > 0:
            self._apply(self.boundary_pending & self.enabled, self.max_promotions, _DECODE_PHASE)
        if self.insert_on_miss:
            self.plans_fresh.fill_(True)
        self._count(tokens, decode=True)

    def flush(self) -> None:
        """Apply a decode boundary still pending before an eager forward's first gather."""
        if self.host_pending:
            self.host_pending = False
            self._apply(self.boundary_pending & self.enabled, self.max_promotions, _DECODE_PHASE)

    def observe_forward(
        self, kind: ForwardKind, tokens: int, boundary: bool, graph_served: bool
    ) -> None:
        """Mirror one forward the host clock observed onto the device counters.

        A forward whose first gather ran on the graph path already counted
        itself as a decode forward, and a prefill that did so is corrected.
        Any other forward first applies a boundary still pending, as its
        first eager gather would have, so a forward without gathers never
        lands in the window it closes, and is then counted here. A qualifying
        prefill applies its boundary now; a decode boundary stays pending for
        the next forward.
        """
        if kind is ForwardKind.DRAFT:
            return
        if graph_served:
            self.host_pending = False
            if kind is not ForwardKind.DECODE and kind is not ForwardKind.VERIFY:
                self.decode_forwards.sub_(1)
                self.boundary_pending.copy_(self._decode_boundary_reached())
        else:
            self.flush()
            self._count(tokens, decode=kind in (ForwardKind.DECODE, ForwardKind.VERIFY))
        if not boundary:
            return
        if kind is ForwardKind.PREFILL:
            self._apply(self.enabled.clone(), self.prefill_promotions, _PREFILL_PHASE)
        else:
            self.host_pending = True

    def reset_after_capture(self, clock) -> None:
        """Restore the device clock from the host clock after graph capture and enable updates."""
        self.forwards.fill_(clock.forwards)
        self.tokens.fill_(clock.tokens_since_boundary)
        self.decode_forwards.fill_(clock.decode_forwards_since_boundary)
        self.boundary_pending.fill_(False)
        self.host_pending = False
        self.enabled.fill_(True)
        if self.insert_on_miss:
            self.plans_fresh.fill_(False)

    def snapshot(self) -> dict[str, list]:
        """Host copies of the device counters, for the metrics trace only."""
        snapshot = {
            "promotions": self.promotions.cpu().tolist(),
            "evictions": self.evictions.cpu().tolist(),
            "boundary_updates": self.boundary_updates.cpu().tolist(),
            "truncated_layers": self.truncated.cpu().tolist(),
        }
        if self.insert_on_miss:
            snapshot["insertions"] = self.insertions.cpu().tolist()
            snapshot["insertion_evictions"] = self.insertion_evictions.cpu().tolist()
            snapshot["insertion_truncated"] = self.insertion_truncated.cpu().tolist()
        return snapshot

    def _decode_boundary_reached(self) -> torch.Tensor:
        if self.update_decode_forwards < 1:
            return torch.zeros_like(self.boundary_pending)
        return self.decode_forwards >= self.update_decode_forwards

    def _count(self, tokens: int, decode: bool) -> None:
        self.forwards.add_(1)
        self.tokens.add_(tokens)
        if decode:
            self.decode_forwards.add_(1)
            self.boundary_pending.logical_or_(self._decode_boundary_reached())

    def _apply(self, gate: torch.Tensor, width: int, phase: int) -> None:
        """Run one masked boundary: every tensor keeps its value where ``gate`` is false."""
        inserting = self.insert_on_miss and phase == _DECODE_PHASE
        if self.insert_on_miss:
            insert_decay = self.insert_decay_table.index_select(
                0, self.tokens.clamp(max=self.insert_decay_table_tokens)
            )
            self.insert_scores.copy_(
                torch.where(gate, self.insert_scores * insert_decay + self.route_counts, self.insert_scores)
            )
        routed = self.route_counts > 0 if inserting else None
        decay = self.decay_table.index_select(0, self.tokens.clamp(max=self.decay_table_tokens))
        self.scores.copy_(torch.where(gate, self.scores * decay + self.route_counts, self.scores))
        self.route_counts.masked_fill_(gate, 0.0)
        eligible = gate & (self.forwards - self.last_update >= self.min_residence_forwards)
        if inserting:
            self._insert_misses(gate, routed)
            self.boundary_updates.add_(eligible.to(torch.long))
        else:
            self._promote(eligible, width, phase)
        self.tokens.masked_fill_(gate, 0)
        self.decode_forwards.masked_fill_(gate, 0)
        self.boundary_pending.masked_fill_(gate, False)

    def _promote(self, eligible: torch.Tensor, width: int, phase: int) -> None:
        """Decide promotions from the scores and copy promoted host rows into their slots."""
        experts = self.num_experts
        decision = decide_residency_on_device(
            self.scores,
            self.mapping[:, :experts] >= 0,
            self.capacity,
            promotion_margin=self.promotion_margin,
            promotion_sigmas=self.promotion_sigmas,
            max_promotions=width,
            active=eligible,
        )
        width = decision.promotions.shape[1]
        columns = self.columns[:width].unsqueeze(0)
        promote = columns < decision.promotion_counts.unsqueeze(1)
        evict = columns < decision.eviction_counts.unsqueeze(1)
        slot_dump = self.max_capacity
        evicted_slots = torch.where(
            evict, self.mapping.gather(1, decision.evictions), torch.full_like(decision.evictions, slot_dump)
        )
        free = (self.slot_state == _FREE) & self.slot_valid
        free.scatter_(1, evicted_slots, evict)
        free[:, slot_dump] = False
        ascending_free = torch.sort(
            torch.where(free, self.slot_ids.unsqueeze(0), torch.full_like(self.slot_ids, slot_dump + 1))
        ).values[:, :width]
        destinations = torch.where(promote, ascending_free, torch.full_like(ascending_free, slot_dump))
        self.mapping.scatter_(1, torch.where(evict, decision.evictions, experts), -1)
        self.mapping.scatter_(1, torch.where(promote, decision.promotions, experts), destinations)
        self.slot_state.scatter_(1, evicted_slots, _FREE)
        self.slot_to_expert.scatter_(1, evicted_slots, -1)
        self.slot_state.scatter_(1, destinations, _READY)
        self.slot_to_expert.scatter_(1, destinations, decision.promotions)
        self.slot_generations.scatter_add_(1, destinations, promote.to(torch.long))
        self.slot_state[:, slot_dump] = _FREE
        self.slot_to_expert[:, slot_dump] = -1
        self.slot_generations[:, slot_dump] = 0
        self.source_rows[:, :width].copy_(torch.where(promote, decision.promotions, 0))
        self.destination_rows[:, :width].copy_(
            torch.where(promote, ascending_free, self.scratch_rows.expand(-1, width))
        )
        self.destination_slots[:, :width].copy_(self.destination_rows[:, :width])
        self.copy_counts.copy_(decision.promotion_counts.unsqueeze(1))
        self._copy_promotions(width)
        self.promotions[phase].add_(decision.promotion_counts)
        self.evictions[phase].add_(decision.eviction_counts)
        self.boundary_updates.add_(eligible.to(torch.long))
        self.truncated.add_((decision.needed_promotions > width).to(torch.long))
        changed = (decision.promotion_counts > 0) | (decision.eviction_counts > 0)
        self.last_update.copy_(torch.where(changed, self.forwards, self.last_update))

    def _insert_misses(self, gate: torch.Tensor, routed: torch.Tensor) -> None:
        """Copy the last forward's fresh, still nonresident misses from scratch rows into slots.

        Plan row ``r`` of a layer names the expert in scratch row ``capacity + r``
        (``plan_graph_routes``). Targets are free slots in slot order, then
        residents with no route since the previous boundary, lowest
        ``(insert score, -expert)`` first; misses beyond them are not inserted.
        """
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        experts, width, slot_dump = self.num_experts, self.miss_rows, self.max_capacity
        columns = self.miss_columns.unsqueeze(0)
        miss_ids = torch.stack([streamer._graph_source_rows[:width] for streamer in self.streamers])
        miss_counts = torch.cat([streamer._graph_miss_count for streamer in self.streamers]).to(torch.long)
        wanted = (
            (columns < miss_counts.unsqueeze(1))
            & (gate & self.plans_fresh).unsqueeze(1)
            & (self.mapping.gather(1, miss_ids) < 0)
        )
        order = torch.argsort((~wanted).to(torch.uint8), dim=1, stable=True)
        wanted_counts = wanted.sum(dim=1)
        new_experts = miss_ids.gather(1, order)
        source_rows = self.capacity.unsqueeze(1) + order

        slot_experts = self.slot_to_expert.clamp(min=0)
        free = (self.slot_state == _FREE) & self.slot_valid
        evictable = (
            (self.slot_state == _READY)
            & self.slot_valid
            & (self.slot_to_expert >= 0)
            & ~routed.gather(1, slot_experts)
        )
        never = torch.iinfo(torch.int64).max
        slot_keys = torch.full(
            (self.num_layers, self.victim_columns), never, dtype=torch.int64, device=self.device
        )
        slot_keys[:, : slot_dump + 1] = torch.where(
            free,
            self.slot_ids.unsqueeze(0) - (slot_dump + 1),
            torch.where(evictable, residency_rank_keys(self.insert_scores).gather(1, slot_experts), never),
        )
        ranked = torch.sort(slot_keys, dim=1)
        targets = ranked.indices[:, :width].clamp(max=slot_dump)
        insert_counts = torch.minimum(wanted_counts, (ranked.values[:, :width] < never).sum(dim=1))
        insert = columns < insert_counts.unsqueeze(1)
        evicted = insert & ~free.gather(1, targets)
        destinations = torch.where(insert, targets, slot_dump)
        self.mapping.scatter_(1, torch.where(evicted, self.slot_to_expert.gather(1, targets), experts), -1)
        self.mapping.scatter_(1, torch.where(insert, new_experts, experts), destinations)
        self.slot_to_expert.scatter_(1, destinations, torch.where(insert, new_experts, -1))
        self.slot_state.scatter_(1, destinations, _READY)
        self.slot_generations.scatter_add_(1, destinations, insert.to(torch.long))
        self.slot_state[:, slot_dump] = _FREE
        self.slot_to_expert[:, slot_dump] = -1
        self.slot_generations[:, slot_dump] = 0
        self.insert_sources.copy_(torch.where(insert, source_rows, 0))
        self.insert_slots.copy_(destinations)
        self.insert_counts.copy_(insert_counts.unsqueeze(1))
        for row, segments in enumerate(self.insert_segments):
            copy_expert_row_segments_gpu(
                segments, self.insert_sources[row], self.insert_slots[row], self.insert_counts[row]
            )
        self.insertions.add_(insert_counts)
        self.insertion_evictions.add_(evicted.sum(dim=1))
        self.insertion_truncated.add_((wanted_counts > insert_counts).to(torch.long))
        self.last_update.copy_(torch.where(insert_counts > 0, self.forwards, self.last_update))
        self.plans_fresh.masked_fill_(gate, False)

    def _copy_promotions(self, width: int) -> None:
        """Copy each layer's promoted rows into its destination slots on the current stream."""
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        for row in range(self.num_layers):
            segments = self.segments[row]
            if segments is not None:
                copy_expert_row_segments_gpu(
                    segments, self.source_rows[row], self.destination_slots[row], self.copy_counts[row]
                )
            source_rows = self.source_rows[row, :width]
            destination_rows = self.destination_rows[row, :width]
            for source, destination in self.device_pairs[row]:
                destination.view(torch.uint8).reshape(destination.shape[0], -1).index_copy_(
                    0,
                    destination_rows,
                    source.view(torch.uint8).reshape(source.shape[0], -1).index_select(0, source_rows),
                )
