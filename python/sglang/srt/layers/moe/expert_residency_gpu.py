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

from sglang.kernels.ops.moe.expert_insert_rows import insert_expert_rows
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
_STAGE_OFF = 0
_STAGE_SCRATCH = 1
_STAGE_DIRECT = 2
# Above every residency_rank_keys key of a nonnegative score (sign-bit-0 float32 bits << 16 < 2**47).
_ROUTED_RANK_OFFSET = 1 << 48


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
        insert_on_miss: bool | int = False,
        insert_on_miss_decay: float = 0.98,
        fused_insert: bool = False,
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

        self.insert_stage = int(insert_on_miss)
        self.insert_on_miss = self.insert_stage != _STAGE_OFF
        self.insert_direct = self.insert_stage == _STAGE_DIRECT
        self.insert_on_miss_decay = float(insert_on_miss_decay)
        self.fused_insert = bool(fused_insert)
        self.insert_scores = None
        self.insert_tensors = None
        self.insert_active = None
        self.victims = None
        if self.insert_on_miss:
            self._init_insert_on_miss()

        streamers[0].residency_update = self
        streamers[0].before_eager_gather = self.flush
        if self.insert_direct:
            for row, streamer in enumerate(streamers):
                streamer.residency_direct = self
                streamer.residency_row = row

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
        self.scratch_columns = self.capacity.unsqueeze(1) + self.miss_columns.unsqueeze(0)
        self.insert_sources = torch.zeros((layers, self.miss_rows), dtype=torch.int64, device=device)
        self.insert_destinations = torch.zeros_like(self.insert_sources)
        self.insertions = torch.zeros(layers, dtype=torch.long, device=device)
        self.insertion_evictions = torch.zeros(layers, dtype=torch.long, device=device)
        self.insertion_truncated = torch.zeros(layers, dtype=torch.long, device=device)
        if self.insert_direct:
            self._init_insert_direct()
            return
        # Scratch rows and slots of every tensor, host-backed or not, live in the cache's own tensors.
        #
        # Byte-row views, D2D on an RTX 5090. Two regimes, because this card has ~128 MB of L2
        # and the answer depends on whether the rows are in it:
        #
        #   cached      ~0.007 ms/row index copy, ~0.054 for copy_expert_row_segments_gpu
        #   production  ~0.0105 ms/row index copy, ~0.125 for the segment kernel,
        #               ~0.0037 for the fused masked kernel below
        #
        # The cached pair is what `iom-cuda-tests.sh` reports: it copies 20 rows back and forth
        # inside a single 221 MB tensor, so ~55 MB stays resident in L2 across every rep and
        # nothing goes to HBM. Quote it only as a cached number. The production pair was measured
        # over 48 separate layer tensors -- a 4.9 GiB working set with no L2 reuse between layers,
        # which is the shape this loop actually runs in, 48 layers x 6 tensors per boundary.
        #
        # The design decision rests on the production regime, and the cached numbers understate
        # the margin rather than inventing it: the index copy beats the segment kernel by 7.7x
        # cached and by 11.9x on a real working set. What the cached numbers *do* hide is the
        # absolute cost of this loop, by 1.5x, and the segment kernel's by 2.3x.
        self.insert_tensors = [
            tuple(tensor.view(torch.uint8).reshape(tensor.shape[0], -1) for tensor in cache.tensors.values())
            for cache in self.caches
        ]
        if self.fused_insert:
            # One active flag per lane, read on the device inside the kernel, so an idle lane
            # moves no bytes while the launch shape stays fixed and no count reaches the host.
            self.insert_active = torch.zeros(
                (layers, self.miss_rows), dtype=torch.int32, device=device
            )
            # Compile every specialisation now, with no lane active so not a byte moves. Triton
            # JITs on first call and specialises on the row length, and the boundary's first call
            # can land inside graph capture, which would have to compile mid-capture. Warming it
            # here costs 48 x 6 no-op launches once at startup and removes that from the path.
            for row, tensors in enumerate(self.insert_tensors):
                for rows_view in tensors:
                    insert_expert_rows(
                        rows_view,
                        self.insert_sources[row],
                        self.insert_destinations[row],
                        self.insert_active[row],
                    )

    def _init_insert_direct(self) -> None:
        """Stage DIRECT: a victim shortlist per layer, and the guarantee that every miss finds one.

        Each boundary ranks ``miss_rows`` candidate slots per layer. A gather
        disqualifies the shortlist entries its own forward routes to and sends
        its miss copies straight into the survivors, so no scratch row is read
        or written and the cache keeps those rows as residency.

        Capacity guarantee. A graph gather serves at most ``miss_rows`` routes,
        so its distinct hits ``H`` and distinct misses ``M`` satisfy
        ``H + M <= miss_rows``. Each hit route disqualifies at most one entry,
        leaving at least ``miss_rows - H >= M`` survivors whenever the shortlist
        is full. It is full whenever a layer holds ``miss_rows`` slots, because
        :meth:`_rank_victims` ranks routed residents last instead of dropping
        them; ``capacity >= 2 * miss_rows`` is required on top of that. Without
        a full shortlist a miss could find no slot, and with no scratch row to
        fall back on its routes would read another expert, so a smaller layer
        is refused rather than allowed to truncate.
        """
        width = self.miss_rows
        for layer_id, streamer in zip(self.layer_ids, self.streamers):
            # Without a host-source tensor the merged segment table would carry this layer's
            # full expert rows, and the segment kernel reads its count on the device, so it
            # cannot size its grid to them: ~0.125 ms/row against ~0.0105 for the index copy
            # it replaces, on a production-shaped working set (see `_init_insert_on_miss` for
            # both regimes; the cached 0.054/0.007 pair understates this gap). That trade is
            # only free because the rows it actually
            # takes over here are the per-expert scalars (8 B/row in production), while the
            # megabyte rows were already on this kernel. A layer with nothing on the host has
            # nothing to stream and no reason to insert on miss, so refuse rather than
            # silently move its rows onto the slower path.
            if (not streamer._graph_host_pair_count
                    and getattr(streamer.format, "key", None) != "exl3"):
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 needs host-source expert tensors; "
                    f"layer {layer_id} keeps every streamed tensor on the device"
                )
        for layer_id, cache in zip(self.layer_ids, self.caches):
            if cache.capacity < 2 * width:
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 needs every layer to hold at least "
                    f"twice its graph-gather rows; layer {layer_id} has {cache.capacity} "
                    f"slots for {width} rows. Raise SGLANG_MOE_HOT_GPU_MB or use stage 1."
                )
        device, layers = self.device, self.num_layers
        # Shortlisted slots per layer, and which of those columns name a real slot.
        self.victims = torch.zeros((layers, width), dtype=torch.long, device=device)
        self.victim_valid = torch.zeros((layers, width), dtype=torch.bool, device=device)
        self.victims_fresh = False
        self.gather_lanes = torch.arange(width, dtype=torch.long, device=device)
        self.gather_insertions = torch.zeros(layers, dtype=torch.long, device=device)
        self.gather_evictions = torch.zeros(layers, dtype=torch.long, device=device)
        self._pending_commit = None

    def check_miss_plans(self) -> None:
        """Refuse every backend that could write a cache row this mode does not control.

        One guard, not three, because these three refusals are the whole safety
        argument and separated ones get dropped piecemeal in a later refactor.

        * The plan must be the gather's own miss buffers. SCRATCH reads each
          missed expert's scratch row from them; DIRECT drives both the copy and
          the residency commit from that one ``count``, which is what makes them
          unable to disagree.
        * The doorbell copier writes slots from its own stream at any time
          between post and completion, and reports a timed-out request
          undelivered while its copy may still land. Under DIRECT that would
          overwrite a slot the mapping has already committed.
        * A prefetch pull owns a dedicated row and covers routes that then never
          enter residency, and it reads cache rows from a side stream.
        """
        from sglang.srt.environ import envs

        if self.insert_direct:
            exl3 = [getattr(s.format, "key", None) == "exl3" for s in self.streamers]
            if any(exl3) and not all(exl3):
                raise ValueError("EXL3 DIRECT requires every GPU-updated layer to use EXL3")

        if self.insert_direct and envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off":
            raise ValueError(
                "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 needs "
                "SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE=off: a side-stream pull reads and writes "
                "cache rows this mode commits residency for"
            )
        for streamer in self.streamers:
            if self.insert_direct and getattr(streamer.format, "key", None) == "exl3":
                from sglang.srt.dsv41_config import Dsv41Config
                from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend

                cfg = Dsv41Config.from_envs()
                if not cfg.enable_ram_miss_leases:
                    raise ValueError("EXL3 DIRECT requires SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1")
                if cfg.enable_ram_miss_two_phase:
                    raise ValueError("EXL3 DIRECT requires SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE=0")
                if cfg.enable_expert_prefetch:
                    raise ValueError("EXL3 DIRECT requires SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0")
                if not isinstance(streamer.row_backend, Exl3RamMissRowBackend):
                    raise ValueError("EXL3 DIRECT requires the native EXL3 RAM-miss backend")
            plan = streamer.row_plan
            if (
                plan.expert_ids.data_ptr() != streamer._graph_source_rows.data_ptr()
                or plan.count.data_ptr() != streamer._graph_miss_count.data_ptr()
            ):
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE needs each graph gather's own miss plan; "
                    "leave SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY at 0"
                )
            if (self.insert_direct and streamer.pinned_host_cache is not None
                    and getattr(streamer.format, "key", None) != "exl3"):
                raise ValueError(
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 cannot admit rows through the "
                    "pinned host cache"
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
        if self.insert_on_miss and not self.insert_direct:
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
        if self.insert_on_miss and not self.insert_direct:
            self.plans_fresh.fill_(False)
        if self.insert_direct:
            # Warm-up and capture gathers insert for real, unlike stage 1's, whose boundary is
            # gated by ``enabled`` and so stays inert until here. They cannot be gated the same
            # way: the redirect has to be inside the captured graph, so it has to run during the
            # capture. What they leave behind is consistent -- each row holds the expert the
            # mapping names -- but the experts are the capture's dummy routes rather than the
            # seed's, for at most one gather width per layer, which demand traffic replaces
            # within a few forwards. Drop their counts, as this call does for every other
            # capture-recorded counter, so the trace measures serving and not the warm-up.
            self.gather_insertions.zero_()
            self.gather_evictions.zero_()
            # Rank a shortlist now, eagerly: the first replay's gather reads it before any
            # boundary has run, and a capture-time shortlist would name pre-warm-up slots.
            self._rank_victims(self.route_counts > 0)

    def snapshot(self) -> dict[str, list]:
        """Host copies of the device counters, for the metrics trace only."""
        snapshot = {
            "promotions": self.promotions.cpu().tolist(),
            "evictions": self.evictions.cpu().tolist(),
            "boundary_updates": self.boundary_updates.cpu().tolist(),
            "truncated_layers": self.truncated.cpu().tolist(),
        }
        for name, counter in self.insertion_counters().items():
            snapshot[name] = counter.cpu().tolist()
        return snapshot

    def insertion_counters(self) -> dict[str, torch.Tensor]:
        """The active stage's insertion counters, keyed by the name both readers publish them under.

        One accessor for the metrics trace and for :meth:`snapshot`, because
        they count the same events and the stages keep them in different
        tensors: SCRATCH increments at the boundary, DIRECT in the gathers. A
        second copy of that choice is a counter that silently reads zero on
        whichever path nobody updated -- and a Stage 2 arm reporting zero
        insertions reads as the feature being off rather than as a bug.

        ``insertion_truncated`` stays a shared tripwire. Under DIRECT it counts
        gathers with a miss that found no victim; those copies land in slot 0,
        so its staying zero is the invariant, not an absence of instrumentation.
        """
        if not self.insert_on_miss:
            return {}
        insertions, evictions = self.insertions, self.insertion_evictions
        if self.insert_direct:
            insertions, evictions = self.gather_insertions, self.gather_evictions
        return {
            "insertions": insertions,
            "insertion_evictions": evictions,
            "insertion_truncated": self.insertion_truncated,
        }

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
        routed = self.route_counts > 0 if self.insert_on_miss else None
        decay = self.decay_table.index_select(0, self.tokens.clamp(max=self.decay_table_tokens))
        self.scores.copy_(torch.where(gate, self.scores * decay + self.route_counts, self.scores))
        self.route_counts.masked_fill_(gate, 0.0)
        eligible = gate & (self.forwards - self.last_update >= self.min_residence_forwards)
        if inserting:
            # DIRECT inserted during the gathers themselves; the boundary only proposes the next
            # forward's victims. SCRATCH still copies the previous forward's misses out of scratch.
            if self.insert_direct:
                self._rank_victims(routed)
            else:
                self._insert_misses(gate, routed)
            self.boundary_updates.add_(eligible.to(torch.long))
        else:
            self._promote(eligible, width, phase)
            if self.insert_direct:
                # A prefill boundary rewrote the mapping, so the shortlist it ranked is stale.
                self._rank_victims(routed)
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
        # Unused columns copy their own scratch row onto itself, so every column's destination is
        # distinct. The fused kernel skips them outright and does not need that, but both paths
        # read the same two buffers and the self-copy keeps them meaningful either way.
        self.insert_sources.copy_(torch.where(insert, source_rows, self.scratch_columns))
        self.insert_destinations.copy_(torch.where(insert, targets, self.scratch_columns))
        if self.fused_insert:
            self.insert_active.copy_(insert)
            for row, tensors in enumerate(self.insert_tensors):
                sources, destinations_row = self.insert_sources[row], self.insert_destinations[row]
                active = self.insert_active[row]
                for rows in tensors:
                    insert_expert_rows(rows, sources, destinations_row, active)
        else:
            for row, tensors in enumerate(self.insert_tensors):
                sources, destinations_row = self.insert_sources[row], self.insert_destinations[row]
                for rows in tensors:
                    rows.index_copy_(0, destinations_row, rows.index_select(0, sources))
        self.insertions.add_(insert_counts)
        self.insertion_evictions.add_(evicted.sum(dim=1))
        self.insertion_truncated.add_((wanted_counts > insert_counts).to(torch.long))
        self.last_update.copy_(torch.where(insert_counts > 0, self.forwards, self.last_update))
        self.plans_fresh.masked_fill_(gate, False)

    def _rank_victims(self, routed: torch.Tensor) -> None:
        """Propose each layer's next ``miss_rows`` victim slots: free slots first, then the
        lowest-scored residents, those this window did not route ahead of those it did.

        This is only a proposal. It is ranked before the next forward's routing
        is known, so it may name a slot that forward reads; the gather that
        commits removes those itself (:meth:`gather_destinations`). Ranking
        routed residents last keeps warm rows out of the shortlist's front
        without dropping them: a short prefill routes every resident, and an
        emptied shortlist sends every miss of the next forward to slot 0.
        """
        slot_dump = self.max_capacity
        width = self.miss_rows
        slot_experts = self.slot_to_expert.clamp(min=0)
        free = (self.slot_state == _FREE) & self.slot_valid
        evictable = (self.slot_state == _READY) & self.slot_valid & (self.slot_to_expert >= 0)
        rank = residency_rank_keys(self.insert_scores).gather(1, slot_experts)
        rank = rank + routed.gather(1, slot_experts).to(torch.int64) * _ROUTED_RANK_OFFSET
        never = torch.iinfo(torch.int64).max
        keys = torch.full(
            (self.num_layers, self.victim_columns), never, dtype=torch.int64, device=self.device
        )
        keys[:, : slot_dump + 1] = torch.where(
            free,
            self.slot_ids.unsqueeze(0) - (slot_dump + 1),
            torch.where(evictable, rank, never),
        )
        ranked = torch.sort(keys, dim=1)
        self.victims.copy_(ranked.indices[:, :width].clamp(max=slot_dump))
        self.victim_valid.copy_(ranked.values[:, :width] < never)
        self.victims_fresh = True

    def gather_destinations(
        self, row: int, remap: torch.Tensor, route_slots: torch.Tensor, scratch_base: int
    ) -> torch.Tensor:
        """Point one layer's gather at victim slots instead of scratch rows.

        ``route_slots`` is the gather's own ``expert_to_slot`` lookup of every
        route: a slot for a hit, ``-1`` for a miss. A shortlist entry equal to
        one of those slots is a row this forward reads, so it is dropped here,
        which is the whole safety argument -- the choice is made with the
        forward's routing in hand even though the shortlist was ranked before
        it. The surviving entries take the miss lanes in rank order, and
        ``_init_insert_direct`` proves there are always enough of them.

        Writes the destinations into the gather's own plan slots and returns
        the remap with every miss lane translated. Fixed shape, device only.
        """
        streamer = self.streamers[row]
        victims, valid = self.victims[row], self.victim_valid[row]
        # A miss lane's slot is -1 and a padded column is clamped to the dump index, so
        # neither can equal a real routed slot; only genuine hits disqualify an entry.
        hazard = (route_slots.unsqueeze(1) == victims.unsqueeze(0)).any(dim=0)
        order = torch.argsort((hazard | ~valid).to(torch.uint8), stable=True)
        usable = victims.index_select(0, order)
        usable_valid = (valid & ~hazard).index_select(0, order)
        live = (self.gather_lanes < streamer._graph_miss_count.long()) & usable_valid
        # Lanes past the miss count are never copied (the copy reads the same count) and
        # never remapped to. A lane inside the count that is not live is still copied to
        # slot 0; the full shortlist rules that out and _commit_gather counts it if not.
        destinations = torch.where(live, usable, torch.zeros_like(usable))
        streamer._graph_destination_slots.copy_(destinations.to(torch.int32))
        self._pending_commit = (row, streamer, destinations, live)
        # The fused planner keeps the router's native ids, so remap may be int32; index_select
        # takes int64 only, and the result carries the destinations' dtype back to the caller,
        # which casts to the router's dtype as it always has.
        rank = (remap - scratch_base).clamp(min=0, max=self.miss_rows - 1).long()
        return torch.where(remap >= scratch_base, destinations.index_select(0, rank), remap)

    def commit_gather(self) -> None:
        """Commit the residency of the gather whose copies were just issued."""
        row, streamer, destinations, live = self._pending_commit
        self._pending_commit = None
        self._commit_gather(row, streamer, destinations, live)

    def _commit_gather(
        self, row: int, streamer, destinations: torch.Tensor, live: torch.Tensor
    ) -> None:
        """Move residency onto the rows this gather is about to copy.

        The mask is the same ``_graph_miss_count`` the copy kernel reads, so the
        mapping can never claim a row the copy did not write: there is one count,
        not two that could disagree. The writes are issued after the copy on the
        gather's own stream, and every backend that could write a slot from
        another stream is refused at startup (:meth:`check_miss_plans`).
        """
        experts, slot_dump = self.num_experts, self.max_capacity
        new_experts = streamer._graph_source_rows[: self.miss_rows]
        slots = self.slot_to_expert[row]
        targets = torch.where(live, destinations, slot_dump)
        evicted = live & (slots.gather(0, destinations) >= 0)
        mapping = self.mapping[row]
        mapping.scatter_(0, torch.where(evicted, slots.gather(0, destinations), experts), -1)
        mapping.scatter_(0, torch.where(live, new_experts, experts), targets)
        slots.scatter_(0, targets, torch.where(live, new_experts, -1))
        self.slot_state[row].scatter_(0, targets, _READY)
        self.slot_generations[row].scatter_add_(0, targets, live.to(torch.long))
        # ``tensor[i, j] = scalar`` stages the value through a pageable CPU tensor, which a graph
        # capture refuses; the batched boundary gets away with a whole-column slice, a per-layer
        # commit does not. Fill a one-element view instead, which stays on the device.
        self.slot_state[row, slot_dump : slot_dump + 1].fill_(_FREE)
        slots[slot_dump : slot_dump + 1].fill_(-1)
        self.slot_generations[row, slot_dump : slot_dump + 1].fill_(0)
        self.gather_insertions[row].add_(live.sum())
        self.gather_evictions[row].add_(evicted.sum())
        self.insertion_truncated[row].add_((streamer._graph_miss_count.long() > live.sum()).long().sum())

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
