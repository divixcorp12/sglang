"""A streamed layer's side of the expert engine service: its pinned slot table and its gather.

- ``NativePinnedSlotTable``: a layer's ``PinnedSlotTable`` over the native bookkeeping.
- Copy-engine helpers: the small-copy table, and the check that refuses an unsafe
  launch configuration.
- Row-range arithmetic (``spill_room_shortfall`` ... ``prefill_share_rows``) the service
  checks its tiers with.
- ``ExpertEngineRowBackend``: a layer's gather, the lease chain of device kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import (
    copy_expert_row_segments_gpu,
    device_copy_grid_blocks,
)
from sglang.kernels.ops.moe.expert_lease_block import MAX_LANES
from sglang.kernels.ops.moe.expert_stream_transport import (
    BuildKey,
    ExpertStreamDevice,
    host_layout,
)
from sglang.srt.layers.moe.expert_engine.config import CopyConfig, CopyMode
from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
from sglang.srt.layers.moe.expert_stream import ram_prefetch_spec_share

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService

# The prefill share's size, as a fraction of a layer's usable RAM rows (prefill_share_rows): 64 of the 187 usable rows
# of the served layout, the share served at 64 rows (/mnt/nvme1/share64-ab-20261016: -3.46 ms/token, TTFT -0.48 s).
# The tier-priced sweep's decode knee is ~0.15 (lane_eval.py --shares --rows-scale on lane-tier-replay, outputs
# /mnt/nvme1/lane-tier-replay/sweep/), but served at 0.15 (28 rows, /mnt/nvme1/share28-ab-20261016) it cost +1.58 s of
# TTFT per prompt for a decode gain within noise (-1.78 ms/token, 95% CI -4.53 .. +0.82): below ~900 generated tokens a
# request loses, so the larger share stays.
PREFILL_SHARE_FRACTION = 64 / 187


class NativePinnedSlotTable:
    """``PinnedSlotTable`` over the C++ service's bookkeeping for one streamed layer.

    Created at pinned-tier construction (from ``pinned_tier_options``) and registered
    with the service; the service itself starts on first use, once every layer's slabs
    exist. ``capacity`` is bound later by ``ExpertPinnedHostCache.__init__`` to the
    tier's row count.

    Host use nests: ``_depth`` counts this table's nesting, and the service counts the
    process-wide pause apart. Only the outermost level of a table pushes its hot set
    and refreshes the device slot map.
    """

    def __init__(
        self,
        service: ExpertEngineService,
        layer_id: int,
        streamer_of: Callable[[], object],
    ):
        self.service = service
        self.layer_id = layer_id
        self.capacity: Optional[int] = None
        self.streamer_of = streamer_of
        self._seen_version = -1
        self._slots: dict[int, int] = {}
        self._slots_version = -1
        self._depth = 0
        service.register(layer_id, self)

    def bind_capacity(self, capacity: int) -> None:
        self.capacity = int(capacity)

    def held_back_per_group(self) -> int:
        """HeldBackSlots, before the service starts: an upper bound on each group's staging and pool slots.

        Staging is at most the victim lanes under spill, else the miss lanes, else the widest wire
        (plan_graph_gather); the pool is the RAM prefetch's share.
        """
        stream = self.service.config.stream
        staging = stream.copy.graph_gather_victim_lanes or stream.copy.graph_gather_miss_lanes or MAX_LANES
        return min(staging, MAX_LANES) + ram_prefetch_spec_share(config=stream)

    @property
    def reserved_rows(self) -> int:
        """The row's staging and RAM prefetch pool slots: the service owns them, so ``assign`` never hands one out."""
        return self.service.staging_for(self.capacity) + self.service.pool_slots()

    @property
    def _row(self) -> int:
        self.service.ensure_started()
        return self.service.row_of(self.layer_id)

    @property
    def slot_to_expert(self) -> list[int]:
        """The READY slots' experts, -1 elsewhere: the inverse of the published map."""
        slots = [-1] * self.capacity
        for expert, slot in enumerate(self.service.host.mapping(self._row)):
            if slot >= 0:
                slots[slot] = expert
        return slots

    @property
    def expert_to_slot(self) -> dict[int, int]:
        """Resident experts and their slots, rebuilt when the C++ map version moves.

        Promotions read this once per promoted row. Every change of membership bumps
        the version. Callers must not mutate the returned dict.
        """
        row = self._row
        version = self.service.host.version()
        if self._slots_version != version:
            self._slots = {
                expert: slot
                for expert, slot in enumerate(self.service.host.mapping(row))
                if slot >= 0
            }
            self._slots_version = version
        return self._slots

    def __contains__(self, expert_id: int) -> bool:
        return self.service.host.contains(self._row, int(expert_id))

    def touch(self, expert_id: int) -> None:
        self.service.host.touch(self._row, int(expert_id))

    def assign(
        self, expert_id: int, protected=frozenset()
    ) -> tuple[int, Optional[int]]:
        return self.service.host.assign(
            self._row, int(expert_id), [int(e) for e in protected]
        )

    def release(self, slot: int) -> None:
        self.service.host.release(self._row, int(slot))

    def mapping(self, num_experts: int) -> list[int]:
        return self.service.host.mapping(self._row)

    # PinnedRowFills (SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS): the service's reader fills
    # this layer's slots while the tier's host use holds the thread paused.
    def fill_begin(self, experts, protected, fallback: bool) -> tuple[list[int], int]:
        return self.service.host.fill_begin(self._row, experts, protected, fallback)

    def fill_wait(self, rows: int) -> None:
        self.service.host.fill_wait(rows, self.service.fill_timeout_s)

    def fill_landed(self) -> int:
        return self.service.host.fill_landed()

    def fill_end(self) -> bool:
        return self.service.host.fill_end()

    def before_host_use(self, cache) -> None:
        """Pause the service thread (the service counts nesting).

        At this table's outermost level only, also push the layer's hot set to C++ and
        refresh the device slot map if the thread changed the C++ map.
        """
        self.service.before_host_use()
        try:
            if self._depth == 0:
                self._push_hot()
                version = self.service.host.version()
                if version != self._seen_version:
                    cache._refresh_mapping()
                    self._seen_version = version
        except BaseException:
            self.service.after_host_use()
            raise
        self._depth += 1

    def after_host_use(self, cache) -> None:
        self._depth -= 1
        self.service.after_host_use()

    def _push_hot(self) -> None:
        """Make the C++ victim choice protect what ``is_pinned`` protects right now.

        Until the updater is attached (start-up promotions) that is the hot cache's
        slots, which it reserves before promoting them in chunks, each in its own host
        use. Without this push, a later chunk's admission could evict an earlier
        chunk's expert from RAM, leaving a VRAM-hot expert with no pinned copy. After
        the updater is attached it is the snapshot taken at the pause.
        """
        streamer = self.streamer_of()
        hot = None if streamer is None else getattr(streamer, "hot_cache", None)
        if hot is not None:
            self.service.host.set_hot(
                self._row,
                self.service.hot_experts(self.layer_id)
                if self.service.gpu_hot_enabled
                else hot.slot_to_expert,
            )


def sm_copy_mask(names: Sequence[str], lanes: int = 8, *, build: BuildKey) -> int:
    """Bitmask of the copy-table entries the copy wait copies itself.

    With ``SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES`` the copy wait moves the host
    layout's small tensors, wherever they sit in ``names``. ``build`` is the
    transport's (format, router) key.
    """
    layout_names, small = host_layout(layout=build, lanes=lanes)
    small_names = {name for i, name in enumerate(layout_names) if small >> i & 1}
    return sum(1 << i for i, name in enumerate(names) if name in small_names)


def sm_copy_table(segments, sm_mask: int) -> torch.Tensor:
    """The rows of ``segments.table`` (the device copy table) that ``sm_mask`` names.

    Raises ``ValueError`` unless every source, destination and row size is 16-byte
    aligned.
    """
    rows = [i for i in range(segments.table.shape[0]) if sm_mask >> i & 1]
    table = segments.table[rows].contiguous()
    for source, destination, row_bytes in table.tolist():
        # The copy wait falls back to 1-byte loads off 16-byte alignment: refuse the
        # slow path rather than take it silently.
        if (source | destination | row_bytes) % 16:
            raise ValueError(
                f"CW reads 16-byte units: source {source:#x}, destination {destination:#x} and rows of "
                f"{row_bytes} B are not all 16-byte aligned"
            )
    return table


def check_sm_small_copies(copy: CopyConfig) -> None:
    """Refuse ``SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES`` without the copy engine.

    It moves part of the copy engine's work into the copy wait, which only the
    copy-engine chain runs.
    """
    if copy.sm_small_copies and copy.mode is not CopyMode.COPY_ENGINE:
        raise RuntimeError(
            "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES needs SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine"
        )


def padded_plan_width(capacity: int, lanes: int) -> int:
    """The length of a row backend's ``planned`` tensor: its plan width, and at least the build's lanes."""
    return max(capacity, lanes)


def spill_room_shortfall(ranges, *, staging: int, lanes: int, hot: int, pool: int = 0) -> list[tuple[int, int, int]]:
    """The node ranges of a layer that cannot give every forced CPU miss a RAM victim, as (group, slots, needed).

    A forced miss is read into a victim of its node's range: neither a staging slot (`staging`), nor a speculative pool
    slot (`pool`, SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE), nor an expert the record routes (at most `lanes` less the
    forced misses themselves), nor a VRAM-hot expert (at most `hot`). So a range of at least
    staging + pool + lanes + hot slots always has one (plan 2026-10-06, Task 9)."""
    need = staging + pool + lanes + hot
    return [(g, hi - lo, need) for g, (lo, hi) in enumerate(ranges) if hi - lo < need]


def pool_room_shortfall(ranges, *, staging_width: int, share: int) -> list[tuple[int, int, int, int]]:
    """The groups of a row the host's reserve_spec_pool would refuse, as (group, lo, hi, staging): each group stages
    min(width, its slots - 1) and needs 2 assignable slots past its ``share`` pool slots (ram_tier.h)."""
    short = []
    for g, (lo, hi) in enumerate(ranges):
        staging = min(staging_width, hi - lo - 1)
        if hi - lo - staging - share < 2:
            short.append((g, lo, hi, staging))
    return short


def full_layer_shortfall(ranges, *, staging_width: int, share: int, experts: int) -> list[tuple[int, int, int]]:
    """The groups of a row that cannot map every expert homed on them at once, as (group, assignable, homed).

    A group keeps min(width, its slots - 1) staging and ``share`` pool slots out of its range (ram_tier.h,
    reserve_staging and reserve_spec_pool); the rest must hold the experts e with e % groups == group.
    """
    short = []
    for g, (lo, hi) in enumerate(ranges):
        assignable = hi - lo - min(staging_width, hi - lo - 1) - share
        homed = len(range(g, experts, len(ranges)))
        if assignable < homed:
            short.append((g, assignable, homed))
    return short


def usable_rows(ranges, *, staging_width: int, share: int) -> int:
    """The slots of a row that can hold a mapped expert, over every group: each group's range less its
    min(width, slots - 1) staging slots and ``share`` pool slots (ram_tier.h, reserve_staging and reserve_spec_pool).
    The prefill share is counted over the same groups (owned_in_row)."""
    return sum(max(0, hi - lo - min(staging_width, hi - lo - 1) - share) for lo, hi in ranges)


def prefill_share_rows(usable: list[int], fraction: Optional[float] = None) -> int:
    """The prefill share for rows of ``usable`` mapped slots each: ``fraction`` (default PREFILL_SHARE_FRACTION) of
    the smallest, at least 1. The host keeps one share for every layer (set_prefill_share), so the smallest row
    decides."""
    if not usable:
        raise ValueError("exl3 RAM miss: the prefill share needs at least one row that admits")
    return max(1, round((PREFILL_SHARE_FRACTION if fraction is None else fraction) * min(usable)))


class ExpertEngineRowBackend(PinnedTierRowBackend):
    """``PinnedTierRowBackend`` whose gather is the lease chain.

    One gather posts a linear chain of kernels in one stream (see
    ``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "The chain"): the post kernel, which
    types the lanes from the device's slot map; C1, which copies the compacted
    ``HIT_SM`` lanes; with the GPU drip, C1s, which copies the ``HIT_SCRATCH`` lanes
    from the drip's VRAM scratch; S, which copies the misses piece by piece; the copy
    wait (CW); a stream wait; and the copy check (CC).

    ``routes`` holds the layer's routed experts (the protect set); ``quant_hook.apply_graph``
    copies them in before each gather. ``planned`` pads the plan's expert ids to at
    least the wire's lanes (-1 past the plan): the post kernel reads
    ``min(count, lanes)`` lanes and ``count`` lives on the device, so no host check
    can bound it. Every lane of a gather is delivered or the process stops, so ``keep``
    stays 1 and the delivered count is the plan's.

    With CPU experts, ``quant_hook.apply_graph`` sets ``cpu_input`` (``x`` and the routes'
    weights) around each gather and the post stages it for the CPU expert thread.
    """

    name = "expert_engine"
    # The residency updater's capability query: this backend's delivered count and lease chain stand behind the gather.
    leased = True

    def __init__(
        self,
        segments,
        host_row_map: torch.Tensor,
        device_side: ExpertStreamDevice,
        row: int,
        capacity: int,
        stream_maps: Mapping[int, torch.Tensor],
        hot_slots: Optional[torch.Tensor] = None,
        hot_capacity: int = 0,
        route_log=None,
        copy_engine: bool = False,
        copy_sm_table: Optional[torch.Tensor] = None,
        cpu_experts: bool = False,
        streamer_of: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__(segments, host_row_map, capacity)
        self.device_side = device_side
        self.row = row
        self.routes = torch.full(
            (capacity,), -1, dtype=torch.int64, device=host_row_map.device
        )
        self.planned = torch.full(
            (padded_plan_width(capacity, device_side.wire.lanes),),
            -1,
            dtype=torch.int64,
            device=host_row_map.device,
        )
        self.hot_slots = hot_slots
        self.hot_capacity = hot_capacity
        # Per tag, the miss stream's map of that tag's copy table (stream_segment_map).
        if set(stream_maps) != set(self.segments):
            raise ValueError(
                "the stream kernel needs a stream segment map for every copy table"
            )
        self.stream_maps = dict(stream_maps)
        # A GraphRouteLog only when the stage trace is on; None is the untraced chain.
        self.route_log = route_log
        # The service may copy a captured post's hits with its copy engine, which the
        # copy wait waits for.
        self.copy_engine = copy_engine
        # SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES: rows the copy wait copies.
        self.copy_sm_table = copy_sm_table
        # SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS: the service may compute resident lanes on the CPU.
        if cpu_experts and not copy_engine:
            raise ValueError("CPU experts are completed by the copy engine's copy wait")
        if cpu_experts and streamer_of is None:
            raise ValueError(
                "CPU experts check the layer's miss order at capture: pass streamer_of"
            )
        self.cpu_experts = cpu_experts
        self.cpu_input = None
        # A verify's dead rows ([max_rows, 1] bool) next to cpu_input, for the lane head; None when all rows are live.
        self.verify_dead = None
        self.spill = None  # DIRECT's (overflow flag, this layer's counter) under spill; set by ExpertEngineService.attach
        self.streamer_of = streamer_of
        self._delivered: Optional[torch.Tensor] = None
        # The GPU scorer (ExpertEngineService._bind_spec_scores): whether this row has a target, and its SpecScoreRow
        # once the target row attached.
        self.spec_expected = False
        self.spec_score = None
        # SGLANG_EXPERT_STREAM_TRACE's residency_capture: a ResidencyHook set by ExpertEngineService.attach; None is the
        # uncaptured post.
        self.residency_hook = None
        # SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS: C1s's segments, from the drip's scratch slots to this row's destinations (set by
        # ExpertEngineService.attach); None: no C1s.
        self.drip_segments = None

    @property
    def delivered_count(self) -> torch.Tensor:
        if self._delivered is None:
            raise RuntimeError(
                "EXL3 DIRECT reads the delivered count of a gather this backend posted"
            )
        return self._delivered

    def _stage_planned(self, plan) -> None:
        """Copy the plan's lane experts into the captured ``planned`` buffer.

        The plan is bounded first; the post kernel reads each lane's expert out of
        ``planned``, which is otherwise still at its -1 fill.
        """
        lanes = plan.expert_ids.numel()
        if lanes > self.planned.numel():
            raise ValueError(
                f"a plan of {lanes} lanes does not fit the backend's {self.planned.numel()}"
            )
        # A device copy: captured, so every replay refreshes it.
        self.planned[:lanes].copy_(plan.expert_ids)

    def post(self, tag, plan) -> None:
        """Enqueue this gather's lease chain on the current stream (see class doc)."""
        if self.route_log is not None:
            # ``routes`` (by quant_hook.apply_graph) and ``plan.count`` (by the planner) were
            # both refreshed earlier in this gather on this stream.
            self.route_log.record(self.row, self.routes, plan.count)
        self._stage_planned(plan)
        # Only a captured graph lets the service copy: an eager forward may load a
        # kernel module while the copy wait holds the stream, and a load blocks the copy
        # thread's driver calls (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine").
        captured = self.copy_engine and torch.cuda.is_current_stream_capturing()
        if captured and self.cpu_experts and self.cpu_input is None:
            raise RuntimeError(
                f"CPU experts: row {self.row} gathered without its input (quant_hook.apply_graph)"
            )
        if captured and self.cpu_experts:
            streamer = self.streamer_of()
            if streamer is None:
                raise RuntimeError(f"CPU experts: row {self.row}'s streamer is gone")
            if getattr(streamer, "_plan_miss_keys", None) is None:
                # attach installs the keys and a later enable_graph_gather resets them.
                # An unsorted plan would send the CPU the last-routed misses, not the
                # lowest-scored ones.
                raise RuntimeError(
                    f"CPU experts: row {self.row}'s route plan has no miss order (ExpertEngineService.attach)"
                )
        if self.spec_expected and self.spec_score is None:
            raise RuntimeError(
                f"RAM prefetch: row {self.row} posted before its target row attached; the GPU scorer's table is built "
                "at attach"
            )
        side = self.device_side
        side.post(
            self.row,
            self.planned,
            plan.count,
            self.routes,
            plan.slots,
            self.hot_slots,
            self.hot_capacity,
            captured=captured,
            cpu_input=self.cpu_input if captured and self.cpu_experts else None,
            spill=self.spill,
        )
        if self.residency_hook is not None:
            self.residency_hook.launch(
                side=side, routes=self.routes, planned=self.planned, count=plan.count, route_log=self.route_log
            )
        if self.spec_expected:
            # Layer T+1's gate on this layer's input, forked off this stream so C1 does not wait for it; joined below.
            x = self.cpu_input[0] if self.cpu_input is not None else None
            side.spec_score(
                entry=self.spec_score,
                x=x,
                routes=self.routes,
                route_weights=self.cpu_input[1] if self.cpu_input is not None else None,
            )
        side.copy_engine_captured |= captured
        copy_expert_row_segments_gpu(
            self.segments[tag], side.host_rows_1, side.dst_slots_1, side.go_1
        )
        if self.drip_segments is not None:
            # C1s: the HIT_SCRATCH lanes, from the GPU drip's scratch slots (LEASE_PROTOCOL.md, "The GPU drip").
            copy_expert_row_segments_gpu(
                self.drip_segments,
                side.scratch_rows_1,
                side.scratch_dst_1,
                side.scratch_go_1,
                grid_blocks=device_copy_grid_blocks(self.drip_segments.table.device),
            )
        side.stream(
            self.row,
            self.planned,
            plan.count,
            plan.slots,
            self.segments[tag],
            self.stream_maps[tag],
        )
        side.copy_wait(plan.count, plan.slots, self.copy_sm_table)
        if self.spec_expected:
            # After the copy wait the scoring overlapped; before the next post rewrites the seq it published under.
            side.spec_join()
        self._delivered = plan.count
