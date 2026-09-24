"""Option C for EXL3 streamed experts (plan D8-D23).

``exl3_ram_miss_tables`` flattens what ``Exl3ShardRowSource`` knows (the per-expert
superset reads and the per-name segment map) plus each layer's pinned slabs into
int64 tensors, so the C++ thread reads and splits rows without Python.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch

from sglang.kernels.ops.moe.exl3_ram_miss import (
    MAX_IDS,
    Exl3RamMissDevice,
    Exl3RamMissHost,
    new_hot_page,
    new_page,
)
from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_read_split import SplitPolicy, StaticSplitPolicy
from sglang.srt.layers.moe.exl3_row_reader import mirror_path
from sglang.srt.layers.moe.expert_host_tier import quarantine_host_slabs
from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES

logger = logging.getLogger(__name__)
_SYNC_WAIT_NVTX = os.environ.get("SGLANG_DSV41_SYNC_WAIT_NVTX") == "1"


def _quarantine_service_at_exit(service: "weakref.ref[Exl3RamMissService]") -> None:
    """Exit hook: no device barrier is attempted in an exit handler, so the tiers are quarantined, never freed."""
    live = service()
    if live is not None:
        live.shutdown(at_exit=True)


@dataclass(frozen=True)
class Exl3RamMissTables:
    layer_ids: list[int]
    paths: list[str]
    source_paths: list[str]  # per file: the source shard it copies (itself with no mirror roots)
    file_sizes: torch.Tensor  # int64 [F]
    # int64 [L, E, P, 4]: file index, aligned offset, aligned length, destination offset in the
    # row's bounce slot. Part p of a row is served by root p; parts sum to the row's aligned length
    # and a zero-length part means "this root serves none of this row" (issue no read).
    extents: torch.Tensor
    starts: torch.Tensor  # int64 [L, E]: where the row starts inside its aligned superset
    parts: int  # P: mirror roots per row (1 with no roots)
    segments: torch.Tensor  # int64 [S, 4]: name index, dst offset, src offset, bytes
    slabs: torch.Tensor  # int64 [L, 6]: slab base addresses in EXL3_STREAMED_NAMES order
    row_bytes: torch.Tensor  # int64 [6]
    capacity: torch.Tensor  # int64 [L]
    slot_bytes: int
    # The slab tensors whose addresses are in ``slabs``: the C++ reader writes through
    # those raw addresses, so the tables own a reference to every slab (M5).
    keepalive: tuple = field(default=(), repr=False, compare=False)


def exl3_ram_miss_tables(
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    roots: Sequence[str] = (),
    policy: Optional[SplitPolicy] = None,
    source_root: Optional[str] = None,
) -> Exl3RamMissTables:
    """Tables for the streamed layers in ``slabs_by_layer`` (ascending layer id = row order).

    With ``roots`` (byte-identical mirrors of the checkpoint under ``source_root``), each row's
    aligned read is split across them as ``policy`` plans it, the same policy the eager
    ``read_split`` uses. ``paths`` is then shard-major, root-minor: part ``p`` of the shard with
    source index ``s`` is file ``s * parts + p``. With no roots ``parts == 1``, the files are the
    source shards and every extent starts at destination offset 0.
    """
    roots = tuple(roots)
    if roots:
        if policy is None or source_root is None:
            raise ValueError("mirror roots need both a split policy and the source root")
    else:
        if policy is not None:
            raise ValueError("a split policy needs mirror roots")
        policy = StaticSplitPolicy((1.0,))
    parts = len(roots) or 1
    planned = len(policy.plan(PAGE_BYTES).part_bytes)
    if planned != parts:
        raise ValueError(f"split policy plans {planned} parts for {parts} roots")
    layer_ids = sorted(slabs_by_layer)
    source_paths: list[str] = []
    source_index: dict[str, int] = {}
    extents = torch.empty((len(layer_ids), layout.num_experts, parts, 4), dtype=torch.int64)
    starts = torch.empty((len(layer_ids), layout.num_experts), dtype=torch.int64)
    splits: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}  # aligned length -> (part bytes, part starts)
    widest = 0
    for row, layer_id in enumerate(layer_ids):
        for expert in range(layout.num_experts):
            record = layout.records[(layer_id, expert)]
            if record.path not in source_index:
                source_index[record.path] = len(source_paths)
                source_paths.append(record.path)
            offset, length, start = record.aligned_read(PAGE_BYTES)
            if length not in splits:
                split = policy.plan(length)
                splits[length] = (split.part_bytes, split.starts)
            part_bytes, part_starts = splits[length]
            for part in range(parts):
                extents[row, expert, part, 0] = source_index[record.path] * parts + part
                extents[row, expert, part, 1] = offset + part_starts[part]
                extents[row, expert, part, 2] = part_bytes[part]
                extents[row, expert, part, 3] = part_starts[part]
            starts[row, expert] = start
            widest = max(widest, length)
    paths = (
        [mirror_path(source_root, root, path) for path in source_paths for root in roots]
        if roots
        else source_paths
    )
    # What each file is a copy of, for the reader's open-time size check to name.
    copied = [path for path in source_paths for _ in range(parts)]
    # A mirror is a byte-identical copy, so every part of a shard is bounded by the source's size
    # (as in the eager reader); the open-time check that each copy really has it is the reader's.
    source_sizes = [os.path.getsize(path) for path in source_paths]
    names = {name: index for index, name in enumerate(EXL3_STREAMED_NAMES)}
    segment_table = torch.tensor(
        [[names[s.name], s.dst_offset, s.src_offset, s.nbytes] for s in segments], dtype=torch.int64
    )
    row_bytes = torch.tensor(
        [sum(s.nbytes for s in segments if s.name == name) for name in EXL3_STREAMED_NAMES], dtype=torch.int64
    )
    slabs = torch.empty((len(layer_ids), len(EXL3_STREAMED_NAMES)), dtype=torch.int64)
    capacity = torch.empty(len(layer_ids), dtype=torch.int64)
    for row, layer_id in enumerate(layer_ids):
        tensors = slabs_by_layer[layer_id]
        rows = {int(tensors[name].shape[0]) for name in EXL3_STREAMED_NAMES}
        if len(rows) != 1:
            raise ValueError(f"layer {layer_id}: pinned slabs disagree on their row count {rows}")
        capacity[row] = rows.pop()
        for name, index in names.items():
            slab = tensors[name]
            if not slab.is_contiguous() or slab.device.type != "cpu":
                raise ValueError(f"layer {layer_id} {name}: slab must be a contiguous CPU tensor")
            per_row = slab.numel() * slab.element_size() // max(int(capacity[row]), 1)
            if per_row != int(row_bytes[index]):
                raise ValueError(f"layer {layer_id} {name}: slab rows hold {per_row} B, expected {int(row_bytes[index])}")
            slabs[row, index] = slab.data_ptr()
    return Exl3RamMissTables(
        layer_ids=layer_ids,
        paths=paths,
        source_paths=copied,
        file_sizes=torch.tensor([size for size in source_sizes for _ in range(parts)], dtype=torch.int64),
        extents=extents,
        starts=starts,
        parts=parts,
        segments=segment_table,
        slabs=slabs,
        row_bytes=row_bytes,
        capacity=capacity,
        slot_bytes=-(-widest // PAGE_BYTES) * PAGE_BYTES,
        keepalive=tuple(slabs_by_layer[layer_id][name] for layer_id in layer_ids for name in EXL3_STREAMED_NAMES),
    )


class NativePinnedSlotTable:
    """``PinnedSlotTable`` over the C++ service's slot bookkeeping for one streamed layer.

    Created at pinned-tier construction (from ``pinned_tier_options``); the service
    starts on first use, once every layer's slabs exist.
    """

    def __init__(self, service: "Exl3RamMissService", layer_id: int, streamer_of: Callable[[], object]):
        self.service = service
        self.layer_id = layer_id
        # Bound by ExpertPinnedHostCache.__init__ (bind_capacity) to the tier's row count.
        self.capacity: Optional[int] = None
        self.streamer_of = streamer_of
        self._seen_version = -1
        self._slots: "OrderedDict[int, int]" = OrderedDict()
        self._slots_version = -1
        # This table's host-use nesting; the service counts the process-wide pause apart.
        self._depth = 0
        service.register(layer_id, self)

    def bind_capacity(self, capacity: int) -> None:
        self.capacity = int(capacity)

    @property
    def _row(self) -> int:
        self.service.ensure_started()
        return self.service.row_of(self.layer_id)

    @property
    def slot_to_expert(self) -> list[int]:
        return self.service.host.slot_to_expert(self._row)

    @property
    def expert_to_slot(self) -> "OrderedDict[int, int]":
        """Resident experts and their slots, rebuilt only when the C++ map's version moves.

        Promotions read this once per promoted row. Every change of membership bumps
        the version; a touch does not, so the order is the LRU order as of the last
        change. Callers must not mutate the returned dict.
        """
        row = self._row
        version = self.service.host.version()
        if self._slots_version != version:
            mapping = self.service.host.mapping(row)
            self._slots = OrderedDict((expert, mapping[expert]) for expert in self.service.host.lru_order(row))
            self._slots_version = version
        return self._slots

    def __contains__(self, expert_id: int) -> bool:
        return self.service.host.contains(self._row, int(expert_id))

    def touch(self, expert_id: int) -> None:
        self.service.host.touch(self._row, int(expert_id))

    def assign(self, expert_id: int, protected=frozenset()) -> tuple[int, Optional[int]]:
        return self.service.host.assign(self._row, int(expert_id), [int(e) for e in protected])

    def release(self, slot: int) -> None:
        self.service.host.release(self._row, int(slot))

    def mapping(self, num_experts: int) -> list[int]:
        return self.service.host.mapping(self._row)

    def before_host_use(self, cache) -> None:
        """Pause the thread (the service counts nesting), then, at this table's outermost
        level only, push the layer's hot set to C++ and refresh the device slot map if the
        thread changed the C++ map."""
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

        The hot cache reserves a residency update's experts before the residency
        listener pushes them, and promotes them in chunks, each in its own host use:
        without this push, a later chunk's admission could evict an earlier chunk's
        expert from RAM (a VRAM-hot expert with no pinned copy).
        """
        streamer = self.streamer_of()
        hot = None if streamer is None else getattr(streamer, "hot_cache", None)
        if hot is not None:
            self.service.host.set_hot(
                self._row,
                self.service.hot_experts(self.layer_id) if self.service.gpu_hot_enabled else hot.slot_to_expert,
            )


class Exl3RamMissRowBackend(PinnedTierRowBackend):
    """``PinnedTierRowBackend`` whose translation posts RAM misses to the thread and waits.

    ``routes`` holds the layer's routed experts (the protect set); ``_apply_graph``
    copies them in before each gather. ``post`` = post kernel, wait kernel (which
    writes ``host_rows``, ``keep`` and ``ram_miss``), then the segment copy.
    ``host_row_map`` is the pinned tier's device slot map: the wait reads the host
    slot map instead, but the streamer checks that the tier's map never moved.
    ``planned`` pads the plan's expert ids to at least ``MAX_IDS`` lanes (-1 past
    the plan): the post kernel reads ``min(count, MAX_IDS)`` lanes, and ``count``
    lives on the device, so no host check can bound it.
    """

    name = "exl3_ram_miss"

    def __init__(
        self,
        segments,
        host_row_map: torch.Tensor,
        device_side: Exl3RamMissDevice,
        row: int,
        next_row: int,
        capacity: int,
        two_phase: bool = False,
        hit_wait_ns: int = 100_000,
        hot_slots: Optional[torch.Tensor] = None,
        hot_capacity: int = 0,
    ) -> None:
        super().__init__(segments, host_row_map, capacity)
        self.device_side = device_side
        self.row = row
        self.next_row = next_row
        self.routes = torch.full((capacity,), -1, dtype=torch.int64, device=host_row_map.device)
        self.planned = torch.full((max(capacity, MAX_IDS),), -1, dtype=torch.int64, device=host_row_map.device)
        # Task 6 V1. Off builds the Task 5 batched chain, which is the A1 arm every V1 measurement is reported
        # against, so both chains have to exist in one build.
        self.two_phase = two_phase
        self.hit_wait_ns = hit_wait_ns
        self.hot_slots = hot_slots
        self.hot_capacity = hot_capacity

    @property
    def delivered_count(self) -> torch.Tensor:
        if self.device_side.go_count is None:
            raise RuntimeError("EXL3 DIRECT requires leased delivery")
        return self.device_side.go_total if self.two_phase else self.device_side.go_count

    def _stage_planned(self, plan) -> None:
        """Copy the plan's lane experts into the captured ``planned`` buffer, bounding the plan first.

        Every chain needs this and none of them can skip it: the post kernel reads each lane's expert
        out of ``planned``, which is otherwise still at the -1 fill it was allocated with, and a -1 lane
        is rejected by the ``expert >= 0`` guard. It lives here rather than in ``translate`` because the
        two-phase chain does not call ``translate``; when the copy lived there, two-phase posted a lane
        array of -1 and fail-stopped on its first layer.

        The bound check belongs with the copy for the same reason. Kept here, a plan wider than the
        buffer raises at capture; left in ``translate``, two-phase degrades it to a device-side
        ``kLeaseReasonCount`` fail-stop.
        """
        lanes = plan.expert_ids.numel()
        if lanes > self.planned.numel():
            raise ValueError(f"a plan of {lanes} lanes does not fit the backend's {self.planned.numel()}")
        self.planned[:lanes].copy_(plan.expert_ids)  # a device copy: captured, refreshed every replay

    def translate(self, tag, plan) -> None:
        self._stage_planned(plan)
        self.device_side.post(
            self.row, self.planned, plan.count, self.routes, self.next_row,
            self.hot_slots, self.hot_capacity,
        )
        self.device_side.wait(self.row, self.planned, plan.count, self.host_rows, self.keep, self.ram_miss)

    def post(self, tag, plan) -> None:
        """Without leases: the inherited translate then copy of ``plan.count`` rows.

        With them (LEASE_PROTOCOL.md 7): the copy's active count is the wait kernel's ``go_count``, not the plan's
        ``count`` (zero on any refusal, so a refused request reads nothing), and the acknowledgement kernel follows the
        copy in the same stream. Whether a record is armed is decided by the post kernel and read back by the service
        from the record, so no arming decision is made here.
        """
        if self.device_side.lease_block is None:
            super().post(tag, plan)
            return
        if not self.two_phase:
            self.translate(tag, plan)
            copy_expert_row_segments_gpu(self.segments[tag], self.host_rows, plan.slots, self.device_side.go_count)
            self.device_side.ack(self.keep)
            return
        # V1 two-phase (D7): post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F, one linear chain in one stream. Each
        # stage copies its own compacted plan, so each copy takes that stage's source rows, destination slots and
        # committed count rather than the plan's lane-ordered arrays.
        self._stage_planned(plan)
        self.device_side.post(
            self.row, self.planned, plan.count, self.routes, self.next_row,
            self.hot_slots, self.hot_capacity,
        )
        self.device_side.hit_wait(self.row, self.planned, plan.count, plan.slots, self.hit_wait_ns)
        copy_expert_row_segments_gpu(
            self.segments[tag], self.device_side.host_rows_1, self.device_side.dst_slots_1, self.device_side.go_1
        )
        self.device_side.stage_ack(1)
        self.device_side.rest_wait(self.row, self.planned, plan.count, plan.slots, self.ram_miss)
        copy_expert_row_segments_gpu(
            self.segments[tag], self.device_side.host_rows_2, self.device_side.dst_slots_2, self.device_side.go_2
        )
        self.device_side.stage_ack(2)
        self.device_side.finalize(plan.count, self.keep)
        # Lanes each stage copied into plan.slots; finalize keeps only a request whose stages copied every lane.
        torch.add(self.device_side.go_1, self.device_side.go_2, out=self.device_side.go_total)


def watchdog_wait_s(timeout_ms: int) -> float:
    """The C++ watchdog's abort limit for a wait timeout of ``timeout_ms``: max(30 s, 3 x timeout).

    It must outlast the device wait (a slow demand fails stop cleanly at the timeout, not
    by abort) and the eager pause bound, 2 x timeout + 1 s (``before_host_use``). Three
    times the timeout exceeds both for any timeout over 1 s; 30 s covers the rest.
    """
    return max(30.0, 3.0 * timeout_ms / 1000)


def parse_fault(spec: str) -> Optional[tuple[int, float]]:
    """``SGLANG_TEST_DSV41_RAM_MISS_FAULT`` = ``"<demands>:<seconds>"``: delay each demand
    read by ``seconds`` once ``demands`` demands have read rows. Empty: no fault."""
    if not spec:
        return None
    demands, seconds = spec.split(":")
    return int(demands), float(seconds)


class Exl3RamMissService:
    """Process-wide option C service: the C++ thread, the device kernels, the hooks."""

    _instance: "Optional[Exl3RamMissService]" = None

    @classmethod
    def get(cls) -> "Exl3RamMissService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.tables: dict[int, NativePinnedSlotTable] = {}
        self.host: Optional[Exl3RamMissHost] = None
        self.device_side: Optional[Exl3RamMissDevice] = None
        self.page = None
        self.slot_map = None
        self._rows: dict[int, int] = {}
        self._manager = None
        # The only caller of host.pause()/resume(), which are not reentrant.
        self._pause_depth = 0
        self._trace_rows: Optional[list[int]] = None
        self._trace_graph: Optional[list[int]] = None
        # One reusable, pinned snapshot. A busy slot coalesces later cumulative
        # register values; neither the scheduler nor the CUDA graph waits for it.
        self._trace_snapshot: Optional[torch.Tensor] = None
        self._trace_event: Optional[torch.cuda.Event] = None
        self._trace_pending_rows: Optional[list[int]] = None
        self._stages_traced = False  # the host records a stage line per request (trace runs only)
        self._stages_dropped = 0
        # Routed rows of one bs-1 decode step over every in-graph layer (attach sums it).
        self.routed_rows_per_step = 0
        self._shut_down = False
        self._quarantined = False
        self._completed = False
        # Fixed once, in ensure_started, from SGLANG_DSV41_ENABLE_RAM_MISS_LEASES: the host and the device are both
        # configured from this one field, so they cannot disagree about whether a record is leased.
        self.lease_mode = False
        # Task 6 V1, fixed in ensure_started beside lease_mode so the host, the device and every backend cannot
        # disagree about which chain this process runs.
        self.two_phase = False
        self.piece_stream = False
        self.hit_wait_ns = 100_000
        self.hot_page = None
        self.gpu_hot_enabled = False
        self._gpu_hot_updater = None
        self._hot_snapshot = None
        self._hot_lists: dict[int, list[int]] = {}

    def register(self, layer_id: int, table: NativePinnedSlotTable) -> None:
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: a pinned tier was built after the service started")
        self.tables[layer_id] = table

    def row_of(self, layer_id: int) -> int:
        return self._rows[layer_id]

    def _refuse_if_shut_down(self) -> None:
        # shutdown() closes the C++ host but keeps self.host, whose calls would then
        # fail with an opaque "unknown handle".
        if self._shut_down:
            raise RuntimeError("exl3 RAM miss: the option C service was shut down; its pinned tiers are closed")

    def ensure_started(self) -> None:
        self._refuse_if_shut_down()
        if self.host is not None:
            return
        cfg = Dsv41Config.from_envs()
        streamers = {layer_id: table.streamer_of() for layer_id, table in sorted(self.tables.items())}
        missing = [layer_id for layer_id, s in streamers.items() if s is None or s.pinned_host_cache is None]
        if missing:
            raise RuntimeError(f"exl3 RAM miss: layers {missing} have no pinned tier yet")
        fmt = next(iter(streamers.values())).format
        # The mirror roots and weights the eager row source reads with, validated by the same code.
        tables = exl3_ram_miss_tables(
            fmt.layout,
            fmt.segment_map(),
            {layer_id: s.pinned_host_cache.tensors for layer_id, s in streamers.items()},
            **fmt.mirror_table_args(),
        )
        pin = torch.cuda.is_available()
        page = new_page(pin=pin)
        hot_page = new_hot_page(tables.starts.shape[1], pin=pin)
        slot_map = torch.full(tuple(tables.starts.shape), -1, dtype=torch.int32)
        slot_map = slot_map.pin_memory() if pin else slot_map
        host = Exl3RamMissHost(
            tables,
            page=page,
            slot_map=slot_map,
            direct=fmt._resolve_direct(),
            hot_page=hot_page,
            pack_workers=cfg.ram_miss_pack_workers,
        )
        try:
            from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

            lease_mode = cfg.enable_ram_miss_leases
            # Two-phase is a mode of lease mode, not an independent one: without leases there are no row results to
            # publish early, so it is refused rather than silently ignored.
            two_phase = cfg.enable_ram_miss_two_phase
            if two_phase and not lease_mode:
                raise RuntimeError(
                    "exl3 RAM miss: SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE needs SGLANG_DSV41_ENABLE_RAM_MISS_LEASES"
                )
            # Piece streaming is a mode of two-phase: the inline no-pool pack path (pack_workers == 0) has no
            # publisher for a piece job, so it is refused rather than silently packing whole rows.
            piece_stream = cfg.enable_ram_miss_piece_stream
            if piece_stream and not (two_phase and lease_mode and cfg.ram_miss_pack_workers > 0):
                raise RuntimeError(
                    "exl3 RAM miss: SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM needs "
                    "SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE, SGLANG_DSV41_ENABLE_RAM_MISS_LEASES and "
                    "SGLANG_DSV41_RAM_MISS_PACK_WORKERS > 0"
                )
            if lease_mode:
                host.enable_lease_mode()  # before the thread starts (the host refuses it afterwards)
            if two_phase:
                host.enable_two_phase()
            if get_exl3_stream_trace().enabled:
                host.enable_trace()  # before the thread starts: without a trace file it takes no timestamps
                self._stages_traced = True
            host.start_thread(fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms))
            fault = parse_fault(cfg.ram_miss_fault)
            if fault is not None:
                demands, seconds = fault
                host.inject(delay_s=seconds, delay_after_demands=demands)
                logger.warning("exl3 RAM miss TEST FAULT: demand reads sleep %.1f s after %d demands", seconds, demands)
        except BaseException:
            # The thread writes into the tiers' slabs through raw addresses: join it
            # before anything can release them.
            host.stop()
            raise
        self.page, self.slot_map, self.host, self.lease_mode = page, slot_map, host, lease_mode
        self.hot_page = hot_page
        self.two_phase = two_phase
        self.piece_stream = piece_stream
        self.hit_wait_ns = cfg.ram_miss_hit_wait_us * 1000
        # Order matters: atexit runs last-registered first, and weakref.finalize installs its single exit hook when the
        # first finalizer (any tier's slab unregister) is created. Registering HERE, after every tier exists (a tier built
        # later is refused by register()), makes this run before that hook, so the slabs are quarantined and their
        # finalizers detached before they could unregister them. Move this earlier and the quarantine is silently undone.
        atexit.register(_quarantine_service_at_exit, weakref.ref(self))
        self._rows = {layer_id: row for row, layer_id in enumerate(tables.layer_ids)}
        logger.info(
            "exl3 RAM miss thread started: %d layers, %d files, slot bytes %d, wait timeout %d ms, leases %s",
            len(tables.layer_ids), len(tables.paths), tables.slot_bytes, cfg.ram_miss_timeout_ms,
            "on" if lease_mode else "off",
        )

    def before_host_use(self) -> None:
        """Eager pinned-tier use: finish queued device work, then pause the thread (nesting counted)."""
        self.ensure_started()
        if self._pause_depth == 0:
            if self.gpu_hot_enabled:
                self._hot_snapshot.copy_(self._gpu_hot_updater.slot_to_expert, non_blocking=True)
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                with (
                    torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_stream_sync")
                    if _SYNC_WAIT_NVTX
                    else nullcontext()
                ):
                    torch.cuda.current_stream().synchronize()
            with (
                torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_thread_pause")
                if _SYNC_WAIT_NVTX
                else nullcontext()
            ):
                self.host.pause(2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0)
            if self.gpu_hot_enabled:
                self._refresh_hot_lists()
        self._pause_depth += 1

    def after_host_use(self) -> None:
        self._pause_depth -= 1
        if self._pause_depth == 0:
            self.host.resume()

    def attach(self, manager, streamer) -> None:
        """The format's ``attach_hot_cache_manager``: hooks once, a row backend per layer."""
        self.ensure_started()
        if self._manager is None:
            self._manager = manager
            manager.register_fail_stop_check(self.fail_stop_check)
            updater = getattr(manager, "gpu_residency", None)
            if updater is None or not updater.insert_direct:
                manager.add_residency_listener(self.on_residency)
            else:
                self._enable_gpu_hot(updater)
        if not getattr(streamer, "_graph_pinned_tier", False):
            return
        if streamer.graph_gather_rows > MAX_IDS:
            # The post kernel requests min(count, MAX_IDS) lanes, so a plan can carry lanes the service is
            # never asked for; the wait kernel then fail-stops on the first of them that is not in RAM.
            raise ValueError(
                f"exl3 RAM miss: layer {streamer.layer_id} gathers up to {streamer.graph_gather_rows} rows "
                f"per call but the service requests at most {MAX_IDS} lanes"
            )
        cache = streamer.hot_cache
        if self.device_side is None:
            from sglang.srt.layers.moe.exl3_expert_format import prefetch_enabled

            self.device_side = Exl3RamMissDevice(
                self.page,
                self.slot_map,
                device=cache.device,
                layers=len(self._rows),
                timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
                advise=prefetch_enabled(),
                # The host owns the block (Exl3RamMissHost allocates it); the device reads the same one.
                lease_block=self.host.lease_block if self.lease_mode else None,
                lease_layout=self.host.lease_layout if self.lease_mode else None,
                hot_page=self.hot_page if self.gpu_hot_enabled else None,
                piece_stream=self.piece_stream,
            )
        self.routed_rows_per_step += streamer.graph_gather_rows
        row = self.row_of(streamer.layer_id)
        next_row = row + 1 if row + 1 < len(self._rows) else -1
        previous = streamer.row_backend
        streamer.row_backend = Exl3RamMissRowBackend(
            previous.segments, previous.host_row_map, self.device_side, row, next_row, streamer.graph_gather_rows,
            two_phase=self.two_phase, hit_wait_ns=self.hit_wait_ns,
            hot_slots=(manager.gpu_residency.slot_to_expert[manager.gpu_residency.layer_ids.index(streamer.layer_id)]
                       if self.gpu_hot_enabled else None),
            hot_capacity=cache.capacity if self.gpu_hot_enabled else 0,
        )

    def _refresh_hot_lists(self) -> None:
        updater = self._gpu_hot_updater
        for row, layer_id in enumerate(updater.layer_ids):
            capacity = updater.caches[row].capacity
            self._hot_lists[layer_id] = [int(e) for e in self._hot_snapshot[row, :capacity].tolist() if e >= 0]

    def _enable_gpu_hot(self, updater) -> None:
        if not self.lease_mode:
            raise ValueError("EXL3 DIRECT requires RAM-miss leases")
        self._gpu_hot_updater = updater
        self._hot_snapshot = torch.empty_like(
            updater.slot_to_expert, device="cpu", pin_memory=updater.device.type == "cuda"
        )
        self._hot_snapshot.copy_(updater.slot_to_expert, non_blocking=True)
        if updater.device.type == "cuda":
            torch.cuda.current_stream(updater.device).synchronize()
        self._refresh_hot_lists()
        for layer_id, experts in self._hot_lists.items():
            self.host.set_hot(self.row_of(layer_id), experts)
        self.host.enable_gpu_hot()
        self.gpu_hot_enabled = True

    def hot_experts(self, layer_id: int) -> list[int]:
        return self._hot_lists.get(layer_id, [])

    def on_residency(self, layer_id: int, slot_to_expert: list[int]) -> None:
        self._refuse_if_shut_down()
        if layer_id in self._rows:
            self.host.set_hot(self.row_of(layer_id), slot_to_expert)

    def fail_stop_check(self) -> None:
        """Per batch (the scheduler's doorbell hook): raise when a wait timed out or failed.

        This is also where a warmup or capture timeout surfaces: no check runs between
        capture and the first batch, so a fatal raised during warmup stops the process
        at the first batch's check (an eager prefill, so no dropped-layer token is
        served), not at startup as plan D20 words it.
        """
        if self.host is None or self._shut_down:
            return
        fatal = self.host.fatal_seq()
        if fatal:
            raise RuntimeError(
                f"exl3 RAM miss: request {fatal} timed out or failed "
                f"(thread {self.host.counters()}); fail-stop"
            )
        self._trace_step()
        self._trace_stages()

    def _trace_stages(self) -> None:
        """The stage records the service produced since the last check, into the stream trace."""
        if not self._stages_traced:
            return
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        get_exl3_stream_trace().record_ram_miss_requests(
            self.host.drain_trace(), sorted(self._rows, key=self._rows.__getitem__)
        )
        dropped = self.host.trace_dropped()
        if dropped > self._stages_dropped:
            logger.warning("exl3 RAM miss: %d stage records dropped (ring full)", dropped - self._stages_dropped)
            self._stages_dropped = dropped

    def _graph_rows(self, rows: list[int], *, final: bool = False) -> Optional[tuple[list[int], list[int]]]:
        """Poll a pinned graph-counter readback and queue the next one without waiting.

        The manager's registers survive the forward observer's zeroing of each
        streamer's graph counters. Cumulative snapshots may coalesce while the
        sole pinned slot is in flight. ``final`` runs only after shutdown's GPU
        barrier; it drains the slot and takes the last register value.
        """
        registers = getattr(self._manager, "_registers", None) or {}
        decode = registers.get("decode")
        if decode is None:
            return None
        source = decode["graph_rows"]
        if not source.is_cuda:
            return [int(value) for value in source.sum(dim=0).tolist()], rows

        if not final and torch.cuda.is_current_stream_capturing():
            return None
        ready = None
        if self._trace_pending_rows is not None:
            if final:
                self._trace_event.synchronize()
            elif not self._trace_event.query():
                return None
            ready = ([int(value) for value in self._trace_snapshot.tolist()], self._trace_pending_rows)
            self._trace_pending_rows = None
        if not final:
            if self._trace_snapshot is None:
                self._trace_snapshot = torch.empty(2, dtype=torch.int64, pin_memory=True)
                self._trace_event = torch.cuda.Event(enable_timing=False)
            self._trace_snapshot.copy_(source.sum(dim=0), non_blocking=True)
            self._trace_event.record(torch.cuda.current_stream(source.device))
            self._trace_pending_rows = rows
        return ready

    def _record_graph_snapshot(self, trace, graph: list[int], rows: list[int]) -> None:
        # A register reset (discard_graph_capture_routes) moves the totals back: re-baseline.
        if self._trace_graph is not None and graph[0] > self._trace_graph[0]:
            routed = graph[0] - self._trace_graph[0]
            # A lagged read can fold multiple bs-1 decode graph steps into one line.
            steps = max(1, round(routed / self.routed_rows_per_step)) if self.routed_rows_per_step else 1
            trace.record_graph_step(
                layer_rows_delta=[a - b for a, b in zip(rows, self._trace_rows)],
                routed_rows=routed,
                routed_misses=graph[1] - self._trace_graph[1],
                thread=self.host.counters(),
                steps=steps,
            )
        self._trace_rows, self._trace_graph = rows, graph

    def _trace_step(self, *, final: bool = False) -> None:
        """One graph decode step's G and RAM misses into the stream trace (trace runs only)."""
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        trace = get_exl3_stream_trace()
        if not trace.enabled:
            return
        rows = self.host.layer_rows()  # demand rows only: advisory reads are not misses
        sample = self._graph_rows(rows, final=final)
        if sample is not None:
            self._record_graph_snapshot(trace, *sample)
        if final and self._trace_snapshot is not None:
            # The orderly shutdown barrier already finished all graph work. This
            # last copy captures registers updated after the prior snapshot.
            registers = self._manager._registers["decode"]
            final_graph = [int(value) for value in registers["graph_rows"].sum(dim=0).tolist()]
            self._record_graph_snapshot(trace, final_graph, rows)

    def _cuda_active(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_initialized()

    def _synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def _barrier_devices(self) -> list[torch.device]:
        """The CUDA devices whose kernels may read the slabs, each with an explicit index. Chosen on the CALLING thread:
        a helper thread starts on device 0, and a device-less synchronize there waits for the wrong device on any rank
        that serves another GPU (and would resolve an index-less ``cuda`` to device 0 the same way)."""
        candidates = []
        if self.device_side is not None:
            candidates.append(self.device_side.state.device)
        for layer_id in sorted(self.tables):
            tier = getattr(self.tables[layer_id].streamer_of(), "pinned_host_cache", None)
            if tier is not None:
                candidates.append(tier.device)
        devices: list[torch.device] = []
        for device in candidates:
            if device.type != "cuda":
                continue
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device not in devices:
                devices.append(device)
        return devices or [torch.device("cuda", torch.cuda.current_device())]

    def _completion_deadline_s(self) -> float:
        # A wait kernel is bounded by the wait timeout (and, once admission is closed, by the header word).
        return envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 5.0

    def _establish_gpu_completion(self) -> Optional[str]:
        """None when every GPU reader is known to have finished, else why that could not be established.

        The device-wide synchronize runs on a helper thread with a deadline: it is not interruptible, so a hung
        kernel would otherwise hang shutdown with no way to say "uncertain".
        """
        if not self._cuda_active():
            return None
        outcome: dict = {}
        devices = self._barrier_devices()

        def run() -> None:
            try:
                for device in devices:
                    self._synchronize(device)
                outcome["done"] = True
            except BaseException as error:  # noqa: BLE001 - any CUDA error means completion is not established
                outcome["error"] = error

        deadline = self._completion_deadline_s()
        helper = threading.Thread(target=run, daemon=True, name="exl3-shutdown-sync")
        helper.start()
        helper.join(deadline)
        if helper.is_alive():
            return f"the device synchronize did not return within {deadline:.1f} s"
        if "error" in outcome:
            return f"the device synchronize failed: {outcome['error']!r}"
        return None

    def shutdown(self, *, at_exit: bool = False) -> None:
        """Stop admission, establish that no GPU reader runs, then free; else quarantine (LEASE_PROTOCOL.md 14.3).

        Idempotent. The tiers must not be used afterwards. ``at_exit``: no device barrier is attempted in an exit
        handler, so the tiers are quarantined unconditionally. The scheduler's graceful shutdown calls the orderly path
        through ``shutdown_exl3_ram_miss_service`` (LEASE_PROTOCOL.md 20.2i); the exit hook calls this with ``at_exit``.
        Anything that leaves it unknown whether the service thread or a GPU reader still runs (the barrier, or
        stopping the thread) makes the shutdown quarantine; it frees only when both are known to be finished.
        """
        if self._completed or (self._shut_down and not at_exit):
            return
        # At exit a shutdown that started and did not complete is finished as a quarantine, never left half done.
        uncertain: Optional[str] = "an earlier shutdown did not complete" if self._shut_down else None
        self._shut_down = True
        stop_error: Optional[BaseException] = None
        interrupt: Optional[BaseException] = None  # a KeyboardInterrupt or SystemExit: re-raised once the slabs are safe
        try:
            if self.host is not None:
                self.host.close_admission()
                if uncertain is None:
                    uncertain = (
                        "process exit: no device barrier is attempted" if at_exit else self._establish_gpu_completion()
                    )
                if uncertain is None and self._stages_traced:
                    # host.stop() closes its native handle, so freeze the reader
                    # between requests before taking its final demand counters.
                    # The GPU barrier above has retired every demand and lease.
                    self.host.pause(2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0)
                    self._trace_step(final=True)
        except BaseException as error:  # noqa: BLE001 - a failure to even close admission is an uncertain state
            uncertain = f"closing admission or the barrier failed: {error!r}"
            if not isinstance(error, Exception):
                interrupt = error
        finally:
            try:
                # The service thread writes into the slabs through raw addresses, so it stops before anything is freed.
                # A service thread hung in a read ends this in the service watchdog's abort (see LEASE_PROTOCOL 20.2j).
                if self.host is not None:
                    logger.info(
                        "exl3 RAM miss: stopping the service thread; a read that hangs ends in the watchdog's abort "
                        "after max(30 s, 3 x the wait timeout)"
                    )
                    self.host.stop()
            except BaseException as error:  # noqa: BLE001 - a thread that may still run must not have its slabs freed
                stop_error = error
                uncertain = uncertain or f"stopping the service thread failed: {error!r}"
                if not isinstance(error, Exception):
                    interrupt = interrupt or error
            finally:
                if uncertain is None:
                    for layer_id in sorted(self.tables):
                        streamer = self.tables[layer_id].streamer_of()
                        tier = getattr(streamer, "pinned_host_cache", None)
                        if tier is not None:
                            tier.close()
                else:
                    self._quarantine(uncertain)
                self._completed = True
        if interrupt is not None:
            raise interrupt  # KeyboardInterrupt and SystemExit go on, after the quarantine
        if stop_error is not None:
            logger.error("exl3 RAM miss: stopping the service thread failed: %r", stop_error)

    def _quarantine(self, reason: str) -> None:
        """Keep everything a GPU kernel may still read or write alive until the process ends; free nothing."""
        logger.error(
            "exl3 RAM miss: shutdown could not establish that no GPU reader is running (%s); the pinned slabs, the "
            "request page, the slot map, the lease block and the device buffers are quarantined until process exit",
            reason,
        )
        owned: list[torch.Tensor] = []
        if self.host is not None:
            owned += [self.host.page, self.host.slot_map, self.host.lease_block]
            if self.host.hot_page is not None:
                owned.append(self.host.hot_page)
        if self.device_side is not None:
            owned += [self.device_side.state, self.device_side.last_routes]
            owned += [t for t in (self.device_side.go_count, self.device_side.lane_ctx) if t is not None]
            owned += [
                t
                for t in (
                    self.device_side.go_1,
                    self.device_side.go_2,
                    self.device_side.go_total,
                    self.device_side.lane_ctx_1,
                    self.device_side.lane_ctx_2,
                    self.device_side.host_rows_1,
                    self.device_side.host_rows_2,
                    self.device_side.dst_slots_1,
                    self.device_side.dst_slots_2,
                    self.device_side.origin_1,
                    self.device_side.origin_2,
                    self.device_side.claimed,
                    self.device_side.violated,
                )
                if t is not None
            ]
        if self._trace_snapshot is not None:
            # A copy can still be writing this pinned block when the device
            # barrier failed. Keep it alive with the other uncertain buffers.
            owned.append(self._trace_snapshot)
        if self._hot_snapshot is not None:
            owned.append(self._hot_snapshot)
        if self._gpu_hot_updater is not None:
            owned.append(self._gpu_hot_updater.slot_to_expert)
        for layer_id in sorted(self.tables):
            streamer = self.tables[layer_id].streamer_of()
            tier = getattr(streamer, "pinned_host_cache", None)
            if tier is not None:
                tier.quarantine()
            backend = getattr(streamer, "row_backend", None)
            for name in ("host_rows", "ram_miss", "keep", "routes", "planned", "hot_slots"):
                tensor = getattr(backend, name, None)
                if isinstance(tensor, torch.Tensor):
                    owned.append(tensor)
        quarantine_host_slabs(owned)
        self._quarantined = True


def shutdown_exl3_ram_miss_service() -> None:
    """The scheduler's graceful-shutdown entry (LEASE_PROTOCOL.md 20.2i): shut the service down if one exists.

    A no-op when no service was ever created: a run without EXL3 must not construct the singleton at shutdown.
    """
    service = Exl3RamMissService._instance
    if service is not None:
        service.shutdown()
