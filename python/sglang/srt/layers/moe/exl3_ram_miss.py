"""Option C for EXL3 streamed experts (plan D8-D23).

``exl3_ram_miss_tables`` flattens what ``Exl3ShardRowSource`` knows (the per-expert
superset reads and the per-name segment map) plus each layer's pinned slabs into
int64 tensors, so the C++ thread reads and splits rows without Python.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch

from sglang.kernels.ops.moe.exl3_ram_miss import MAX_IDS, Exl3RamMissDevice, Exl3RamMissHost, new_page
from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Exl3RamMissTables:
    layer_ids: list[int]
    paths: list[str]
    file_sizes: torch.Tensor  # int64 [F]
    reads: torch.Tensor  # int64 [L, E, 4]: file index, aligned offset, aligned length, row start
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
) -> Exl3RamMissTables:
    """Tables for the streamed layers in ``slabs_by_layer`` (ascending layer id = row order)."""
    layer_ids = sorted(slabs_by_layer)
    paths: list[str] = []
    file_index: dict[str, int] = {}
    reads = torch.empty((len(layer_ids), layout.num_experts, 4), dtype=torch.int64)
    widest = 0
    for row, layer_id in enumerate(layer_ids):
        for expert in range(layout.num_experts):
            record = layout.records[(layer_id, expert)]
            if record.path not in file_index:
                file_index[record.path] = len(paths)
                paths.append(record.path)
            offset, length, start = record.aligned_read(PAGE_BYTES)
            reads[row, expert] = torch.tensor([file_index[record.path], offset, length, start])
            widest = max(widest, length)
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
        file_sizes=torch.tensor([os.path.getsize(p) for p in paths], dtype=torch.int64),
        reads=reads,
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
        row = self._row
        mapping = self.service.host.mapping(row)
        return OrderedDict((expert, mapping[expert]) for expert in self.service.host.lru_order(row))

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
        level only, refresh the device slot map if the thread changed the C++ map."""
        self.service.before_host_use()
        try:
            if self._depth == 0:
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
    ) -> None:
        super().__init__(segments, host_row_map, capacity)
        self.device_side = device_side
        self.row = row
        self.next_row = next_row
        self.routes = torch.full((capacity,), -1, dtype=torch.int64, device=host_row_map.device)
        self.planned = torch.full((max(capacity, MAX_IDS),), -1, dtype=torch.int64, device=host_row_map.device)
        if self.planned.numel() < MAX_IDS:
            raise ValueError(f"planned has {self.planned.numel()} lanes; the post kernel reads up to {MAX_IDS}")

    def translate(self, tag, plan) -> None:
        lanes = plan.expert_ids.numel()
        if lanes > self.planned.numel():
            raise ValueError(f"a plan of {lanes} lanes does not fit the backend's {self.planned.numel()}")
        self.planned[:lanes].copy_(plan.expert_ids)  # a device copy: captured, refreshed every replay
        self.device_side.post(self.row, self.planned, plan.count, self.routes, self.next_row)
        self.device_side.wait(self.row, self.planned, plan.count, self.host_rows, self.keep, self.ram_miss)


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
        self._shut_down = False

    def register(self, layer_id: int, table: NativePinnedSlotTable) -> None:
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: a pinned tier was built after the service started")
        self.tables[layer_id] = table

    def row_of(self, layer_id: int) -> int:
        return self._rows[layer_id]

    def ensure_started(self) -> None:
        if self.host is not None:
            return
        if self._shut_down:
            raise RuntimeError("exl3 RAM miss: the service was shut down")
        streamers = {layer_id: table.streamer_of() for layer_id, table in sorted(self.tables.items())}
        missing = [layer_id for layer_id, s in streamers.items() if s is None or s.pinned_host_cache is None]
        if missing:
            raise RuntimeError(f"exl3 RAM miss: layers {missing} have no pinned tier yet")
        fmt = next(iter(streamers.values())).format
        tables = exl3_ram_miss_tables(
            fmt.layout, fmt.segment_map(), {layer_id: s.pinned_host_cache.tensors for layer_id, s in streamers.items()}
        )
        pin = torch.cuda.is_available()
        page = new_page(pin=pin)
        slot_map = torch.full(tuple(tables.reads.shape[:2]), -1, dtype=torch.int32)
        slot_map = slot_map.pin_memory() if pin else slot_map
        host = Exl3RamMissHost(tables, page=page, slot_map=slot_map, direct=fmt._resolve_direct())
        try:
            host.start_thread(fatal_wait_s=30.0)
            fault = parse_fault(envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get())
            if fault is not None:
                demands, seconds = fault
                host.inject(delay_s=seconds, delay_after_demands=demands)
                logger.warning("exl3 RAM miss TEST FAULT: demand reads sleep %.1f s after %d demands", seconds, demands)
        except BaseException:
            # The thread writes into the tiers' slabs through raw addresses: join it
            # before anything can release them.
            host.stop()
            raise
        self.page, self.slot_map, self.host = page, slot_map, host
        self._rows = {layer_id: row for row, layer_id in enumerate(tables.layer_ids)}
        logger.info(
            "exl3 RAM miss thread started: %d layers, %d files, slot bytes %d, wait timeout %d ms",
            len(tables.layer_ids), len(tables.paths), tables.slot_bytes, envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
        )

    def before_host_use(self) -> None:
        """Eager pinned-tier use: finish queued device work, then pause the thread (nesting counted)."""
        self.ensure_started()
        if self._pause_depth == 0:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.current_stream().synchronize()
            self.host.pause(2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0)
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
            manager.add_residency_listener(self.on_residency)
        if not getattr(streamer, "_graph_pinned_tier", False):
            return
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
            )
        row = self.row_of(streamer.layer_id)
        next_row = row + 1 if row + 1 < len(self._rows) else -1
        previous = streamer.row_backend
        streamer.row_backend = Exl3RamMissRowBackend(
            previous.segments, previous.host_row_map, self.device_side, row, next_row, streamer.graph_gather_rows
        )

    def on_residency(self, layer_id: int, slot_to_expert: list[int]) -> None:
        if layer_id in self._rows:
            self.host.set_hot(self.row_of(layer_id), slot_to_expert)

    def fail_stop_check(self) -> None:
        """Per batch (the scheduler's doorbell hook): raise when a wait timed out or failed."""
        if self.host is None:
            return
        fatal = self.host.fatal_seq()
        if fatal:
            raise RuntimeError(
                f"exl3 RAM miss: request {fatal} timed out or failed "
                f"(thread {self.host.counters()}); fail-stop"
            )
        self._trace_step()

    def _graph_rows(self) -> Optional[list[int]]:
        """Routed rows and routed misses of every decode graph gather so far, from the
        manager's registers (the streamers' own graph_counters are zeroed every forward
        by the forward observer, before this per-batch check runs; plan D23)."""
        registers = getattr(self._manager, "_registers", None) or {}
        decode = registers.get("decode")
        if decode is None:
            return None
        return [int(value) for value in decode["graph_rows"].sum(dim=0).tolist()]

    def _trace_step(self) -> None:
        """One graph decode step's G and RAM misses into the stream trace (trace runs only)."""
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        trace = get_exl3_stream_trace()
        if not trace.enabled:
            return
        graph = self._graph_rows()
        if graph is None:
            return
        rows = self.host.layer_rows()  # demand rows only: advisory reads are not misses
        # A register reset (discard_graph_capture_routes) moves the totals back: re-baseline.
        if self._trace_graph is not None and graph[0] > self._trace_graph[0]:
            trace.record_graph_step(
                layer_rows_delta=[a - b for a, b in zip(rows, self._trace_rows)],
                routed_rows=graph[0] - self._trace_graph[0],
                routed_misses=graph[1] - self._trace_graph[1],
            )
        self._trace_rows, self._trace_graph = rows, graph

    def shutdown(self) -> None:
        """Join the thread, then unregister every pinned tier's slabs; idempotent.

        The thread writes into the slabs through raw addresses, so it stops first.
        The tiers must not be used afterwards.
        """
        if self._shut_down:
            return
        self._shut_down = True
        try:
            if self.host is not None:
                self.host.stop()
        finally:
            for layer_id in sorted(self.tables):
                streamer = self.tables[layer_id].streamer_of()
                tier = getattr(streamer, "pinned_host_cache", None)
                if tier is not None:
                    tier.close()
