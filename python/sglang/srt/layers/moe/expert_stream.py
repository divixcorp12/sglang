"""Selected-expert staging for host-resident MoE weight rows."""

from __future__ import annotations

import contextlib
import functools
import logging
import json
import os
import weakref
from dataclasses import asdict, dataclass, fields, replace
from operator import index
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, Sequence, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.moe import moe_side_stream
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend, _aot_transfer_available
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    graph_source_kind_of,
    iter_expert_streamers,
    pinned_tier_options_of,
    require_graph_gather_support,
    resolve_row_source_kind,
)
from sglang.srt.layers.moe.expert_host_tier import (
    PinnedGatherResult,
    PinnedRowFills,
    PinnedSlotLRU,
    PinnedSlotTable,
    allocate_host_slab,
    quarantine_host_slabs,
    release_host_slabs,
)
from sglang.srt.layers.moe.expert_row_source import (
    ExpertRowSource,
    RowReadStats,
    TensorRowSource,
)
from sglang.srt.layers.moe.expert_route_plan import NO_DEDUP_LIMIT as _NO_DEDUP_LIMIT
from sglang.srt.layers.moe.expert_route_plan import (
    plan_graph_routes,
    plan_graph_routes_fused,
    should_dedup,
    supports_fused_graph_routes,
)
from sglang.srt.layers.moe.expert_row_plan import (
    ExpertRowPlan,
    ExpertRowPlanner,
    InGraphRowBackend,
    PinnedTierRowBackend,
)
from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor

if TYPE_CHECKING:
    from sglang.srt.layers.moe.host_numa import Placement

logger = logging.getLogger(__name__)
_SYNC_WAIT_NVTX = os.environ.get("SGLANG_DSV41_SYNC_WAIT_NVTX") == "1"

_STAGING: Dict[Tuple, torch.Tensor] = {}
_PINNED_STAGING: Dict[Tuple, torch.Tensor] = {}
_PINNED_INDEX: Dict[Tuple, torch.Tensor] = {}
_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}
_LOGGED_SOURCE_SIGNATURES: set[Tuple] = set()

NVFP4_STREAM_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "g1_alphas",
    "g2_alphas",
)
# ExpertStreamer's row_source default: resolve it from SGLANG_MOE_EXPERT_ROW_SOURCE.
_DEFAULT_ROW_SOURCE = object()


@dataclass(frozen=True)
class ExpertGatherStats:
    """Actual source rows and transfer bytes after the gather's deduplication.

    All-hot slot remapping has no transfer bytes. source_bytes counts misses
    read from backing tensors, whether their source device is CPU or CUDA.
    routed_rows and routed_miss_rows count routes with multiplicity;
    unique_miss_rows counts the distinct missed experts actually gathered.
    """

    requested_rows: int = 0
    hot_hit_rows: int = 0
    miss_rows: int = 0
    d2d_bytes: int = 0
    h2d_bytes: int = 0
    source_bytes: int = 0
    pinned_host_hit_rows: int = 0
    pinned_host_miss_rows: int = 0
    pinned_host_populated_bytes: int = 0

    transfer_wait_ns: int = 0
    gather_fallback_used: bool = False
    copy_engine_bytes: int = 0
    routed_rows: int = 0
    routed_miss_rows: int = 0
    unique_miss_rows: int = 0
    # Host rows the row sources read during this gather (summed RowReadStats).
    # Trailing and defaulted: the stats are constructed positionally.
    host_read_rows: int = 0
    host_read_file_bytes: int = 0
    host_read_split_bytes: int = 0
    host_read_ns: int = 0
    host_split_ns: int = 0


def _sum_gather_stats(stats: Sequence[ExpertGatherStats]) -> ExpertGatherStats:
    """Add gather stats field by field; a fallback in any gather marks the sum."""
    values = {}
    for field in fields(ExpertGatherStats):
        items = [getattr(item, field.name) for item in stats]
        values[field.name] = any(items) if isinstance(items[0], bool) else sum(items)
    return ExpertGatherStats(**values)


@dataclass
class PinnedHostCacheStats:
    """Cumulative counters for the bounded pinned-host expert cache."""

    lookup_hits: int = 0
    lookup_misses: int = 0
    populated_rows: int = 0
    populated_bytes: int = 0
    evictions: int = 0


def _host_use(method):
    """Run a pinned-tier method inside ``self.host_use()``."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.host_use():
            return method(self, *args, **kwargs)

    return wrapper


class ExpertPinnedHostCache:
    """Bounded, on-demand pinned host rows shared by one expert layer.

    Each host tensor gets one page-aligned slab registered with CUDA and sized
    to exactly ``capacity`` rows; PyTorch's pinned allocator would round each
    slab up to a power of two. ``device`` holds the slot lookup. It defaults
    to the layer's CUDA source device, else the current CUDA device; a CPU
    ``device`` keeps the tier on the host with unregistered slabs, so it runs
    without a GPU. ``is_pinned(expert_id)`` protects experts from eviction.

    ``slot_table`` replaces the default ``PinnedSlotLRU``. Such a table chooses
    its own victims: ``is_pinned`` is then used only to size requests
    (``evictable_rows``), so the table must protect the same experts itself.

    ``row_fills`` (SGLANG_DSV41_ENABLE_PREFILL_FILLS) reads missing rows in place
    of the streamer's row source: ``ensure_rows`` reads through it, and
    ``prefetch_rows`` starts a layer's reads ahead of its chunked gather, which
    then waits per chunk for only its own rows. None keeps every read as it was.
    """

    def __init__(
        self,
        streamer: "ExpertStreamer",
        capacity: int,
        *,
        device: torch.device | str | None = None,
        is_pinned: Callable[[int], bool] | None = None,
        slot_table: PinnedSlotTable | None = None,
        placement: "Placement" = (),
        row_fills: PinnedRowFills | None = None,
    ):
        capacity = index(capacity)
        if not 0 <= capacity <= streamer.num_experts:
            raise ValueError("pinned host cache capacity must be within expert count")
        self.streamer = streamer
        self.capacity = capacity
        self.cached_names = tuple(
            spec.name for spec in streamer.specs if spec.residence == "host"
        )
        if capacity and not self.cached_names:
            raise ValueError(
                "pinned host cache requires at least one CPU source tensor"
            )
        self.bytes_per_expert = streamer.host_bytes_per_expert
        self.residency_bytes = capacity * self.bytes_per_expert
        if device is None:
            devices = {
                streamer.source(spec.name).device
                for spec in streamer.specs
                if spec.residence == "device"
            }
            if len(devices) > 1:
                raise ValueError("pinned host cache CUDA sources must share one device")
            device = next(
                iter(devices), torch.device("cuda", torch.cuda.current_device())
            )
        self.device = torch.device(device)
        register = self.device.type == "cuda"
        self.tensors: dict[str, torch.Tensor] = {}
        registered: list[torch.Tensor] = []
        try:
            for name in self.cached_names:
                spec = streamer.spec(name)
                slab = allocate_host_slab(
                    capacity,
                    spec.row_shape,
                    spec.dtype,
                    register=register,
                    placement=placement,
                )
                self.tensors[name] = slab
                if register and slab.numel():
                    registered.append(slab)
        except BaseException:
            release_host_slabs(registered)
            raise
        # Unregisters the slabs when the cache is collected, at exit, or on close().
        self._release_slabs = weakref.finalize(self, release_host_slabs, registered)
        row_source = streamer.row_source
        if row_source is not None:
            row_source.register_destinations(self.tensors.values())
        self.expert_to_slot = torch.full(
            (streamer.num_experts,), -1, dtype=torch.long, device=self.device
        )
        self.is_pinned = is_pinned
        if slot_table is not None and hasattr(slot_table, "bind_capacity"):
            # A table built before the tier's size was known (a format's
            # pinned_tier_options) learns it here.
            slot_table.bind_capacity(capacity)
        if slot_table is not None and slot_table.capacity != capacity:
            raise ValueError(
                f"pinned slot table capacity {slot_table.capacity} does not match "
                f"the tier's {capacity} rows"
            )
        self._lru = (
            slot_table
            if slot_table is not None
            else PinnedSlotLRU(capacity, is_pinned=is_pinned)
        )
        self.stats = PinnedHostCacheStats()
        self.row_fills = row_fills
        # The running prefetch's claim order by expert (fill_wait counts rows in that order); None when none runs.
        self._fill_order: dict[int, int] | None = None
        streamer.pinned_host_cache = self

    @staticmethod
    def capacity_for_budget(streamer: "ExpertStreamer", budget_bytes: int) -> int:
        """Round a pinned-host byte budget down to complete expert rows."""
        budget_bytes = index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("pinned host cache byte budget cannot be negative")
        if streamer.host_bytes_per_expert == 0:
            return 0
        return min(streamer.num_experts, budget_bytes // streamer.host_bytes_per_expert)

    @property
    def slot_to_expert(self) -> list[int]:
        return self._lru.slot_to_expert

    @property
    def _expert_to_slot(self) -> dict[int, int]:
        # ExpertHotCache._prepare_promotion reads resident slots through this.
        return self._lru.expert_to_slot

    @contextlib.contextmanager
    def host_use(self) -> Iterator[None]:
        """Hold the slot table's host use: ``before_host_use`` now, ``after_host_use`` on exit.

        Every read of the slot map or the slabs from the host, and every copy
        that reads slots chosen here, belongs inside one. Nested host uses call
        both hooks again (``lookup``, ``ensure_rows``, ``copy_rows`` and
        ``gather_rows`` each open one), so a table whose owner must pause
        counts depth and acts only on the outermost pair.
        """
        self._lru.before_host_use(self)
        try:
            yield
        finally:
            self._lru.after_host_use(self)

    def close(self) -> None:
        """Unregister the slabs; the cache must not be used afterwards."""
        self._release_slabs()

    def quarantine(self) -> None:
        """Keep every slab registered and alive until the process ends, and never unregister it.

        For when a GPU reader of unknown state may still run (LEASE_PROTOCOL.md section 14). The finalizer that
        unregisters the slabs at exit is detached, or it would undo this; ``close`` is then a no-op.
        """
        self._release_slabs.detach()
        quarantine_host_slabs(self.tensors.values())

    def evictable_rows(self) -> int:
        """Slots a request can use: the capacity minus residents ``is_pinned`` protects."""
        if self.is_pinned is None:
            return self.capacity
        return self.capacity - sum(
            1 for expert_id in self._lru.expert_to_slot if self.is_pinned(expert_id)
        )

    def _refresh_mapping(self) -> None:
        self.expert_to_slot.copy_(
            torch.tensor(
                self._lru.mapping(self.streamer.num_experts),
                dtype=torch.long,
                device=self.device,
            )
        )

    @_host_use
    def lookup(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slots and record row-level hit/miss counters."""
        if source_ids.device != self.device:
            raise ValueError(
                "selected expert IDs must use the pinned cache CUDA device"
            )
        slots = self.expert_to_slot[source_ids.long()]
        hit_mask = slots >= 0
        hit_ids = source_ids[hit_mask].tolist()
        for expert_id in hit_ids:
            self._lru.touch(int(expert_id))
        self.stats.lookup_hits += len(hit_ids)
        self.stats.lookup_misses += source_ids.numel() - len(hit_ids)
        return slots, hit_mask

    @_host_use
    def ensure_rows(
        self, source_ids: torch.Tensor, protected: Iterable[int] = ()
    ) -> None:
        """Read missing source rows into pinned slots, evicting least-recently-used rows.

        The requested experts, and any others in ``protected`` (a caller's
        whole chunk), are evicted only when nothing else can be.
        """
        if not self.cached_names or self.capacity == 0 or source_ids.numel() == 0:
            return
        requested = list(dict.fromkeys(int(value) for value in source_ids.tolist()))
        missing = [expert_id for expert_id in requested if expert_id not in self._lru]
        if not missing:
            return
        protected = frozenset(requested).union(int(value) for value in protected)
        if self.row_fills is not None:
            self._fill_rows(missing, protected)
            return
        assignments = []
        evictions = 0
        # Assignment and read are one transaction: on any failure every slot this
        # call assigned is freed, so no expert stays mapped to a slot never read.
        try:
            for expert_id in missing:
                slot, evicted = self._lru.assign(expert_id, protected)
                evictions += evicted is not None
                assignments.append((expert_id, slot))
            # More misses than slots reassign a slot within this call. Read only the
            # slot's final expert: batched file reads complete in any order.
            final_slots = {slot: expert_id for expert_id, slot in assignments}
            source_ids_cpu = torch.tensor(list(final_slots.values()), dtype=torch.long)
            slots_cpu = torch.tensor(list(final_slots), dtype=torch.long)
            self.streamer.read_host_rows(
                source_ids_cpu,
                {name: self.tensors[name] for name in self.cached_names},
                slots_cpu,
            )
        except BaseException:
            for slot in dict.fromkeys(slot for _, slot in assignments):
                self._lru.release(slot)
            self._refresh_mapping()
            raise
        self._refresh_mapping()
        self.stats.evictions += evictions
        self.stats.populated_rows += len(final_slots)
        self.stats.populated_bytes += len(final_slots) * self.bytes_per_expert

    def _fill_rows(self, missing: list[int], protected: frozenset[int]) -> None:
        """``ensure_rows`` through ``row_fills``: claim and read ``missing`` now, after any prefetch has ended."""
        self.finish_fills()
        slots, evictions = self.row_fills.fill_begin(missing, protected, True)
        landed = self.row_fills.fill_end()
        self._refresh_mapping()
        if not landed:
            raise RuntimeError(f"reading pinned host rows of experts {missing[: len(slots)]} failed")
        if len(slots) < len(missing):
            raise RuntimeError("every pinned host slot holds a protected or leased expert")
        self.stats.evictions += evictions
        self.stats.populated_rows += len(slots)
        self.stats.populated_bytes += len(slots) * self.bytes_per_expert

    @_host_use
    def prefetch_rows(self, expert_ids: Sequence[int], protected: Iterable[int]) -> int:
        """Start reading the experts of ``expert_ids`` that are not resident, in that order; returns how many.

        ``row_fills`` only, inside a host use the caller holds until ``finish_fills``: the reads run while the
        caller gathers, and ``gather_rows`` waits per chunk for only its own rows. Slots are claimed until one has
        no victim outside ``protected``; the rest are left to ``gather_rows``' own admission.
        """
        self.finish_fills()
        missing = [expert_id for expert_id in dict.fromkeys(int(e) for e in expert_ids) if expert_id not in self._lru]
        if not missing:
            return 0
        slots, evictions = self.row_fills.fill_begin(missing, [int(e) for e in protected], False)
        self._fill_order = {expert_id: position for position, expert_id in enumerate(missing[: len(slots)])}
        self._refresh_mapping()
        self.stats.evictions += evictions
        self.stats.populated_rows += len(slots)
        self.stats.populated_bytes += len(slots) * self.bytes_per_expert
        return len(slots)

    def finish_fills(self) -> None:
        """Join the running prefetch, if any; raise if it failed (its rows that did not land were released)."""
        if self._fill_order is None:
            return
        self._fill_order = None
        if not self.row_fills.fill_end():
            self._refresh_mapping()
            raise RuntimeError("a prefetch of pinned host rows failed")

    def _await_fills(self, expert_ids: Iterable[int]) -> None:
        """Wait until the prefetched rows among ``expert_ids`` are in their slabs."""
        order = self._fill_order
        if order is None:
            return
        rows = max((order.get(expert_id, -1) for expert_id in expert_ids), default=-1) + 1
        if rows:
            self.row_fills.fill_wait(rows)

    @_host_use
    def copy_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> bool:
        """Gather resident pinned rows directly into CUDA (or, on a CPU tier, CPU) outputs."""
        if source_ids.numel() == 0:
            return False
        slots = self.expert_to_slot[source_ids.long()]
        fallback_used = False
        for name in self.cached_names:
            source = self.tensors[name]
            output = outputs[name]
            if output.device.type == "cpu":
                torch.index_select(source, 0, slots.cpu(), out=output)
            elif source.is_contiguous() and output.is_contiguous():
                row_bytes = source.numel() * source.element_size() // self.capacity
                _gather_host_rows_kernel[
                    (source_ids.numel(), triton.cdiv(row_bytes, 1024))
                ](
                    source.view(torch.uint8),
                    slots,
                    output.view(torch.uint8),
                    row_bytes,
                    BLOCK=1024,
                )
            else:
                fallback_used = True
                slots_cpu = _copy_indices_to_cpu(slots, source_ids.numel())
                host_output = _pinned_staging_buffer(
                    ("pinned_host_cache_fallback", name),
                    source_ids.numel(),
                    max(source_ids.numel(), _NO_DEDUP_LIMIT),
                    tuple(source.shape[1:]),
                    source.dtype,
                )
                torch.index_select(source, 0, slots_cpu, out=host_output)
                output.copy_(host_output, non_blocking=True)
        return fallback_used

    @_host_use
    def gather_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> PinnedGatherResult:
        """Copy the rows of ``source_ids`` into the leading rows of ``outputs``, admitting misses.

        Rows are admitted and copied chunk by chunk. Each chunk is sized when
        it starts, from ``evictable_rows()``: residents that ``is_pinned``
        protects never make room, and an admitted row may itself become
        protected (an inclusive hierarchy pins rows the hot cache reserves), so
        the room shrinks during a call. A chunk holds at most that many
        distinct experts, all of them protected while it is admitted, and its
        rows are checked resident before its copy, so no copy reads slot -1.
        A call that fits in one chunk is the pre-chunking sequence ``lookup``,
        ``ensure_rows``, ``copy_rows``, plus that host-side check. With no
        evictable slot, a chunk with misses raises; an all-hit chunk is still
        copied. Each chunk's hit count (``.item()``) syncs the stream on the
        host, so the previous chunk's copy has run before its slots can be
        refilled.
        """
        if self.capacity == 0:
            raise ValueError("pinned host cache has no rows")
        hit_rows = 0
        miss_rows = 0
        populated_before = self.stats.populated_bytes
        fallback_used = False
        total = source_ids.numel()
        start = 0
        while start < total:
            evictable = self.evictable_rows()
            chunk_rows = total - start
            if (
                evictable >= 1
                and chunk_rows > evictable
                and torch.unique(source_ids[start:]).numel() > evictable
            ):
                chunk_rows = evictable
            chunk = source_ids[start : start + chunk_rows]
            _, hit_mask = self.lookup(chunk)
            chunk_hits = int(hit_mask.sum().item())
            hit_rows += chunk_hits
            miss_rows += chunk.numel() - chunk_hits
            chunk_ids = [int(value) for value in chunk.tolist()]
            if chunk_hits < chunk.numel():
                if evictable < 1:
                    raise RuntimeError("every pinned host slot holds a protected expert")
                self.ensure_rows(chunk[~hit_mask], protected=chunk_ids)
            lost = sorted({expert_id for expert_id in chunk_ids if expert_id not in self._lru})
            if lost:
                raise RuntimeError(
                    f"pinned host rows of experts {lost} were evicted before their copy"
                )
            chunk_outputs = {
                name: output[start : start + chunk_rows]
                for name, output in outputs.items()
            }
            self._await_fills(chunk_ids)
            fallback_used = self.copy_rows(chunk, chunk_outputs) or fallback_used
            start += chunk_rows
        return PinnedGatherResult(
            hit_rows,
            miss_rows,
            self.stats.populated_bytes - populated_before,
            fallback_used,
        )


def pinned_host_placement(budget_bytes: int) -> "Placement":
    """SGLANG_MOE_PINNED_HOST_NUMA_MB, checked against the budget and the nodes; () when unset."""
    from sglang.srt.layers.moe.host_numa import check_capacity, parse_placement

    placement = parse_placement(envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.get())
    if not placement:
        return ()
    placed = sum(nbytes for _, nbytes in placement)
    if placed != budget_bytes:
        raise ValueError(
            f"SGLANG_MOE_PINNED_HOST_NUMA_MB places {placed >> 20} MiB but "
            f"SGLANG_MOE_PINNED_HOST_MB is {budget_bytes >> 20} MiB; they must agree"
        )
    check_capacity(placement)
    return placement


def _placement_report(placement: "Placement", caches) -> dict | None:
    """The requested MiB per node and, per node, how many sampled tier pages it holds (-2: not yet resident)."""
    if not placement:
        return None
    from collections import Counter

    from sglang.srt.layers.moe.host_numa import page_nodes

    sampled = Counter()
    try:
        for cache in caches:
            for slab in cache.tensors.values():
                sampled.update(page_nodes(slab, samples=16))
    except OSError as error:  # a diagnostic; the tier is already bound and registered
        sampled = Counter({f"unavailable: {error.strerror}": 0})
    return {
        "mib": {str(node): nbytes >> 20 for node, nbytes in placement},
        "sampled_pages": {str(node): count for node, count in sorted(sampled.items())},
    }


class ExpertPinnedHostCacheManager:
    """Allocate a global complete-row budget across streamed expert layers."""

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, budget_bytes: int
    ) -> "ExpertPinnedHostCacheManager | None":
        budget_bytes = index(budget_bytes)
        if budget_bytes == 0:
            return None
        if budget_bytes < 0:
            raise ValueError("pinned host cache byte budget cannot be negative")
        streamers = {}
        for streamer in iter_expert_streamers(model):
            if streamer.host_bytes_per_expert == 0:
                continue
            layer_id = index(streamer.layer_id)
            if layer_id < 0 or layer_id in streamers:
                raise ValueError(
                    "pinned host cache requires unique nonnegative layer IDs"
                )
            streamers[layer_id] = streamer
        placement = pinned_host_placement(budget_bytes)
        if not streamers:
            return None
        capacities = {layer_id: 0 for layer_id in streamers}
        remaining = budget_bytes
        progress = True
        while progress:
            progress = False
            for layer_id in sorted(streamers):
                streamer = streamers[layer_id]
                if capacities[layer_id] >= streamer.num_experts:
                    continue
                if streamer.host_bytes_per_expert > remaining:
                    continue
                capacities[layer_id] += 1
                remaining -= streamer.host_bytes_per_expert
                progress = True
        if not any(capacities.values()):
            return None
        manager = cls()
        # The format supplies tier options such as an is_pinned filter; the dense
        # format supplies none, so NVFP4 tiers are built exactly as before.
        manager.caches = {
            layer_id: ExpertPinnedHostCache(
                streamers[layer_id],
                capacity,
                # Only when set: an unplaced tier is built with the same arguments as before.
                **({"placement": placement} if placement else {}),
                **pinned_tier_options_of(
                    streamers[layer_id].format, streamers[layer_id].layer
                ),
            )
            for layer_id, capacity in capacities.items()
            if capacity
        }
        manager.streamers = streamers
        manager.requested_bytes = budget_bytes
        logger.info(
            "Pinned host expert cache startup %s",
            json.dumps(
                {
                    "requested_bytes": budget_bytes,
                    "residency_bytes": manager.residency_bytes,
                    "rows": sum(cache.capacity for cache in manager.caches.values()),
                    "layers": len(manager.caches),
                    "numa": _placement_report(placement, manager.caches.values()),
                },
                sort_keys=True,
            ),
        )
        return manager

    @property
    def residency_bytes(self) -> int:
        return sum(cache.residency_bytes for cache in self.caches.values())

    def snapshot_stats(self) -> dict[str, dict[str, int]]:
        """Return cumulative row and byte counters grouped by layer."""
        return {
            str(layer_id): {
                "capacity_rows": cache.capacity,
                "residency_bytes": cache.residency_bytes,
                **asdict(cache.stats),
            }
            for layer_id, cache in self.caches.items()
        }


def expert_streaming_enabled() -> bool:
    return envs.SGLANG_MOE_EXPERT_STREAM.get()


@triton.jit
def _gather_host_rows_kernel(
    src_ptr,
    index_ptr,
    output_ptr,
    row_bytes,
    BLOCK: tl.constexpr,
):
    source_row = tl.load(index_ptr + tl.program_id(0)).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_bytes
    values = tl.load(
        src_ptr + source_row * row_bytes + offsets,
        mask=mask,
        other=0,
    )
    output_row = tl.program_id(0).to(tl.int64)
    tl.store(output_ptr + output_row * row_bytes + offsets, values, mask=mask)


def _tensor_data(value: torch.Tensor) -> torch.Tensor:
    return value.data if isinstance(value, torch.nn.Parameter) else value


@triton.jit
def _scatter_hot_rows_kernel(
    src_ptr,
    source_ids,
    destination_ids,
    output_ptr,
    row_bytes,
    BLOCK: tl.constexpr,
):
    source_row = tl.load(source_ids + tl.program_id(0)).to(tl.int64)
    destination_row = tl.load(destination_ids + tl.program_id(0)).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        src_ptr + source_row * row_bytes + offsets, mask=offsets < row_bytes, other=0
    )
    tl.store(
        output_ptr + destination_row * row_bytes + offsets,
        values,
        mask=offsets < row_bytes,
    )


def _cached_arange(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, str(device), dtype)
    value = _ARANGE_CACHE.get(key)
    if value is None:
        value = torch.arange(size, device=device, dtype=dtype)
        _ARANGE_CACHE[key] = value
    return value


def _staging_buffer(
    name: str,
    rows: int,
    max_rows: int,
    row_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the first ``rows`` rows of a reused buffer of at least ``max_rows``.

    Allocated zeroed: deduplicated eager gathers hand the fused-MoE kernel rows
    past the gathered ones, which no route selects and the CUTLASS kernel skips
    as token-less experts (flashinfer 0.6.18 ``cutlass_fused_moe_kernels.cuh:1452-1457``);
    zeros only keep it from reading uninitialised memory on first use.
    """
    key = (name, dtype, str(device), row_shape)
    buffer = _STAGING.get(key)
    if buffer is None or buffer.shape[0] < max_rows:
        buffer = torch.zeros(
            (max_rows,) + row_shape,
            dtype=dtype,
            device=device,
        )
        _STAGING[key] = buffer
        logger.info(
            "MoE expert staging buffer %s: shape=%s size=%.1f MiB",
            name,
            tuple(buffer.shape),
            buffer.numel() * buffer.element_size() / 1024**2,
        )
    return buffer[:rows]


def _pinned_staging_buffer(
    name: str,
    rows: int,
    capacity: int,
    row_shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (name, dtype, row_shape)
    buffer = _PINNED_STAGING.get(key)
    if buffer is None or buffer.shape[0] < capacity:
        buffer = torch.empty(
            (capacity,) + row_shape,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        _PINNED_STAGING[key] = buffer
        logger.info(
            "MoE expert pinned transfer buffer %s: shape=%s size=%.1f MiB",
            name,
            tuple(buffer.shape),
            buffer.numel() * buffer.element_size() / 1024**2,
        )
    return buffer[:rows]


def _copy_indices_to_cpu(source_ids: torch.Tensor, capacity: int) -> torch.Tensor:
    key = (str(source_ids.device),)
    buffer = _PINNED_INDEX.get(key)
    if buffer is None or buffer.numel() < capacity:
        buffer = torch.empty(capacity, dtype=torch.long, pin_memory=True)
        _PINNED_INDEX[key] = buffer
    indices = buffer[: source_ids.numel()]
    indices.copy_(source_ids, non_blocking=True)
    with (
        torch.cuda.nvtx.range("dsv41.expert_stream.copy_indices_stream_sync")
        if _SYNC_WAIT_NVTX
        else contextlib.nullcontext()
    ):
        torch.cuda.current_stream(source_ids.device).synchronize()
    return indices


def _read_rows(
    source: ExpertRowSource,
    rows_cpu: torch.Tensor,
    destinations: dict[str, torch.Tensor],
    destination_rows: torch.Tensor | None,
) -> RowReadStats:
    # Without destination rows, call read with two arguments, as the pageable
    # gather path always called the file reader.
    if destination_rows is None:
        return source.read(rows_cpu, destinations)
    return source.read(rows_cpu, destinations, destination_rows)


class ExpertStreamer:
    """Gather aligned expert rows into compact, reusable CUDA buffers.

    Residency-policy recording is skipped during CUDA stream capture in phase one.

    """

    def __init__(
        self,
        layer: torch.nn.Module,
        tensor_names: Iterable[str],
        *,
        layer_id: int | None = None,
        format: ExpertFormat | None = None,
        row_source: ExpertRowSource | None | object = _DEFAULT_ROW_SOURCE,
    ):
        self.layer = layer
        self.layer_id = (
            getattr(layer, "layer_id", None) if layer_id is None else layer_id
        )
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")
        # The format owns the row schema. Sources stay dynamic lookups because
        # the host arena rebinds layer tensors after this streamer exists.
        self.format = (
            DenseLayerFormat(self.tensor_names) if format is None else format
        )
        self.num_experts = self._validate_sources()
        if row_source is _DEFAULT_ROW_SOURCE:
            row_source = self.format.default_row_source(
                layer, self.specs, resolve_row_source_kind()
            )
        self._row_source: ExpertRowSource | None = None
        self.row_source = row_source  # validated against num_experts by the setter
        # Dense sources serve every name the row source does not cover.
        self._tensor_rows = TensorRowSource(
            self.source, self.tensor_names, self.num_experts
        )
        self.hot_cache = None
        self.expert_copy_backend = "gpu"
        self._dma_backend = ExpertDMABackend()
        self.pinned_host_cache = None
        self.residency_policy = None
        self.residency_update = None
        self.row_planner = None
        self.row_plan = None
        self.row_backend = None
        self._graph_pinned_tier = False
        self.row_tag = 0
        self.before_eager_gather = None
        # Set by ExpertPredictionRuntime when SGLANG_MOE_EXPERT_PREFETCH_PULL is on and this
        # layer is a scored prefetch target; see PrefetchPuller.join_target in serving/runtime.py.
        self.prefetch_puller = None
        self.graph_gather_rows = 0
        self.graph_counters: torch.Tensor | None = None
        self.last_gather_stats = ExpertGatherStats()
        # Row-source reads outside eager gathers (promotions, seeding, direct calls).
        self.background_read_stats = RowReadStats()
        self._gather_read_stats: RowReadStats | None = None
        self.bytes_per_expert = sum(spec.row_bytes for spec in self.specs)
        self.host_bytes_per_expert = sum(
            spec.row_bytes for spec in self.specs if spec.residence == "host"
        )
        signature = tuple(
            (
                spec.name,
                (self.num_experts,) + spec.row_shape,
                str(spec.dtype),
                spec.residence,
            )
            for spec in self.specs
        )
        if signature not in _LOGGED_SOURCE_SIGNATURES:
            _LOGGED_SOURCE_SIGNATURES.add(signature)
            logger.info(
                "MoE selected-expert streaming active: experts=%d tensors=%s",
                self.num_experts,
                ",".join(self.tensor_names),
            )

    @property
    def specs(self) -> tuple[ExpertTensorSpec, ...]:
        """The format's row specs, in ``tensor_names`` order."""
        return tuple(self._specs.values())

    def spec(self, name: str) -> ExpertTensorSpec:
        return self._specs[name]

    def source(self, name: str) -> torch.Tensor | None:
        """The dense ``[experts, ...]`` source of ``name`` now, or None when it has none."""
        return self.format.source(self.layer, name)

    @property
    def has_spec_only_tensors(self) -> bool:
        """Whether a streamed tensor has no dense source, so only the row source reads it."""
        return any(self.source(name) is None for name in self.tensor_names)

    @property
    def row_source(self) -> ExpertRowSource | None:
        return self._row_source

    @row_source.setter
    def row_source(self, value: ExpertRowSource | None) -> None:
        """Assign the row source; a non-None value must hold ``num_experts`` rows.

        None is always accepted without a check, so the host arena can drop
        the source (``file_row_reader = None``) at no extra cost.
        """
        if value is not None and value.num_experts != self.num_experts:
            raise ValueError(
                f"row source holds {value.num_experts} experts, but the layer "
                f"has {self.num_experts}"
            )
        self._row_source = value

    @property
    def file_row_reader(self) -> ExpertRowSource | None:
        """Alias of ``row_source``; the host arena drops it by assigning None."""
        return self.row_source

    @file_row_reader.setter
    def file_row_reader(self, value: ExpertRowSource | None) -> None:
        self.row_source = value

    @property
    def file_source_bytes_per_expert(self) -> int | None:
        """File bytes one expert row reads; None keeps eager gathers out of the pinned tier."""
        return self.format.file_source_bytes_per_expert(self.layer, self.row_source)

    def read_host_rows(
        self,
        rows_cpu: torch.Tensor,
        destinations: dict[str, torch.Tensor],
        destination_rows: torch.Tensor | None = None,
    ) -> RowReadStats:
        """Fill host ``destinations`` with expert ``rows_cpu``.

        Every name the row source covers is read in one call, so a source that
        reads a whole on-disk expert row per request pays for it once. The other
        names are read from their dense sources. Rows land in
        ``destination_rows`` of each destination, or in its leading rows.
        """
        row_source = self.row_source
        covered = {
            name: destination
            for name, destination in destinations.items()
            if row_source is not None and row_source.covers(name)
        }
        uncovered = {
            name: destination
            for name, destination in destinations.items()
            if name not in covered
        }
        missing = [name for name in uncovered if not self._tensor_rows.covers(name)]
        if missing:
            raise ValueError(
                f"no row source covers expert tensors {missing} and they have "
                "no dense source"
            )
        stats = RowReadStats()
        if uncovered:
            stats = stats + _read_rows(
                self._tensor_rows, rows_cpu, uncovered, destination_rows
            )
        if covered:
            stats = stats + _read_rows(row_source, rows_cpu, covered, destination_rows)
        if stats.rows:
            stats = replace(stats, rows=rows_cpu.numel())
        self._record_read(stats)
        return stats

    def _record_read(self, stats: RowReadStats) -> None:
        if self._gather_read_stats is not None:
            self._gather_read_stats = self._gather_read_stats + stats
        else:
            self.background_read_stats = self.background_read_stats + stats

    def serves_graph_gather(self, topk_output) -> bool:
        """Whether ``topk_output`` fits the sync-free gather enabled at startup."""
        topk_ids = getattr(topk_output, "topk_ids", None)
        return (
            self.graph_gather_rows > 0
            and isinstance(topk_ids, torch.Tensor)
            and 0 < topk_ids.numel() <= self.graph_gather_rows
        )

    def enable_graph_gather(self, max_rows: int, scratch_destinations: bool = True) -> None:
        """Serve gathers of at most ``max_rows`` routes with device-only operations.

        Misses are pulled from registered host rows into the hot cache's scratch
        rows by a kernel that reads its row count on the device, and routes are
        remapped with ``torch.where``. Nothing reads a CUDA value on the host, so
        a CUDA graph can capture the gather and replay it for any routes.
        All host-backed tensors of the layer are pulled in one kernel launch.
        Its plan holds the tensors it addresses. Capture and eager gathers
        raise if a layer tensor was rebound after this call; replays cannot
        check, and keep reading the tensors frozen here.
        """
        pinned_tier = graph_source_kind_of(self.format) == "pinned_tier"
        require_graph_gather_support((self,), pinned_tier_ok=True)
        from sglang.kernels.ops.moe.expert_cache_transfer import (
            copy_expert_row_segments_gpu,
            expert_row_segments,
        )

        max_rows = index(max_rows)
        cache = self.hot_cache
        if max_rows < 1:
            raise ValueError("graph gather needs at least one route row")
        if cache is None:
            raise ValueError("graph gather needs a hot cache")
        if scratch_destinations and cache.scratch_rows < max_rows:
            raise ValueError("graph gather needs a hot cache scratch row per route")
        if self.pinned_host_cache is not None and not pinned_tier:
            raise ValueError(
                "graph gather cannot admit rows through the pinned host cache"
            )
        if (
            getattr(self, "prefetch_coordinator", None) is not None
            or getattr(self, "next_layer_prefetch", None) is not None
        ):
            raise ValueError("graph gather cannot run with expert prefetch")
        if pinned_tier:
            # Missed rows are read from the pinned tier's registered slabs by pinned
            # slot (PinnedTierRowBackend), never from layer attributes.
            sources = dict(self.pinned_host_cache.tensors)
        else:
            sources = {
                name: _tensor_data(getattr(self.layer, name))
                for name in self.tensor_names
            }
        for name, source in sources.items():
            if source.device.type == "cpu" and not is_gpu_readable_host_tensor(source):
                raise ValueError(
                    f"graph gather needs registered host rows; {name!r} is pageable"
                )
        device = cache.device
        self._graph_sources = sources
        self._graph_pinned_tier = pinned_tier
        device_pairs = tuple(
            (source, cache.tensors[name])
            for name, source in self._graph_sources.items()
            if source.device.type == "cuda"
        )
        host_pairs = [
            (source, cache.tensors[name])
            for name, source in self._graph_sources.items()
            if source.device.type == "cpu"
        ]
        self._copy_row_segments_gpu = copy_expert_row_segments_gpu
        if scratch_destinations:
            # Device-source tensors take a separate fixed-shape index copy of every route row.
            # Its padding lanes past the miss count land in spare scratch rows, harmlessly.
            self._graph_device_pairs = device_pairs
            segment_pairs = host_pairs
        else:
            # Without scratch there is no harmless landing place for a padding lane, so every
            # pair goes through the segment kernel, which copies exactly ``count`` rows.
            # ``_validate_row_pair`` accepts CUDA sources, and one launch replaces two.
            self._graph_device_pairs = ()
            segment_pairs = host_pairs + list(device_pairs)
        self._graph_row_segments = (
            expert_row_segments(segment_pairs) if segment_pairs else None
        )
        # Stage 2 merges the device pairs into the segment table, so the table's existence no
        # longer says whether anything is host-backed. Record that separately.
        self._graph_host_pair_count = len(host_pairs)
        self._graph_scratch_slots = (
            torch.arange(
                cache.capacity, cache.capacity + max_rows, dtype=torch.long, device=device
            )
            if scratch_destinations
            # Overwritten with victim slots by every gather; zero is a valid row until then.
            else torch.zeros(max_rows, dtype=torch.long, device=device)
        )
        self._graph_source_rows = torch.zeros(
            max_rows, dtype=torch.int64, device=device
        )
        self._graph_destination_slots = self._graph_scratch_slots.to(torch.int32)
        self._graph_miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        # Read once here rather than in `_gather_graph`: `EnvBool.get()` is an
        # uncached `os.getenv` read, and reading it again per call could
        # disagree with this setup-time read and index a buffer that was
        # never allocated.
        self._fused_plan_enabled = envs.SGLANG_MOE_EXPERT_FUSED_PLAN.get()
        self._graph_fused_slots_scratch = (
            torch.empty(max_rows, dtype=torch.int32, device=device)
            if self._fused_plan_enabled
            else None
        )
        # The fused planner preserves a router's native int32 ids (or int64
        # where a model uses them) and writes directly into a stable remap.
        # Separate buffers avoid a dtype conversion/allocation during replay.
        self._graph_fused_remaps = (
            {
                dtype: torch.empty(max_rows, dtype=dtype, device=device)
                for dtype in (torch.int32, torch.int64)
            }
            if self._fused_plan_enabled
            else None
        )
        self.row_planner = ExpertRowPlanner(cache, cache.capacity, max_rows)
        self.row_plan = ExpertRowPlan(
            expert_ids=self._graph_source_rows,
            slots=self._graph_destination_slots,
            count=self._graph_miss_count,
        )
        if pinned_tier:
            self.row_backend = PinnedTierRowBackend(
                {self.row_tag: self._graph_row_segments},
                self.pinned_host_cache.expert_to_slot,
                max_rows,
            )
        else:
            self.row_backend = (
                InGraphRowBackend({self.row_tag: self._graph_row_segments})
                if self._graph_row_segments is not None
                else None
            )
        self._graph_ones = torch.ones(max_rows, dtype=torch.float32, device=device)
        self.graph_counters = torch.zeros(2, dtype=torch.int64, device=device)
        self.graph_unique_counters = torch.zeros(2, dtype=torch.int64, device=device)
        self.graph_gather_rows = max_rows

    def _gather_graph(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._check_graph_sources()
        if self.residency_update is not None:
            self.residency_update.on_graph_forward(topk_ids.shape[0])
        cache = self.hot_cache
        expert_to_slot = self.row_planner.expert_to_slot
        fused = self._fused_plan_enabled and supports_fused_graph_routes(
            topk_ids, expert_to_slot, self.graph_gather_rows
        )
        # The JIT planner supports the router's native IDs.  Its generic
        # counterpart relies on index_select, which requires int64.
        flat = topk_ids.reshape(-1) if fused else topk_ids.reshape(-1).long()
        count = flat.numel()
        prefetch_puller = getattr(self, "prefetch_puller", None)
        # Read before planning, not after: the planner needs the posted prediction to
        # exclude its covered row from the demand-scratch plan (the row-skip this
        # dispatch exists for). Safe without the pull's stream join -- see
        # `PrefetchPuller.predicted_expert_for`.
        prefetch_expert = (
            prefetch_puller.predicted_expert_for(self.layer_id)
            if prefetch_puller is not None
            else None
        )
        prefetch_slot = (
            prefetch_puller.slot_for(self.layer_id) if prefetch_expert is not None else -1
        )
        prefetch_count = (
            prefetch_puller.posted_count_for(self.layer_id)
            if prefetch_expert is not None
            else None
        )
        prefetch_outcomes = (
            prefetch_puller.outcome_counters_for(self.layer_id)
            if prefetch_expert is not None
            else None
        )
        if fused:
            route_counts = (
                self.residency_policy.pending_counts
                if self.residency_policy is not None
                else None
            )
            remap = plan_graph_routes_fused(
                flat,
                expert_to_slot,
                self.row_planner.scratch_base,
                topk_ids.dtype,
                self._graph_source_rows[:count],
                self._graph_fused_slots_scratch[:count],
                self._graph_miss_count,
                self.graph_counters,
                self.graph_unique_counters,
                route_counts,
                prefetch_expert=prefetch_expert,
                prefetch_slot=prefetch_slot,
                prefetch_count=prefetch_count,
                outcome_counters=prefetch_outcomes,
                remap_out=self._graph_fused_remaps[flat.dtype][:count],
            )
            source_rows = self._graph_source_rows[:count]
            scratch = self._graph_scratch_slots[: source_rows.numel()]
        else:
            plan = self.row_planner.route_plan(
                flat,
                prefetch_expert=prefetch_expert,
                prefetch_slot=prefetch_slot,
                prefetch_count=prefetch_count,
            )
            scratch = self._graph_scratch_slots[: plan.source_rows.numel()]
            self.row_planner.fill_routes(plan, self.row_plan)
            remap = plan.remap
            source_rows = plan.source_rows
        direct = getattr(self, "residency_direct", None)
        if direct is not None and direct.layer_fusion:
            # One kernel for the branch below. Without a prefetch join it writes the router's own
            # dtype, so the cast on return is a no-op; the join keeps the int64 the branch returns.
            remap = direct.fused_gather_destinations(
                self.residency_row,
                remap,
                flat,
                expert_to_slot,
                self.row_planner.scratch_base,
                topk_ids.dtype if prefetch_puller is None else torch.int64,
            )
        elif direct is not None:
            # Stage DIRECT: send the miss lanes into victim slots instead of scratch rows.
            # `expert_to_slot` is still this forward's pre-gather mapping here, so its slots
            # are exactly the rows the gather is about to read.
            remap = direct.gather_destinations(
                self.residency_row,
                remap,
                expert_to_slot.index_select(0, flat.long()),
                self.row_planner.scratch_base,
            )
        if prefetch_puller is not None:
            # Join after actual routing: `remap`/`expert_to_slot` above are this
            # forward's real routing decision, not the prediction that posted the
            # pull. The dedicated slot the pull wrote is never in `scratch`
            # (plan section 7.1), so the ordinary demand copy below cannot race it.
            remap = prefetch_puller.join_target(
                self.layer_id,
                flat_ids=flat,
                missed_mask=expert_to_slot[flat] < 0,
                demand_remap=remap,
                planner_owned=fused,
                clear_after_join=True,
            )
        calibration = getattr(self, "prefetch_calibration", None)
        if calibration is not None:
            calibration.record_target(
                self.layer_id,
                flat,
                expert_to_slot[flat] < 0,
                self.row_plan.count,
                expert_to_slot,
            )
        for source, destination in self._graph_device_pairs:
            destination.view(torch.uint8).reshape(destination.shape[0], -1).index_copy_(
                0,
                scratch,
                source.view(torch.uint8)
                .reshape(source.shape[0], -1)
                .index_select(0, source_rows),
            )
        if self.row_backend is not None:
            self.row_backend.post(self.row_tag, self.row_plan)
            delivery = self.row_backend.resolve(self.row_tag, self.row_plan)
            self.row_backend.copy_residual(self.row_tag, delivery)
        if direct is not None:
            # Strictly after the copies, on this same stream: residency only claims a row
            # once the copy that fills it has been issued ahead of it.
            if moe_side_stream.active():
                # The side stream forks after the copies; the MoE layer joins it before the next layer's gather.
                moe_side_stream.fork(direct.commit_gather, inputs=direct.pending_commit_tensors())
            else:
                direct.commit_gather()
        if not fused:
            self.graph_counters[0].add_(count)
            self.graph_counters[1].add_(plan.routed_miss_rows)
            self.graph_unique_counters[0].add_(plan.unique_hit_rows)
            self.graph_unique_counters[1].add_(plan.unique_miss_rows)
            if self.residency_policy is not None:
                self.residency_policy.pending_counts.index_add_(
                    0, flat, self._graph_ones[:count]
                )
        return remap.reshape(topk_ids.shape).to(topk_ids.dtype), cache.tensors

    def _check_graph_sources(self) -> None:
        """Raise if a layer tensor was rebound after its graph gather plan froze it."""
        for name, source in self._graph_sources.items():
            current = (
                self.pinned_host_cache.tensors[name]
                if self._graph_pinned_tier
                else _tensor_data(getattr(self.layer, name))
            )
            if current.data_ptr() != source.data_ptr() or current.device != source.device:
                raise RuntimeError(
                    f"expert tensor {name!r} moved after graph gather was enabled"
                )
        if self._graph_pinned_tier and (
            self.pinned_host_cache.expert_to_slot.data_ptr()
            != self.row_backend.host_row_map.data_ptr()
        ):
            raise RuntimeError(
                "the pinned host tier's slot map moved after graph gather was enabled"
            )

    def _read_pageable_rows(
        self,
        cpu_ids: torch.Tensor | None,
        outputs: dict[str, torch.Tensor],
        row_count: int,
        capacity: int,
    ) -> None:
        """Read ``outputs``' rows on the host in one batch into pinned staging, then copy them up."""
        assert cpu_ids is not None
        host_outputs = {
            name: _pinned_staging_buffer(
                name,
                row_count,
                capacity,
                self.spec(name).row_shape,
                self.spec(name).dtype,
            )
            for name in outputs
        }
        self.read_host_rows(cpu_ids, host_outputs)
        for name, output in outputs.items():
            output.copy_(host_outputs[name], non_blocking=True)

    def _validate_sources(self) -> int:
        """Check the format's specs against the streamer names and any dense sources."""
        specs = tuple(self.format.tensor_specs(self.layer))
        names = tuple(spec.name for spec in specs)
        if names != self.tensor_names:
            raise ValueError(
                f"expert format specs {names} do not match tensor names {self.tensor_names}"
            )
        expert_count = index(self.format.num_experts(self.layer))
        if expert_count < 1:
            raise ValueError("expert format has no expert rows")
        for spec in specs:
            source = self.format.source(self.layer, spec.name)
            if source is None:
                if spec.residence != "host":
                    raise ValueError(
                        f"expert tensor {spec.name!r} has no dense source, so its "
                        "rows must be host-resident"
                    )
                continue
            if (
                tuple(source.shape) != (expert_count,) + spec.row_shape
                or source.dtype != spec.dtype
            ):
                raise ValueError(
                    f"expert source tensor {spec.name!r} does not match its spec"
                )
            if (source.device.type == "cpu") != (spec.residence == "host"):
                raise ValueError(
                    f"expert source tensor {spec.name!r} on {source.device} does not "
                    f"match its spec's {spec.residence!r} residence"
                )
        self._specs = {spec.name: spec for spec in specs}
        return expert_count

    def _copy_source_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> int:
        """Fill supplied CUDA rows through the existing bounded host buffers.

        With the ``dma`` copy backend, rows of registered or pinned host tensors
        go through the CUDA copy engine, merged into runs of consecutive rows.
        Graph capture keeps the pull kernel. Rows of pageable host tensors, and
        of tensors without a dense source, are read by the row sources in one
        batch into pinned staging. Returns the bytes the copy engine moved, so
        a build without it shows up as zero.
        """
        row_count = source_ids.numel()
        use_dma = (
            self.expert_copy_backend == "dma"
            and _aot_transfer_available()
            and not torch.cuda.is_current_stream_capturing()
        )
        dma_rows = None
        copy_engine_bytes = 0
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        sources = {name: self.source(name) for name in self.tensor_names}
        host_sources = {
            name: source
            for name, source in sources.items()
            if source is None or source.device.type == "cpu"
        }
        readable = {
            name: source is not None and is_gpu_readable_host_tensor(source)
            for name, source in host_sources.items()
        }
        pageable_source = not all(readable.values())
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        pageable_outputs: dict[str, torch.Tensor] = {}
        for name, output in outputs.items():
            source = sources[name]
            if source is not None and source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif readable[name] and use_dma and source.ndim >= 2:
                if dma_rows is None:
                    dma_rows = source_ids.tolist()
                self._dma_backend.copy_rows(source, output, dma_rows, range(row_count))
                copy_engine_bytes += (
                    row_count * source.numel() * source.element_size() // self.num_experts
                )
            elif readable[name]:
                row_bytes = source.numel() * source.element_size() // self.num_experts
                _gather_host_rows_kernel[(row_count, triton.cdiv(row_bytes, 1024))](
                    source.view(torch.uint8),
                    source_ids,
                    output.view(torch.uint8),
                    row_bytes,
                    BLOCK=1024,
                )
            else:
                pageable_outputs[name] = output
        if pageable_outputs:
            self._read_pageable_rows(cpu_ids, pageable_outputs, row_count, capacity)
        return copy_engine_bytes

    def _staging_floor_rows(self) -> int:
        """Rows every eager staging buffer is allocated with from the first gather."""
        cap = self.format.max_gather_rows
        return self.num_experts if cap is None else min(self.num_experts, cap)

    def _gather_cached(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather routed rows through the hot cache for the fused-MoE kernel.

        When ``should_dedup`` applies, returned tensors keep a leading dimension
        of ``max(row_count, NO_DEDUP_LIMIT)``: rows are assembled into the first
        ``row_count`` rows of each staging buffer and the padded view is
        returned. Deduplicated gathers give a different expert count on almost
        every call, which made the fused-MoE kernel 4x slower (experiment E10).
        A forward without dedup already has a constant count and returns
        ``row_count`` rows.
        """
        cache = self.hot_cache
        slots, hit_mask = cache.lookup(source_ids)
        row_count = source_ids.numel()
        routed_rows = compact_ids.numel()
        if routed_rows == row_count:
            hit_rows = routed_hit_rows = int(hit_mask.sum().item())
        else:
            hit_rows, routed_hit_rows = torch.stack(
                (hit_mask.sum(), hit_mask[compact_ids.long()].sum())
            ).tolist()
        miss_rows = row_count - hit_rows
        if miss_rows == 0:
            self.last_gather_stats = ExpertGatherStats(
                row_count, hit_rows, routed_rows=routed_rows
            )
            return slots[compact_ids.long()].reshape(topk_ids.shape).to(
                topk_ids.dtype
            ), cache.tensors
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        # Allocated at a whole layer's experts from the first gather: prefill chunks climb to
        # nearly all of them, and growing one step at a time left every outgrown buffer in
        # the allocator's cache at the prefill peak. A format with large rows caps that
        # floor at its max_gather_rows and gathers in chunks (iter_gather_experts).
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self._staging_floor_rows()),
                self.spec(name).row_shape,
                self.spec(name).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
        gathered = {name: buffer[:row_count] for name, buffer in padded.items()}
        miss_source_ids = source_ids[~hit_mask]
        # Misses take the first rows so their copies land in place, with no second
        # staging buffer as large as a layer's experts; hits fill the rows after them.
        misses = {name: output[:miss_rows] for name, output in gathered.items()}
        assembly_bytes = hit_rows * self.bytes_per_expert
        if hit_rows:
            hot_slots = slots[hit_mask]
            order = torch.cat(((~hit_mask).nonzero().flatten(), hit_mask.nonzero().flatten()))
            rows = _cached_arange(row_count, order.device, order.dtype)
            row_of_source = torch.empty_like(order)
            row_of_source[order] = rows
            compact_ids = row_of_source[compact_ids.long()]
            hit_positions = rows[miss_rows:]
        pinned_hit_rows = 0
        pinned_miss_rows = 0
        pinned_populated_bytes = 0
        gather_fallback_used = False
        copy_engine_bytes = 0
        pinned_cache = self.pinned_host_cache
        source_bytes = miss_rows * self.bytes_per_expert
        if (
            pinned_cache is not None
            and pinned_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            pinned_outputs = {
                name: output
                for name, output in misses.items()
                if name in pinned_cache.cached_names
            }
            # Chunked so misses beyond the pinned capacity are never copied from slot -1.
            pinned = pinned_cache.gather_rows(miss_source_ids, pinned_outputs)
            pinned_hit_rows = pinned.hit_rows
            pinned_miss_rows = pinned.miss_rows
            pinned_populated_bytes = pinned.populated_bytes
            gather_fallback_used = pinned.fallback_used
            source_bytes = pinned_miss_rows * self.host_bytes_per_expert + miss_rows * (
                self.bytes_per_expert - self.host_bytes_per_expert
            )
            uncached_outputs = {
                name: output
                for name, output in misses.items()
                if name not in pinned_cache.cached_names
            }
            if uncached_outputs:
                copy_engine_bytes = self._copy_source_rows(
                    miss_source_ids, uncached_outputs
                )
        else:
            copy_engine_bytes = self._copy_source_rows(miss_source_ids, misses)
        if hit_rows:
            for name, output in gathered.items():
                row_bytes = output.numel() * output.element_size() // row_count
                _scatter_hot_rows_kernel[(hit_rows, triton.cdiv(row_bytes, 1024))](
                    cache.tensors[name].view(torch.uint8),
                    hot_slots,
                    hit_positions,
                    output.view(torch.uint8),
                    row_bytes,
                    BLOCK=1024,
                )
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            hit_rows,
            miss_rows,
            assembly_bytes
            + miss_rows * (self.bytes_per_expert - self.host_bytes_per_expert),
            miss_rows * self.host_bytes_per_expert,
            source_bytes,
            pinned_hit_rows,
            pinned_miss_rows,
            pinned_populated_bytes,
            0,
            gather_fallback_used,
            copy_engine_bytes=copy_engine_bytes,
            routed_rows=routed_rows,
            routed_miss_rows=routed_rows - routed_hit_rows,
            unique_miss_rows=miss_rows,
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded

    def _gather_pinned_host(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather routed rows through the pinned host cache.

        Returned tensors follow the leading-dimension rule of ``_gather_cached``.
        Pinned rows are copied straight into the one staging set of
        ``max(rows, NO_DEDUP_LIMIT)`` rows by ``gather_rows``, which admits
        misses in chunks the tier can hold, so a format's ``max_gather_rows``
        bounds this path's VRAM as it bounds ``_gather_cached``'s.
        """
        cache = self.pinned_host_cache
        assert cache is not None
        row_count = source_ids.numel()
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                capacity,
                self.spec(name).row_shape,
                self.spec(name).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
        gathered = {name: buffer[:row_count] for name, buffer in padded.items()}
        pinned = cache.gather_rows(
            source_ids,
            {
                name: output
                for name, output in gathered.items()
                if name in cache.cached_names
            },
        )
        copy_engine_bytes = 0
        uncached = {
            name: output
            for name, output in gathered.items()
            if name not in cache.cached_names
        }
        if uncached:
            copy_engine_bytes = self._copy_source_rows(source_ids, uncached)
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            0,
            row_count * self.host_bytes_per_expert,
            pinned.miss_rows * self.bytes_per_expert,
            pinned.hit_rows,
            pinned.miss_rows,
            pinned.populated_bytes,
            copy_engine_bytes=copy_engine_bytes,
            routed_rows=compact_ids.numel(),
            routed_miss_rows=compact_ids.numel(),
            unique_miss_rows=row_count,
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded

    @contextlib.contextmanager
    def prefill_fills(self, source_ids: torch.Tensor) -> Iterator[None]:
        """SGLANG_DSV41_ENABLE_PREFILL_FILLS: read a layer's pinned-tier misses while its chunks gather.

        ``source_ids`` are the layer's distinct routed experts, gathered inside this context. The pinned tier is
        held in one host use for the whole layer (one stream sync, one pause of its owner), the experts that miss
        VRAM and RAM are claimed in ascending order (the chunks' order) and read on the tier's fill thread, and each
        chunk's ``gather_rows`` waits only for its own rows. Nothing happens without the tier's ``row_fills``.
        """
        cache = self.pinned_host_cache
        if cache is None or cache.row_fills is None or source_ids.numel() == 0:
            yield
            return
        if self.before_eager_gather is not None:
            # The first gather would apply a pending residency boundary; apply it before reading the hot set.
            self.before_eager_gather()
        ids = source_ids.long()
        with cache.host_use():
            hot = self.hot_cache
            if hot is not None and hot.capacity:
                experts, hot_slots = torch.stack((ids, hot.expert_to_slot[ids].long())).tolist()
            else:
                experts = ids.tolist()
                hot_slots = [-1] * len(experts)
            cache.prefetch_rows(
                sorted(expert for expert, slot in zip(experts, hot_slots) if slot < 0), protected=experts
            )
            try:
                yield
            finally:
                cache.finish_fills()

    def _plan_eager_routes(
        self, topk_ids: torch.Tensor, record: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the source IDs to gather and each route's index into them.

        Multi-token forwards gather one row per distinct expert; the residency
        policy still records every route, keeping routed multiplicity.
        """
        flat_ids = topk_ids.reshape(-1)
        if should_dedup(topk_ids):
            source_ids, compact_ids = torch.unique(
                flat_ids, sorted=True, return_inverse=True
            )
        else:
            source_ids = flat_ids
            compact_ids = _cached_arange(
                flat_ids.numel(), flat_ids.device, topk_ids.dtype
            )
        if record:
            self.record_routes(topk_ids)
        return source_ids, compact_ids

    def record_routes(self, topk_ids: torch.Tensor) -> None:
        """Count every route of a forward in the residency policy; call once per forward.

        Contract: every id in ``topk_ids`` must be in ``[0, num_experts)``,
        on the residency policy's device, with the forward's full
        multiplicity. A caller whose ``topk_ids`` can carry negative or
        sentinel ids (padding, masked routes) must filter them out itself
        before calling this; ``record_routes`` does not filter, so that
        NVFP4 prefill, which never has such ids, pays no extra masked-select.
        ``ExpertResidencyPolicy.record_routes`` runs ``torch.bincount`` on
        the ids and raises on a negative id or a device mismatch.

        Skipped during CUDA stream capture. ``gather`` calls it itself with
        ids it has already range-checked; a consumer of ``gather_experts``
        calls it separately with the forward's full ``topk_ids``, filtered
        to satisfy this contract.
        """
        residency_policy = self.residency_policy
        flat_ids = topk_ids.reshape(-1)
        if residency_policy is not None and not (
            flat_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            residency_policy.record_routes(flat_ids)

    def gather(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return compact route IDs and the expert rows they index.

        Eager results follow the leading-dimension rule of ``_gather_cached``;
        graph gathers return the fixed hot-cache tensors.
        """
        if topk_ids.device.type != "cuda":
            raise ValueError("selected expert IDs must be on CUDA")
        if 0 < topk_ids.numel() <= self.graph_gather_rows:
            return self._gather_graph(topk_ids)
        # A source score can still have forked this target's side pull when an
        # eager/tail shape falls outside graph-gather support. It is not usable
        # there, but must join before this state can be reused.
        prefetch_puller = getattr(self, "prefetch_puller", None)
        if prefetch_puller is not None:
            prefetch_puller.join_unsupported_target(self.layer_id)
        if self.before_eager_gather is not None:
            self.before_eager_gather()
        flat_ids = topk_ids.reshape(-1)
        prefetch_coordinator = getattr(self, "prefetch_coordinator", None)
        if flat_ids.numel() == 0:
            raise ValueError("selected expert IDs cannot be empty")
        if bool(((flat_ids < 0) | (flat_ids >= self.num_experts)).any().item()):
            raise ValueError(
                f"selected expert ID is outside [0, {self.num_experts - 1}]"
            )

        source_ids, compact_ids = self._plan_eager_routes(topk_ids, record=False)
        cap = self.format.max_gather_rows
        if cap is not None and source_ids.numel() > cap:
            raise ValueError(
                f"eager gather of {source_ids.numel()} experts exceeds the format's "
                f"max_gather_rows={cap}; gather in chunks with iter_gather_experts"
            )
        self.record_routes(topk_ids)
        if prefetch_coordinator is not None:
            prefetch_coordinator.synchronous_correction(
                source_ids.tolist(), lambda _: None
            )
        next_layer_prefetch = getattr(self, "next_layer_prefetch", None)
        if next_layer_prefetch is not None:
            next_layer_prefetch(source_ids)
        return self._gather_eager_rows(source_ids, compact_ids, topk_ids)

    def gather_experts(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather the rows of the distinct experts ``source_ids`` for an eager consumer.

        Returns ``(row_of_source, rows)``, where ``rows[name][row_of_source[i]]``
        holds expert ``source_ids[i]``. ``rows`` are the hot cache's slot
        tensors when every expert is resident, else staging buffers that the
        next eager gather of any layer reuses: enqueue any consuming work on
        the current stream before calling into this streamer again (``next()``
        on an ``iter_gather_experts`` iterator included), and clone any row
        you need to keep past that point. An all-hit chunk returns the hot
        cache's own slot tensors, not a staging copy. Routes are not
        recorded: call ``record_routes`` once with the forward's full
        ``topk_ids``. At most the format's ``max_gather_rows`` experts per
        call; see ``iter_gather_experts``.

        Validates ``source_ids`` itself (1-D, in range, distinct). A caller
        that already validated a larger id set once, such as
        ``iter_gather_experts``, should not call this method per chunk;
        use the unvalidated ``_gather_experts`` instead.
        """
        if source_ids.ndim != 1:
            raise ValueError("gather_experts needs a 1-D tensor of expert IDs")
        count = source_ids.numel()
        if count == 0:
            raise ValueError("gather_experts needs a nonempty tensor of expert IDs")
        if bool(((source_ids < 0) | (source_ids >= self.num_experts)).any().item()):
            raise ValueError(
                f"selected expert ID is outside [0, {self.num_experts - 1}]"
            )
        if torch.unique(source_ids).numel() != count:
            raise ValueError("gather_experts needs distinct expert IDs")
        return self._gather_experts(source_ids)

    def _gather_experts(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The unvalidated core of ``gather_experts``.

        Callers must have already checked that ``source_ids`` is a 1-D,
        in-range, distinct tensor of expert IDs; ``gather_experts`` and
        ``iter_gather_experts`` are the only callers, and each validates
        once before any chunk reaches here. This still enforces the
        format's ``max_gather_rows`` cap and the prefetch refusal per call,
        since both can vary by chunk.
        """
        count = source_ids.numel()
        cap = self.format.max_gather_rows
        if cap is not None and count > cap:
            raise ValueError(
                f"gather of {count} experts exceeds the format's max_gather_rows={cap}"
            )
        if (
            getattr(self, "prefetch_coordinator", None) is not None
            or getattr(self, "next_layer_prefetch", None) is not None
        ):
            raise ValueError("gather_experts does not drive expert prefetch")
        if self.before_eager_gather is not None:
            self.before_eager_gather()
        compact_ids = _cached_arange(count, source_ids.device, source_ids.dtype)
        row_of_source, rows = self._gather_eager_rows(
            source_ids, compact_ids, source_ids.reshape(1, -1)
        )
        return row_of_source.reshape(-1), rows

    def iter_gather_experts(
        self, source_ids: torch.Tensor, chunk_rows: int | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]:
        """Yield ``(chunk_ids, row_of_source, rows)`` over chunks of distinct experts.

        Each chunk is one gather of at most ``chunk_rows`` experts (default:
        the format's ``max_gather_rows``, else all at once). A chunk's rows
        share staging with the next chunk, so consume them (see
        ``gather_experts``'s docstring for the staging-reuse contract) before
        advancing to the next chunk. When the iteration ends,
        ``last_gather_stats`` holds the sum over its chunks, so observers
        count the forward once.

        Validates ``source_ids`` once, up front (1-D, in range, distinct),
        then dispatches every chunk through the unvalidated
        ``_gather_experts`` so a multi-chunk gather does not repeat the
        range and distinctness syncs per chunk.
        """
        if source_ids.ndim != 1:
            raise ValueError("iter_gather_experts needs a 1-D tensor of expert IDs")
        count = source_ids.numel()
        if count == 0:
            return
        if bool(((source_ids < 0) | (source_ids >= self.num_experts)).any().item()):
            raise ValueError(
                f"selected expert ID is outside [0, {self.num_experts - 1}]"
            )
        if torch.unique(source_ids).numel() != count:
            raise ValueError("iter_gather_experts needs distinct expert IDs")
        cap = self.format.max_gather_rows
        if chunk_rows is None:
            chunk_rows = cap if cap is not None else count
        chunk_rows = index(chunk_rows)
        if chunk_rows < 1:
            raise ValueError("gather chunks need at least one row")
        if cap is not None and chunk_rows > cap:
            raise ValueError(
                f"gather chunks of {chunk_rows} rows exceed the format's "
                f"max_gather_rows={cap}"
            )
        chunk_stats = []
        try:
            for start in range(0, count, chunk_rows):
                chunk = source_ids[start : start + chunk_rows]
                row_of_source, rows = self._gather_experts(chunk)
                chunk_stats.append(self.last_gather_stats)
                yield chunk, row_of_source, rows
        finally:
            if chunk_stats:
                self.last_gather_stats = _sum_gather_stats(chunk_stats)

    def _gather_eager_rows(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one eager gather and add the host reads it caused to its stats."""
        self._gather_read_stats = RowReadStats()
        try:
            result = self._dispatch_eager_rows(source_ids, compact_ids, topk_ids)
            read = self._gather_read_stats
        finally:
            self._gather_read_stats = None
        if read.rows:
            self.last_gather_stats = replace(
                self.last_gather_stats,
                host_read_rows=read.rows,
                host_read_file_bytes=read.file_bytes,
                host_read_split_bytes=read.split_bytes,
                host_read_ns=read.read_ns,
                host_split_ns=read.split_ns,
            )
        return result

    def _dispatch_eager_rows(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
        if (
            self.pinned_host_cache is not None
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
        return self._gather_uncached(source_ids, compact_ids, topk_ids)

    def _gather_uncached(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather rows straight from their sources, with no hot or pinned cache.

        Returned tensors follow the leading-dimension rule of ``_gather_cached``.
        """
        row_count = source_ids.numel()
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            row_count * (self.bytes_per_expert - self.host_bytes_per_expert),
            row_count * self.host_bytes_per_expert,
            row_count * self.bytes_per_expert,
            routed_rows=compact_ids.numel(),
            routed_miss_rows=compact_ids.numel(),
            unique_miss_rows=row_count,
        )
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        sources = {name: self.source(name) for name in self.tensor_names}
        pageable_source = any(
            source is None
            or (source.device.type == "cpu" and not is_gpu_readable_host_tensor(source))
            for source in sources.values()
        )
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        padded: dict[str, torch.Tensor] = {}
        pageable_outputs: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            source = sources[name]
            spec = self.spec(name)
            padded[name] = _staging_buffer(
                name,
                kernel_rows,
                capacity,
                spec.row_shape,
                spec.dtype,
                topk_ids.device,
            )
            output = padded[name][:row_count]
            if source is not None and source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif source is not None and is_gpu_readable_host_tensor(source):
                source_bytes = source.view(torch.uint8)
                output_bytes = output.view(torch.uint8)
                row_bytes = source_bytes.numel() // self.num_experts
                block = 1024
                _gather_host_rows_kernel[(row_count, triton.cdiv(row_bytes, block))](
                    source_bytes,
                    source_ids,
                    output_bytes,
                    row_bytes,
                    BLOCK=block,
                )
            else:
                pageable_outputs[name] = output
        if pageable_outputs:
            self._read_pageable_rows(cpu_ids, pageable_outputs, row_count, capacity)
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded
