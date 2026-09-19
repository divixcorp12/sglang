"""Selected-expert staging for host-resident MoE weight rows."""

from __future__ import annotations

import logging
import json
import os
from dataclasses import asdict, dataclass
from operator import index
from typing import Dict, Iterable, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend, _aot_transfer_available
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
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
)
from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor

logger = logging.getLogger(__name__)

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


@dataclass
class PinnedHostCacheStats:
    """Cumulative counters for the bounded pinned-host expert cache."""

    lookup_hits: int = 0
    lookup_misses: int = 0
    populated_rows: int = 0
    populated_bytes: int = 0
    evictions: int = 0


class ExpertPinnedHostCache:
    """Bounded, on-demand pinned host rows shared by one expert layer."""

    def __init__(self, streamer: "ExpertStreamer", capacity: int):
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
        devices = {
            streamer.source(spec.name).device
            for spec in streamer.specs
            if spec.residence == "device"
        }
        if len(devices) > 1:
            raise ValueError("pinned host cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity,) + streamer.spec(name).row_shape,
                dtype=streamer.spec(name).dtype,
                device="cpu",
                pin_memory=True,
            )
            for name in self.cached_names
        }
        file_row_reader = getattr(streamer, "file_row_reader", None)
        if file_row_reader is not None:
            file_row_reader.register_destinations(self.tensors.values())
        self.expert_to_slot = torch.full(
            (streamer.num_experts,), -1, dtype=torch.long, device=self.device
        )
        self.slot_to_expert = [-1] * capacity
        self._expert_to_slot: dict[int, int] = {}
        self._last_used = [0] * capacity
        self._clock = 0
        self.stats = PinnedHostCacheStats()
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

    def _refresh_mapping(self) -> None:
        mapping = [-1] * self.streamer.num_experts
        for slot, expert_id in enumerate(self.slot_to_expert):
            if expert_id >= 0:
                mapping[expert_id] = slot
        self.expert_to_slot.copy_(
            torch.tensor(mapping, dtype=torch.long, device=self.device)
        )

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
            slot = self._expert_to_slot[int(expert_id)]
            self._clock += 1
            self._last_used[slot] = self._clock
        self.stats.lookup_hits += len(hit_ids)
        self.stats.lookup_misses += source_ids.numel() - len(hit_ids)
        return slots, hit_mask

    def ensure_rows(self, source_ids: torch.Tensor) -> None:
        """Read missing source rows into pinned slots, evicting least-recently-used rows."""
        if not self.cached_names or self.capacity == 0 or source_ids.numel() == 0:
            return
        requested = list(dict.fromkeys(int(value) for value in source_ids.tolist()))
        missing = [
            expert_id
            for expert_id in requested
            if int(expert_id) not in self._expert_to_slot
        ]
        assignments = []
        for expert_id in missing:
            free = next(
                (
                    slot
                    for slot, resident in enumerate(self.slot_to_expert)
                    if resident < 0
                ),
                None,
            )
            evicted = free is None
            if free is None:
                free = min(range(self.capacity), key=self._last_used.__getitem__)
            if evicted:
                self.stats.evictions += 1
                self._expert_to_slot.pop(self.slot_to_expert[free], None)
            assignments.append((expert_id, free))
            self.slot_to_expert[free] = expert_id
            self._expert_to_slot[expert_id] = free
            self._clock += 1
            self._last_used[free] = self._clock
        if not assignments:
            return
        # More misses than slots reassign a slot within this call. Read only the
        # slot's final expert: batched file reads complete in any order.
        final_slots = {slot: expert_id for expert_id, slot in assignments}
        source_ids_cpu = torch.tensor(list(final_slots.values()), dtype=torch.long)
        slots_cpu = torch.tensor(list(final_slots), dtype=torch.long)
        file_row_reader = getattr(self.streamer, "file_row_reader", None)
        try:
            for name in self.cached_names:
                if file_row_reader is not None and file_row_reader.covers(name):
                    continue
                source = _tensor_data(getattr(self.streamer.layer, name))
                for source_id, slot in zip(source_ids_cpu, slots_cpu):
                    torch.index_select(
                        source,
                        0,
                        source_id.reshape(1),
                        out=self.tensors[name][int(slot) : int(slot) + 1],
                    )
            if file_row_reader is not None:
                file_row_reader.read(
                    source_ids_cpu,
                    {
                        name: self.tensors[name]
                        for name in self.cached_names
                        if file_row_reader.covers(name)
                    },
                    slots_cpu,
                )
        except BaseException:
            for slot, expert_id in final_slots.items():
                self.slot_to_expert[slot] = -1
                self._expert_to_slot.pop(expert_id, None)
            self._refresh_mapping()
            raise
        self._refresh_mapping()
        self.stats.populated_rows += len(final_slots)
        self.stats.populated_bytes += len(final_slots) * self.bytes_per_expert

    def copy_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> bool:
        """Gather resident pinned rows directly into CUDA outputs."""
        if source_ids.numel() == 0:
            return False
        slots = self.expert_to_slot[source_ids.long()]
        fallback_used = False
        for name in self.cached_names:
            source = self.tensors[name]
            output = outputs[name]
            if source.is_contiguous() and output.is_contiguous():
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
        manager.caches = {
            layer_id: ExpertPinnedHostCache(streamers[layer_id], capacity)
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
    return os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"


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
    torch.cuda.current_stream(source_ids.device).synchronize()
    return indices


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
        self.file_row_reader = self.format.default_row_source(
            layer, self.specs, "auto"
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
        self.row_tag = 0
        self.before_eager_gather = None
        # Set by ExpertPredictionRuntime when SGLANG_MOE_EXPERT_PREFETCH_PULL is on and this
        # layer is a scored prefetch target; see PrefetchPuller.join_target in serving/runtime.py.
        self.prefetch_puller = None
        self.graph_gather_rows = 0
        self.graph_counters: torch.Tensor | None = None
        self.last_gather_stats = ExpertGatherStats()
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
    def file_source_bytes_per_expert(self) -> int | None:
        """File bytes one expert row reads; None keeps eager gathers out of the pinned tier."""
        return self.format.file_source_bytes_per_expert(
            self.layer, self.file_row_reader
        )

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
        if self.pinned_host_cache is not None:
            raise ValueError(
                "graph gather cannot admit rows through the pinned host cache"
            )
        if (
            getattr(self, "prefetch_coordinator", None) is not None
            or getattr(self, "next_layer_prefetch", None) is not None
        ):
            raise ValueError("graph gather cannot run with expert prefetch")
        for name in self.tensor_names:
            source = _tensor_data(getattr(self.layer, name))
            if source.device.type == "cpu" and not is_gpu_readable_host_tensor(source):
                raise ValueError(
                    f"graph gather needs registered host rows; {name!r} is pageable"
                )
        device = cache.device
        self._graph_sources = {
            name: _tensor_data(getattr(self.layer, name)) for name in self.tensor_names
        }
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
        if direct is not None:
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
            current = _tensor_data(getattr(self.layer, name))
            if current.data_ptr() != source.data_ptr() or current.device != source.device:
                raise RuntimeError(
                    f"expert tensor {name!r} moved after graph gather was enabled"
                )

    def _read_host_rows(
        self,
        name: str,
        source: torch.Tensor,
        cpu_ids: torch.Tensor,
        host_output: torch.Tensor,
    ) -> None:
        if self.file_row_reader is not None and self.file_row_reader.covers(name):
            self.file_row_reader.read(cpu_ids, {name: host_output})
        else:
            torch.index_select(source, 0, cpu_ids, out=host_output)

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
        Graph capture keeps the pull kernel. Returns the bytes the copy engine
        moved, so a build without it shows up as zero.
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
        host_sources = {
            name: source
            for name in self.tensor_names
            for source in [_tensor_data(getattr(self.layer, name))]
            if source.device.type == "cpu"
        }
        readable = {
            name: is_gpu_readable_host_tensor(source)
            for name, source in host_sources.items()
        }
        pageable_source = not all(readable.values())
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        for name, output in outputs.items():
            source = _tensor_data(getattr(self.layer, name))
            if source.device.type == "cuda":
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
                assert cpu_ids is not None
                host_output = _pinned_staging_buffer(
                    name, row_count, capacity, tuple(source.shape[1:]), source.dtype
                )
                self._read_host_rows(name, source, cpu_ids, host_output)
                output.copy_(host_output, non_blocking=True)
        return copy_engine_bytes

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
        # the allocator's cache at the prefill peak.
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self.num_experts),
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
            _, pinned_hit_mask = pinned_cache.lookup(miss_source_ids)
            pinned_hit_rows = int(pinned_hit_mask.sum().item())
            pinned_miss_rows = miss_rows - pinned_hit_rows
            populated_before = pinned_cache.stats.populated_bytes
            if pinned_miss_rows:
                pinned_cache.ensure_rows(miss_source_ids[~pinned_hit_mask])
            pinned_populated_bytes = (
                pinned_cache.stats.populated_bytes - populated_before
            )
            source_bytes = pinned_miss_rows * self.host_bytes_per_expert + miss_rows * (
                self.bytes_per_expert - self.host_bytes_per_expert
            )
            pinned_outputs = {
                name: output
                for name, output in misses.items()
                if name in pinned_cache.cached_names
            }
            gather_fallback_used = pinned_cache.copy_rows(
                miss_source_ids, pinned_outputs
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
        """
        cache = self.pinned_host_cache
        assert cache is not None
        row_count = source_ids.numel()
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        slots, hit_mask = cache.lookup(source_ids)
        del slots
        hit_rows = int(hit_mask.sum().item())
        miss_rows = row_count - hit_rows
        copy_engine_bytes = 0
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
        hit_positions = hit_mask.nonzero().flatten()
        miss_positions = (~hit_mask).nonzero().flatten()
        if hit_rows:
            hit_outputs = {
                name: _staging_buffer(
                    ("pinned_host_cache_hits", name),
                    hit_rows,
                    capacity,
                    tuple(output.shape[1:]),
                    output.dtype,
                    output.device,
                )
                for name, output in gathered.items()
                if name in cache.cached_names
            }
            cache.copy_rows(source_ids[hit_mask], hit_outputs)
            for name, output in hit_outputs.items():
                gathered[name].index_copy_(0, hit_positions, output)
        if miss_rows:
            populated_before = cache.stats.populated_bytes
            cache.ensure_rows(source_ids[~hit_mask])
            resident_mask = cache.expert_to_slot[source_ids[~hit_mask].long()] >= 0
            resident_count = int(resident_mask.sum().item())
            miss_outputs = {
                name: _staging_buffer(
                    ("pinned_host_cache_misses", name),
                    resident_count,
                    capacity,
                    tuple(output.shape[1:]),
                    output.dtype,
                    output.device,
                )
                for name, output in gathered.items()
                if name in cache.cached_names
            }
            if bool(resident_mask.any().item()):
                cache.copy_rows(source_ids[~hit_mask][resident_mask], miss_outputs)
                resident_positions = miss_positions[resident_mask]
                for name, output in miss_outputs.items():
                    gathered[name].index_copy_(0, resident_positions, output)
            cold_mask = ~resident_mask
            if bool(cold_mask.any().item()):
                cold_ids = source_ids[~hit_mask][cold_mask]
                cold_positions = miss_positions[cold_mask]
                cold_outputs = {
                    name: _staging_buffer(
                        ("pinned_host_cache_cold", name),
                        cold_ids.numel(),
                        capacity,
                        tuple(output.shape[1:]),
                        output.dtype,
                        output.device,
                    )
                    for name, output in gathered.items()
                    if name in cache.cached_names
                }
                copy_engine_bytes += self._copy_source_rows(cold_ids, cold_outputs)
                for name, output in cold_outputs.items():
                    gathered[name].index_copy_(0, cold_positions, output)
            populated_bytes = cache.stats.populated_bytes - populated_before
        else:
            populated_bytes = 0
        uncached = {
            name: output
            for name, output in gathered.items()
            if name not in cache.cached_names
        }
        if uncached:
            copy_engine_bytes += self._copy_source_rows(source_ids, uncached)
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            0,
            row_count * self.host_bytes_per_expert,
            miss_rows * self.bytes_per_expert,
            hit_rows,
            miss_rows,
            populated_bytes,
            copy_engine_bytes=copy_engine_bytes,
            routed_rows=compact_ids.numel(),
            routed_miss_rows=compact_ids.numel(),
            unique_miss_rows=row_count,
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded

    def _plan_eager_routes(
        self, topk_ids: torch.Tensor
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
        residency_policy = self.residency_policy
        if residency_policy is not None and not (
            flat_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            residency_policy.record_routes(flat_ids)
        return source_ids, compact_ids

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

        source_ids, compact_ids = self._plan_eager_routes(topk_ids)
        if prefetch_coordinator is not None:
            prefetch_coordinator.synchronous_correction(
                source_ids.tolist(), lambda _: None
            )
        next_layer_prefetch = getattr(self, "next_layer_prefetch", None)
        if next_layer_prefetch is not None:
            next_layer_prefetch(source_ids)
        row_count = source_ids.numel()
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
        if (
            self.pinned_host_cache is not None
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
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
        pageable_source = any(
            _tensor_data(getattr(self.layer, name)).device.type == "cpu"
            and not is_gpu_readable_host_tensor(_tensor_data(getattr(self.layer, name)))
            for name in self.tensor_names
        )
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        padded: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            source = _tensor_data(getattr(self.layer, name))
            padded[name] = _staging_buffer(
                name,
                kernel_rows,
                capacity,
                tuple(source.shape[1:]),
                source.dtype,
                topk_ids.device,
            )
            output = padded[name][:row_count]
            if source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif is_gpu_readable_host_tensor(source):
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
                assert cpu_ids is not None
                host_output = _pinned_staging_buffer(
                    name,
                    row_count,
                    capacity,
                    tuple(source.shape[1:]),
                    source.dtype,
                )
                self._read_host_rows(name, source, cpu_ids, host_output)
                output.copy_(host_output, non_blocking=True)

        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded
