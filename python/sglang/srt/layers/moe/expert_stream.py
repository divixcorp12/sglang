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

logger = logging.getLogger(__name__)

_NO_DEDUP_LIMIT = 64
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
            name
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cpu"
        )
        if capacity and not self.cached_names:
            raise ValueError(
                "pinned host cache requires at least one CPU source tensor"
            )
        self.bytes_per_expert = streamer.host_bytes_per_expert
        self.residency_bytes = capacity * self.bytes_per_expert
        devices = {
            _tensor_data(getattr(streamer.layer, name)).device
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cuda"
        }
        if len(devices) > 1:
            raise ValueError("pinned host cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity,)
                + tuple(_tensor_data(getattr(streamer.layer, name)).shape[1:]),
                dtype=_tensor_data(getattr(streamer.layer, name)).dtype,
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
        for module in model.modules():
            streamer = getattr(module, "_nvfp4_expert_streamer", None)
            if streamer is None or streamer.host_bytes_per_expert == 0:
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
    key = (name, dtype, str(device), row_shape)
    buffer = _STAGING.get(key)
    if buffer is None or buffer.shape[0] < max_rows:
        buffer = torch.empty(
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
    ):
        self.layer = layer
        self.layer_id = (
            getattr(layer, "layer_id", None) if layer_id is None else layer_id
        )
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")

        self.num_experts = self._validate_sources()
        # Imported here: the reader pulls in sglang.srt.model_loader, whose
        # package import reaches modelopt_quant, which imports this module.
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        self.file_row_reader = ExpertFileRowReader.from_layer(layer, self.tensor_names)
        self.hot_cache = None
        self.pinned_host_cache = None
        self.residency_policy = None
        self.last_gather_stats = ExpertGatherStats()
        self.bytes_per_expert = sum(
            _tensor_data(getattr(layer, name)).numel()
            * _tensor_data(getattr(layer, name)).element_size()
            // self.num_experts
            for name in self.tensor_names
        )
        self.host_bytes_per_expert = sum(
            _tensor_data(getattr(layer, name)).numel()
            * _tensor_data(getattr(layer, name)).element_size()
            // self.num_experts
            for name in self.tensor_names
            if _tensor_data(getattr(layer, name)).device.type == "cpu"
        )
        signature = tuple(
            (
                name,
                tuple(_tensor_data(getattr(layer, name)).shape),
                str(_tensor_data(getattr(layer, name)).dtype),
                str(_tensor_data(getattr(layer, name)).device),
            )
            for name in self.tensor_names
        )
        if signature not in _LOGGED_SOURCE_SIGNATURES:
            _LOGGED_SOURCE_SIGNATURES.add(signature)
            logger.info(
                "MoE selected-expert streaming active: experts=%d tensors=%s",
                self.num_experts,
                ",".join(self.tensor_names),
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
        expert_count = None
        for name in self.tensor_names:
            if not hasattr(self.layer, name):
                raise ValueError(f"expert source tensor {name!r} is missing")
            tensor = _tensor_data(getattr(self.layer, name))
            if tensor.ndim == 0:
                raise ValueError(
                    f"expert source tensor {name!r} has no expert dimension"
                )
            if tensor.shape[0] == 0:
                raise ValueError(f"expert source tensor {name!r} has no expert rows")
            if not tensor.is_contiguous():
                raise ValueError(f"expert source tensor {name!r} must be contiguous")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"expert source tensor {name!r} uses unsupported device {tensor.device}"
                )
            if expert_count is None:
                expert_count = tensor.shape[0]
            elif tensor.shape[0] != expert_count:
                raise ValueError(
                    f"expert count mismatch for {name!r}: {tensor.shape[0]} != {expert_count}"
                )
        assert expert_count is not None
        return expert_count

    def _copy_source_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> None:
        """Fill supplied CUDA rows through the existing bounded host buffers."""
        row_count = source_ids.numel()
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        pageable_source = any(
            _tensor_data(getattr(self.layer, name)).device.type == "cpu"
            and not _tensor_data(getattr(self.layer, name)).is_pinned()
            for name in self.tensor_names
        )
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        for name, output in outputs.items():
            source = _tensor_data(getattr(self.layer, name))
            if source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif source.is_pinned():
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

    def _gather_cached(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache = self.hot_cache
        slots, hit_mask = cache.lookup(source_ids)
        row_count = source_ids.numel()
        hit_rows = int(hit_mask.sum().item())
        miss_rows = row_count - hit_rows
        if miss_rows == 0:
            self.last_gather_stats = ExpertGatherStats(row_count, hit_rows)
            return slots[compact_ids.long()].reshape(topk_ids.shape).to(
                topk_ids.dtype
            ), cache.tensors
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        gathered = {
            name: _staging_buffer(
                name,
                row_count,
                capacity,
                tuple(source.shape[1:]),
                source.dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
            for source in [_tensor_data(getattr(self.layer, name))]
        }
        miss_source_ids = source_ids[~hit_mask]
        if hit_rows == 0:
            misses = gathered
            assembly_bytes = 0
        else:
            hit_positions = hit_mask.nonzero().flatten()
            hot_slots = slots[hit_mask]
            miss_positions = (~hit_mask).nonzero().flatten()
            misses = {
                name: _staging_buffer(
                    ("hot_cache_misses", name),
                    miss_rows,
                    capacity,
                    tuple(output.shape[1:]),
                    output.dtype,
                    output.device,
                )
                for name, output in gathered.items()
            }
            assembly_bytes = row_count * self.bytes_per_expert
        pinned_hit_rows = 0
        pinned_miss_rows = 0
        pinned_populated_bytes = 0
        gather_fallback_used = False
        pinned_cache = self.pinned_host_cache
        source_bytes = miss_rows * self.bytes_per_expert
        if (
            pinned_cache is not None
            and pinned_cache.capacity
            and getattr(self.layer, "_nvfp4_file_source_bytes_per_expert", None)
            is not None
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
                self._copy_source_rows(miss_source_ids, uncached_outputs)
        else:
            self._copy_source_rows(miss_source_ids, misses)
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
                output.view(torch.uint8).reshape(row_count, -1).index_copy_(
                    0,
                    miss_positions,
                    misses[name].view(torch.uint8).reshape(miss_rows, -1),
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
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered

    def _gather_pinned_host(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache = self.pinned_host_cache
        assert cache is not None
        row_count = source_ids.numel()
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        slots, hit_mask = cache.lookup(source_ids)
        del slots
        hit_rows = int(hit_mask.sum().item())
        miss_rows = row_count - hit_rows
        gathered = {
            name: _staging_buffer(
                name,
                row_count,
                capacity,
                tuple(_tensor_data(getattr(self.layer, name)).shape[1:]),
                _tensor_data(getattr(self.layer, name)).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
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
                self._copy_source_rows(cold_ids, cold_outputs)
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
            self._copy_source_rows(source_ids, uncached)
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
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered

    def gather(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if topk_ids.device.type != "cuda":
            raise ValueError("selected expert IDs must be on CUDA")
        flat_ids = topk_ids.reshape(-1)
        prefetch_coordinator = getattr(self, "prefetch_coordinator", None)
        if flat_ids.numel() == 0:
            raise ValueError("selected expert IDs cannot be empty")
        if bool(((flat_ids < 0) | (flat_ids >= self.num_experts)).any().item()):
            raise ValueError(
                f"selected expert ID is outside [0, {self.num_experts - 1}]"
            )

        if flat_ids.numel() <= _NO_DEDUP_LIMIT:
            source_ids = flat_ids
            compact_ids = _cached_arange(
                flat_ids.numel(), flat_ids.device, topk_ids.dtype
            )
        else:
            source_ids, compact_ids = torch.unique(
                flat_ids, sorted=True, return_inverse=True
            )

        residency_policy = self.residency_policy
        if (
            residency_policy is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            residency_policy.record_routes(source_ids)
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
            and getattr(self.layer, "_nvfp4_file_source_bytes_per_expert", None)
            is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            row_count * (self.bytes_per_expert - self.host_bytes_per_expert),
            row_count * self.host_bytes_per_expert,
            row_count * self.bytes_per_expert,
        )
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        pageable_source = any(
            _tensor_data(getattr(self.layer, name)).device.type == "cpu"
            and not _tensor_data(getattr(self.layer, name)).is_pinned()
            for name in self.tensor_names
        )
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        gathered: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            source = _tensor_data(getattr(self.layer, name))
            output = _staging_buffer(
                name,
                row_count,
                capacity,
                tuple(source.shape[1:]),
                source.dtype,
                topk_ids.device,
            )
            if source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif source.is_pinned():
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
            gathered[name] = output

        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered
