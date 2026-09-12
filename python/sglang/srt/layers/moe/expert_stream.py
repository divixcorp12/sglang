"""Selected-expert staging for host-resident MoE weight rows."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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
    """Gather aligned expert rows into compact, reusable CUDA buffers."""

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
        self.hot_cache = None
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
                torch.index_select(source, 0, cpu_ids, out=host_output)
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
        if hit_rows == 0:
            self._copy_source_rows(source_ids, gathered)
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
            self._copy_source_rows(source_ids[~hit_mask], misses)
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
            assembly_bytes = row_count * self.bytes_per_expert
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            hit_rows,
            miss_rows,
            assembly_bytes
            + miss_rows * (self.bytes_per_expert - self.host_bytes_per_expert),
            miss_rows * self.host_bytes_per_expert,
            miss_rows * self.bytes_per_expert,
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered

    def gather(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if topk_ids.device.type != "cuda":
            raise ValueError("selected expert IDs must be on CUDA")
        flat_ids = topk_ids.reshape(-1)
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

        row_count = source_ids.numel()
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
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
                torch.index_select(source, 0, cpu_ids, out=host_output)
                output.copy_(host_output, non_blocking=True)
            gathered[name] = output

        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered
