"""Selected-expert staging for host-resident MoE weight rows."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_NO_DEDUP_LIMIT = 64
_STAGING: Dict[Tuple, torch.Tensor] = {}
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


def _cached_arange(
    size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
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


class ExpertStreamer:
    """Gather aligned expert rows into compact, reusable CUDA buffers."""

    def __init__(self, layer: torch.nn.Module, tensor_names: Iterable[str]):
        self.layer = layer
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")

        self.num_experts = self._validate_sources()
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
                raise ValueError(f"expert source tensor {name!r} has no expert dimension")
            if tensor.shape[0] == 0:
                raise ValueError(f"expert source tensor {name!r} has no expert rows")
            if not tensor.is_contiguous():
                raise ValueError(f"expert source tensor {name!r} must be contiguous")
            if tensor.device.type == "cpu" and not tensor.is_pinned():
                raise RuntimeError(
                    f"expert source tensor {name!r} must use pinned CPU memory"
                )
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"expert source tensor {name!r} uses unsupported device {tensor.device}"
                )
            if expert_count is None:
                expert_count = tensor.shape[0]
            elif tensor.shape[0] != expert_count:
                raise ValueError(
                    f"expert count mismatch for {name!r}: "
                    f"{tensor.shape[0]} != {expert_count}"
                )
        assert expert_count is not None
        return expert_count

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
        gathered: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            source = _tensor_data(getattr(self.layer, name))
            output = _staging_buffer(
                name,
                row_count,
                max(self.num_experts, _NO_DEDUP_LIMIT),
                tuple(source.shape[1:]),
                source.dtype,
                topk_ids.device,
            )
            if source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            else:
                source_bytes = source.view(torch.uint8)
                output_bytes = output.view(torch.uint8)
                row_bytes = source_bytes.numel() // self.num_experts
                block = 1024
                _gather_host_rows_kernel[
                    (row_count, triton.cdiv(row_bytes, block))
                ](
                    source_bytes,
                    source_ids,
                    output_bytes,
                    row_bytes,
                    BLOCK=block,
                )
            gathered[name] = output

        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), gathered
