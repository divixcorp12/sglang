"""Pinned host storage for per-token state carried between layers of a layer-major prefill, plus parked per-chunk
objects. Holds named fields only: what a field means belongs to the model adapter."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import msgspec
import torch

from sglang.srt.layer_major.tensor_tree import map_tensors


class FieldSpec(msgspec.Struct, frozen=True):
    name: str
    per_token_shape: tuple[int, ...]
    dtype: str


def _dtype(spec: FieldSpec) -> torch.dtype:
    return getattr(torch, spec.dtype)


def state_store_bytes(fields: list[FieldSpec], capacity_tokens: int) -> int:
    return capacity_tokens * sum(math.prod(f.per_token_shape) * _dtype(f).itemsize for f in fields)


class StateStore:
    def __init__(self, fields: list[FieldSpec], capacity_tokens: int, *, numa_node: int | None, pin: bool):
        self.capacity_tokens = capacity_tokens
        self._fields = {f.name: f for f in fields}
        self._host = {f.name: _allocate(f, capacity_tokens, numa_node=numa_node, pin=pin) for f in fields}
        self._pin = pin
        self._parked: dict[int, Any] = {}

    def _rows(self, field: str, start: int, count: int) -> torch.Tensor:
        if start < 0 or start + count > self.capacity_tokens:
            raise ValueError(
                f"rows [{start}, {start + count}) of {field!r} exceed the store's {self.capacity_tokens} tokens"
            )
        return self._host[field][start : start + count]

    def write(self, field: str, start: int, rows: torch.Tensor) -> None:
        self._rows(field, start, rows.shape[0]).copy_(rows)

    def read_into(self, field: str, start: int, out: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        src = self._rows(field, start, out.shape[0])
        with _on(stream):
            out.copy_(src, non_blocking=self._pin)

    def write_from(self, field: str, start: int, src: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        dst = self._rows(field, start, src.shape[0])
        with _on(stream):
            dst.copy_(src, non_blocking=self._pin)

    def park(self, key: int, obj: Any) -> None:
        self._parked[key] = map_tensors(obj, self._to_host)

    def unpark(self, key: int, device: torch.device) -> Any:
        return map_tensors(self._parked[key], lambda t: t.to(device, non_blocking=self._pin))

    def clear_parked(self) -> None:
        self._parked.clear()

    def _to_host(self, t: torch.Tensor) -> torch.Tensor:
        if t.device.type == "cpu":
            return t.clone()
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=self._pin)
        host.copy_(t, non_blocking=self._pin)
        return host


def _allocate(spec: FieldSpec, capacity_tokens: int, *, numa_node: int | None, pin: bool) -> torch.Tensor:
    shape = (capacity_tokens,) + tuple(spec.per_token_shape)
    # cudaHostAlloc (pin_memory=True) faults in pages at allocation time, so the
    # preferred-node bias below actually steers physical placement; a plain
    # torch.empty is lazily paged and this context is then a no-op.
    with _numa_preferred(numa_node):
        return torch.empty(shape, dtype=_dtype(spec), pin_memory=pin)


def _get_libnuma():
    # Load directly rather than via sglang.srt.utils.numa_utils: that module pulls in
    # ServerArgs and the whole server import graph, which this leaf allocator must not.
    for name in ("libnuma.so", "libnuma.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class _numa_preferred:
    """Best-effort MPOL_PREFERRED bias for the calling thread during allocation. Silently does nothing if libnuma
    is unavailable; this is a placement hint, not a hard guarantee."""

    def __init__(self, node: int | None):
        self._libnuma = _get_libnuma() if node is not None else None
        self._node = node

    def __enter__(self):
        if self._libnuma is not None and self._libnuma.numa_available() >= 0:
            self._libnuma.numa_set_preferred(ctypes.c_int(self._node))

    def __exit__(self, *exc):
        if self._libnuma is not None and self._libnuma.numa_available() >= 0:
            self._libnuma.numa_set_preferred(ctypes.c_int(-1))


class _on:
    def __init__(self, stream: torch.cuda.Stream | None):
        self._ctx = torch.cuda.stream(stream) if stream is not None else None

    def __enter__(self):
        if self._ctx is not None:
            self._ctx.__enter__()

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
