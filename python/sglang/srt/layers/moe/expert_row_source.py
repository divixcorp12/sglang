"""Host row sources: the disk-to-RAM half of expert streaming.

A row source fills host rows of the streamed tensors it covers. It knows
files and extents and never kernels: the pinned host tier and the streamer's
staging paths call it and copy the rows to the GPU themselves. This module is
CPU-only and imports nothing from ``sglang.srt.model_loader``, whose package
import reaches the quantization methods that import the streamer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch


class HostSlotLayout(Enum):
    """How a host tier stores one expert row.

    ``PER_NAME`` is one ``[capacity, *row_shape]`` slab per streamed tensor, the
    only layout the framework implements. ``BLOB`` is one byte slot per expert
    holding a source's on-disk row, reserved for a later RAM tier.
    """

    PER_NAME = "per_name"
    BLOB = "blob"


@dataclass(frozen=True)
class RowReadStats:
    """What one or more row reads cost.

    ``file_bytes`` counts bytes read from storage, including any superset
    waste; ``split_bytes`` counts CPU copies that split a read into
    per-tensor rows (0 for sources that read in place).
    """

    rows: int = 0
    file_bytes: int = 0
    split_bytes: int = 0
    read_ns: int = 0
    split_ns: int = 0

    def __add__(self, other: "RowReadStats") -> "RowReadStats":
        return RowReadStats(
            self.rows + other.rows,
            self.file_bytes + other.file_bytes,
            self.split_bytes + other.split_bytes,
            self.read_ns + other.read_ns,
            self.split_ns + other.split_ns,
        )


@runtime_checkable
class ReadTicket(Protocol):
    """A submitted read; ``wait`` returns its stats or raises its error."""

    def done(self) -> bool: ...

    def wait(self) -> RowReadStats: ...


class CompletedReadTicket:
    """A ticket for a read that already ran, successfully or not."""

    def __init__(
        self,
        stats: Optional[RowReadStats] = None,
        error: Optional[BaseException] = None,
    ):
        if (stats is None) == (error is None):
            raise ValueError("a completed read ticket holds either stats or an error")
        self._stats = stats
        self._error = error

    def done(self) -> bool:
        return True

    def wait(self) -> RowReadStats:
        if self._error is not None:
            raise self._error
        assert self._stats is not None
        return self._stats


@runtime_checkable
class ExpertRowSource(Protocol):
    """Fills host rows of the streamed tensors it covers.

    ``read`` copies expert ``rows`` (a CPU integer tensor) of every named
    tensor into ``destinations[name][destination_rows]``, or into the leading
    rows when ``destination_rows`` is None. Destinations are CPU and
    contiguous. It runs synchronously on the calling (model) thread and does
    no CUDA work. A source that reads a whole on-disk expert row per request
    should be given every name it covers in one call. ``submit`` returns a
    ticket; the default (``SynchronousSubmit``) reads immediately.
    ``register_destinations`` registers long-lived host buffers with the
    source's reader and must run on the reader's owner thread.
    ``preferred_batch_rows`` of 0 means no preference.
    """

    names: tuple[str, ...]
    num_experts: int
    host_layouts: frozenset[HostSlotLayout]
    preferred_batch_rows: int
    file_bytes_per_expert: int
    requires_page_aligned_destinations: bool

    def covers(self, name: str) -> bool: ...

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int: ...

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats: ...

    def submit(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> ReadTicket: ...

    def close(self) -> None: ...


class SynchronousSubmit:
    """``submit`` that runs ``read`` at once and returns a completed ticket.

    An error is kept in the ticket and raised by ``wait``, as an asynchronous
    source would report it.
    """

    def submit(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> CompletedReadTicket:
        try:
            if destination_rows is None:
                stats = self.read(rows, destinations)
            else:
                stats = self.read(rows, destinations, destination_rows)
        except Exception as error:
            return CompletedReadTicket(error=error)
        return CompletedReadTicket(stats)


class TensorRowSource(SynchronousSubmit):
    """Rows selected with ``index_select`` from dense ``[experts, ...]`` tensors.

    This is the mmap/tensor fallback: over a file mapping it faults the page
    cache in. ``lookup(name)`` is called on every read, because the host arena
    rebinds layer tensors after startup; a name whose lookup is None is not
    covered. It cannot tell page-cache hits from storage reads, so its stats
    report rows and time but no file bytes.
    """

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    preferred_batch_rows = 0
    file_bytes_per_expert = 0
    requires_page_aligned_destinations = False

    def __init__(
        self,
        lookup: Callable[[str], Optional[torch.Tensor]],
        names: Sequence[str],
        num_experts: int,
    ):
        self._lookup = lookup
        self.names = tuple(names)
        self.num_experts = int(num_experts)

    def covers(self, name: str) -> bool:
        return name in self.names and self._lookup(name) is not None

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        return 0

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        start = time.perf_counter_ns()
        rows = rows.reshape(-1)
        count = rows.numel()
        for name, destination in destinations.items():
            source = self._lookup(name) if name in self.names else None
            if source is None:
                raise ValueError(f"tensor row source does not cover {name!r}")
            if destination_rows is None:
                torch.index_select(
                    source,
                    0,
                    rows,
                    out=destination
                    if destination.shape[0] == count
                    else destination[:count],
                )
            else:
                for row, slot in zip(rows, destination_rows.reshape(-1)):
                    torch.index_select(
                        source,
                        0,
                        row.reshape(1),
                        out=destination[int(slot) : int(slot) + 1],
                    )
        return RowReadStats(rows=count, read_ns=time.perf_counter_ns() - start)

    def close(self) -> None:
        pass
