"""Expert rows for the streaming framework, read straight from the EXL3 shards.

Each expert is one page-aligned superset read (``Exl3RowReader``) into a
page-aligned bounce ring. The format's segment map then splits it into the
per-name destination rows: nine copies per row, with ``mul1`` dropped. There
is no re-layout on disk. The reader and the bounce ring are shared by every
layer, because all reads run synchronously on the model thread.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.expert_row_source import (
    HostSlotLayout,
    RowReadStats,
    SynchronousSubmit,
)
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES

# Superset reads per reader call: 8 x 13.3 MB = 107 MB of bounce for all layers.
BOUNCE_ROWS = 8

_SHARED_READERS: dict[tuple[int, bool], Exl3RowReader] = {}
_SHARED_BOUNCE: dict[tuple[int, int], torch.Tensor] = {}


def shared_row_reader(layout: Exl3ExpertLayout, direct: bool) -> Exl3RowReader:
    """One ``Exl3RowReader`` per layout and read mode, so shard files open once."""
    key = (id(layout), bool(direct))
    reader = _SHARED_READERS.get(key)
    if reader is None or reader.layout is not layout:
        reader = _SHARED_READERS[key] = Exl3RowReader(layout, direct=bool(direct))
    return reader


def shared_bounce(rows: int, row_bytes: int) -> torch.Tensor:
    """A page-aligned uint8 ``[rows, row_bytes]`` host buffer, one per shape."""
    key = (int(rows), int(row_bytes))
    bounce = _SHARED_BOUNCE.get(key)
    if bounce is None:
        storage = torch.empty(rows * row_bytes + PAGE_BYTES, dtype=torch.uint8)
        start = (-storage.data_ptr()) % PAGE_BYTES
        bounce = storage[start : start + rows * row_bytes].view(rows, row_bytes)
        _SHARED_BOUNCE[key] = bounce
    return bounce


class Exl3ShardRowSource(SynchronousSubmit):
    """An ``ExpertRowSource`` (``PER_NAME`` host layout) over one layer's shard rows."""

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    requires_page_aligned_destinations = False

    def __init__(
        self,
        reader: Exl3RowReader,
        layer_id: int,
        segments: Sequence[RowSegment],
        *,
        bounce_rows: int = BOUNCE_ROWS,
    ) -> None:
        layout = reader.layout
        if not 0 <= layer_id < layout.num_layers:
            raise ValueError(
                f"exl3 layer {layer_id} is outside the checkpoint's "
                f"{layout.num_layers} layers"
            )
        if bounce_rows < 1:
            raise ValueError("the bounce ring needs at least one row")
        self.reader = reader
        self.layer_id = layer_id
        self.segments = tuple(segments)
        self.names = tuple(
            name
            for name in EXL3_STREAMED_NAMES
            if any(segment.name == name for segment in self.segments)
        )
        self.row_bytes = {
            name: sum(s.nbytes for s in self.segments if s.name == name)
            for name in self.names
        }
        self.num_experts = layout.num_experts
        self.slot_bytes = -(-reader.buffer_bytes // PAGE_BYTES) * PAGE_BYTES
        self.bounce = shared_bounce(bounce_rows, self.slot_bytes)
        self.preferred_batch_rows = bounce_rows
        # Bytes each row's superset read transfers, page-alignment waste included.
        # A shard's last superset runs past end of file; only bytes that exist
        # are read, as Exl3RowReader counts them.
        file_sizes: dict[str, int] = {}
        self._read_bytes = []
        for expert in range(self.num_experts):
            record = layout.records[(layer_id, expert)]
            offset, length, _ = record.aligned_read(PAGE_BYTES)
            if record.path not in file_sizes:
                file_sizes[record.path] = os.path.getsize(record.path)
            self._read_bytes.append(min(length, file_sizes[record.path] - offset))
        self.file_bytes_per_expert = sum(self._read_bytes) // len(self._read_bytes)

    @classmethod
    def for_layer(
        cls,
        layout: Exl3ExpertLayout,
        layer_id: int,
        segments: Sequence[RowSegment],
        *,
        direct: bool,
    ) -> "Exl3ShardRowSource":
        return cls(shared_row_reader(layout, direct), layer_id, segments)

    def covers(self, name: str) -> bool:
        return name in self.row_bytes

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        # Rows reach their destinations by CPU copies out of the bounce ring,
        # so no destination is handed to io_uring.
        return 0

    def close(self) -> None:
        # The reader and the bounce ring are shared by every layer's source.
        return None

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        unknown = [name for name in destinations if name not in self.row_bytes]
        if unknown:
            raise ValueError(f"exl3 shard row source does not cover {unknown}")
        experts = [int(row) for row in rows.reshape(-1).tolist()]
        if not experts:
            return RowReadStats()
        slots = (
            list(range(len(experts)))
            if destination_rows is None
            else [int(slot) for slot in destination_rows.reshape(-1).tolist()]
        )
        if len(slots) != len(experts):
            raise ValueError("expert rows and destination rows must match in length")
        if any(not 0 <= expert < self.num_experts for expert in experts):
            raise ValueError(
                f"expert row is outside [0, {self.num_experts - 1}]"
            )
        targets = {}
        for name, destination in destinations.items():
            if destination.device.type != "cpu" or not destination.is_contiguous():
                raise ValueError(f"{name}: destination must be a contiguous CPU tensor")
            flat = destination.view(destination.shape[0], -1).view(torch.uint8)
            if flat.shape[1] != self.row_bytes[name]:
                raise ValueError(
                    f"{name}: destination rows hold {flat.shape[1]} bytes, "
                    f"expected {self.row_bytes[name]}"
                )
            if any(not 0 <= slot < flat.shape[0] for slot in slots):
                raise ValueError(f"{name}: destination row is out of range")
            targets[name] = flat
        active = [segment for segment in self.segments if segment.name in targets]
        split_row_bytes = sum(segment.nbytes for segment in active)
        stats = RowReadStats()
        for start in range(0, len(experts), self.preferred_batch_rows):
            chunk = experts[start : start + self.preferred_batch_rows]
            began = time.perf_counter_ns()
            starts = self.reader.read(
                [(self.layer_id, expert) for expert in chunk],
                [self.bounce[i].data_ptr() for i in range(len(chunk))],
            )
            read_done = time.perf_counter_ns()
            for i, (slot, row_start) in enumerate(
                zip(slots[start : start + len(chunk)], starts)
            ):
                source = self.bounce[i]
                for segment in active:
                    at = row_start + segment.src_offset
                    targets[segment.name][
                        slot, segment.dst_offset : segment.dst_offset + segment.nbytes
                    ].copy_(source[at : at + segment.nbytes])
            split_done = time.perf_counter_ns()
            stats = stats + RowReadStats(
                rows=len(chunk),
                file_bytes=sum(self._read_bytes[expert] for expert in chunk),
                split_bytes=len(chunk) * split_row_bytes,
                read_ns=read_done - began,
                split_ns=split_done - read_done,
            )
        return stats
