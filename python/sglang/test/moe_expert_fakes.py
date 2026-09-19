"""Test doubles for the MoE expert streaming plugin seams (formats and row sources)."""

from __future__ import annotations

from typing import Iterable, Mapping, NamedTuple, Optional

import torch

from sglang.srt.layers.moe.expert_row_source import (
    HostSlotLayout,
    RowReadStats,
    SynchronousSubmit,
)


class RowSourceCall(NamedTuple):
    rows: list[int]
    names: tuple[str, ...]
    destination_rows: Optional[list[int]]


class CountingRowSource(SynchronousSubmit):
    """A row source over in-memory ``[experts, ...]`` tensors that records every read.

    ``read_ns`` reports one nanosecond per row, so tests can check how stats
    are summed without a clock.
    """

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    preferred_batch_rows = 0
    requires_page_aligned_destinations = False

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        self.tensors = dict(tensors)
        self.names = tuple(self.tensors)
        self.num_experts = next(iter(self.tensors.values())).shape[0]
        self.file_bytes_per_expert = sum(self._row_bytes(name) for name in self.names)
        self.calls: list[RowSourceCall] = []
        self.registered: list[torch.Tensor] = []
        self.closed = False

    def _row_bytes(self, name: str) -> int:
        tensor = self.tensors[name]
        return tensor[0].numel() * tensor.element_size()

    def covers(self, name: str) -> bool:
        return name in self.tensors

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        self.registered.extend(tensors)
        return 0

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        missing = [name for name in destinations if name not in self.tensors]
        if missing:
            raise ValueError(f"counting row source does not cover {missing}")
        row_list = [int(row) for row in rows.reshape(-1).tolist()]
        slots = (
            list(range(len(row_list)))
            if destination_rows is None
            else [int(slot) for slot in destination_rows.reshape(-1).tolist()]
        )
        self.calls.append(
            RowSourceCall(
                row_list,
                tuple(destinations),
                None if destination_rows is None else slots,
            )
        )
        for name, destination in destinations.items():
            source = self.tensors[name]
            for row, slot in zip(row_list, slots):
                destination[slot].copy_(source[row])
        return RowReadStats(
            rows=len(row_list),
            file_bytes=len(row_list)
            * sum(self._row_bytes(name) for name in destinations),
            read_ns=len(row_list),
        )

    def close(self) -> None:
        self.closed = True
