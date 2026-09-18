"""Read whole EXL3 expert rows from the original shards, one aligned read each.

A row's file offset is not page-aligned, so each read covers the page-aligned
superset of the row and the caller finds the row at a per-expert offset inside
its buffer. That keeps O_DIRECT possible with no on-disk re-layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES, shared_uring_file_reader

if TYPE_CHECKING:
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader


class Exl3RowReader:
    def __init__(
        self,
        layout: Exl3ExpertLayout,
        reader: Optional[UringFileReader] = None,
        *,
        direct: bool = True,
    ) -> None:
        self.layout = layout
        self._reader = reader if reader is not None else shared_uring_file_reader()
        self._direct = direct
        self._files: dict[str, tuple[int, int]] = {}
        self.buffer_bytes = max(
            record.aligned_read(PAGE_BYTES)[1] for record in layout.records.values()
        )

    def _file(self, path: str) -> tuple[int, int]:
        entry = self._files.get(path)
        if entry is None:
            file_id = self._reader.open(path, direct=self._direct)
            entry = self._files[path] = (file_id, self._reader.file_size(file_id))
        return entry

    def read(self, keys: Sequence[tuple[int, int]], destinations: Sequence[int]) -> list[int]:
        """Read each expert into a page-aligned host address with room for
        ``buffer_bytes``; return where each row starts inside its buffer."""
        if len(keys) != len(destinations):
            raise ValueError("one destination per expert")
        if not keys:
            return []
        if any(address % PAGE_BYTES for address in destinations):
            raise ValueError("expert row destinations must be page-aligned")
        file_ids, offsets, lengths, starts = [], [], [], []
        expected = 0
        for key in keys:
            record = self.layout.records[key]
            offset, length, start = record.aligned_read(PAGE_BYTES)
            file_id, file_bytes = self._file(record.path)
            file_ids.append(file_id)
            offsets.append(offset)
            lengths.append(length)
            starts.append(start)
            # The superset of a shard's last row may run past end of file.
            expected += min(length, file_bytes - offset)
        actual = self._reader.read(
            torch.tensor(file_ids, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(list(destinations), dtype=torch.int64),
            torch.tensor(lengths, dtype=torch.int64),
        )
        if actual != expected:
            raise RuntimeError(f"EXL3 expert read ended early: read {actual} of {expected} bytes")
        return starts
