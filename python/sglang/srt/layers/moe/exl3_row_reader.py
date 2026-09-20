"""Read whole EXL3 expert rows from the original shards, one aligned read each.

A row's file offset is not page-aligned, so each read covers the page-aligned
superset of the row and the caller finds the row at a per-expert offset inside
its buffer. That keeps O_DIRECT possible with no on-disk re-layout.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_read_split import SplitPolicy
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
        source_root: Optional[str] = None,
    ) -> None:
        self.layout = layout
        self._reader = reader if reader is not None else shared_uring_file_reader()
        self._direct = direct
        # Open files by (root, path): root is None for the layout's own paths
        # (`read`) and a mirror root for that root's copy (`read_split`).
        self._files: dict[tuple[Optional[str], str], tuple[int, int]] = {}
        # The directory the layout's paths live under; a mirror root holds the
        # same tree, so a record's path relative to this picks its file there.
        self._source_root = source_root
        self.buffer_bytes = max(
            record.aligned_read(PAGE_BYTES)[1] for record in layout.records.values()
        )

    def _file(self, path: str, root: Optional[str] = None) -> tuple[int, int]:
        """(file id, size) of ``path``, or of ``root``'s copy of it."""
        entry = self._files.get((root, path))
        if entry is None:
            target = path if root is None else self._mirror_path(root, path)
            file_id = self._reader.open(target, direct=self._direct)
            entry = self._files[(root, path)] = (
                file_id,
                self._reader.file_size(file_id),
            )
        return entry

    def _mirror_path(self, root: str, path: str) -> str:
        if self._source_root is None:
            self._source_root = os.path.commonpath(
                [os.path.dirname(r.path) for r in self.layout.records.values()]
            )
        relative = os.path.relpath(path, self._source_root)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            raise ValueError(f"{path} is not under source root {self._source_root}")
        return os.path.join(root, relative)

    def _submit(
        self,
        file_ids: list[int],
        offsets: list[int],
        destinations: list[int],
        lengths: list[int],
        expected: int,
    ) -> None:
        """The one ``UringFileReader.read`` of a batch; fails if it fell short."""
        actual = self._reader.read(
            torch.tensor(file_ids, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(destinations, dtype=torch.int64),
            torch.tensor(lengths, dtype=torch.int64),
        )
        if actual != expected:
            raise RuntimeError(
                f"EXL3 expert read ended early: read {actual} of {expected} bytes"
            )

    def read(
        self, keys: Sequence[tuple[int, int]], destinations: Sequence[int]
    ) -> list[int]:
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
        self._submit(file_ids, offsets, list(destinations), lengths, expected)
        return starts

    def read_split(
        self,
        keys: Sequence[tuple[int, int]],
        destinations: Sequence[int],
        *,
        roots: Sequence[str],
        policy: SplitPolicy,
    ) -> list[int]:
        """Like ``read``, but each row is served by ``len(roots)`` mirrored copies
        of the checkpoint, one sub-range per root as ``policy`` plans it.

        Part ``i`` of a row reads ``roots[i]``'s copy at ``offset + starts[i]``
        into ``destination + starts[i]``, so the row's aligned superset still
        lands contiguously and the returned row starts are the same as
        ``read``'s. Every part of every row goes out in one submit; a 0-byte
        part issues no read.
        """
        if len(keys) != len(destinations):
            raise ValueError("one destination per expert")
        if not roots:
            raise ValueError("read_split needs at least one root")
        if not keys:
            return []
        if any(address % PAGE_BYTES for address in destinations):
            raise ValueError("expert row destinations must be page-aligned")
        file_ids, offsets, dests, lengths, starts = [], [], [], [], []
        expected = 0
        for key, destination in zip(keys, destinations):
            record = self.layout.records[key]
            offset, length, start = record.aligned_read(PAGE_BYTES)
            split = policy.plan(length)
            if len(split.part_bytes) != len(roots):
                raise ValueError(
                    f"split policy planned {len(split.part_bytes)} parts "
                    f"for {len(roots)} roots"
                )
            for root, part_start, part_bytes in zip(
                roots, split.starts, split.part_bytes
            ):
                if part_bytes == 0:
                    continue
                file_id, file_bytes = self._file(record.path, root)
                part_offset = offset + part_start
                file_ids.append(file_id)
                offsets.append(part_offset)
                dests.append(destination + part_start)
                lengths.append(part_bytes)
                # Only a shard's last row can run past end of file, and within
                # it only its last non-empty part; clamp each part on its own
                # offset rather than the row's.
                expected += min(part_bytes, file_bytes - part_offset)
            starts.append(start)
        self._submit(file_ids, offsets, dests, lengths, expected)
        return starts
