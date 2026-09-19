"""Read fixed-width rows of file-backed tensors with io_uring instead of an mmap.

Two layouts are served:

``AlignedRowSource``
    Rows are whole multiples of a 4 KiB page (NVFP4 expert rows). Each row is
    one extent read straight into its destination row. With ``direct`` and a
    page-aligned destination the read uses ``O_DIRECT``, so the file never
    populates the page cache.

``PagedRowSource``
    Rows are smaller than a page (PLE rows are 160 B). The distinct pages under
    the requested rows are coalesced into runs of consecutive pages, each run is
    one extent read into an aligned bounce buffer, and the rows are gathered out
    of it. ``direct`` selects ``O_DIRECT`` for those page reads.

Reader modes, selected per consumer by environment variable:

``mmap``          the existing shared-mapping reads (default)
``uring``         buffered io_uring reads, served from the page cache when warm
``uring_direct``  ``O_DIRECT`` io_uring reads that bypass the page cache
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader

FILE_READER_MODES = ("mmap", "uring", "uring_direct")
PAGE_BYTES = 4096
_PAGE_SHIFT = 12


def validate_file_reader_mode(mode: str) -> str:
    """Return ``mode`` when it names a supported reader, else raise ``ValueError``."""
    if mode not in FILE_READER_MODES:
        raise ValueError(
            f"unknown file reader mode {mode!r}; choose from {FILE_READER_MODES}"
        )
    return mode


def shared_uring_file_reader() -> UringFileReader:
    """The process-wide io_uring reader, sized by ``SGLANG_URING_FILE_READER_QUEUE_DEPTH``."""
    from sglang.kernels.ops.io.uring_file_reader import get_shared_uring_file_reader

    return get_shared_uring_file_reader(envs.SGLANG_URING_FILE_READER_QUEUE_DEPTH.get())


class FileRowPlan(NamedTuple):
    """Extents for one batched read: file ids, offsets, addresses, byte lengths."""

    file_ids: torch.Tensor
    offsets: torch.Tensor
    destinations: torch.Tensor
    lengths: torch.Tensor


def read_plans(reader: UringFileReader, plans: Sequence[FileRowPlan]) -> None:
    """Submit every plan as one batch and require every byte to arrive."""
    plans = [plan for plan in plans if plan.file_ids.numel()]
    if not plans:
        return
    lengths = torch.cat([plan.lengths for plan in plans])
    expected = int(lengths.sum())
    actual = reader.read(
        torch.cat([plan.file_ids for plan in plans]),
        torch.cat([plan.offsets for plan in plans]),
        torch.cat([plan.destinations for plan in plans]),
        lengths,
    )
    if actual != expected:
        raise RuntimeError(
            f"io_uring row read ended early: read {actual} of {expected} bytes"
        )


def _cpu_row_ids(rows: torch.Tensor, row_count: int) -> torch.Tensor:
    if rows.device.type != "cpu":
        raise ValueError("file row ids must be a CPU tensor")
    rows = rows.reshape(-1).to(torch.int64)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= row_count):
        raise IndexError(f"file row id outside [0, {row_count})")
    return rows


def _row_bytes_of(destination: torch.Tensor) -> int:
    if destination.device.type != "cpu":
        raise ValueError("file row destination must be a CPU tensor")
    if not destination.is_contiguous():
        raise ValueError("file row destination must be contiguous")
    if destination.ndim == 0 or destination.shape[0] == 0:
        return 0
    return destination[0].numel() * destination.element_size()


class AlignedRowSource:
    """Rows of ``row_bytes`` at file offset ``row * row_bytes``, read in place."""

    def __init__(
        self,
        reader: UringFileReader,
        path: str | os.PathLike[str],
        row_bytes: int,
        row_count: int,
        *,
        direct: bool,
    ) -> None:
        self.path = os.fspath(path)
        self.row_bytes = int(row_bytes)
        self.row_count = int(row_count)
        if self.row_bytes <= 0:
            raise ValueError("file row width must be positive")
        self._buffered_file = reader.open(self.path, direct=False)
        size = reader.file_size(self._buffered_file)
        if size < self.row_bytes * self.row_count:
            raise ValueError(
                f"{self.path} holds {size} bytes, fewer than "
                f"{self.row_count} rows of {self.row_bytes} bytes"
            )
        self.direct = bool(direct) and self.row_bytes % PAGE_BYTES == 0
        self._direct_file = reader.open(self.path, direct=True) if self.direct else None

    def plan(
        self,
        rows: torch.Tensor,
        destination: torch.Tensor,
        destination_rows: Optional[torch.Tensor] = None,
    ) -> FileRowPlan:
        """Extents reading ``rows`` into ``destination[destination_rows]``."""
        rows = _cpu_row_ids(rows, self.row_count)
        if rows.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device="cpu")
            return FileRowPlan(empty, empty, empty, empty)
        if _row_bytes_of(destination) != self.row_bytes:
            raise ValueError(
                f"file row destination rows hold {_row_bytes_of(destination)} bytes, "
                f"expected {self.row_bytes}"
            )
        if destination_rows is None:
            if rows.numel() > destination.shape[0]:
                raise ValueError("file row destination has fewer rows than requested")
            destination_rows = torch.arange(
                rows.numel(), dtype=torch.int64, device="cpu"
            )
        else:
            destination_rows = _cpu_row_ids(destination_rows, destination.shape[0])
            if destination_rows.numel() != rows.numel():
                raise ValueError("file rows and destination rows must match in length")
        base = destination.data_ptr()
        use_direct = self._direct_file is not None and base % PAGE_BYTES == 0
        file_id = self._direct_file if use_direct else self._buffered_file
        count = rows.numel()
        return FileRowPlan(
            torch.full((count,), file_id, dtype=torch.int64, device="cpu"),
            rows * self.row_bytes,
            destination_rows * self.row_bytes + base,
            torch.full((count,), self.row_bytes, dtype=torch.int64, device="cpu"),
        )


class PagedRowSource:
    """Sub-page rows read through coalesced page runs and an aligned bounce buffer."""

    def __init__(
        self,
        reader: UringFileReader,
        path: str | os.PathLike[str],
        row_bytes: int,
        row_count: int,
        *,
        direct: bool,
        base_offset: int = 0,
    ) -> None:
        self.path = os.fspath(path)
        self.row_bytes = int(row_bytes)
        self.row_count = int(row_count)
        # Row 0 starts here (a table inside a safetensors shard).
        self.base_offset = int(base_offset)
        if not 0 < self.row_bytes <= PAGE_BYTES:
            raise ValueError("paged file rows must be between 1 byte and one page")
        self._reader = reader
        self._buffered_file = reader.open(self.path, direct=False)
        self._file_bytes = reader.file_size(self._buffered_file)
        if self._file_bytes < self.base_offset + self.row_bytes * self.row_count:
            raise ValueError(
                f"{self.path} holds {self._file_bytes} bytes, fewer than "
                f"{self.row_count} rows of {self.row_bytes} bytes"
            )
        self.direct = bool(direct)
        self._file = (
            reader.open(self.path, direct=True) if direct else self._buffered_file
        )
        self._row_offsets = torch.arange(self.row_bytes, dtype=torch.int64, device="cpu")
        self._bounce_storage: Optional[torch.Tensor] = None
        self._bounce = torch.empty(0, dtype=torch.uint8, device="cpu")

    @property
    def bounce_bytes(self) -> int:
        return self._bounce.numel()

    def _bounce_for(self, page_count: int) -> torch.Tensor:
        needed = page_count * PAGE_BYTES
        if self._bounce.numel() < needed:
            capacity = max(needed, 2 * self._bounce.numel())
            storage = torch.empty(capacity + PAGE_BYTES, dtype=torch.uint8, device="cpu")
            start = (-storage.data_ptr()) % PAGE_BYTES
            self._bounce_storage = storage
            self._bounce = storage[start : start + capacity]
        return self._bounce[:needed]

    def read_rows(self, rows: torch.Tensor, destination: torch.Tensor) -> None:
        """Fill ``destination[i]`` with file row ``rows[i]``."""
        rows = _cpu_row_ids(rows, self.row_count)
        count = rows.numel()
        if count == 0:
            return
        if destination.shape[0] < count or _row_bytes_of(destination) != self.row_bytes:
            raise ValueError(
                f"paged row destination must hold {count} rows of {self.row_bytes} bytes"
            )
        starts = self.base_offset + rows * self.row_bytes
        first_pages = starts >> _PAGE_SHIFT
        last_pages = (starts + (self.row_bytes - 1)) >> _PAGE_SHIFT
        pages = torch.unique(torch.cat([first_pages, last_pages]), sorted=True)
        run_starts = torch.ones(pages.numel(), dtype=torch.bool, device="cpu")
        run_starts[1:] = pages[1:] != pages[:-1] + 1
        run_ids = torch.cumsum(run_starts, 0) - 1
        run_first_pages = pages[run_starts]
        run_page_counts = torch.bincount(run_ids)
        run_bounce_pages = torch.cumsum(run_page_counts, 0) - run_page_counts
        page_bounce_pages = run_bounce_pages[run_ids] + (
            pages - run_first_pages[run_ids]
        )

        bounce = self._bounce_for(int(run_page_counts.sum()))
        offsets = run_first_pages * PAGE_BYTES
        lengths = run_page_counts * PAGE_BYTES
        # O_DIRECT needs whole-page lengths, so the final page of the file may
        # be requested past its end; only the bytes before it are owed.
        expected = int(
            torch.clamp(torch.minimum(lengths, self._file_bytes - offsets), min=0).sum()
        )
        actual = self._reader.read(
            torch.full((offsets.numel(),), self._file, dtype=torch.int64, device="cpu"),
            offsets,
            run_bounce_pages * PAGE_BYTES + bounce.data_ptr(),
            lengths,
        )
        if actual != expected:
            raise RuntimeError(
                f"io_uring page read ended early: read {actual} of {expected} bytes"
            )

        row_pages = torch.searchsorted(pages, first_pages)
        row_bounce_bytes = page_bounce_pages[row_pages] * PAGE_BYTES + (
            starts & (PAGE_BYTES - 1)
        )
        gather_index = (row_bounce_bytes[:, None] + self._row_offsets).reshape(-1)
        torch.index_select(
            bounce,
            0,
            gather_index,
            out=destination[:count].view(torch.uint8).reshape(-1),
        )


class PagedRowBatch:
    """Rows of several ``PagedRowSource`` files read with one native call.

    The native reader plans the pages under every source's rows, coalesces
    them into runs, reads all runs in one io_uring batch and scatters the rows,
    so a step that touches many tables pays one Python call and one batch.
    Source ``i`` treats ids inside ``row_windows[i] = (start, end)`` as file row
    ``id - start`` and writes zeros for every other id.
    """

    def __init__(
        self,
        sources: Sequence[PagedRowSource],
        row_windows: Sequence[tuple[int, int]],
    ) -> None:
        if not sources:
            raise ValueError("a paged row batch needs at least one source")
        if len(row_windows) != len(sources):
            raise ValueError("paged row batch needs one row window per source")
        self._reader = sources[0]._reader
        if any(source._reader is not self._reader for source in sources):
            raise ValueError("paged row batch sources must share one reader")
        self.row_bytes = tuple(source.row_bytes for source in sources)
        self._segments = torch.tensor(
            [
                [source._file, source.row_bytes, source.row_count, int(start), int(end), 0, 0]
                for source, (start, end) in zip(sources, row_windows)
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._layout: Optional[tuple] = None
        self._id_total = 0

    def read_rows(
        self,
        ids: torch.Tensor,
        id_counts: Sequence[int],
        destination: torch.Tensor,
    ) -> None:
        """Fill ``destination`` with every source's rows, back to back in source order.

        Source ``i`` reads the next ``id_counts[i]`` entries of ``ids`` into the
        next ``id_counts[i] * row_bytes[i]`` bytes of the flat ``destination``.
        """
        layout = (tuple(id_counts), destination.data_ptr(), destination.numel())
        if layout != self._layout:
            self._set_layout(id_counts, destination)
            self._layout = layout
        if ids.numel() != self._id_total:
            raise ValueError(
                f"paged row batch expected {self._id_total} ids, got {ids.numel()}"
            )
        if ids.device.type != "cpu":
            raise ValueError("file row ids must be a CPU tensor")
        self._reader.read_paged_rows(self._segments, ids)

    def _set_layout(self, id_counts: Sequence[int], destination: torch.Tensor) -> None:
        if len(id_counts) != len(self.row_bytes):
            raise ValueError("paged row batch needs one id count per source")
        if destination.device.type != "cpu" or not destination.is_contiguous():
            raise ValueError("file row destination must be a contiguous CPU tensor")
        counts = torch.tensor([int(count) for count in id_counts], dtype=torch.int64)
        if bool((counts < 0).any()):
            raise ValueError("paged row id counts must be non-negative")
        byte_counts = counts * self._segments[:, 1]
        total_bytes = int(byte_counts.sum())
        if total_bytes > destination.numel() * destination.element_size():
            raise ValueError(
                f"paged row destination must hold {total_bytes} bytes"
            )
        self._segments[:, 5] = counts
        self._segments[:, 6] = (
            destination.data_ptr() + torch.cumsum(byte_counts, 0) - byte_counts
        )
        self._id_total = int(counts.sum())
