"""CUDA copy-engine backend for uncaptured NVFP4 expert-row transfers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from operator import index

import torch

from sglang.srt.utils.cuda_host_registry import (
    cuda_host_registration_end,
    is_gpu_readable_host_tensor,
)

try:
    from sgl_kernel.kvcacheio import transfer_embedding_ranges_direct
except ImportError:
    transfer_embedding_ranges_direct = None


class ExpertDMABackend:
    """Copy selected pinned or CUDA-registered host expert rows into CUDA rows.

    Row pairs that advance together are merged into one range, so a run of
    consecutive experts is one copy-engine transfer rather than one per row.
    """

    requested_backend = "dma"

    def __init__(self) -> None:
        self.actual_backend: str | None = None

    def copy_rows(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_rows: Sequence[int],
        destination_slots: Sequence[int],
    ) -> str:
        """Copy selected rows on the current CUDA stream and return the path used."""
        self.actual_backend = ExpertDMARowRoute(source, destination).copy_rows(
            source_rows, destination_slots
        )
        return self.actual_backend

    @staticmethod
    def _validate_tensors(
        source: torch.Tensor, destination: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.device.type != "cpu":
            raise ValueError("expert DMA source must be a CPU tensor")
        if not is_gpu_readable_host_tensor(source):
            raise ValueError("expert DMA source must be pinned or CUDA-registered")
        if destination.device.type != "cuda":
            raise ValueError("expert DMA destination must be a CUDA tensor")
        if source.dtype != destination.dtype:
            raise ValueError("expert DMA source and destination dtypes must match")
        if source.ndim < 2 or destination.ndim < 2:
            raise ValueError("expert DMA tensors must include rows and row values")
        if not source.is_contiguous() or not destination.is_contiguous():
            raise ValueError("expert DMA tensors must be contiguous")
        source_matrix = source.view(source.shape[0], -1)
        destination_matrix = destination.view(destination.shape[0], -1)
        if source_matrix.shape[1] != destination_matrix.shape[1]:
            raise ValueError("expert DMA source and destination row widths must match")
        return source_matrix, destination_matrix

    @staticmethod
    def _validate_indices(rows: Sequence[int], limit: int, name: str) -> list[int]:
        indices = [index(row) for row in rows]
        if any(row < 0 or row >= limit for row in indices):
            raise ValueError(f"expert DMA {name} are outside tensor rows")
        return indices


class ExpertDMARowRoute:
    """One validated source and destination pair for repeated DMA row copies.

    Tensor validation, the AOT-availability check and the registration bound
    are resolved once, so a caller that copies the same pair every update
    pays only index checks, range coalescing and the copy-engine call.
    """

    def __init__(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        self.source_matrix, self.destination_matrix = (
            ExpertDMABackend._validate_tensors(source, destination)
        )
        self.aot_available = _aot_transfer_available()
        self.run_end = (
            _registration_run_end(self.source_matrix) if self.aot_available else None
        )

    def copy_rows(
        self,
        source_rows: Sequence[int],
        destination_slots: Sequence[int],
        coalesced: dict[object, tuple[list[int], list[int], list[int]]] | None = None,
    ) -> str:
        """Copy selected rows on the current CUDA stream and return the path used.

        ``coalesced`` shares merged ranges between routes whose registration
        bound is the same, for callers copying one row list into several pairs.
        """
        source_indices = ExpertDMABackend._validate_indices(
            source_rows, self.source_matrix.shape[0], "source_rows"
        )
        destination_indices = ExpertDMABackend._validate_indices(
            destination_slots, self.destination_matrix.shape[0], "destination_slots"
        )
        if len(source_indices) != len(destination_indices):
            raise ValueError(
                "source_rows and destination_slots must have the same number"
            )
        if not source_indices:
            raise ValueError("expert DMA transfer requires at least one row")
        if not self.aot_available:
            for source_row, destination_slot in zip(
                source_indices, destination_indices
            ):
                self.destination_matrix[destination_slot : destination_slot + 1].copy_(
                    self.source_matrix[source_row : source_row + 1], non_blocking=True
                )
            return "fallback"
        key = (self.run_end, id(source_rows), id(destination_slots))
        ranges = None if coalesced is None or self.run_end is not None else coalesced.get(key)
        if ranges is None:
            ranges = _coalesce_ranges(source_indices, destination_indices, self.run_end)
            if coalesced is not None and self.run_end is None:
                coalesced[key] = ranges
        transfer_embedding_ranges_direct(
            self.source_matrix, self.destination_matrix, *ranges
        )
        return "dma"


def _coalesce_ranges(
    source_rows: Sequence[int],
    destination_slots: Sequence[int],
    run_end: Callable[[int], int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Merge row pairs that advance together into start and length lists.

    ``run_end(row)`` is the first source row a run starting at ``row`` may not
    reach; without it runs are unbounded.
    """
    source_starts: list[int] = []
    destination_starts: list[int] = []
    lengths: list[int] = []
    limit = 0
    for source_row, destination_slot in zip(source_rows, destination_slots):
        if (
            lengths
            and source_row == source_starts[-1] + lengths[-1]
            and destination_slot == destination_starts[-1] + lengths[-1]
            and source_row < limit
        ):
            lengths[-1] += 1
        else:
            source_starts.append(source_row)
            destination_starts.append(destination_slot)
            lengths.append(1)
            limit = run_end(source_row) if run_end is not None else sys.maxsize
    return source_starts, destination_starts, lengths


def _registration_run_end(matrix: torch.Tensor) -> Callable[[int], int] | None:
    """Bound runs of a registered host matrix to the registration of their first row.

    The arena registers large tensors in row-aligned chunks, and one copy-engine
    range spanning two registrations is not guaranteed to be read in place.
    PyTorch pinned allocations are one block and need no bound.
    """
    row_bytes = matrix.shape[1] * matrix.element_size()
    if matrix.is_pinned() or row_bytes == 0:
        return None
    base = matrix.data_ptr()

    def run_end(row: int) -> int:
        end = cuda_host_registration_end(base + row * row_bytes)
        return row + 1 if end is None else (end - base) // row_bytes

    return run_end


def _aot_transfer_available() -> bool:
    return transfer_embedding_ranges_direct is not None and hasattr(
        torch.ops.sgl_kernel, "transfer_embedding_ranges_direct"
    )
