"""CUDA copy-engine backend for uncaptured NVFP4 expert-row transfers."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index

import torch

try:
    from sgl_kernel.kvcacheio import transfer_embedding_ranges_direct
except ImportError:
    transfer_embedding_ranges_direct = None


class ExpertDMABackend:
    """Copy selected pinned host expert rows into fixed CUDA cache slots."""

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
        source_matrix, destination_matrix = self._validate_tensors(source, destination)
        source_indices = self._validate_indices(
            source_rows, source_matrix.shape[0], "source_rows"
        )
        destination_indices = self._validate_indices(
            destination_slots, destination_matrix.shape[0], "destination_slots"
        )
        if len(source_indices) != len(destination_indices):
            raise ValueError(
                "source_rows and destination_slots must have the same number"
            )
        if not source_indices:
            raise ValueError("expert DMA transfer requires at least one row")

        if _aot_transfer_available():
            transfer_embedding_ranges_direct(
                source_matrix,
                destination_matrix,
                source_indices,
                destination_indices,
                [1] * len(source_indices),
            )
            self.actual_backend = "dma"
        else:
            for source_row, destination_slot in zip(
                source_indices, destination_indices
            ):
                destination_matrix[destination_slot : destination_slot + 1].copy_(
                    source_matrix[source_row : source_row + 1], non_blocking=True
                )
            self.actual_backend = "fallback"
        return self.actual_backend

    @staticmethod
    def _validate_tensors(
        source: torch.Tensor, destination: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.device.type != "cpu":
            raise ValueError("expert DMA source must be a CPU tensor")
        if not source.is_pinned():
            raise ValueError("expert DMA source must be pinned")
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


def _aot_transfer_available() -> bool:
    return transfer_embedding_ranges_direct is not None and hasattr(
        torch.ops.sgl_kernel, "transfer_embedding_ranges_direct"
    )
