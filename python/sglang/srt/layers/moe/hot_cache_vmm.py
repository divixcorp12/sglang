"""Expert hot cache tensors on CUDA VMM memory, so trailing slot rows can be released after allocation
(SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP).

Each tensor gets its own virtual-address reservation, backed one granule (the device's recommended VMM granularity,
2 MiB on the RTX 5090) per physical handle. ``release_tail`` unmaps and frees the granules that lie wholly past a kept
row count. Every address below the kept rows is unchanged, so the tables that recorded them at startup (the RAM-miss
copy table, graph-gather segment tables, slot pointer tables) stay valid, and a stray access to a released row faults
instead of reading or writing another tensor's memory.

The memory sits outside torch's caching allocator (as torch.empty would not): torch.cuda.memory_reserved() does not
count it, while cudaMemGetInfo, and so KV pool sizing, does. It uses its own reservations, separate from the ones
torch makes under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TailRelease:
    """Granules [first, end) of a tensor's reservation, wholly past its kept rows, and their byte span."""

    first: int
    end: int
    granularity: int

    @property
    def nbytes(self) -> int:
        return (self.end - self.first) * self.granularity


def granule_count(nbytes: int, granularity: int) -> int:
    return -(-int(nbytes) // granularity)


def tail_release(*, row_bytes: int, rows: int, keep_rows: int, granularity: int) -> TailRelease:
    """The granules a tensor of ``rows`` rows can release when only its first ``keep_rows`` rows stay: those starting
    at or after the first granule boundary past the kept bytes, through the last granule the tensor maps."""
    if not 0 <= keep_rows <= rows:
        raise ValueError(f"keep_rows {keep_rows} outside [0, {rows}]")
    end = granule_count(rows * row_bytes, granularity)
    first = min(end, granule_count(keep_rows * row_bytes, granularity))
    return TailRelease(first, end, granularity)


# A VMM tensor rounds up to whole granules, so it can waste up to one. Tensors smaller than this many granules (the
# per-row scale vectors, ~0.5 MiB a layer) stay on torch.empty: rounding them cost ~140 MiB over 40 layers in the
# step-3 smoke, and their retired rows are too small to be worth releasing. At 32 granules the rounding is <= 3%.
MIN_VMM_GRANULES = 32


class HotCacheMapOutOfMemory(RuntimeError):
    """The device had no memory for a granule a grow needed; the grow was rolled back."""


class VmmHotCacheBacking:
    """One device's hot cache tensors: the large ones each on their own VMM reservation, mapped one handle per
    granule; the small ones (below MIN_VMM_GRANULES) on torch.empty."""

    def __init__(
        self,
        device: torch.device,
        specs: Sequence[Tuple[str, int, Tuple[int, ...], torch.dtype]],
        backed_rows: Optional[int] = None,
    ):
        """``specs``: (name, rows, row_shape, dtype) per tensor; ``rows`` is the reservation (the upper bound).
        Only the first ``backed_rows`` rows (all when None) get physical memory now; ``set_backed_rows`` moves that
        boundary either way later, while every address stays where it is."""
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            get_device_granularity,
            make_device_allocation_prop,
            tensor_from_pointer,
        )

        self.device_id = device.index if device.index is not None else torch.cuda.current_device()
        self.granularity = get_device_granularity(self.device_id)
        prop = make_device_allocation_prop(self.device_id, handle_types=None)
        self.tensors: Dict[str, torch.Tensor] = {}
        self.row_bytes: Dict[str, int] = {}
        self.rows: Dict[str, int] = {}
        self._reservations = {}
        self._handles: Dict[str, List] = {}
        self.backed_rows = None
        with torch.cuda.device(self.device_id):
            for name, rows, row_shape, dtype in specs:
                row_bytes = prod(row_shape) * torch.empty((), dtype=dtype).element_size()
                granules = granule_count(rows * row_bytes, self.granularity)
                if granules < MIN_VMM_GRANULES:
                    self.tensors[name] = torch.empty((rows,) + tuple(row_shape), dtype=dtype, device=device)
                    continue
                reservation = VmmReservation(
                    granules * self.granularity, prop, self.device_id, alignment=self.granularity
                )
                self._reservations[name] = reservation
                self._handles[name] = [None] * granules
                self.tensors[name] = tensor_from_pointer(
                    reservation.base, rows * row_bytes, shape=(rows,) + tuple(row_shape), dtype=dtype,
                    device_id=self.device_id,
                )
                self.row_bytes[name], self.rows[name] = row_bytes, rows
        self.upper_rows = max(self.rows.values(), default=0)
        self.set_backed_rows(self.upper_rows if backed_rows is None else backed_rows)

    @property
    def mapped_bytes(self) -> int:
        return sum(h is not None for handles in self._handles.values() for h in handles) * self.granularity

    def _needed_granules(self, name: str, rows: int) -> int:
        """Granules [0, n) of tensor ``name`` hold a byte of its first ``rows`` rows."""
        return tail_release(
            row_bytes=self.row_bytes[name], rows=self.rows[name], keep_rows=rows, granularity=self.granularity
        ).first

    def bytes_to_back(self, rows: int) -> int:
        """Device memory ``set_backed_rows(rows)`` would map on top of what is mapped now: whole granules, so a slot
        costs its rows rounded up per tensor, not ``bytes_per_expert``. 0 when shrinking."""
        if not 0 <= rows <= self.upper_rows:
            raise ValueError(f"backed rows {rows} outside [0, {self.upper_rows}]")
        missing = 0
        for name, handles in self._handles.items():
            missing += sum(h is None for h in handles[: self._needed_granules(name, rows)])
        return missing * self.granularity

    def _unmap(self, name: str, g: int, drv) -> None:
        from sglang.srt.utils.cuda_vmm_utils import check_drv

        reservation, handles = self._reservations[name], self._handles[name]
        address = reservation.base + g * self.granularity
        check_drv(drv.cuMemUnmap(address, self.granularity), "cuMemUnmap(hot tail)")
        check_drv(drv.cuMemRelease(handles[g]), "cuMemRelease(hot tail)")
        handles[g] = None
        # The reservation's teardown must skip what is already unmapped and released.
        reservation._mappings = [m for m in reservation._mappings if m[0] != address]

    def set_backed_rows(self, rows: int) -> int:
        """Back exactly the granules that hold any of the first ``rows`` rows of every VMM tensor: map the missing
        ones (growing) or unmap and free those wholly past them (shrinking). Returns the bytes this added (negative
        when it freed). Addresses never move; rows past the boundary fault on access.

        A grow that fails part-way unmaps every granule it mapped first, so the backing is exactly as before the
        call; a device out-of-memory is raised as ``HotCacheMapOutOfMemory`` for the caller to stop growing on."""
        from sglang.srt.utils.cuda_vmm_utils import _get_cuda_driver

        if not 0 <= rows <= self.upper_rows:
            raise ValueError(f"backed rows {rows} outside [0, {self.upper_rows}]")
        drv = _get_cuda_driver() if self._reservations else None
        if self.backed_rows is not None and rows < self.backed_rows:
            torch.cuda.synchronize(self.device_id)
        delta = 0
        mapped_now = []
        with torch.cuda.device(self.device_id):
            for name, reservation in self._reservations.items():
                need = self._needed_granules(name, rows)  # granules [0, need) hold a kept byte
                handles = self._handles[name]
                for g in range(len(handles)):
                    if g < need and handles[g] is None:
                        try:
                            handles[g] = reservation.map(g * self.granularity, self.granularity, retain_handle=True)
                        except RuntimeError as error:
                            for undo_name, undo_g in reversed(mapped_now):
                                self._unmap(undo_name, undo_g, drv)
                            if "OUT_OF_MEMORY" in str(error):
                                raise HotCacheMapOutOfMemory(
                                    f"hot cache VMM: out of device memory backing rows {self.backed_rows} -> {rows} "
                                    f"({len(mapped_now)} granules mapped and rolled back)"
                                ) from error
                            raise
                        mapped_now.append((name, g))
                        delta += self.granularity
                    elif g >= need and handles[g] is not None:
                        self._unmap(name, g, drv)
                        delta -= self.granularity
        if self.backed_rows is not None and rows != self.backed_rows:
            logger.info(
                "hot cache VMM: backed rows %d -> %d of %d (%+.1f MiB)",
                self.backed_rows, rows, self.upper_rows, delta / 2**20,
            )
        self.backed_rows = rows
        return delta

    def release_tail(self, keep_rows: int) -> int:
        """Shrink the backing to ``keep_rows`` rows; return the bytes freed."""
        return -self.set_backed_rows(keep_rows)

    def close(self) -> None:
        torch.cuda.synchronize(self.device_id)
        for reservation in self._reservations.values():
            reservation.close(release_handles=True)
        self._reservations.clear()
