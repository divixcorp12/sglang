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
from typing import Dict, List, Sequence, Tuple

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


class VmmHotCacheBacking:
    """One device's hot cache tensors, each on its own VMM reservation, mapped one handle per granule."""

    def __init__(self, device: torch.device, specs: Sequence[Tuple[str, int, Tuple[int, ...], torch.dtype]]):
        """``specs``: (name, rows, row_shape, dtype) per tensor."""
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
        self._released: Dict[str, TailRelease] = {}
        with torch.cuda.device(self.device_id):
            for name, rows, row_shape, dtype in specs:
                row_bytes = prod(row_shape) * torch.empty((), dtype=dtype).element_size()
                granules = granule_count(rows * row_bytes, self.granularity)
                reservation = VmmReservation(
                    granules * self.granularity, prop, self.device_id, alignment=self.granularity
                )
                self._reservations[name] = reservation
                self._handles[name] = [
                    reservation.map(g * self.granularity, self.granularity, retain_handle=True)
                    for g in range(granules)
                ]
                self.tensors[name] = tensor_from_pointer(
                    reservation.base, rows * row_bytes, shape=(rows,) + tuple(row_shape), dtype=dtype,
                    device_id=self.device_id,
                )
                self.row_bytes[name], self.rows[name] = row_bytes, rows

    @property
    def mapped_bytes(self) -> int:
        return sum(h is not None for handles in self._handles.values() for h in handles) * self.granularity

    def release_tail(self, keep_rows: int) -> int:
        """Unmap and free every tensor's granules wholly past its first ``keep_rows`` rows; return the bytes freed.
        Once per backing: the kept tensors keep their addresses, the released rows fault on access."""
        if self._released:
            raise RuntimeError("hot cache VMM tail already released")
        from sglang.srt.utils.cuda_vmm_utils import _get_cuda_driver, check_drv

        drv = _get_cuda_driver()
        torch.cuda.synchronize(self.device_id)
        freed = 0
        for name, reservation in self._reservations.items():
            tail = tail_release(
                row_bytes=self.row_bytes[name], rows=self.rows[name], keep_rows=keep_rows,
                granularity=self.granularity,
            )
            released = set()
            for g in range(tail.first, tail.end):
                address = reservation.base + g * self.granularity
                check_drv(drv.cuMemUnmap(address, self.granularity), "cuMemUnmap(hot tail)")
                check_drv(drv.cuMemRelease(self._handles[name][g]), "cuMemRelease(hot tail)")
                self._handles[name][g] = None
                released.add(address)
            # The reservation's teardown must skip what is already unmapped and released.
            reservation._mappings = [m for m in reservation._mappings if m[0] not in released]
            self._released[name] = tail
            freed += tail.nbytes
            if tail.end > tail.first:
                logger.info(
                    "hot cache VMM %s: released rows %d.. (granules %d-%d of %d, %.1f MiB) at 0x%x",
                    name, keep_rows, tail.first, tail.end - 1, tail.end, tail.nbytes / 2**20,
                    reservation.base + tail.first * self.granularity,
                )
        return freed

    def close(self) -> None:
        torch.cuda.synchronize(self.device_id)
        for reservation in self._reservations.values():
            reservation.close(release_handles=True)
        self._reservations.clear()
