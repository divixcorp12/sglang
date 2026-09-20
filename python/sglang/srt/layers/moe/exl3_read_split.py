"""Splitting an EXL3 expert row's bytes into K parts, one per drive.

A row is split into K fragments so each fragment can be read with its own
O_DIRECT request in parallel, one per drive. This module is the pure math: it
has no file I/O and does not import torch, because it is imported during
layout construction alongside code that must stay usable before torch is
available.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

# The moe package already defines PAGE_BYTES = 4096 in expert_host_arena.py and
# expert_host_tier.py, but both pull in torch at import time. This module must
# not import torch (see module docstring), so the constant is repeated here
# rather than imported.
PAGE_BYTES = 4096


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


@dataclass(frozen=True)
class ReadSplit:
    """Splits one row's bytes into K parts, one per drive.

    Every part but the last is `align4096(ceil(row_bytes * w_i / sum(w)))`;
    the last absorbs the rounding remainder so the parts sum to exactly
    `row_bytes`. This keeps the K-1 leading parts' size a multiple of 4096,
    but the last part generally is not (it is whatever bytes are left over).
    """

    row_bytes: int
    weights: tuple[float, ...]
    part_bytes: tuple[int, ...] = field(init=False)
    starts: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        weights = tuple(self.weights)
        if not weights or any(w <= 0 for w in weights):
            raise ValueError(f"weights must be non-empty and strictly positive, got {weights}")
        object.__setattr__(self, "weights", weights)

        total = sum(weights)
        parts = []
        for w in weights[:-1]:
            part = _align_up(math.ceil(self.row_bytes * w / total), PAGE_BYTES)
            if part <= 0:
                raise ValueError(f"weight {w} produced a non-positive part")
            parts.append(part)
        last = self.row_bytes - sum(parts)
        if last <= 0:
            raise ValueError(
                f"too many parts for row_bytes={self.row_bytes}: leading parts "
                f"{parts} leave {last} bytes for the last part"
            )
        parts.append(last)
        parts = tuple(parts)

        object.__setattr__(self, "part_bytes", parts)
        object.__setattr__(
            self, "starts", tuple(itertools.accumulate(parts[:-1], initial=0))
        )
