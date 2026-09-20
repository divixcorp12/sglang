"""Splitting an already page-aligned read length across mirror roots.

`Exl3ExpertRecord.aligned_read` already rounds a row's read up to a
page-aligned `(offset, length, row_start)` before any of this runs, so by the
time a length reaches `ReadSplit` it is a multiple of 4096. That is the
difference from the retired striping design: there, a *row's own byte count*
was the thing being split, which is essentially never a multiple of 4096
itself, so the split needed a padded "stride" distinct from the real payload
("fragment") size, and reconstructing a row meant stripping that per-part
padding back out. Here `length` is already a whole number of pages, so a
split of it into per-part page counts has no remainder fragment and no
padded stride to strip -- every part, including the last, is directly a
number of whole O_DIRECT-aligned pages, and they sum to `length` exactly.
That hazard is what mirroring removes, not something this module works
around.

`ReadSplit` is deliberately about *bytes for one read*, not row geometry: for
mirroring, every root holds a byte-identical copy of the checkpoint, so
"which root serves which bytes of this one read" is the only thing left to
plan, and it can change per read (a `SplitPolicy` may adapt to per-root
throughput; Task 7). A part of 0 is a legal outcome, not a degenerate one --
it is how a slow or unhealthy root is dropped from a read without dropping
the root from the configuration.

This module has no file I/O and does not import torch, because it is
imported during layout construction alongside code that must stay usable
before torch is available.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

# The moe package already defines PAGE_BYTES = 4096 in expert_host_arena.py and
# expert_host_tier.py, but both pull in torch at import time. This module must
# not import torch (see module docstring), so the constant is repeated here
# rather than imported.
PAGE_BYTES = 4096


@dataclass(frozen=True)
class ReadSplit:
    """Splits an already page-aligned `length` into `len(weights)` parts.

    Every part, including the last, is a multiple of `PAGE_BYTES`, and the
    parts sum exactly to `length` -- there is no remainder to absorb because
    `length` itself is already whole pages. Each part is rounded down to a
    page count proportional to its weight; the leftover pages (from the
    rounding-down) all go to the single largest-weight root, so the split
    stays deterministic and the sum stays exact.

    A weight of 0 is legal and produces a part of 0 bytes for that root --
    this is how a root is dropped from one read.
    """

    length: int
    weights: tuple[float, ...]
    part_bytes: tuple[int, ...] = field(init=False)
    starts: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError(f"length must be non-negative, got {self.length}")
        if self.length % PAGE_BYTES != 0:
            raise ValueError(f"length must be a multiple of {PAGE_BYTES}, got {self.length}")

        weights = tuple(self.weights)
        if not weights:
            raise ValueError("weights must be non-empty")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights}")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError(f"weights must not be all zero, got {weights}")
        object.__setattr__(self, "weights", weights)

        total_pages = self.length // PAGE_BYTES
        pages = [int(total_pages * w / total_weight) for w in weights]
        remainder = total_pages - sum(pages)
        if remainder:
            largest = weights.index(max(weights))
            pages[largest] += remainder

        part_bytes = tuple(p * PAGE_BYTES for p in pages)
        object.__setattr__(self, "part_bytes", part_bytes)
        object.__setattr__(
            self, "starts", tuple(itertools.accumulate(part_bytes[:-1], initial=0))
        )


class SplitPolicy:
    """How to divide one read's bytes across mirror roots.

    A policy may be static (fixed weights) or adaptive (Task 7: weights
    derived from observed per-root throughput). Either way, `plan` is the
    only thing a caller needs: it never sees the weights directly.
    """

    def plan(self, length: int) -> ReadSplit:
        raise NotImplementedError


class StaticSplitPolicy(SplitPolicy):
    """Fixed weights, unaffected by observed drive behavior."""

    def __init__(self, weights: Sequence[float]) -> None:
        self._weights = tuple(weights)

    def plan(self, length: int) -> ReadSplit:
        return ReadSplit(length=length, weights=self._weights)
