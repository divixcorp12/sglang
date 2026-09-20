"""Splitting an already page-aligned read length across mirror roots.

For mirroring, every root holds a byte-identical copy of the checkpoint, so
the only thing to plan for one read is which root serves which of its bytes.
`ReadSplit` divides an already page-aligned `length` (the aligned length
`Exl3ExpertRecord.aligned_read` produces) into `len(weights)` parts, one per
root: every part, including the last, is a multiple of `PAGE_BYTES`, and the
parts sum exactly to `length`. A part of 0 is a legal outcome, not a
degenerate one -- it is how a slow or unhealthy root is dropped from a read
without dropping the root from the configuration.

`SplitPolicy` is the interface a reader plans against; `StaticSplitPolicy`
is the fixed-weights implementation.

This module has no file I/O and does not import torch, because it is
imported during layout construction alongside code that must stay usable
before torch is available.
"""

from __future__ import annotations

import itertools
import math
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
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise ValueError(f"length must be an int, got {type(self.length).__name__}")
        if self.length < 0:
            raise ValueError(f"length must be non-negative, got {self.length}")
        if self.length % PAGE_BYTES != 0:
            raise ValueError(
                f"length must be a multiple of {PAGE_BYTES}, got {self.length}"
            )

        weights = tuple(self.weights)
        if not weights:
            raise ValueError("weights must be non-empty")
        for w in weights:
            if (
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not math.isfinite(w)
            ):
                raise ValueError(f"weights must be finite numbers, got {weights}")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights}")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError(f"weights must not be all zero, got {weights}")
        object.__setattr__(self, "weights", weights)

        # Floor division (rather than true division followed by `int()`)
        # keeps the apportionment integer by construction: the sum of the
        # floors can never exceed `total_pages`, so `remainder` below can
        # never go negative -- the asserts turn that guarantee into a loud
        # failure if a future edit breaks it, instead of a silently
        # misaligned O_DIRECT read.
        total_pages = self.length // PAGE_BYTES
        pages = [int(total_pages * w // total_weight) for w in weights]
        remainder = total_pages - sum(pages)
        assert remainder >= 0, f"apportionment overshot total_pages: pages={pages}"
        if remainder:
            largest = weights.index(max(weights))
            pages[largest] += remainder

        part_bytes = tuple(p * PAGE_BYTES for p in pages)
        assert all(p >= 0 for p in part_bytes), f"negative part in {part_bytes}"
        assert sum(part_bytes) == self.length, (
            f"parts {part_bytes} do not sum to {self.length}"
        )
        object.__setattr__(self, "part_bytes", part_bytes)
        object.__setattr__(
            self, "starts", tuple(itertools.accumulate(part_bytes[:-1], initial=0))
        )


class SplitPolicy:
    """How to divide one read's bytes across mirror roots.

    A policy may be static (fixed weights) or adaptive (weights derived from
    observed per-root throughput). Either way, `plan` is the only thing a
    caller needs: it never sees the weights directly.
    """

    def plan(self, length: int) -> ReadSplit:
        raise NotImplementedError


class StaticSplitPolicy(SplitPolicy):
    """Fixed weights, unaffected by observed drive behavior."""

    def __init__(self, weights: Sequence[float]) -> None:
        self._weights = tuple(weights)

    def plan(self, length: int) -> ReadSplit:
        return ReadSplit(length=length, weights=self._weights)
