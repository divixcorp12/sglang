"""Geometry and on-disk manifest for splitting an EXL3 expert row across NVMe drives.

A row is split into K page-aligned fragments, one per drive, so each fragment
can be read with its own O_DIRECT request in parallel. This module is the pure
math: it has no file I/O and does not import torch, because it is imported
during layout construction alongside code that must stay usable before torch
is available.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Sequence

# The moe package already defines PAGE_BYTES = 4096 in expert_host_arena.py and
# expert_host_tier.py, but both pull in torch at import time. This module must
# not import torch (see module docstring), so the constant is repeated here
# rather than imported.
PAGE_BYTES = 4096


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


@dataclass(frozen=True)
class StripeGeometry:
    """Splits one row's bytes into K fragments, one per drive.

    Every fragment but the last is `align4096(ceil(row_bytes * w_i / sum(w)))`;
    the last absorbs the rounding remainder so the fragments sum to exactly
    `row_bytes`. This keeps the K-1 leading fragments O_DIRECT-aligned while
    still landing on the row's true byte count.
    """

    row_bytes: int
    weights: tuple[float, ...]
    fragment_bytes: tuple[int, ...] = field(init=False)
    starts: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        weights = tuple(self.weights)
        if not weights or any(w <= 0 for w in weights):
            raise ValueError(f"weights must be non-empty and strictly positive, got {weights}")
        object.__setattr__(self, "weights", weights)

        total = sum(weights)
        fragments = []
        for w in weights[:-1]:
            fragment = _align_up(math.ceil(self.row_bytes * w / total), PAGE_BYTES)
            if fragment <= 0:
                raise ValueError(f"weight {w} produced a non-positive fragment")
            fragments.append(fragment)
        last = self.row_bytes - sum(fragments)
        if last <= 0:
            raise ValueError(
                f"too many stripes for row_bytes={self.row_bytes}: leading fragments "
                f"{fragments} leave {last} bytes for the last stripe"
            )
        fragments.append(last)

        object.__setattr__(self, "fragment_bytes", tuple(fragments))
        object.__setattr__(
            self, "starts", tuple(itertools.accumulate(fragments[:-1], initial=0))
        )

    def slot_offset(self, stripe: int, expert: int) -> int:
        """Byte offset of `expert`'s slot inside stripe `stripe`'s per-drive file."""
        return expert * self.fragment_bytes[stripe]


@dataclass(frozen=True)
class StripeInfo:
    index: int
    weight: float
    fragment_bytes: int
    dir_hint: str


@dataclass(frozen=True)
class StripeManifest:
    """One stripe directory's `manifest.json`.

    `index` names which stripe directory this manifest describes; every other
    field must be identical across the K manifests of one striped repack
    (`validate_set` enforces this).
    """

    version: int
    source: str
    num_layers: int
    num_experts: int
    row_bytes: int
    tensor_order: tuple[str, ...]
    stripes: tuple[StripeInfo, ...]
    row_sha256_sample: dict[str, str]
    index: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "source": self.source,
                "num_layers": self.num_layers,
                "num_experts": self.num_experts,
                "row_bytes": self.row_bytes,
                "tensor_order": list(self.tensor_order),
                "stripes": [
                    {
                        "index": s.index,
                        "weight": s.weight,
                        "fragment_bytes": s.fragment_bytes,
                        "dir_hint": s.dir_hint,
                    }
                    for s in self.stripes
                ],
                "row_sha256_sample": self.row_sha256_sample,
                "index": self.index,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "StripeManifest":
        data = json.loads(text)
        return cls(
            version=data["version"],
            source=data["source"],
            num_layers=data["num_layers"],
            num_experts=data["num_experts"],
            row_bytes=data["row_bytes"],
            tensor_order=tuple(data["tensor_order"]),
            stripes=tuple(
                StripeInfo(s["index"], s["weight"], s["fragment_bytes"], s["dir_hint"])
                for s in data["stripes"]
            ),
            row_sha256_sample=data["row_sha256_sample"],
            index=data["index"],
        )


def validate_set(manifests: Sequence[StripeManifest]) -> None:
    """Raise unless `manifests` is one complete, mutually agreeing stripe set.

    Every manifest must agree on everything except `index`; the indices present
    must be exactly `0..K-1` with no duplicates and no fewer than the set's own
    `stripes` list declares.
    """
    if not manifests:
        raise ValueError("no manifests to validate")

    reference = manifests[0]
    for manifest in manifests[1:]:
        for name in (
            "version",
            "source",
            "num_layers",
            "num_experts",
            "row_bytes",
            "tensor_order",
            "stripes",
            "row_sha256_sample",
        ):
            if getattr(manifest, name) != getattr(reference, name):
                raise ValueError(
                    f"manifest index={manifest.index} disagrees with index={reference.index} "
                    f"on {name}: {getattr(manifest, name)!r} != {getattr(reference, name)!r}"
                )

    expected = len(reference.stripes)
    if len(manifests) != expected:
        raise ValueError(f"expected {expected} manifests (per stripes list), got {len(manifests)}")

    indices = [m.index for m in manifests]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate manifest index in {sorted(indices)}")
    if sorted(indices) != list(range(expected)):
        raise ValueError(f"manifest indices must be exactly 0..{expected - 1}, got {sorted(indices)}")
