"""Padded layout of an EXL3 expert inside a cache slot.

A raw EXL3 row packs its 12 tensors with no padding, which leaves the trellis
misaligned for exllamav3's 128-bit loads. A slot re-places each tensor at an
aligned offset; the segment list is the copy plan from raw row to slot.

The default alignment is 16 bytes, not 128: exllamav3's fused MoE kernel loads
the trellis (`exl3_moe_kernel`/`exl3_gemm_kernel_inner`) through `cp.async.cg`
on `int4*`-cast pointers, which is a hard 16-byte requirement
(`exllamav3_ext/quant/exl3_gemm_inner.cuh:253-265`, `ptx.cuh:159-168`), and
loads `suh`/`svh` through an 8-byte-aligned `half4` cast
(`exllamav3_ext/util.cuh:8`, `hadamard_inner.cuh:106-113`). Nothing in
exllamav3 needs wider than 16 bytes; see Phase 0 §14.1 in DSV41_REFERENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sglang.srt.layers.moe.exl3_expert_layout import Exl3TensorSpan


@dataclass(frozen=True)
class Exl3Segment:
    name: str
    src_offset: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class Exl3SlotLayout:
    alignment: int
    slot_bytes: int
    segments: tuple[Exl3Segment, ...]

    def dst_offset(self, name: str) -> int:
        for segment in self.segments:
            if segment.name == name:
                return segment.dst_offset
        raise KeyError(name)


def _round_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def build_exl3_slot_layout(
    tensors: Sequence[Exl3TensorSpan], alignment: int = 16
) -> Exl3SlotLayout:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a power of two, got {alignment}")
    cursor, segments = 0, []
    for tensor in tensors:
        cursor = _round_up(cursor, alignment)
        segments.append(Exl3Segment(tensor.name, tensor.rel_offset, cursor, tensor.nbytes))
        cursor += tensor.nbytes
    return Exl3SlotLayout(alignment, _round_up(cursor, alignment), tuple(segments))
