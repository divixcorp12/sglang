"""The EXL3 routed-expert format plugin of the expert streaming framework.

An EXL3 expert row on disk is 12 tensors back to back (w1/w2/w3 x
suh/svh/mul1/trellis). The framework streams six per-name tensors instead:
w1 and w3 stack into the ``w13_*`` rows as parts 0 and 1, w2 is part 0 of the
``w2_*`` rows, and the three ``mul1`` scalars are dropped (the codebook is
always mul1, so ``Exl3Tensors`` takes ``mul1=True``). ``segment_map`` says
where every streamed byte sits in the on-disk row, so a row source can split
one superset read into the six per-name rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.expert_format import ExpertTensorSpec

EXL3_STREAMED_NAMES = (
    "w13_trellis",
    "w13_suh",
    "w13_svh",
    "w2_trellis",
    "w2_suh",
    "w2_svh",
)
# Rows one eager gather stages: 64 x 13.3 MB = 852 MB of VRAM staging.
EXL3_MAX_GATHER_ROWS = 64
_DTYPES = {"I16": torch.int16, "F16": torch.float16}
# On-disk linear -> (streamed prefix, part of the row's leading dimension).
_PARTS = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}
_KINDS = ("trellis", "suh", "svh")


@dataclass(frozen=True)
class RowSegment:
    """``nbytes`` at ``src_offset`` of the on-disk row fill part ``part`` of
    streamed tensor ``name``'s row, at byte ``dst_offset`` of that row."""

    name: str
    part: int
    dst_offset: int
    src_offset: int
    nbytes: int


def _row_schema(
    layout: Exl3ExpertLayout,
) -> tuple[tuple[ExpertTensorSpec, ...], tuple[RowSegment, ...]]:
    spans = {span.name: span for span in layout.tensors}
    expected = {f"{w}.{kind}" for w in _PARTS for kind in _KINDS + ("mul1",)}
    if set(spans) != expected:
        raise ValueError(
            f"exl3 expert rows hold {sorted(spans)}, expected {sorted(expected)}"
        )
    for w in _PARTS:
        mul1 = spans[f"{w}.mul1"]
        if mul1.dtype != "I32" or tuple(mul1.shape) != ():
            raise ValueError(f"exl3 {mul1.name} is {mul1.dtype} {mul1.shape}, expected I32 []")
    specs: dict[str, ExpertTensorSpec] = {}
    segments = []
    for w, (prefix, part) in _PARTS.items():
        for kind in _KINDS:
            span = spans[f"{w}.{kind}"]
            dtype = _DTYPES.get(span.dtype)
            shape = tuple(span.shape)
            if dtype is None or math.prod(shape) * dtype.itemsize != span.nbytes:
                raise ValueError(
                    f"exl3 expert tensor {span.name}: unexpected {span.dtype} "
                    f"{shape} ({span.nbytes} bytes)"
                )
            name = f"{prefix}_{kind}"
            parts = 2 if prefix == "w13" else 1
            spec = ExpertTensorSpec(name, (parts,) + shape, dtype, "host")
            previous = specs.setdefault(name, spec)
            if previous != spec:
                raise ValueError(
                    f"exl3 {name}: w1 and w3 disagree "
                    f"({previous.row_shape} vs {spec.row_shape})"
                )
            segments.append(
                RowSegment(name, part, part * span.nbytes, span.rel_offset, span.nbytes)
            )
    return (
        tuple(specs[name] for name in EXL3_STREAMED_NAMES),
        tuple(sorted(segments, key=lambda segment: segment.src_offset)),
    )


class Exl3ExpertFormat:
    """``ExpertFormat`` for one layer of an EXL3 checkpoint's routed experts.

    Spec-only: there is no dense ``[experts, ...]`` source, so every host row
    comes from the row source. ``direct`` picks O_DIRECT shard reads; None
    takes it from ``SGLANG_MOE_EXPERT_FILE_READER``.
    """

    key = "exl3"
    supports_graph_gather = False
    supports_host_arena = False
    max_gather_rows: Optional[int] = EXL3_MAX_GATHER_ROWS
    names = EXL3_STREAMED_NAMES

    def __init__(
        self, layout: Exl3ExpertLayout, layer_id: int, *, direct: Optional[bool] = None
    ) -> None:
        if not 0 <= layer_id < layout.num_layers:
            raise ValueError(
                f"exl3 layer {layer_id} is outside the checkpoint's "
                f"{layout.num_layers} layers"
            )
        self.layout = layout
        self.layer_id = layer_id
        self.direct = direct
        self._specs, self._segments = _row_schema(layout)

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        return self._specs

    def num_experts(self, layer: torch.nn.Module) -> int:
        return self.layout.num_experts

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return None

    def segment_map(self) -> tuple[RowSegment, ...]:
        return self._segments
