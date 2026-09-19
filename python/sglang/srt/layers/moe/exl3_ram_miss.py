"""Option C for EXL3 streamed experts (plan D8-D23).

``exl3_ram_miss_tables`` flattens what ``Exl3ShardRowSource`` knows (the per-expert
superset reads and the per-name segment map) plus each layer's pinned slabs into
int64 tensors, so the C++ thread reads and splits rows without Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES


@dataclass(frozen=True)
class Exl3RamMissTables:
    layer_ids: list[int]
    paths: list[str]
    file_sizes: torch.Tensor  # int64 [F]
    reads: torch.Tensor  # int64 [L, E, 4]: file index, aligned offset, aligned length, row start
    segments: torch.Tensor  # int64 [S, 4]: name index, dst offset, src offset, bytes
    slabs: torch.Tensor  # int64 [L, 6]: slab base addresses in EXL3_STREAMED_NAMES order
    row_bytes: torch.Tensor  # int64 [6]
    capacity: torch.Tensor  # int64 [L]
    slot_bytes: int
    # The slab tensors whose addresses are in ``slabs``: the C++ reader writes through
    # those raw addresses, so the tables own a reference to every slab (M5).
    keepalive: tuple = field(default=(), repr=False, compare=False)


def exl3_ram_miss_tables(
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
) -> Exl3RamMissTables:
    """Tables for the streamed layers in ``slabs_by_layer`` (ascending layer id = row order)."""
    layer_ids = sorted(slabs_by_layer)
    paths: list[str] = []
    file_index: dict[str, int] = {}
    reads = torch.empty((len(layer_ids), layout.num_experts, 4), dtype=torch.int64)
    widest = 0
    for row, layer_id in enumerate(layer_ids):
        for expert in range(layout.num_experts):
            record = layout.records[(layer_id, expert)]
            if record.path not in file_index:
                file_index[record.path] = len(paths)
                paths.append(record.path)
            offset, length, start = record.aligned_read(PAGE_BYTES)
            reads[row, expert] = torch.tensor([file_index[record.path], offset, length, start])
            widest = max(widest, length)
    names = {name: index for index, name in enumerate(EXL3_STREAMED_NAMES)}
    segment_table = torch.tensor(
        [[names[s.name], s.dst_offset, s.src_offset, s.nbytes] for s in segments], dtype=torch.int64
    )
    row_bytes = torch.tensor(
        [sum(s.nbytes for s in segments if s.name == name) for name in EXL3_STREAMED_NAMES], dtype=torch.int64
    )
    slabs = torch.empty((len(layer_ids), len(EXL3_STREAMED_NAMES)), dtype=torch.int64)
    capacity = torch.empty(len(layer_ids), dtype=torch.int64)
    for row, layer_id in enumerate(layer_ids):
        tensors = slabs_by_layer[layer_id]
        rows = {int(tensors[name].shape[0]) for name in EXL3_STREAMED_NAMES}
        if len(rows) != 1:
            raise ValueError(f"layer {layer_id}: pinned slabs disagree on their row count {rows}")
        capacity[row] = rows.pop()
        for name, index in names.items():
            slab = tensors[name]
            if not slab.is_contiguous() or slab.device.type != "cpu":
                raise ValueError(f"layer {layer_id} {name}: slab must be a contiguous CPU tensor")
            per_row = slab.numel() * slab.element_size() // max(int(capacity[row]), 1)
            if per_row != int(row_bytes[index]):
                raise ValueError(f"layer {layer_id} {name}: slab rows hold {per_row} B, expected {int(row_bytes[index])}")
            slabs[row, index] = slab.data_ptr()
    return Exl3RamMissTables(
        layer_ids=layer_ids,
        paths=paths,
        file_sizes=torch.tensor([os.path.getsize(p) for p in paths], dtype=torch.int64),
        reads=reads,
        segments=segment_table,
        slabs=slabs,
        row_bytes=row_bytes,
        capacity=capacity,
        slot_bytes=-(-widest // PAGE_BYTES) * PAGE_BYTES,
        keepalive=tuple(slabs_by_layer[layer_id][name] for layer_id in layer_ids for name in EXL3_STREAMED_NAMES),
    )
