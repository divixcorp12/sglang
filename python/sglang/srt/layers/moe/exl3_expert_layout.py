"""Byte layout of EXL3 routed experts inside the original safetensors shards.

An EXL3 export writes each routed expert's 12 tensors (w1/w2/w3 x suh/svh/mul1/trellis)
back to back, so one expert can be fetched with a single O_DIRECT read and no re-layout
copy. This module finds that byte range for every expert and checks the property that
makes it possible (contiguous, one file, one tensor layout per layer) instead of
assuming it.

Layers may differ: a mixed-K export quantizes some layers' experts at another bit width,
so their rows are a different size. Every expert of one layer shares a row geometry; the
layout holds the distinct geometries and each layer's index into them. Each tensor's K is
read from its trellis shape ``[k/16, n/16, 16 * K]``, never from the quantization config.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass

import msgspec

_HEADER_LEN_BYTES = 8
# Not part of the EXL3 format: some mixed-K exports write it next to the shards. When present, the headers must agree.
K_MAP_FILENAME = "exl3_k_map.json"


@dataclass(frozen=True)
class Exl3TensorSpan:
    name: str
    rel_offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Exl3ExpertRecord:
    layer: int
    expert: int
    path: str
    file_offset: int
    nbytes: int

    def aligned_read(self, page: int = 4096) -> tuple[int, int, int]:
        """(aligned_offset, aligned_len, row_start_in_buffer) of one O_DIRECT read."""
        start = self.file_offset // page * page
        end = -(-(self.file_offset + self.nbytes) // page) * page
        return start, end - start, self.file_offset - start


class Exl3RowGeometry(msgspec.Struct, frozen=True):
    """One layer's expert row: its tensors by offset in the row, and the row's size."""

    tensors: tuple[Exl3TensorSpan, ...]
    row_bytes: int

    def trellis_bits(self, projection: str) -> int:
        """K of ``projection`` (``w1``, ``w2`` or ``w3``): its trellis's last dimension is ``16 * K``."""
        for span in self.tensors:
            if span.name == f"{projection}.trellis":
                return span.shape[-1] // 16
        raise KeyError(f"exl3 row has no {projection}.trellis")


@dataclass(frozen=True)
class Exl3ExpertLayout:
    geometries: tuple[Exl3RowGeometry, ...]
    # Per layer, its index into geometries.
    layer_geometry: tuple[int, ...]
    records: dict[tuple[int, int], Exl3ExpertRecord]
    num_layers: int
    num_experts: int

    def geometry(self, layer: int) -> Exl3RowGeometry:
        return self.geometries[self.layer_geometry[layer]]

    @property
    def max_row_bytes(self) -> int:
        return max(geometry.row_bytes for geometry in self.geometries)

    @property
    def uniform_geometry(self) -> Exl3RowGeometry:
        """The one geometry of a checkpoint whose layers all share it; raises for a mixed-K checkpoint."""
        if len(self.geometries) != 1:
            raise ValueError(
                f"this EXL3 checkpoint has {len(self.geometries)} expert row geometries (a mixed-K export); "
                "read each layer's with layout.geometry(layer)"
            )
        return self.geometries[0]

    @property
    def tensors(self) -> tuple[Exl3TensorSpan, ...]:
        return self.uniform_geometry.tensors

    @property
    def row_bytes(self) -> int:
        return self.uniform_geometry.row_bytes


def read_safetensors_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return _HEADER_LEN_BYTES + length, header


def build_exl3_expert_layout(model_dir: str, prefix: str = "layers") -> Exl3ExpertLayout:
    pattern = re.compile(
        rf"^{re.escape(prefix)}\.(\d+)\.ffn\.experts\.(\d+)\."
        r"(w[123]\.(?:suh|svh|mul1|trellis))$"
    )
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({shard for name, shard in weight_map.items() if pattern.match(name)})
    if not shards:
        raise ValueError(f"no tensors match {prefix}.<L>.ffn.experts.<E>.* in {model_dir}")

    pieces: dict[tuple[int, int], list[tuple]] = {}
    for shard in shards:
        path = os.path.join(model_dir, shard)
        data_start, header = read_safetensors_header(path)
        for name, info in header.items():
            match = pattern.match(name)
            if match is None:
                continue
            begin, end = info["data_offsets"]
            key = (int(match.group(1)), int(match.group(2)))
            pieces.setdefault(key, []).append(
                (data_start + begin, end - begin, match.group(3), info["dtype"],
                 tuple(info["shape"]), path)
            )

    layer_spans: dict[int, tuple[tuple[int, int], tuple[Exl3TensorSpan, ...]]] = {}
    records = {}
    for key in sorted(pieces):
        spans, record = _expert_row(key, pieces[key])
        reference = layer_spans.setdefault(key[0], (key, spans))
        if spans != reference[1]:
            raise ValueError(
                f"expert {key} tensor layout differs from expert {reference[0]} of the same layer"
            )
        records[key] = record

    num_layers = 1 + max(layer for layer, _ in records)
    num_experts = 1 + max(expert for _, expert in records)
    missing = [
        (layer, expert)
        for layer in range(num_layers)
        for expert in range(num_experts)
        if (layer, expert) not in records
    ]
    if missing:
        raise ValueError(f"{len(missing)} experts missing, first {missing[:4]}")
    geometries: list[Exl3RowGeometry] = []
    layer_geometry = []
    for layer in range(num_layers):
        spans = layer_spans[layer][1]
        geometry = Exl3RowGeometry(tensors=spans, row_bytes=sum(span.nbytes for span in spans))
        if geometry not in geometries:
            geometries.append(geometry)
        layer_geometry.append(geometries.index(geometry))
    layout = Exl3ExpertLayout(
        geometries=tuple(geometries),
        layer_geometry=tuple(layer_geometry),
        records=records,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    _check_k_map(layout, model_dir, prefix)
    return layout


def _expert_row(
    key: tuple[int, int], pieces: list[tuple]
) -> tuple[tuple[Exl3TensorSpan, ...], Exl3ExpertRecord]:
    """One expert's tensor spans (contiguous, in one file) and its record."""
    parts = sorted(pieces)
    paths = sorted({part[5] for part in parts})
    if len(paths) != 1:
        raise ValueError(f"expert {key} spans {len(paths)} files: {paths}")
    start = cursor = parts[0][0]
    spans = []
    for offset, nbytes, suffix, dtype, shape, _ in parts:
        if offset != cursor:
            raise ValueError(
                f"expert {key}: {suffix} starts at byte {offset}, expected {cursor}; "
                "its tensors are not contiguous"
            )
        spans.append(Exl3TensorSpan(suffix, offset - start, nbytes, dtype, shape))
        cursor += nbytes
    return tuple(spans), Exl3ExpertRecord(key[0], key[1], paths[0], start, cursor - start)


def _check_k_map(layout: Exl3ExpertLayout, model_dir: str, prefix: str) -> None:
    """Refuse a checkpoint whose ``exl3_k_map.json`` disagrees with its trellis shapes.

    Only the routed experts of ``layers`` (``routed``: layer -> K) and of ``mtp``
    (``mtp_bits``) are checked; the file's other entries describe tensors this layout
    does not read.
    """
    path = os.path.join(model_dir, K_MAP_FILENAME)
    if not os.path.exists(path):
        return
    with open(path) as f:
        k_map = json.load(f)
    if prefix == "layers":
        expected = {int(layer): int(k) for layer, k in k_map.get("routed", {}).items()}
    elif prefix == "mtp" and "mtp_bits" in k_map:
        expected = {layer: int(k_map["mtp_bits"]) for layer in range(layout.num_layers)}
    else:
        return
    for layer, k in sorted(expected.items()):
        if layer >= layout.num_layers:
            raise ValueError(f"{path}: layer {layer} is not in the checkpoint's {layout.num_layers} {prefix}")
        actual = {w: layout.geometry(layer).trellis_bits(w) for w in ("w1", "w2", "w3")}
        if set(actual.values()) != {k}:
            raise ValueError(f"{path}: {prefix} layer {layer} has K {k}, its trellises say {actual}")
