"""Byte layout of EXL3 routed experts inside the original safetensors shards.

An EXL3 export writes each routed expert's 12 tensors (w1/w2/w3 x suh/svh/mul1/trellis)
back to back, so one expert can be fetched with a single O_DIRECT read and no re-layout
copy. This module finds that byte range for every expert and checks the property that
makes it possible (contiguous, one file, the same tensor layout everywhere) instead of
assuming it.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass

_HEADER_LEN_BYTES = 8


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


@dataclass(frozen=True)
class Exl3ExpertLayout:
    tensors: tuple[Exl3TensorSpan, ...]
    row_bytes: int
    records: dict[tuple[int, int], Exl3ExpertRecord]
    num_layers: int
    num_experts: int


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

    reference_key, reference, records = None, None, {}
    for key in sorted(pieces):
        parts = sorted(pieces[key])
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
        spans = tuple(spans)
        if reference is None:
            reference_key, reference = key, spans
        elif spans != reference:
            raise ValueError(
                f"expert {key} tensor layout differs from expert {reference_key}"
            )
        records[key] = Exl3ExpertRecord(key[0], key[1], paths[0], start, cursor - start)

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
    return Exl3ExpertLayout(
        tensors=reference,
        row_bytes=sum(span.nbytes for span in reference),
        records=records,
        num_layers=num_layers,
        num_experts=num_experts,
    )
