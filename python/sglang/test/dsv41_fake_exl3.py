"""A tiny EXL3 checkpoint on disk whose expert bytes are known (tests only).

Each shard starts with a 6-byte plain tensor so no expert row is page- or
16-byte-aligned in the file, as in the real export.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

HIDDEN = 128
INTER = 128
BITS = 3


def expert_spec(hidden: int = HIDDEN, inter: int = INTER, bits: int = BITS):
    """[(suffix, dtype, shape, nbytes)] in on-disk order, as in the real export."""
    spec = []
    for w, (d_in, d_out) in (("w1", (hidden, inter)), ("w2", (inter, hidden)), ("w3", (hidden, inter))):
        spec += [
            (f"{w}.suh", "F16", (d_in,), 2 * d_in),
            (f"{w}.svh", "F16", (d_out,), 2 * d_out),
            (f"{w}.mul1", "I32", (), 4),
            (f"{w}.trellis", "I16", (d_in // 16, d_out // 16, 16 * bits), d_in * d_out * bits // 8),
        ]
    return spec


def expert_bytes(layer: int, expert: int, row_bytes: int) -> bytes:
    rng = np.random.default_rng(layer * 100_003 + expert)
    return rng.integers(0, 256, row_bytes, dtype=np.uint8).tobytes()


def _write_shard(path: str, tensors: list[tuple[str, str, tuple, bytes]]) -> None:
    header, offset = {}, 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for *_, data in tensors:
            f.write(data)


def write_fake_exl3(
    directory: str,
    num_layers: int,
    num_experts: int,
    experts_per_shard: int = 3,
    hidden: int = HIDDEN,
    inter: int = INTER,
) -> dict[tuple[int, int], bytes]:
    """Write shards plus model.safetensors.index.json; return each expert's raw row."""
    spec = expert_spec(hidden, inter)
    row_bytes = sum(nbytes for *_, nbytes in spec)
    keys = [(layer, expert) for layer in range(num_layers) for expert in range(num_experts)]
    rows, weight_map = {}, {}
    for shard_index, first in enumerate(range(0, len(keys), experts_per_shard)):
        filename = f"model-{shard_index + 1:05d}.safetensors"
        tensors = [(f"layers.{keys[first][0]}.attn.norm{shard_index}.weight", "BF16", (3,), b"\x01" * 6)]
        for layer, expert in keys[first : first + experts_per_shard]:
            row = expert_bytes(layer, expert, row_bytes)
            rows[(layer, expert)] = row
            cursor = 0
            for suffix, dtype, shape, nbytes in spec:
                name = f"layers.{layer}.ffn.experts.{expert}.{suffix}"
                tensors.append((name, dtype, shape, row[cursor : cursor + nbytes]))
                cursor += nbytes
        _write_shard(os.path.join(directory, filename), tensors)
        weight_map.update({name: filename for name, *_ in tensors})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return rows
