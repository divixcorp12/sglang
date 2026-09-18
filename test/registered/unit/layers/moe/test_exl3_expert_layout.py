"""Byte layout of EXL3 routed experts inside safetensors shards."""

import json
import os
import struct

import pytest

from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertRecord,
    build_exl3_expert_layout,
    read_safetensors_header,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (suffix, dtype, shape, nbytes) in on-disk order; a tiny stand-in for the real 12 tensors.
EXPERT = [
    (f"{w}.{part}", dtype, shape, nbytes)
    for w in ("w1", "w2", "w3")
    for part, dtype, shape, nbytes in (
        ("suh", "F16", (8,), 16),
        ("svh", "F16", (4,), 8),
        ("mul1", "I32", (), 4),
        ("trellis", "I16", (2, 3), 12),
    )
]
ROW_BYTES = 3 * (16 + 8 + 4 + 12)


def _write_shard(path, tensors):
    """tensors: [(name, dtype, shape, nbytes)] written in exactly this order."""
    header, offset = {}, 0
    for name, dtype, shape, nbytes in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)
    return 8 + len(blob)


def _expert(layer, expert, spec=EXPERT):
    return [
        (f"layers.{layer}.ffn.experts.{expert}.{suffix}", dtype, shape, nbytes)
        for suffix, dtype, shape, nbytes in spec
    ]


def _write_model(directory, shards):
    """shards: {filename: [(name, dtype, shape, nbytes)]}; also writes the index."""
    starts, weight_map = {}, {}
    for filename, tensors in shards.items():
        starts[filename] = _write_shard(os.path.join(directory, filename), tensors)
        weight_map.update({name: filename for name, *_ in tensors})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return starts


def test_contiguous_experts_across_two_shards(tmp_path):
    spacer = [("layers.1.attn.wq.weight", "BF16", (3,), 6)]
    starts = _write_model(
        tmp_path,
        {
            "model-00001.safetensors": _expert(0, 0) + _expert(0, 1),
            "model-00002.safetensors": spacer + _expert(1, 0) + _expert(1, 1),
        },
    )
    layout = build_exl3_expert_layout(str(tmp_path))

    assert (layout.num_layers, layout.num_experts) == (2, 2)
    assert layout.row_bytes == ROW_BYTES
    assert [t.name for t in layout.tensors] == [s for s, *_ in EXPERT]
    assert layout.tensors[3].name == "w1.trellis"
    assert layout.tensors[3].rel_offset == 16 + 8 + 4
    record = layout.records[(1, 0)]
    assert record.path.endswith("model-00002.safetensors")
    assert record.file_offset == starts["model-00002.safetensors"] + 6
    assert record.nbytes == ROW_BYTES
    assert layout.records[(0, 1)].file_offset == (
        starts["model-00001.safetensors"] + ROW_BYTES
    )


def test_read_safetensors_header_drops_metadata(tmp_path):
    path = tmp_path / "x.safetensors"
    start = _write_shard(path, [("a", "F16", (2,), 4)])
    data_start, header = read_safetensors_header(str(path))
    assert data_start == start
    assert set(header) == {"a"}


def test_aligned_read_covers_row():
    record = Exl3ExpertRecord(layer=0, expert=0, path="x", file_offset=5000, nbytes=10000)
    assert record.aligned_read(4096) == (4096, 12288, 904)
    aligned = Exl3ExpertRecord(layer=0, expert=0, path="x", file_offset=8192, nbytes=4096)
    assert aligned.aligned_read(4096) == (8192, 4096, 0)


def test_gap_between_tensors_raises(tmp_path):
    tensors = _expert(0, 0)
    tensors.insert(4, ("layers.0.attn.wq.weight", "BF16", (2,), 4))
    _write_model(tmp_path, {"model-00001.safetensors": tensors})
    with pytest.raises(ValueError, match="not contiguous"):
        build_exl3_expert_layout(str(tmp_path))


def test_expert_split_across_files_raises(tmp_path):
    tensors = _expert(0, 0)
    _write_model(
        tmp_path,
        {
            "model-00001.safetensors": tensors[:6],
            "model-00002.safetensors": tensors[6:],
        },
    )
    with pytest.raises(ValueError, match="spans 2 files"):
        build_exl3_expert_layout(str(tmp_path))


def test_layout_mismatch_raises(tmp_path):
    other = [
        (s, d, sh, 14 if s == "w3.trellis" else n) for s, d, sh, n in EXPERT
    ]
    _write_model(
        tmp_path,
        {"model-00001.safetensors": _expert(0, 0) + _expert(0, 1, spec=other)},
    )
    with pytest.raises(ValueError, match="layout differs"):
        build_exl3_expert_layout(str(tmp_path))


def test_missing_expert_raises(tmp_path):
    _write_model(
        tmp_path,
        {"model-00001.safetensors": _expert(0, 0) + _expert(0, 1) + _expert(1, 0)},
    )
    with pytest.raises(ValueError, match="missing"):
        build_exl3_expert_layout(str(tmp_path))


def test_draft_prefix_selects_mtp_experts(tmp_path):
    mtp = [(n.replace("layers.", "mtp.", 1), d, s, b) for n, d, s, b in _expert(0, 0)]
    _write_model(tmp_path, {"model-00001.safetensors": _expert(0, 0) + mtp})
    layout = build_exl3_expert_layout(str(tmp_path), prefix="mtp")
    assert list(layout.records) == [(0, 0)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
