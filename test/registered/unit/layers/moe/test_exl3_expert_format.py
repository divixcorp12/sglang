"""The EXL3 expert format's row schema: six streamed names and the segment map."""

import dataclasses

import pytest
import torch

from sglang.srt.layers.moe.exl3_expert_format import (
    EXL3_MAX_GATHER_ROWS,
    EXL3_STREAMED_NAMES,
    Exl3ExpertFormat,
)
from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    Exl3ExpertRecord,
    Exl3TensorSpan,
    build_exl3_expert_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import expert_spec, write_fake_exl3

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(tmp_path, num_layers=2, num_experts=4):
    write_fake_exl3(str(tmp_path), num_layers=num_layers, num_experts=num_experts)
    return build_exl3_expert_layout(str(tmp_path))


def _synthetic_layout(hidden, inter):
    """A layout with the real export's tensor order and sizes, and no files."""
    spans, cursor = [], 0
    for suffix, dtype, shape, nbytes in expert_spec(hidden, inter):
        spans.append(Exl3TensorSpan(suffix, cursor, nbytes, dtype, tuple(shape)))
        cursor += nbytes
    record = Exl3ExpertRecord(0, 0, "unused.safetensors", 0, cursor)
    return Exl3ExpertLayout(tuple(spans), cursor, {(0, 0): record}, 1, 1)


def test_specs_are_the_six_streamed_names(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1)
    specs = fmt.tensor_specs(None)
    assert tuple(spec.name for spec in specs) == EXL3_STREAMED_NAMES == fmt.names
    shapes = {spec.name: (spec.row_shape, spec.dtype) for spec in specs}
    assert shapes == {
        "w13_trellis": ((2, 8, 8, 48), torch.int16),
        "w13_suh": ((2, 128), torch.float16),
        "w13_svh": ((2, 128), torch.float16),
        "w2_trellis": ((1, 8, 8, 48), torch.int16),
        "w2_suh": ((1, 128), torch.float16),
        "w2_svh": ((1, 128), torch.float16),
    }
    assert all(spec.residence == "host" for spec in specs)
    # Every byte of the row except the three 4-byte mul1 scalars is streamed.
    assert sum(spec.row_bytes for spec in specs) == layout.row_bytes - 12


def test_segment_map_covers_every_streamed_byte_once(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 0)
    row_bytes = {spec.name: spec.row_bytes for spec in fmt.tensor_specs(None)}
    source = [0] * layout.row_bytes
    target = {name: [0] * nbytes for name, nbytes in row_bytes.items()}
    for segment in fmt.segment_map():
        for i in range(segment.nbytes):
            source[segment.src_offset + i] += 1
            target[segment.name][segment.dst_offset + i] += 1
    mul1 = {span.rel_offset + i for span in layout.tensors if span.name.endswith("mul1") for i in range(4)}
    assert [i for i, n in enumerate(source) if n != 1] == sorted(mul1)
    assert all(n == 1 for counts in target.values() for n in counts)


def test_w1_and_w3_are_parts_zero_and_one(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 0)
    by_offset = {span.rel_offset: span.name for span in layout.tensors}
    parts = {by_offset[s.src_offset]: (s.name, s.part) for s in fmt.segment_map()}
    for kind in ("trellis", "suh", "svh"):
        assert parts[f"w1.{kind}"] == (f"w13_{kind}", 0)
        assert parts[f"w3.{kind}"] == (f"w13_{kind}", 1)
        assert parts[f"w2.{kind}"] == (f"w2_{kind}", 0)
    assert len(fmt.segment_map()) == 9


def test_real_export_row_sizes():
    fmt = Exl3ExpertFormat(_synthetic_layout(5120, 2304), 0)
    sizes = {spec.name: spec.row_bytes for spec in fmt.tensor_specs(None)}
    assert sizes == {
        "w13_trellis": 8_847_360,
        "w13_suh": 20_480,
        "w13_svh": 9_216,
        "w2_trellis": 4_423_680,
        "w2_suh": 4_608,
        "w2_svh": 10_240,
    }
    assert sum(sizes.values()) == 13_315_584
    assert all(size % 16 == 0 for size in sizes.values())


def test_format_flags_and_sources(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1)
    assert fmt.key == "exl3"
    assert fmt.supports_graph_gather is False
    assert fmt.supports_host_arena is False
    assert fmt.max_gather_rows == EXL3_MAX_GATHER_ROWS == 64
    assert fmt.num_experts(None) == 4
    assert fmt.source(None, "w13_trellis") is None


def test_rejects_a_layer_outside_the_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="outside the checkpoint"):
        Exl3ExpertFormat(_layout(tmp_path), 2)


def test_rejects_an_unexpected_row(tmp_path):
    layout = _layout(tmp_path)
    broken = dataclasses.replace(layout, tensors=layout.tensors[:-1])
    with pytest.raises(ValueError, match="expected"):
        Exl3ExpertFormat(broken, 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
