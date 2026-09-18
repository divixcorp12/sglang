"""Padded slot layout for an EXL3 expert row."""

import pytest

from sglang.srt.layers.moe.exl3_expert_layout import Exl3TensorSpan
from sglang.srt.layers.moe.exl3_slot_layout import build_exl3_slot_layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _real_expert_spans():
    """The measured DeepSeek-V4.1-Flash EXL3 3.0bpw expert (13,315,596 raw bytes)."""
    spans, offset = [], 0
    for w, (d_in, d_out) in (("w1", (5120, 2304)), ("w2", (2304, 5120)), ("w3", (5120, 2304))):
        for part, dtype, shape, nbytes in (
            ("suh", "F16", (d_in,), 2 * d_in),
            ("svh", "F16", (d_out,), 2 * d_out),
            ("mul1", "I32", (), 4),
            ("trellis", "I16", (d_in // 16, d_out // 16, 48), d_in * d_out * 3 // 8),
        ):
            spans.append(Exl3TensorSpan(f"{w}.{part}", offset, nbytes, dtype, shape))
            offset += nbytes
    assert offset == 13_315_596
    return spans


def test_real_expert_128_byte_slots():
    layout = build_exl3_slot_layout(_real_expert_spans(), alignment=128)
    assert layout.slot_bytes == 13_315_968
    assert layout.dst_offset("w1.trellis") == 14_976
    assert all(s.dst_offset % 128 == 0 for s in layout.segments)


def test_real_expert_16_byte_slots():
    layout = build_exl3_slot_layout(_real_expert_spans(), alignment=16)
    assert layout.slot_bytes == 13_315_632
    assert layout.dst_offset("w1.trellis") == 14_864


def test_segments_preserve_order_and_do_not_overlap():
    spans = _real_expert_spans()
    layout = build_exl3_slot_layout(spans, alignment=128)
    assert [s.name for s in layout.segments] == [t.name for t in spans]
    assert [s.src_offset for s in layout.segments] == [t.rel_offset for t in spans]
    for prev, cur in zip(layout.segments, layout.segments[1:]):
        assert prev.dst_offset + prev.nbytes <= cur.dst_offset
    last = layout.segments[-1]
    assert last.dst_offset + last.nbytes <= layout.slot_bytes
    assert layout.slot_bytes % 128 == 0


def test_alignment_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        build_exl3_slot_layout(_real_expert_spans(), alignment=24)


def test_unknown_tensor_name_raises():
    layout = build_exl3_slot_layout(_real_expert_spans())
    with pytest.raises(KeyError):
        layout.dst_offset("w4.trellis")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
