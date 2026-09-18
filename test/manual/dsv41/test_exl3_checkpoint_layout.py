"""The real EXL3 export: every routed and draft expert is one contiguous, single-file row."""

import os

import pytest

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout

EXL3_DIR = os.environ.get("DSV41_EXL3_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(EXL3_DIR, "model.safetensors.index.json")),
    reason="needs the DeepSeek-V4.1-Flash EXL3 checkpoint",
)


def test_routed_experts():
    layout = build_exl3_expert_layout(EXL3_DIR)
    assert (layout.num_layers, layout.num_experts) == (40, 384)
    assert len(layout.tensors) == 12
    assert layout.row_bytes == 13_315_596
    names = [t.name for t in layout.tensors]
    assert names[:4] == ["w1.suh", "w1.svh", "w1.mul1", "w1.trellis"]
    assert layout.tensors[3].rel_offset == 14_852


def test_draft_experts():
    layout = build_exl3_expert_layout(EXL3_DIR, prefix="mtp")
    assert (layout.num_layers, layout.num_experts) == (3, 128)
    assert layout.row_bytes == 17_739_276


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
