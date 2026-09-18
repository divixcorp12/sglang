"""The EXL3 weight adapter for DeepSeek V4.1 (fake dequant, no kernels)."""

import pytest
import torch

from sglang.srt.models.deepseek_v4_exl3_weights import adapt_exl3_weights
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _exl3(prefix, in_f, out_f, fill):
    return [
        (f"{prefix}.suh", torch.full((in_f,), 1.0, dtype=torch.float16)),
        (f"{prefix}.svh", torch.full((out_f,), 1.0, dtype=torch.float16)),
        (f"{prefix}.mul1", torch.tensor(1, dtype=torch.int32)),
        (f"{prefix}.trellis", torch.full((in_f // 16, out_f // 16, 80), fill, dtype=torch.int16)),
    ]


def fake_dequant(t):
    # [in, out] filled with the trellis fill value, so the test can track tensors.
    return torch.full((t.in_features, t.out_features), float(t.trellis[0, 0, 0]), dtype=torch.float16)


def run(weights, groups=2):
    return dict(adapt_exl3_weights(iter(weights), fake_dequant, groups))


def test_native_exl3_and_plain_tensors_pass_through():
    weights = _exl3("layers.2.attn.wq_a", 64, 32, 1) + [("layers.2.attn_norm.weight", torch.ones(64))]
    out = run(weights)
    assert set(out) == {
        "layers.2.attn.wq_a.suh",
        "layers.2.attn.wq_a.svh",
        "layers.2.attn.wq_a.mul1",
        "layers.2.attn.wq_a.trellis",
        "layers.2.attn_norm.weight",
    }


@pytest.mark.parametrize("module", ["compressor.wkv", "compressor.wgate", "indexer.wk"])
def test_bf16_modules_are_dequantized_to_out_in(module):
    out = run(_exl3(f"layers.2.attn.{module}", 64, 32, 5))
    w = out[f"layers.2.attn.{module}.weight"]
    assert w.shape == (32, 64) and w.dtype == torch.bfloat16 and float(w[0, 0]) == 5.0
    assert len(out) == 1


def test_wo_a_slices_concatenate_in_group_order():
    weights = _exl3("layers.0.attn.wo_a.slice.1", 64, 16, 11) + _exl3("layers.0.attn.wo_a.slice.0", 64, 16, 10)
    out = run(weights, groups=2)
    w = out["layers.0.attn.wo_a.weight"]
    assert w.shape == (32, 64)
    assert float(w[0, 0]) == 10.0 and float(w[16, 0]) == 11.0


def test_interleaved_layers_do_not_mix():
    weights = _exl3("layers.0.attn.indexer.wk", 64, 32, 1)[:2] + _exl3("layers.1.attn.indexer.wk", 64, 32, 2)
    weights += _exl3("layers.0.attn.indexer.wk", 64, 32, 1)[2:]
    out = run(weights)
    assert float(out["layers.0.attn.indexer.wk.weight"][0, 0]) == 1.0
    assert float(out["layers.1.attn.indexer.wk.weight"][0, 0]) == 2.0


def test_incomplete_group_raises():
    with pytest.raises(ValueError, match="incomplete"):
        run(_exl3("layers.0.attn.wo_a.slice.0", 64, 16, 1), groups=2)


@pytest.mark.parametrize(
    "stem",
    [
        "layers.2.attn.wq_a",
        "layers.2.attn.wq_b",
        "layers.2.attn.wkv",
        "layers.2.attn.wo_b",
        "layers.2.attn.indexer.wq_b",
        "layers.2.ffn.shared_experts.w1",
        "layers.2.ffn.shared_experts.w3",
        "layers.2.ffn.shared_experts.w2",
        "layers.1.engram.wkv",
        "head",
    ],
)
@pytest.mark.parametrize("suffix", ["suh", "svh", "mul1", "trellis"])
def test_remap_treats_exl3_suffixes_like_weight(stem, suffix):
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    as_weight = remap(f"{stem}.weight")
    as_exl3 = remap(f"{stem}.{suffix}")
    assert as_exl3.rsplit(".", 1)[0] == as_weight.rsplit(".", 1)[0]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
