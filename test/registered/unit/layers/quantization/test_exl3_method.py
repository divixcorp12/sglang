"""exl3 config parsing, parameter registration and lazy part loading (no kernels)."""

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HF_QUANT = {
    "quant_method": "exl3",
    "version": "1.4.2",
    "bits": 3.02,
    "head_bits": 6,
    "codebook": "mul1",
    "out_scales": "always",
}


def _layer(parts, in_f, out_part):
    layer = nn.Module()
    method = Exl3LinearMethod(Exl3Config.from_config(HF_QUANT))
    method.create_weights(layer, in_f, [out_part] * parts, in_f, out_part * parts, torch.bfloat16)
    return layer, method


def _tensors(in_f, out_f, bits, fill):
    return {
        "trellis": torch.full((in_f // 16, out_f // 16, 16 * bits), fill, dtype=torch.int16),
        "suh": torch.full((in_f,), float(fill), dtype=torch.float16),
        "svh": torch.full((out_f,), float(fill), dtype=torch.float16),
        "mul1": torch.tensor(-2082680531, dtype=torch.int32),
    }


def test_registered():
    assert QUANTIZATION_METHODS["exl3"] is Exl3Config
    assert Exl3Config.from_config(HF_QUANT).get_name() == "exl3"


def test_rejects_other_codebooks():
    with pytest.raises(ValueError, match="mul1"):
        Exl3Config.from_config({**HF_QUANT, "codebook": "mcg"})


def test_single_linear_loads_and_builds_tensors():
    layer, method = _layer(1, 5120, 1280)
    for name, tensor in _tensors(5120, 1280, 5, 3).items():
        getattr(layer, name).weight_loader(getattr(layer, name), tensor)
    method.process_weights_after_loading(layer)
    (t,) = layer.exl3_tensors
    assert (t.in_features, t.out_features, t.bits, t.mul1) == (5120, 1280, 5, True)
    assert int(t.trellis[0, 0, 0]) == 3


def test_merged_linear_keeps_parts_separate():
    layer, method = _layer(2, 5120, 2304)
    for shard, fill in ((1, 7), (0, 5)):
        for name, tensor in _tensors(5120, 2304, 5, fill).items():
            getattr(layer, name).weight_loader(getattr(layer, name), tensor, shard)
    method.process_weights_after_loading(layer)
    gate, up = layer.exl3_tensors
    assert int(gate.trellis[0, 0, 0]) == 5 and int(up.trellis[0, 0, 0]) == 7
    assert float(gate.suh[0]) == 5.0 and float(up.suh[0]) == 7.0


def test_unequal_parts_rejected():
    method = Exl3LinearMethod(Exl3Config.from_config(HF_QUANT))
    with pytest.raises(ValueError, match="equal"):
        method.create_weights(nn.Module(), 5120, [1280, 512], 5120, 1792, torch.bfloat16)


def test_missing_part_detected():
    layer, method = _layer(2, 5120, 2304)
    for name, tensor in _tensors(5120, 2304, 5, 1).items():
        getattr(layer, name).weight_loader(getattr(layer, name), tensor, 0)
    with pytest.raises(RuntimeError, match="part 1"):
        method.process_weights_after_loading(layer)


def test_weights_proj_stays_unquantized():
    # DeepSeek V4.1's indexer.weights_proj is a ReplicatedLinear (LinearBase), so
    # exl3's get_quant_method is actually consulted for it and must route it to
    # the unquantized path. The router gate (model.layers.N.mlp.gate) is a plain
    # MoEGate nn.Module (not LinearBase) in this model family, so get_quant_method
    # is never called for it in practice; that case is intentionally not tested
    # here (see task-3-report.md).
    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    cfg = Exl3Config.from_config(HF_QUANT)
    prefix = "model.layers.2.self_attn.indexer.weights_proj"
    layer = ReplicatedLinear(64, 32, bias=False, quant_config=None, prefix=prefix)
    assert isinstance(cfg.get_quant_method(layer, prefix), UnquantizedLinearMethod)


def test_lm_head_loads_via_vocab_parallel_embedding_create_weights_order():
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    cfg = Exl3Config.from_config(HF_QUANT)
    layer = ParallelLMHead(129280, 5120, quant_config=cfg, enable_tp=False)
    assert layer.exl3_in == 5120
    assert layer.exl3_out_part == 129280


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
