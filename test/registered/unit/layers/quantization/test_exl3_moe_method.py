"""exl3 MoE parameter registration and per-expert loading (no kernels)."""

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
E, HIDDEN, INTER = 4, 256, 128


def _moe():
    layer = nn.Module()
    layer.num_experts = E
    method = Exl3MoEMethod(Exl3Config.from_config(CFG))
    method.create_weights(layer, E, HIDDEN, INTER, torch.bfloat16)
    return layer, method


def _expert(in_f, out_f, fill):
    return {
        "trellis": torch.full((in_f // 16, out_f // 16, 48), fill, dtype=torch.int16),
        "suh": torch.full((in_f,), float(fill), dtype=torch.float16),
        "svh": torch.full((out_f,), float(fill), dtype=torch.float16),
        "mul1": torch.tensor(1, dtype=torch.int32),
    }


def _load_all(layer):
    for e in range(E):
        for shard, (in_f, out_f), prefix in (
            ("w1", (HIDDEN, INTER), "w13"),
            ("w3", (HIDDEN, INTER), "w13"),
            ("w2", (INTER, HIDDEN), "w2"),
        ):
            fill = 10 * e + {"w1": 1, "w3": 3, "w2": 2}[shard]
            for name, tensor in _expert(in_f, out_f, fill).items():
                param = getattr(layer, f"{prefix}_{name}")
                param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=e)


def test_loads_every_expert_into_its_slot():
    layer, method = _moe()
    _load_all(layer)
    method.process_weights_after_loading(layer)
    gate, up = layer.exl3_w13[2]
    assert int(gate.trellis[0, 0, 0]) == 21 and int(up.trellis[0, 0, 0]) == 23
    assert int(layer.exl3_w2[3].trellis[0, 0, 0]) == 32
    assert (gate.in_features, gate.out_features, gate.bits) == (HIDDEN, INTER, 3)


def test_missing_expert_detected():
    layer, method = _moe()
    _load_all(layer)
    layer.exl3_loaded.discard(("w2", "trellis", 1, 0))
    with pytest.raises(RuntimeError, match="expert 1"):
        method.process_weights_after_loading(layer)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
