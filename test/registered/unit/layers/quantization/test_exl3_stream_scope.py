"""EXL3 expert streaming applies only to the target model's routed experts:
the DSpark draft's stages (module names like ``stages.<S>.mlp.experts``) must
load resident even when SGLANG_DSV41_EXPERT_STREAM is on."""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from sglang.srt.models.deepseek_v4_exl3_weights import is_streamed_expert_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.0.mlp.experts", True),
        ("model.layers.39.mlp.experts", True),
        ("stages.0.mlp.experts", False),
        ("model.stages.2.mlp.experts", False),
    ],
)
def test_only_target_routed_experts_stream(prefix, expected):
    assert is_streamed_expert_module(prefix) is expected


def test_a_non_streamed_module_never_streams_even_with_the_env_on():
    layer = torch.nn.Module()
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True):
        method = Exl3MoEMethod(Exl3Config(3.0, 6, "mul1", "1.4"), streamed=False)
        method.create_weights(layer, 128, 64, 32, torch.bfloat16)
    assert layer.exl3_streamed is False
    assert "w13_trellis" in dict(layer.named_parameters())
