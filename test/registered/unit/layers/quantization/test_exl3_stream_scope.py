"""EXL3 expert streaming applies only to the target model's routed experts:
the DSpark draft's stages (module names like ``stages.<S>.mlp.experts``) must
load resident even when SGLANG_DSV41_EXPERT_STREAM is on."""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from sglang.srt.models.deepseek_v4_exl3_weights import (
    is_dspark_draft_expert_module,
    is_streamed_expert_module,
)
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


def test_draft_routes_are_appended_one_json_line_per_call(tmp_path):
    import json

    from sglang.srt.layers.quantization.exl3.exl3 import record_draft_routes

    path = tmp_path / "routes.jsonl"
    ids = torch.tensor([[3, 7, 9], [3, 1, 2]], dtype=torch.int32)
    with envs.SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH.override(str(path)):
        record_draft_routes(1, ids)
        record_draft_routes(2, ids[:1])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        {"layer": 1, "ids": [[3, 7, 9], [3, 1, 2]]},
        {"layer": 2, "ids": [[3, 7, 9]]},
    ]


def test_draft_routes_are_not_written_when_the_path_is_unset(tmp_path):
    from sglang.srt.layers.quantization.exl3.exl3 import record_draft_routes

    with envs.SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH.override(""):
        record_draft_routes(1, torch.zeros(1, 3, dtype=torch.int32))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("stages.0.mlp.experts", True),
        ("model.stages.2.mlp.experts", True),
        ("model.layers.0.mlp.experts", False),
        ("stages.0.mlp.shared_experts", False),
    ],
)
def test_only_draft_stage_experts_are_draft_modules(prefix, expected):
    assert is_dspark_draft_expert_module(prefix) is expected
