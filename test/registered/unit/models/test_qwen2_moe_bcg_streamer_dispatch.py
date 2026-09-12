"""Dispatch regressions for Qwen2MoE streamed-expert CUDA graph paths."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen2_moe as qwen2_moe_module
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoDeepEP:
    def is_deepep(self):
        return False

    def is_deepep_v2(self):
        return False

    def is_mori(self):
        return False


def _make_block(streamed_experts):
    block = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.alt_stream = object()
    block.experts = SimpleNamespace()
    if streamed_experts:
        block.experts._nvfp4_expert_streamer = object()
    block.shared_expert_gate = None
    block.tp_size = 1
    calls = []

    def dual_stream(hidden_states, **_):
        calls.append("dual_stream")
        return hidden_states + 10, None

    def shared_experts(hidden_states, **_):
        calls.append("shared_experts")
        return None

    def router_experts(hidden_states, **_):
        calls.append("router_experts")
        return hidden_states + 20

    block.forward_normal_dual_stream = dual_stream
    block._forward_shared_experts = shared_experts
    block._forward_router_experts = router_experts
    return block, calls


@pytest.mark.parametrize(
    ("breakable", "streamed_experts", "expected_calls", "expected_offset"),
    [
        (True, True, ["shared_experts", "router_experts"], 20),
        (False, True, ["dual_stream"], 10),
        (True, False, ["dual_stream"], 10),
    ],
)
def test_qwen2_moe_streamed_experts_avoid_dual_stream_only_in_bcg(
    monkeypatch,
    breakable,
    streamed_experts,
    expected_calls,
    expected_offset,
):
    block, calls = _make_block(streamed_experts)
    monkeypatch.setattr(qwen2_moe_module, "get_moe_a2a_backend", _NoDeepEP)
    monkeypatch.setattr(qwen2_moe_module, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(
        qwen2_moe_module, "is_in_breakable_cuda_graph", lambda: breakable
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    hidden_states = torch.ones((1, 2))
    output = block(hidden_states)

    assert calls == expected_calls
    torch.testing.assert_close(output, hidden_states + expected_offset)
