"""SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION against the unfused chains it replaces, bit for bit.

Every comparison is of raw bits (``.view(torch.int16)``), never a tolerance: the fusion only moves where each
rounding happens, so any difference is a bug.
"""

import os

import pytest
import torch
from torch import nn

from sglang.kernels.ops.attention.dsv4.moe import silu_and_mul_clamp
from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm, hc_combine_norm_half
from sglang.kernels.ops.moe.dsv41_cast_fusion import exl3_scale_to_bf16, exl3_silu_mul_clamp_half
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod, exl3_swiglu_mlp
from sglang.srt.layers.quantization.exl3_ops import EXL3_HALF_INPUT, random_exl3_tensors
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    capturing_host_node_count,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)
CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
LIMIT = 10.0


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def assert_bits_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, a.shape, b.dtype, b.shape)
    diff = (bits(a) != bits(b)).sum().item()
    assert diff == 0, f"{diff} of {a.numel()} elements differ"


def _method(fusion: bool) -> Exl3LinearMethod:
    with envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.override(fusion):
        return Exl3LinearMethod(Exl3Config.from_config(CFG))


def _linear(in_features: int, out_features: int, parts: int, seed: int, bits_: int = 5) -> nn.Module:
    layer = nn.Module()
    with torch.device("cuda"):  # the weight loader materializes the parameters on the default device
        _method(False).create_weights(
            layer, in_features, [out_features] * parts, in_features, out_features * parts, torch.bfloat16
        )
        for part in range(parts):
            t = random_exl3_tensors(in_features, out_features, bits_, device="cuda", seed=seed + part)
            for name in ("trellis", "suh", "svh"):
                getattr(layer, name).weight_loader(getattr(layer, name), getattr(t, name), part)
            layer.mul1.weight_loader(layer.mul1, torch.tensor(1, dtype=torch.int32, device="cuda"), part)
        _method(False).process_weights_after_loading(layer)
    assert all(t.trellis.is_cuda for t in layer.exl3_tensors)
    return layer


@pytest.fixture(autouse=True)
def _clear_published_input():
    EXL3_HALF_INPUT.clear()
    yield
    EXL3_HALF_INPUT.clear()


# ---- kernels against their torch chains


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("inter", [2304, 8, 1032])
@pytest.mark.parametrize("scale", [1.0, 8.0, 3000.0])
def test_silu_mul_clamp_half_matches_the_cast_silu_cast_chain(rows, inter, scale):
    gen = torch.Generator(device="cuda").manual_seed(rows * 7919 + inter)
    gate_up = (torch.randn(rows, 2 * inter, device="cuda", generator=gen) * scale).to(torch.float16)
    gate_up[0, :4] = torch.tensor([LIMIT, -LIMIT, 65504.0, -65504.0], dtype=torch.float16)
    unfused_in = gate_up.to(torch.bfloat16)
    unfused_out = unfused_in.new_empty(rows, inter)
    silu_and_mul_clamp(unfused_in, unfused_out, LIMIT)
    assert_bits_equal(exl3_silu_mul_clamp_half(gate_up, LIMIT), unfused_out.to(torch.float16))


@pytest.mark.parametrize("n", [5120, 1, 5121, 100_000])
@pytest.mark.parametrize("factor", [1.5, 2.5, 0.7])
def test_scale_to_bf16_matches_cast_then_multiply(n, factor):
    gen = torch.Generator(device="cuda").manual_seed(n)
    routed = torch.randn(n, device="cuda", generator=gen) * torch.logspace(-30, 30, n, device="cuda")
    routed[: min(n, 3)] = torch.tensor([0.0, -0.0, 3.3e38], device="cuda")[: min(n, 3)]
    assert_bits_equal(exl3_scale_to_bf16(routed, factor), routed.to(torch.bfloat16) * factor)
    shaped = routed[: n - n % 5].view(-1, 5) if n >= 5 else routed.view(1, -1)
    assert_bits_equal(exl3_scale_to_bf16(shaped, factor), shaped.to(torch.bfloat16) * factor)


@pytest.mark.parametrize("rows", [1, 2, 8, 96])
def test_hc_combine_norm_half_keeps_the_bf16_output_and_adds_its_fp16_copy(rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 20480, device="cuda", generator=gen) * 4).to(torch.bfloat16)
    pre = torch.rand(rows, 4, device="cuda", generator=gen) * 2
    weight = (torch.randn(5120, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    reference = hc_combine_norm(x, pre, weight, 1e-6)
    y, y16 = hc_combine_norm_half(x, pre, weight, 1e-6)
    assert_bits_equal(y, reference)
    assert_bits_equal(y16, reference.to(torch.float16))


# ---- the published fp16 input


def test_a_published_input_serves_only_the_same_unmodified_tensor_on_the_same_stream():
    x = torch.randn(1, 64, device="cuda").to(torch.bfloat16)
    half = x.to(torch.float16)
    EXL3_HALF_INPUT.publish(x, half)
    assert EXL3_HALF_INPUT.take(x) is half
    assert EXL3_HALF_INPUT.take(x.clone()) is None
    assert EXL3_HALF_INPUT.take(x.view(1, 64)) is None
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert EXL3_HALF_INPUT.take(x) is None
    x.add_(1)
    assert EXL3_HALF_INPUT.take(x) is None


# ---- EXL3 linears, flag on against flag off


@pytest.mark.parametrize(
    "in_features, out_features, parts",
    [(5120, 1280, 1), (5120, 512, 1), (1280, 32768, 1), (5120, 2304, 2), (2304, 5120, 1), (256, 128, 3)],
)
@pytest.mark.parametrize("published", [False, True])
def test_linear_apply_is_bit_identical_with_the_flag_on(in_features, out_features, parts, published):
    layer = _linear(in_features, out_features, parts, seed=in_features + out_features)
    fused, unfused = _method(True), _method(False)
    for trial in range(4):
        gen = torch.Generator(device="cuda").manual_seed(trial)
        x = (torch.randn(1, in_features, device="cuda", generator=gen) * 2).to(torch.bfloat16)
        if published:
            EXL3_HALF_INPUT.publish(x, x.to(torch.float16))
        assert_bits_equal(fused.apply(layer, x), unfused.apply(layer, x))


@pytest.mark.parametrize("rows", [2, 300])
def test_linear_apply_leaves_more_than_one_row_to_the_unfused_path(rows):
    layer = _linear(5120, 2304, 2, seed=5)
    x = torch.randn(rows, 5120, device="cuda").to(torch.bfloat16)
    assert_bits_equal(_method(True).apply(layer, x), _method(False).apply(layer, x))


# ---- the shared expert


def _shared_expert(seed: int):
    return _linear(5120, 2304, 2, seed=seed), _linear(2304, 5120, 1, seed=seed + 100)


def _unfused_shared_expert(gate_up, down, x):
    method = _method(False)
    h = method.apply(gate_up, x)
    act = h.new_empty(h.shape[0], h.shape[1] // 2)
    silu_and_mul_clamp(h, act, LIMIT)
    return method.apply(down, act)


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("scale", [1.0, 30.0])
def test_swiglu_mlp_matches_the_unfused_shared_expert(published, scale):
    gate_up, down = _shared_expert(seed=11)
    for trial in range(4):
        gen = torch.Generator(device="cuda").manual_seed(trial)
        x = (torch.randn(1, 5120, device="cuda", generator=gen) * scale).to(torch.bfloat16)
        if published:
            EXL3_HALF_INPUT.publish(x, x.to(torch.float16))
        assert_bits_equal(exl3_swiglu_mlp(x, gate_up, down, LIMIT), _unfused_shared_expert(gate_up, down, x))


# ---- captured in a CUDA graph


def test_the_fused_chain_captures_without_host_nodes_and_replays_bit_identically():
    gate_up, down = _shared_expert(seed=21)
    wq_a = _linear(5120, 1280, 1, seed=31)
    wkv = _linear(5120, 512, 1, seed=41)
    fused = _method(True)
    residual = torch.zeros(1, 20480, device="cuda", dtype=torch.bfloat16)
    pre = torch.zeros(1, 4, device="cuda")
    weight = (torch.randn(5120, device="cuda") * 2).to(torch.bfloat16)
    routed = torch.zeros(1, 5120, device="cuda")

    def step():
        x, x16 = hc_combine_norm_half(residual, pre, weight, 1e-6)
        EXL3_HALF_INPUT.publish(x, x16)
        return (
            fused.apply(wq_a, x),
            fused.apply(wkv, x),
            exl3_swiglu_mlp(x, gate_up, down, LIMIT),
            exl3_scale_to_bf16(routed, 1.5),
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = step()
        host_nodes = capturing_host_node_count(torch.cuda.current_stream().cuda_stream)
    assert host_nodes == 0

    unfused = _method(False)
    for trial in range(8):
        gen = torch.Generator(device="cuda").manual_seed(100 + trial)
        residual.copy_((torch.randn(1, 20480, device="cuda", generator=gen) * 3).to(torch.bfloat16))
        pre.copy_(torch.rand(1, 4, device="cuda", generator=gen))
        routed.copy_(torch.randn(1, 5120, device="cuda", generator=gen) * 50)
        graph.replay()
        EXL3_HALF_INPUT.clear()
        x = hc_combine_norm(residual, pre, weight, 1e-6)
        assert_bits_equal(outs[0], unfused.apply(wq_a, x))
        assert_bits_equal(outs[1], unfused.apply(wkv, x))
        assert_bits_equal(outs[2], _unfused_shared_expert(gate_up, down, x))
        assert_bits_equal(outs[3], routed.to(torch.bfloat16) * 1.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
