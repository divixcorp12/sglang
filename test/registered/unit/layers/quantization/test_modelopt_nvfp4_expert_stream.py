"""Integration tests for ModelOpt NVFP4 selected-expert streaming."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerBackend
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.layers.quantization import modelopt_quant


def _cpu_parameter(shape, dtype, start=0, pinned=True):
    values = torch.arange(
        start,
        start + torch.tensor(shape).prod().item(),
        dtype=torch.float32,
    ).reshape(shape)
    data = torch.empty(shape, dtype=dtype, pin_memory=pinned)
    data.copy_(values.to(dtype))
    return torch.nn.Parameter(data, requires_grad=False)


class _Dispatcher:
    def set_quant_config(self, config):
        self.quant_config = config


class _CaptureRunner:
    def run(self, dispatch_output, quant_info):
        self.dispatch_output = dispatch_output
        self.quant_info = quant_info
        return "captured"


def _backend():
    return MoeRunnerBackend.FLASHINFER_CUTLASS


def _method():
    method = modelopt_quant.ModelOptNvFp4FusedMoEMethod.__new__(
        modelopt_quant.ModelOptNvFp4FusedMoEMethod
    )
    method.quant_config = SimpleNamespace(
        group_size=16,
        use_per_token_activation=False,
    )
    method.enable_flashinfer_trtllm_moe = False
    method._moe_runner_backend = _backend()
    method.moe_runner_config = SimpleNamespace(
        activation="silu",
        apply_router_weight_on_input=False,
        is_gated=True,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        gemm1_alpha=None,
    )
    return method


def _finalization_layer():
    layer = torch.nn.Module()
    layer.num_experts = 4
    layer.num_local_experts = 4
    layer.moe_ep_size = 1
    layer.moe_ep_rank = 0
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        gemm1_alpha=None,
    )
    layer.dispatcher = _Dispatcher()
    layer.inference_moe_w13_interleaved = True
    layer._w13_deinterleaved = False
    layer.register_parameter(
        "w13_weight", _cpu_parameter((4, 256, 32), torch.uint8, pinned=False)
    )
    layer.register_parameter(
        "w2_weight", _cpu_parameter((4, 64, 64), torch.uint8, pinned=False)
    )
    layer.register_parameter(
        "w13_weight_scale",
        _cpu_parameter((4, 256, 4), torch.float8_e4m3fn, pinned=False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        _cpu_parameter((4, 64, 8), torch.float8_e4m3fn, pinned=False),
    )
    layer.register_parameter(
        "w13_blockscale_swizzled",
        _cpu_parameter((4, 256, 4), torch.float8_e4m3fn, pinned=False),
    )
    layer.register_parameter(
        "w2_blockscale_swizzled",
        _cpu_parameter((4, 128, 8), torch.float8_e4m3fn, pinned=False),
    )
    layer.register_parameter(
        "w13_weight_scale_2",
        torch.nn.Parameter(torch.ones((4, 2), device="cuda"), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale_2",
        torch.nn.Parameter(torch.ones(4, device="cuda"), requires_grad=False),
    )
    layer.register_parameter(
        "w13_input_scale",
        torch.nn.Parameter(torch.ones((4, 2), device="cuda"), requires_grad=False),
    )
    layer.register_parameter(
        "w2_input_scale",
        torch.nn.Parameter(torch.ones(4, device="cuda"), requires_grad=False),
    )
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ModelOptNvfp4ExpertStreamTests(unittest.TestCase):
    def test_finalization_keeps_large_tensors_pageable_and_small_scales_on_cuda(self):
        method = _method()
        layer = _finalization_layer()

        with (
            patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}),
            patch.object(modelopt_quant, "get_moe_runner_backend", _backend),
            patch.object(
                modelopt_quant.ModelOptNvFp4FusedMoEMethod,
                "enable_flashinfer_cutlass_moe",
                property(lambda unused_self: True),
            ),
            patch.object(
                modelopt_quant,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
                create=True,
            ),
        ):
            method.process_weights_after_loading(layer)

        for name in (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
        ):
            tensor = getattr(layer, name)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.is_pinned(), name)
        for name in (
            "g1_alphas",
            "g2_alphas",
            "w13_input_scale_quant",
            "w2_input_scale_quant",
        ):
            self.assertEqual(getattr(layer, name).device.type, "cuda", name)
        self.assertIsInstance(layer._nvfp4_expert_streamer, ExpertStreamer)

    def test_apply_remaps_ids_and_builds_compact_cutlass_payload(self):
        method = _method()
        method.runner = _CaptureRunner()
        layer = torch.nn.Module()
        layer.moe_ep_size = 1
        layer.moe_ep_rank = 0
        layer.moe_tp_size = 1
        layer.moe_tp_rank = 0
        layer.w13_weight = _cpu_parameter((4, 2, 2), torch.uint8, 0, pinned=False)
        layer.w2_weight = _cpu_parameter((4, 2, 2), torch.uint8, 32, pinned=False)
        layer.w13_blockscale_swizzled = _cpu_parameter(
            (4, 2, 2), torch.float8_e4m3fn, 1, pinned=False
        )
        layer.w2_blockscale_swizzled = _cpu_parameter(
            (4, 2, 2), torch.float8_e4m3fn, 17, pinned=False
        )
        layer.g1_alphas = torch.nn.Parameter(
            torch.tensor([10, 20, 30, 40], device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        layer.g2_alphas = torch.nn.Parameter(
            torch.tensor([50, 60, 70, 80], device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        layer.w13_input_scale_quant = torch.tensor(0.5, device="cuda")
        layer.w2_input_scale_quant = torch.tensor(0.25, device="cuda")
        names = (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
            "g1_alphas",
            "g2_alphas",
        )
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, names)
        topk_weights = torch.tensor([[0.5, 0.3, 0.2]], device="cuda")
        router_logits = torch.ones((1, 4), device="cuda")
        dispatch = StandardDispatchOutput(
            hidden_states=torch.ones((1, 2), device="cuda", dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=torch.tensor([[3, 1, 3]], device="cuda", dtype=torch.int32),
                router_logits=router_logits,
            ),
        )

        with (
            patch.object(modelopt_quant, "get_moe_runner_backend", _backend),
            patch.object(
                modelopt_quant.ModelOptNvFp4FusedMoEMethod,
                "enable_flashinfer_cutlass_moe",
                property(lambda unused_self: True),
            ),
        ):
            result = method.apply(layer, dispatch)

        self.assertEqual(result, "captured")
        captured_dispatch = method.runner.dispatch_output
        self.assertEqual(captured_dispatch.topk_output.topk_ids.tolist(), [[0, 1, 2]])
        self.assertIs(captured_dispatch.topk_output.topk_weights, topk_weights)
        self.assertIs(captured_dispatch.topk_output.router_logits, router_logits)
        quant_info = method.runner.quant_info
        self.assertEqual(quant_info.w13_weight.shape[0], 3)
        self.assertEqual(quant_info.w2_weight.shape[0], 3)
        self.assertTrue(
            torch.equal(quant_info.w13_weight.cpu(), layer.w13_weight[[3, 1, 3]])
        )
        self.assertIs(quant_info.quant_scales[0], layer.w13_input_scale_quant)
        self.assertEqual(quant_info.quant_scales[1].shape[0], 3)
        self.assertEqual(quant_info.quant_scales[2].tolist(), [40, 20, 40])
        self.assertIs(quant_info.quant_scales[3], layer.w2_input_scale_quant)
        self.assertEqual(quant_info.quant_scales[4].shape[0], 3)
        self.assertEqual(quant_info.quant_scales[5].tolist(), [80, 60, 80])

    def test_finalization_rejects_expert_parallel_streaming(self):
        method = _method()
        layer = _finalization_layer()
        layer.moe_ep_size = 2

        with (
            patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}),
            patch.object(modelopt_quant, "get_moe_runner_backend", _backend),
            patch.object(
                modelopt_quant.ModelOptNvFp4FusedMoEMethod,
                "enable_flashinfer_cutlass_moe",
                property(lambda unused_self: True),
            ),
            patch.object(
                modelopt_quant,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "EP size 1"):
                method.process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
