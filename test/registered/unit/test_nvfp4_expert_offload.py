"""Tests for gated ModelOpt NVFP4 selected-expert CPU offload."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.utils import offloader as offloader_module


OFFLOAD_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
)


class ModelOptNvFp4FusedMoEMethod:
    pass


class FakeFp8Method:
    pass


class FakeExperts(torch.nn.Module):
    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        for name in OFFLOAD_NAMES:
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 4),
                    requires_grad=False,
                ),
            )
        self.register_parameter(
            "w13_weight_scale_2",
            torch.nn.Parameter(torch.ones(2, device="cuda"), requires_grad=False),
        )


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "dense_weight",
            torch.nn.Parameter(torch.ones(2, device="cuda"), requires_grad=False),
        )
        self.decoder_experts = FakeExperts(ModelOptNvFp4FusedMoEMethod())
        self.mtp_experts = FakeExperts(FakeFp8Method())
        self.ple = torch.nn.Linear(2, 2, bias=False, device="cuda")

    def forward(self, value):
        return value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Nvfp4ExpertOffloadTests(unittest.TestCase):
    def setUp(self):
        torch.cuda.empty_cache()

    def test_parameter_selection_excludes_dense_ple_and_fp8_experts(self):
        layer = FakeLayer()

        selected = list(offloader_module._iter_streamed_nvfp4_parameters(layer))

        self.assertEqual(
            selected,
            [getattr(layer.decoder_experts, name) for name in OFFLOAD_NAMES],
        )

    def test_streaming_offloads_only_decoder_nvfp4_tensors_without_wrapper(self):
        layer = FakeLayer()
        target_bytes = sum(
            getattr(layer.decoder_experts, name).numel()
            * getattr(layer.decoder_experts, name).element_size()
            for name in OFFLOAD_NAMES
        )

        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            result = offloader_module.OffloaderV1(target_bytes).maybe_offload_to_cpu(
                layer
            )

        self.assertIs(result, layer)
        self.assertNotIn("forward", layer.__dict__)
        for name in OFFLOAD_NAMES:
            parameter = getattr(layer.decoder_experts, name)
            self.assertEqual(parameter.device.type, "cpu")
            self.assertFalse(parameter.is_pinned())
        self.assertEqual(layer.dense_weight.device.type, "cuda")
        self.assertEqual(layer.ple.weight.device.type, "cuda")
        self.assertEqual(layer.mtp_experts.w13_weight.device.type, "cuda")
        self.assertEqual(layer.decoder_experts.w13_weight_scale_2.device.type, "cuda")

    def test_streaming_budget_is_atomic_for_an_expert_module(self):
        layer = FakeLayer()
        target_bytes = sum(
            getattr(layer.decoder_experts, name).numel()
            * getattr(layer.decoder_experts, name).element_size()
            for name in OFFLOAD_NAMES
        )

        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            with self.assertRaisesRegex(RuntimeError, "complete ModelOpt NVFP4 expert"):
                offloader_module.OffloaderV1(
                    target_bytes - 1
                ).maybe_offload_to_cpu(layer)

        for name in OFFLOAD_NAMES:
            self.assertEqual(
                getattr(layer.decoder_experts, name).device.type,
                "cuda",
            )

    def test_disabled_streaming_preserves_generic_offload(self):
        layer = torch.nn.Linear(2, 2, bias=False, device="cuda")

        with patch.dict(os.environ, {}, clear=True):
            offloader_module.OffloaderV1(1024).maybe_offload_to_cpu(layer)

        self.assertEqual(layer.weight.device.type, "cpu")
        self.assertIn("forward", layer.__dict__)
        output = layer(torch.ones(1, 2, device="cuda"))
        self.assertEqual(output.device.type, "cuda")


class OffloadCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _args(cpu_offload_gb=0, offload_group_size=0):
        return SimpleNamespace(
            ple_offload_embedding=True,
            cpu_offload_gb=cpu_offload_gb,
            offload_group_size=offload_group_size,
            ple_offload_backend=None,
        )

    def test_ple_and_cpu_offload_still_rejected_without_streaming(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ple-offload-embedding"):
                server_args_module.ServerArgs._handle_offload_compatibility(
                    self._args(cpu_offload_gb=1)
                )

    def test_ple_and_cpu_offload_allowed_for_selected_expert_streaming(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            server_args_module.ServerArgs._handle_offload_compatibility(
                self._args(cpu_offload_gb=1)
            )

    def test_ple_and_group_offload_rejected_with_streaming(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            with self.assertRaisesRegex(ValueError, "offload-group-size"):
                server_args_module.ServerArgs._handle_offload_compatibility(
                    self._args(offload_group_size=1)
                )


if __name__ == "__main__":
    unittest.main()
