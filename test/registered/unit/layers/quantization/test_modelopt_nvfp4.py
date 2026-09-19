import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.utils.offloader import _iter_streamed_nvfp4_parameters
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))

    def test_missing_input_scale_defaults_to_one_and_checkpoint_overwrites(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        layer = nn.Module()
        ModelOptFp4LinearMethod(config).create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[16],
            input_size=16,
            output_size=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )

        torch.testing.assert_close(layer.input_scale, torch.ones(1))
        default_weight_loader(layer.input_scale, torch.tensor(0.25))
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.25]))

    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.envs."
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get",
        return_value=True,
    )
    def test_modelopt_fp4_per_token_activation_contract(self, _):
        # Serialized ModelOpt FP4 retains the existing environment-controlled
        # per-token activation path.
        serialized_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        # Online modelopt_fp4 always uses per-tensor activation scaling, even
        # when the serialized-checkpoint environment switch is enabled.
        online_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=False,
            group_size=16,
        )

        self.assertTrue(serialized_config.use_per_token_activation)
        self.assertFalse(online_config.use_per_token_activation)
        # nvfp4_online is the public interface for online per-token scaling.
        with self.assertRaisesRegex(ValueError, "Use nvfp4_online"):
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=16,
                use_per_token_activation=True,
            )


_DRAFT_EXPERTS = "mtp.layers.0.mlp.experts"


@patch(
    "sglang.srt.layers.quantization.modelopt_quant.get_platform",
    return_value=SimpleNamespace(is_blackwell=True),
)
class TestDraftMoeNvfp4Requant(CustomTestCase):
    """FP8-block draft experts load as NVFP4 only when asked, and only in the draft."""

    def setUp(self):
        self.config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    _DRAFT_EXPERTS: {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128},
                },
            }
        )
        self.moe = FusedMoE.__new__(FusedMoE)

    def test_flag_on_in_draft_scope_requantizes_with_per_tensor_activation(self, _):
        with envs.SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT.override(True):
            with draft_model_build_scope():
                method = self.config.get_quant_method(self.moe, _DRAFT_EXPERTS)
        self.assertIsInstance(method, ModelOptNvFp4OnlineFusedMoEMethod)
        self.assertTrue(method.quant_config.is_checkpoint_fp8_serialized)
        self.assertEqual(method.quant_config.weight_block_size, [128, 128])
        self.assertFalse(method.quant_config.use_per_token_activation)

    def test_requantized_draft_experts_stay_on_the_gpu(self, _):
        with envs.SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT.override(True):
            with draft_model_build_scope():
                method = self.config.get_quant_method(self.moe, _DRAFT_EXPERTS)
        module = nn.Module()
        module.quant_method = method
        self.assertEqual(list(_iter_streamed_nvfp4_parameters(module)), [])

    def test_flag_on_outside_draft_scope_keeps_fp8(self, _):
        with envs.SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT.override(True):
            method = self.config.get_quant_method(self.moe, _DRAFT_EXPERTS)
        self.assertIs(type(method), Fp8MoEMethod)

    def test_flag_off_keeps_fp8(self, _):
        with envs.SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT.override(False):
            with draft_model_build_scope():
                method = self.config.get_quant_method(self.moe, _DRAFT_EXPERTS)
        self.assertIs(type(method), Fp8MoEMethod)


if __name__ == "__main__":
    unittest.main()
