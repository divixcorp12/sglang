"""Regressions for the NVIDIA Qwen checkpoint port to current upstream APIs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.fields.exec_ import ExecOffload
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.srt.models.qwen3_5_mtp import _mtp_quant_config
from sglang.srt.models.qwen4_exp import _ple_table_is_fp8


class TestNvfp4MainPort(unittest.TestCase):
    def test_nvidia_checkpoint_resolves_text_ple_and_block_fp8_mtp(self):
        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": {},
                "quantized_layers": {
                    "model.language_model.layers.3.mlp.experts": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                        "quant_algo": "FP8",
                    },
                    "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES"},
                    "mtp.layers.0.linear": {"quant_algo": "FP8_BLOCK_SCALES"},
                },
            }
        )
        self.assertEqual(config.exclude_modules, [])
        self.assertEqual(
            config.resolve_quant_algo("model.layers.3.mlp.experts"), "NVFP4"
        )
        self.assertTrue(
            _ple_table_is_fp8(
                SimpleNamespace(ple_embedding_dtype=None),
                config,
                "model.layers.1.ple.ple_embedding.ngram_embedding",
            )
        )
        self.assertIs(_mtp_quant_config(config), config)
        method = config.get_quant_method(
            FusedMoE.__new__(FusedMoE), "mtp.layers.0.mlp.experts"
        )
        self.assertIsInstance(method, Fp8MoEMethod)
        self.assertEqual(method.quant_config.weight_block_size, [128, 128])
        linear = config.get_quant_method(
            ReplicatedLinear.__new__(ReplicatedLinear), "mtp.layers.0.linear"
        )
        self.assertIsInstance(linear, Fp8LinearMethod)
        self.assertIs(linear.quant_config, config.fp8_block_config)

    def test_unquantized_mtp_stays_unquantized(self):
        config = SimpleNamespace(
            get_name=lambda: "modelopt_mixed",
            quantized_layers={"model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}},
        )
        self.assertIsNone(_mtp_quant_config(config))

    def test_file_backend_fields_live_in_exec_offload(self):
        self.assertEqual(ExecOffload().ple_offload_backend, "pinned")
        fields = ExecOffload(ple_offload_backend="file", ple_offload_dir="/tmp/ple")
        self.assertEqual(fields.ple_offload_dir, "/tmp/ple")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_post_load_stages_small_state_and_preserves_pageable_experts(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "expert", torch.nn.Parameter(torch.ones(2, 4), requires_grad=False)
        )
        layer.expert._sglang_skip_device_loading = True
        layer.register_buffer("small_scale", torch.ones(1))
        original_ptr = layer.expert.data_ptr()
        with stage_module_for_post_load(layer, torch.device("cuda:0"), pin_memory=True):
            self.assertEqual(layer.expert.device.type, "cpu")
            self.assertFalse(layer.expert.is_pinned())
            self.assertEqual(layer.small_scale.device.type, "cuda")
            layer.expert.add_(1)
        self.assertEqual(layer.expert.data_ptr(), original_ptr)
        self.assertFalse(layer.expert.is_pinned())
        self.assertEqual(layer.small_scale.device.type, "cpu")
        self.assertTrue(torch.equal(layer.expert, torch.full((2, 4), 2.0)))


if __name__ == "__main__":
    unittest.main()
