"""CPU tests for load-time FP8 of BF16 linear layers (``online_fp8``).

Covers module selection, the per-channel quantization error bound, the weight-only
path, the hyper-connection mix, and that the feature is a no-op when its env is unset.
Run with CUDA hidden; nothing here needs a GPU.
"""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import online_fp8
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.online_fp8 import (
    FP8_DTYPE,
    GROUPS_ENV,
    LAYERS_ENV,
    SCHEME_ENV,
    OnlineFp8LinearMethod,
    dequantize_fp8_per_channel,
    fp8_weight_only_linear,
    hc_mix_online_fp8_scheme,
    linear_group_for_prefix,
    online_fp8_or_unquantized,
    online_fp8_selection,
    quantize_fp8_per_channel,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

_ENV_KEYS = (GROUPS_ENV, SCHEME_ENV, LAYERS_ENV)
_SUBNORMAL_HALF_STEP = 2.0**-10


def _env(**values):
    environ = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    environ.update(values)
    return mock.patch.dict(os.environ, environ, clear=True)


def _weights() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(37, 64, generator=generator) * 0.02
    weight[3, 5] = 1.7
    weight[4] = 0.0
    weight[5] = torch.randn(64, generator=generator) * 1e-6
    weight[6, :8] = torch.tensor([448.0, -448.0, 1.0, -1.0, 0.5, 3.0, 1e-3, -2e-4])
    return weight


class TestSelection(unittest.TestCase):
    def test_unset_or_empty_is_off(self):
        self.assertIsNone(online_fp8_selection({}))
        self.assertIsNone(online_fp8_selection({GROUPS_ENV: " "}))

    def test_all_default_scheme_and_layers(self):
        selection = online_fp8_selection(
            {GROUPS_ENV: "all", LAYERS_ENV: "0-2, 40"}
        )
        self.assertEqual(selection.groups, frozenset(online_fp8.GROUP_NAMES))
        self.assertEqual(selection.scheme, "w8a8")
        self.assertEqual(selection.layers, frozenset({0, 1, 2, 40}))
        self.assertTrue(selection.includes("gdn", 40))
        self.assertFalse(selection.includes("gdn", 3))
        self.assertFalse(selection.includes("lm_head", None))

    def test_rejects_bad_values(self):
        for environ in (
            {GROUPS_ENV: "gdn,mlp"},
            {GROUPS_ENV: "gdn", SCHEME_ENV: "w4a8"},
            {GROUPS_ENV: "gdn", LAYERS_ENV: "5-2"},
            {GROUPS_ENV: "gdn", LAYERS_ENV: "x"},
            {GROUPS_ENV: "gdn", LAYERS_ENV: ","},
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    online_fp8_selection(environ)

    def test_linear_group_for_prefix(self):
        cases = {
            "model.language_model.layers.0.linear_attn.in_proj_qkvz": ("gdn", 0),
            "model.layers.36.linear_attn.out_proj": ("gdn", 36),
            "model.layers.3.self_attn.qkv_proj": ("full_attn", 3),
            "model.layers.47.self_attn.o_proj": ("full_attn", 47),
            "model.layers.3.mlp.shared_expert.gate_up_proj": ("shared_expert", 3),
            "model.layers.2.mlp.shared_expert.down_proj": ("shared_expert", 2),
            "lm_head": ("lm_head", None),
            "model.layers.0.linear_attn.in_proj_ba": None,
            "model.layers.0.linear_attn.conv1d": None,
            "model.layers.3.mlp.shared_expert_gate": None,
            "model.layers.3.mlp.gate": None,
            "model.layers.1.ple.key_proj": None,
            "model.layers.3.self_attn.indexer.index_qk_proj": None,
            "mtp.layers.0.self_attn.qkv_proj": None,
            "mtp.layers.0.mlp.shared_expert.gate_up_proj": None,
            "model.shared_head.head": None,
        }
        for prefix, expected in cases.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(linear_group_for_prefix(prefix), expected)

    def test_hc_mix_scheme(self):
        self.assertIsNone(hc_mix_online_fp8_scheme("model.layers.3.self_attn", {}))
        environ = {GROUPS_ENV: "hc_mix", SCHEME_ENV: "w8a16"}
        self.assertEqual(
            hc_mix_online_fp8_scheme("model.layers.3.self_attn", environ), "w8a16"
        )
        self.assertEqual(hc_mix_online_fp8_scheme("model", environ), "w8a16")
        self.assertIsNone(hc_mix_online_fp8_scheme("mtp.layers.0.self_attn", environ))
        self.assertIsNone(
            hc_mix_online_fp8_scheme("model.layers.3.self_attn", {GROUPS_ENV: "gdn"})
        )
        environ[LAYERS_ENV] = "0-2"
        self.assertIsNone(hc_mix_online_fp8_scheme("model.layers.3.self_attn", environ))
        self.assertIsNone(hc_mix_online_fp8_scheme("model", environ))
        self.assertEqual(
            hc_mix_online_fp8_scheme("model.layers.2.linear_attn", environ), "w8a16"
        )


class TestMixedPrecisionDispatch(unittest.TestCase):
    """The ModelOpt MIXED_PRECISION config hands selected BF16 layers to online FP8 only."""

    def setUp(self):
        self.config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": [
                    "lm_head",
                    "model.language_model.layers.0.linear_attn*",
                    "model.language_model.layers.0.mlp.shared_expert*",
                    "model.language_model.layers.3.self_attn*",
                ],
                "quantized_layers": {
                    "model.language_model.layers.0.mlp.experts": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.language_model.layers.7.self_attn.qkv_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
                "packed_modules_mapping": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
                    "in_proj_ba": ["in_proj_b", "in_proj_a"],
                },
            }
        )
        self.linear = ReplicatedLinear.__new__(ReplicatedLinear)
        self.lm_head = ParallelLMHead.__new__(ParallelLMHead)
        self.bf16_prefixes = (
            "model.language_model.layers.0.linear_attn.in_proj_qkvz",
            "model.language_model.layers.0.linear_attn.out_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_up_proj",
            "model.language_model.layers.3.self_attn.qkv_proj",
        )

    def test_flag_off_keeps_unquantized_method(self):
        with _env():
            for prefix in self.bf16_prefixes + (
                "model.language_model.layers.0.linear_attn.in_proj_ba",
            ):
                with self.subTest(prefix=prefix):
                    method = self.config.get_quant_method(self.linear, prefix)
                    self.assertIs(type(method), UnquantizedLinearMethod)
            self.assertIs(
                type(self.config.get_quant_method(self.lm_head, "lm_head")),
                UnquantizedLinearMethod,
            )

    def test_flag_on_selects_groups_and_leaves_the_rest(self):
        with _env(**{GROUPS_ENV: "all"}):
            for prefix in self.bf16_prefixes:
                with self.subTest(prefix=prefix):
                    method = self.config.get_quant_method(self.linear, prefix)
                    self.assertIsInstance(method, OnlineFp8LinearMethod)
            self.assertIsInstance(
                self.config.get_quant_method(self.lm_head, "lm_head"),
                OnlineFp8LinearMethod,
            )
            self.assertIs(
                type(
                    self.config.get_quant_method(
                        self.linear, "model.language_model.layers.0.linear_attn.in_proj_ba"
                    )
                ),
                UnquantizedLinearMethod,
            )
            self.assertIsInstance(
                self.config.get_quant_method(
                    self.linear, "model.language_model.layers.7.self_attn.qkv_proj"
                ),
                ModelOptFp4LinearMethod,
            )

    def test_group_subset(self):
        with _env(**{GROUPS_ENV: "full_attn"}):
            self.assertIsInstance(
                self.config.get_quant_method(
                    self.linear, "model.language_model.layers.3.self_attn.qkv_proj"
                ),
                OnlineFp8LinearMethod,
            )
            self.assertIs(
                type(
                    self.config.get_quant_method(
                        self.linear,
                        "model.language_model.layers.0.linear_attn.in_proj_qkvz",
                    )
                ),
                UnquantizedLinearMethod,
            )

    def test_online_method_is_not_an_unquantized_method(self):
        """GDN fuses in_proj weights only for UnquantizedLinearMethod; FP8 must opt out."""
        self.assertNotIsInstance(OnlineFp8LinearMethod("w8a8"), UnquantizedLinearMethod)
        self.assertIsInstance(online_fp8_or_unquantized("lm_head", {}), UnquantizedLinearMethod)


class TestQuantization(unittest.TestCase):
    def test_round_trip_error_bound(self):
        """E4M3 round-to-nearest: |err| <= |w|/16 above the subnormal range, 2^-10 scale below."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                weight = _weights().to(dtype)
                qweight, scale = quantize_fp8_per_channel(weight)
                self.assertEqual(qweight.dtype, FP8_DTYPE)
                self.assertEqual(scale.shape, (weight.shape[0], 1))
                self.assertEqual(scale.dtype, torch.float32)
                reference = weight.to(torch.float32)
                restored = dequantize_fp8_per_channel(qweight, scale, torch.float32)
                bound = reference.abs() / 16 + scale * _SUBNORMAL_HALF_STEP
                slack = 1e-6 * scale + 1e-12
                self.assertTrue(bool(((restored - reference).abs() <= bound + slack).all()))
                row_max = reference.abs().amax(dim=1)
                restored_max = restored.abs().amax(dim=1)
                torch.testing.assert_close(restored_max, row_max, rtol=1e-6, atol=0.0)
                self.assertTrue(bool((restored[4] == 0).all()))

    def test_bound_rejects_truncation(self):
        """The bound is tight enough that truncating toward zero would violate it."""
        weight = torch.linspace(-1.0, 1.0, 4001).reshape(1, -1)
        _, scale = quantize_fp8_per_channel(weight)
        fp8_max = torch.finfo(FP8_DTYPE).max
        truncated = torch.trunc(weight / scale * 8) / 8 * scale
        bound = weight.abs() / 16 + scale * _SUBNORMAL_HALF_STEP
        self.assertFalse(bool(((truncated - weight).abs() <= bound + 1e-9).all()))
        self.assertEqual(float(fp8_max), 448.0)

    def test_sqnr_on_gaussian_rows(self):
        weight = torch.randn(256, 512, generator=torch.Generator().manual_seed(1))
        qweight, scale = quantize_fp8_per_channel(weight)
        error = dequantize_fp8_per_channel(qweight, scale, torch.float32) - weight
        sqnr_db = 10 * torch.log10(weight.pow(2).sum() / error.pow(2).sum())
        self.assertGreater(float(sqnr_db), 20 * torch.log10(torch.tensor(16.0)).item())

    def test_row_chunking_is_exact(self):
        weight = _weights()
        whole = quantize_fp8_per_channel(weight, row_chunk=1 << 20)
        chunked = quantize_fp8_per_channel(weight, row_chunk=5)
        self.assertTrue(torch.equal(whole[0].view(torch.uint8), chunked[0].view(torch.uint8)))
        self.assertTrue(torch.equal(whole[1], chunked[1]))

    def _loaded_layer(self, scheme: str) -> torch.nn.Linear:
        layer = torch.nn.Linear(64, 37, bias=False)
        layer.quant_method = OnlineFp8LinearMethod(scheme)
        with torch.no_grad():
            layer.weight.copy_(_weights())
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_process_weights_layouts(self):
        reference = _weights()
        w8a8 = self._loaded_layer("w8a8")
        self.assertEqual(w8a8.weight.dtype, FP8_DTYPE)
        self.assertEqual(tuple(w8a8.weight.shape), (64, 37))
        self.assertEqual(w8a8.weight_scale.numel(), w8a8.weight.shape[1])
        w8a16 = self._loaded_layer("w8a16")
        self.assertEqual(tuple(w8a16.weight.shape), (37, 64))
        torch.testing.assert_close(
            dequantize_fp8_per_channel(w8a8.weight.t(), w8a8.weight_scale, torch.float32),
            dequantize_fp8_per_channel(w8a16.weight, w8a16.weight_scale, torch.float32),
        )
        restored = dequantize_fp8_per_channel(w8a16.weight, w8a16.weight_scale, torch.float32)
        self.assertLess(float((restored - reference).abs().max()), 1.7 / 16 + 1e-6)
        w8a16.quant_method.process_weights_after_loading(w8a16)
        self.assertEqual(tuple(w8a16.weight.shape), (37, 64))

    def test_weight_only_apply_matches_dequantized_linear(self):
        layer = self._loaded_layer("w8a16")
        x = torch.randn(5, 64, generator=torch.Generator().manual_seed(3))
        bias = torch.randn(37, generator=torch.Generator().manual_seed(4))
        expected = F.linear(
            x, dequantize_fp8_per_channel(layer.weight, layer.weight_scale, x.dtype), bias
        )
        torch.testing.assert_close(layer.quant_method.apply(layer, x, bias), expected)
        chunked = fp8_weight_only_linear(
            x, layer.weight, layer.weight_scale, bias, row_chunk=4
        )
        torch.testing.assert_close(chunked, expected, rtol=1e-5, atol=1e-6)


class TestHyperConnectionMix(unittest.TestCase):
    def _build(self, scheme):
        from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

        config = HyperConnectionConfig(
            hc_count=4,
            hidden_size=32,
            params_dtype=torch.float32,
            hc_lowrank=8,
            rms_norm_eps=1e-6,
            hc_per_branch_norm=True,
        )
        with mock.patch("torch.cuda.current_device", return_value="cpu"), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            return GatedResidual(config, use_combine=False, online_fp8_scheme=scheme)

    def test_flag_off_leaves_mix_untouched(self):
        module = self._build(None)
        self.assertFalse(module._online_fp8_mix)
        self.assertFalse(hasattr(module.input_mix_weight_down, "quant_method"))
        self.assertFalse(hasattr(module.input_mix_weight_up, "quant_method"))

    def test_fp8_mix_matches_reference_math(self):
        module = self._build("w8a16")
        self.assertTrue(module._online_fp8_mix)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            module.input_mix_weight_down.weight.copy_(torch.randn(8, 128, generator=generator))
            module.input_mix_weight_up.weight.copy_(torch.randn(128, 8, generator=generator))
        down_bf = module.input_mix_weight_down.weight.detach().clone()
        up_bf = module.input_mix_weight_up.weight.detach().clone()
        for linear in (module.input_mix_weight_down, module.input_mix_weight_up):
            linear.quant_method.process_weights_after_loading(linear)
        x = torch.randn(3, 128, generator=generator)

        def reference(down, up):
            gate = torch.sigmoid(F.linear(F.silu(F.linear(x, down) / 4), up))
            return (gate.unflatten(-1, (4, 32)) * x.unflatten(-1, (4, 32))).mean(dim=-2)

        down_q = dequantize_fp8_per_channel(
            module.input_mix_weight_down.weight, module.input_mix_weight_down.weight_scale, x.dtype
        )
        up_q = dequantize_fp8_per_channel(
            module.input_mix_weight_up.weight, module.input_mix_weight_up.weight_scale, x.dtype
        )
        actual = module._mix_online_fp8(x)
        torch.testing.assert_close(actual, reference(down_q, up_q))
        relative = (actual - reference(down_bf, up_bf)).norm() / reference(down_bf, up_bf).norm()
        self.assertLess(float(relative), 0.05)


if __name__ == "__main__":
    unittest.main()
