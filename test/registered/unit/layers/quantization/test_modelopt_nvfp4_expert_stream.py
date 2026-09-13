"""Integration tests for ModelOpt NVFP4 selected-expert streaming."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.utils.offloader import NVFP4_FILE_PARAMETER_NAMES, OffloaderV1


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
        layer_id=7,
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
        layer_id=7,
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
    def _cache_layer(self):
        layer = FusedMoE.__new__(FusedMoE)
        layer.__dict__.update(_finalization_layer().__dict__)
        method = _method()
        method.moe_runner_config.layer_id = 7
        layer.quant_method = method
        layer.scheme = None
        layer.quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4")
        layer.use_triton_kernels = False
        layer.use_flashinfer_trtllm_moe = False
        layer.use_padded_loading = False
        layer.use_presharded_weights = False
        layer._maybe_load_fp8_shared_expert_as_fp4 = lambda **kwargs: False
        shapes = ((4, 256, 64), (4, 128, 64), (4, 256, 8), (4, 128, 8))
        for index, (tag, shape) in enumerate(zip(NVFP4_FILE_PARAMETER_NAMES, shapes)):
            dtype = torch.uint8 if index < 2 else torch.float8_e4m3fn
            values = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
            values = ((values * 17 + values // 7 + 23 * index) % 113).to(torch.uint8)
            setattr(
                layer,
                tag,
                torch.nn.Parameter(values.view(dtype), requires_grad=False),
            )
        layer.w13_blockscale_swizzled = None
        layer.w2_blockscale_swizzled = None
        return layer

    def _finalize(self, layer):
        with (
            patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}),
            patch.object(modelopt_quant, "get_moe_runner_backend", _backend),
            patch.object(
                modelopt_quant,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
        ):
            layer.quant_method.process_weights_after_loading(layer)

    def _bind_cache(self, layer, directory):
        offloader = OffloaderV1(1000000, {"model_path": "/models/deterministic"})
        self.addCleanup(offloader.abort)
        with patch.dict(
            os.environ,
            {
                "SGLANG_MOE_EXPERT_STREAM": "1",
                "SGLANG_MOE_EXPERT_FILE_DIR": directory,
            },
        ):
            offloader.maybe_offload_to_cpu(layer)
        return offloader

    def _load_cache(self, layer, payload):
        for tag, source in payload.items():
            for expert_id in range(4):
                shards = ("w1", "w3") if tag.startswith("w13") else ("w2",)
                chunks = source[expert_id].chunk(len(shards), dim=0)
                for shard, chunk in zip(shards, chunks):
                    layer._weight_loader_impl(
                        getattr(layer, tag), chunk, tag, shard, expert_id
                    )

    def test_file_runtime_layout_matches_anonymous_and_warm_gather(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._cache_layer()
            payload = {
                tag: getattr(cold, tag).detach().clone()
                for tag in NVFP4_FILE_PARAMETER_NAMES
            }
            offloader = self._bind_cache(cold, directory)
            self._load_cache(cold, payload)
            anonymous = self._cache_layer()
            self._load_cache(anonymous, payload)
            raw_w13 = cold.w13_weight.detach().clone()
            self._finalize(anonymous)
            self._finalize(cold)
            self.assertEqual(cold._nvfp4_expert_streamer.layer_id, 7)
            self.assertEqual(cold.layer_id, 7)
            self.assertEqual(cold._nvfp4_file_source_bytes_per_expert, 27648)
            self.assertFalse(hasattr(anonymous, "_nvfp4_file_source_bytes_per_expert"))
            self.assertFalse(torch.equal(raw_w13, cold.w13_weight))
            self.assertTrue(
                torch.equal(
                    cold.w13_weight,
                    torch.cat((raw_w13[:, ::2], raw_w13[:, 1::2]), dim=1),
                )
            )
            group = cold.w13_weight._sglang_file_cache_group
            runtime_names = (
                "w13_weight",
                "w2_weight",
                "w13_blockscale_swizzled",
                "w2_blockscale_swizzled",
            )
            for tag, name in zip(NVFP4_FILE_PARAMETER_NAMES, runtime_names):
                actual = getattr(cold, name)
                self.assertEqual(actual.data_ptr(), group.tensors[tag].data_ptr())
                self.assertTrue(
                    torch.equal(
                        actual.view(torch.uint8),
                        getattr(anonymous, name).view(torch.uint8),
                    ),
                    name,
                )
                disk = torch.from_file(
                    group.paths[tag],
                    shared=True,
                    size=actual.numel(),
                    dtype=torch.uint8,
                )
                self.assertTrue(
                    torch.equal(disk, actual.view(torch.uint8).reshape(-1)),
                    name,
                )
            offloader.post_init()
            manifest = json.loads(Path(group.manifest_path).read_text())
            self.assertIn("runtime_layout", manifest["cache_identity"])
            self.assertEqual(len(manifest["cache_identity"]["runtime_layout"]), 4)
            for entry, tag in zip(
                manifest["cache_identity"]["runtime_layout"], NVFP4_FILE_PARAMETER_NAMES
            ):
                tensor = getattr(cold, tag)
                self.assertEqual(entry["shape"], list(tensor.shape))
                self.assertEqual(entry["stride"], list(tensor.stride()))
                self.assertEqual(
                    entry["nbytes"], tensor.numel() * tensor.element_size()
                )
            ids = torch.tensor([[3, 1, 3]], device="cuda", dtype=torch.int32)

            def gather(layer):
                remapped, tensors = layer._nvfp4_expert_streamer.gather(ids)
                return {
                    name: value[remapped.long()].cpu().view(torch.uint8)
                    for name, value in tensors.items()
                }, remapped.cpu().clone()

            cold_payload, cold_ids = gather(cold)
            anonymous_payload, anonymous_ids = gather(anonymous)
            warm = self._cache_layer()
            warm_offloader = self._bind_cache(warm, directory)
            self.assertTrue(warm.w13_weight._sglang_file_cache_hit)
            with patch.object(
                FusedMoE,
                "_weight_loader_impl_uncached",
                side_effect=AssertionError("large loader copy"),
            ):
                self._load_cache(warm, payload)
            with ExitStack() as stack:
                for name in ("deinterleave_w13", "swizzle_blockscale"):
                    stack.enter_context(
                        patch.object(
                            modelopt_quant,
                            name,
                            side_effect=AssertionError("warm transform"),
                        )
                    )
                self._finalize(warm)
            self.assertTrue(warm._w13_deinterleaved)
            self.assertIs(warm.w13_blockscale_swizzled, warm.w13_weight_scale)
            self.assertIs(warm.w2_blockscale_swizzled, warm.w2_weight_scale)
            warm_payload, warm_ids = gather(warm)
            for name in cold_payload:
                self.assertTrue(
                    torch.equal(cold_payload[name], warm_payload[name]), name
                )
                self.assertTrue(
                    torch.equal(cold_payload[name], anonymous_payload[name]),
                    name,
                )
            self.assertTrue(torch.equal(cold_ids, warm_ids))
            self.assertTrue(torch.equal(cold_ids, anonymous_ids))
            warm_offloader.post_init()

    def test_warm_start_loads_tiny_scales_and_recomputes_alphas(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._cache_layer()
            payload = {
                tag: getattr(cold, tag).detach().clone()
                for tag in NVFP4_FILE_PARAMETER_NAMES
            }
            offloader = self._bind_cache(cold, directory)
            self._load_cache(cold, payload)
            self._finalize(cold)
            offloader.post_init()
            warm = self._cache_layer()
            warm_offloader = self._bind_cache(warm, directory)
            for expert in range(4):
                for name, value, shards in (
                    ("w13_input_scale", 3.0, ("w1", "w3")),
                    ("w2_input_scale", 5.0, ("w2",)),
                    ("w13_weight_scale_2", 2.0 + expert, ("w1", "w3")),
                    ("w2_weight_scale_2", 7.0 + expert, ("w2",)),
                ):
                    for shard in shards:
                        warm._weight_loader_impl(
                            getattr(warm, name),
                            torch.tensor(value),
                            name,
                            shard,
                            expert,
                        )
            with (
                patch.object(
                    modelopt_quant,
                    "deinterleave_w13",
                    side_effect=AssertionError("warm deinterleave"),
                ),
                patch.object(
                    modelopt_quant,
                    "swizzle_blockscale",
                    side_effect=AssertionError("warm swizzle"),
                ),
            ):
                self._finalize(warm)
            self.assertEqual(warm.g1_alphas.tolist(), [6, 9, 12, 15])
            self.assertEqual(warm.g2_alphas.tolist(), [35, 40, 45, 50])
            warm_offloader.post_init()

    def test_file_runtime_rejects_swizzle_padding_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            layer = self._cache_layer()
            layer.w2_weight_scale.data = layer.w2_weight_scale.data[:, :64].contiguous()
            self._bind_cache(layer, directory)
            with self.assertRaisesRegex(RuntimeError, "byte length"):
                self._finalize(layer)

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


class TestStreamingCompatibilityGuards(unittest.TestCase):
    def test_attachment_attributes_only_verified_final_file_views(self):
        from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS

        method = _method()
        layer = torch.nn.Module()
        layer.layer_id = 19
        layer.moe_tp_size = layer.moe_ep_size = 1
        tensors = {}
        specs = []
        for name, tag in zip(NVFP4_STREAM_TENSORS[:4], NVFP4_FILE_PARAMETER_NAMES):
            tensor = torch.zeros((4, 8), dtype=torch.uint8)
            setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))
            tensors[tag] = tensor
            specs.append(
                SimpleNamespace(tag=tag, shape=(4, 8), stride=(8, 1), dtype=torch.uint8)
            )
        layer.g1_alphas = layer.g2_alphas = torch.ones(4)
        layer.w13_weight._sglang_file_cache_group = SimpleNamespace(
            tensors=tensors,
            _nvfp4_runtime_specs=specs,
        )
        with patch.object(
            modelopt_quant, "get_moe_runner_backend", _backend
        ), patch.object(
            modelopt_quant, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ):
            method._attach_expert_streamer(layer)
            self.assertEqual(layer._nvfp4_expert_streamer.layer_id, 19)
            self.assertEqual(layer._nvfp4_file_source_bytes_per_expert, 32)
            layer.w2_blockscale_swizzled.data = layer.w2_blockscale_swizzled.clone()
            method._attach_expert_streamer(layer)
            self.assertFalse(hasattr(layer, "_nvfp4_file_source_bytes_per_expert"))

    def test_create_streaming_weights_does_not_allocate_derived_scales(self):
        method = _method()
        method.quant_config.is_checkpoint_nvfp4_serialized = True
        method.quant_config.get_name = lambda: "modelopt_fp4"
        layer = torch.nn.Module()
        layer.num_local_experts = 4
        layer.num_experts = 4
        layer.moe_runner_config = method.moe_runner_config
        with (
            patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}),
            patch.object(modelopt_quant, "get_moe_runner_backend", _backend),
            patch.object(
                modelopt_quant,
                "swizzle_blockscale",
                side_effect=AssertionError("derived scale allocation"),
            ),
        ):
            method.create_weights(layer, 4, 128, 128, torch.bfloat16)
        self.assertIsNone(layer.w13_blockscale_swizzled)
        self.assertIsNone(layer.w2_blockscale_swizzled)
        self.assertEqual(layer.w13_weight_scale.shape, (4, 256, 8))

    def test_backend_tp_ep_and_a2a_are_rejected_before_attaching(self):
        cases = (
            (False, 1, 1, True, "flashinfer_cutlass"),
            (True, 2, 1, True, "TP size 1"),
            (True, 1, 2, True, "EP size 1"),
            (True, 1, 1, False, "moe-a2a-backend none"),
        )
        for cutlass, tp_size, ep_size, no_a2a, message in cases:
            with self.subTest(message=message):
                backend = SimpleNamespace(is_flashinfer_cutlass=lambda: cutlass)
                method = modelopt_quant.ModelOptNvFp4FusedMoEMethod.__new__(
                    modelopt_quant.ModelOptNvFp4FusedMoEMethod
                )
                method._moe_runner_backend = backend
                layer = SimpleNamespace(moe_tp_size=tp_size, moe_ep_size=ep_size)
                with (
                    patch.object(
                        modelopt_quant, "get_moe_runner_backend", return_value=backend
                    ),
                    patch.object(
                        modelopt_quant,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_none=lambda: no_a2a),
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    method._attach_expert_streamer(layer)


if __name__ == "__main__":
    unittest.main()
