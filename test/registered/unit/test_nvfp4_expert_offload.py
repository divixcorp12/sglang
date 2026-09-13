"""Tests for gated ModelOpt NVFP4 selected-expert CPU offload."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import memory_hook
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    ModelOptNvFp4FusedMoEMethod as RealModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    _checkpoint_cache_identity,
)
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
            [
                (layer.decoder_experts, name, getattr(layer.decoder_experts, name))
                for name in OFFLOAD_NAMES
            ],
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
            self.assertTrue(parameter._sglang_skip_device_loading)
        self.assertEqual(layer.dense_weight.device.type, "cuda")
        self.assertEqual(layer.ple.weight.device.type, "cuda")
        self.assertEqual(layer.mtp_experts.w13_weight.device.type, "cuda")
        self.assertEqual(layer.decoder_experts.w13_weight_scale_2.device.type, "cuda")

    def test_streamed_experts_stay_on_cpu_during_post_load_processing(self):
        from sglang.srt.model_loader.loader import device_loading_context

        layer = FakeLayer()
        target_bytes = sum(
            getattr(layer.decoder_experts, name).numel()
            * getattr(layer.decoder_experts, name).element_size()
            for name in OFFLOAD_NAMES
        )
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            offloader_module.OffloaderV1(target_bytes).maybe_offload_to_cpu(layer)

        with device_loading_context(layer.decoder_experts, torch.device("cuda")):
            for name in OFFLOAD_NAMES:
                self.assertEqual(
                    getattr(layer.decoder_experts, name).device.type,
                    "cpu",
                )

    def test_streaming_budget_is_atomic_for_an_expert_module(self):
        layer = FakeLayer()
        target_bytes = sum(
            getattr(layer.decoder_experts, name).numel()
            * getattr(layer.decoder_experts, name).element_size()
            for name in OFFLOAD_NAMES
        )

        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            with self.assertRaisesRegex(RuntimeError, "complete ModelOpt NVFP4 expert"):
                offloader_module.OffloaderV1(target_bytes - 1).maybe_offload_to_cpu(
                    layer
                )

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


class FileExpertOffloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        environment = patch.dict(
            os.environ,
            {
                "SGLANG_MOE_EXPERT_STREAM": "1",
                "SGLANG_MOE_EXPERT_FILE_DIR": self.directory.name,
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _layer(self, layer_id=7):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.quant_method = RealModelOptNvFp4FusedMoEMethod.__new__(
            RealModelOptNvFp4FusedMoEMethod
        )
        layer.moe_runner_config = SimpleNamespace(layer_id=layer_id, is_gated=True)
        layer.quant_method.moe_runner_config = layer.moe_runner_config
        layer.scheme = None
        layer.quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4")
        layer.moe_tp_rank = 0
        layer.moe_tp_size = 1
        layer.moe_ep_rank = 0
        layer.moe_ep_size = 1
        layer.num_local_experts = 2
        layer.use_triton_kernels = False
        layer.use_flashinfer_trtllm_moe = False
        layer.use_padded_loading = False
        layer.use_presharded_weights = False
        for name in OFFLOAD_NAMES:
            layer.register_parameter(
                name,
                torch.nn.Parameter(torch.full((2, 4, 4), 99.0), requires_grad=False),
            )
        layer.register_parameter(
            "w13_weight_scale_2",
            torch.nn.Parameter(torch.ones(2, 2), requires_grad=False),
        )
        layer.register_parameter(
            "w13_input_scale", torch.nn.Parameter(torch.ones(2), requires_grad=False)
        )
        layer._maybe_load_fp8_shared_expert_as_fp4 = lambda **kwargs: False
        return layer

    def _bind(self, layer=None):
        layer = layer if layer is not None else self._layer()
        offloader = offloader_module.OffloaderV1(
            100000,
            checkpoint_cache_identity={
                "model_path": "/models/test",
                "commit_hash": "abc",
            },
        )
        self.addCleanup(offloader.abort)
        offloader.maybe_offload_to_cpu(layer)
        return offloader, layer

    def _load_all(self, layer, omit=None):
        for tag in OFFLOAD_NAMES[:4]:
            for expert_id in range(2):
                for shard_id in ("w1", "w3") if tag.startswith("w13") else ("w2",):
                    if (tag, expert_id, shard_id) == omit:
                        continue
                    shape = (2, 4) if shard_id in ("w1", "w3") else (4, 4)
                    layer._weight_loader_impl(
                        getattr(layer, tag),
                        torch.full(shape, 3.0),
                        tag,
                        shard_id,
                        expert_id,
                    )

    def test_file_binding_does_not_copy_initial_values_or_move_tiny_scales(self):
        layer = self._layer()
        initial_tiny = layer.w13_weight_scale_2.data_ptr()
        initial_swizzled = layer.w13_blockscale_swizzled.data_ptr()
        with patch.object(
            torch.Tensor, "copy_", side_effect=AssertionError("initial copy")
        ):
            offloader, layer = self._bind(layer)
        for tag in OFFLOAD_NAMES[:4]:
            parameter = getattr(layer, tag)
            self.assertTrue(parameter._sglang_skip_device_loading)
            self.assertFalse(parameter._sglang_file_cache_hit)
            self.assertEqual(parameter._sglang_file_cache_tag, tag)
            self.assertEqual(parameter.count_nonzero().item(), 0)
        self.assertEqual(layer.w13_weight_scale_2.data_ptr(), initial_tiny)
        self.assertEqual(layer.w13_blockscale_swizzled.data_ptr(), initial_swizzled)
        self.assertFalse(hasattr(layer.w13_input_scale, "_sglang_file_cache_group"))
        group = layer.w13_weight._sglang_file_cache_group
        self.assertEqual(set(group.tensors), set(OFFLOAD_NAMES[:4]))
        self.assertFalse(Path(group.manifest_path).exists())

    def test_cold_load_publishes_and_warm_hit_closes_without_republishing(self):
        offloader, layer = self._bind()
        self._load_all(layer)
        group = layer.w13_weight._sglang_file_cache_group
        self.assertFalse(Path(group.manifest_path).exists())
        self.assertTrue(torch.equal(layer.w13_weight, torch.full((2, 4, 4), 3.0)))
        layer.w13_weight.data.add_(1)
        offloader.finalize_expert_files(layer)
        offloader.post_init()
        self.assertTrue(Path(group.manifest_path).exists())
        warm, warm_layer = self._bind()
        self.assertTrue(warm_layer.w13_weight._sglang_file_cache_hit)
        self._load_all(warm_layer)
        self.assertTrue(torch.equal(warm_layer.w13_weight, torch.full((2, 4, 4), 4.0)))
        warm_layer._weight_loader_impl(
            warm_layer.w13_input_scale, torch.tensor(8.0), "input_scale", "w1", 0
        )
        self.assertEqual(warm_layer.w13_input_scale[0].item(), 8)
        warm_layer._weight_loader_impl(
            warm_layer.w13_weight_scale_2, torch.tensor(9.0), "weight_scale_2", "w3", 1
        )
        self.assertEqual(warm_layer.w13_weight_scale_2[1, 1].item(), 9)
        warm_group = warm_layer.w13_weight._sglang_file_cache_group
        with patch.object(
            warm_group,
            "complete",
            side_effect=AssertionError("warm hit must not be republished"),
        ):
            warm.post_init()
        self.assertIsNone(warm_group._lock)

    def test_each_missing_logical_shard_aborts_all_groups(self):
        for tag, expert_id, shard_id in (
            ("w13_weight", 1, "w3"),
            ("w13_weight_scale", 0, "w1"),
            ("w2_weight", 0, "w2"),
            ("w2_weight_scale", 1, "w2"),
        ):
            with self.subTest(tag=tag, shard=shard_id):
                offloader, layer = self._bind()
                self._load_all(layer, omit=(tag, expert_id, shard_id))
                second = self._layer(layer_id=8)
                offloader.maybe_offload_to_cpu(second)
                self._load_all(second)
                groups = [
                    layer.w13_weight._sglang_file_cache_group,
                    second.w13_weight._sglang_file_cache_group,
                ]
                with self.assertRaisesRegex(RuntimeError, "coverage"):
                    offloader.post_init()
                for group in groups:
                    self.assertFalse(Path(group.manifest_path).exists())
                    self.assertIsNone(group._lock)

    def test_fused_loader_records_both_w13_shards(self):
        offloader, layer = self._bind()
        for tag in OFFLOAD_NAMES[:4]:
            if tag.startswith("w13"):
                layer.weight_loader_fused(
                    getattr(layer, tag), torch.full((2, 4, 4), 5.0), tag, "w13"
                )
            else:
                for expert_id in range(2):
                    layer._weight_loader_impl(
                        getattr(layer, tag),
                        torch.full((4, 4), 5.0),
                        tag,
                        "w2",
                        expert_id,
                    )
        offloader.finalize_expert_files(layer)
        offloader.post_init()
        self.assertTrue(
            Path(layer.w13_weight._sglang_file_cache_group.manifest_path).exists()
        )
        self.assertEqual(layer.w13_weight.sum().item(), 160)

    def test_fused_loader_rejects_broadcast_expert_coverage(self):
        _, layer = self._bind()
        with self.assertRaisesRegex(ValueError, "expert"):
            layer.weight_loader_fused(
                layer.w13_weight, torch.ones(1, 4, 4), "w13_weight", "w13"
            )
        self.assertIsNone(layer.w13_weight._sglang_file_cache_group._lock)

    def test_empty_directory_keeps_anonymous_cpu_offload(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_FILE_DIR": ""}):
            _, layer = self._bind()
        for tag in OFFLOAD_NAMES:
            parameter = getattr(layer, tag)
            self.assertFalse(hasattr(parameter, "_sglang_file_cache_group"))
            self.assertEqual(parameter.sum().item(), 3168)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_file_directory_does_not_enable_streaming(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "0"}):
            _, layer = self._bind()
        self.assertFalse(hasattr(layer.w13_weight, "_sglang_file_cache_group"))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_optional_swizzled_parameters_are_not_required_in_file_mode(self):
        layer = self._layer()
        layer.w13_blockscale_swizzled = None
        layer.w2_blockscale_swizzled = None
        _, layer = self._bind(layer)
        self.assertFalse(layer.w13_weight._sglang_file_cache_hit)

    def test_anonymous_streaming_accepts_absent_derived_placeholders(self):
        layer = self._layer()
        layer.w13_blockscale_swizzled = None
        layer.w2_blockscale_swizzled = None
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_FILE_DIR": ""}):
            _, layer = self._bind(layer)
        self.assertTrue(layer.w13_weight._sglang_skip_device_loading)
        self.assertIsNone(layer.w13_blockscale_swizzled)

    def test_checkpoint_coverage_cannot_publish_before_runtime_finalization(
        self,
    ):
        offloader, layer = self._bind()
        self._load_all(layer)
        with self.assertRaisesRegex(RuntimeError, "runtime layout"):
            offloader.post_init()
        self.assertFalse(
            Path(layer.w13_weight._sglang_file_cache_group.manifest_path).exists()
        )

    def test_runtime_finalization_rejects_anonymous_replacement(self):
        offloader, layer = self._bind()
        self._load_all(layer)
        layer.w13_weight.data = layer.w13_weight.data.clone()
        with self.assertRaisesRegex(RuntimeError, "runtime layout"):
            offloader.finalize_expert_files(layer)
        with self.assertRaisesRegex(RuntimeError, "runtime layout"):
            offloader.post_init()

    def test_changed_runtime_manifest_is_a_cold_miss(self):
        offloader, layer = self._bind()
        self._load_all(layer)
        offloader.finalize_expert_files(layer)
        offloader.post_init()
        path = Path(layer.w13_weight._sglang_file_cache_group.manifest_path)
        manifest = json.loads(path.read_text())
        manifest["cache_identity"]["runtime_layout"][0]["stride"] = [1, 2, 3]
        path.write_text(json.dumps(manifest))
        _, reloaded = self._bind()
        self.assertFalse(reloaded.w13_weight._sglang_file_cache_hit)
        self.assertEqual(reloaded.w13_weight.count_nonzero().item(), 0)

    def test_interleaved_checkpoint_layout_uses_a_distinct_cache(self):
        offloader, layer = self._bind()
        self._load_all(layer)
        offloader.finalize_expert_files(layer)
        offloader.post_init()
        interleaved = self._layer()
        interleaved.inference_moe_w13_interleaved = True
        _, interleaved = self._bind(interleaved)
        self.assertFalse(interleaved.w13_weight._sglang_file_cache_hit)

    def test_invalid_logical_shard_does_not_publish_coverage(self):
        offloader, layer = self._bind()
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            layer._weight_loader_impl(
                layer.w13_weight, torch.zeros(4, 4), "w13_weight", "w2", 0
            )
        group = layer.w13_weight._sglang_file_cache_group
        self.assertIsNone(group._lock)
        self.assertFalse(Path(group.manifest_path).exists())

    def test_duplicate_layer_binding_releases_existing_lock(self):
        offloader, layer = self._bind()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            offloader.maybe_offload_to_cpu(self._layer())
        self.assertIsNone(layer.w13_weight._sglang_file_cache_group._lock)

    def test_loader_failure_aborts_every_group(self):
        offloader, layer = self._bind()
        second = self._layer(layer_id=8)
        offloader.maybe_offload_to_cpu(second)
        with self.assertRaises((RuntimeError, ValueError)):
            layer._weight_loader_impl(
                layer.w13_weight, torch.zeros(9, 9), "w13_weight", "w1", 0
            )
        for current in (layer, second):
            group = current.w13_weight._sglang_file_cache_group
            self.assertIsNone(group._lock)
            self.assertFalse(Path(group.manifest_path).exists())

    def test_model_load_or_postprocessing_failure_aborts_file_cache(self):
        for phase in ("load", "postprocess"):
            with self.subTest(phase=phase):
                offloader, layer = self._bind()

                def fail_load(**kwargs):
                    if phase == "postprocess":
                        self._load_all(layer)
                    raise RuntimeError(phase)

                with (
                    patch.object(load_model_utils, "monkey_patch_vllm_parallel_state"),
                    patch.object(
                        load_model_utils,
                        "get_exec",
                        return_value=SimpleNamespace(
                            offload=SimpleNamespace(ple_offload_embedding=False),
                            features=SimpleNamespace(
                                enable_weights_cpu_backup=False,
                                enable_draft_weights_cpu_backup=False,
                            ),
                        ),
                    ),
                    patch.object(
                        load_model_utils,
                        "get_model",
                        return_value=SimpleNamespace(
                            weight_cache_mode="off",
                            is_startup_weight_load_overlap=False,
                        ),
                    ),
                    patch.object(
                        load_model_utils,
                        "get_model_loader",
                        return_value=SimpleNamespace(load_model=fail_load),
                    ),
                    patch.object(offloader_module, "_instance", offloader),
                ):
                    with self.assertRaisesRegex(RuntimeError, phase):
                        load_model_utils.load_model_with_memory_saver(
                            model_config=SimpleNamespace(
                                hf_config=SimpleNamespace(architectures=[])
                            ),
                            load_config=SimpleNamespace(),
                            device="cpu",
                            gpu_id=0,
                            memory_saver_adapter=SimpleNamespace(
                                region=lambda *args, **kwargs: nullcontext()
                            ),
                            is_draft_worker=False,
                        )
                group = layer.w13_weight._sglang_file_cache_group
                self.assertIsNone(group._lock)
                self.assertFalse(Path(group.manifest_path).exists())


class OffloaderIdentityTests(unittest.TestCase):
    @staticmethod
    def _exec_config(cpu_offload_gb=1):
        return SimpleNamespace(
            offload=SimpleNamespace(
                cpu_offload_gb=cpu_offload_gb,
                offload_group_size=0,
                offload_num_in_group=1,
                offload_prefetch_step=1,
                offload_mode="cpu",
            )
        )

    def test_factory_passes_immutable_checkpoint_identity_to_v1(self):
        model_config = SimpleNamespace(
            model_path="~/models/checkpoint",
            revision="requested-revision",
            hf_config=SimpleNamespace(_commit_hash="resolved-commit"),
            hf_text_config=SimpleNamespace(),
        )
        server_args = SimpleNamespace(
            model_path="ignored-model", revision="ignored-revision"
        )
        expected = _checkpoint_cache_identity(
            model_config=model_config, server_args=server_args
        )

        with patch.object(
            offloader_module, "get_exec", return_value=self._exec_config()
        ):
            offloader = offloader_module.create_offloader_from_server_args(
                server_args=server_args,
                dp_rank=0,
                model_config=model_config,
            )

        self.assertIsInstance(offloader, offloader_module.OffloaderV1)
        self.assertEqual(offloader.checkpoint_cache_identity, expected)
        self.assertIsNot(offloader.checkpoint_cache_identity, expected)
        with self.assertRaises(TypeError):
            offloader.checkpoint_cache_identity["revision"] = "changed"

    def test_factory_keeps_model_config_optional(self):
        with patch.object(
            offloader_module, "get_exec", return_value=self._exec_config()
        ):
            offloader = offloader_module.create_offloader_from_server_args(
                server_args=SimpleNamespace(), dp_rank=0
            )

        self.assertIsInstance(offloader, offloader_module.OffloaderV1)
        self.assertEqual(offloader.checkpoint_cache_identity, {})

    def test_v1_copies_checkpoint_identity_before_storing_it(self):
        source_identity = {
            "model_path": "/models/checkpoint",
            "revision": "requested-revision",
            "commit_hash": "resolved-commit",
        }

        offloader = offloader_module.OffloaderV1(
            1024, checkpoint_cache_identity=source_identity
        )
        source_identity["revision"] = "changed"

        self.assertEqual(
            offloader.checkpoint_cache_identity["revision"], "requested-revision"
        )
        with self.assertRaises(TypeError):
            offloader.checkpoint_cache_identity["revision"] = "changed"


class OffloadCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.resolving_view = patch.object(
            memory_hook, "resolving_view", lambda args: args
        )
        self.resolving_view.start()
        self.addCleanup(self.resolving_view.stop)

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
                memory_hook.handle_offload_compatibility(self._args(cpu_offload_gb=1))

    def test_ple_and_cpu_offload_allowed_for_selected_expert_streaming(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            memory_hook.handle_offload_compatibility(self._args(cpu_offload_gb=1))

    def test_ple_and_group_offload_rejected_with_streaming(self):
        with patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"}):
            with self.assertRaisesRegex(ValueError, "offload-group-size"):
                memory_hook.handle_offload_compatibility(
                    self._args(offload_group_size=1)
                )


class HotCacheConfigurationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        view = patch.object(memory_hook, "resolving_view", lambda args: args)
        view.start()
        self.addCleanup(view.stop)

    def args(self, **changes):
        values = dict(
            ple_offload_embedding=False,
            cpu_offload_gb=1,
            offload_group_size=0,
            ple_offload_backend=None,
            moe_runner_backend="flashinfer_cutlass",
            tp_size=1,
            ep_size=1,
            moe_a2a_backend="none",
            disable_overlap_schedule=True,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            max_running_requests=1,
            expert_distribution_recorder_mode="stat",
            elastic_ep_backend=None,
            elastic_ep_rejoin=False,
            ep_join_mode=None,
            enable_elastic_expert_backup=False,
            enable_eplb=False,
            enable_waterfill=False,
            ep_join_rank_offset=0,
            elastic_ep_initial_size=None,
            max_ep_size=None,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="disabled"),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def enable(self):
        os.environ.update(SGLANG_MOE_EXPERT_STREAM="1", SGLANG_MOE_HOT_GPU_MB="1")

    def test_hot_cache_defaults_are_disabled_with_bounded_update_policy(self):
        expected = dict(
            SGLANG_MOE_EXPERT_FILE_DIR="",
            SGLANG_MOE_PINNED_HOST_MB=0,
            SGLANG_MOE_HOT_GPU_MB=0,
            SGLANG_MOE_HOT_SEED="",
            SGLANG_MOE_HOT_DYNAMIC=False,
            SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS=1024,
            SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=0,
            SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS=8,
            SGLANG_MOE_HOT_BENEFIT_RATIO=1.0,
            SGLANG_MOE_HOT_LOG_INTERVAL=100,
            SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0,
        )
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(envs, name).get(), value)

    def test_prefetch_requires_the_complete_single_request_cache_envelope(self):
        os.environ["SGLANG_MOE_PREFETCH_MAX_CANDIDATES"] = "1"
        with self.assertRaisesRegex(ValueError, "hot GPU and pinned host"):
            memory_hook.handle_offload_compatibility(self.args())
        self.enable()
        with self.assertRaisesRegex(ValueError, "hot GPU and pinned host"):
            memory_hook.handle_offload_compatibility(self.args())
        os.environ["SGLANG_MOE_PINNED_HOST_MB"] = "1"
        memory_hook.handle_offload_compatibility(self.args())

    def test_hot_cache_requires_streaming(self):
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "1"
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_STREAM"):
            memory_hook.handle_offload_compatibility(self.args())

    def test_pinned_host_cache_requires_streaming(self):
        os.environ["SGLANG_MOE_PINNED_HOST_MB"] = "1"
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_STREAM"):
            memory_hook.handle_offload_compatibility(self.args())

    def test_unsupported_runtime_combinations_fail_early(self):
        self.enable()
        cases = (
            (dict(moe_runner_backend="auto"), "flashinfer_cutlass"),
            (dict(tp_size=2), "TP size 1"),
            (dict(ep_size=2), "EP size 1"),
            (dict(moe_a2a_backend="deepep"), "moe-a2a-backend none"),
            (dict(disable_overlap_schedule=False), "disable-overlap-schedule"),
            (dict(enable_two_batch_overlap=True), "overlap"),
            (dict(enable_single_batch_overlap=True), "overlap"),
            (dict(max_running_requests=2), "max-running-requests 1"),
            (dict(max_running_requests=None), "max-running-requests 1"),
            (dict(elastic_ep_backend="mooncake"), "elastic"),
            (dict(elastic_ep_rejoin=True), "elastic"),
            (dict(ep_join_mode="scale"), "elastic"),
            (dict(enable_elastic_expert_backup=True), "elastic"),
            (dict(elastic_ep_initial_size=1), "elastic"),
            (dict(max_ep_size=2), "elastic"),
            (dict(enable_eplb=True), "EPLB"),
        )
        for changes, message in cases:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, message),
            ):
                memory_hook.handle_offload_compatibility(self.args(**changes))

    def test_waterfill_rejected_before_a2a_backend_resolution(self):
        self.enable()
        args = self.args(enable_waterfill=True, moe_a2a_backend="none")
        with self.assertRaisesRegex(ValueError, "Waterfill"):
            memory_hook.handle_offload_compatibility(args)
        self.assertEqual(args.moe_a2a_backend, "none")
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "0"
        memory_hook.handle_offload_compatibility(args)

    def test_nonzero_ep_join_rank_offset_is_rejected(self):
        self.enable()
        with self.assertRaisesRegex(ValueError, "elastic EP"):
            memory_hook.handle_offload_compatibility(self.args(ep_join_rank_offset=1))

    def test_every_routed_graph_phase_is_rejected(self):
        self.enable()
        for phase, backends in (
            ("decode", ("full", "tc_piecewise")),
            ("prefill", ("full", "breakable", "tc_piecewise")),
        ):
            for backend in backends:
                args = self.args()
                getattr(args.cuda_graph_config, phase).backend = backend
                with (
                    self.subTest(phase=phase, backend=backend),
                    self.assertRaisesRegex(ValueError, "CUDA graph"),
                ):
                    memory_hook.handle_offload_compatibility(args)

    def test_decode_breakable_cuda_graph_is_allowed(self):
        self.enable()
        allowed = self.args()
        allowed.cuda_graph_config.decode.backend = "breakable"
        memory_hook.handle_offload_compatibility(allowed)

        for backend in ("full", "tc_piecewise"):
            args = self.args()
            args.cuda_graph_config.decode.backend = backend
            with (
                self.subTest(phase="decode", backend=backend),
                self.assertRaisesRegex(ValueError, "CUDA graph"),
            ):
                memory_hook.handle_offload_compatibility(args)

        for backend in ("full", "breakable", "tc_piecewise"):
            args = self.args()
            args.cuda_graph_config.decode.backend = "breakable"
            args.cuda_graph_config.prefill.backend = backend
            with (
                self.subTest(phase="prefill", backend=backend),
                self.assertRaisesRegex(ValueError, "CUDA graph"),
            ):
                memory_hook.handle_offload_compatibility(args)

    def test_pinned_host_cache_allows_decode_breakable_cuda_graph(self):
        self.enable()
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "0"
        os.environ["SGLANG_MOE_PINNED_HOST_MB"] = "1"
        allowed = self.args()
        allowed.cuda_graph_config.decode.backend = "breakable"
        memory_hook.handle_offload_compatibility(allowed)

    def enable_graph_gather(self):
        self.enable()
        os.environ.update(
            SGLANG_MOE_EXPERT_GRAPH_GATHER="1", SGLANG_MOE_EXPERT_HOST_ARENA="1"
        )

    def decode_args(self, backend, **changes):
        args = self.args(**changes)
        args.cuda_graph_config.decode.backend = backend
        return args

    def test_graph_gather_allows_full_decode_graphs_without_file_ple(self):
        self.enable_graph_gather()
        for backend in ("breakable", "full"):
            with self.subTest(backend=backend):
                memory_hook.handle_offload_compatibility(self.decode_args(backend))

        file_ple = dict(ple_offload_embedding=True, ple_offload_backend="file")
        with self.assertRaisesRegex(ValueError, "file-backed PLE"):
            memory_hook.handle_offload_compatibility(
                self.decode_args("full", **file_ple)
            )
        memory_hook.handle_offload_compatibility(
            self.decode_args("breakable", **file_ple)
        )
        with self.assertRaisesRegex(ValueError, "CUDA graph"):
            memory_hook.handle_offload_compatibility(self.decode_args("tc_piecewise"))
        args = self.decode_args("full")
        args.cuda_graph_config.prefill.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "CUDA graph"):
            memory_hook.handle_offload_compatibility(args)

    def test_ple_staging_before_replay_allows_full_decode_graphs(self):
        self.enable_graph_gather()
        os.environ["SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY"] = "1"
        file_ple = dict(ple_offload_embedding=True, ple_offload_backend="file")
        memory_hook.handle_offload_compatibility(self.decode_args("full", **file_ple))

        cases = (
            (dict(speculative_algorithm="EAGLE", **file_ple), "speculative"),
            (dict(dllm_algorithm="LowConfidence", **file_ple), "DLLM"),
            (dict(disable_overlap_schedule=False, **file_ple), "overlap"),
            (dict(), "file-backed PLE"),
        )
        for changes, message in cases:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, message),
            ):
                memory_hook.handle_offload_compatibility(
                    self.decode_args("breakable", **changes)
                )
        args = self.decode_args("breakable", **file_ple)
        args.cuda_graph_config.prefill.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "disable prefill CUDA graph"):
            memory_hook.handle_offload_compatibility(args)

    def test_graph_gather_requires_its_cache_envelope(self):
        self.enable_graph_gather()
        cases = (
            (dict(SGLANG_MOE_HOT_GPU_MB="0"), "requires SGLANG_MOE_HOT_GPU_MB"),
            (dict(SGLANG_MOE_EXPERT_HOST_ARENA="0"), "SGLANG_MOE_EXPERT_HOST_ARENA=1"),
            (dict(SGLANG_MOE_PINNED_HOST_MB="1"), "SGLANG_MOE_PINNED_HOST_MB=0"),
            (dict(SGLANG_MOE_PREFETCH_MAX_CANDIDATES="1"), "expert prefetch"),
        )
        for environment, message in cases:
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment),
                self.assertRaisesRegex(ValueError, message),
            ):
                memory_hook.handle_offload_compatibility(self.decode_args("breakable"))
        with self.assertRaisesRegex(ValueError, "requires decode CUDA graphs"):
            memory_hook.handle_offload_compatibility(self.decode_args("disabled"))

    def test_host_arena_rejects_the_pinned_host_cache(self):
        os.environ.update(
            SGLANG_MOE_EXPERT_STREAM="1",
            SGLANG_MOE_EXPERT_HOST_ARENA="1",
            SGLANG_MOE_PINNED_HOST_MB="1",
        )
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_PINNED_HOST_MB=0"):
            memory_hook.handle_offload_compatibility(self.args())

    def test_early_graph_resolution_and_repeated_validation_are_safe(self):
        self.enable()
        args = self.args(cuda_graph_config=None)
        memory_hook.handle_offload_compatibility(args)
        args.cuda_graph_config = self.args().cuda_graph_config
        memory_hook.handle_offload_compatibility(args)
        memory_hook.handle_offload_compatibility(args)

    def test_dynamic_mode_requires_exact_stat_recorder(self):
        self.enable()
        os.environ["SGLANG_MOE_HOT_DYNAMIC"] = "1"
        for recorder in (None, "per_pass", "stat_approx"):
            with (
                self.subTest(recorder=recorder),
                self.assertRaisesRegex(ValueError, "stat"),
            ):
                memory_hook.handle_offload_compatibility(
                    self.args(expert_distribution_recorder_mode=recorder)
                )
        memory_hook.handle_offload_compatibility(self.args())

    def test_real_server_args_reads_resolved_graphs_without_mutating_inputs(self):
        from sglang.srt.arg_groups.model_override_base import resolving_view
        from sglang.srt.server_args import ServerArgs

        self.enable()
        args = ServerArgs(
            model_path="/unused/model", **vars(self.args(cuda_graph_config=None))
        )
        with patch.object(memory_hook, "resolving_view", resolving_view):
            memory_hook.handle_offload_compatibility(args)
            graphs = self.args().cuda_graph_config
            args._resolved_overrides = [("test", {"cuda_graph_config": graphs})]
            memory_hook.handle_offload_compatibility(args)
            graphs.decode.backend = "full"
            with self.assertRaisesRegex(ValueError, "CUDA graph"):
                memory_hook.handle_offload_compatibility(args)
        self.assertIsNone(args.cuda_graph_config)

    def test_zero_budget_ignores_unused_seed_dynamic_and_policy_settings(self):
        os.environ.update(
            SGLANG_MOE_HOT_GPU_MB="0",
            SGLANG_MOE_HOT_SEED="/missing/seed",
            SGLANG_MOE_HOT_DYNAMIC="1",
            SGLANG_MOE_HOT_LOG_INTERVAL="0",
        )
        memory_hook.handle_offload_compatibility(
            self.args(
                moe_runner_backend="auto",
                disable_overlap_schedule=False,
                max_running_requests=None,
                expert_distribution_recorder_mode=None,
            )
        )

    def test_invalid_policy_is_rejected_before_loading(self):
        self.enable()
        for name, value in (
            ("SGLANG_MOE_HOT_GPU_MB", "-1"),
            ("SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS", "0"),
            ("SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS", "-1"),
            ("SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS", "-1"),
            ("SGLANG_MOE_HOT_BENEFIT_RATIO", "nan"),
            ("SGLANG_MOE_HOT_LOG_INTERVAL", "0"),
        ):
            with patch.dict(os.environ, {name: value}), self.subTest(name=name):
                with self.assertRaises(ValueError):
                    memory_hook.handle_offload_compatibility(self.args())


class HotCacheStartupTests(unittest.TestCase):
    def test_manager_allocation_follows_topk_and_precedes_remaining_initialization(
        self,
    ):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor import model_runner

        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic), ExitStack() as stack:
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "SGLANG_MOE_HOT_GPU_MB": "2",
                            "SGLANG_MOE_HOT_SEED": "seed.pt",
                            "SGLANG_MOE_HOT_DYNAMIC": str(int(dynamic)),
                            "SGLANG_MOE_HOT_LOG_INTERVAL": "7",
                        },
                    )
                )
                runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
                runner.model = torch.nn.Module()
                runner.model_config = object()
                runner.ps = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
                events = []
                for name in (
                    "init_memory_saver_adapter",
                    "maybe_init_remote_instance_transfer_engine",
                    "maybe_init_expert_location_metadata",
                    "maybe_init_lplb_solvers",
                    "maybe_init_eplb_manager",
                    "maybe_init_elastic_ep",
                    "init_token_oracle",
                ):
                    stack.enter_context(patch.object(runner, name))
                stack.enter_context(patch.object(model_runner, "ExpertLocationUpdater"))
                stack.enter_context(patch.object(model_runner, "create_sampler"))
                stack.enter_context(
                    patch.object(
                        runner, "load_model", side_effect=lambda: events.append("load")
                    )
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "maybe_init_expert_pinned_host_cache",
                        side_effect=lambda: events.append("pinned"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        model_runner,
                        "prepare_moe_topk",
                        side_effect=lambda **kw: events.append("topk"),
                    )
                )
                manager = SimpleNamespace(
                    on_expert_distribution=lambda *args: None,
                    enable_next_layer_prefetch=lambda candidates: events.append(
                        ("prefetch", candidates)
                    ),
                )

                def allocate(model, **options):
                    self.assertIs(model, runner.model)
                    self.assertEqual(events, ["load", "topk", "pinned"])
                    self.assertEqual(options["budget_bytes"], 2 * 1024 * 1024)
                    self.assertEqual(options["seed_path"], "seed.pt")
                    self.assertEqual(options["dynamic"], dynamic)
                    self.assertEqual(options["log_interval"], 7)
                    events.append("allocate")
                    return manager

                stack.enter_context(
                    patch.object(
                        ExpertHotCacheManager, "from_model", side_effect=allocate
                    )
                )
                observer = stack.enter_context(
                    patch.object(
                        model_runner, "get_global_expert_distribution_recorder"
                    )
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "maybe_init_dwdp",
                        side_effect=RuntimeError("stop after allocation"),
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "stop after allocation"):
                    runner.initialize()
                self.assertIs(
                    getattr(runner, "expert_hot_cache_manager", None), manager
                )
                observer.return_value.register_forward_observer.assert_called_once_with(
                    manager.on_expert_distribution
                )

                self.assertIn(("prefetch", 0), events)

    def test_zero_budget_skips_manager_creation_and_recorder_registration(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor import model_runner

        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_MOE_HOT_GPU_MB": "0",
                    "SGLANG_MOE_HOT_SEED": "/missing/seed",
                    "SGLANG_MOE_HOT_DYNAMIC": "1",
                    "SGLANG_MOE_HOT_LOG_INTERVAL": "invalid",
                },
            ),
            patch.object(ExpertHotCacheManager, "from_model") as factory,
        ):
            runner.maybe_init_expert_hot_cache()
        self.assertIsNone(runner.expert_hot_cache_manager)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
