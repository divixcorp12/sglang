"""CPU tests for the per-format server-args gate of MoE expert caching."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import expert_stream_requirements as requirements_module
from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.expert_stream_requirements import (
    NVFP4_EXPERT_STREAM_REQUIREMENTS,
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    expert_quant_method,
    expert_stream_requirements_for,
    register_expert_stream_requirements,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _launch(**changes):
    """Server arguments of an eager launch with neither NVFP4-only setting."""
    values = dict(
        ple_offload_embedding=False,
        cpu_offload_gb=0,
        offload_group_size=0,
        ple_offload_backend=None,
        moe_runner_backend="auto",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        disable_overlap_schedule=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        max_running_requests=4,
        expert_distribution_recorder_mode="per_pass",
        enable_waterfill=False,
        enable_eplb=False,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="disabled"),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


class _GateTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        view = patch.object(memory_hook, "resolving_view", lambda args: args)
        view.start()
        self.addCleanup(view.stop)
        registry = patch.dict(requirements_module._REGISTRY)
        registry.start()
        self.addCleanup(registry.stop)


class TestQuantMethodResolution(_GateTest):
    def _model_dir(self, directory, quantization_config):
        config = {"architectures": ["X"]}
        if quantization_config is not None:
            config["quantization_config"] = quantization_config
        with open(os.path.join(directory, "config.json"), "w") as stream:
            json.dump(config, stream)
        return directory

    def test_nothing_known_resolves_to_none(self):
        self.assertIsNone(expert_quant_method(SimpleNamespace(), SimpleNamespace()))
        args = SimpleNamespace(quantization=None, model_path="/no/such/model")
        self.assertIsNone(expert_quant_method(args, args))

    def test_whitespace_only_quantization_falls_through_to_config_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self._model_dir(directory, {"quant_method": "EXL3"})
            args = SimpleNamespace(quantization="   ", model_path=directory)
            self.assertEqual(expert_quant_method(args, args), "exl3")

    def test_config_json_names_the_method(self):
        with tempfile.TemporaryDirectory() as directory:
            self._model_dir(directory, {"quant_method": "EXL3", "bits": 3})
            args = SimpleNamespace(quantization=None, model_path=directory)
            self.assertEqual(expert_quant_method(args, args), "exl3")

    def test_config_json_without_a_method_or_readable_json_resolves_to_none(self):
        # The production NVFP4 checkpoint's quantization_config has no quant_method.
        for quantization_config in ({"quant_algo": "MIXED_PRECISION"}, None):
            with tempfile.TemporaryDirectory() as directory:
                self._model_dir(directory, quantization_config)
                args = SimpleNamespace(quantization=None, model_path=directory)
                self.assertIsNone(expert_quant_method(args, args))
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as stream:
                stream.write("{not json")
            args = SimpleNamespace(quantization=None, model_path=directory)
            self.assertIsNone(expert_quant_method(args, args))

    def test_explicit_quantization_beats_the_model_config_which_beats_config_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self._model_dir(directory, {"quant_method": "exl3"})
            args = SimpleNamespace(
                quantization=None,
                model_path=directory,
                _model_config=SimpleNamespace(quantization="modelopt_mixed"),
            )
            self.assertEqual(expert_quant_method(args, args), "modelopt_mixed")
            args.quantization = "ModelOpt_FP4"
            self.assertEqual(expert_quant_method(args, args), "modelopt_fp4")


class TestRequirementsLookup(_GateTest):
    def test_undetermined_and_modelopt_methods_get_the_nvfp4_requirements(self):
        self.assertIs(
            expert_stream_requirements_for(SimpleNamespace(), SimpleNamespace()),
            NVFP4_EXPERT_STREAM_REQUIREMENTS,
        )
        for method in (
            "modelopt",
            "modelopt_fp4",
            "modelopt_mixed",
            "MODELOPT_MIXED",
            "nvfp4_online",
            "fp8",
            "mxfp8",
            "inkling_nvfp4",
        ):
            args = SimpleNamespace(quantization=method)
            with self.subTest(method=method):
                self.assertIs(
                    expert_stream_requirements_for(args, args),
                    NVFP4_EXPERT_STREAM_REQUIREMENTS,
                )

    def test_an_unknown_method_with_budgets_is_rejected(self):
        os.environ.update(SGLANG_MOE_EXPERT_STREAM="1", SGLANG_MOE_HOT_GPU_MB="1")
        with self.assertRaisesRegex(ValueError, "does not support quantization method 'awq'"):
            memory_hook.handle_offload_compatibility(_launch(quantization="awq"))
        os.environ.update(SGLANG_MOE_HOT_GPU_MB="0", SGLANG_MOE_PINNED_HOST_MB="1")
        with self.assertRaisesRegex(ValueError, "does not support quantization method 'awq'"):
            memory_hook.handle_offload_compatibility(_launch(quantization="awq"))

    def test_an_unknown_method_without_budgets_is_not_checked(self):
        memory_hook.handle_offload_compatibility(_launch(quantization="awq"))

    def test_a_plugin_module_registers_its_method_on_first_use(self):
        imported = []
        requirements = ExpertStreamRequirements("Lazy", lambda cfg, budgets: None)

        def import_module(name):
            imported.append(name)
            register_expert_stream_requirements(("lazy-fmt",), requirements)

        args = SimpleNamespace(quantization="lazy-fmt")
        with patch.object(requirements_module.importlib, "import_module", import_module):
            self.assertIs(expert_stream_requirements_for(args, args), requirements)
            self.assertIs(expert_stream_requirements_for(args, args), requirements)
        self.assertEqual(
            imported, ["sglang.srt.arg_groups.expert_stream_requirements_lazy_fmt"]
        )

    def test_a_plugin_failing_on_its_own_import_is_not_hidden(self):
        def import_module(name):
            raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")

        args = SimpleNamespace(quantization="broken")
        with patch.object(requirements_module.importlib, "import_module", import_module):
            with self.assertRaisesRegex(ModuleNotFoundError, "missing_dep"):
                expert_stream_requirements_for(args, args)

    def test_a_method_cannot_be_registered_twice(self):
        first = ExpertStreamRequirements("A", lambda cfg, budgets: None)
        register_expert_stream_requirements(("twice",), first)
        register_expert_stream_requirements(("twice",), first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_expert_stream_requirements(
                ("TWICE",), ExpertStreamRequirements("B", lambda cfg, budgets: None)
            )


class TestEagerFormatRequirements(_GateTest):
    """A format registered with ``eager_expert_stream_requirements`` (EXL3's shape)."""

    def setUp(self):
        super().setUp()
        self.streaming = {"on": True}
        register_expert_stream_requirements(
            ("eager_test",),
            eager_expert_stream_requirements(
                "EAGER",
                enabled=lambda: self.streaming["on"],
                enable_hint="the test stream switch",
            ),
        )
        os.environ.update(
            SGLANG_MOE_HOT_GPU_MB="1",
            SGLANG_MOE_PINNED_HOST_MB="1",
            SGLANG_MOE_HOT_DYNAMIC="1",
        )

    def test_it_needs_none_of_the_nvfp4_only_settings(self):
        # No SGLANG_MOE_EXPERT_STREAM, overlap on, 4 running requests, auto backend.
        memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))
        os.environ["SGLANG_MOE_HOT_DYNAMIC"] = "0"
        memory_hook.handle_offload_compatibility(
            _launch(quantization="eager_test", expert_distribution_recorder_mode=None)
        )

    def test_its_own_requirements_are_enforced(self):
        cases = (
            (dict(), {"decode": "breakable"}, "runs eagerly"),
            (dict(), {"prefill": "breakable"}, "runs eagerly"),
            (dict(expert_distribution_recorder_mode=None), {}, "stat or per_pass"),
            (dict(expert_distribution_recorder_mode="per_token"), {}, "stat or per_pass"),
        )
        for changes, graphs, message in cases:
            args = _launch(quantization="eager_test", **changes)
            for phase, backend in graphs.items():
                getattr(args.cuda_graph_config, phase).backend = backend
            with self.subTest(changes=changes, graphs=graphs):
                with self.assertRaisesRegex(ValueError, message):
                    memory_hook.handle_offload_compatibility(args)
        self.streaming["on"] = False
        with self.assertRaisesRegex(ValueError, "requires the test stream switch"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_graph_gather_and_the_host_arena_are_refused(self):
        # Satisfy the generic graph-gather checks so the format's own refusal is reached.
        os.environ.update(
            SGLANG_MOE_PINNED_HOST_MB="0",
            SGLANG_MOE_EXPERT_GRAPH_GATHER="1",
            SGLANG_MOE_EXPERT_HOST_ARENA="1",
        )
        args = _launch(quantization="eager_test")
        args.cuda_graph_config.decode.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "does not support SGLANG_MOE_EXPERT_GRAPH_GATHER"):
            memory_hook.handle_offload_compatibility(args)
        os.environ["SGLANG_MOE_EXPERT_GRAPH_GATHER"] = "0"
        with self.assertRaisesRegex(ValueError, "does not support SGLANG_MOE_EXPERT_HOST_ARENA"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_the_policy_knobs_are_checked_for_it_too(self):
        os.environ["SGLANG_MOE_HOT_LOG_INTERVAL"] = "0"
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_LOG_INTERVAL"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_expert_prefetch_is_refused(self):
        os.environ["SGLANG_MOE_PREFETCH_MAX_CANDIDATES"] = "1"
        with self.assertRaisesRegex(
            ValueError, "set SGLANG_MOE_PREFETCH_MAX_CANDIDATES to 0"
        ):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_gpu_residency_update_is_refused(self):
        os.environ["SGLANG_MOE_GPU_RESIDENCY_UPDATE"] = "1"
        with self.assertRaisesRegex(
            ValueError, "SGLANG_MOE_GPU_RESIDENCY_UPDATE; set it to 0"
        ):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_doorbell_is_refused(self):
        os.environ["SGLANG_MOE_EXPERT_DOORBELL"] = "1"
        with self.assertRaisesRegex(
            ValueError, "SGLANG_MOE_EXPERT_DOORBELL; set it to 0"
        ):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_it_accepts_a_launch_with_all_three_settings_off(self):
        os.environ.update(
            SGLANG_MOE_PREFETCH_MAX_CANDIDATES="0",
            SGLANG_MOE_GPU_RESIDENCY_UPDATE="0",
            SGLANG_MOE_EXPERT_DOORBELL="0",
        )
        memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))


if __name__ == "__main__":
    unittest.main()
