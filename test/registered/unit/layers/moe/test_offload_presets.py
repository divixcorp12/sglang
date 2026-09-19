"""MoE offload presets: the preset values, the merge and derivation rules, and validation."""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import moe_offload_hook
from sglang.srt.environ import envs
from sglang.srt.layers.moe import offload_presets as presets
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A real config.json, so TestPipelineWiring's resolve_once() runs past the
# pipeline's dummy-model early return without loading actual weights.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# The offload variables of divix01:/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh as
# of 2026-09-19, minus site-specific paths. The graph-gather preset must reproduce them.
PROD_OFFLOAD_ENV = {
    "SGLANG_MOE_EXPERT_STREAM": "1",
    "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
    "SGLANG_QWEN4_PLE_FILE_READER": "uring",
    "SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY": "1",
    "SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING": "1",
    "SGLANG_MOE_HOT_GPU_MB": "15360",
    "SGLANG_MOE_PINNED_HOST_MB": "0",
    "SGLANG_MOE_EXPERT_HOST_ARENA": "1",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
    "SGLANG_MOE_EXPERT_FUSED_PLAN": "1",
    "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT": "1",
    "SGLANG_MOE_HOT_DYNAMIC": "1",
    "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "1",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
    "SGLANG_MOE_HOT_DECAY_TOKENS": "1",
    "SGLANG_MOE_HOT_PROMOTION_SIGMAS": "0",
    "SGLANG_MOE_HOT_BENEFIT_RATIO": "2",
    "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "0",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
    "SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS": "64",
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
    "SGLANG_MOE_EXPERT_COPY_BACKEND": "dma",
}

ALL_CPUS = frozenset(range(72))


def parsed(values):
    return {name: getattr(envs, name).parse(value) for name, value in values.items()}


def check(values, **overrides):
    context = dict(
        speculative=False,
        decode_graphs_disabled=False,
        decode_max_bs=1,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        dp_attention=False,
        allowed_cpus=ALL_CPUS,
        nvfp4_hot_cache=True,
    )
    context.update(overrides)
    presets.check_offload_config(values, **context)


class TestPresetValues(unittest.TestCase):
    def test_graph_gather_preset_is_the_prod_offload_env(self):
        self.assertEqual(parsed(presets.preset_env(presets.GRAPH_GATHER_PRESET)), parsed(PROD_OFFLOAD_ENV))

    def test_doorbell_preset_departs_from_graph_gather_only_where_the_doorbell_forces_it(self):
        graph = parsed(presets.preset_env(presets.GRAPH_GATHER_PRESET))
        doorbell = parsed(presets.preset_env(presets.DOORBELL_PRESET))
        changed = {name for name in graph.keys() | doorbell.keys() if graph.get(name) != doorbell.get(name)}
        self.assertEqual(
            changed,
            {
                "SGLANG_MOE_EXPERT_DOORBELL",
                "SGLANG_MOE_EXPERT_DOORBELL_MODE",
                "SGLANG_MOE_EXPERT_DOORBELL_CPU",
                "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
                "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT",
            },
        )

    def test_every_preset_field_names_an_envs_descriptor(self):
        self.assertEqual(set(presets.ENV_NAMES), set(presets.MoeOffloadPreset.__struct_fields__))
        for name in presets.ENV_NAMES.values():
            self.assertTrue(hasattr(envs, name), name)
        self.assertEqual(set(presets.PRESETS), {"off", "graph-gather", "doorbell"})
        self.assertIsNone(presets.PRESETS["off"])


class TestResolution(unittest.TestCase):
    def test_an_explicit_variable_wins_over_the_preset(self):
        resolved = presets.resolve_offload_env(presets.GRAPH_GATHER_PRESET, {"SGLANG_MOE_HOT_GPU_MB": "12288"})
        self.assertEqual(resolved.effective["SGLANG_MOE_HOT_GPU_MB"], "12288")
        self.assertNotIn("SGLANG_MOE_HOT_GPU_MB", resolved.filled)
        self.assertEqual(resolved.overridden, {"SGLANG_MOE_HOT_GPU_MB": "15360"})

    def test_an_explicit_variable_equal_to_the_preset_is_not_an_override(self):
        resolved = presets.resolve_offload_env(presets.GRAPH_GATHER_PRESET, {"SGLANG_MOE_HOT_BENEFIT_RATIO": "2"})
        self.assertEqual(resolved.overridden, {})

    def test_off_with_no_offload_env_sets_nothing(self):
        resolved = presets.resolve_offload_env(None, {})
        self.assertEqual((resolved.effective, resolved.filled, resolved.overridden), ({}, {}, {}))

    def test_insert_on_miss_derives_the_residency_update_and_its_boundary_cadence(self):
        resolved = presets.resolve_offload_env(None, {"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2"})
        self.assertEqual(
            parsed(resolved.filled),
            {
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE": True,
                "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": 1,
                "SGLANG_MOE_HOT_DYNAMIC": True,
            },
        )

    def test_an_explicit_value_contradicting_a_derivation_is_refused(self):
        for explicit, culprit in (
            ({"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "1", "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0"}, "SGLANG_MOE_GPU_RESIDENCY_UPDATE"),
            ({"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2", "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "4"}, "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS"),
            ({"SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1", "SGLANG_MOE_HOT_DYNAMIC": "0"}, "SGLANG_MOE_HOT_DYNAMIC"),
        ):
            with self.subTest(culprit=culprit), self.assertRaisesRegex(ValueError, culprit):
                presets.resolve_offload_env(None, explicit)

    def test_explicit_offload_env_keeps_only_offload_variables(self):
        environ = {"SGLANG_MOE_HOT_GPU_MB": "1024", "PATH": "/bin", "SGLANG_LOG_MS": "1"}
        self.assertEqual(presets.explicit_offload_env(environ), {"SGLANG_MOE_HOT_GPU_MB": "1024"})


class TestOverlapRule(unittest.TestCase):
    def test_graph_gather_keeps_overlap_and_doorbell_turns_it_off(self):
        graph = presets.preset_env(presets.GRAPH_GATHER_PRESET)
        doorbell = presets.preset_env(presets.DOORBELL_PRESET)
        self.assertFalse(presets.needs_overlap_off(graph, nvfp4_hot_cache=True))
        self.assertTrue(presets.needs_overlap_off(doorbell, nvfp4_hot_cache=True))

    def test_a_hot_cache_without_graph_gather_and_the_residency_update_turns_overlap_off(self):
        self.assertTrue(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}, nvfp4_hot_cache=True))
        self.assertFalse(presets.needs_overlap_off({}, nvfp4_hot_cache=True))

    def test_the_hot_cache_rule_is_nvfp4s_but_the_doorbell_rule_is_every_formats(self):
        self.assertFalse(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}, nvfp4_hot_cache=False))
        doorbell = presets.preset_env(presets.DOORBELL_PRESET)
        self.assertTrue(presets.needs_overlap_off(doorbell, nvfp4_hot_cache=False))


class TestValidation(unittest.TestCase):
    def test_both_presets_pass_their_intended_setups(self):
        check(presets.preset_env(presets.GRAPH_GATHER_PRESET), speculative=True)
        check(presets.preset_env(presets.DOORBELL_PRESET))

    def test_invalid_combinations_are_refused_before_the_weight_load(self):
        graph = presets.preset_env(presets.GRAPH_GATHER_PRESET)
        doorbell = presets.preset_env(presets.DOORBELL_PRESET)
        cases = (
            (doorbell, dict(speculative=True), "speculative"),
            (dict(doorbell, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE="2"), {}, "INSERT_ON_MISS_STAGE=2"),
            (doorbell, dict(allowed_cpus=frozenset(range(64))), "DOORBELL_CPU=71"),
            (doorbell, dict(tp_size=2), "parallel"),
            (doorbell, dict(dp_attention=True), "parallel"),
            (graph, dict(decode_graphs_disabled=True), "decode CUDA graphs"),
            (graph, dict(decode_max_bs=2), "cuda-graph-max-bs-decode 1"),
            (dict(graph, SGLANG_MOE_EXPERT_STREAM="0"), {}, "SGLANG_MOE_EXPERT_STREAM=1"),
        )
        for values, context, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check(values, **context)

    def test_another_format_streams_experts_without_the_nvfp4_stream_variable(self):
        # DSV4.1's EXL3 experts stream under SGLANG_DSV41_EXPERT_STREAM.
        check(
            {"SGLANG_MOE_HOT_GPU_MB": "14336", "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"},
            nvfp4_hot_cache=False,
        )


class TestPresetHook(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        for target, replacement in (
            ("resolving_view", lambda args: args),
            ("declare_resolution", self.record_declaration),
        ):
            patcher = patch.object(moe_offload_hook, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        affinity = patch.object(moe_offload_hook.os, "sched_getaffinity", return_value=set(range(72)))
        affinity.start()
        self.addCleanup(affinity.stop)
        self.declared = []

    def record_declaration(self, server_args, source, **fields):
        self.declared.append(fields)

    def args(self, preset, **changes):
        values = dict(
            moe_offload_preset=preset,
            disable_overlap_schedule=False,
            speculative_algorithm=None,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="breakable", max_bs=1),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_graph_gather_sets_every_unset_variable_and_keeps_overlap(self):
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "12288"
        moe_offload_hook.handle_moe_offload_preset(self.args("graph-gather", speculative_algorithm="NEXTN"))
        expected = dict(presets.preset_env(presets.GRAPH_GATHER_PRESET), SGLANG_MOE_HOT_GPU_MB="12288")
        self.assertEqual(presets.explicit_offload_env(os.environ), expected)
        self.assertEqual(self.declared, [])

    def test_doorbell_turns_overlap_off(self):
        moe_offload_hook.handle_moe_offload_preset(self.args("doorbell"))
        self.assertEqual(self.declared, [{"disable_overlap_schedule": True}])
        self.assertEqual(envs.SGLANG_MOE_EXPERT_DOORBELL_CPU.get(), 71)

    def test_off_with_no_offload_env_touches_nothing(self):
        moe_offload_hook.handle_moe_offload_preset(self.args("off"))
        self.assertEqual(presets.explicit_offload_env(os.environ), {})
        self.assertEqual(self.declared, [])

    def test_an_exl3_hot_cache_keeps_overlap_and_passes_without_the_nvfp4_stream_variable(self):
        os.environ.update(SGLANG_MOE_HOT_GPU_MB="14336", SGLANG_MOE_EXPERT_GRAPH_GATHER="1")
        args = self.args("off", quantization="exl3")
        moe_offload_hook.handle_moe_offload_preset(args)
        self.assertEqual(self.declared, [])
        moe_offload_hook.check_moe_offload_config(args)

    def test_a_hot_cache_of_unknown_format_gets_the_nvfp4_rules(self):
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "14336"
        args = self.args("off")
        moe_offload_hook.handle_moe_offload_preset(args)
        self.assertEqual(self.declared, [{"disable_overlap_schedule": True}])
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_STREAM=1"):
            moe_offload_hook.check_moe_offload_config(args)

    def test_a_refusal_names_the_preset(self):
        args = self.args("doorbell", speculative_algorithm="NEXTN")
        moe_offload_hook.handle_moe_offload_preset(args)
        with self.assertRaisesRegex(ValueError, "--moe-offload-preset doorbell: .*speculative"):
            moe_offload_hook.check_moe_offload_config(args)


class TestPipelineWiring(unittest.TestCase):
    """Runs the real resolution pipeline (past its dummy-model early return,
    same technique as test_resolution_declarations.py's _resolve) to pin
    where check_moe_offload_config is wired: after handle_cuda_graph_config
    has parsed --cuda-graph-max-bs-decode, which handle_moe_offload_preset
    alone cannot see (that field is still the CLI default there)."""

    def setUp(self):
        self._environ_snapshot = dict(os.environ)
        self.addCleanup(self._restore_environ)

    def _restore_environ(self):
        os.environ.clear()
        os.environ.update(self._environ_snapshot)

    def _resolve(self, **fields):
        path = tempfile.mkdtemp(prefix="offload_preset_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        server_args = ServerArgs(model_path=path, device="cuda", random_seed=42, **fields)
        server_args.resolve_once()
        return server_args

    def test_graph_gather_with_decode_max_bs_2_is_refused(self):
        # graph-gather also sets SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY, which
        # handle_offload_compatibility (run earlier) requires file-backed PLE
        # offload for; satisfy it so this refusal is the one under test.
        with self.assertRaisesRegex(
            ValueError, "--moe-offload-preset graph-gather: .*cuda-graph-max-bs-decode 1"
        ):
            self._resolve(
                moe_offload_preset="graph-gather",
                cuda_graph_max_bs_decode=2,
                ple_offload_backend="file",
                ple_offload_embedding=True,
                moe_runner_backend="flashinfer_cutlass",
                max_running_requests=1,
                disable_prefill_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
            )

    def test_off_with_no_offload_env_resolves_cleanly(self):
        self._resolve(moe_offload_preset="off")
        self.assertEqual(presets.explicit_offload_env(os.environ), {})
