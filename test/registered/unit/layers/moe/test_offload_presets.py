"""MoE offload presets: the preset values, the merge and derivation rules, and validation."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.moe import offload_presets as presets
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
        self.assertFalse(presets.needs_overlap_off(presets.preset_env(presets.GRAPH_GATHER_PRESET)))
        self.assertTrue(presets.needs_overlap_off(presets.preset_env(presets.DOORBELL_PRESET)))

    def test_a_hot_cache_without_graph_gather_and_the_residency_update_turns_overlap_off(self):
        self.assertTrue(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}))
        self.assertFalse(presets.needs_overlap_off({}))


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
