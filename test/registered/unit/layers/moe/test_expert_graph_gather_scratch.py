"""CPU tests for speculative graph-gather scratch sizing and its validation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import memory_hook
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERT_ROW_BYTES = 2_764_808
LAYERS = 48
TOP_K = 10


class GraphGatherScratchValidationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        view = patch.object(memory_hook, "resolving_view", lambda args: args)
        view.start()
        self.addCleanup(view.stop)

    @staticmethod
    def args(speculative_algorithm="NEXTN"):
        return SimpleNamespace(
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
            speculative_algorithm=speculative_algorithm,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="breakable"),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )

    def test_scratch_row_cap_matrix(self):
        for graph_gather, rows, speculative, message in (
            ("1", None, None, None),
            ("1", None, "NEXTN", None),
            ("1", "24", "NEXTN", None),
            ("1", "0", None, None),
            ("1", "-1", "NEXTN", "must be nonnegative"),
            ("0", "24", "NEXTN", "requires SGLANG_MOE_EXPERT_GRAPH_GATHER=1"),
            ("1", "24", None, "requires speculative decoding"),
        ):
            environment = dict(
                SGLANG_MOE_EXPERT_STREAM="1",
                SGLANG_MOE_HOT_GPU_MB="1",
                SGLANG_MOE_EXPERT_HOST_ARENA="1",
                SGLANG_MOE_EXPERT_GRAPH_GATHER=graph_gather,
            )
            if rows is not None:
                environment["SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS"] = rows
            args = self.args(speculative)
            if graph_gather == "0":
                args.cuda_graph_config.decode.backend = "disabled"
            with (
                self.subTest(graph_gather=graph_gather, rows=rows, spec=speculative),
                patch.dict(os.environ, environment, clear=True),
            ):
                if message is None:
                    memory_hook.handle_offload_compatibility(args)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        memory_hook.handle_offload_compatibility(args)


class GraphGatherScratchArithmeticTests(unittest.TestCase):
    def test_rows_follow_verify_routes_and_cap(self):
        from sglang.srt.layers.moe.expert_hot_cache import graph_gather_scratch_rows

        for tokens, cap, rows in (
            (1, 0, 10),
            (4, 0, 40),
            (4, 24, 24),
            (4, 16, 16),
            (1, 24, 10),
            (8, 0, 80),
        ):
            with self.subTest(tokens=tokens, cap=cap):
                self.assertEqual(graph_gather_scratch_rows(tokens, TOP_K, cap), rows)
        with self.assertRaises(ValueError):
            graph_gather_scratch_rows(4, TOP_K, -1)

    def test_plan_budget_table(self):
        from sglang.srt.layers.moe.expert_hot_cache import graph_gather_scratch_rows

        for tokens, cap, gib in ((1, 0, 1.24), (4, 16, 1.98), (4, 24, 2.97), (4, 0, 4.94)):
            rows = graph_gather_scratch_rows(tokens, TOP_K, cap)
            with self.subTest(tokens=tokens, cap=cap):
                self.assertAlmostEqual(
                    LAYERS * rows * EXPERT_ROW_BYTES / 2**30, gib, places=2
                )


class GraphGatherStartupSizingTests(unittest.TestCase):
    def sized_options(self, speculative, environment):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor import model_runner
        from sglang.srt.speculative import spec_utils
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        streamed = torch.nn.Module()
        streamed._nvfp4_expert_streamer = SimpleNamespace(
            layer=SimpleNamespace(top_k=TOP_K)
        )
        runner.model = torch.nn.Sequential(streamed)
        runner.expert_host_arena = object()
        runner.is_draft_worker = False
        runner.spec_algorithm = (
            SpeculativeAlgorithm.from_string("NEXTN")
            if speculative
            else SimpleNamespace(is_speculative=lambda: False)
        )
        captured = {}

        def allocate(model, **options):
            captured.update(options)
            return None

        graph_config = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=2))
            )
        )
        with (
            patch.dict(
                os.environ,
                dict(
                    SGLANG_MOE_HOT_GPU_MB="1024",
                    SGLANG_MOE_EXPERT_GRAPH_GATHER="1",
                    **environment,
                ),
                clear=True,
            ),
            patch.object(model_runner, "get_exec", lambda: graph_config),
            patch.object(model_runner, "max_speculative_num_draft_tokens", lambda: 4),
            patch.object(
                spec_utils,
                "get_spec",
                lambda: SimpleNamespace(speculative_num_draft_tokens=4),
            ),
            patch.object(ExpertHotCacheManager, "from_model", side_effect=allocate),
        ):
            runner.maybe_init_expert_hot_cache()
        return captured

    def test_non_speculative_sizing_is_decode_batch(self):
        options = self.sized_options(False, {})
        self.assertEqual(options["graph_gather_batch_size"], 2)
        self.assertEqual(options.get("graph_gather_max_rows", 0), 0)

    def test_speculative_sizing_multiplies_verify_tokens(self):
        options = self.sized_options(True, {})
        self.assertEqual(options["graph_gather_batch_size"], 8)
        self.assertEqual(options["graph_gather_max_rows"], 0)

    def test_speculative_cap_is_passed_to_the_hot_cache(self):
        for cap in ("40", "60"):
            with self.subTest(cap=cap):
                options = self.sized_options(
                    True, {"SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS": cap}
                )
                self.assertEqual(options["graph_gather_batch_size"], 8)
                self.assertEqual(options["graph_gather_max_rows"], int(cap))

    def test_cap_below_one_verify_request_routes_is_rejected(self):
        for cap in ("16", "24", "39"):
            with (
                self.subTest(cap=cap),
                self.assertRaisesRegex(ValueError, "below one verify request's 40 routes"),
            ):
                self.sized_options(
                    True, {"SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS": cap}
                )


if __name__ == "__main__":
    unittest.main()
