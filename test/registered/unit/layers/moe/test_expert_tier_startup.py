"""CPU tests of the tiers' startup path: the pinned manager and inclusive hot slots."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_format, expert_hot_cache, expert_stream
from sglang.srt.layers.moe.expert_format import (
    inclusive_hot_slot_limit,
    pinned_tier_options_of,
)
from sglang.srt.layers.moe.expert_hot_cache import (
    ExpertHotCacheManager,
    HotCacheUpdateStats,
)
from sglang.srt.layers.moe.expert_stream import (
    ExpertPinnedHostCache,
    ExpertPinnedHostCacheManager,
    ExpertStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXPERTS = 8


def _reference(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "w13_trellis": torch.randint(
            -(2**15), 2**15, (EXPERTS, 2, 6), dtype=torch.int16, generator=generator
        ),
        "w2_trellis": torch.randint(
            -(2**15), 2**15, (EXPERTS, 1, 6), dtype=torch.int16, generator=generator
        ),
    }


def _model(**format_options):
    """Two spec-only layers attached the way a quantization method attaches them."""
    model = torch.nn.Module()
    references = {}
    for layer_id in range(2):
        references[layer_id] = _reference(layer_id)
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(
            layer,
            tuple(references[layer_id]),
            format=SpecOnlyFormat(references[layer_id], **format_options),
        )
        model.add_module(str(layer_id), layer)
    return model, references


class TestPinnedManagerStartup(unittest.TestCase):
    def setUp(self):
        expert_format._WARNED_WITHOUT_TIER_OPTIONS.clear()

    def test_the_manager_builds_spec_only_tiers_end_to_end(self):
        model, references = _model(tier_options={"device": "cpu"})
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        manager = ExpertPinnedHostCacheManager.from_model(
            model, budget_bytes=6 * streamer.host_bytes_per_expert
        )
        self.assertEqual(sorted(manager.caches), [0, 1])
        for layer_id, cache in manager.caches.items():
            self.assertEqual(cache.capacity, 3)
            self.assertEqual(cache.device, torch.device("cpu"))
            outputs = {
                name: torch.zeros((5,) + tuple(tensor.shape[1:]), dtype=tensor.dtype)
                for name, tensor in references[layer_id].items()
            }
            ids = torch.tensor([6, 1, 4, 0, 7])
            result = cache.gather_rows(ids, outputs)
            self.assertEqual(result.miss_rows, 5)
            for name, tensor in references[layer_id].items():
                self.assertTrue(torch.equal(outputs[name], tensor[ids]), (layer_id, name))

    def test_a_format_without_pinned_tier_options_gets_none(self):
        class NoOptions:
            key = "no_options_test"

        with self.assertLogs(
            "sglang.srt.layers.moe.expert_format", level="WARNING"
        ) as logs:
            self.assertEqual(dict(pinned_tier_options_of(NoOptions(), None)), {})
            self.assertEqual(dict(pinned_tier_options_of(NoOptions(), None)), {})
        self.assertEqual(len(logs.records), 1)
        self.assertIn("no_options_test", logs.output[0])

    def test_the_manager_tolerates_a_format_without_pinned_tier_options(self):
        model, _ = _model()
        for layer in model.children():
            streamer = layer._nvfp4_expert_streamer
            streamer.format.pinned_tier_options = None  # hide the hook
        built = []

        def fake_cache(streamer, capacity, **options):
            built.append((streamer.layer_id, capacity, options))
            return SimpleNamespace(capacity=capacity, residency_bytes=0)

        with patch.object(expert_stream, "ExpertPinnedHostCache", fake_cache):
            streamer = model.get_submodule("0")._nvfp4_expert_streamer
            ExpertPinnedHostCacheManager.from_model(
                model, budget_bytes=2 * streamer.host_bytes_per_expert
            )
        self.assertEqual(built, [(0, 1, {}), (1, 1, {})])


class TestPinnedLayerWeights(unittest.TestCase):
    """SGLANG_MOE_PINNED_HOST_LAYER_WEIGHTS splits the pinned budget by per-layer weight instead of evenly."""

    def _capacities(self, budget_rows, weights):
        model, _ = _model()
        row_bytes = model.get_submodule("0")._nvfp4_expert_streamer.host_bytes_per_expert
        built = {}

        def fake_cache(streamer, capacity, **options):
            built[streamer.layer_id] = capacity
            return SimpleNamespace(capacity=capacity, residency_bytes=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = ""
            if weights is not None:
                path = f"{tmp}/weights.json"
                with open(path, "w") as f:
                    json.dump({"layer_rows": weights}, f)
            with patch.object(expert_stream, "ExpertPinnedHostCache", fake_cache), \
                    envs.SGLANG_MOE_PINNED_HOST_LAYER_WEIGHTS.override(path):
                ExpertPinnedHostCacheManager.from_model(model, budget_bytes=budget_rows * row_bytes)
        return [built.get(layer_id, 0) for layer_id in range(2)]

    def test_unset_splits_evenly(self):
        self.assertEqual(self._capacities(7, None), [4, 3])

    def test_rows_follow_the_weights(self):
        self.assertEqual(self._capacities(8, [3, 1]), [6, 2])

    def test_the_total_matches_the_even_split(self):
        self.assertEqual(sum(self._capacities(7, [2, 5])), 7)
        self.assertEqual(self._capacities(7, [1, 1]), [4, 3])

    def test_a_layer_is_capped_at_its_experts_and_the_rest_goes_elsewhere(self):
        self.assertEqual(self._capacities(10, [9, 1]), [EXPERTS, 2])

    def test_a_weight_list_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one weight per streamed layer"):
            self._capacities(4, [1, 1, 1])

    def test_negative_or_all_zero_weights_are_refused(self):
        with self.assertRaisesRegex(ValueError, "weights"):
            self._capacities(4, [1, -1])
        with self.assertRaisesRegex(ValueError, "weights"):
            self._capacities(4, [0, 0])


class TestInclusiveHotSlotLimit(unittest.TestCase):
    def test_the_limit_is_pinned_rows_minus_gather_rows_when_opted_in(self):
        model, _ = _model(
            tier_options={"device": "cpu"}, max_gather_rows=2, inclusive_pinned_tier=True
        )
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        self.assertIsNone(inclusive_hot_slot_limit(streamer))  # no pinned tier yet
        ExpertPinnedHostCache(streamer, 5, device="cpu")
        self.assertEqual(inclusive_hot_slot_limit(streamer), 3)
        streamer.format.max_gather_rows = 7
        self.assertEqual(inclusive_hot_slot_limit(streamer), 0)
        streamer.format.inclusive_pinned_tier = False
        streamer.format.max_gather_rows = None
        self.assertIsNone(inclusive_hot_slot_limit(streamer))

    def test_the_dense_format_does_not_opt_in(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        streamer = ExpertStreamer(layer, ("rows",))
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertIsNone(inclusive_hot_slot_limit(streamer))

    def test_an_inclusive_format_with_no_positive_max_gather_rows_is_refused(self):
        model, _ = _model(
            tier_options={"device": "cpu"}, max_gather_rows=None, inclusive_pinned_tier=True
        )
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        ExpertPinnedHostCache(streamer, 5, device="cpu")
        with self.assertRaisesRegex(ValueError, "no positive max_gather_rows"):
            inclusive_hot_slot_limit(streamer)
        streamer.format.max_gather_rows = 0
        with self.assertRaisesRegex(ValueError, "no positive max_gather_rows"):
            inclusive_hot_slot_limit(streamer)


class _FakeHotCache:
    """Stands in for ExpertHotCache, which needs CUDA, in the manager's selection."""

    def __init__(self, streamer, capacity, scratch_rows=0):
        self.streamer = streamer
        self.capacity = capacity
        self.device = torch.device("cpu")
        self.capacity_bytes = capacity * streamer.bytes_per_expert
        self.allocation_bytes = self.capacity_bytes
        self.scratch_bytes = 0
        self.prefetch_pull_bytes = 0
        self.last_copy_submission = None
        self.experts = []

    def reassign(self, expert_ids):
        self.experts = list(expert_ids)
        return HotCacheUpdateStats(len(self.experts), 0, 0)


class TestInclusiveHotSelection(unittest.TestCase):
    def _hot_manager(self, model, slots):
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        with (
            patch.object(expert_hot_cache, "ExpertHotCache", _FakeHotCache),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            return ExpertHotCacheManager.from_model(
                model,
                budget_bytes=slots * streamer.bytes_per_expert,
                seed_path=None,
                dynamic=False,
                update_prefill_tokens=16,
                min_residence_forwards=0,
                benefit_ratio=1.0,
            )

    def _pinned(self, model, rows):
        for layer_id, capacity in rows.items():
            streamer = model.get_submodule(str(layer_id))._nvfp4_expert_streamer
            ExpertPinnedHostCache(streamer, capacity, device="cpu")

    def test_without_inclusion_the_budget_splits_evenly(self):
        model, _ = _model(tier_options={"device": "cpu"}, max_gather_rows=2)
        self._pinned(model, {0: 5, 1: 8})
        manager = self._hot_manager(model, 8)
        self.assertEqual(
            {layer_id: cache.capacity for layer_id, cache in manager.caches.items()},
            {0: 4, 1: 4},
        )

    def test_an_inclusive_layer_is_clamped_and_its_slots_go_elsewhere(self):
        model, _ = _model(
            tier_options={"device": "cpu"}, max_gather_rows=2, inclusive_pinned_tier=True
        )
        self._pinned(model, {0: 5, 1: 8})
        with self.assertLogs(
            "sglang.srt.layers.moe.expert_hot_cache", level="INFO"
        ) as logs:
            manager = self._hot_manager(model, 8)
        self.assertEqual(
            {layer_id: cache.capacity for layer_id, cache in manager.caches.items()},
            {0: 3, 1: 5},
        )
        clamp_lines = [line for line in logs.output if "inclusive pinned tier" in line]
        self.assertEqual(len(clamp_lines), 1)
        self.assertIn("layer 0", clamp_lines[0])
        for layer_id, cache in manager.caches.items():
            streamer = model.get_submodule(str(layer_id))._nvfp4_expert_streamer
            self.assertLessEqual(
                cache.capacity, streamer.pinned_host_cache.capacity - 2
            )


if __name__ == "__main__":
    unittest.main()
