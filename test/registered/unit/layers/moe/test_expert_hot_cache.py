import json
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.expert_stream import (
    NVFP4_STREAM_TENSORS,
    ExpertStreamer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertHotCache(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        self.cache_type = ExpertHotCache
        self.layer = torch.nn.Module()
        for name in NVFP4_STREAM_TENSORS[:4]:
            setattr(
                self.layer,
                name,
                torch.arange(8 * 12, dtype=torch.uint8).reshape(8, 3, 4),
            )
        self.layer.g1_alphas = torch.arange(8, dtype=torch.float32)
        self.layer.g2_alphas = torch.arange(8, dtype=torch.float32) + 10
        self.streamer = ExpertStreamer(self.layer, NVFP4_STREAM_TENSORS)

    def assert_routes(self, route_ids):
        ids = torch.tensor(route_ids, dtype=torch.int32, device="cuda")
        compact, tensors = self.streamer.gather(ids)
        self.assertEqual(compact.dtype, ids.dtype)
        self.assertEqual(compact.shape, ids.shape)
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].cpu(),
                    getattr(self.layer, name)[ids.long().cpu()],
                ),
                name,
            )
        return compact, tensors

    def test_byte_budget_counts_all_runtime_rows_including_alphas(self):
        cache = self.cache_type(self.streamer, capacity=2)
        self.assertEqual(cache.bytes_per_expert, 56)
        self.assertEqual(cache.capacity_bytes, 112)
        self.assertEqual(self.cache_type.capacity_for_budget(self.streamer, 111), 1)
        self.assertEqual(self.cache_type.capacity_for_budget(self.streamer, 112), 2)
        self.assertEqual(self.cache_type.capacity_for_budget(self.streamer, 10**9), 8)
        self.assertEqual(self.cache_type.capacity_for_budget(self.streamer, 55), 0)
        with self.assertRaises(ValueError):
            self.cache_type.capacity_for_budget(self.streamer, -1)

    def test_reassignment_preserves_pointers_and_retained_rows(self):
        cache = self.cache_type(self.streamer, capacity=2)
        pointers = cache.data_ptrs()
        first = cache.reassign([3, 7])
        self.assertEqual((first.promoted_experts, first.evicted_experts), (2, 0))
        self.assertEqual(first.migration_bytes, 112)
        retained = {name: tensor[0].clone() for name, tensor in cache.tensors.items()}
        self.layer.w13_weight[3].zero_()
        second = cache.reassign([7, 3])
        self.assertEqual(second.migration_bytes, 0)
        self.assertEqual((second.promoted_experts, second.evicted_experts), (0, 0))
        third = cache.reassign([3, 5])
        self.assertEqual((third.promoted_experts, third.evicted_experts), (1, 1))
        self.assertEqual(third.migration_bytes, 56)
        self.assertEqual(cache.data_ptrs(), pointers)
        self.assertEqual(cache.slot_to_expert, [3, 5])
        for name, value in retained.items():
            self.assertTrue(torch.equal(cache.tensors[name][0], value), name)
        slots, hits = cache.lookup(torch.tensor([3, 7, 5, 3], device="cuda"))
        self.assertEqual(slots.tolist(), [0, -1, 1, 0])
        self.assertEqual(hits.tolist(), [True, False, True, True])
        emptied = cache.reassign([])
        self.assertEqual((emptied.promoted_experts, emptied.evicted_experts), (0, 2))
        self.assertEqual(emptied.migration_bytes, 0)
        self.assertEqual(cache.data_ptrs(), pointers)
        self.assertEqual(cache.expert_to_slot.tolist(), [-1] * 8)

    def test_prefetch_uses_only_unprotected_victims_and_clamps_to_capacity(self):
        cache = self.cache_type(self.streamer, capacity=3)
        cache.reassign([1, 2, 3])

        placements = cache.prefetch_destinations([4, 5, 6], protected_slots={1, 2})

        self.assertEqual(placements, ((4, 0),))
        update = cache.assign_prefetch(placements)
        self.assertEqual(cache.slot_to_expert, [4, 2, 3])
        self.assertEqual((update.promoted_experts, update.evicted_experts), (1, 1))

    def test_reservation_hides_victim_until_matching_generation_is_ready(self):
        cache = self.cache_type(self.streamer, capacity=1)
        cache.reassign([1])

        first = cache.reserve(((2, 0),), consumer_complete=True)[0]
        self.assertEqual(cache.slot_states[0].name, "RESERVED")
        self.assertEqual(cache.resident_experts(), frozenset())
        self.assertEqual(
            cache.lookup(torch.tensor([1, 2], device="cuda"))[0].tolist(), [-1, -1]
        )
        self.assertFalse(cache.publish_ready(first))
        self.assertTrue(cache.begin_loading(first))
        self.assertEqual(cache.slot_states[0].name, "LOADING")

        self.assertTrue(cache.cancel(first))
        replacement = cache.reserve(((3, 0),))[0]
        self.assertNotEqual(first.generation, replacement.generation)
        self.assertTrue(cache.begin_loading(replacement))
        self.assertFalse(cache.publish_ready(first))
        self.assertTrue(cache.publish_ready(replacement))
        self.assertEqual(cache.slot_states[0].name, "READY")
        self.assertEqual(cache.resident_experts(), frozenset({3}))
        self.assertEqual(
            cache.lookup(torch.tensor([2, 3], device="cuda"))[0].tolist(), [-1, 0]
        )

    def test_ready_slot_waits_for_consumer_before_reuse(self):
        cache = self.cache_type(self.streamer, capacity=1)
        reservation = cache.reserve(((2, 0),))[0]
        self.assertTrue(cache.begin_loading(reservation))
        self.assertTrue(cache.publish_ready(reservation))

        self.assertFalse(cache.retire(reservation, consumer_complete=False))
        self.assertEqual(cache.slot_states[0].name, "READY")
        with self.assertRaisesRegex(RuntimeError, "active consumer"):
            cache.reserve(((3, 0),))
        self.assertTrue(cache.retire(reservation, consumer_complete=True))
        self.assertEqual(cache.slot_states[0].name, "FREE")
        self.assertEqual(cache.prefetch_destinations((3,), ()), ((3, 0),))

    def test_all_hot_returns_fixed_slots_without_compact_assembly(self):
        cache = self.cache_type(self.streamer, capacity=2)
        cache.reassign([3, 7])
        compact, tensors = self.assert_routes([[7, 3, 7]])
        self.assertEqual(compact.tolist(), [[1, 0, 1]])
        self.assertIs(tensors, cache.tensors)
        stats = self.streamer.last_gather_stats
        self.assertEqual(
            (stats.requested_rows, stats.hot_hit_rows, stats.miss_rows), (3, 3, 0)
        )
        self.assertEqual(
            (stats.d2d_bytes, stats.h2d_bytes, stats.source_bytes), (0, 0, 0)
        )

    def test_mixed_file_rows_preserve_duplicates_and_count_actual_transfers(self):
        with tempfile.NamedTemporaryFile() as backing:
            backing.truncate(8 * 12)
            mapped = torch.from_file(
                backing.name, shared=True, size=8 * 12, dtype=torch.uint8
            ).reshape(8, 3, 4)
            mapped.copy_(self.layer.w13_weight)
            self.layer.w13_weight = mapped
            cache = self.cache_type(self.streamer, capacity=2)
            cache.reassign([3, 7])
            compact, _ = self.assert_routes([[7, 1, 7, 2]])
            self.assertEqual(compact.tolist(), [[0, 1, 2, 3]])
            stats = self.streamer.last_gather_stats
            self.assertEqual(
                (stats.requested_rows, stats.hot_hit_rows, stats.miss_rows), (4, 2, 2)
            )
            self.assertEqual(stats.h2d_bytes, 112)
            self.assertEqual(stats.source_bytes, 112)
            self.assertEqual(stats.d2d_bytes, 224)

    def test_mixed_file_rows_and_cuda_alphas_use_their_own_row_spaces(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = torch.nn.Module()
        with ExitStack() as files:
            for name, dtype in zip(
                NVFP4_STREAM_TENSORS[:4],
                (
                    torch.uint8,
                    torch.uint8,
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fn,
                ),
            ):
                backing = files.enter_context(tempfile.NamedTemporaryFile())
                backing.truncate(8 * 12)
                bytes_view = torch.from_file(
                    backing.name, shared=True, size=8 * 12, dtype=torch.uint8
                ).reshape(8, 3, 4)
                bytes_view.copy_(
                    torch.arange(8 * 12, dtype=torch.uint8).reshape(8, 3, 4)
                )
                setattr(
                    layer,
                    name,
                    bytes_view if dtype is torch.uint8 else bytes_view.view(dtype),
                )
            layer.g1_alphas = torch.arange(8, device="cuda", dtype=torch.float32)
            layer.g2_alphas = layer.g1_alphas + 100
            streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            pinned = ExpertPinnedHostCache(streamer, capacity=2)
            pinned.ensure_rows(torch.tensor([5, 3], device="cuda"))

            self.assertEqual(pinned.slot_to_expert, [5, 3])
            for backend in ("gpu", "dma"):
                with self.subTest(backend=backend):
                    cache = self.cache_type(streamer, capacity=2)
                    cache.copy_backend = backend
                    cache.reassign([3, 5])
                    self.assertEqual(cache.last_copy_submission.requested_backend, backend)
                    if backend == "gpu":
                        self.assertEqual(cache.last_copy_submission.actual_backend, "mixed")
                    else:
                        self.assertIn(
                            cache.last_copy_submission.actual_backend,
                            ("mixed", "fallback"),
                        )
                    for name in NVFP4_STREAM_TENSORS:
                        actual = cache.tensors[name][0:2].cpu()
                        expected = getattr(layer, name)[torch.tensor([3, 5])].cpu()
                        if actual.dtype is torch.float8_e4m3fn:
                            actual = actual.view(torch.uint8)
                            expected = expected.view(torch.uint8)
                        self.assertTrue(torch.equal(actual, expected), name)

    def test_prefill_deduplicates_hot_and_cold_source_rows(self):
        cache = self.cache_type(self.streamer, capacity=2)
        cache.reassign([3, 7])
        routes = [[7, 1, 7, 3]] * 17
        compact, tensors = self.assert_routes(routes)
        self.assertEqual(compact.tolist(), [[2, 0, 2, 1]] * 17)
        self.assertEqual(tensors["w13_weight"].shape[0], 3)
        stats = self.streamer.last_gather_stats
        self.assertEqual(
            (stats.requested_rows, stats.hot_hit_rows, stats.miss_rows), (3, 2, 1)
        )
        self.assertEqual((stats.h2d_bytes, stats.source_bytes), (56, 56))

    def test_all_cold_and_zero_budget_preserve_uncached_results(self):
        cache = self.cache_type(self.streamer, capacity=2)
        cache.reassign([3, 7])
        compact, _ = self.assert_routes([[1, 2, 1]])
        self.assertEqual(compact.tolist(), [[0, 1, 2]])
        stats = self.streamer.last_gather_stats
        self.assertEqual((stats.hot_hit_rows, stats.miss_rows), (0, 3))
        self.assertEqual((stats.h2d_bytes, stats.source_bytes), (168, 168))
        self.assertEqual(stats.d2d_bytes, 0)
        cache = self.cache_type(self.streamer, capacity=0)
        self.assertEqual(cache.capacity_bytes, 0)
        cache.reassign([])
        self.assert_routes([[7, 1, 7]])
        self.assertEqual(self.streamer.last_gather_stats.hot_hit_rows, 0)

    def test_invalid_reassignment_does_not_change_residency(self):
        cache = self.cache_type(self.streamer, capacity=2)
        cache.reassign([3, 7])
        for invalid in ([3, 3], [0, 1, 2], [-1], [8], [1.5]):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    cache.reassign(invalid)
                self.assert_routes([[3, 7]])
        for capacity in (-1, 9, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                self.cache_type(self.streamer, capacity=capacity)

    def test_mixed_gather_preserves_float8_scale_bytes_across_kernel_tiles(self):
        layer = torch.nn.Module()
        layer.scales = (
            torch.arange(8 * 2052, dtype=torch.float32)
            .remainder(40)
            .reshape(8, 2052)
            .to(torch.float8_e4m3fn)
        )
        streamer = ExpertStreamer(layer, ("scales",))
        cache = self.cache_type(streamer, capacity=2)
        cache.reassign([3, 7])
        ids = torch.tensor([[7, 1, 3, 2]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self.assertTrue(
            torch.equal(
                tensors["scales"].view(torch.uint8)[compact.long()].cpu(),
                layer.scales.view(torch.uint8)[ids.long().cpu()],
            )
        )

    def test_all_hot_prefill_remaps_inverse_ids_to_stable_slots(self):
        cache = self.cache_type(self.streamer, capacity=2)
        cache.reassign([7, 3])
        compact, tensors = self.assert_routes([[3, 7, 3, 7]] * 17)
        self.assertEqual(compact.tolist(), [[1, 0, 1, 0]] * 17)
        self.assertIs(tensors, cache.tensors)
        self.assertEqual(self.streamer.last_gather_stats.requested_rows, 2)


class TestExpertFrequencySeed(unittest.TestCase):
    def test_stat_steps_are_summed_and_count_beats_mass(self):
        from sglang.srt.layers.moe.expert_hot_cache import (
            normalize_expert_frequency_seed,
        )

        for payload in (
            {"logical_count": [[[1, 2], [3, 4]], [[4, 3], [2, 1]]]},
            {"count": [[5, 5], [5, 5]], "mass": [[99, 0], [0, 99]], "tokens": 10},
        ):
            self.assertEqual(
                normalize_expert_frequency_seed(payload).tolist(), [[5, 5], [5, 5]]
            )
        self.assertEqual(
            normalize_expert_frequency_seed({"mass": [[1, 2]]}).tolist(), [[1, 2]]
        )

    def test_enabled_prefetch_requires_at_least_one_installable_coordinator(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

        manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
        manager.streamers = {0: object()}
        manager.caches = {}

        with self.assertRaisesRegex(ValueError, "no eligible adjacent layers"):
            manager.enable_next_layer_prefetch(1)

    def test_disabled_prefetch_is_a_noop(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

        manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
        manager.enable_next_layer_prefetch(0)

    def test_invalid_seed_is_rejected(self):
        from sglang.srt.layers.moe.expert_hot_cache import (
            normalize_expert_frequency_seed,
        )

        for payload in (
            {},
            {"count": [1, 2]},
            {"count": [[-1, 2]]},
            {"mass": [[float("nan")]]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                normalize_expert_frequency_seed(payload)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertHotCacheManager(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.manager_type = ExpertHotCacheManager
        self.mode = ForwardMode
        self.model = torch.nn.Module()
        for layer_id, width in ((0, 3), (2, 6)):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            layer._nvfp4_file_source_bytes_per_expert = 0
            for name in NVFP4_STREAM_TENSORS[:4]:
                setattr(
                    layer,
                    name,
                    torch.arange(4 * width, dtype=torch.uint8).reshape(4, width),
                )
            layer.g1_alphas = torch.ones(4)
            layer.g2_alphas = torch.ones(4)
            layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            self.model.add_module(str(layer_id), layer)

    def manager(self, seed=None, **kwargs):
        options = dict(
            budget_bytes=52,
            seed_path=None,
            dynamic=True,
            update_prefill_tokens=16,
            min_residence_forwards=2,
            benefit_ratio=2.0,
            metrics_path=None,
            route_history_limit=2,
        )
        options.update(kwargs)
        if seed is None:
            return self.manager_type.from_model(self.model, **options)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(seed, f.name)
            options["seed_path"] = f.name
            return self.manager_type.from_model(self.model, **options)

    def batch(self, mode=None, tokens=32):
        return SimpleNamespace(
            forward_mode=mode or self.mode.EXTEND, extend_num_tokens=tokens
        )

    def record_routes(self, manager, counts):
        for layer_id, streamer in manager.streamers.items():
            route_ids = [
                expert_id
                for expert_id, count in enumerate(counts[layer_id])
                for _ in range(int(count))
            ]
            if route_ids:
                streamer.gather(torch.tensor([route_ids], device="cuda"))

    def observe(self, manager, counts, *, record_routes=False, **kwargs):
        if record_routes:
            self.record_routes(manager, counts)
        manager.on_expert_distribution(
            self.batch(**kwargs), {"global_physical_count": torch.tensor(counts)}
        )

    def test_seed_allocates_complete_slots_by_global_expected_byte_savings(self):
        manager = self.manager({"count": [[9, 0, 0, 0], [0] * 4, [0, 10, 0, 0]]})
        self.assertEqual(
            {k: c.capacity for k, c in manager.caches.items()}, {0: 1, 2: 1}
        )
        self.assertEqual(manager.caches[0].slot_to_expert, [0])
        self.assertEqual(manager.caches[2].slot_to_expert, [1])
        self.assertEqual(manager.residency_bytes, 52)
        self.assertEqual(self.manager(budget_bytes=19), None)
        self.assertEqual(self.manager(budget_bytes=0), None)

    def test_copy_backend_reaches_each_cache_before_initial_population(self):
        manager = self.manager(copy_backend="dma")

        self.assertTrue(manager.caches)
        self.assertTrue(
            all(cache.copy_backend == "dma" for cache in manager.caches.values())
        )

    def test_larger_layer_can_win_entire_budget_and_json_seed_loads(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump({"count": [[0] * 4, [0] * 4, [9, 8, 7, 6]]}, f)
            f.flush()
            manager = self.manager(seed_path=f.name, budget_bytes=64)
        self.assertEqual({k: c.capacity for k, c in manager.caches.items()}, {2: 2})
        self.assertEqual(manager.caches[2].slot_to_expert, [0, 1])

    def test_dynamic_update_uses_recorded_routes_at_prefill_boundary(self):
        manager = self.manager({"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]})
        self.assertIn(0, manager.residency_policies)
        cold = [[0, 10, 0, 0], [0] * 4, [0, 10, 0, 0]]
        for mode in (self.mode.DECODE, self.mode.TARGET_VERIFY):
            self.observe(manager, cold, mode=mode, record_routes=True)
        self.observe(manager, cold, tokens=15, record_routes=True)
        self.assertEqual(manager.caches[0].slot_to_expert, [0])
        self.observe(manager, cold, record_routes=True)
        self.assertEqual(manager.caches[0].slot_to_expert, [1])
        metrics = manager.snapshot_counters()["residency_policy"]["0"]
        self.assertGreater(metrics["recorded_routes"], 0)
        self.assertEqual(metrics["boundary_updates"], 1)
        self.assertEqual(metrics["background_promotion_experts"], 1)

    def test_dynamic_scores_start_from_frequency_seed(self):
        manager = self.manager(
            {"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]},
            min_residence_forwards=0,
            benefit_ratio=0.0,
        )
        challenger = [[0, 3, 0, 0], [0] * 4, [0, 3, 0, 0]]

        self.observe(manager, challenger, record_routes=True)

        self.assertEqual(manager.caches[0].slot_to_expert, [0])
        self.assertEqual(manager.caches[2].slot_to_expert, [0])

        for _ in range(3):
            self.observe(manager, challenger, record_routes=True)

        self.assertEqual(manager.caches[0].slot_to_expert, [1])
        self.assertEqual(manager.caches[2].slot_to_expert, [1])

    def test_decode_forwards_update_residency_only_when_configured(self):
        seed = {"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]}
        challenger = [[0, 3, 0, 0], [0] * 4, [0, 3, 0, 0]]
        options = dict(min_residence_forwards=0, benefit_ratio=0.0)

        never = self.manager(seed, **options)
        for _ in range(8):
            self.observe(never, challenger, mode=self.mode.DECODE, record_routes=True)
        self.assertEqual(never.caches[0].slot_to_expert, [0])

        manager = self.manager(seed, update_decode_forwards=2, **options)
        for _ in range(3):
            self.observe(
                manager, challenger, mode=self.mode.DECODE, record_routes=True
            )
        self.assertEqual(manager.caches[0].slot_to_expert, [0])
        self.observe(manager, challenger, mode=self.mode.DECODE, record_routes=True)

        self.assertEqual(manager.caches[0].slot_to_expert, [1])
        self.assertEqual(manager.caches[2].slot_to_expert, [1])
        counters = manager.snapshot_counters()
        self.assertEqual(counters["decode"]["0"]["promotions"], 1)
        self.assertEqual(counters["decode"]["0"]["evictions"], 1)
        self.assertEqual(counters["prefill"]["0"]["evictions"], 0)
        self.assertEqual(counters["residency_policy"]["0"]["boundary_updates"], 2)

    def test_boundaries_decay_by_routed_tokens_and_restart_the_decode_count(self):
        manager = self.manager(
            {"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]},
            decay_tokens=16,
            update_decode_forwards=4,
            min_residence_forwards=100,
        )
        scores = manager.residency_policies[0]._scores
        one = [[0, 1, 0, 0], [0] * 4, [0, 1, 0, 0]]
        decode = self.mode.DECODE

        self.observe(manager, one, record_routes=True, tokens=8)
        for _ in range(4):
            self.observe(manager, one, record_routes=True, mode=decode)
        expected = [10 * 0.95 ** (12 / 16), 5.0, 0.0, 0.0]
        torch.testing.assert_close(scores.cpu(), torch.tensor(expected))

        for _ in range(2):
            self.observe(manager, one, record_routes=True, mode=decode)
        self.observe(manager, one, record_routes=True, tokens=16)
        decay = 0.95 ** (18 / 16)
        expected = [expected[0] * decay, expected[1] * decay + 3.0, 0.0, 0.0]
        torch.testing.assert_close(scores.cpu(), torch.tensor(expected))

        for _ in range(3):
            self.observe(manager, one, record_routes=True, mode=decode)
        torch.testing.assert_close(scores.cpu(), torch.tensor(expected))
        self.observe(manager, one, record_routes=True, mode=decode)
        decay = 0.95 ** (4 / 16)
        expected = [expected[0] * decay, expected[1] * decay + 4.0, 0.0, 0.0]
        torch.testing.assert_close(scores.cpu(), torch.tensor(expected))
        self.assertEqual(manager.caches[0].slot_to_expert, [0])

    def test_static_seed_never_changes_and_counts_actual_gathers_once(self):
        manager = self.manager(
            {"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]}, dynamic=False
        )
        streamer = self.model.get_submodule("0")._nvfp4_expert_streamer
        streamer.gather(torch.tensor([[0, 1, 0]], device="cuda"))
        cold = [[2, 1, 0, 0], [0] * 4, [0] * 4]
        self.observe(manager, cold, mode=self.mode.DECODE)
        self.observe(manager, cold, mode=self.mode.DECODE)
        counters = manager.snapshot_counters()
        self.assertEqual(counters["decode"]["0"]["hot_hits"], 2)
        self.assertEqual(counters["decode"]["0"]["h2d_bytes"], 20)
        self.assertEqual(counters["decode"]["0"]["d2d_bytes"], 60)
        self.assertEqual(counters["decode"]["0"]["requested_unique_experts"], 2)
        self.assertEqual(counters["decode"]["0"]["file_source_bytes"], 0)
        self.assertEqual(counters["prefill"]["0"]["promotions"], 1)
        self.assertEqual(counters["prefill"]["0"]["migration_bytes"], 20)
        json.dumps(counters, allow_nan=False)
        self.assertEqual(manager.caches[0].slot_to_expert, [0])

    def test_initial_residence_and_delta_migration_keep_retained_expert(self):
        manager = self.manager(
            {"count": [[20, 10, 0, 0], [0] * 4, [0] * 4]}, budget_bytes=40
        )
        counts = [[8, 1, 4, 0], [0] * 4, [0] * 4]
        self.observe(manager, counts, record_routes=True)
        self.assertEqual(manager.caches[0].slot_to_expert, [0, 1])
        self.observe(manager, counts, record_routes=True)
        self.assertEqual(manager.caches[0].slot_to_expert, [0, 1])
        self.observe(manager, counts, record_routes=True)
        self.observe(manager, counts, record_routes=True)
        self.assertEqual(manager.caches[0].slot_to_expert, [0, 2])
        counters = manager.snapshot_counters()["prefill"]["0"]
        self.assertEqual(counters["promotions"], 3)
        self.assertEqual(counters["evictions"], 1)
        self.assertEqual(counters["migration_bytes"], 60)

    def test_file_attribution_requires_explicit_metadata_and_writes_totals(self):
        layer = self.model.get_submodule("0")
        del layer._nvfp4_file_source_bytes_per_expert
        with tempfile.NamedTemporaryFile() as trace:
            manager = self.manager(dynamic=False, metrics_path=trace.name)
            streamer = layer._nvfp4_expert_streamer
            streamer.gather(torch.tensor([[2, 3]], device="cuda"))
            counts = [[0, 0, 1, 1], [0] * 4, [0] * 4]
            self.observe(manager, counts)
            counters = manager.snapshot_counters()["prefill"]["0"]
            self.assertIsNone(counters["file_source_bytes"])
            self.assertIsNone(counters["file_misses"])
            self.assertEqual(counters["backing_source_bytes"], 40)
            self.assertEqual(counters["h2d_bytes"], 40)
            for _ in range(98):
                self.observe(manager, counts)
            with self.assertNoLogs(
                "sglang.srt.layers.moe.expert_hot_cache", level="INFO"
            ):
                self.observe(manager, counts)
            trace.seek(0)
            logged = json.loads(trace.read().splitlines()[0])
        self.assertEqual(logged["counters"]["prefill"]["0"]["backing_source_bytes"], 40)

    def test_configured_log_interval_controls_metrics_file_emission(self):
        with tempfile.NamedTemporaryFile() as trace:
            manager = self.manager(
                dynamic=False, log_interval=2, metrics_path=trace.name
            )
            counts = [[0] * 4 for _ in range(3)]
            with self.assertNoLogs(
                "sglang.srt.layers.moe.expert_hot_cache", level="INFO"
            ):
                self.observe(manager, counts)
                self.observe(manager, counts)
            trace.seek(0)
            self.assertEqual(len(trace.read().splitlines()), 1)

    def test_phase_aware_trace_persists_bounded_route_popularity_and_affinity(self):
        with tempfile.NamedTemporaryFile() as trace:
            manager = self.manager(
                dynamic=False, log_interval=1, metrics_path=trace.name
            )
            counts = [[1, 4, 0, 0], [0] * 4, [3, 0, 2, 0]]
            self.observe(manager, counts, mode=self.mode.TARGET_VERIFY)
            self.observe(manager, counts, mode=self.mode.DRAFT_EXTEND_V2)
            self.observe(manager, counts, mode=self.mode.DECODE)
            trace.seek(0)
            records = [json.loads(line) for line in trace.read().splitlines()]
        self.assertEqual(
            [record["phase"] for record in records], ["speculative"] * 2 + ["decode"]
        )
        speculative = records[1]["route_statistics"]["speculative"]
        self.assertEqual(speculative["popularity"]["0"], [[1, 8.0], [0, 2.0]])
        self.assertEqual(speculative["affinity"]["0->2"], [[1, 0, 24.0], [1, 2, 16.0]])
        counters = records[1]["counters"]["speculative"]["0"]
        self.assertIn("pinned_hits", counters)
        self.assertIn("file_fallbacks", counters)
        self.assertIn("transfer_wait_ns", counters)

    def test_no_metrics_file_never_uses_server_logger(self):
        manager = self.manager(dynamic=False, log_interval=1)
        counts = [[0] * 4 for _ in range(3)]
        with self.assertNoLogs("sglang.srt.layers.moe.expert_hot_cache", level="INFO"):
            self.observe(manager, counts)

    def test_startup_summary_reports_actual_slots_and_cuda_memory(self):
        with self.assertLogs(
            "sglang.srt.layers.moe.expert_hot_cache", level="INFO"
        ) as logs:
            manager = self.manager(dynamic=False)
        summary = json.loads(
            logs.records[-1].getMessage().split("Expert hot cache startup ")[1]
        )
        self.assertEqual(summary["requested_bytes"], 52)
        self.assertEqual(summary["residency_bytes"], manager.residency_bytes)
        self.assertEqual(summary["slots"], 2)
        self.assertEqual(summary["layers"], 2)
        self.assertGreater(summary["cuda_allocated_bytes"], 0)
        self.assertGreaterEqual(
            summary["cuda_reserved_bytes"], summary["cuda_allocated_bytes"]
        )

    def test_observer_counts_requested_and_missed_rows_once(self):
        manager = self.manager(dynamic=False)
        streamer = self.model.get_submodule("0")._nvfp4_expert_streamer
        streamer.gather(torch.tensor([[0, 0, 3]], device="cuda"))
        counts = [[2, 0, 0, 1], [0] * 4, [0] * 4]
        self.observe(manager, counts)
        self.observe(manager, counts)
        counters = manager.snapshot_counters()["prefill"]["0"]
        self.assertEqual(counters["requested_rows"], 3)
        self.assertEqual(counters["miss_rows"], 1)
        self.assertEqual(counters["requested_unique_experts"], 2)

    def test_observer_accumulates_metrics_without_host_synchronization(self):
        manager = self.manager(dynamic=False, log_interval=1000)
        counts = torch.tensor([[2, 0, 0, 1], [0] * 4, [0, 3, 1, 0]], device="cuda")
        batch = self.batch(mode=self.mode.DECODE)
        manager.on_expert_distribution(batch, {"global_physical_count": counts})
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            manager.on_expert_distribution(batch, {"global_physical_count": counts})
        finally:
            torch.cuda.set_sync_debug_mode("default")

        statistics = manager.snapshot_route_statistics()["decode"]
        self.assertEqual(statistics["popularity"]["2"], [[1, 6.0], [2, 2.0]])
        self.assertEqual(statistics["affinity"]["0->2"], [[0, 1, 12.0], [3, 1, 6.0]])

    def test_zero_budget_ignores_seed_and_inactive_policy(self):
        self.assertIsNone(
            self.manager(
                budget_bytes=0,
                seed_path="/missing/seed",
                dynamic=True,
                update_prefill_tokens=0,
                min_residence_forwards=-1,
                benefit_ratio=float("nan"),
                log_interval=0,
            )
        )

    def test_known_file_bytes_do_not_include_other_host_sources(self):
        self.model.get_submodule("0")._nvfp4_file_source_bytes_per_expert = 12
        manager = self.manager(dynamic=False)
        streamer = self.model.get_submodule("0")._nvfp4_expert_streamer
        streamer.gather(torch.tensor([[2, 3]], device="cuda"))
        self.observe(manager, [[0, 0, 1, 1], [0] * 4, [0] * 4])
        counters = manager.snapshot_counters()["prefill"]["0"]
        self.assertEqual(counters["file_source_bytes"], 24)
        self.assertEqual(counters["file_misses"], 2)
        self.assertEqual(counters["backing_source_bytes"], 40)

    def test_invalid_config_and_seed_do_not_allocate_slots(self):
        for options in (
            {"budget_bytes": -1},
            {"update_prefill_tokens": 0},
            {"update_decode_forwards": -1},
            {"decay_tokens": -1},
            {"promotion_sigmas": float("nan")},
            {"min_residence_forwards": -1},
            {"benefit_ratio": float("nan")},
            {"benefit_ratio": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.manager(**options)
        with self.assertRaises(ValueError):
            self.manager({"count": [[1, 2, 3, 4]]})
        self.assertTrue(
            all(
                module._nvfp4_expert_streamer.hot_cache is None
                for module in self.model.children()
            )
        )


if __name__ == "__main__":
    unittest.main()
