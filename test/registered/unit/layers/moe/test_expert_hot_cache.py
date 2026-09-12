import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
