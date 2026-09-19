"""CPU tests for the pinned host expert tier: slot LRU, slabs and chunked gathers."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_host_tier import (
    PAGE_BYTES,
    PinnedSlotLRU,
    allocate_host_slab,
)
from sglang.srt.layers.moe.expert_stream import (
    ExpertPinnedHostCache,
    ExpertPinnedHostCacheManager,
    ExpertStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _LegacyLRU:
    """The pinned tier's slot choice before PinnedSlotLRU (expert_stream.py at e54c84a7c6)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.slot_to_expert = [-1] * capacity
        self.expert_to_slot = {}
        self.last_used = [0] * capacity
        self.clock = 0
        self.evictions = 0

    def touch(self, expert_id):
        self.clock += 1
        self.last_used[self.expert_to_slot[expert_id]] = self.clock

    def admit(self, missing):
        for expert_id in missing:
            free = next(
                (slot for slot, resident in enumerate(self.slot_to_expert) if resident < 0),
                None,
            )
            evicted = free is None
            if free is None:
                free = min(range(self.capacity), key=self.last_used.__getitem__)
            if evicted:
                self.evictions += 1
                self.expert_to_slot.pop(self.slot_to_expert[free], None)
            self.slot_to_expert[free] = expert_id
            self.expert_to_slot[expert_id] = free
            self.clock += 1
            self.last_used[free] = self.clock


class TestPinnedSlotLRU(unittest.TestCase):
    def test_free_slots_go_lowest_first_then_the_least_recent_is_evicted(self):
        lru = PinnedSlotLRU(3)
        self.assertEqual([lru.assign(expert)[0] for expert in (7, 2, 9)], [0, 1, 2])
        lru.touch(7)
        self.assertEqual(lru.assign(4), (1, 2))
        self.assertEqual(lru.slot_to_expert, [7, 4, 9])
        self.assertNotIn(2, lru)
        self.assertEqual(lru.mapping(10)[4], 1)
        self.assertEqual(lru.mapping(10)[2], -1)

    def test_release_frees_the_slot_for_the_next_assignment(self):
        lru = PinnedSlotLRU(2)
        lru.assign(5)
        lru.assign(6)
        lru.release(0)
        self.assertNotIn(5, lru)
        self.assertEqual(lru.assign(8), (0, None))

    def test_is_pinned_protects_experts_from_eviction(self):
        lru = PinnedSlotLRU(2, is_pinned=lambda expert: expert == 1)
        lru.assign(1)
        lru.assign(2)
        self.assertEqual(lru.assign(3), (1, 2))
        self.assertIn(1, lru)
        everything = PinnedSlotLRU(1, is_pinned=lambda expert: True)
        everything.assign(0)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            everything.assign(1)

    def test_the_calls_own_experts_are_evicted_last(self):
        lru = PinnedSlotLRU(3)
        for expert in (4, 5, 6):
            lru.assign(expert)
        self.assertEqual(lru.assign(7, protected={4, 7}), (1, 5))
        self.assertEqual(lru.assign(8, protected={6, 7, 8}), (0, 4))
        # Only the call's own experts are left: the oldest of them goes.
        self.assertEqual(lru.assign(9, protected={6, 7, 8, 9}), (2, 6))

    def test_random_traffic_matches_the_legacy_slot_choice(self):
        generator = random.Random(4)
        legacy, lru = _LegacyLRU(3), PinnedSlotLRU(3)
        evictions = 0
        for _ in range(500):
            request = generator.sample(range(12), generator.randint(1, 5))
            for expert in request:
                if expert in legacy.expert_to_slot:
                    legacy.touch(expert)
                if expert in lru:
                    lru.touch(expert)
            missing = [expert for expert in request if expert not in legacy.expert_to_slot]
            self.assertEqual(missing, [expert for expert in request if expert not in lru])
            legacy.admit(missing)
            for expert in missing:
                evictions += lru.assign(expert)[1] is not None
            self.assertEqual(lru.slot_to_expert, legacy.slot_to_expert)
            self.assertEqual(evictions, legacy.evictions)


class TestHostSlab(unittest.TestCase):
    def test_slabs_are_page_aligned_and_sized_exactly(self):
        slab = allocate_host_slab(3, (1000,), torch.float32, register=False)
        self.assertEqual(tuple(slab.shape), (3, 1000))
        self.assertEqual(slab.dtype, torch.float32)
        self.assertEqual(slab.data_ptr() % PAGE_BYTES, 0)
        self.assertTrue(slab.is_contiguous())
        self.assertEqual(slab.untyped_storage().nbytes(), 3 * 1000 * 4 + PAGE_BYTES)

    def test_empty_and_scalar_rows(self):
        self.assertEqual(
            tuple(allocate_host_slab(0, (4,), torch.uint8, register=False).shape), (0, 4)
        )
        self.assertEqual(
            tuple(allocate_host_slab(5, (), torch.float32, register=False).shape), (5,)
        )


def _host_layer(experts=8):
    layer = torch.nn.Module()
    layer.host_rows = torch.nn.Parameter(
        torch.arange(experts * 12, dtype=torch.uint8).reshape(experts, 3, 4),
        requires_grad=False,
    )
    layer._nvfp4_file_source_bytes_per_expert = 12
    return layer


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


class TestCpuPinnedTier(unittest.TestCase):
    def test_ensure_and_copy_rows_on_a_cpu_tier(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertIs(streamer.pinned_host_cache, cache)
        self.assertEqual(cache.tensors["host_rows"].data_ptr() % PAGE_BYTES, 0)
        cache.ensure_rows(torch.tensor([1, 3]))
        self.assertEqual(cache.slot_to_expert, [1, 3])
        self.assertEqual(cache._expert_to_slot.get(3, -1), 1)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        self.assertFalse(cache.copy_rows(torch.tensor([3, 1]), {"host_rows": output}))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[3, 1]]))
        cache.close()

    def test_gather_rows_keeps_the_legacy_counts(self):
        layer = _host_layer(experts=4)
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        output = torch.zeros(3, 3, 4, dtype=torch.uint8)
        first = cache.gather_rows(torch.tensor([1, 3, 1]), {"host_rows": output})
        self.assertEqual(first.hit_rows, 0)
        self.assertEqual(first.miss_rows, 3)
        self.assertEqual(first.populated_bytes, 24)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[1, 3, 1]]))
        second = cache.gather_rows(torch.tensor([1, 2, 1]), {"host_rows": output})
        self.assertEqual((second.hit_rows, second.miss_rows), (2, 1))
        self.assertEqual(second.populated_bytes, 12)
        self.assertEqual(cache.stats.populated_rows, 3)
        self.assertEqual(cache.stats.evictions, 1)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[1, 2, 1]]))

    def test_misses_beyond_the_capacity_are_copied_correctly(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        ids = torch.tensor([5, 0, 4, 2, 1])
        output = torch.zeros(5, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(ids, {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 5))
        self.assertTrue(torch.equal(output, layer.host_rows.data[ids]))
        self.assertEqual(cache.stats.evictions, 3)

    def test_is_pinned_rows_survive_admissions(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        cache.ensure_rows(torch.tensor([3]))
        self.assertEqual(cache.slot_to_expert, [1, 3])

    def test_a_protected_resident_shrinks_the_chunks(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1]))
        self.assertEqual(cache.evictable_rows(), 1)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([5, 7]), {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 2))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[5, 7]]))
        self.assertEqual(cache._expert_to_slot.get(1), 0)
        self.assertEqual(cache.slot_to_expert, [1, 7])

    def test_promotion_chunks_of_evictable_rows_are_all_resident(self):
        # ExpertHotCache._load_reserved_in_chunks admits evictable_rows() experts per
        # chunk, and _prepare_promotion then needs every one of them in the tier.
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1]))
        chunk_rows = cache.evictable_rows()
        self.assertEqual(chunk_rows, 2)
        experts = [5, 7, 0, 3, 6]
        for start in range(0, len(experts), chunk_rows):
            chunk = experts[start : start + chunk_rows]
            cache.ensure_rows(torch.tensor(chunk))
            slots = [cache._expert_to_slot.get(expert, -1) for expert in chunk]
            self.assertTrue(all(slot >= 0 for slot in slots), (chunk, slots))
            self.assertTrue(
                torch.equal(
                    cache.tensors["host_rows"][slots], layer.host_rows.data[chunk]
                )
            )
        self.assertIn(1, cache._expert_to_slot)

    def test_every_slot_protected_leaves_the_tier_unchanged(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert in (1, 2)
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        slots = list(cache.slot_to_expert)
        mapping = dict(cache._expert_to_slot)
        device_map = cache.expert_to_slot.clone()
        populated = cache.stats.populated_rows
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.ensure_rows(torch.tensor([5]))
        self.assertEqual(cache.slot_to_expert, slots)
        self.assertEqual(dict(cache._expert_to_slot), mapping)
        self.assertTrue(torch.equal(cache.expert_to_slot, device_map))
        self.assertEqual(cache.stats.populated_rows, populated)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.gather_rows(torch.tensor([5, 7]), {"host_rows": output})
        self.assertEqual(cache.slot_to_expert, slots)

    def test_a_failed_assignment_rolls_back_the_calls_slots(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        pinned = {1, 2}
        cache = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert in pinned
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        pinned.add(5)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.ensure_rows(torch.tensor([5, 7]))
        # 5 took the free slot before 7 found no victim; neither may stay mapped.
        self.assertEqual(cache.slot_to_expert, [1, 2, -1])
        self.assertNotIn(5, cache._expert_to_slot)
        self.assertEqual(int(cache.expert_to_slot[5]), -1)
        pinned.discard(5)
        output = torch.zeros(1, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([5]), {"host_rows": output})
        self.assertEqual(result.miss_rows, 1)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[5]]))

    def test_a_failed_read_rolls_back_the_calls_slots(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        cache.ensure_rows(torch.tensor([1]))
        with patch.object(streamer, "read_host_rows", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                cache.ensure_rows(torch.tensor([4, 6]))
        # 4 took the free slot and 6 evicted 1; both slots are unread, so both are freed.
        self.assertEqual(cache.slot_to_expert, [-1, -1])
        self.assertEqual(dict(cache._expert_to_slot), {})
        self.assertTrue(torch.equal(cache.expert_to_slot, torch.full((8,), -1)))
        self.assertEqual(cache.stats.evictions, 0)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        cache.gather_rows(torch.tensor([4, 1]), {"host_rows": output})
        self.assertTrue(torch.equal(output, layer.host_rows.data[[4, 1]]))


class TestGrowingProtection(unittest.TestCase):
    """``is_pinned`` whose protected set grows as the tier admits rows.

    An inclusive hierarchy pins the experts the hot cache reserves, so rows
    become protected in the middle of a call.
    """

    def _growing_tier(self, capacity, becomes_pinned):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        pinned = set()
        cache = ExpertPinnedHostCache(
            streamer, capacity, device="cpu", is_pinned=lambda expert: expert in pinned
        )
        read = streamer.read_host_rows

        def read_and_pin(rows_cpu, destinations, destination_rows=None):
            stats = read(rows_cpu, destinations, destination_rows)
            pinned.update(e for e in rows_cpu.tolist() if becomes_pinned(e))
            return stats

        streamer.read_host_rows = read_and_pin
        return layer, cache

    def _assert_tier_consistent(self, layer, cache):
        for slot, expert in enumerate(cache.slot_to_expert):
            if expert >= 0:
                self.assertEqual(cache._expert_to_slot[expert], slot)
                self.assertEqual(int(cache.expert_to_slot[expert]), slot)
                self.assertTrue(
                    torch.equal(cache.tensors["host_rows"][slot], layer.host_rows.data[expert])
                )
        self.assertEqual(
            int((cache.expert_to_slot >= 0).sum()),
            sum(expert >= 0 for expert in cache.slot_to_expert),
        )

    def test_chunks_shrink_as_admitted_rows_become_protected(self):
        # Capacity 3; expert 0 becomes protected once admitted. A chunk size fixed
        # at the call's start (3) would make the second chunk [3, 4, 5] evict one
        # of its own rows; per-chunk sizing gives [0, 1, 2], [3, 4], [5].
        layer, cache = self._growing_tier(3, lambda expert: expert == 0)
        ids = torch.arange(6)
        output = torch.zeros(6, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(ids, {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 6))
        self.assertTrue(torch.equal(output, layer.host_rows.data[ids]))
        self.assertIn(0, cache._expert_to_slot)
        self._assert_tier_consistent(layer, cache)

    def test_a_tier_filling_with_protected_rows_fails_cleanly(self):
        # Capacity 4; even experts become protected once admitted. The chunks are
        # [0..3], [4, 5], [6]; then no slot is evictable and 7 is refused.
        layer, cache = self._growing_tier(4, lambda expert: expert % 2 == 0)
        ids = torch.arange(8)
        output = torch.zeros(8, 3, 4, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.gather_rows(ids, {"host_rows": output})
        self.assertTrue(torch.equal(output[:7], layer.host_rows.data[:7]))
        self.assertEqual(sorted(cache._expert_to_slot), [0, 2, 4, 6])
        self._assert_tier_consistent(layer, cache)

    def test_an_all_hit_call_needs_no_evictable_slot(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert in (1, 2)
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        self.assertEqual(cache.evictable_rows(), 0)
        output = torch.zeros(3, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([2, 1, 2]), {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (3, 0))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[2, 1, 2]]))


class TestPinnedTierOptions(unittest.TestCase):
    def test_the_manager_passes_the_format_options_to_each_tier(self):
        def pinned(expert):
            return expert == 3

        class OptionsFormat(DenseLayerFormat):
            def pinned_tier_options(self, layer):
                return {"device": "cpu", "is_pinned": pinned}

        model = torch.nn.Module()
        for layer_id in range(2):
            layer = _host_layer()
            layer.layer_id = layer_id
            layer._nvfp4_expert_streamer = ExpertStreamer(
                layer, ("host_rows",), format=OptionsFormat(("host_rows",))
            )
            model.add_module(str(layer_id), layer)
        manager = ExpertPinnedHostCacheManager.from_model(model, budget_bytes=4 * 12)
        self.assertEqual(sorted(manager.caches), [0, 1])
        for cache in manager.caches.values():
            self.assertEqual(cache.capacity, 2)
            self.assertEqual(cache.device, torch.device("cpu"))
            self.assertIs(cache.is_pinned, pinned)

    def test_the_dense_format_adds_no_options(self):
        options = DenseLayerFormat(("rows",)).pinned_tier_options(torch.nn.Module())
        self.assertEqual(dict(options), {})


class TestCachedGatherPinnedOverflow(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()

    def test_cached_gather_with_more_misses_than_pinned_rows(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        ids = torch.tensor([[0, 5], [7, 2], [3, 5]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_cached(source_ids, compact_ids, ids)
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()], layer.host_rows.data[ids])
        )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (0, 5))
        self.assertEqual(stats.pinned_host_populated_bytes, 5 * 12)
        self.assertEqual(stats.source_bytes, 5 * 12)

    def test_a_pinned_only_gather_stages_one_set_of_rows(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        cache.ensure_rows(torch.tensor([5]))
        ids = torch.tensor([[0, 5], [7, 2], [3, 5]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_eager_rows(source_ids, compact_ids, ids)
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()], layer.host_rows.data[ids])
        )
        # One staging set: no hit, miss or cold buffers beside it.
        self.assertEqual({key[0] for key in expert_stream._STAGING}, {"host_rows"})
        self.assertEqual(tensors["host_rows"].shape[0], 64)
        stats = streamer.last_gather_stats
        self.assertEqual((stats.requested_rows, stats.miss_rows), (5, 5))
        # Chunks of 2 evict 5 before its own chunk reaches it, so it counts as a miss
        # (declined review finding M2).
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (0, 5))
        self.assertEqual(stats.source_bytes, 5 * 12)


if __name__ == "__main__":
    unittest.main()
