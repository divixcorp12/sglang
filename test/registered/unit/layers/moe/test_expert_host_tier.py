"""CPU tests for the pinned host expert tier: slot LRU, slabs and chunked gathers."""

import gc
import heapq
import json
import random
import tempfile
import unittest
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import engram_row_cache
from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_host_tier import (
    PAGE_BYTES,
    PinnedSlotLRU,
    allocate_host_slab,
    quarantined_slab_count,
    tier_snapshot,
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


class _UncountedLRU:
    """PinnedSlotLRU as it was before its counters (expert_host_tier.py at 099eadba33)."""

    def __init__(self, capacity, is_pinned=None):
        self.capacity = int(capacity)
        self.is_pinned = is_pinned
        self.slot_to_expert = [-1] * self.capacity
        self.expert_to_slot = OrderedDict()
        self._free = list(range(self.capacity))
        heapq.heapify(self._free)

    def __contains__(self, expert_id):
        return expert_id in self.expert_to_slot

    def touch(self, expert_id):
        self.expert_to_slot.move_to_end(expert_id)

    def assign(self, expert_id, protected=frozenset()):
        if expert_id in self.expert_to_slot:
            raise ValueError(f"expert {expert_id} already holds a pinned slot")
        evicted = None
        if self._free:
            slot = heapq.heappop(self._free)
        else:
            evicted = self._victim(protected)
            slot = self.expert_to_slot.pop(evicted)
        self.slot_to_expert[slot] = expert_id
        self.expert_to_slot[expert_id] = slot
        return slot, evicted

    def _victim(self, protected):
        fallback = None
        for expert_id in self.expert_to_slot:
            if self.is_pinned is not None and self.is_pinned(expert_id):
                continue
            if expert_id not in protected:
                return expert_id
            if fallback is None:
                fallback = expert_id
        if fallback is not None:
            return fallback
        raise RuntimeError("every pinned host slot holds a protected expert")

    def release(self, slot):
        expert_id = self.slot_to_expert[slot]
        if expert_id < 0:
            return
        self.expert_to_slot.pop(expert_id, None)
        self.slot_to_expert[slot] = -1
        heapq.heappush(self._free, slot)


class TestPinnedSlotLRUCounters(unittest.TestCase):
    def test_counters_start_at_zero(self):
        lru = PinnedSlotLRU(3)
        self.assertEqual(
            lru.stats(),
            {
                "capacity": 3,
                "occupancy": 0,
                "hits": 0,
                "admissions": 0,
                "evictions": 0,
                "protected_evictions": 0,
                "releases": 0,
            },
        )

    def test_counters_follow_admission_hit_eviction_and_release(self):
        lru = PinnedSlotLRU(2)
        lru.assign(1)
        lru.assign(2)
        self.assertEqual((lru.stats()["admissions"], lru.stats()["evictions"]), (2, 0))
        lru.touch(1)
        lru.touch(1)
        self.assertEqual(lru.stats()["hits"], 2)
        lru.assign(3)  # evicts 2, the oldest
        stats = lru.stats()
        self.assertEqual((stats["admissions"], stats["evictions"], stats["occupancy"]), (3, 1, 2))
        self.assertEqual(stats["protected_evictions"], 0)
        lru.release(0)
        stats = lru.stats()
        self.assertEqual((stats["releases"], stats["occupancy"]), (1, 1))
        lru.release(0)  # already free: not a release
        self.assertEqual(lru.stats()["releases"], 1)

    def test_an_eviction_of_the_calls_own_expert_is_counted_apart(self):
        lru = PinnedSlotLRU(2)
        lru.assign(1)
        lru.assign(2)
        lru.assign(3, protected={1, 2, 3})
        stats = lru.stats()
        self.assertEqual((stats["evictions"], stats["protected_evictions"]), (1, 1))

    def test_a_refused_assignment_counts_nothing(self):
        lru = PinnedSlotLRU(1, is_pinned=lambda expert: True)
        lru.assign(0)
        before = lru.stats()
        with self.assertRaises(RuntimeError):
            lru.assign(1)
        with self.assertRaises(ValueError):
            lru.assign(0)
        self.assertEqual(lru.stats(), before)

    def test_counting_changes_no_decision(self):
        # The counted table and a verbatim copy of the uncounted one see the same
        # random traffic (assign with and without protection, touch, release, a
        # pinned filter): every return value, the slot list and the eviction order
        # must match, and the counters must agree with what was observed.
        generator = random.Random(11)
        pinned = {0, 5}
        counted = PinnedSlotLRU(6, is_pinned=pinned.__contains__)
        plain = _UncountedLRU(6, is_pinned=pinned.__contains__)
        assigned = evicted_seen = touched = released = 0
        for _ in range(4000):
            action = generator.random()
            expert = generator.randrange(20)
            if action < 0.45:
                if expert in plain:
                    counted.touch(expert)
                    plain.touch(expert)
                    touched += 1
                continue
            if action < 0.9:
                protected = frozenset(generator.sample(range(20), generator.randint(0, 8)))
                if expert in plain:
                    with self.assertRaises(ValueError):
                        counted.assign(expert, protected)
                    continue
                try:
                    expected = plain.assign(expert, protected)
                except RuntimeError:
                    with self.assertRaises(RuntimeError):
                        counted.assign(expert, protected)
                    continue
                self.assertEqual(counted.assign(expert, protected), expected)
                assigned += 1
                evicted_seen += expected[1] is not None
            else:
                slot = generator.randrange(6)
                released += plain.slot_to_expert[slot] >= 0
                plain.release(slot)
                counted.release(slot)
            self.assertEqual(counted.slot_to_expert, plain.slot_to_expert)
            self.assertEqual(list(counted.expert_to_slot.items()), list(plain.expert_to_slot.items()))
            self.assertEqual(counted._free, plain._free)
        stats = counted.stats()
        self.assertEqual(
            (stats["hits"], stats["admissions"], stats["evictions"], stats["releases"]),
            (touched, assigned, evicted_seen, released),
        )
        self.assertEqual(stats["occupancy"], len(plain.expert_to_slot))
        self.assertGreater(evicted_seen, 100)


class TestQuarantine(unittest.TestCase):
    """LEASE_PROTOCOL.md section 14: a tier whose GPU readers are uncertain is never unregistered or freed."""

    def _cache(self, released):
        streamer = ExpertStreamer(_host_layer(experts=4), ("host_rows",))
        with patch.object(expert_stream, "release_host_slabs", released.append):
            return ExpertPinnedHostCache(streamer, 2, device="cpu")

    def test_close_releases_the_slabs_once(self):
        released = []
        self._cache(released).close()
        self.assertEqual(len(released), 1)

    def test_a_quarantined_tier_is_never_released_not_even_at_collection(self):
        released = []
        cache = self._cache(released)
        cache.quarantine()
        cache.close()
        del cache
        gc.collect()
        self.assertEqual(released, [])

    def test_quarantined_slabs_outlive_the_cache_and_the_module_list(self):
        released = []
        cache = self._cache(released)
        slab = weakref.ref(next(iter(cache.tensors.values())))
        before = quarantined_slab_count()
        cache.quarantine()
        self.assertEqual(quarantined_slab_count(), before + len(cache.tensors))
        del cache
        gc.collect()
        # Interpreter finalization clears module globals: only the extra reference can keep the slab.
        from sglang.srt.layers.moe import expert_host_tier

        kept = list(expert_host_tier._QUARANTINED)
        expert_host_tier._QUARANTINED.clear()
        gc.collect()
        self.assertIsNotNone(slab())
        expert_host_tier._QUARANTINED.extend(kept)

    def test_an_unquarantined_slab_is_freed_with_its_cache(self):
        released = []
        cache = self._cache(released)
        slab = weakref.ref(next(iter(cache.tensors.values())))
        del cache
        gc.collect()
        self.assertIsNone(slab())


class TestTierSnapshot(unittest.TestCase):
    def test_the_snapshot_adds_the_tier_route_counters(self):
        layer = _host_layer(experts=4)
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        output = torch.zeros(3, 3, 4, dtype=torch.uint8)
        cache.gather_rows(torch.tensor([1, 3, 1]), {"host_rows": output})
        cache.gather_rows(torch.tensor([1, 2, 1]), {"host_rows": output})
        lru = cache._lru
        # Table-level: 3 rows admitted (1, 3, 2), one evicted (3), routes 1 and 1 hit.
        self.assertEqual(
            (lru.stats()["admissions"], lru.stats()["evictions"], lru.stats()["hits"]),
            (3, 1, 2),
        )
        # Tier-level, route units: the first call misses all 3 routes, the second hits 2 of 3.
        self.assertEqual(
            (cache.stats.lookup_hits, cache.stats.lookup_misses), (2, 4)
        )
        snapshot = tier_snapshot()
        mine = lru.stats()
        # Other tests' tables may be alive: the snapshot sums at least this one.
        self.assertGreaterEqual(snapshot["admissions"], mine["admissions"])
        self.assertGreaterEqual(snapshot["lookup_misses"], 4)
        self.assertGreaterEqual(snapshot["populated_bytes"], cache.stats.populated_bytes)
        self.assertIn(mine["occupancy"], snapshot["layers"]["occupancy"])
        cache.close()


class TestCacheStatsSink(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trace = f"{self.directory.name}/trace.jsonl"
        self.addCleanup(setattr, engram_row_cache, "_SINK", engram_row_cache._SINK)
        self.addCleanup(setattr, engram_row_cache, "_SINK_PATH", engram_row_cache._SINK_PATH)
        engram_row_cache._SINK = None
        engram_row_cache._SINK_PATH = ""

    def lines(self):
        with open(self.trace + ".cache-stats") as f:
            return [json.loads(line) for line in f]

    def test_no_trace_path_means_no_sink_and_no_file(self):
        with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(""):
            lru = PinnedSlotLRU(2)
        self.assertIsNone(lru._sink)
        lru.assign(1)
        lru.touch(1)

    def test_snapshots_are_throttled_and_cumulative(self):
        with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(self.trace):
            lru = PinnedSlotLRU(4)
            lru.assign(1)
            for _ in range(50):
                lru.touch(1)
            # One write per interval: the first call wrote, the rest were throttled.
            self.assertEqual(len(self.lines()), 1)
            lru._sink._interval_s = 0.0
            lru._sink._next.clear()
            lru.assign(2)
            lru.touch(2)
        lines = self.lines()
        self.assertEqual([line["kind"] for line in lines], ["pinned_tier"] * 3)
        self.assertEqual(lines[0]["admissions"], 1)
        self.assertEqual(lines[-1]["admissions"], 2)
        self.assertEqual(lines[-1]["hits"], 51)
        self.assertEqual(lines[-1]["occupancy"], 2)
        self.assertLessEqual(lines[0]["t"], lines[-1]["t"])

    def test_a_sink_writes_no_decision_of_its_own(self):
        with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(self.trace):
            counted = PinnedSlotLRU(3)
            counted._sink._interval_s = 0.0
            plain = _UncountedLRU(3)
            generator = random.Random(3)
            for _ in range(300):
                expert = generator.randrange(8)
                if expert in plain:
                    counted.touch(expert)
                    plain.touch(expert)
                else:
                    self.assertEqual(counted.assign(expert), plain.assign(expert))
        self.assertEqual(counted.slot_to_expert, plain.slot_to_expert)
        self.assertGreater(len(self.lines()), 100)



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

    def test_gather_rows_raises_when_a_chunk_expert_is_evicted_before_its_copy(self):
        # ensure_rows protects the whole chunk, so this can only happen if
        # something outside the call's own bookkeeping evicts a chunk member
        # mid-admission; force that race by patching _lru.assign.
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        cache.ensure_rows(torch.tensor([1]))
        original_assign = cache._lru.assign

        def sneaky_assign(expert_id, protected):
            slot, evicted = original_assign(expert_id, protected)
            if expert_id == 5:
                victim_slot = cache._lru.expert_to_slot.pop(1, None)
                if victim_slot is not None:
                    cache._lru.slot_to_expert[victim_slot] = -1
                    heapq.heappush(cache._lru._free, victim_slot)
            return slot, evicted

        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        with patch.object(cache._lru, "assign", side_effect=sneaky_assign):
            with self.assertRaisesRegex(RuntimeError, "evicted before their copy"):
                cache.gather_rows(torch.tensor([1, 5]), {"host_rows": output})

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
