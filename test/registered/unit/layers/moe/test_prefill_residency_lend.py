"""Layer-major expert residency over lent late-layer hot-cache slots (prefill_residency.LentPrefillResidency) (CPU)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layer_major.driver import run_pass
from sglang.srt.layers.moe import expert_stream, hot_cache_retire
from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.srt.layers.moe.prefill_residency import Exl3LendingLayerMajorResidency
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERTS = 16
FREE, READY = 0, 3


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


class _Tier:
    """The donor layer's pinned tier: serves any expert's row from the layer's source."""

    cached_names = ("rows",)
    capacity = EXPERTS
    expert_to_slot = torch.arange(EXPERTS)

    def __init__(self, layer):
        self.layer = layer
        self.fail = False
        self.requests = []

    def gather_rows(self, ids, outputs):
        if self.fail:
            raise RuntimeError("tier read failed")
        self.requests.append(ids.tolist())
        for name, output in outputs.items():
            output.copy_(getattr(self.layer, name)[ids])
        return SimpleNamespace(hit_rows=ids.numel(), miss_rows=0, populated_bytes=0, fallback_used=False)


def _streamer(layer_id):
    layer = torch.nn.Module()
    layer.layer_id = layer_id
    layer.rows = torch.arange(EXPERTS * 4, dtype=torch.int32).reshape(EXPERTS, 4) + 1000 * layer_id
    streamer = ExpertStreamer(layer, ("rows",))
    streamer.hot_cache = SimpleNamespace(
        lookup=_all_miss, capacity=1, expert_to_slot=torch.full((EXPERTS,), -1), device=torch.device("cpu")
    )
    reads = []

    def copy_rows(source_ids, outputs):
        reads.append(source_ids.tolist())
        for name, output in outputs.items():
            torch.index_select(getattr(layer, name), 0, source_ids.long(), out=output)
        return 0

    streamer._copy_source_rows = copy_rows
    return layer, streamer, reads


class _Manager:
    prefill_lend_active = False

    def __init__(self):
        self.caches = {}
        self.gpu_residency = None
        self.quiesced = 0
        self.lend_calls = []

    def quiesce_for_prefill_lend(self):
        self.quiesced += 1

    def set_prefill_lend(self, active):
        self.lend_calls.append(active)
        self.prefill_lend_active = active


def _world(donors, early=(0, 1)):
    """A manager over hot-cached early layers and donor layers ``{layer: [expert per slot, -1 for FREE]}``.

    Donor slot s holds its expert's row, as a promotion wrote it, in the device bank the GPU residency update owns."""
    manager = _Manager()
    streamers, reads = {}, {}
    for layer_id in early:
        _, streamer, reads[layer_id] = _streamer(layer_id)
        streamers[layer_id] = streamer
        manager.caches[layer_id] = SimpleNamespace(capacity=1, streamer=streamer)
    slots = max((len(experts) for experts in donors.values()), default=0)
    layer_ids = list(donors)
    bank = SimpleNamespace(
        layer_ids=layer_ids,
        slot_to_expert=torch.full((len(layer_ids), slots + 1), -1, dtype=torch.long),
        slot_state=torch.zeros((len(layer_ids), slots + 1), dtype=torch.uint8),
        slot_generations=torch.zeros((len(layer_ids), slots + 1), dtype=torch.long),
        mapping=torch.full((len(layer_ids), EXPERTS + 1), -1, dtype=torch.long),
    )
    manager.gpu_residency = bank
    layers, tiers = {}, {}
    for row, (layer_id, experts) in enumerate(donors.items()):
        layer, streamer, _ = _streamer(layer_id)
        tier = _Tier(layer)
        streamer.pinned_host_cache = tier
        capacity = len(experts)
        tensors = {"rows": torch.zeros((capacity, 4), dtype=torch.int32)}
        for slot, expert in enumerate(experts):
            if expert >= 0:
                tensors["rows"][slot] = layer.rows[expert]
                bank.mapping[row, expert] = slot
                bank.slot_state[row, slot] = READY
                bank.slot_to_expert[row, slot] = expert
                bank.slot_generations[row, slot] = 10 * layer_id + slot
        manager.caches[layer_id] = SimpleNamespace(
            capacity=capacity,
            streamer=streamer,
            tensors=tensors,
            slot_to_expert=list(experts),
            slot_generations=bank.slot_generations[row, :capacity],
            expert_to_slot=bank.mapping[row, :EXPERTS],
            slot_state=bank.slot_state[row, :capacity],
            device_residency=bank,
        )
        layers[layer_id], tiers[layer_id] = layer, tier
    return manager, streamers, reads, layers, tiers


def _bytes(manager):
    return {layer: cache.tensors["rows"].clone() for layer, cache in manager.caches.items() if hasattr(cache, "tensors")}


def _maps(manager):
    bank = manager.gpu_residency
    return [t.clone() for t in (bank.mapping, bank.slot_to_expert, bank.slot_state, bank.slot_generations)]


class _Base(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()
        patch = mock.patch.object(hot_cache_retire, "_service", lambda: None)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        expert_stream._STAGING.clear()

    def residency(self, manager, rows):
        return Exl3LendingLayerMajorResidency(SimpleNamespace(expert_hot_cache_manager=manager), rows)


class TestLendPlan(_Base):
    def test_slots_are_taken_from_the_last_layer_down_in_slot_order(self):
        manager, *_ = _world({21: [0, 1, 2], 22: [3, 4], 23: [5, 6, 7, 8]})
        residency = self.residency(manager, rows=6)
        residency.begin(range(0, 2))
        plan = [(row.layer_id, row.slot) for row in residency.buffer.plan]
        self.assertEqual(plan, [(23, 0), (23, 1), (23, 2), (23, 3), (22, 0), (22, 1)])
        self.assertEqual(manager.quiesced, 1)
        self.assertEqual(manager.lend_calls, [True])

    def test_fewer_lendable_slots_than_rows_makes_a_smaller_buffer(self):
        manager, *_ = _world({21: [0, 1], 22: [2]})
        residency = self.residency(manager, rows=8)
        residency.begin(range(0, 2))
        self.assertEqual(residency.buffer.rows, 3)

    def test_the_early_layers_never_donate(self):
        manager, *_ = _world({21: [0, 1]})
        residency = self.residency(manager, rows=8)
        residency.begin(range(0, 2))
        self.assertEqual({row.layer_id for row in residency.buffer.plan}, {21})

    def test_rows_never_exceed_the_expert_count(self):
        manager, *_ = _world({21: list(range(EXPERTS)), 22: list(range(EXPERTS))})
        residency = self.residency(manager, rows=1000)
        residency.begin(range(0, 2))
        self.assertEqual(residency.buffer.rows, EXPERTS)

    def test_no_donor_leaves_the_pass_unlent(self):
        manager, *_ = _world({})
        residency = self.residency(manager, rows=4)
        residency.begin(range(0, 2))
        residency.make_resident(0)
        residency.restore()
        residency.end()
        self.assertIsNone(residency.buffer)
        self.assertEqual(manager.lend_calls, [])

    def test_a_slot_mid_promotion_is_not_lent(self):
        manager, *_ = _world({21: [0, 1, 2]})
        manager.gpu_residency.slot_state[0, 1] = 2  # LOADING
        residency = self.residency(manager, rows=8)
        residency.begin(range(0, 2))
        self.assertEqual([row.slot for row in residency.buffer.plan], [0, 2])


class TestPassThroughLentSlots(_Base):
    def _pass(self, manager, residency):
        """Layer 0 sees two chunks that route to overlapping experts; returns what it read from its source."""
        residency.begin(range(0, 2))
        residency.make_resident(0)
        streamer = manager.caches[0].streamer
        gathered = []
        for ids in ([1, 3, 5], [3, 5, 7], [1, 7]):
            row_of_source, rows = streamer._gather_experts(torch.tensor(ids))
            gathered.append(rows["rows"][row_of_source.long()].clone())
        return gathered

    def test_chunks_read_each_expert_once_and_stage_the_same_bytes(self):
        manager, streamers, reads, *_ = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        residency = self.residency(manager, rows=6)
        layer0 = streamers[0].layer
        gathered = self._pass(manager, residency)
        for ids, rows in zip(([1, 3, 5], [3, 5, 7], [1, 7]), gathered):
            self.assertTrue(torch.equal(rows, layer0.rows[ids]))
        self.assertEqual(reads[0], [[1, 3, 5], [7]])
        stats = streamers[0].last_gather_stats
        self.assertEqual((stats.miss_rows, stats.residency_hit_rows), (0, 2))

    def test_restore_puts_every_original_expert_back_and_leaves_the_maps_alone(self):
        manager, _, _, _, tiers = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        before, maps = _bytes(manager), _maps(manager)
        residency = self.residency(manager, rows=6)
        self._pass(manager, residency)
        self.assertFalse(torch.equal(manager.caches[22].tensors["rows"], before[22]))  # the lend overwrote them
        residency.restore()
        self.assertTrue(manager.prefill_lend_active is False)
        for layer in (21, 22):
            self.assertTrue(torch.equal(manager.caches[layer].tensors["rows"], before[layer]), layer)
        for now, then in zip(_maps(manager), maps):
            self.assertTrue(torch.equal(now, then))
        self.assertEqual(residency.buffer.evicted_rows, 0)
        self.assertEqual(residency.buffer.reloaded_rows, 4)  # chunk 1 and 2 adopted 4 distinct experts (1,3,5,7)
        residency.end()

    def test_only_written_slots_are_reloaded(self):
        manager, _, _, _, tiers = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        residency = self.residency(manager, rows=6)
        self._pass(manager, residency)
        residency.restore()
        self.assertEqual(sum(len(r) for t in tiers.values() for r in t.requests), 4)

    def test_a_free_slot_is_lent_but_stays_free_and_is_not_reloaded(self):
        manager, _, _, layers, tiers = _world({21: [-1, -1, 2, 3]})
        before, maps = _bytes(manager), _maps(manager)
        residency = self.residency(manager, rows=4)
        self._pass(manager, residency)  # all four slots are written
        residency.restore()
        for now, then in zip(_maps(manager), maps):
            self.assertTrue(torch.equal(now, then))
        self.assertEqual(residency.buffer.evicted_rows, 0)
        self.assertEqual(tiers[21].requests, [[2, 3]])
        self.assertTrue(torch.equal(manager.caches[21].tensors["rows"][2:], before[21][2:]))

    def test_a_map_that_changed_under_the_lend_is_evicted_not_reloaded(self):
        manager, _, _, _, tiers = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        before = _bytes(manager)
        residency = self.residency(manager, rows=6)
        self._pass(manager, residency)
        bank = manager.gpu_residency
        # Layer 22 (bank row 1) is the first donor in the plan: slot 0 is promoted behind the lend's back.
        bank.slot_generations[1, 0] += 1
        residency.restore()
        self.assertEqual(residency.buffer.evicted_rows, 1)
        self.assertEqual(int(bank.slot_to_expert[1, 0]), -1)
        self.assertEqual(int(bank.slot_state[1, 0]), FREE)
        self.assertEqual(int(bank.mapping[1, 4]), -1)  # the expert it mapped is unmapped
        self.assertEqual(manager.caches[22].slot_to_expert[0], -1)
        self.assertFalse(torch.equal(manager.caches[22].tensors["rows"][0], before[22][0]))  # not reloaded over
        self.assertEqual(tiers[22].requests, [[5, 6, 7]])
        # The other lent slots were restored.
        self.assertEqual(int(bank.slot_to_expert[1, 1]), 5)
        self.assertEqual(int(bank.slot_state[1, 1]), READY)
        self.assertTrue(torch.equal(manager.caches[22].tensors["rows"][1:], before[22][1:]))

    def test_a_failed_reload_evicts_the_layers_slots_not_yet_restored(self):
        manager, _, _, _, tiers = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        residency = self.residency(manager, rows=8)
        self._pass(manager, residency)
        # Adopted: experts 1, 3, 5, 7, into plan slots (22,0) to (22,3).
        tiers[22].fail = True
        residency.restore()
        bank = manager.gpu_residency
        self.assertEqual(residency.buffer.evicted_rows, 4)
        self.assertEqual(bank.slot_state[1, :4].tolist(), [FREE] * 4)
        self.assertEqual(bank.slot_to_expert[1, :4].tolist(), [-1] * 4)
        self.assertEqual([int(bank.mapping[1, e]) for e in (4, 5, 6, 7)], [-1] * 4)
        self.assertEqual(bank.slot_state[0, :4].tolist(), [READY] * 4)
        self.assertEqual(manager.lend_calls[-1], False)

    def test_a_chunk_that_fails_midway_restores_the_chunks_before_it(self):
        manager, streamers, _, _, tiers = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        residency = self.residency(manager, rows=8)
        streamers_of = {layer: cache.streamer for layer, cache in manager.caches.items()}
        streamers_of[22]._staging_floor_rows = lambda: 2
        before = _bytes(manager)
        self._pass(manager, residency)
        calls = []
        real = tiers[22].gather_rows

        def second_chunk_fails(ids, outputs):
            calls.append(ids.tolist())
            if len(calls) == 2:
                raise RuntimeError("tier read failed")
            return real(ids, outputs)

        tiers[22].gather_rows = second_chunk_fails
        residency.restore()
        bank = manager.gpu_residency
        self.assertEqual(residency.buffer.evicted_rows, 2)
        self.assertTrue(torch.equal(manager.caches[22].tensors["rows"][:2], before[22][:2]))
        self.assertEqual(bank.slot_state[1, :4].tolist(), [READY, READY, FREE, FREE])

    def test_a_layer_without_a_pinned_tier_is_evicted(self):
        manager, _, _, _, tiers = _world({21: [0, 1, 2, 3]})
        manager.caches[21].streamer.pinned_host_cache = None
        residency = self.residency(manager, rows=4)
        self._pass(manager, residency)
        residency.restore()
        self.assertEqual(residency.buffer.evicted_rows, 4)

    def test_end_logs_and_drops_the_buffer(self):
        manager, *_ = _world({21: [0, 1, 2, 3]})
        residency = self.residency(manager, rows=4)
        self._pass(manager, residency)
        residency.restore()
        with self.assertLogs("sglang.srt.layers.moe.prefill_residency", level="INFO") as logs:
            residency.end()
        self.assertIn("reloaded", logs.output[-1])
        self.assertIsNone(residency.buffer)


class TestFailurePaths(_Base):
    class _Adapter:
        def __init__(self, fail_at=None):
            self.fail_at = fail_at
            self.events = []

        def begin_pass(self, forward_batch, schedule_batch, store):
            return "handle"

        def layer_ids(self, handle):
            return range(0, 2)

        def num_chunks(self, handle):
            return 2

        def run_layer(self, handle, layer_id, chunk, store):
            self.events.append(("run", layer_id, chunk))
            if (layer_id, chunk) == self.fail_at:
                raise RuntimeError("chunk failed")

        def finish_pass(self, handle, store):
            self.events.append(("finish",))
            return "output"

        def release_pass(self, handle, store, failed):
            self.events.append(("release", failed))

    def test_a_chunk_that_raises_still_restores_the_lent_slots_and_closes_the_lend(self):
        manager, _, _, _, _ = _world({21: [0, 1, 2, 3], 22: [4, 5, 6, 7]})
        before = _bytes(manager)
        residency = self.residency(manager, rows=6)
        adapter = self._Adapter(fail_at=(1, 0))
        # Layer 0's chunks gather through the lent slots; layer 1 then fails.
        streamer = manager.caches[0].streamer

        def run_layer(handle, layer_id, chunk, store):
            adapter.events.append(("run", layer_id, chunk))
            if layer_id == 0:
                streamer._gather_experts(torch.tensor([1, 3, 5, 7]))
            if (layer_id, chunk) == (1, 0):
                raise RuntimeError("chunk failed")

        adapter.run_layer = run_layer
        with self.assertRaisesRegex(RuntimeError, "chunk failed"):
            run_pass(adapter, residency, None, None, None)
        for layer in (21, 22):
            self.assertTrue(torch.equal(manager.caches[layer].tensors["rows"], before[layer]))
        self.assertEqual(manager.lend_calls, [True, False])
        self.assertNotIn(("finish",), adapter.events)

    def test_a_pass_that_succeeds_restores_before_the_tail_runs(self):
        manager, *_ = _world({21: [0, 1, 2, 3]})
        residency = self.residency(manager, rows=4)
        adapter = self._Adapter()
        order = []
        real_restore = residency.restore

        def restore():
            order.append(("restore", manager.prefill_lend_active))
            real_restore()

        residency.restore = restore
        run_pass(adapter, residency, None, None, None)
        self.assertEqual(order, [("restore", True)])
        self.assertEqual(adapter.events[-2:], [("finish",), ("release", False)])
        self.assertFalse(manager.prefill_lend_active)

    def test_a_restore_that_cannot_finish_leaves_the_lend_open_for_the_fail_stop_check(self):
        manager, *_ = _world({21: [0, 1, 2, 3]})
        residency = self.residency(manager, rows=4)
        residency.begin(range(0, 2))
        with mock.patch.object(type(residency.buffer), "verify", side_effect=RuntimeError("device lost")):
            with self.assertRaisesRegex(RuntimeError, "device lost"):
                residency.restore()
        self.assertTrue(manager.prefill_lend_active)


class TestManagerFailStop(unittest.TestCase):
    def _manager(self):
        manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
        manager._inflight_promotions = []
        manager.gpu_residency = SimpleNamespace(flushed=0, flush=lambda: setattr(manager, "flushed", 1))
        return manager

    def test_quiesce_settles_promotions_and_applies_the_pending_boundary(self):
        manager = self._manager()
        manager.quiesce_for_prefill_lend()
        self.assertEqual(manager.flushed, 1)

    def test_the_check_is_registered_once_and_fires_only_while_a_lend_is_open(self):
        manager = self._manager()
        manager.run_fail_stop_checks()
        manager.set_prefill_lend(True)
        manager.set_prefill_lend(False)
        manager.set_prefill_lend(True)
        self.assertEqual(len(manager.fail_stop_checks), 1)
        with self.assertRaisesRegex(RuntimeError, "still lent"):
            manager.run_fail_stop_checks()
        manager.set_prefill_lend(False)
        manager.run_fail_stop_checks()

    def test_a_boundary_or_graph_gather_while_lent_raises(self):
        manager = self._manager()
        manager.set_prefill_lend(True)
        with self.assertRaisesRegex(RuntimeError, "residency boundary"):
            manager._update_residency(0, "decode")
        with self.assertRaisesRegex(RuntimeError, "lent"):
            manager.require_no_prefill_lend("a graph gather")


if __name__ == "__main__":
    unittest.main()
