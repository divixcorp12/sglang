"""Device slot publication and asynchronous promotions of the expert hot cache.

``_ReferenceSlots`` is a frozen host model of the d0a3e55b16 per-ticket
lifecycle: evicted slots retire in slot order, promoted experts reserve the
ascending free slots, and each reservation bumps its slot generation. That
code mirrored exactly these host lists onto the device after every ticket, so
the device tensors it left behind are the ones this model computes.
"""

import json
import random
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-a", runner_config="1-gpu-small")

FREE, RESERVED, LOADING, READY = 0, 1, 2, 3


class _ReferenceSlots:
    """Frozen d0a3e55b16 host lifecycle for ``reassign`` and ``assign_prefetch``."""

    def __init__(self, num_experts, capacity):
        self.num_experts = num_experts
        self.slot_to_expert = [-1] * capacity
        self.states = [FREE] * capacity
        self.generations = [0] * capacity

    def reassign(self, desired):
        existing = {
            expert
            for expert, state in zip(self.slot_to_expert, self.states)
            if state == READY
        }
        promoted = [expert for expert in desired if expert not in existing]
        evicted = existing - set(desired)
        if not promoted and not evicted:
            return 0, 0
        for slot, expert in enumerate(self.slot_to_expert):
            if expert in evicted:
                self.slot_to_expert[slot] = -1
                self.states[slot] = FREE
        free = [slot for slot, state in enumerate(self.states) if state == FREE]
        self._reserve_ready(zip(promoted, free))
        return len(promoted), len(evicted)

    def assign(self, placements):
        for _, slot in placements:
            if self.states[slot] == READY:
                self.slot_to_expert[slot] = -1
                self.states[slot] = FREE
        self._reserve_ready(placements)

    def _reserve_ready(self, placements):
        for expert, slot in placements:
            self.generations[slot] += 1
            self.slot_to_expert[slot] = expert
            self.states[slot] = READY

    def mapping(self):
        mapping = [-1] * self.num_experts
        for slot, expert in enumerate(self.slot_to_expert):
            if self.states[slot] == READY:
                mapping[expert] = slot
        return mapping


def assert_slots_match(test, cache, reference, context=""):
    """Fail unless the cache's host lists and device tensors equal the reference."""
    test.assertEqual(cache.slot_to_expert, reference.slot_to_expert, f"{context} host experts")
    test.assertEqual([int(state) for state in cache.slot_states], reference.states, f"{context} host states")
    test.assertEqual(cache._slot_generations, reference.generations, f"{context} host generations")
    test.assertEqual(cache.expert_to_slot.tolist(), reference.mapping(), f"{context} expert_to_slot")
    test.assertEqual(cache.slot_state.tolist(), reference.states, f"{context} slot_state")
    test.assertEqual(cache.slot_generations.tolist(), reference.generations, f"{context} slot_generations")


def assert_ready_rows(test, cache, layer, context=""):
    """Fail unless every READY slot holds its expert's six source rows byte for byte."""
    for slot, (expert, state) in enumerate(zip(cache.slot_to_expert, cache.slot_states)):
        if int(state) != READY:
            continue
        for name in NVFP4_STREAM_TENSORS:
            test.assertTrue(
                torch.equal(cache.tensors[name][slot].cpu(), getattr(layer, name)[expert].cpu()),
                f"{context} slot {slot} expert {expert} {name}",
            )


def _layer(num_experts, variant):
    layer = torch.nn.Module()
    for position, name in enumerate(NVFP4_STREAM_TENSORS[:4]):
        values = (torch.arange(num_experts * 12, dtype=torch.int64) * (position + 3)).remainder(251)
        tensor = values.to(torch.uint8).reshape(num_experts, 3, 4)
        setattr(layer, name, tensor.pin_memory() if variant != "pageable" else tensor)
    alphas = torch.arange(num_experts, dtype=torch.float32)
    if variant == "cuda_alphas":
        alphas = alphas.cuda()
    elif variant == "pinned":
        alphas = alphas.pin_memory()
    layer.g1_alphas = alphas + 0.5
    layer.g2_alphas = alphas + 10.25
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestHotCacheSlotPublication(unittest.TestCase):
    num_experts = 24
    capacity = 8

    def setUp(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        self.cache_type = ExpertHotCache

    def make(self, variant):
        layer = _layer(self.num_experts, variant)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        cache = self.cache_type(streamer, capacity=self.capacity)
        return layer, cache, _ReferenceSlots(self.num_experts, self.capacity)

    def test_random_reassignments_publish_the_per_ticket_device_state(self):
        for variant in ("pageable", "pinned", "cuda_alphas"):
            generator = random.Random(variant)
            layer, cache, reference = self.make(variant)
            previous = []
            for step in range(40):
                keep = [expert for expert in previous if generator.random() < 0.6]
                pool = [expert for expert in range(self.num_experts) if expert not in keep]
                size = generator.randrange(self.capacity + 1)
                desired = (keep + generator.sample(pool, len(pool)))[:size]
                generator.shuffle(desired)
                stats = cache.reassign(desired)
                expected = reference.reassign(desired)
                context = f"{variant} step {step}"
                self.assertEqual((stats.promoted_experts, stats.evicted_experts), expected, context)
                assert_slots_match(self, cache, reference, context)
                assert_ready_rows(self, cache, layer, context)
                previous = desired
            self.assertEqual(cache.promotion_in_flight, None)

    def test_prefetch_assignments_publish_the_per_ticket_device_state(self):
        layer, cache, reference = self.make("pinned")
        cache.reassign([1, 2, 3, 4, 5, 6])
        reference.reassign([1, 2, 3, 4, 5, 6])
        for candidates, protected in (((7, 8, 9), (0,)), ((10, 11), (1, 2, 3)), ((1, 12), ())):
            placements = cache.prefetch_destinations(candidates, protected)
            cache.assign_prefetch(placements)
            reference.assign(placements)
            assert_slots_match(self, cache, reference, str(candidates))
            assert_ready_rows(self, cache, layer, str(candidates))

    def test_lifecycle_calls_outside_a_reassignment_publish_before_returning(self):
        _, cache, _ = self.make("pinned")
        cache.reassign([4])
        ticket = cache.reserve(((5, 1),))[0]
        self.assertEqual(cache.slot_state.tolist()[:2], [READY, RESERVED])
        self.assertEqual(cache.slot_generations.tolist()[:2], [1, 1])
        self.assertEqual(cache.lookup(torch.tensor([4, 5], device="cuda"))[0].tolist(), [0, -1])
        self.assertTrue(cache.begin_loading(ticket))
        self.assertEqual(cache.slot_state.tolist()[1], LOADING)
        self.assertTrue(cache.publish_ready(ticket))
        self.assertEqual(cache.lookup(torch.tensor([4, 5], device="cuda"))[0].tolist(), [0, 1])
        self.assertTrue(cache.retire(cache.ticket_for_slot(0), consumer_complete=True))
        self.assertEqual(cache.slot_state.tolist()[:2], [FREE, READY])
        self.assertEqual(cache.lookup(torch.tensor([4, 5], device="cuda"))[0].tolist(), [-1, 1])
        replacement = cache.reserve(((6, 0),))[0]
        self.assertTrue(cache.cancel(replacement))
        self.assertEqual(cache.slot_state.tolist()[0], FREE)
        self.assertEqual(cache.slot_generations.tolist()[0], 2)

    def test_staged_promotion_hides_its_slots_until_completed(self):
        layer, cache, reference = self.make("pinned")
        cache.reassign([0, 1, 2])
        reference.reassign([0, 1, 2])
        stats, promotion = cache.stage_reassign([0, 5, 6], publish=True)
        self.assertEqual((stats.promoted_experts, stats.evicted_experts), (2, 2))
        self.assertIs(cache.promotion_in_flight, promotion)
        self.assertEqual(cache.lookup(torch.tensor([0, 1, 2, 5, 6], device="cuda"))[0].tolist(), [0, -1, -1, -1, -1])
        self.assertEqual(cache.slot_state.tolist()[:3], [READY, LOADING, LOADING])
        with self.assertRaisesRegex(RuntimeError, "still in flight"):
            cache.stage_reassign([0], publish=True)
        from sglang.srt.layers.moe.expert_hot_cache import submit_hot_cache_promotions

        stream = torch.cuda.current_stream()
        ticket = submit_hot_cache_promotions([promotion], producer_stream=stream)
        cache._transfer_executor.wait(ticket, stream)
        cache.complete_promotion(promotion)
        reference.reassign([0, 5, 6])
        assert_slots_match(self, cache, reference)
        assert_ready_rows(self, cache, layer)
        with self.assertRaisesRegex(RuntimeError, "not in flight"):
            cache.complete_promotion(promotion)
        assert_slots_match(self, cache, reference)

    def test_slot_equality_helpers_fail_on_perturbed_state(self):
        layer, cache, reference = self.make("pinned")
        cache.reassign([3, 9])
        reference.reassign([3, 9])
        assert_slots_match(self, cache, reference)
        assert_ready_rows(self, cache, layer)
        perturbations = (
            lambda ref: ref.generations.__setitem__(0, 7),
            lambda ref: ref.states.__setitem__(1, LOADING),
            lambda ref: ref.slot_to_expert.__setitem__(0, 4),
        )
        for perturb in perturbations:
            copy = _ReferenceSlots(self.num_experts, self.capacity)
            copy.slot_to_expert = list(reference.slot_to_expert)
            copy.states = list(reference.states)
            copy.generations = list(reference.generations)
            perturb(copy)
            with self.assertRaises(AssertionError):
                assert_slots_match(self, cache, copy)
        cache.slot_generations[0] = 99
        with self.assertRaises(AssertionError):
            assert_slots_match(self, cache, reference)
        cache.slot_generations[0] = 1
        cache.expert_to_slot[3] = -1
        with self.assertRaises(AssertionError):
            assert_slots_match(self, cache, reference)
        cache.expert_to_slot[3] = 0
        cache.tensors["w13_weight"][1, 0, 0] += 1
        with self.assertRaises(AssertionError):
            assert_ready_rows(self, cache, layer)


class _HeldEvent:
    """A completion event whose ``query`` the test controls."""

    held = True
    instances = []

    def __init__(self):
        self.complete = False
        _HeldEvent.instances.append(self)

    def record(self, stream=None):
        self.complete = not _HeldEvent.held

    def query(self):
        return self.complete

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass

    @classmethod
    def release_all(cls):
        cls.held = False
        for event in cls.instances:
            event.complete = True


class _SameStream:
    def wait_stream(self, stream):
        pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestAsyncHotCachePromotions(unittest.TestCase):
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
            for position, name in enumerate(NVFP4_STREAM_TENSORS[:4]):
                values = torch.arange(4 * width, dtype=torch.int64) * (position + 7) + layer_id
                setattr(layer, name, values.remainder(251).to(torch.uint8).reshape(4, width))
            layer.g1_alphas = torch.arange(4, dtype=torch.float32) + 1.5
            layer.g2_alphas = torch.arange(4, dtype=torch.float32) + 20.5
            layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            self.model.add_module(str(layer_id), layer)
        _HeldEvent.held = True
        _HeldEvent.instances = []

    def manager(self, async_promotions, held_events):
        with tempfile.NamedTemporaryFile(suffix=".pt") as seed:
            torch.save({"count": [[10, 0, 0, 0], [0] * 4, [10, 0, 0, 0]]}, seed.name)
            manager = self.manager_type.from_model(
                self.model,
                budget_bytes=52,
                seed_path=seed.name,
                dynamic=True,
                update_prefill_tokens=16,
                update_decode_forwards=2,
                min_residence_forwards=0,
                benefit_ratio=0.0,
                metrics_path=None,
                route_history_limit=2,
                async_promotions=async_promotions,
            )
        if held_events:
            from sglang.srt.layers.moe.expert_transfer import AsyncExpertTransferExecutor

            executor = AsyncExpertTransferExecutor(
                device=manager.caches[0].device,
                max_inflight=8,
                stream=_SameStream(),
                event_factory=_HeldEvent,
                stream_context=lambda _: nullcontext(),
            )
            for cache in manager.caches.values():
                cache._transfer_executor = executor
        return manager

    def observe(self, manager, counts, mode=None, tokens=1):
        for layer_id, streamer in manager.streamers.items():
            route_ids = [expert for expert, count in enumerate(counts[layer_id]) for _ in range(count)]
            if route_ids:
                streamer.gather(torch.tensor([route_ids], device="cuda"))
        manager.on_expert_distribution(
            SimpleNamespace(forward_mode=mode or self.mode.DECODE, extend_num_tokens=tokens),
            {"global_physical_count": torch.tensor(counts)},
        )

    def lookup(self, manager, layer_id, experts):
        return manager.caches[layer_id].lookup(torch.tensor(experts, device="cuda"))[0].tolist()

    def challenger(self, expert):
        row = [0] * 4
        row[expert] = 3
        return [list(row), [0] * 4, list(row)]

    def test_promoted_slot_is_not_a_hit_until_its_copy_completes(self):
        manager = self.manager(async_promotions=True, held_events=True)
        self.assertEqual(self.lookup(manager, 0, [0, 1]), [0, -1])
        for _ in range(2):
            self.observe(manager, self.challenger(1))
        cache = manager.caches[0]
        self.assertIsNotNone(cache.promotion_in_flight)
        self.assertEqual(cache.slot_to_expert, [1])
        self.assertEqual(cache.slot_state.tolist(), [LOADING])
        self.assertEqual(self.lookup(manager, 0, [0, 1]), [-1, -1])
        self.assertEqual(cache.resident_experts(), frozenset())
        self.observe(manager, self.challenger(1))
        self.assertEqual(self.lookup(manager, 0, [0, 1]), [-1, -1])
        _HeldEvent.release_all()
        self.observe(manager, self.challenger(1))
        self.assertIsNone(cache.promotion_in_flight)
        self.assertEqual(manager._inflight_promotions, [])
        self.assertEqual(self.lookup(manager, 0, [0, 1]), [-1, 0])
        self.assertEqual(cache.slot_state.tolist(), [READY])
        for layer_id in (0, 2):
            assert_ready_rows(self, manager.caches[layer_id], self.model.get_submodule(str(layer_id)), f"layer {layer_id}")

    def test_in_flight_slot_is_neither_evicted_nor_published_twice(self):
        manager = self.manager(async_promotions=True, held_events=True)
        for _ in range(2):
            self.observe(manager, self.challenger(1))
        cache = manager.caches[0]
        promotion = cache.promotion_in_flight
        generation = cache._slot_generations[0]
        for _ in range(6):
            self.observe(manager, self.challenger(2))
        self.observe(manager, self.challenger(2), mode=self.mode.EXTEND, tokens=32)
        self.assertIs(cache.promotion_in_flight, promotion)
        self.assertEqual((cache.slot_to_expert, cache._slot_generations[0]), ([1], generation))
        self.assertEqual(cache.slot_state.tolist(), [LOADING])
        self.assertGreater(manager.deferred_residency_updates, 0)
        counters = manager.snapshot_counters()
        self.assertEqual(counters["residency_async"]["inflight_submissions"], 1)
        _HeldEvent.release_all()
        self.observe(manager, self.challenger(2))
        self.assertEqual(self.lookup(manager, 0, [1]), [0])
        with self.assertRaisesRegex(RuntimeError, "not in flight"):
            cache.complete_promotion(promotion)
        manager.finish_promotions()
        self.assertEqual(manager._inflight_promotions, [])
        self.assertEqual(cache._slot_generations[0], generation)
        json.dumps(manager.snapshot_counters(), allow_nan=False)

    def test_finish_promotions_publishes_real_copies_behind_the_current_stream(self):
        manager = self.manager(async_promotions=True, held_events=False)
        for _ in range(2):
            self.observe(manager, self.challenger(1))
        manager.finish_promotions()
        self.assertEqual(manager._inflight_promotions, [])
        torch.cuda.synchronize()
        for layer_id in (0, 2):
            cache = manager.caches[layer_id]
            self.assertEqual(cache.slot_to_expert, [1])
            self.assertEqual(self.lookup(manager, layer_id, [0, 1]), [-1, 0])
            assert_ready_rows(self, cache, self.model.get_submodule(str(layer_id)), f"layer {layer_id}")

    def test_flag_off_publishes_promotions_at_the_boundary(self):
        manager = self.manager(async_promotions=False, held_events=True)
        for _ in range(2):
            self.observe(manager, self.challenger(1))
        self.assertEqual(manager._inflight_promotions, [])
        for layer_id in (0, 2):
            cache = manager.caches[layer_id]
            self.assertIsNone(cache.promotion_in_flight)
            self.assertEqual(self.lookup(manager, layer_id, [0, 1]), [-1, 0])
            self.assertEqual(cache.slot_state.tolist(), [READY])
            assert_ready_rows(self, cache, self.model.get_submodule(str(layer_id)), f"layer {layer_id}")
        self.assertNotIn("residency_async", manager.snapshot_counters())


if __name__ == "__main__":
    unittest.main()
