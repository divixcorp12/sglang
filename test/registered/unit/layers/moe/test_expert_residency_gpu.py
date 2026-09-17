"""The in-graph residency update against the host (Python) residency path.

Two managers are built over identical layers: ``host`` runs the reference
Python update after each forward, ``gpu`` runs the device update inside one
captured decode-shaped CUDA graph (the gather of every layer). The device
applies a decode boundary at the first gather of the next forward, so after
the gpu replay of forward ``f + 1`` its mapping, slot state, generations,
scores and route counts must equal the host's after its forward ``f + 1``
gathers, which follow the host update of forward ``f``. Every READY slot must
hold its expert's host rows byte for byte.
"""

import os
import random
import tempfile
import unittest
import unittest.mock
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")

LAYERS = 3
EXPERTS = 12
TOP_K = 2
READY = 3


def _model():
    model = torch.nn.Module()
    for layer_id in range(LAYERS):
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer.top_k = TOP_K
        layer._nvfp4_file_source_bytes_per_expert = 0
        for position, name in enumerate(NVFP4_STREAM_TENSORS[:4]):
            values = (torch.arange(EXPERTS * 12, dtype=torch.int64) * (position + 5) + layer_id * 17).remainder(251)
            setattr(layer, name, values.to(torch.uint8).reshape(EXPERTS, 3, 4).pin_memory())
        layer.g1_alphas = (torch.arange(EXPERTS, dtype=torch.float32) + 0.5 + layer_id).cuda()
        layer.g2_alphas = (torch.arange(EXPERTS, dtype=torch.float32) * 3 + layer_id).cuda()
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        model.add_module(str(layer_id), layer)
    return model


def _manager(model, gpu, max_promotions=EXPERTS, seed_scale=(1, 1, 1), **overrides):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

    seed = [
        [float((expert * 7 + layer) % 5) * seed_scale[layer] for expert in range(EXPERTS)]
        for layer in range(LAYERS)
    ]
    options = dict(
        budget_bytes=56 * (12 + LAYERS * TOP_K),
        dynamic=True,
        update_prefill_tokens=16,
        update_decode_forwards=4,
        decay_tokens=1,
        min_residence_forwards=0,
        benefit_ratio=0.5,
        metrics_path=None,
        graph_gather_batch_size=1,
        gpu_residency_update=gpu,
        gpu_residency_max_promotions=max_promotions,
    )
    options.update(overrides)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save({"count": seed}, handle.name)
        return ExpertHotCacheManager.from_model(model, seed_path=handle.name, **options)


def device_state(manager):
    """Mapping, slot state, generations, scores and route counts of every layer, on the host."""
    state = {}
    for layer_id, cache in manager.caches.items():
        policy = manager.residency_policies[layer_id]
        state[layer_id] = dict(
            expert_to_slot=cache.expert_to_slot.tolist(),
            slot_state=cache.slot_state.tolist(),
            slot_generations=cache.slot_generations.tolist(),
            scores=policy._scores.tolist(),
            route_counts=policy.pending_counts.tolist(),
        )
    return state


def assert_states_equal(test, actual, expected, context=""):
    """Fail unless two ``device_state`` snapshots are identical."""
    for layer_id in expected:
        for field in expected[layer_id]:
            test.assertEqual(actual[layer_id][field], expected[layer_id][field], f"{context} layer {layer_id} {field}")


def assert_slot_rows(test, manager, model, context=""):
    """Fail unless every mapped expert's slot holds its six host rows byte for byte."""
    for layer_id, cache in manager.caches.items():
        layer = model.get_submodule(str(layer_id))
        mapping = cache.expert_to_slot.tolist()
        states = cache.slot_state.tolist()
        for expert, slot in enumerate(mapping):
            if slot < 0:
                continue
            test.assertEqual(states[slot], READY, f"{context} layer {layer_id} slot {slot} state")
            for name in NVFP4_STREAM_TENSORS:
                actual = cache.tensors[name][slot].cpu()
                expected = getattr(layer, name)[expert].cpu()
                test.assertTrue(
                    torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)),
                    f"{context} layer {layer_id} expert {expert} slot {slot} {name}",
                )


def _decode_batch():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(forward_mode=ForwardMode.DECODE, extend_num_tokens=1, batch_size=1)


def _idle_batch():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(forward_mode=ForwardMode.IDLE, extend_num_tokens=0, batch_size=1)


def _prefill_batch(tokens):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(forward_mode=ForwardMode.EXTEND, extend_num_tokens=tokens, batch_size=1)


def _counts(routes):
    counts = torch.zeros(LAYERS, EXPERTS, dtype=torch.int64)
    for layer, layer_routes in enumerate(routes):
        for expert in torch.as_tensor(layer_routes).reshape(-1).tolist():
            counts[layer, expert] += 1
    return counts


def _decode_routes(generator, step):
    hot = [(step // 3 + layer * 2) % EXPERTS for layer in range(LAYERS)]
    routes = []
    for layer in range(LAYERS):
        first = hot[layer] if generator.random() < 0.7 else generator.randrange(EXPERTS)
        second = generator.choice([expert for expert in range(EXPERTS) if expert != first])
        routes.append([[first, second]])
    return routes


def _burst_routes(window):
    """Every token of a boundary window routes to one fresh expert pair per layer.

    Both experts gain the whole window's routes at once, so a boundary can
    promote two experts into the same layer.
    """
    return [
        [[(window * 4 + 5 + layer) % EXPERTS, (window * 4 + 9 + layer) % EXPERTS]]
        for layer in range(LAYERS)
    ]


def _decode_promotions(manager):
    counters = manager.snapshot_counters()["decode"]
    return [counters[str(layer)]["promotions"] for layer in range(LAYERS)]


DOORBELL_SPIN_CORE = int(os.environ.get("DOORBELL_SPIN_CORE", "71"))


def _reference_gather_graph(self, topk_ids):
    """``ExpertStreamer._gather_graph`` as of 7de955329a, verbatim: the pre-doorbell path."""
    from sglang.srt.layers.moe.expert_route_plan import plan_graph_routes

    self._check_graph_sources()
    if self.residency_update is not None:
        self.residency_update.on_graph_forward(topk_ids.shape[0])
    cache = self.hot_cache
    flat = topk_ids.reshape(-1).long()
    count = flat.numel()
    plan = plan_graph_routes(
        flat, cache.expert_to_slot, self.graph_gather_rows, cache.capacity
    )
    plan_rows = plan.source_rows.numel()
    scratch = self._graph_scratch_slots[:plan_rows]
    self._graph_source_rows[:plan_rows].copy_(plan.source_rows)
    self._graph_miss_count.copy_(plan.miss_plan_rows.reshape(1))
    for source, destination in self._graph_device_pairs:
        destination.view(torch.uint8).reshape(destination.shape[0], -1).index_copy_(
            0,
            scratch,
            source.view(torch.uint8)
            .reshape(source.shape[0], -1)
            .index_select(0, plan.source_rows),
        )
    if self._graph_row_segments is not None:
        self._copy_row_segments_gpu(
            self._graph_row_segments,
            self._graph_source_rows,
            self._graph_destination_slots,
            self._graph_miss_count,
        )
    self.graph_counters[0].add_(count)
    self.graph_counters[1].add_(plan.routed_miss_rows)
    self.graph_unique_counters[0].add_(plan.unique_hit_rows)
    self.graph_unique_counters[1].add_(plan.unique_miss_rows)
    if self.residency_policy is not None:
        self.residency_policy.pending_counts.index_add_(
            0, flat, self._graph_ones[:count]
        )
    return plan.remap.reshape(topk_ids.shape).to(topk_ids.dtype), cache.tensors


def _gather_harness(manager):
    """Static routes, per-layer output rows and a forward that gathers every layer into them."""
    streamers = [manager.streamers[layer_id] for layer_id in sorted(manager.streamers)]
    static = torch.zeros((LAYERS, 1, TOP_K), dtype=torch.int32, device="cuda")
    static[:, 0, 1] = 1
    outputs = [
        {
            name: torch.zeros((TOP_K,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device="cuda")
            for name, tensor in streamer.hot_cache.tensors.items()
        }
        for streamer in streamers
    ]

    def forward():
        for layer, streamer in enumerate(streamers):
            compact, tensors = streamer.gather(static[layer])
            for name, output in outputs[layer].items():
                output.copy_(tensors[name][compact.reshape(-1).long()])

    return static, outputs, forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGatherAcrossResidencyUpdates(unittest.TestCase):
    def _captured(self, manager, forward):
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        manager.quiesce_doorbell()
        try:
            with torch.cuda.graph(graph):
                forward()
        finally:
            manager.resume_doorbell()
        manager.discard_graph_capture_routes()
        return graph

    def test_gathers_match_the_pre_doorbell_path_across_residency_updates(self):
        """Gathers, then residency updates, then more gathers, with the host (Python) update and
        with SGLANG_MOE_GPU_RESIDENCY_UPDATE, the doorbell off and on. Each step's gathered rows,
        per-layer hit and miss counters and residency state must equal a twin manager whose
        streamers run 7de955329a's ``_gather_graph``, and the rows must equal the source rows.
        The GPU update rebinds each cache's ``expert_to_slot`` to its own table, so a gather
        planner holding the startup mapping diverges at the first update."""
        for gpu in (False, True):
            for doorbell in (False, True):
                with self.subTest(gpu_residency_update=gpu, doorbell=doorbell):
                    self._run_mode(gpu, doorbell)

    def test_gathers_match_the_pre_doorbell_path_across_insert_on_miss_updates(self):
        """The same cross-residency check with SGLANG_MOE_HOT_INSERT_ON_MISS: every forward's
        misses are copied from scratch rows into slots before the next forward's gathers, so a
        planner, doorbell plan or remap that read a stale mapping or stale scratch row would
        return rows that differ from the reference path or from the source rows."""
        for doorbell in (False, True):
            with self.subTest(doorbell=doorbell):
                self._run_mode(True, doorbell, insert_on_miss=True)

    def _run_mode(self, gpu, doorbell, insert_on_miss=False):
        current_model, reference_model = _model(), _model()
        mode = (
            dict(update_decode_forwards=1, insert_on_miss=True, insert_on_miss_decay=0.98)
            if insert_on_miss
            else {}
        )
        current = _manager(
            current_model,
            gpu,
            expert_doorbell=doorbell,
            doorbell_cpu_core=DOORBELL_SPIN_CORE,
            **mode,
        )
        reference = _manager(reference_model, gpu, **mode)
        for streamer in reference.streamers.values():
            streamer._gather_graph = MethodType(_reference_gather_graph, streamer)
        try:
            harnesses = []
            for manager in (current, reference):
                static, outputs, forward = _gather_harness(manager)
                graph = self._captured(manager, forward) if gpu else None
                harnesses.append((manager, static, outputs, forward, graph))
            generator = random.Random(11)
            promotions = 0
            for step in range(17):
                routes = _decode_routes(generator, step)
                route_tensor = torch.tensor(routes, dtype=torch.int32, device="cuda")
                for manager, static, outputs, forward, graph in harnesses:
                    static.copy_(route_tensor)
                    if graph is not None:
                        graph.replay()
                    else:
                        forward()
                torch.cuda.synchronize()
                context = f"gpu={gpu} doorbell={doorbell} step {step}"
                current_outputs, reference_outputs = harnesses[0][2], harnesses[1][2]
                for layer in range(LAYERS):
                    source_layer = current_model.get_submodule(str(layer))
                    experts = torch.tensor(routes[layer][0])
                    for name in NVFP4_STREAM_TENSORS:
                        actual = current_outputs[layer][name].view(torch.uint8).cpu()
                        self.assertTrue(
                            torch.equal(actual, reference_outputs[layer][name].view(torch.uint8).cpu()),
                            f"{context} layer {layer} {name} differs from the reference path",
                        )
                        expected = getattr(source_layer, name)[experts.to(getattr(source_layer, name).device)]
                        self.assertTrue(
                            torch.equal(actual.reshape(-1), expected.reshape(-1).view(torch.uint8).cpu()),
                            f"{context} layer {layer} {name} differs from the source rows",
                        )
                current_counters = current.snapshot_counters()["decode"]
                reference_counters = reference.snapshot_counters()["decode"]
                fields = ("hot_hits", "miss_rows", "promotions", "evictions")
                if insert_on_miss:
                    fields += ("insertions", "insertion_evictions")
                for layer in range(LAYERS):
                    for field in fields:
                        self.assertEqual(
                            current_counters[str(layer)][field],
                            reference_counters[str(layer)][field],
                            f"{context} layer {layer} {field}",
                        )
                assert_states_equal(self, device_state(current), device_state(reference), context)
                changed = "insertions" if insert_on_miss else "promotions"
                promotions = sum(current_counters[str(layer)][changed] for layer in range(LAYERS))
                if insert_on_miss:
                    self.assertEqual(
                        sum(current_counters[str(layer)]["promotions"] for layer in range(LAYERS)),
                        0,
                        f"{context}: insert-on-miss decode boundaries must not promote host rows",
                    )
                counts = {"global_physical_count": _counts(routes)}
                current.on_expert_distribution(_decode_batch(), counts)
                reference.on_expert_distribution(_decode_batch(), counts)
            self.assertGreater(promotions, 0, f"gpu={gpu} doorbell={doorbell}: no residency update happened")
        finally:
            if current.doorbell is not None:
                current.doorbell.stop()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGpuResidencyUpdate(unittest.TestCase):
    def setUp(self):
        self.host_model = _model()
        self.gpu_model = _model()

    def capture(self, manager):
        streamers = [manager.streamers[layer_id] for layer_id in sorted(manager.streamers)]
        static = torch.zeros((LAYERS, 1, TOP_K), dtype=torch.int32, device="cuda")
        static[:, 0, 1] = 1

        def forward():
            for layer, streamer in enumerate(streamers):
                streamer.gather(static[layer])

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        manager.discard_graph_capture_routes()
        return graph, static

    def host_forward(self, manager, routes, batch):
        for layer, streamer in sorted(manager.streamers.items()):
            streamer.gather(torch.tensor(routes[layer], dtype=torch.int32, device="cuda"))

    def run_decode(self, host, gpu, graph, static, generator, steps, start=0):
        for step in range(start, start + steps):
            routes = _decode_routes(generator, step)
            self.host_forward(host, routes, _decode_batch())
            static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
            graph.replay()
            context = f"decode step {step}"
            assert_states_equal(self, device_state(gpu), device_state(host), context)
            assert_slot_rows(self, gpu, self.gpu_model, context)
            counts = {"global_physical_count": _counts(routes)}
            host.on_expert_distribution(_decode_batch(), counts)
            gpu.on_expert_distribution(_decode_batch(), counts)

    def test_captured_update_matches_host_path_at_every_boundary(self):
        host = _manager(self.host_model, gpu=False)
        gpu = _manager(self.gpu_model, gpu=True)
        self.assertIsNone(host.gpu_residency)
        assert_states_equal(self, device_state(gpu), device_state(host), "startup")
        graph, static = self.capture(gpu)
        assert_states_equal(self, device_state(gpu), device_state(host), "after capture")
        generator = random.Random(1)
        self.run_decode(host, gpu, graph, static, generator, steps=17)
        promotions = sum(gpu.snapshot_counters()["decode"][str(layer)]["promotions"] for layer in range(LAYERS))
        host_promotions = sum(host.snapshot_counters()["decode"][str(layer)]["promotions"] for layer in range(LAYERS))
        self.assertGreater(promotions, 0)
        self.assertEqual(sum(gpu.gpu_residency.snapshot()["truncated_layers"]), 0)
        routes = _decode_routes(generator, 17)
        self.host_forward(host, routes, _decode_batch())
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        self.assertEqual(
            sum(gpu.snapshot_counters()["decode"][str(layer)]["promotions"] for layer in range(LAYERS)),
            host_promotions,
        )

    def test_eager_prefill_flushes_pending_boundary_and_applies_its_own(self):
        host = _manager(self.host_model, gpu=False)
        gpu = _manager(self.gpu_model, gpu=True)
        graph, static = self.capture(gpu)
        generator = random.Random(2)
        self.run_decode(host, gpu, graph, static, generator, steps=4)
        self.assertTrue(gpu.gpu_residency.host_pending)
        for tokens in (24, 6):
            routes = [
                [[generator.randrange(EXPERTS), generator.randrange(EXPERTS)] for _ in range(tokens)]
                for _ in range(LAYERS)
            ]
            routes = [[[a, b if b != a else (a + 1) % EXPERTS] for a, b in layer] for layer in routes]
            self.host_forward(host, routes, _prefill_batch(tokens))
            self.host_forward(gpu, routes, _prefill_batch(tokens))
            counts = {"global_physical_count": _counts(routes)}
            host.on_expert_distribution(_prefill_batch(tokens), counts)
            gpu.on_expert_distribution(_prefill_batch(tokens), counts)
            assert_states_equal(self, device_state(gpu), device_state(host), f"prefill {tokens}")
            assert_slot_rows(self, gpu, self.gpu_model, f"prefill {tokens}")
        self.assertFalse(gpu.gpu_residency.host_pending)
        self.run_decode(host, gpu, graph, static, generator, steps=9, start=20)

    def test_residence_margins_and_uneven_capacities_match_host_path(self):
        options = dict(min_residence_forwards=6, benefit_ratio=0.25, promotion_sigmas=0.5, seed_scale=(1, 3, 9))
        host = _manager(self.host_model, gpu=False, **options)
        gpu = _manager(self.gpu_model, gpu=True, **options)
        capacities = [cache.capacity for _, cache in sorted(gpu.caches.items())]
        self.assertGreater(len(set(capacities)), 1, capacities)
        graph, static = self.capture(gpu)
        generator = random.Random(6)
        self.run_decode(host, gpu, graph, static, generator, steps=25)
        self.assertGreater(sum(_decode_promotions(gpu)), 0)
        self.assertEqual(_decode_promotions(gpu), _decode_promotions(host))

    def test_idle_and_graph_served_prefill_keep_the_clocks_aligned(self):
        host = _manager(self.host_model, gpu=False, min_residence_forwards=3)
        gpu = _manager(self.gpu_model, gpu=True, min_residence_forwards=3)
        graph, static = self.capture(gpu)
        generator = random.Random(7)
        self.run_decode(host, gpu, graph, static, generator, steps=4)
        self.assertTrue(gpu.gpu_residency.host_pending)
        idle_counts = {"global_physical_count": torch.zeros(LAYERS, EXPERTS, dtype=torch.int64)}
        host.on_expert_distribution(_idle_batch(), idle_counts)
        gpu.on_expert_distribution(_idle_batch(), idle_counts)
        self.assertFalse(gpu.gpu_residency.host_pending)
        assert_states_equal(self, device_state(gpu), device_state(host), "after idle")
        routes = [[[layer, (layer + 3) % EXPERTS]] for layer in range(LAYERS)]
        self.host_forward(host, routes, _prefill_batch(1))
        self.host_forward(gpu, routes, _prefill_batch(1))
        counts = {"global_physical_count": _counts(routes)}
        host.on_expert_distribution(_prefill_batch(1), counts)
        gpu.on_expert_distribution(_prefill_batch(1), counts)
        assert_states_equal(self, device_state(gpu), device_state(host), "after graph-served prefill")
        self.run_decode(host, gpu, graph, static, generator, steps=11, start=40)
        clock = host._boundary_clock
        updater = gpu.gpu_residency
        self.assertEqual(
            (int(updater.forwards), int(updater.tokens), int(updater.decode_forwards)),
            (clock.forwards, clock.tokens_since_boundary, clock.decode_forwards_since_boundary),
        )

    def test_gpu_owned_slots_refuse_host_publication(self):
        gpu = _manager(self.gpu_model, gpu=True)
        cache = next(cache for _, cache in sorted(gpu.caches.items()) if cache.capacity)
        with self.assertRaisesRegex(RuntimeError, "owned by the GPU residency update"):
            cache.reassign([])
        with self.assertRaisesRegex(RuntimeError, "owned by the GPU residency update"):
            cache.resident_experts()

    def test_decode_forward_hook_is_sync_free(self):
        gpu = _manager(self.gpu_model, gpu=True)
        graph, static = self.capture(gpu)
        counts = {"global_physical_count": torch.zeros(LAYERS, EXPERTS, dtype=torch.int64, device="cuda")}
        graph.replay()
        gpu.on_expert_distribution(_decode_batch(), counts)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(5):
                graph.replay()
                gpu.on_expert_distribution(_decode_batch(), counts)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_capped_update_truncates_and_keeps_slots_consistent(self):
        host = _manager(self.host_model, gpu=False)
        gpu = _manager(self.gpu_model, gpu=True, max_promotions=1)
        graph, static = self.capture(gpu)
        host_most_promotions = 0
        before = _decode_promotions(host)
        for step in range(24):
            routes = _burst_routes(step // 4)
            self.host_forward(host, routes, _decode_batch())
            static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
            graph.replay()
            counts = {"global_physical_count": _counts(routes)}
            host.on_expert_distribution(_decode_batch(), counts)
            gpu.on_expert_distribution(_decode_batch(), counts)
            after = _decode_promotions(host)
            host_most_promotions = max(host_most_promotions, *(a - b for a, b in zip(after, before)))
            before = after
            assert_slot_rows(self, gpu, self.gpu_model, f"capped step {step}")
            updater = gpu.gpu_residency
            for row, cache in enumerate(updater.caches):
                mapping = cache.expert_to_slot.tolist()
                slots = updater.slot_to_expert[row, : cache.capacity].tolist()
                for expert, slot in enumerate(mapping):
                    if slot >= 0:
                        self.assertEqual(slots[slot], expert)
                self.assertEqual(sum(slot >= 0 for slot in mapping), sum(expert >= 0 for expert in slots))
        self.assertGreaterEqual(
            host_most_promotions, 2, "the uncapped host path never promoted two experts into one layer"
        )
        self.assertGreater(sum(gpu.gpu_residency.snapshot()["truncated_layers"]), 0)

    def test_state_helpers_fail_on_perturbed_state(self):
        host = _manager(self.host_model, gpu=False)
        gpu = _manager(self.gpu_model, gpu=True)
        expected = device_state(host)
        assert_states_equal(self, device_state(gpu), expected)
        assert_slot_rows(self, gpu, self.gpu_model)
        cache = gpu.caches[0]
        slot = cache.expert_to_slot.tolist().index(next(s for s in cache.expert_to_slot.tolist() if s >= 0))
        for field, perturb, restore in (
            ("expert_to_slot", lambda: cache.expert_to_slot.__setitem__(slot, -1), lambda: cache.expert_to_slot.__setitem__(slot, expected[0]["expert_to_slot"][slot])),
            ("slot_generations", lambda: cache.slot_generations.add_(1), lambda: cache.slot_generations.sub_(1)),
            ("scores", lambda: gpu.residency_policies[0]._scores.add_(0.5), lambda: gpu.residency_policies[0]._scores.sub_(0.5)),
        ):
            perturb()
            with self.subTest(field=field), self.assertRaises(AssertionError):
                assert_states_equal(self, device_state(gpu), expected)
            restore()
        mapped_slot = expected[0]["expert_to_slot"][slot]
        cache.tensors["w13_weight"][mapped_slot, 0, 0] += 1
        with self.assertRaises(AssertionError):
            assert_slot_rows(self, gpu, self.gpu_model)


IOM_DECAY = 0.98
IOM = dict(update_decode_forwards=1, insert_on_miss=True, insert_on_miss_decay=IOM_DECAY)


class _InsertOnMissReference:
    """Host model of insert-on-miss decode boundaries, written without the device code.

    A decode forward first applies the previous forward's boundary: every layer's
    scores become ``scores * decay**tokens + route counts`` in float32, and each
    of that forward's missed experts still nonresident takes, in plan order, a
    free slot (lowest first) or else the slot of the resident with the lowest
    ``(score, -expert)`` among residents with no route since the last boundary.
    Misses beyond the free and evictable slots stay out and count as truncated.
    Then the forward's routes gather against the resulting mapping.
    """

    def __init__(self, manager, decay=IOM_DECAY):
        self.decay = decay
        self.caches = [cache for _, cache in sorted(manager.caches.items())]
        self.mapping = [cache.expert_to_slot.tolist() for cache in self.caches]
        self.slots = []
        for row, cache in enumerate(self.caches):
            slots = [-1] * cache.capacity
            for expert, slot in enumerate(self.mapping[row]):
                if slot >= 0:
                    slots[slot] = expert
            self.slots.append(slots)
        self.generations = [cache.slot_generations.tolist() for cache in self.caches]
        self.scores = torch.stack(
            [manager.residency_policies[layer_id]._scores.detach().cpu() for layer_id in sorted(manager.caches)]
        ).to(torch.float32)
        self.pending = torch.zeros_like(self.scores)
        self.tokens = 0
        self.plans = None
        self.insertions = [0] * LAYERS
        self.evictions = [0] * LAYERS
        self.truncated = [0] * LAYERS
        self.hits = [0] * LAYERS
        self.misses = [0] * LAYERS

    def boundary(self):
        factor = torch.tensor(self.decay**self.tokens, dtype=torch.float32)
        self.scores = self.scores * factor + self.pending
        routed = self.pending > 0
        self.pending = torch.zeros_like(self.pending)
        self.tokens = 0
        plans, self.plans = self.plans, None
        if plans is None:
            return
        for row, plan in enumerate(plans):
            mapping, slots = self.mapping[row], self.slots[row]
            want = [expert for expert in plan if mapping[expert] < 0]
            free = [slot for slot, expert in enumerate(slots) if expert < 0]
            victims = sorted(
                (slot for slot, expert in enumerate(slots) if expert >= 0 and not routed[row, expert]),
                key=lambda slot: (float(self.scores[row, slots[slot]]), -slots[slot]),
            )
            targets = free + victims
            if len(want) > len(targets):
                self.truncated[row] += 1
            for expert, slot in zip(want, targets):
                if slots[slot] >= 0:
                    mapping[slots[slot]] = -1
                    self.evictions[row] += 1
                mapping[expert] = slot
                slots[slot] = expert
                self.generations[row][slot] += 1
                self.insertions[row] += 1

    def decode_forward(self, routes):
        """Apply the pending boundary, then gather one decode token's routes."""
        if self.plans is not None:
            self.boundary()
        self.plans = []
        for row, layer_routes in enumerate(routes):
            plan = []
            for expert in [expert for token in layer_routes for expert in token]:
                self.pending[row, expert] += 1
                if self.mapping[row][expert] >= 0:
                    self.hits[row] += 1
                elif expert not in plan:
                    plan.append(expert)
                    self.misses[row] += 1
            self.plans.append(plan)
        self.tokens += 1

    def eager_prefill(self, routes, tokens):
        """A prefill below the prefill boundary: flush a pending boundary, then only count."""
        if self.plans is not None:
            self.boundary()
        for row, layer_routes in enumerate(routes):
            for token in layer_routes:
                for expert in token:
                    self.pending[row, expert] += 1
        self.tokens += tokens

    def assert_matches(self, test, manager, context):
        updater = manager.gpu_residency
        for row, cache in enumerate(self.caches):
            test.assertEqual(cache.expert_to_slot.tolist(), self.mapping[row], f"{context} layer {row} mapping")
            test.assertEqual(
                cache.slot_state.tolist(),
                [READY if expert >= 0 else 0 for expert in self.slots[row]],
                f"{context} layer {row} slot state",
            )
            test.assertEqual(cache.slot_generations.tolist(), self.generations[row], f"{context} layer {row} generations")
            test.assertEqual(
                updater.slot_to_expert[row, : cache.capacity].tolist(), self.slots[row], f"{context} layer {row} slots"
            )
        test.assertEqual(updater.insert_scores.cpu().tolist(), self.scores.tolist(), f"{context} scores")
        test.assertEqual(updater.route_counts.cpu().tolist(), self.pending.tolist(), f"{context} route counts")
        snapshot = updater.snapshot()
        test.assertEqual(snapshot["insertions"], self.insertions, f"{context} insertions")
        test.assertEqual(snapshot["insertion_evictions"], self.evictions, f"{context} insertion evictions")
        test.assertEqual(snapshot["insertion_truncated"], self.truncated, f"{context} insertion truncated")
        decode = manager.snapshot_counters()["decode"]
        for row in range(LAYERS):
            test.assertEqual(decode[str(row)]["hot_hits"], self.hits[row], f"{context} layer {row} hot hits")
            test.assertEqual(decode[str(row)]["miss_rows"], self.misses[row], f"{context} layer {row} misses")
            test.assertEqual(decode[str(row)]["promotions"], 0, f"{context} layer {row} decode promotions")
            test.assertEqual(decode[str(row)]["insertions"], self.insertions[row], f"{context} layer {row} insertions")


def _random_routes(generator):
    return [[generator.sample(range(EXPERTS), TOP_K)] for _ in range(LAYERS)]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestInsertOnMiss(unittest.TestCase):
    """SGLANG_MOE_HOT_INSERT_ON_MISS: a decode forward's missed experts move from scratch rows into slots."""

    def setUp(self):
        self.model = _model()

    def capture(self, manager):
        static, outputs, forward = _gather_harness(manager)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        manager.discard_graph_capture_routes()
        return graph, static, outputs

    def assert_outputs(self, routes, outputs, context):
        for layer in range(LAYERS):
            source_layer = self.model.get_submodule(str(layer))
            experts = torch.tensor(routes[layer][0])
            for name in NVFP4_STREAM_TENSORS:
                source = getattr(source_layer, name)
                expected = source[experts.to(source.device)].reshape(-1).view(torch.uint8).cpu()
                actual = outputs[layer][name].view(torch.uint8).reshape(-1).cpu()
                self.assertTrue(torch.equal(actual, expected), f"{context} layer {layer} {name} gathered rows")

    def replay(self, manager, reference, graph, static, outputs, routes, context):
        reference.decode_forward(routes)
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        self.assert_outputs(routes, outputs, context)
        manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        reference.assert_matches(self, manager, context)
        assert_slot_rows(self, manager, self.model, context)

    def test_flag_off_builds_the_unchanged_updater(self):
        """Off (the default, or explicitly), the updater has no insertion state, counters or trace
        buffers; the environment switch turns the mode on without a model runner argument."""
        from sglang.srt.environ import envs

        for explicit in (None, False):
            with self.subTest(explicit=explicit):
                options = {} if explicit is None else dict(insert_on_miss=explicit)
                manager = _manager(_model(), gpu=True, **options)
                updater = manager.gpu_residency
                self.assertFalse(updater.insert_on_miss)
                self.assertIsNone(updater.insert_tensors)
                self.assertNotIn("insertions", updater.snapshot())
                self.assertFalse(any("insertion" in name for name in manager._trace_sources()))
                self.assertNotIn("insertions", manager.snapshot_counters()["decode"]["0"])
        stage = envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE
        with stage.override(1), envs.SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY.override(0.99):
            manager = _manager(_model(), gpu=True, update_decode_forwards=1)
            self.assertTrue(manager.gpu_residency.insert_on_miss)
            self.assertFalse(manager.gpu_residency.insert_direct)
            self.assertAlmostEqual(manager.gpu_residency.insert_on_miss_decay, 0.99)

    def test_the_retired_boolean_still_selects_stage_one(self):
        """`combined-iom`'s measured arms set SGLANG_MOE_HOT_INSERT_ON_MISS=1. The stage enum keeps
        that value, so those arms keep running stage 1 -- with a DeprecationWarning, not silently
        falling back to off and simply looking slower."""
        from sglang.srt.environ import InsertOnMissStage, envs

        self.assertEqual(int(InsertOnMissStage.OFF), 0)
        self.assertEqual(int(InsertOnMissStage.SCRATCH), 1)
        with unittest.mock.patch.dict(os.environ, {"SGLANG_MOE_HOT_INSERT_ON_MISS": "1"}):
            envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.clear()
            with self.assertWarns(DeprecationWarning):
                resolved = envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get()
            self.assertEqual(resolved, InsertOnMissStage.SCRATCH)
        envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.clear()

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE"):
            _manager(_model(), gpu=True, update_decode_forwards=1, insert_on_miss=3)

    def test_mode_requires_the_gpu_update_and_a_boundary_every_decode_forward(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_GPU_RESIDENCY_UPDATE"):
            _manager(_model(), gpu=False, **IOM)
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1"):
            _manager(_model(), gpu=True, **dict(IOM, update_decode_forwards=4))
        with self.assertRaisesRegex(ValueError, "decay"):
            _manager(_model(), gpu=True, **dict(IOM, insert_on_miss_decay=1.5))

    def test_captured_decode_inserts_every_miss_before_the_next_forward(self):
        """Each replay applies the previous forward's insertions: mapping, slots, generations, scores,
        counters and every resident slot's bytes match the host reference at every step, and the
        gathered rows match the source rows."""
        manager = _manager(self.model, gpu=True, seed_scale=(1, 3, 9), **IOM)
        capacities = [cache.capacity for _, cache in sorted(manager.caches.items())]
        self.assertGreater(len(set(capacities)), 1, capacities)
        reference = _InsertOnMissReference(manager)
        graph, static, outputs = self.capture(manager)
        reference.assert_matches(self, manager, "after capture")
        generator = random.Random(3)
        for step in range(40):
            self.replay(manager, reference, graph, static, outputs, _random_routes(generator), f"step {step}")
        self.assertGreater(sum(reference.insertions), 0)
        self.assertGreater(sum(reference.evictions), 0)

    def test_missed_experts_are_resident_after_one_forward(self):
        manager = _manager(self.model, gpu=True, **IOM)
        graph, static, outputs = self.capture(manager)
        reference = _InsertOnMissReference(manager)
        generator = random.Random(4)
        previous = None
        for step in range(12):
            self.replay(manager, reference, graph, static, outputs, _random_routes(generator), f"step {step}")
            if previous is not None:
                for row, cache in enumerate(reference.caches):
                    for expert in previous[row]:
                        self.assertGreaterEqual(int(cache.expert_to_slot[expert]), 0, f"step {step} layer {row} expert {expert}")
            previous = [list(plan) for plan in reference.plans]
        self.assertGreater(sum(reference.insertions), 0)
        self.assertEqual(sum(reference.truncated), 0)

    def test_victim_is_the_lowest_score_resident_not_routed_in_that_forward(self):
        manager = _manager(self.model, gpu=True, **IOM)
        updater = manager.gpu_residency
        graph, static, outputs = self.capture(manager)
        cache = manager.caches[0]
        mapping = cache.expert_to_slot.tolist()
        residents = [expert for expert, slot in enumerate(mapping) if slot >= 0]
        outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
        self.assertGreaterEqual(len(residents), 3)
        self.assertTrue(outsiders)
        self.assertEqual(len(residents), cache.capacity, "a free slot would take the miss before any victim")
        lowest, second, miss = residents[0], residents[1], outsiders[0]
        scores = torch.full((EXPERTS,), 100.0)
        scores[lowest], scores[second] = 1.0, 2.0
        updater.insert_scores[0].copy_(scores)
        others = [[[0, 1]] for _ in range(LAYERS - 1)]
        routes = [[[lowest, miss]]] + others
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        torch.cuda.synchronize()
        self.assertEqual(int(cache.expert_to_slot[miss]), -1)
        routes = [[[lowest, residents[2]]]] + others
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(cache.expert_to_slot[miss]), mapping[second], "the miss takes the lowest unrouted resident's slot")
        self.assertEqual(int(cache.expert_to_slot[second]), -1)
        self.assertEqual(int(cache.expert_to_slot[lowest]), mapping[lowest], "a routed resident is never evicted")
        assert_slot_rows(self, manager, self.model, "victim")
        self.assert_outputs(routes, outputs, "victim")

    def test_layers_without_free_or_evictable_slots_truncate(self):
        manager = _manager(self.model, gpu=True, seed_scale=(0, 1, 1), **IOM)
        self.assertEqual(manager.caches[0].capacity, 0)
        reference = _InsertOnMissReference(manager)
        graph, static, outputs = self.capture(manager)
        generator = random.Random(5)
        for step in range(10):
            self.replay(manager, reference, graph, static, outputs, _random_routes(generator), f"step {step}")
        self.assertGreater(reference.truncated[0], 0)
        self.assertEqual(reference.insertions[0], 0)
        self.assertGreater(sum(reference.insertions[1:]), 0)

    def test_eager_prefills_flush_insertions_and_keep_rows_exact(self):
        """A short eager prefill flushes the pending insertions before its first gather and adds its
        routes and tokens to the next boundary; a boundary-sized prefill promotes host rows as before,
        and decode insertions resume on consistent slots."""
        manager = _manager(self.model, gpu=True, **IOM)
        updater = manager.gpu_residency
        reference = _InsertOnMissReference(manager)
        graph, static, outputs = self.capture(manager)
        generator = random.Random(6)

        def prefill(tokens):
            routes = [[generator.sample(range(EXPERTS), TOP_K) for _ in range(tokens)] for _ in range(LAYERS)]
            for layer, streamer in sorted(manager.streamers.items()):
                streamer.gather(torch.tensor(routes[layer], dtype=torch.int32, device="cuda"))
            manager.on_expert_distribution(_prefill_batch(tokens), {"global_physical_count": _counts(routes)})
            torch.cuda.synchronize()
            return routes

        for step in range(4):
            self.replay(manager, reference, graph, static, outputs, _random_routes(generator), f"decode {step}")
        self.assertTrue(updater.host_pending)
        pending_plans = [list(plan) for plan in reference.plans]
        reference.eager_prefill(prefill(6), 6)
        reference.assert_matches(self, manager, "short prefill")
        assert_slot_rows(self, manager, self.model, "short prefill")
        self.assertTrue(any(pending_plans), "the flushed boundary had no misses to insert")
        for step in range(4, 8):
            self.replay(manager, reference, graph, static, outputs, _random_routes(generator), f"decode {step}")
        prefill(24)
        assert_slot_rows(self, manager, self.model, "boundary prefill")
        self.assertEqual(sum(updater.snapshot()["promotions"][0]), 0)
        inserted = sum(updater.snapshot()["insertions"])
        for step in range(8, 14):
            routes = _random_routes(generator)
            static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            self.assert_outputs(routes, outputs, f"decode {step}")
            assert_slot_rows(self, manager, self.model, f"decode {step}")
            for row, cache in enumerate(updater.caches):
                mapping = cache.expert_to_slot.tolist()
                slots = updater.slot_to_expert[row, : cache.capacity].tolist()
                for expert, slot in enumerate(mapping):
                    if slot >= 0:
                        self.assertEqual(slots[slot], expert)
                self.assertEqual(sum(slot >= 0 for slot in mapping), sum(expert >= 0 for expert in slots))
            manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        self.assertGreater(sum(updater.snapshot()["insertions"]), inserted)
        self.assertEqual(sum(updater.snapshot()["promotions"][0]), 0)

    def test_decode_forward_hook_is_sync_free(self):
        manager = _manager(self.model, gpu=True, **IOM)
        graph, static, _ = self.capture(manager)
        counts = {"global_physical_count": torch.zeros(LAYERS, EXPERTS, dtype=torch.int64, device="cuda")}
        graph.replay()
        manager.on_expert_distribution(_decode_batch(), counts)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for step in range(5):
                static.copy_((static + 3 + step).remainder(EXPERTS))
                graph.replay()
                manager.on_expert_distribution(_decode_batch(), counts)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        self.assertGreater(sum(manager.gpu_residency.snapshot()["insertions"]), 0)

    def test_reference_helper_fails_on_a_wrong_victim(self):
        manager = _manager(self.model, gpu=True, **IOM)
        reference = _InsertOnMissReference(manager)
        reference.assert_matches(self, manager, "startup")
        row = 1
        slot = next(slot for slot, expert in enumerate(reference.slots[row]) if expert >= 0)
        reference.mapping[row][reference.slots[row][slot]] = -1
        reference.slots[row][slot] = -1
        with self.assertRaises(AssertionError):
            reference.assert_matches(self, manager, "perturbed")


DIRECT = dict(update_decode_forwards=1, insert_on_miss=2, insert_on_miss_decay=IOM_DECAY)


def _cache_bytes(cache):
    """Every row of every tensor of one cache, as host bytes, for a no-touch comparison."""
    return {
        name: tensor.view(torch.uint8).reshape(tensor.shape[0], -1).cpu().clone()
        for name, tensor in cache.tensors.items()
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestInsertOnMissDirect(unittest.TestCase):
    """Stage DIRECT: a gather copies each miss straight into a victim slot, and no scratch is held.

    The victim shortlist is ranked by the previous boundary, before this forward's routing exists.
    Safety therefore does not come from the ranking; it comes from the gather dropping every
    shortlist entry its own forward routes to. These tests pin that split down.
    """

    def setUp(self):
        self.model = _model()

    def capture(self, manager):
        static, outputs, forward = _gather_harness(manager)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        manager.discard_graph_capture_routes()
        return graph, static, outputs

    def step(self, manager, graph, static, outputs, routes, context):
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            source_layer = self.model.get_submodule(str(layer))
            experts = torch.tensor(routes[layer][0])
            for name in NVFP4_STREAM_TENSORS:
                source = getattr(source_layer, name)
                expected = source[experts.to(source.device)].reshape(-1).view(torch.uint8).cpu()
                actual = outputs[layer][name].view(torch.uint8).reshape(-1).cpu()
                self.assertTrue(torch.equal(actual, expected), f"{context} layer {layer} {name}")
        manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        torch.cuda.synchronize()

    # ----- shape of the allocation -----

    def test_direct_holds_no_scratch_and_spends_the_rows_on_residency(self):
        """The whole point: the graph gather keeps its route width but stops reserving rows for it,
        so the same budget buys strictly more slots than stage 1 does."""
        scratch = _manager(_model(), gpu=True, **IOM)
        direct = _manager(_model(), gpu=True, **DIRECT)
        for cache in direct.caches.values():
            self.assertEqual(cache.scratch_rows, 0)
            self.assertEqual(cache.scratch_bytes, 0)
            for tensor in cache.tensors.values():
                self.assertEqual(tensor.shape[0], cache.capacity)
        for streamer in direct.streamers.values():
            self.assertEqual(streamer.graph_gather_rows, TOP_K)
            self.assertEqual(streamer._graph_device_pairs, ())
        scratch_slots = sum(cache.capacity for cache in scratch.caches.values())
        direct_slots = sum(cache.capacity for cache in direct.caches.values())
        self.assertGreater(direct_slots, scratch_slots)
        self.assertEqual(direct_slots - scratch_slots, LAYERS * TOP_K)

    def test_a_layer_too_small_to_guarantee_a_victim_is_refused(self):
        """Below twice the gather width a forward could route every shortlist entry, and with no
        scratch row left there would be nowhere safe for that miss to land. Refuse, never truncate."""
        with self.assertRaisesRegex(ValueError, "twice its graph-gather rows"):
            _manager(_model(), gpu=True, seed_scale=(0, 1, 1), **DIRECT)

    # ----- the consolidated safety guard (one guard, three reasons) -----

    def test_every_backend_that_could_write_a_slot_off_stream_is_refused(self):
        """Stage DIRECT commits residency for a row the moment its copy is issued. Anything that
        can write that row from another stream, or serve it from somewhere else, breaks that."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

        with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("always"):
            with self.assertRaisesRegex(ValueError, "PREFETCH_PULL_MODE=off"):
                _manager(_model(), gpu=True, **DIRECT)

        # A plan whose buffers are not the gather's own: the doorbell backend posts one, and then
        # its thread owns those slots on another stream.
        manager = _manager(_model(), gpu=True, **DIRECT)
        streamer = next(iter(manager.streamers.values()))
        streamer.row_plan = ExpertRowPlan.for_scratch(
            TOP_K, streamer.hot_cache.capacity, TOP_K, streamer.hot_cache.device
        )
        with self.assertRaisesRegex(ValueError, "DOORBELL_PLAN_CAPACITY"):
            manager.gpu_residency.check_miss_plans()

        manager = _manager(_model(), gpu=True, **DIRECT)
        next(iter(manager.streamers.values())).pinned_host_cache = SimpleNamespace(capacity=1)
        with self.assertRaisesRegex(ValueError, "pinned host cache"):
            manager.gpu_residency.check_miss_plans()

    def test_the_copy_and_the_commit_read_one_count(self):
        """Structural, and the reason a short copy cannot leave a slot wrongly READY: the residency
        commit is masked by the very tensor the copy kernel reads its row count from. There are not
        two counts that could disagree, so there is no ordering window to get wrong."""
        manager = _manager(_model(), gpu=True, **DIRECT)
        for streamer in manager.streamers.values():
            self.assertIs(streamer.row_plan.count, streamer._graph_miss_count)
            self.assertIs(streamer.row_plan.slots, streamer._graph_destination_slots)
            self.assertIs(streamer.row_plan.expert_ids, streamer._graph_source_rows)

    # ----- the capacity proof, exercised -----

    def test_no_forward_ever_lands_a_copy_in_a_row_it_reads(self):
        """The safety property itself, checked per replay against the pre-gather mapping: a
        destination is never a slot this forward routes to, destinations are distinct, and a
        shortlist entry the forward does route is skipped rather than overwritten."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        updater = manager.gpu_residency
        graph, static, outputs = self.capture(manager)
        generator = random.Random(11)
        seen_disqualified = 0
        for step in range(30):
            routes = _random_routes(generator)
            before = [cache.expert_to_slot.tolist() for _, cache in sorted(manager.caches.items())]
            shortlist = updater.victims.tolist()
            valid = updater.victim_valid.tolist()
            self.step(manager, graph, static, outputs, routes, f"step {step}")
            for row, mapping in enumerate(before):
                read = {mapping[expert] for expert in routes[row][0] if mapping[expert] >= 0}
                wanted = [expert for expert in routes[row][0] if mapping[expert] < 0]
                destinations = [
                    slot
                    for slot, ok in zip(shortlist[row], valid[row])
                    if ok and slot not in read
                ][: len(wanted)]
                self.assertEqual(
                    len(destinations), len(wanted), f"step {step} layer {row} ran out of victims"
                )
                self.assertEqual(len(set(destinations)), len(destinations), "destinations collide")
                cache = manager.caches[sorted(manager.caches)[row]]
                for expert, slot in zip(wanted, destinations):
                    self.assertEqual(int(cache.expert_to_slot[expert]), slot, f"step {step}")
                seen_disqualified += sum(
                    1 for slot, ok in zip(shortlist[row], valid[row]) if ok and slot in read
                )
        self.assertGreater(seen_disqualified, 0, "no forward ever routed a shortlisted slot")
        self.assertEqual(sum(updater.snapshot()["insertion_truncated"]), 0)
        self.assertGreater(sum(updater.snapshot()["insertions"]), 0)

    def test_every_resident_slot_holds_its_own_expert_after_every_forward(self):
        """Byte-exactness of the copy path where it now matters most, since a miss copy lands in a
        live cache row rather than scratch."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        graph, static, outputs = self.capture(manager)
        generator = random.Random(12)
        for step in range(20):
            self.step(manager, graph, static, outputs, _random_routes(generator), f"step {step}")
            assert_slot_rows(self, manager, self.model, f"step {step}")

    def test_a_forward_changes_only_the_rows_it_inserts_into(self):
        """No-touch: the fixed-shape lanes past the miss count must not write anywhere, which is
        why every tensor copies through the count-exact segment kernel instead of an index copy."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        graph, static, outputs = self.capture(manager)
        generator = random.Random(13)
        for step in range(8):
            routes = _random_routes(generator)
            before = {
                layer_id: _cache_bytes(cache) for layer_id, cache in sorted(manager.caches.items())
            }
            mappings = {
                layer_id: cache.expert_to_slot.tolist()
                for layer_id, cache in sorted(manager.caches.items())
            }
            self.step(manager, graph, static, outputs, routes, f"step {step}")
            for row, (layer_id, cache) in enumerate(sorted(manager.caches.items())):
                after = _cache_bytes(cache)
                inserted = {
                    int(cache.expert_to_slot[expert])
                    for expert in routes[row][0]
                    if mappings[layer_id][expert] < 0
                }
                for name, rows in after.items():
                    for slot in range(cache.capacity):
                        if slot in inserted:
                            continue
                        self.assertTrue(
                            torch.equal(rows[slot], before[layer_id][name][slot]),
                            f"step {step} layer {layer_id} slot {slot} {name} changed unexpectedly",
                        )

    def test_the_victim_is_the_lowest_scored_resident_the_forward_does_not_route(self):
        """Ranking quality is stage 1's rule, unchanged; what is new is that routing the shortlisted
        slot pushes the miss onto the next entry instead of evicting a row in use."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        updater = manager.gpu_residency
        graph, static, outputs = self.capture(manager)
        cache = manager.caches[0]
        mapping = cache.expert_to_slot.tolist()
        residents = [expert for expert, slot in enumerate(mapping) if slot >= 0]
        outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
        self.assertGreaterEqual(len(residents), 3)
        self.assertTrue(outsiders)
        lowest, second, miss = residents[0], residents[1], outsiders[0]
        scores = torch.full((EXPERTS,), 100.0)
        scores[lowest], scores[second] = 1.0, 2.0
        updater.insert_scores[0].copy_(scores)
        updater._rank_victims(updater.route_counts > 0)
        torch.cuda.synchronize()
        free = [slot for slot in range(cache.capacity) if slot not in mapping]
        self.assertFalse(free, "a free slot would take the miss before any scored victim")
        others = [[[0, 1]] for _ in range(LAYERS - 1)]
        routes = [[[lowest, miss]]] + others
        self.step(manager, graph, static, outputs, routes, "routed victim")
        self.assertEqual(
            int(cache.expert_to_slot[miss]),
            mapping[second],
            "routing the top-ranked victim must push the miss to the next entry",
        )
        self.assertEqual(int(cache.expert_to_slot[lowest]), mapping[lowest], "a routed row survives")
        self.assertEqual(int(cache.expert_to_slot[second]), -1)
        assert_slot_rows(self, manager, self.model, "routed victim")

    def test_the_shortlist_is_ranked_before_the_first_replay(self):
        """`reset_after_capture` must leave a usable shortlist: the first replay's gather reads it
        before any boundary has run, and a shortlist captured during warm-up would name stale slots."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        updater = manager.gpu_residency
        self.capture(manager)
        self.assertTrue(updater.victims_fresh)
        self.assertTrue(bool(updater.victim_valid.any()))

    def test_decode_forward_is_sync_free(self):
        manager = _manager(self.model, gpu=True, **DIRECT)
        graph, static, _ = self.capture(manager)
        counts = {"global_physical_count": torch.zeros(LAYERS, EXPERTS, dtype=torch.int64, device="cuda")}
        graph.replay()
        manager.on_expert_distribution(_decode_batch(), counts)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for step in range(5):
                static.copy_((static + 3 + step).remainder(EXPERTS))
                graph.replay()
                manager.on_expert_distribution(_decode_batch(), counts)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        self.assertGreater(sum(manager.gpu_residency.snapshot()["insertions"]), 0)

    def test_stage_one_is_unchanged_by_stage_two_existing(self):
        """Stage 1 must stay byte-identical as the fallback: it keeps its scratch rows, its
        device-pair index copy and its boundary insertion path."""
        manager = _manager(_model(), gpu=True, **IOM)
        updater = manager.gpu_residency
        self.assertTrue(updater.insert_on_miss)
        self.assertFalse(updater.insert_direct)
        self.assertIsNone(updater.victims)
        self.assertIsNotNone(updater.insert_tensors)
        for cache in manager.caches.values():
            self.assertEqual(cache.scratch_rows, TOP_K)
        for streamer in manager.streamers.values():
            self.assertNotEqual(streamer._graph_device_pairs, ())


if __name__ == "__main__":
    unittest.main()
