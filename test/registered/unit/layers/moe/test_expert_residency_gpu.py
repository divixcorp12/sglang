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

from sglang.kernels.ops.moe.expert_insert_rows import insert_expert_rows
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


def _gather_harness(manager, tokens=1):
    """Static routes, per-layer output rows and a forward that gathers every layer into them."""
    streamers = [manager.streamers[layer_id] for layer_id in sorted(manager.streamers)]
    static = torch.zeros((LAYERS, tokens, TOP_K), dtype=torch.int32, device="cuda")
    static[:, :, 1] = 1
    outputs = [
        {
            name: torch.zeros((tokens * TOP_K,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device="cuda")
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

    def test_prefetch_covered_misses_and_the_pull_row_stay_out_of_insertions(self):
        """A route served from the dedicated pull row is not a demand miss, so it is not inserted,
        and insertion never reads or writes the pull row (``capacity + scratch_rows``)."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.runtime import PrefetchPuller

        with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL.override("true"):
            manager = _manager(self.model, gpu=True, **IOM)
        cache, streamer = manager.caches[0], manager.streamers[0]
        self.assertTrue(cache.reserves_prefetch_pull_row)
        bank = PrefetchCandidateBank(layer_ids=[0], width=1, device="cuda")
        puller = PrefetchPuller(bank=bank, layer_ids=[0], hot_caches={0: cache}, device="cuda")
        streamer.prefetch_puller = puller
        pull_row = puller.slot_for(0)
        self.assertEqual(pull_row, cache.capacity + cache.scratch_rows)
        manager.discard_graph_capture_routes()
        mapping = cache.expert_to_slot.tolist()
        covered, missed = [expert for expert, slot in enumerate(mapping) if slot < 0][:2]
        residents = [expert for expert, slot in enumerate(mapping) if slot >= 0]
        others = [[[0, 1]] for _ in range(LAYERS - 1)]

        def forward(routes):
            for layer, layer_streamer in sorted(manager.streamers.items()):
                layer_streamer.gather(torch.tensor(routes[layer], dtype=torch.int32, device="cuda"))
            torch.cuda.synchronize()
            manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})

        scores = torch.zeros(1, EXPERTS, device="cuda")
        scores[0, covered] = 1.0
        bank.write(0, scores, expert_to_slot=cache.expert_to_slot)
        puller.post_target(0)
        forward([[[covered, missed]]] + others)
        self.assertEqual(puller.stats[0].snapshot()[0], 1, "the covered route was not served from the pull row")
        self.assertEqual(
            int(streamer.row_plan.count.item()), 1, "the covered route still took a demand scratch row"
        )
        pull_bytes = {name: tensor[pull_row].clone() for name, tensor in cache.tensors.items()}
        # The next forward's first gather applies that forward's boundary; a later forward would be
        # free to evict the insertion again, which is ordinary victim churn, not this invariant.
        forward([[residents[:2]]] + others)
        self.assertGreaterEqual(int(cache.expert_to_slot[missed]), 0, "the demand miss was not inserted")
        self.assertEqual(int(cache.expert_to_slot[covered]), -1, "a pull-covered route was inserted")
        self.assertTrue((cache.expert_to_slot < cache.capacity).all())
        self.assertEqual(manager.gpu_residency.snapshot()["insertions"][0], 1)
        for name, tensor in cache.tensors.items():
            self.assertTrue(torch.equal(tensor[pull_row], pull_bytes[name]), f"pull row {name} was written")
        assert_slot_rows(self, manager, self.model, "pull row")

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


FUSED = dict(IOM, fused_insert=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFusedInsert(unittest.TestCase):
    """SGLANG_MOE_HOT_FUSED_INSERT: the same boundary copy in one masked pass instead of two.

    The claim being defended is that this is a cost change and nothing else, so
    the tests compare it byte for byte against the ``index_copy_`` path it
    replaces -- over the whole cache, not only the slots the boundary wrote,
    because a kernel that also disturbed a scratch row or an unrelated slot
    would satisfy the residency reference and still be wrong.
    """

    def _run(self, fused, steps=30, seed=3):
        """Drive one manager through a fixed route sequence; return its bytes and state."""
        model = _model()
        options = dict(FUSED) if fused else dict(IOM)
        manager = _manager(model, gpu=True, seed_scale=(1, 3, 9), **options)
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
        generator = random.Random(seed)
        for _ in range(steps):
            routes = _random_routes(generator)
            static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        return (
            manager,
            model,
            [_cache_bytes(cache) for _, cache in sorted(manager.caches.items())],
            device_state(manager),
            manager.gpu_residency.snapshot(),
        )

    def test_the_fused_kernel_is_byte_identical_to_the_index_copy_path(self):
        """Same routes, same seed, both paths: every byte of every cache tensor agrees, including
        the scratch rows and the slots this run never inserted into, and so does every piece of
        residency state and every insertion counter."""
        eager_manager, eager_model, eager_bytes, eager_state, eager_counts = self._run(False)
        fused_manager, _, fused_bytes, fused_state, fused_counts = self._run(True)
        self.assertEqual(fused_state, eager_state)
        self.assertEqual(fused_counts, eager_counts)
        self.assertGreater(sum(eager_counts["insertions"]), 0, "the run must actually insert")
        for row, (eager_rows, fused_rows) in enumerate(zip(eager_bytes, fused_bytes)):
            self.assertEqual(sorted(fused_rows), sorted(eager_rows))
            for name, expected in eager_rows.items():
                self.assertTrue(
                    torch.equal(fused_rows[name], expected),
                    f"layer {row} {name} differs between the fused and index-copy paths",
                )
        assert_slot_rows(self, fused_manager, eager_model, "fused")

    def test_the_fused_boundary_still_matches_the_host_reference(self):
        """Byte-for-byte agreement with the other path would be worthless if both were wrong, so
        the fused path is also checked against the independent host model of the boundary."""
        model = _model()
        manager = _manager(model, gpu=True, seed_scale=(1, 3, 9), **FUSED)
        reference = _InsertOnMissReference(manager)
        # Borrow stage 1's own capture/replay/assert loop verbatim, so the fused path is held to
        # the same checks rather than a re-implementation of them that might drift.
        harness = TestInsertOnMiss(methodName="test_an_unknown_stage_is_refused")
        harness.model = model
        graph, static, outputs = harness.capture(manager)
        reference.assert_matches(self, manager, "after capture")
        generator = random.Random(3)
        for step in range(20):
            harness.replay(
                manager, reference, graph, static, outputs, _random_routes(generator), f"step {step}"
            )

    def test_an_idle_lane_moves_no_bytes_and_a_masked_lane_is_not_read(self):
        """The whole point of the mask is that traffic follows the insertion count while the launch
        shape stays fixed. Drive the kernel directly with one active lane out of many and a
        deliberately poisoned source for every inactive lane: a kernel that loaded an idle lane
        would copy poison into a live slot."""
        lanes, slots, row_bytes = 6, 10, 777
        rows = torch.randint(0, 256, (slots + lanes, row_bytes), dtype=torch.uint8, device="cuda")
        before = rows.clone()
        sources = torch.full((lanes,), slots, dtype=torch.int64, device="cuda")
        sources[0] = slots + 1
        destinations = torch.zeros(lanes, dtype=torch.int64, device="cuda")
        destinations[0] = 4
        active = torch.zeros(lanes, dtype=torch.int32, device="cuda")
        active[0] = 1
        insert_expert_rows(rows, sources, destinations, active)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(rows[4], before[slots + 1]), "the active lane must copy its row")
        untouched = [row for row in range(slots + lanes) if row != 4]
        self.assertTrue(
            torch.equal(rows[untouched], before[untouched]),
            "an inactive lane wrote a row it should never have read",
        )

    def test_the_flag_is_refused_where_it_would_measure_the_wrong_path(self):
        """Stage 0 has no boundary copy loop and stage 2 lands its copies in the gather, so the flag
        does nothing there. A silent no-op would let an arm report a fused number for an unfused
        run, so both are refused rather than accepted."""
        for stage in (0, 2):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_FUSED_INSERT"):
                    _manager(
                        _model(),
                        gpu=True,
                        update_decode_forwards=1,
                        insert_on_miss=stage,
                        fused_insert=True,
                    )

    def test_the_environment_switch_selects_the_kernel_and_defaults_off(self):
        from sglang.srt.environ import envs

        manager = _manager(_model(), gpu=True, **IOM)
        self.assertFalse(manager.gpu_residency.fused_insert)
        self.assertIsNone(manager.gpu_residency.insert_active)
        with envs.SGLANG_MOE_HOT_FUSED_INSERT.override(True):
            manager = _manager(_model(), gpu=True, **IOM)
        self.assertTrue(manager.gpu_residency.fused_insert)
        self.assertEqual(
            manager.gpu_residency.insert_active.shape,
            (LAYERS, manager.gpu_residency.miss_rows),
        )

    def test_the_startup_warm_up_moves_no_bytes(self):
        """Triton JITs on first call and specialises on row length, and the boundary's first call
        can land inside graph capture, so startup compiles every specialisation with no lane
        active. A warm-up that moved anything would corrupt a seeded cache before the first
        forward. Replay startup's own launch on the built caches and require it to be inert.

        Comparing two separately built managers would not test this: cache rows no slot holds are
        allocated with ``torch.empty`` and differ between allocations whatever the warm-up did.
        """
        model = _model()
        manager = _manager(model, gpu=True, **FUSED)
        updater = manager.gpu_residency
        self.assertTrue(torch.equal(updater.insert_active, torch.zeros_like(updater.insert_active)))
        before = [_cache_bytes(cache) for _, cache in sorted(manager.caches.items())]
        for row, tensors in enumerate(updater.insert_tensors):
            for rows_view in tensors:
                insert_expert_rows(
                    rows_view,
                    updater.insert_sources[row],
                    updater.insert_destinations[row],
                    updater.insert_active[row],
                )
        torch.cuda.synchronize()
        for row, (cache, expected) in enumerate(zip([c for _, c in sorted(manager.caches.items())], before)):
            actual = _cache_bytes(cache)
            for name, rows in expected.items():
                self.assertTrue(torch.equal(actual[name], rows), f"layer {row} {name} moved during warm-up")
        assert_slot_rows(self, manager, model, "after warm-up")

    def test_a_malformed_plan_is_refused_instead_of_corrupting_a_row(self):
        rows = torch.zeros((8, 16), dtype=torch.uint8, device="cuda")
        lane = torch.zeros(2, dtype=torch.int64, device="cuda")
        active = torch.ones(2, dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "contiguous 2D row view"):
            insert_expert_rows(rows.t(), lane, lane, active)
        with self.assertRaisesRegex(ValueError, "same lanes"):
            insert_expert_rows(rows, lane, lane[:1], active)
        with self.assertRaisesRegex(TypeError, "integer tensor"):
            insert_expert_rows(rows, lane, lane, active.to(torch.float32))
        with self.assertRaisesRegex(ValueError, "same device"):
            insert_expert_rows(rows, lane, lane, active.cpu())


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

    def capture(self, manager, tokens=1):
        static, outputs, forward = _gather_harness(manager, tokens)
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

    def test_a_layer_with_nothing_on_the_host_is_refused(self):
        """Stage 2 folds the device-source tensors into the segment kernel, which reads its row
        count on the device and so cannot size its grid to them -- ~0.125 ms/row against ~0.0105
        for the index copy it replaces, on a production-shaped working set (see
        `_init_insert_on_miss` for both regimes). That is free only
        because the rows it takes over are the per-expert scalars, 8 B/row in production, while the
        megabyte rows were already on that kernel. A layer holding every tensor on the device would
        put its full rows on the slower path instead, so refuse it."""
        model = _model()
        for layer_id in range(LAYERS):
            layer = model.get_submodule(str(layer_id))
            for name in NVFP4_STREAM_TENSORS[:4]:
                setattr(layer, name, getattr(layer, name).cuda())
            layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        with self.assertRaisesRegex(ValueError, "host-source expert tensors"):
            _manager(model, gpu=True, **DIRECT)

    def test_a_layer_too_small_to_guarantee_a_victim_is_refused(self):
        """Below twice the gather width a forward could route every shortlist entry, and with no
        scratch row left there would be nowhere safe for that miss to land. Refuse, never truncate.
        The allocator gives every layer that floor first, so only a budget short of all the floors
        together leaves a layer under it: here one slot short of three floors."""
        floors = LAYERS * 2 * TOP_K
        with self.assertRaisesRegex(ValueError, "twice its graph-gather rows"):
            _manager(_model(), gpu=True, budget_bytes=56 * (floors - 1), **DIRECT)

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

    def test_exl3_direct_ack_violation_cannot_publish_a_resident(self):
        """The copy may have started (go_count > 0) before ack detects a
        source generation violation. Only the post-ack keep word permits commit."""
        manager = _manager(_model(), gpu=True, **DIRECT)
        updater = manager.gpu_residency
        streamer = updater.streamers[0]
        row = 0
        target = 0
        before = updater.slot_to_expert[row].clone()
        mapping_before = updater.mapping[row].clone()
        new_expert = next(e for e in range(EXPERTS) if mapping_before[e].item() < 0)
        streamer._graph_source_rows[0] = new_expert
        delivered = torch.tensor([1], dtype=torch.int32, device="cuda")
        keep = torch.tensor([0.0], dtype=torch.float32, device="cuda")
        streamer.row_backend = SimpleNamespace(
            name="exl3_ram_miss", delivered_count=delivered, keep=keep,
        )
        destinations = torch.zeros(updater.miss_rows, dtype=torch.long, device="cuda")
        live = torch.zeros(updater.miss_rows, dtype=torch.bool, device="cuda")
        live[0] = True
        updater._pending_commit = (row, streamer, destinations, live)
        updater.commit_gather()
        self.assertTrue(torch.equal(updater.slot_to_expert[row], before))
        self.assertTrue(torch.equal(updater.mapping[row], mapping_before))
        self.assertEqual(updater.insertion_truncated[row].item(), 0)
        keep.fill_(1.0)
        updater._pending_commit = (row, streamer, destinations, live)
        updater.commit_gather()
        self.assertEqual(updater.slot_to_expert[row, target].item(), new_expert)
        self.assertEqual(updater.mapping[row, new_expert].item(), target)

    def test_exl3_prefill_boundary_scores_without_dense_promotion(self):
        from sglang.srt.layers.moe.expert_residency_gpu import _PREFILL_PHASE

        manager = _manager(_model(), gpu=True, **DIRECT)
        updater = manager.gpu_residency
        for streamer in updater.streamers:
            streamer.format = SimpleNamespace(key="exl3")
        updater.enabled.fill_(True)
        updater.route_counts[:, 3].fill_(4.0)
        before = updater.scores[:, 3].clone()
        updater._promote = lambda *args: self.fail("EXL3 prefill cannot index dense host rows")
        updater._apply(updater.enabled.clone(), updater.prefill_promotions, _PREFILL_PHASE)
        self.assertTrue(bool((updater.scores[:, 3] > before).all()))
        self.assertTrue(bool((updater.route_counts == 0).all()))
        self.assertTrue(updater.victims_fresh)

    # ----- the capacity proof, exercised -----

    def test_no_forward_ever_lands_a_copy_in_a_row_it_reads(self):
        """The safety property itself, checked per replay against the pre-gather mapping: a
        destination is never a slot this forward routes to, destinations are distinct, every miss
        finds a slot, and a routed resident is never the one evicted.

        The shortlist this forward uses is ranked by the boundary at the top of the same forward,
        before any gather, so it cannot be read from outside beforehand. These are the observable
        consequences instead: where each miss landed, and what survived.
        """
        manager = _manager(self.model, gpu=True, **DIRECT)
        updater = manager.gpu_residency
        graph, static, outputs = self.capture(manager)
        generator = random.Random(11)
        inserted_total = 0
        for step in range(30):
            routes = _random_routes(generator)
            before = [cache.expert_to_slot.tolist() for _, cache in sorted(manager.caches.items())]
            self.step(manager, graph, static, outputs, routes, f"step {step}")
            for row, mapping in enumerate(before):
                cache = manager.caches[sorted(manager.caches)[row]]
                read = {mapping[expert] for expert in routes[row][0] if mapping[expert] >= 0}
                wanted = [expert for expert in routes[row][0] if mapping[expert] < 0]
                landed = [int(cache.expert_to_slot[expert]) for expert in wanted]
                self.assertNotIn(-1, landed, f"step {step} layer {row}: a miss found no slot")
                self.assertFalse(
                    read & set(landed),
                    f"step {step} layer {row}: copied into a row this forward reads",
                )
                self.assertEqual(len(set(landed)), len(landed), f"step {step} destinations collide")
                for expert in routes[row][0]:
                    if mapping[expert] >= 0:
                        self.assertEqual(
                            int(cache.expert_to_slot[expert]),
                            mapping[expert],
                            f"step {step} layer {row}: a routed resident was evicted",
                        )
                inserted_total += len(wanted)
        self.assertGreater(inserted_total, 0, "no forward ever missed")
        self.assertEqual(sum(updater.snapshot()["insertion_truncated"]), 0)
        self.assertEqual(sum(updater.snapshot()["insertions"]), inserted_total)

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
        """Unrouted residents rank lowest score first, as in stage 1; what is new is that routing the
        shortlisted slot pushes the miss onto the next entry instead of evicting a row in use."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        updater = manager.gpu_residency
        graph, static, outputs = self.capture(manager)
        cache = manager.caches[0]
        mapping = cache.expert_to_slot.tolist()
        residents = [expert for expert, slot in enumerate(mapping) if slot >= 0]
        outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
        self.assertGreaterEqual(len(residents), 4)
        self.assertTrue(outsiders)
        lowest, second, miss = residents[0], residents[1], outsiders[0]
        self.assertEqual(len(residents), cache.capacity, "a free slot would take the miss first")
        others = [[[0, 1]] for _ in range(LAYERS - 1)]
        # Neutral step first, so neither candidate carries a route count into the ranking that
        # the boundary at the top of the scored step folds into its score.
        self.step(manager, graph, static, outputs, [[[residents[2], residents[3]]]] + others, "warm")
        scores = torch.full((EXPERTS,), 100.0)
        # Far enough apart that the boundary's decay and route counts cannot reorder them.
        scores[lowest], scores[second] = 1.0, 50.0
        updater.insert_scores[0].copy_(scores)
        torch.cuda.synchronize()
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

    def test_the_fused_route_planner_drives_direct_exactly_like_the_generic_one(self):
        """`gather_destinations` takes a miss lane's rank from ``remap - scratch_base``, its expert
        from ``_graph_source_rows[rank]`` and its liveness from ``_graph_miss_count``. Under
        SGLANG_MOE_EXPERT_FUSED_PLAN one kernel writes all three, so a drift in that layout would
        copy misses into the wrong slots. Twins, one per planner, must stay identical."""
        from sglang.srt.environ import envs

        generic = _manager(self.model, gpu=True, **DIRECT)
        with envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override("true"):
            fused = _manager(_model(), gpu=True, **DIRECT)
        for streamer in fused.streamers.values():
            streamer.row_planner.route_plan = unittest.mock.Mock(
                side_effect=AssertionError("the fused manager fell back to the generic planner")
            )
        twins = [(manager, *self.capture(manager)) for manager in (generic, fused)]
        generator = random.Random(14)
        for step in range(30):
            routes = _random_routes(generator)
            for manager, graph, static, outputs in twins:
                self.step(manager, graph, static, outputs, routes, f"step {step}")
            context = f"step {step}"
            assert_states_equal(self, device_state(fused), device_state(generic), context)
            assert_slot_rows(self, fused, self.model, context)
            self.assertEqual(fused.gpu_residency.snapshot(), generic.gpu_residency.snapshot(), context)
            for layer_id, streamer in generic.streamers.items():
                twin = fused.streamers[layer_id]
                self.assertEqual(twin.graph_counters.tolist(), streamer.graph_counters.tolist(), context)
                self.assertEqual(
                    twin.graph_unique_counters.tolist(), streamer.graph_unique_counters.tolist(), context
                )
        self.assertGreater(sum(fused.gpu_residency.snapshot()["insertions"]), 0, "no forward missed")

    def test_a_speculative_verify_forward_keeps_direct_exact(self):
        """NEXTN replays a TARGET_VERIFY graph whose gather serves several tokens, with repeated
        experts across them, and the speculative worker then commits fewer tokens than it drafted.
        The commit only corrects the host clock's decay tokens, never a boundary, so the device
        clock must still count every forward the host does. Each miss must still land in a slot no
        token of the forward reads. Runs with the fused planner on, as production does: a
        multi-token gather must fall back to the generic planner rather than fail."""
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        tokens = 2
        with envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override("true"):
            manager = _manager(
                self.model, gpu=True, graph_gather_batch_size=tokens,
                budget_bytes=56 * LAYERS * (EXPERTS - 2), **DIRECT,
            )
        updater = manager.gpu_residency
        verify = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=tokens, is_draft_input=lambda: False),
            extend_num_tokens=tokens,
            batch_size=1,
        )
        graph, static, outputs = self.capture(manager, tokens)
        generator = random.Random(15)
        inserted_total = 0
        for step in range(30):
            context = f"step {step}"
            routes = [
                [generator.sample(range(EXPERTS), TOP_K) for _ in range(tokens)]
                for _ in range(LAYERS)
            ]
            before = [cache.expert_to_slot.tolist() for _, cache in sorted(manager.caches.items())]
            static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            for layer, mapping in enumerate(before):
                source_layer = self.model.get_submodule(str(layer))
                flat = [expert for token in routes[layer] for expert in token]
                for name in NVFP4_STREAM_TENSORS:
                    source = getattr(source_layer, name)
                    expected = source[torch.tensor(flat).to(source.device)].reshape(-1).view(torch.uint8).cpu()
                    actual = outputs[layer][name].view(torch.uint8).reshape(-1).cpu()
                    self.assertTrue(torch.equal(actual, expected), f"{context} layer {layer} {name}")
                cache = manager.caches[sorted(manager.caches)[layer]]
                read = {mapping[expert] for expert in flat if mapping[expert] >= 0}
                wanted = sorted({expert for expert in flat if mapping[expert] < 0})
                landed = [int(cache.expert_to_slot[expert]) for expert in wanted]
                self.assertNotIn(-1, landed, f"{context} layer {layer}: a miss found no slot")
                self.assertFalse(read & set(landed), f"{context} layer {layer}: copied into a row it reads")
                self.assertEqual(len(set(landed)), len(landed), f"{context} destinations collide")
                for expert in set(flat) - set(wanted):
                    self.assertEqual(
                        int(cache.expert_to_slot[expert]), mapping[expert],
                        f"{context} layer {layer}: a routed resident was evicted",
                    )
                inserted_total += len(wanted)
            manager.on_expert_distribution(verify, {"global_physical_count": _counts(routes)})
            manager.on_speculative_commit(generator.randint(1, tokens))
            torch.cuda.synchronize()
            assert_slot_rows(self, manager, self.model, context)
            self.assertEqual(int(updater.forwards.item()), manager._boundary_clock.forwards, context)
        self.assertGreater(inserted_total, 0, "no forward ever missed")
        self.assertEqual(sum(updater.snapshot()["insertion_truncated"]), 0)
        self.assertEqual(sum(updater.snapshot()["insertions"]), inserted_total)

    def test_a_low_scored_layer_still_gets_its_slot_floor(self):
        """The seed splits the budget across layers by score, so a layer the seed rates low could
        fall under DIRECT's twice-the-gather-rows floor and refuse a budget that holds every
        layer's floor. Here the budget is exactly three floors: by score alone layers 1 and 2 would
        take 10 slots each and leave layer 0 with 4 for a 2-token gather's 8. Every layer gets
        its floor, from its own best experts."""
        tokens, scale = 2, (0.01, 1, 1)
        floor = 2 * tokens * TOP_K
        manager = _manager(
            self.model, gpu=True, seed_scale=scale, graph_gather_batch_size=tokens,
            budget_bytes=56 * LAYERS * floor, **DIRECT,
        )
        capacities = [manager.caches[layer].capacity for layer in range(LAYERS)]
        self.assertEqual(capacities, [floor] * LAYERS)
        best = sorted(range(EXPERTS), key=lambda expert: (-((expert * 7) % 5), expert))[:floor]
        resident = {expert for expert, slot in enumerate(manager.caches[0].expert_to_slot.tolist()) if slot >= 0}
        self.assertEqual(resident, set(best))

    def test_misses_after_a_short_prefill_that_routed_every_resident_land_in_their_own_slots(self):
        """A replay after a boundary-less prefill that routed every resident must still copy each
        miss into its own slot, never all of them into slot 0."""
        manager = _manager(self.model, gpu=True, **DIRECT)
        graph, static, outputs = self.capture(manager)

        def eager_prefill(tokens, experts):
            routes = [[[experts[(token * TOP_K + k) % len(experts)] for k in range(TOP_K)] for token in range(tokens)]
                      for _ in range(LAYERS)]
            for layer, streamer in sorted(manager.streamers.items()):
                streamer.gather(torch.tensor(routes[layer], dtype=torch.int32, device="cuda"))
            manager.on_expert_distribution(_prefill_batch(tokens), {"global_physical_count": _counts(routes)})
            torch.cuda.synchronize()

        eager_prefill(24, list(range(EXPERTS)))
        assert_slot_rows(self, manager, self.model, "boundary prefill")
        eager_prefill(6, list(range(EXPERTS)))
        for step in range(3):
            routes = []
            for _, cache in sorted(manager.caches.items()):
                absent = [expert for expert, slot in enumerate(cache.expert_to_slot.tolist()) if slot < 0]
                self.assertGreaterEqual(len(absent), TOP_K, "the cache holds every expert; nothing can miss")
                routes.append([absent[:TOP_K]])
            self.step(manager, graph, static, outputs, routes, f"verify {step}")
            assert_slot_rows(self, manager, self.model, f"verify {step}")
        self.assertEqual(sum(manager.gpu_residency.snapshot()["insertion_truncated"]), 0)

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

    def test_the_metrics_trace_and_the_snapshot_report_the_same_insertions(self):
        """`_trace_sources` fills hot-cache.metrics.jsonl; `snapshot()` has no caller in the serving
        tree. So every test here can pass while a real arm's metrics report insertions = 0, which
        reads as the feature being off rather than as a bug.

        Both stages, because the durable failure is someone moving a counter for one stage and
        updating only one of the two readers. Pinning them to the same tensor object is what makes
        that impossible rather than merely currently-true.
        """
        for stage, options in ((1, IOM), (2, DIRECT)):
            with self.subTest(stage=stage):
                model = _model()
                manager = _manager(model, gpu=True, **options)
                updater = manager.gpu_residency
                graph, static, outputs = self.capture(manager)
                generator = random.Random(20 + stage)
                for step in range(6):
                    self.step(manager, graph, static, outputs, _random_routes(generator), f"s{step}")

                sources = manager._trace_sources()
                snapshot = updater.snapshot()
                for name in ("insertions", "insertion_evictions", "insertion_truncated"):
                    published = sources[f"gpu_residency:{name}"]
                    self.assertIs(
                        published,
                        updater.insertion_counters()[name],
                        f"stage {stage}: the trace publishes a different tensor than snapshot reads",
                    )
                    self.assertEqual(published.cpu().tolist(), snapshot[name], f"stage {stage} {name}")
                self.assertGreater(
                    sum(snapshot["insertions"]), 0, f"stage {stage} inserted nothing to report"
                )
                self.assertGreater(sum(sources["gpu_residency:insertions"].cpu().tolist()), 0)
                # The per-layer decode counters are folded from the same device values.
                decode = manager.snapshot_counters()["decode"]
                self.assertEqual(
                    sum(decode[str(row)]["insertions"] for row in range(LAYERS)),
                    sum(snapshot["insertions"]),
                    f"stage {stage}: per-layer counters disagree with the snapshot",
                )

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
