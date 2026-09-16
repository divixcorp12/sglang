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

    def _run_mode(self, gpu, doorbell):
        current_model, reference_model = _model(), _model()
        current = _manager(
            current_model,
            gpu,
            expert_doorbell=doorbell,
            doorbell_cpu_core=DOORBELL_SPIN_CORE,
        )
        reference = _manager(reference_model, gpu)
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
                for layer in range(LAYERS):
                    for field in ("hot_hits", "miss_rows", "promotions", "evictions"):
                        self.assertEqual(
                            current_counters[str(layer)][field],
                            reference_counters[str(layer)][field],
                            f"{context} layer {layer} {field}",
                        )
                assert_states_equal(self, device_state(current), device_state(reference), context)
                promotions = sum(current_counters[str(layer)]["promotions"] for layer in range(LAYERS))
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


if __name__ == "__main__":
    unittest.main()
