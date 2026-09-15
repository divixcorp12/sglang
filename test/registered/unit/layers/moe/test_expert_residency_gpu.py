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

import random
import tempfile
import unittest
from types import SimpleNamespace

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
