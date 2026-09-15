"""Live prefetch scoring runs from generic tap writes, rewrites stable bank rows, and never synchronizes."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

HIDDEN, EXPERTS, TOP_K = 16, 32, 4


class _Cache:
    def __init__(self, device):
        self.expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        self.expert_to_slot[:6] = torch.arange(6, device=device)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringRuntime(unittest.TestCase):
    def _build(self):
        from sglang.srt.layers.moe.expert_prediction.serving import runtime as serving_runtime
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint
        from sglang.srt.layers.moe.expert_prediction.training import llapor
        from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

        device = torch.device("cuda")
        specs = [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=HIDDEN) for i in range(3)]
        checkpoints = {
            target: LlaporCheckpoint(
                source_layer=target - 1, target_layer=target, group="outer",
                pca=PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(4, HIDDEN), explained_variance=torch.ones(4)),
                model=llapor.build_predictor("outer", pca_rank=4, num_experts=EXPERTS).eval(),
            )
            for target in (1, 2)
        }
        store = FeatureStore(specs=specs, features=serving_runtime.PrefetchScoring.features_for("llapor"),
                             max_rows=1, device=device, hidden_dtype=torch.bfloat16)
        hot_caches = {i: _Cache(device) for i in range(3)}
        scoring = serving_runtime.PrefetchScoring.from_checkpoints(
            predictor="llapor", checkpoints=checkpoints, specs=specs, store=store,
            hot_caches=hot_caches,
            width=8, budget=2, tau=0.95, dtype=torch.bfloat16, device=device,
        )
        return scoring, store, hot_caches

    def _tap(self, store, layer):
        generator = torch.Generator().manual_seed(layer)
        store.write(layer, RouteFeature.ROUTER_INPUT, torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
        store.write(layer, RouteFeature.TOPK_IDS, torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).cuda())
        store.write(layer, RouteFeature.TOPK_WEIGHTS, torch.rand(1, TOP_K, generator=generator).cuda())

    def test_source_tap_rewrites_the_next_layers_bank_row_in_place(self):
        scoring, store, _ = self._build()
        self.assertEqual((scoring.targets, scoring.next_target), ([1, 2], {0: 1, 1: 2}))
        ids, scores = scoring.bank.ids_for(2), scoring.bank.scores_for(2)
        pointer = ids.data_ptr()
        self._tap(store, 1)
        torch.cuda.synchronize()
        self.assertEqual(scoring.bank.ids_for(2).data_ptr(), pointer)
        self.assertNotEqual(scores.abs().sum().item(), 0.0)
        self.assertEqual(scoring.bank.scores_for(1).abs().sum().item(), 0.0)

    def test_forward_taps_never_synchronize(self):
        _, store, _ = self._build()
        for layer in range(3):
            self._tap(store, layer)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for layer in range(3):
                self._tap(store, layer)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_residency_rebind_is_picked_up_live_not_captured_at_init(self):
        # Regression guard: crypto-c9's live A/B failed because a planner captured
        # cache.expert_to_slot by reference at init; the GPU residency update rebinds
        # that attribute (expert_hot_cache.py <- expert_residency_gpu.py), so a captured
        # reference goes stale. PrefetchScoring must read hot_caches[layer].expert_to_slot
        # fresh on every observe(), not once at construction time.
        scoring, store, hot_caches = self._build()
        self._tap(store, 0)
        torch.cuda.synchronize()
        before = scoring.recall.snapshot()[1]
        # Rebind (not mutate in place) the cache's expert_to_slot attribute, as the GPU
        # residency update does when it takes over slot bookkeeping.
        hot_caches[1].expert_to_slot = torch.arange(EXPERTS, device="cuda")
        self._tap(store, 1)
        torch.cuda.synchronize()
        after = scoring.recall.snapshot()[1]
        # With every expert now resident (rebound mapping), no route can be a miss.
        self.assertEqual(after[0] - before[0], 0)

    def test_replay_updates_candidates_and_metrics_without_python(self):
        scoring, store, _ = self._build()
        inputs = {layer: (torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                          torch.zeros(1, TOP_K, dtype=torch.long, device="cuda"),
                          torch.zeros(1, TOP_K, device="cuda")) for layer in range(3)}
        for layer, (x, ids, w) in inputs.items():
            ids.copy_(torch.arange(TOP_K, device="cuda").unsqueeze(0) + 6 + layer)

        def forward():
            for layer, (x, ids, w) in inputs.items():
                store.write(layer, RouteFeature.ROUTER_INPUT, x)
                store.write(layer, RouteFeature.TOPK_IDS, ids)
                store.write(layer, RouteFeature.TOPK_WEIGHTS, w)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        before = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        for _ in range(4):
            inputs[1][0].copy_(torch.randn(1, HIDDEN, device="cuda").to(torch.bfloat16))
            graph.replay()
        torch.cuda.synchronize()
        after = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        self.assertEqual(after - before, 4 * TOP_K)


if __name__ == "__main__":
    unittest.main()
