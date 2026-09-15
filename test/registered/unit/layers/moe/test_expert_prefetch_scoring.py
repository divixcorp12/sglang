"""Prefetch scoring is device-only: bf16 scorers match training, bank writes and counters never sync."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _llapor_checkpoint(group="middle"):
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint

    torch.manual_seed(1)
    model = llapor.build_predictor(group, pca_rank=RANK, num_experts=EXPERTS).eval()
    pca = PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(RANK, HIDDEN), explained_variance=torch.ones(RANK))
    return LlaporCheckpoint(source_layer=0, target_layer=1, group=group, pca=pca, model=model)


def _apex_checkpoint():
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint

    torch.manual_seed(2)
    return ApexCheckpoint(
        layer_id=1, top_k=TOP_K, ranker=apex.Ranker(HIDDEN, EXPERTS).eval(),
        cdf=apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1).eval(),
    )


def _route_features(rows=3, device="cpu"):
    generator = torch.Generator().manual_seed(3)
    router_input = torch.randn(rows, HIDDEN, generator=generator)
    topk_ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(rows)])
    topk_weights = torch.rand(rows, TOP_K, generator=generator)
    return router_input.to(device), topk_ids.to(device), topk_weights.to(device)


class TestPrefetchScoringCpu(unittest.TestCase):
    def test_llapor_fp32_scorer_equals_training_forward(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        for group in ("outer", "middle"):
            checkpoint = _llapor_checkpoint(group)
            router_input, topk_ids, topk_weights = _route_features()
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=checkpoint.pca, num_experts=EXPERTS)
            expected = torch.sigmoid(checkpoint.model(u))
            scorer = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cpu"))
            torch.testing.assert_close(scorer(router_input, topk_ids, topk_weights), expected, rtol=1e-5, atol=1e-6)

    def test_apex_scorer_zeroes_ranks_beyond_top_k_plus_depth(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer

        checkpoint = _apex_checkpoint()
        pre_mixer = torch.randn(2, HIDDEN)
        scorer = ApexScorer(checkpoint, num_experts=EXPERTS, tau=0.9, dtype=torch.float32, device=torch.device("cpu"))
        scores = scorer(pre_mixer)
        probabilities = torch.softmax(checkpoint.ranker(pre_mixer), dim=-1)
        depth = apex.select_depth(checkpoint.cdf(pre_mixer), 0.9, EXPERTS - TOP_K)
        for row in range(2):
            kept = (scores[row] > 0).sum().item()
            self.assertEqual(kept, min(EXPERTS, TOP_K + depth[row].item()))
            best = torch.topk(probabilities[row], kept).indices
            torch.testing.assert_close(scores[row, best], probabilities[row, best], rtol=1e-5, atol=1e-6)

    def test_bank_keeps_top_width_of_summed_rows(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank

        bank = PrefetchCandidateBank(layer_ids=[3, 7], width=2, device=torch.device("cpu"))
        scores = torch.zeros(2, EXPERTS)
        scores[0, 5], scores[1, 5], scores[0, 9], scores[1, 11] = 0.4, 0.4, 0.7, 0.6
        bank.write(7, scores)
        self.assertEqual(bank.ids_for(7).tolist(), [5, 9])
        torch.testing.assert_close(bank.scores_for(7), torch.tensor([0.8, 0.7]))
        self.assertEqual(bank.ids_for(3).tolist(), [0, 1])

    def test_budget_recall_counts_nonresident_routes_within_budget(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall

        recall = BudgetRecall(layer_ids=[1], budget=2, device=torch.device("cpu"))
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[[2, 4]] = torch.tensor([0, 1])
        candidates = torch.tensor([2, 6, 4, 8, 10], dtype=torch.long)
        topk_ids = torch.tensor([[2, 8, 10, 12]])
        recall.observe(target_layer=1, candidate_ids=candidates, topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        self.assertEqual(recall.snapshot(), {1: (3, 1)})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringCuda(unittest.TestCase):
    def _parts(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        device = torch.device("cuda")
        scorer = LlaporScorer(_llapor_checkpoint(), num_experts=EXPERTS, dtype=torch.bfloat16, device=device)
        bank = PrefetchCandidateBank(layer_ids=[0, 1], width=6, device=device)
        recall = BudgetRecall(layer_ids=[1], budget=3, device=device)
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        expert_to_slot[:10] = torch.arange(10, device=device)
        return scorer, bank, recall, expert_to_slot

    def test_decode_step_never_synchronizes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)
        bank.write(1, scorer(router_input, topk_ids, topk_weights))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_graph_replay_matches_eager_for_new_routes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)

        def step():
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for seed in range(5):
            generator = torch.Generator().manual_seed(100 + seed)
            router_input.copy_(torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
            topk_ids.copy_(torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).to("cuda"))
            topk_weights.copy_(torch.rand(1, TOP_K, generator=generator).to("cuda"))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.topk(scorer(router_input, topk_ids, topk_weights).sum(0), bank.width)
            self.assertEqual(bank.ids_for(1).tolist(), expected.indices.tolist())

    def test_bf16_scorer_agrees_with_fp32_top_candidates(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        checkpoint = _llapor_checkpoint()
        router_input, topk_ids, topk_weights = _route_features(rows=64, device="cuda")
        full = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cuda"))
        half = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.bfloat16, device=torch.device("cuda"))
        top_full = torch.topk(full(router_input, topk_ids, topk_weights), 8).indices
        top_half = torch.topk(half(router_input.to(torch.bfloat16), topk_ids, topk_weights), 8).indices
        overlap = (top_full.unsqueeze(2) == top_half.unsqueeze(1)).any(2).float().mean().item()
        self.assertGreaterEqual(overlap, 0.95)


if __name__ == "__main__":
    unittest.main()
