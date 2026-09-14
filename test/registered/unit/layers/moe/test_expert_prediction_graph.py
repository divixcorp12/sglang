import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

ROWS = 8
HIDDEN = 32
EXPERTS = 16
TOP_K = 4


class FakeTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.float().softmax(dim=-1), TOP_K, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class FakeMoE(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = EXPERTS
        self.hidden_size = HIDDEN

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        self.topk = FakeTopK()
        self.experts = FakeMoE(layer_id)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in (0, 1, 2))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertPredictionGraph(unittest.TestCase):
    def test_replay_refreshes_taps_and_scoring_never_syncs(self):
        device = torch.device("cuda", 0)
        model = FakeModel().to(device)
        runtime = ExpertPredictionRuntime.build(
            model=model,
            predictor_names=("popularity", "affinity"),
            device=device,
            hidden_dtype=torch.float32,
            max_rows=ROWS,
            max_candidates=8,
            hot_caches={1: SimpleNamespace(expert_to_slot=torch.full((EXPERTS,), -1, device=device))},
            log_interval=10**9,
            metrics_path=None,
            topk_type=FakeTopK,
            experts_type=FakeMoE,
        )
        static_input = torch.zeros(ROWS, HIDDEN, device=device)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.zeros(ROWS, dtype=torch.int64, device=device),
            batch_size=ROWS,
            spec_info=None,
            extend_num_tokens=ROWS,
        )
        with torch.inference_mode():
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):
                    model(static_input)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                model(static_input)

            for seed in (1, 2):
                generator = torch.Generator(device=device).manual_seed(seed)
                static_input.copy_(torch.randn(ROWS, HIDDEN, device=device, generator=generator))
                graph.replay()
                torch.cuda.synchronize()
                for layer_id, block in zip((0, 1, 2), model.layers):
                    logits = block.gate(static_input)
                    expected = torch.topk(logits.float().softmax(dim=-1), TOP_K, dim=-1).indices
                    self.assertTrue(
                        torch.equal(
                            runtime.store.view(layer_id, RouteFeature.TOPK_IDS, ROWS),
                            expected.long(),
                        )
                    )
                torch.cuda.set_sync_debug_mode("error")
                try:
                    runtime.on_forward_end(batch)
                finally:
                    torch.cuda.set_sync_debug_mode("default")

        snapshot = runtime.metrics.snapshot()
        self.assertEqual(runtime.forwards, 2)
        self.assertEqual(snapshot["popularity"]["total"]["routes"], 2 * ROWS * TOP_K * 3)
        self.assertEqual(snapshot["affinity"]["total"]["routes"], 2 * ROWS * TOP_K * 2)
        self.assertEqual(
            snapshot["popularity"]["layers"]["1"]["cold_routes"],
            snapshot["popularity"]["layers"]["1"]["routes"],
        )


if __name__ == "__main__":
    unittest.main()
