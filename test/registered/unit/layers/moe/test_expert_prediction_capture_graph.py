import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.capture import CaptureSettings
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

DECODE_ROWS = 2
PREFILL_ROWS = 8
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
        self.num_fused_shared_experts = 0

    def forward(self, hidden_states, topk_output):
        return hidden_states * 1.0


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
        self.layers = nn.ModuleList(FakeBlock(i) for i in range(3))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _batch(mode, positions, token_ids, rids, extend_lens):
    rows = positions.shape[0]
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=token_ids,
        positions=positions,
        batch_size=len(rids),
        spec_info=None,
        extend_num_tokens=rows,
        rids=rids,
        extend_seq_lens_cpu=extend_lens,
    )


class TestCaptureGraph(unittest.TestCase):
    def test_graph_decode_and_eager_prefill_are_captured_without_sync(self):
        device = torch.device("cuda")
        model = FakeModel().to(device=device, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            runtime = ExpertPredictionRuntime.build(
                model=model, predictor_names=(), device=device, hidden_dtype=torch.bfloat16,
                max_rows=DECODE_ROWS, max_candidates=4, hot_caches={}, log_interval=100,
                metrics_path=None, score_interval=1, topk_type=FakeTopK, experts_type=FakeMoE,
                capture=CaptureSettings(
                    directory=directory, capacity=PREFILL_ROWS, frames=2,
                    shard_rows=1000, max_bytes=1 << 30,
                ),
            )
            static_input = torch.zeros(DECODE_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model(static_input)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    model(static_input)

                prefill_input = torch.randn(PREFILL_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
                decode_input = torch.randn(DECODE_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
                prefill_positions = torch.arange(4, device=device).repeat(2)
                decode_positions = torch.tensor([4, 4], device=device)
                # Distinct tokens per request so request b's prefill is not deduplicated.
                prefill_tokens = torch.arange(PREFILL_ROWS, device=device) + 100
                decode_tokens = torch.tensor([200, 201], device=device)
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    model(prefill_input)
                    runtime.on_forward_end(
                        _batch(ForwardMode.EXTEND, prefill_positions, prefill_tokens,
                               ["a", "b"], [4, 4])
                    )
                    static_input.copy_(decode_input)
                    graph.replay()
                    runtime.on_forward_end(
                        _batch(ForwardMode.DECODE, decode_positions, decode_tokens,
                               ["a", "b"], None)
                    )
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                expected_ids = model.layers[0].topk(
                    decode_input, model.layers[0].gate(decode_input)
                ).topk_ids.cpu()
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.violations, report.stopped_reason), ([], None))
            self.assertEqual((report.prefill_rows, report.decode_rows), (PREFILL_ROWS, DECODE_ROWS))
            tensors = load_shard(directory, read_manifest(directory)[0]["shard"]).tensors
            torch.testing.assert_close(
                tensors["layer.0.router_input"], torch.cat([prefill_input, decode_input]).cpu()
            )
            self.assertEqual(
                tensors["layer.0.topk_ids"][PREFILL_ROWS:].tolist(), expected_ids.tolist()
            )


if __name__ == "__main__":
    unittest.main()
