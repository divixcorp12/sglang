import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.capture import CaptureSettings
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FakeTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=2, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), 2, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class FakeMoE(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = 8
        self.hidden_size = 6
        self.num_fused_shared_experts = 0

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(6, 8, bias=False)
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


def _runtime(directory, *, capacity=8, capture=True):
    model = FakeModel()
    runtime = ExpertPredictionRuntime.build(
        model=model,
        predictor_names=(),
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        max_rows=1,
        max_candidates=4,
        hot_caches={},
        log_interval=100,
        metrics_path=None,
        score_interval=1,
        topk_type=FakeTopK,
        experts_type=FakeMoE,
        capture=(
            CaptureSettings(
                directory=directory, capacity=capacity, frames=2,
                shard_rows=1000, max_bytes=1 << 30,
            )
            if capture
            else None
        ),
    )
    return model, runtime


def _forward(model, runtime, *, mode, rid, positions, token_ids):
    rows = len(positions)
    hidden = torch.randn(rows, 6)
    with torch.no_grad():
        model(hidden)
    runtime.on_forward_end(
        SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.tensor(token_ids),
            positions=torch.tensor(positions),
            batch_size=1,
            spec_info=None,
            extend_num_tokens=rows,
            rids=[rid],
            extend_seq_lens_cpu=[rows] if mode is ForwardMode.EXTEND else None,
        )
    )
    return hidden


class TestRouteCapture(unittest.TestCase):
    def test_records_spilled_prefill_and_buffered_decode_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory)
            prefill = _forward(model, runtime, mode=ForwardMode.EXTEND, rid="r",
                               positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            decode = _forward(model, runtime, mode=ForwardMode.DECODE, rid="r",
                              positions=[5], token_ids=[6])
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.prefill_rows, report.decode_rows), (6, 5, 1))
            self.assertEqual((report.violations, report.stopped_reason), ([], None))
            tensors = load_shard(directory, read_manifest(directory)[0]["shard"]).tensors
            expected = torch.cat([prefill, decode])
            torch.testing.assert_close(tensors["layer.0.router_input"], expected)
            torch.testing.assert_close(tensors["layer.2.pre_mixer"], expected)
            block = model.layers[1]
            expected_ids = block.topk(expected, block.gate(expected)).topk_ids
            self.assertEqual(tensors["layer.1.topk_ids"].tolist(), expected_ids.tolist())
            self.assertEqual(tensors["row.token_id"].tolist(), [1, 2, 3, 4, 5, 6])

    def test_second_turn_prefill_drops_the_seen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory)
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="turn1",
                     positions=[0, 1, 2, 3], token_ids=[1, 2, 3, 4])
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="turn2",
                     positions=[0, 1, 2, 3, 4, 5], token_ids=[1, 2, 3, 4, 9, 9])
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.ran_rows, report.requests), (6, 10, 2))

    def test_oversized_forward_stops_capture_without_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory, capacity=4)
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="r",
                     positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            for step in range(3):
                _forward(model, runtime, mode=ForwardMode.DECODE, rid="r",
                         positions=[5 + step], token_ids=[6])
            runtime.close()
            report = check_capture(directory)
            self.assertIn("exceeds capture capacity", report.stopped_reason)
            self.assertEqual(report.rows, 0)

    def test_capture_off_leaves_store_without_spill(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runtime = _runtime(Path(tmp) / "capture", capture=False)
            self.assertIsNone(runtime.capture)
            self.assertIsNone(runtime.store.spill)

    def test_from_env_rejects_unsupported_capture_launches(self):
        common = dict(
            model=FakeModel(), gpu_id=0, hidden_dtype=torch.float32, decode_max_bs=1,
            tp_size=1, moe_ep_size=1, attn_dp_size=None, pp_size=1,
            expert_hot_cache_manager=None,
        )
        with tempfile.TemporaryDirectory() as tmp, \
                envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.override(str(Path(tmp) / "c")):
            with self.assertRaisesRegex(ValueError, "speculative"):
                ExpertPredictionRuntime.from_env(
                    tokens_per_request=2, max_prefill_rows=4096, **common
                )
            with self.assertRaisesRegex(ValueError, "chunked-prefill-size"):
                ExpertPredictionRuntime.from_env(
                    tokens_per_request=1, max_prefill_rows=-1, **common
                )


if __name__ == "__main__":
    unittest.main()
