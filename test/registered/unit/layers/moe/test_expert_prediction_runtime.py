import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.runtime import (
    ExpertPredictionRuntime,
    layer_pairs,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FakeTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=top_k, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), self.topk_config.top_k, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class TupleTopK(FakeTopK):
    def forward(self, hidden_states, router_logits):
        output = super().forward(hidden_states, router_logits)
        return (output.topk_weights, output.topk_ids)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size, num_fused_shared_experts=0):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_fused_shared_experts = num_fused_shared_experts

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(6, 8, bias=False)
        self.topk = FakeTopK(2)
        self.experts = FakeMoE(layer_id, 8, 6)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self, layer_ids=(0, 1, 2)):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in layer_ids)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _batch(mode, rows):
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        batch_size=rows,
        spec_info=None,
        extend_num_tokens=rows,
    )


def _runtime(
    *,
    model=None,
    metrics_path=None,
    hot_caches=None,
    max_rows=4,
    log_interval=2,
    score_interval=1,
):
    model = model or FakeModel()
    runtime = ExpertPredictionRuntime.build(
        model=model,
        predictor_names=("popularity", "affinity"),
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        max_rows=max_rows,
        max_candidates=4,
        hot_caches=hot_caches or {},
        log_interval=log_interval,
        metrics_path=metrics_path,
        score_interval=score_interval,
        topk_type=FakeTopK,
        experts_type=FakeMoE,
    )
    return model, runtime


def _decode(model, runtime, rows=3, mode=ForwardMode.DECODE):
    with torch.no_grad():
        model(torch.randn(rows, 6))
    runtime.on_forward_end(_batch(mode, rows))


class TestExpertPredictionRuntime(unittest.TestCase):
    def test_layer_pairs_by_offset(self):
        self.assertEqual(layer_pairs((0, 2, 5), 0), ((0, 0), (2, 2), (5, 5)))
        self.assertEqual(layer_pairs((0, 2, 5), 1), ((0, 2), (2, 5)))

    def test_decode_forwards_score_every_predictor_and_write_metrics(self):
        path = Path(tempfile.mkdtemp()) / "prediction.jsonl"
        model, runtime = _runtime(metrics_path=path)
        for _ in range(2):
            _decode(model, runtime)
        self.assertEqual(runtime.forwards, 2)
        (line,) = path.read_text().splitlines()
        record = json.loads(line)
        self.assertEqual(record["forwards"], 2)
        popularity = record["predictors"]["popularity"]
        affinity = record["predictors"]["affinity"]
        self.assertEqual(set(popularity["layers"]), {"0", "1", "2"})
        self.assertEqual(popularity["total"]["routes"], 2 * 3 * 2 * 3)
        self.assertEqual(set(affinity["layers"]), {"1", "2"})
        self.assertEqual(affinity["total"]["routes"], 2 * 3 * 2 * 2)
        for predictor in (popularity, affinity):
            self.assertTrue(0.0 <= predictor["total"]["recall_at_m"] <= 1.0)

    def test_verify_forwards_are_scored(self):
        model, runtime = _runtime()
        _decode(model, runtime, mode=ForwardMode.TARGET_VERIFY)
        self.assertEqual(runtime.forwards, 1)

    def test_prefill_idle_and_oversized_forwards_are_not_scored(self):
        model, runtime = _runtime(max_rows=4)
        _decode(model, runtime, mode=ForwardMode.EXTEND)
        runtime.on_forward_end(_batch(ForwardMode.IDLE, 0))
        _decode(model, runtime, rows=5)
        self.assertEqual(runtime.forwards, 0)

    def test_residency_mask_splits_cold_routes(self):
        all_resident = SimpleNamespace(expert_to_slot=torch.arange(8))
        model, runtime = _runtime(hot_caches={1: all_resident})
        _decode(model, runtime)
        layers = runtime.metrics.snapshot()["popularity"]["layers"]
        self.assertEqual(layers["1"]["cold_routes"], 0)
        self.assertEqual(layers["2"]["cold_routes"], layers["2"]["routes"])

    def test_unsupported_topk_disables_scoring(self):
        model = FakeModel()
        model.layers[0].topk = TupleTopK(2)
        model, runtime = _runtime(model=model)
        with self.assertLogs("sglang.srt.layers.moe.expert_prediction", level="WARNING"):
            _decode(model, runtime)
        self.assertEqual(runtime.forwards, 0)

    def test_close_removes_hooks(self):
        model, runtime = _runtime()
        _decode(model, runtime)
        before = runtime.store.view(0, RouteFeature.TOPK_IDS, 3).clone()
        runtime.close()
        with torch.no_grad():
            model(torch.randn(3, 6) * 100)
        self.assertTrue(torch.equal(runtime.store.view(0, RouteFeature.TOPK_IDS, 3), before))

    def test_from_env_rejects_multi_gpu_and_unsized_buffers(self):
        common = dict(
            model=FakeModel(),
            gpu_id=0,
            hidden_dtype=torch.float32,
            tokens_per_request=1,
            moe_ep_size=1,
            attn_dp_size=None,
            pp_size=1,
            expert_hot_cache_manager=None,
        )
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override("popularity"):
            with self.assertRaisesRegex(ValueError, "single GPU"):
                ExpertPredictionRuntime.from_env(decode_max_bs=1, tp_size=2, **common)
            with self.assertRaisesRegex(ValueError, "MAX_ROWS"):
                ExpertPredictionRuntime.from_env(decode_max_bs=0, tp_size=1, **common)
            with self.assertRaisesRegex(ValueError, "single GPU"):
                ExpertPredictionRuntime.from_env(
                    decode_max_bs=1, tp_size=1, **{**common, "pp_size": 2}
                )

    def test_score_interval_scores_every_nth_eligible_forward(self):
        model, runtime = _runtime(score_interval=3)
        for _ in range(6):
            _decode(model, runtime)
        self.assertEqual(runtime.forwards, 2)
        self.assertEqual(runtime.eligible_forwards, 6)

    def test_build_rejects_non_positive_score_interval(self):
        with self.assertRaisesRegex(
            ValueError, "SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL must be positive"
        ):
            _runtime(score_interval=0)

    def test_metrics_write_failure_disables_further_writes_without_raising(self):
        path = Path(tempfile.mkdtemp()) / "missing" / "m.jsonl"
        model, runtime = _runtime(metrics_path=path, log_interval=2)
        with self.assertLogs("sglang.srt.layers.moe.expert_prediction", "WARNING"):
            for _ in range(2):
                _decode(model, runtime)
        self.assertEqual(runtime.forwards, 2)

    def test_from_env_rejects_misconfigured_prefetch(self):
        common = dict(
            model=FakeModel(),
            gpu_id=0,
            hidden_dtype=torch.float32,
            decode_max_bs=1,
            tp_size=1,
            moe_ep_size=1,
            attn_dp_size=None,
            pp_size=1,
        )
        # envs.override()'s restore-on-exit runs only after the `with` body exits
        # normally, so assertRaisesRegex (which swallows the raise) must be the
        # inner context and override the outer one -- otherwise the env var leaks
        # into the next block.
        with envs.SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR.override("llapor"):
            with envs.SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR.override("/tmp/does-not-matter"):
                with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR"):
                    ExpertPredictionRuntime.from_env(
                        tokens_per_request=1, expert_hot_cache_manager=None, **common
                    )
                with envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.override("/tmp/capture"):
                    with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR"):
                        ExpertPredictionRuntime.from_env(
                            tokens_per_request=1,
                            expert_hot_cache_manager=SimpleNamespace(caches={}),
                            max_prefill_rows=1,
                            **common,
                        )
                with envs.SGLANG_MOE_PREFETCH_MAX_CANDIDATES.override(4):
                    with self.assertRaisesRegex(ValueError, "SGLANG_MOE_PREFETCH_MAX_CANDIDATES"):
                        ExpertPredictionRuntime.from_env(
                            tokens_per_request=1,
                            expert_hot_cache_manager=SimpleNamespace(caches={}),
                            **common,
                        )
            with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR"):
                ExpertPredictionRuntime.from_env(
                    tokens_per_request=1,
                    expert_hot_cache_manager=SimpleNamespace(caches={}),
                    **common,
                )


if __name__ == "__main__":
    unittest.main()
