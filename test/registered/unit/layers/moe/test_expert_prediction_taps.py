import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import RouteTaps, discover_moe_layers
from sglang.srt.layers.moe.topk import StandardTopKOutput


class FakeTopK(nn.Module):
    def __init__(self, top_k, num_fused_shared_experts=0):
        super().__init__()
        self.topk_config = SimpleNamespace(
            top_k=top_k, num_fused_shared_experts=num_fused_shared_experts
        )

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(
            router_logits.float().softmax(dim=-1), self.topk_config.top_k, dim=-1
        )
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class TupleTopK(FakeTopK):
    def forward(self, hidden_states, router_logits):
        output = super().forward(hidden_states, router_logits)
        return (output.topk_weights, output.topk_ids)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(
        self, layer_id, num_experts=8, hidden_size=6, top_k=2, num_fused_shared_experts=0
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.topk = FakeTopK(top_k + num_fused_shared_experts, num_fused_shared_experts)
        self.experts = FakeMoE(layer_id, num_experts + num_fused_shared_experts, hidden_size)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeDecoderLayer(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        self.mlp = FakeBlock(layer_id, **kwargs)

    def forward(self, positions, hidden_states):
        return self.mlp(hidden_states)


class FakeModel(nn.Module):
    def __init__(self, layer_ids=(0, 1, 2), **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(FakeDecoderLayer(i, **kwargs) for i in layer_ids)

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[0])
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


def _discover(model):
    return discover_moe_layers(model, topk_type=FakeTopK, experts_type=FakeMoE)


class TestContracts(unittest.TestCase):
    def test_feature_widths_and_dtypes_follow_layer_spec(self):
        spec = MoeLayerSpec(layer_id=3, num_experts=8, top_k=2, hidden_size=6)
        self.assertEqual(feature_width(RouteFeature.ROUTER_INPUT, spec), 6)
        self.assertEqual(feature_width(RouteFeature.PRE_MIXER, spec), 6)
        self.assertEqual(feature_width(RouteFeature.ROUTER_LOGITS, spec), 8)
        self.assertEqual(feature_width(RouteFeature.TOPK_IDS, spec), 2)
        self.assertEqual(feature_width(RouteFeature.TOPK_WEIGHTS, spec), 2)
        self.assertEqual(feature_dtype(RouteFeature.TOPK_IDS, torch.bfloat16), torch.int64)
        self.assertEqual(
            feature_dtype(RouteFeature.ROUTER_LOGITS, torch.bfloat16), torch.float32
        )
        self.assertEqual(
            feature_dtype(RouteFeature.ROUTER_INPUT, torch.bfloat16), torch.bfloat16
        )

    def test_predictor_env_parses_comma_list_and_empty_is_off(self):
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override("affinity, popularity"):
            self.assertEqual(
                envs.SGLANG_MOE_EXPERT_PREDICTOR.get(), ("affinity", "popularity")
            )
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override(""):
            self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR.get(), ())
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES.get(), 16)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS.get(), 0)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL.get(), 100)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE.get(), "")


class TestFeatureStore(unittest.TestCase):
    def _store(self, max_rows=4):
        spec = MoeLayerSpec(layer_id=0, num_experts=8, top_k=2, hidden_size=6)
        return FeatureStore(
            specs=[spec],
            features=[RouteFeature.TOPK_IDS, RouteFeature.ROUTER_INPUT],
            max_rows=max_rows,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )

    def test_write_keeps_buffer_address_and_casts_ids(self):
        store = self._store()
        before = store.view(0, RouteFeature.TOPK_IDS, 4).data_ptr()
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
        view = store.view(0, RouteFeature.TOPK_IDS, 2)
        self.assertEqual(view.data_ptr(), before)
        self.assertEqual(view.dtype, torch.int64)
        self.assertEqual(view.tolist(), [[1, 2], [3, 4]])

    def test_topk_writes_drop_trailing_shared_expert_columns(self):
        store = self._store()
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[1, 2, 8]]))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 1).tolist(), [[1, 2]])

    def test_hidden_width_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "width 12, expected 6"):
            self._store().write(0, RouteFeature.ROUTER_INPUT, torch.zeros(1, 12))

    def test_oversized_empty_and_unstored_writes_are_skipped(self):
        store = self._store(max_rows=2)
        store.write(0, RouteFeature.TOPK_IDS, torch.ones(3, 2, dtype=torch.int64))
        store.write(0, RouteFeature.TOPK_IDS, torch.ones(0, 2, dtype=torch.int64))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 2).tolist(), [[0, 0], [0, 0]])
        store.write(0, RouteFeature.ROUTER_LOGITS, torch.ones(1, 8))
        self.assertFalse(store.holds(0, RouteFeature.ROUTER_LOGITS))
        self.assertTrue(store.holds(0, RouteFeature.TOPK_IDS))
        self.assertEqual(store.nbytes, 2 * 2 * 8 + 2 * 6 * 4)

    def test_rejects_zero_rows(self):
        with self.assertRaisesRegex(ValueError, "at least one row"):
            self._store(max_rows=0)


class TestDiscovery(unittest.TestCase):
    def test_pairs_siblings_sorts_layers_and_excludes_fused_shared_experts(self):
        model = FakeModel(layer_ids=(4, 2), num_fused_shared_experts=1)
        layers = _discover(model)
        self.assertEqual([layer.spec.layer_id for layer in layers], [2, 4])
        self.assertEqual(
            layers[0].spec, MoeLayerSpec(layer_id=2, num_experts=8, top_k=2, hidden_size=6)
        )
        self.assertIs(layers[1].topk, model.layers[0].mlp.topk)
        self.assertIs(layers[1].block, model.layers[0].mlp)

    def test_block_with_two_topk_children_raises(self):
        model = FakeModel(layer_ids=(0,))
        model.layers[0].mlp.extra_topk = FakeTopK(2)
        with self.assertRaisesRegex(ValueError, "2 TopK and 1 FusedMoE"):
            _discover(model)

    def test_duplicate_layer_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "layer_id 0"):
            _discover(FakeModel(layer_ids=(0, 0)))

    def test_model_without_moe_blocks_raises(self):
        with self.assertRaisesRegex(ValueError, "no TopK"):
            _discover(nn.Sequential(nn.Linear(2, 2)))


class TestRouteTaps(unittest.TestCase):
    def test_taps_copy_router_tensors_until_removed(self):
        model = FakeModel(layer_ids=(0, 1))
        layers = _discover(model)
        store = FeatureStore(
            specs=[layer.spec for layer in layers],
            features=[
                RouteFeature.ROUTER_INPUT,
                RouteFeature.ROUTER_LOGITS,
                RouteFeature.TOPK_IDS,
                RouteFeature.TOPK_WEIGHTS,
            ],
            max_rows=4,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        hidden = torch.randn(3, 6)
        with torch.no_grad():
            model(hidden)
            for layer_id, decoder in zip((0, 1), model.layers):
                logits = decoder.mlp.gate(hidden)
                weights, ids = torch.topk(logits.softmax(dim=-1), 2, dim=-1)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.ROUTER_INPUT, 3), hidden)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.ROUTER_LOGITS, 3), logits)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.TOPK_WEIGHTS, 3), weights)
                self.assertEqual(store.view(layer_id, RouteFeature.TOPK_IDS, 3).tolist(), ids.tolist())
            taps.remove()
            model(torch.randn(3, 6))
        torch.testing.assert_close(store.view(0, RouteFeature.ROUTER_INPUT, 3), hidden)
        self.assertEqual(taps.unsupported_layers, set())

    def test_non_standard_topk_output_marks_layer_unsupported(self):
        model = FakeModel(layer_ids=(0, 1))
        model.layers[0].mlp.topk = TupleTopK(2)
        layers = _discover(model)
        store = FeatureStore(
            specs=[layer.spec for layer in layers],
            features=[RouteFeature.TOPK_IDS],
            max_rows=4,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        with torch.no_grad(), self.assertLogs(
            "sglang.srt.layers.moe.expert_prediction.taps", level="WARNING"
        ):
            model(torch.randn(2, 6))
        self.assertEqual(taps.unsupported_layers, {0})


if __name__ == "__main__":
    unittest.main()
