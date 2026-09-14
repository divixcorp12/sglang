import unittest

import torch

from sglang.srt.layers.moe.expert_prediction import registry
from sglang.srt.layers.moe.expert_prediction.affinity import AffinityPredictor
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.popularity import PopularityPredictor

CPU = torch.device("cpu")


def _specs(num_layers, num_experts=4, top_k=1):
    return [
        MoeLayerSpec(layer_id=i, num_experts=num_experts, top_k=top_k, hidden_size=3)
        for i in range(num_layers)
    ]


def _store(specs, rows=2):
    return FeatureStore(
        specs=specs,
        features=[RouteFeature.TOPK_IDS],
        max_rows=rows,
        device=CPU,
        hidden_dtype=torch.float32,
    )


class ConstantPredictor(ExpertPredictor):
    name = "test-constant"
    target_offset = 0
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def predict(self, *, source_layer, target_layer, store, rows):
        return torch.zeros((rows, self.max_candidates), dtype=torch.int64)


class TestBaseAndRegistry(unittest.TestCase):
    def test_pad_candidates_pads_with_minus_one_and_truncates(self):
        ranked = torch.tensor([[3, 1], [2, 0]])
        self.assertEqual(pad_candidates(ranked, 4).tolist(), [[3, 1, -1, -1], [2, 0, -1, -1]])
        self.assertEqual(pad_candidates(ranked, 1).tolist(), [[3], [2]])

    def test_base_rejects_non_positive_candidates(self):
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            ConstantPredictor(specs=_specs(1), device=CPU, max_candidates=0)

    def test_builds_builtins_in_requested_order(self):
        self.assertIn("affinity", registry.registered_predictor_names())
        self.assertIn("popularity", registry.registered_predictor_names())
        built = registry.build_predictors(
            ("popularity", "affinity"), specs=_specs(2), device=CPU, max_candidates=3
        )
        self.assertIsInstance(built[0], PopularityPredictor)
        self.assertIsInstance(built[1], AffinityPredictor)
        self.assertEqual(built[1].max_candidates, 3)

    def test_rejects_unknown_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "unknown expert predictors \\['nope'\\]"):
            registry.build_predictors(("nope",), specs=_specs(1), device=CPU, max_candidates=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.build_predictors(
                ("affinity", "affinity"), specs=_specs(2), device=CPU, max_candidates=2
            )

    def test_register_custom_predictor_and_reject_name_clash(self):
        self.addCleanup(registry._PREDICTORS.pop, "test-constant", None)
        self.assertIs(registry.register_predictor(ConstantPredictor), ConstantPredictor)
        (built,) = registry.build_predictors(
            ("test-constant",), specs=_specs(1), device=CPU, max_candidates=2
        )
        self.assertEqual(built.layer_ids, (0,))
        with self.assertRaisesRegex(ValueError, "test-constant"):
            registry.register_predictor(ConstantPredictor)


class TestPopularityPredictor(unittest.TestCase):
    def test_ranks_decayed_counts_ignores_invalid_ids_and_pads(self):
        specs = _specs(1, num_experts=4, top_k=2)
        store = _store(specs)
        predictor = PopularityPredictor(specs=specs, device=CPU, max_candidates=6)
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[3, 1], [3, -1]]))
        predictor.observe(store=store, rows=2)
        candidates = predictor.predict(source_layer=0, target_layer=0, store=store, rows=2)
        self.assertEqual(candidates.shape, (2, 6))
        self.assertEqual(candidates.dtype, torch.int64)
        self.assertEqual(candidates[:, :2].tolist(), [[3, 1], [3, 1]])
        self.assertEqual(candidates[:, 4:].tolist(), [[-1, -1], [-1, -1]])
        self.assertEqual(predictor.target_offset, 0)
        self.assertEqual(predictor.state_nbytes, 4 * 4)


class TestAffinityPredictor(unittest.TestCase):
    def _trained(self):
        specs = _specs(2, num_experts=4, top_k=1)
        store = _store(specs, rows=1)
        predictor = AffinityPredictor(specs=specs, device=CPU, max_candidates=2)
        for source_id, target_id in ((1, 3), (2, 0)):
            store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[source_id]]))
            store.write(1, RouteFeature.TOPK_IDS, torch.tensor([[target_id]]))
            predictor.observe(store=store, rows=1)
        return predictor, store

    def _first(self, predictor, store, source_id):
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[source_id]]))
        candidates = predictor.predict(source_layer=0, target_layer=1, store=store, rows=1)
        self.assertEqual(candidates.shape, (1, 2))
        return candidates[0, 0].item()

    def test_prefers_cooccurring_targets(self):
        predictor, store = self._trained()
        self.assertEqual(self._first(predictor, store, 1), 3)
        self.assertEqual(self._first(predictor, store, 2), 0)

    def test_falls_back_to_popularity_without_cooccurrence(self):
        predictor, store = self._trained()
        self.assertEqual(self._first(predictor, store, 0), 0)

    def test_metadata_and_state_size(self):
        predictor, _ = self._trained()
        self.assertEqual(predictor.target_offset, 1)
        self.assertEqual(predictor.state_nbytes, 4 * 4 * 4 + 2 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
