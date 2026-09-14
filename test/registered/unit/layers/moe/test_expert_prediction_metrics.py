import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.metrics import (
    COUNTER_NAMES,
    ShadowMetrics,
    score_candidates,
)


def _counts(values):
    return dict(zip(COUNTER_NAMES, values.tolist()))


class TestScoreCandidates(unittest.TestCase):
    candidates = torch.tensor([[2, 5, 7, -1], [1, 0, 3, 4]])
    actual = torch.tensor([[5, 1], [4, 6]])

    def test_counts_hits_at_k_and_m_without_residency(self):
        counts = score_candidates(
            candidates=self.candidates, actual=self.actual, top_k=2, num_experts=8, resident=None
        )
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(
            _counts(counts),
            {
                "rows": 2,
                "routes": 4,
                "hits_at_k": 1,
                "hits_at_m": 2,
                "cold_routes": 4,
                "cold_hits_at_m": 2,
                "cold_candidates": 7,
            },
        )

    def test_residency_marks_cold_routes_and_candidates(self):
        resident = torch.zeros(8, dtype=torch.bool)
        resident[[0, 4, 5]] = True
        counts = score_candidates(
            candidates=self.candidates,
            actual=self.actual,
            top_k=2,
            num_experts=8,
            resident=resident,
        )
        self.assertEqual(
            _counts(counts),
            {
                "rows": 2,
                "routes": 4,
                "hits_at_k": 1,
                "hits_at_m": 2,
                "cold_routes": 2,
                "cold_hits_at_m": 0,
                "cold_candidates": 4,
            },
        )

    def test_invalid_actual_ids_are_not_routes(self):
        counts = score_candidates(
            candidates=torch.tensor([[3]]),
            actual=torch.tensor([[-1, 3]]),
            top_k=1,
            num_experts=4,
            resident=None,
        )
        self.assertEqual(_counts(counts)["routes"], 1)
        self.assertEqual(_counts(counts)["hits_at_m"], 1)


class TestShadowMetrics(unittest.TestCase):
    def test_snapshot_aggregates_layers_and_appends_jsonl(self):
        metrics = ShadowMetrics(
            predictor_names=("a", "b"), layer_ids=(0, 1), device=torch.device("cpu")
        )
        for _ in range(2):
            metrics.add(
                predictor_index=0, target_layer=1, counts=torch.tensor([2, 4, 1, 2, 4, 2, 7])
            )
        snapshot = metrics.snapshot()
        self.assertEqual(
            snapshot["a"]["layers"],
            {
                "1": {
                    "rows": 4,
                    "routes": 8,
                    "hits_at_k": 2,
                    "hits_at_m": 4,
                    "cold_routes": 8,
                    "cold_hits_at_m": 4,
                    "cold_candidates": 14,
                }
            },
        )
        total = snapshot["a"]["total"]
        self.assertAlmostEqual(total["recall_at_k"], 0.25)
        self.assertAlmostEqual(total["recall_at_m"], 0.5)
        self.assertAlmostEqual(total["cold_recall_at_m"], 0.5)
        self.assertAlmostEqual(total["cold_precision_at_m"], 4 / 14)
        self.assertEqual(snapshot["b"]["layers"], {})
        self.assertEqual(snapshot["b"]["total"]["routes"], 0)
        self.assertEqual(snapshot["b"]["total"]["recall_at_m"], 0.0)

        path = Path(tempfile.mkdtemp()) / "metrics.jsonl"
        metrics.append_jsonl(path, forwards=7)
        metrics.append_jsonl(path, forwards=9)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["forwards"] for record in records], [7, 9])
        self.assertEqual(records[0]["predictors"], snapshot)
        self.assertIsInstance(records[0]["timestamp_ns"], int)


if __name__ == "__main__":
    unittest.main()
