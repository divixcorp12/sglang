"""Prefetch value model: misses counted against residency, hits bounded by the budget, late requests charged."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    FORWARD_INDEX,
    FORWARD_KIND,
    FORWARD_RESIDENCY,
    FORWARD_ROWS,
    ROW_FORWARD,
    ROW_REQUEST,
    CaptureKind,
    feature_key,
)
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.training.dataset import LayerRows, SessionSplits, load_layer_rows


def _load_single_layer_capture(root: Path, *, residency: torch.Tensor) -> LayerRows:
    """One shard: two decode forwards, one row each, layer 0's topk_ids and residency."""
    forwards = residency.shape[0]
    tensors = {
        ROW_FORWARD: torch.arange(forwards, dtype=torch.int64),
        ROW_REQUEST: torch.zeros(forwards, dtype=torch.int32),
        FORWARD_INDEX: torch.arange(forwards, dtype=torch.int64),
        FORWARD_KIND: torch.full((forwards,), int(CaptureKind.DECODE), dtype=torch.uint8),
        FORWARD_ROWS: torch.ones(forwards, dtype=torch.int64),
        FORWARD_RESIDENCY: residency,
        feature_key(0, RouteFeature.TOPK_IDS): torch.zeros((forwards, 1), dtype=torch.int16),
    }
    save_file(tensors, str(root / "shard-000000.safetensors"), metadata={"request_ids": json.dumps(["sess-t0"])})
    manifest = {"shard": "shard-000000.safetensors", "rows": forwards, "forwards": forwards, "first_forward": 0,
                "last_forward": forwards - 1, "bytes": 0}
    (root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    splits = SessionSplits(split_of_session={})
    return load_layer_rows(root, splits, layer_id=0, features=(RouteFeature.TOPK_IDS,), residency_layer=0)


class TestPrefetchPricing(unittest.TestCase):
    def test_budget_hits_skip_residents_and_respect_budget(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import budget_hits

        resident = torch.zeros(1, 8, dtype=torch.bool)
        resident[0, [1, 2]] = True
        scores = torch.tensor([[0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0]])
        topk_ids = torch.tensor([[1, 3, 5, 6]])
        missed, hits = budget_hits(scores, topk_ids, resident, budget=2)
        self.assertEqual((missed.item(), hits.item()), (3, 1))
        missed, hits = budget_hits(scores, topk_ids, resident, budget=3)
        self.assertEqual((missed.item(), hits.item()), (3, 2))

    def test_saving_charges_the_wait_past_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            DOORBELL_COMPLETED_WAIT_MS, DOORBELL_FIXED_MS, DOORBELL_ROW_MS, IN_GRAPH_ROW_MS, doorbell_saving_ms,
        )

        hits = torch.tensor([0, 1, 2])
        on_time = doorbell_saving_ms(hits=hits, budget=1, window_ms=1.0, reaction_ms=0.03)
        torch.testing.assert_close(on_time, hits.double() * IN_GRAPH_ROW_MS - DOORBELL_COMPLETED_WAIT_MS)
        late = doorbell_saving_ms(hits=hits, budget=2, window_ms=0.2, reaction_ms=0.05)
        wait = 0.05 + 2 * DOORBELL_ROW_MS + DOORBELL_FIXED_MS - 0.2
        torch.testing.assert_close(late, hits.double() * IN_GRAPH_ROW_MS - wait - DOORBELL_COMPLETED_WAIT_MS)

    def test_side_stream_lands_whole_rows_inside_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            IN_GRAPH_FIXED_MS, IN_GRAPH_ROW_MS, side_stream_ready_rows,
        )

        row = IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=0.267), 1)
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=3 * row + 1e-9), 3)
        self.assertEqual(side_stream_ready_rows(budget=2, window_ms=3 * row), 2)
        self.assertEqual(side_stream_ready_rows(budget=4, window_ms=0.1), 0)

    def test_oracle_hits_posts_min_of_misses_and_budget(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import oracle_hits

        missed = torch.tensor([0, 1, 3, 7])
        torch.testing.assert_close(oracle_hits(missed, budget=3), torch.tensor([0, 1, 3, 3]))

    def test_prefix_wait_charges_only_through_the_deepest_hit_rank(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            DOORBELL_FIXED_MS, DOORBELL_ROW_MS, offered_ids, prefix_wait_ms,
        )

        resident = torch.zeros(2, 8, dtype=torch.bool)
        resident[:, [4, 5, 6, 7]] = True
        # Row 0: native miss is expert 2, ranked 2nd of 3 offered (the rank-1 false-positive costs nothing).
        # Row 1: native miss (expert 3) isn't among the top-3 offered, so it waits nothing past the base charge.
        scores = torch.tensor([[0.4, 0.9, 0.6, 0.1, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.05, 0, 0, 0, 0]])
        topk_ids = torch.tensor([[2], [3]])
        offered = offered_ids(scores, resident, budget=3)
        self.assertEqual(offered[0].tolist(), [1, 2, 0])  # row 0: expert 2 (the hit) at rank index 1 (2nd)
        self.assertEqual(offered[1].tolist(), [2, 1, 0])  # row 1: expert 3 (the miss) isn't offered
        wait = prefix_wait_ms(offered=offered, topk_ids=topk_ids, resident=resident, window_ms=0.1, reaction_ms=0.0)
        expected_row0 = max(0.0, 2 * DOORBELL_ROW_MS + DOORBELL_FIXED_MS - 0.1)
        torch.testing.assert_close(wait, torch.tensor([expected_row0, 0.0], dtype=torch.float64))

    def test_load_layer_rows_joins_residency_by_forward(self):
        # Shard: forwards 0 (decode) and 1 (decode), one row each, layer 0 residency differs per forward.
        residency = torch.full((2, 1, 8), -1, dtype=torch.int16)
        residency[0, 0, 3] = 0
        residency[1, 0, 5] = 0
        with tempfile.TemporaryDirectory() as tmp:
            rows = _load_single_layer_capture(Path(tmp), residency=residency)
        self.assertEqual(rows.resident[0].nonzero().flatten().tolist(), [3])
        self.assertEqual(rows.resident[1].nonzero().flatten().tolist(), [5])


if __name__ == "__main__":
    unittest.main()
