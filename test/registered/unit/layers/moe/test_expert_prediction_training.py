import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.dataset import (
    carve_apex_train_subsets,
    load_session_splits,
)
from sglang.srt.layers.moe.expert_prediction.training.pca import encode, fit_pca, reconstruct


class TestLLaPorLossWeights(unittest.TestCase):
    def test_rarer_expert_gets_more_weight(self):
        # expert 0 selected in 0.5% of rows, expert 1 in 2%: rarer expert must
        # receive strictly larger q under the relative inverse-frequency formula.
        topk_ids = torch.zeros((1000, 1), dtype=torch.int64)
        topk_ids[:5] = 0  # 0.5%
        topk_ids[5:25] = 1  # 2.0%
        topk_ids[25:] = 2
        q = llapor.expert_frequency_weights(topk_ids, num_experts=3)
        self.assertGreater(float(q[0]), float(q[1]))

    def test_clipped_experts_keep_equal_weight_after_renormalization(self):
        # Experts 1 and 2 are rare enough that their raw inverse-frequency ratio
        # would differ a lot; clipping to the same upper bound before
        # normalizing must leave them tied, unlike unclipped inverse frequency.
        topk_ids = torch.zeros((10000, 1), dtype=torch.int64)
        topk_ids[0] = 1  # freq 1e-4 -> raw ratio far above 10
        topk_ids[1] = 2  # freq 1e-4 -> raw ratio far above 10
        q = llapor.expert_frequency_weights(topk_ids, num_experts=3)
        self.assertAlmostEqual(float(q.mean()), 1.0, places=5)
        self.assertAlmostEqual(float(q[1]), float(q[2]), places=5)

    def test_weights_always_normalize_to_mean_one(self):
        topk_ids = torch.randint(0, 5, (200, 1))
        q = llapor.expert_frequency_weights(topk_ids, num_experts=5)
        self.assertAlmostEqual(float(q.mean()), 1.0, places=5)


class TestLLaPorFocalLoss(unittest.TestCase):
    def test_outer_loss_exceeds_middle_loss_when_confidently_wrong(self):
        # A confidently wrong prediction (pt near 0) should be penalized more
        # heavily by the outer group's focal term than by the plain-BCE middle loss.
        logits = torch.tensor([[8.0, -8.0]])
        y = torch.tensor([[0.0, 1.0]])
        q = torch.ones(2)
        outer = llapor.llapor_loss(logits, y, group="outer", q=q)
        middle = llapor.llapor_loss(logits, y, group="middle", q=q)
        self.assertGreater(float(outer), float(middle))

    def test_losses_and_gradients_are_finite_for_always_and_never_selected(self):
        logits = torch.zeros((4, 3), requires_grad=True)
        y = torch.tensor([[1.0, 0.0, 0.0]] * 4)
        q = llapor.expert_frequency_weights(torch.zeros((4, 1), dtype=torch.int64), 3)
        loss = llapor.llapor_loss(logits, y, group="outer", q=q)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestApexOracleDelta(unittest.TestCase):
    def test_oracle_depth_uses_one_based_rank(self):
        logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
        actual = torch.tensor([[0, 3]])
        self.assertEqual(apex.oracle_delta(logits, actual).tolist(), [2])

    def test_top_k_exact_gives_zero_and_last_expert_gives_e_minus_k(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        self.assertEqual(apex.oracle_delta(logits, torch.tensor([[0, 1]])).tolist(), [0])
        self.assertEqual(apex.oracle_delta(logits, torch.tensor([[0, 4]])).tolist(), [3])


class TestApexOrdinalCDF(unittest.TestCase):
    def test_thresholds_are_strictly_increasing(self):
        torch.manual_seed(0)
        cdf = apex.OrdinalCDF(hidden_size=8, num_depths=6)
        cdf.raw_increments.data = torch.randn(5)
        thresholds = cdf.thresholds()
        self.assertEqual(thresholds.numel(), 6)
        self.assertTrue(bool((thresholds.diff() > 0).all()))

    def test_all_cdf_parameters_receive_finite_gradients(self):
        torch.manual_seed(0)
        cdf = apex.OrdinalCDF(hidden_size=4, num_depths=5)
        x = torch.randn(6, 4)
        delta_star = torch.randint(0, 5, (6,))
        loss = apex.cdf_loss(cdf(x), delta_star)
        loss.backward()
        for param in cdf.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


class TestPCAReconstruction(unittest.TestCase):
    def test_full_rank_pca_reconstructs_inputs(self):
        torch.manual_seed(0)
        x = torch.randn(200, 16)
        pca = fit_pca(x, rank=16)
        recon = reconstruct(encode(x, pca), pca)
        self.assertTrue(torch.allclose(recon, x, atol=1e-3))


class TestDatasetSplitJoin(unittest.TestCase):
    def _write_sessions(self, path, records):
        path.write_text("\n".join(json.dumps(r) for r in records))

    def test_split_of_rid_resolves_through_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            self._write_sessions(
                path,
                [
                    {"session_id": "cfq-a-1", "split": "train"},
                    {"session_id": "cfq-b-2", "split": "val"},
                    {"session_id": "cfq-c-3", "split": "holdout"},
                ],
            )
            splits = load_session_splits(path)
            self.assertEqual(splits.split_of_rid("cfq-a-1-t0"), "train")
            self.assertEqual(splits.split_of_rid("cfq-b-2-t12"), "dev")
            self.assertEqual(splits.split_of_rid("cfq-c-3-t1"), "shifted_test")

    def test_no_session_assigned_to_two_splits(self):
        # A join bug that derives split from something other than session_id
        # (e.g. rid or row index) could put two turns of the same session in
        # different named splits or subsets; this must never happen.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            sessions = [
                {"session_id": f"cfq-{i}", "split": ["train", "val", "holdout"][i % 3]}
                for i in range(30)
            ]
            self._write_sessions(path, sessions)
            splits = load_session_splits(path)
            subsets = carve_apex_train_subsets(splits, seed=0)
            for session in sessions:
                sid = session["session_id"]
                named = splits.split_of_session[sid]
                if named == "train":
                    self.assertIn(sid, subsets)
                else:
                    self.assertNotIn(sid, subsets)
            self.assertEqual(len(set(splits.split_of_session.values()) - {"train", "dev", "shifted_test"}), 0)


if __name__ == "__main__":
    unittest.main()
