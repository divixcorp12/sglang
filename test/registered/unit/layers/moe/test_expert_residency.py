import unittest

import torch

from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyPolicy,
    TransferPriority,
)


class TestExpertResidencyPolicy(unittest.TestCase):
    def test_selects_deterministic_budgeted_topk_from_activation_counts(self):
        policy = ExpertResidencyPolicy(num_experts=5, capacity=2, decay=1.0)
        counter_address = policy.pending_counts.data_ptr()
        policy.record_counts(torch.tensor([4, 4, 7, 1, 0]))
        self.assertEqual(policy.pending_counts.data_ptr(), counter_address)

        decision = policy.materialize_boundary(resident_experts=())

        self.assertEqual(decision.desired_experts, (2, 0))
        self.assertEqual(decision.promotions, (2, 0))
        self.assertEqual(decision.evictions, ())

    def test_decay_and_hysteresis_prevent_near_cutoff_churn(self):
        policy = ExpertResidencyPolicy(
            num_experts=3,
            capacity=2,
            decay=0.5,
            promotion_margin=0.25,
        )
        policy.record_counts(torch.tensor([10, 9, 0]))
        first = policy.materialize_boundary(resident_experts=())
        self.assertEqual(first.desired_experts, (0, 1))

        policy.record_counts(torch.tensor([0, 0, 4.7]))
        stable = policy.materialize_boundary(resident_experts=first.desired_experts)
        self.assertEqual(stable.desired_experts, (0, 1))
        self.assertEqual(stable.promotions, ())
        self.assertEqual(stable.evictions, ())

        policy.record_counts(torch.tensor([0, 0, 0.4]))
        promoted = policy.materialize_boundary(resident_experts=stable.desired_experts)
        self.assertEqual(promoted.desired_experts, (2, 0))
        self.assertEqual(promoted.promotions, (2,))
        self.assertEqual(promoted.evictions, (1,))

    def test_exact_demand_transfers_precede_background_promotions(self):
        policy = ExpertResidencyPolicy(num_experts=5, capacity=2)
        policy.record_counts(torch.tensor([0, 2, 3, 0, 0]))
        decision = policy.materialize_boundary(resident_experts=())

        schedule = policy.schedule_transfers(exact_demand=(4, 1, 4), decision=decision)

        self.assertEqual(
            [(item.expert_id, item.priority) for item in schedule],
            [
                (4, TransferPriority.EXACT_DEMAND),
                (1, TransferPriority.EXACT_DEMAND),
                (2, TransferPriority.BACKGROUND_PROMOTION),
            ],
        )
        metrics = policy.snapshot_metrics()
        self.assertEqual(metrics["exact_demand_experts"], 2)
        self.assertEqual(metrics["background_promotion_experts"], 1)


if __name__ == "__main__":
    unittest.main()
