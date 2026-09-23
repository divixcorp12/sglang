import unittest

import torch

from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyPolicy,
    TransferPriority,
    decide_residency_policies_from_host_scores,
)


class TestExpertResidencyPolicy(unittest.TestCase):
    def test_host_score_snapshot_uses_current_residents_without_device_readback(self):
        policy = ExpertResidencyPolicy(num_experts=3, capacity=1, decay=1.0)
        policy.record_counts(torch.tensor([1.0, 3.0, 0.0]))
        policy.advance()
        snapshot = policy._scores.clone()
        policy.record_counts(torch.tensor([9.0, 0.0, 0.0]))
        policy.advance()

        decision = decide_residency_policies_from_host_scores(
            [policy], [(0,)], [snapshot]
        )[0]

        self.assertEqual(decision.promotions, (1,))
        self.assertEqual(decision.evictions, (0,))
        self.assertEqual(decision.desired_experts, (1,))
        current = decide_residency_policies_from_host_scores(
            [policy], [(1,)], [snapshot]
        )[0]
        self.assertEqual(current.promotions, ())
        self.assertEqual(current.evictions, ())

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

    def test_decay_follows_routed_tokens_when_decay_tokens_is_set(self):
        for tokens, expected in ((10, (0,)), (20, (1,))):
            policy = ExpertResidencyPolicy(
                num_experts=2, capacity=1, decay=0.5, decay_tokens=10
            )
            policy.record_counts(torch.tensor([8.0, 0.0]))
            first = policy.materialize_boundary(resident_experts=(), tokens=tokens)
            self.assertEqual(first.desired_experts, (0,))
            policy.record_counts(torch.tensor([0.0, 3.0]))

            second = policy.materialize_boundary(
                resident_experts=first.desired_experts, tokens=tokens
            )

            with self.subTest(tokens=tokens):
                self.assertEqual(second.desired_experts, expected)

    def test_noise_scaled_margin_ignores_indistinguishable_leads(self):
        policies = {
            sigmas: ExpertResidencyPolicy(
                num_experts=2, capacity=1, decay=1.0, promotion_sigmas=sigmas
            )
            for sigmas in (0.0, 2.0)
        }
        for sigmas, policy in policies.items():
            policy.record_counts(torch.tensor([16.0, 0.0]))
            resident = policy.materialize_boundary(resident_experts=()).desired_experts
            policy.record_counts(torch.tensor([0.0, 22.0]))

            close = policy.materialize_boundary(resident_experts=resident)

            with self.subTest(sigmas=sigmas):
                self.assertEqual(close.desired_experts, (1,) if sigmas == 0.0 else (0,))

        noisy = policies[2.0]
        noisy.record_counts(torch.tensor([0.0, 8.0]))
        clear = noisy.materialize_boundary(resident_experts=(0,))
        self.assertEqual(clear.desired_experts, (1,))

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
