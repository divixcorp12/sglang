"""Equivalence of the batched residency update with the per-layer reference.

``_ReferencePolicy`` is a frozen copy of ``ExpertResidencyPolicy``'s decision
algorithm at d0a3e55b16, before the boundary update was batched across layers.
Every batched decision must match it exactly, tie order included.
"""

import math
import random
import unittest
from operator import index

import torch

from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyPolicy,
    ResidencyDecision,
    advance_residency_policies,
    decide_residency_policies,
)


class _ReferencePolicy:
    """Frozen d0a3e55b16 ``advance``/``decide`` over host scores."""

    def __init__(self, num_experts, capacity, *, decay, promotion_margin, decay_tokens, promotion_sigmas, initial_scores):
        self.num_experts = num_experts
        self.capacity = capacity
        self.decay = float(decay)
        self.promotion_margin = float(promotion_margin)
        self.decay_tokens = decay_tokens
        self.promotion_sigmas = float(promotion_sigmas)
        self._pending_counts = torch.zeros(num_experts, dtype=torch.float32)
        self._scores = torch.as_tensor(initial_scores, dtype=torch.float32).clone()

    def advance(self, tokens=None):
        decay = self.decay
        if self.decay_tokens is not None and tokens is not None:
            decay = self.decay ** (index(tokens) / self.decay_tokens)
        self._scores.mul_(decay).add_(self._pending_counts)
        self._pending_counts.zero_()

    def decide(self, resident_experts):
        score_values = self._scores.detach().cpu().tolist()
        observed_resident = self._deduplicate_and_validate(resident_experts)
        resident = self._limited_resident(observed_resident, score_values)
        ranked = tuple(
            expert_id
            for expert_id, _ in sorted(
                enumerate(score_values), key=lambda item: (-item[1], item[0])
            )
        )
        desired = self._select_desired(ranked, score_values, resident)
        observed_resident_set = set(observed_resident)
        promotions = tuple(
            expert_id for expert_id in desired if expert_id not in observed_resident_set
        )
        evictions = tuple(
            expert_id for expert_id in observed_resident if expert_id not in desired
        )
        return ResidencyDecision(
            desired_experts=desired,
            promotions=promotions,
            evictions=evictions,
            ranked_experts=ranked,
        )

    def _select_desired(self, ranked, score_values, resident):
        if self.capacity == 0:
            return ()
        desired = list(resident[: self.capacity])
        resident_set = set(desired)
        for expert_id in ranked:
            if len(desired) >= self.capacity:
                break
            if score_values[expert_id] <= 0.0:
                break
            if expert_id not in resident_set:
                desired.append(expert_id)
                resident_set.add(expert_id)
        for candidate in ranked:
            if candidate in resident_set:
                continue
            if score_values[candidate] <= 0.0 or not desired:
                break
            victim = min(
                desired, key=lambda expert_id: (score_values[expert_id], -expert_id)
            )
            if score_values[candidate] <= score_values[victim] + self._promotion_threshold(
                score_values[candidate], score_values[victim]
            ):
                break
            desired.remove(victim)
            resident_set.remove(victim)
            desired.append(candidate)
            resident_set.add(candidate)
        return tuple(
            sorted(desired, key=lambda expert_id: (-score_values[expert_id], expert_id))
        )

    def _promotion_threshold(self, candidate, victim):
        return self.promotion_margin + self.promotion_sigmas * math.sqrt(
            max(candidate + victim, 0.0)
        )

    def _limited_resident(self, resident, score_values):
        if len(resident) <= self.capacity:
            return resident
        return tuple(
            sorted(resident, key=lambda expert_id: (-score_values[expert_id], expert_id))[
                : self.capacity
            ]
        )

    def _deduplicate_and_validate(self, expert_ids):
        result = []
        seen = set()
        for value in expert_ids:
            expert_id = index(value)
            if not 0 <= expert_id < self.num_experts:
                raise ValueError("expert ID is outside the policy range")
            if expert_id not in seen:
                result.append(expert_id)
                seen.add(expert_id)
        return tuple(result)


def assert_decisions_equal(test, actual, expected, context=""):
    """Fail unless two decisions agree field by field, order included."""
    for field in ("desired_experts", "promotions", "evictions", "ranked_experts"):
        test.assertEqual(
            tuple(getattr(actual, field)),
            tuple(getattr(expected, field)),
            f"{context} {field}",
        )


def _tied_scores(generator, num_experts, levels):
    """Scores drawn from a few integer levels so many experts tie exactly."""
    return [float(generator.randrange(levels)) * 0.5 for _ in range(num_experts)]


class _Layers:
    """Paired batched and reference policies with the same configuration."""

    def __init__(self, generator, device, layers, num_experts, capacities, **options):
        self.generator = generator
        self.num_experts = num_experts
        self.batched = []
        self.reference = []
        self.residents = []
        for layer in range(layers):
            scores = _tied_scores(generator, num_experts, 6)
            capacity = capacities[layer]
            self.batched.append(
                ExpertResidencyPolicy(
                    num_experts,
                    capacity,
                    device=torch.zeros(0, device=device).device,
                    initial_scores=scores,
                    **options,
                )
            )
            self.reference.append(
                _ReferencePolicy(
                    num_experts,
                    capacity,
                    decay=options.get("decay", 0.95),
                    promotion_margin=options.get("promotion_margin", 0.0),
                    decay_tokens=options.get("decay_tokens"),
                    promotion_sigmas=options.get("promotion_sigmas", 0.0),
                    initial_scores=scores,
                )
            )
            resident = generator.sample(range(num_experts), generator.randrange(capacity + 1))
            self.residents.append(frozenset(resident))

    def record(self, levels):
        for batched, reference in zip(self.batched, self.reference):
            counts = torch.tensor(
                [float(self.generator.randrange(levels)) for _ in range(self.num_experts)]
            )
            batched.record_counts(counts.to(batched.device))
            reference._pending_counts.add_(counts)

    def boundary(self, test, tokens, context):
        advance_residency_policies(self.batched, tokens)
        for reference in self.reference:
            reference.advance(tokens)
        decisions = decide_residency_policies(self.batched, self.residents)
        for layer, (decision, reference) in enumerate(zip(decisions, self.reference)):
            expected = reference.decide(self.residents[layer])
            assert_decisions_equal(test, decision, expected, f"{context} layer {layer}")
            test.assertTrue(
                torch.equal(self.batched[layer]._scores.cpu(), reference._scores),
                f"{context} layer {layer} scores",
            )
            self.residents[layer] = frozenset(expected.desired_experts)
        return decisions


class TestBatchedResidencyDecisions(unittest.TestCase):
    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_randomized_tied_scores_match_reference_over_consecutive_boundaries(self):
        for device in self.devices:
            for seed, options in enumerate(
                (
                    dict(promotion_margin=0.0),
                    dict(promotion_margin=2.0, decay_tokens=1),
                    dict(promotion_margin=0.5, promotion_sigmas=1.0, decay_tokens=4),
                    dict(promotion_margin=0.0, decay=1.0),
                )
            ):
                generator = random.Random(seed)
                capacities = [generator.randrange(0, 21) for _ in range(12)]
                layers = _Layers(generator, device, 12, 24, capacities, **options)
                promotions = 0
                for step in range(12):
                    layers.record(levels=4)
                    decisions = layers.boundary(
                        self, tokens=generator.randrange(1, 9), context=f"{device} {seed} {step}"
                    )
                    promotions += sum(len(d.promotions) for d in decisions)
                self.assertGreater(promotions, 0)

    def test_capacity_edges_match_reference(self):
        for device in self.devices:
            generator = random.Random(7)
            num_experts = 16
            capacities = [0, 1, num_experts, num_experts - 1, 8, 8]
            layers = _Layers(generator, device, len(capacities), num_experts, capacities, promotion_margin=0.0)
            layers.residents[4] = frozenset()
            for step in range(4):
                layers.record(levels=3)
                layers.boundary(self, tokens=None, context=f"{device} edge {step}")

    def test_all_resident_scores_zero_changes_every_slot(self):
        for device in self.devices:
            num_experts, capacity = 12, 4
            scores = [0.0] * 4 + [5.0, 5.0, 4.0, 3.0] + [1.0] * 4
            batched = ExpertResidencyPolicy(num_experts, capacity, device=device, decay=1.0, initial_scores=scores)
            reference = _ReferencePolicy(
                num_experts, capacity, decay=1.0, promotion_margin=0.0,
                decay_tokens=None, promotion_sigmas=0.0, initial_scores=scores,
            )
            residents = [frozenset(range(4))]
            (decision,) = decide_residency_policies([batched], residents)
            expected = reference.decide(residents[0])
            assert_decisions_equal(self, decision, expected, device)
            self.assertEqual(len(decision.promotions), capacity)
            self.assertEqual(set(decision.evictions), set(range(4)))

    def test_no_change_and_oversized_residents_match_reference(self):
        for device in self.devices:
            num_experts, capacity = 10, 3
            scores = [9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            batched = ExpertResidencyPolicy(num_experts, capacity, device=device, promotion_margin=1.0, initial_scores=scores)
            reference = _ReferencePolicy(
                num_experts, capacity, decay=0.95, promotion_margin=1.0,
                decay_tokens=None, promotion_sigmas=0.0, initial_scores=scores,
            )
            for residents in ((0, 1, 2), (5, 4, 3, 2, 1, 0), (3, 3, 4), ()):
                (decision,) = decide_residency_policies([batched], [residents])
                assert_decisions_equal(self, decision, reference.decide(residents), f"{device} {residents}")
            (decision,) = decide_residency_policies([batched], [(0, 1, 2)])
            self.assertEqual((decision.promotions, decision.evictions), ((), ()))

    def test_single_policy_decide_matches_reference(self):
        for device in self.devices:
            generator = random.Random(11)
            layers = _Layers(generator, device, 6, 32, [generator.randrange(33) for _ in range(6)], promotion_margin=1.0)
            for step in range(6):
                layers.record(levels=5)
                for batched, reference in zip(layers.batched, layers.reference):
                    batched.advance(3)
                    reference.advance(3)
                for layer, (batched, reference) in enumerate(zip(layers.batched, layers.reference)):
                    expected = reference.decide(layers.residents[layer])
                    assert_decisions_equal(self, batched.decide(layers.residents[layer]), expected, f"{device} {step} {layer}")
                    layers.residents[layer] = frozenset(expected.desired_experts)

    def test_batched_advance_and_decide_count_metrics_like_the_per_layer_path(self):
        policies = [ExpertResidencyPolicy(8, 2, initial_scores=[float(i) for i in range(8)]) for _ in range(3)]
        advance_residency_policies(policies, None)
        decisions = decide_residency_policies(policies, [(), (7, 6), (0, 1)])
        for policy, decision in zip(policies, decisions):
            metrics = policy.snapshot_metrics()
            self.assertEqual(metrics["boundary_updates"], 1)
            self.assertEqual(metrics["promotions"], len(decision.promotions))
            self.assertEqual(metrics["evictions"], len(decision.evictions))

    def test_decision_equality_helper_fails_on_perturbed_decisions(self):
        scores = [3.0, 3.0, 2.0, 2.0, 1.0, 0.0]
        policy = ExpertResidencyPolicy(6, 2, initial_scores=scores)
        reference = _ReferencePolicy(
            6, 2, decay=0.95, promotion_margin=0.0, decay_tokens=None,
            promotion_sigmas=0.0, initial_scores=scores,
        )
        (decision,) = decide_residency_policies([policy], [(4, 5)])
        expected = reference.decide((4, 5))
        assert_decisions_equal(self, decision, expected)
        ranked = list(expected.ranked_experts)
        ranked[0], ranked[1] = ranked[1], ranked[0]
        perturbations = (
            expected.__class__(expected.desired_experts[::-1], expected.promotions, expected.evictions, expected.ranked_experts),
            expected.__class__(expected.desired_experts, expected.promotions[:-1], expected.evictions, expected.ranked_experts),
            expected.__class__(expected.desired_experts, expected.promotions, expected.evictions[::-1], expected.ranked_experts),
            expected.__class__(expected.desired_experts, expected.promotions, expected.evictions, tuple(ranked)),
        )
        for perturbed in perturbations:
            with self.subTest(perturbed=perturbed), self.assertRaises(AssertionError):
                assert_decisions_equal(self, perturbed, expected)


if __name__ == "__main__":
    unittest.main()
