"""Device-only residency decisions against the frozen per-layer reference.

``decide_residency_on_device`` computes promotions and evictions for every
layer with fixed-shape tensor ops and no host readback. Its prefix form (fill
free slots with the best candidates, then swap candidate i against the i-th
worst member while the lead clears the threshold) must reproduce the frozen
d0a3e55b16 ``_select_desired`` exactly, tie order included. The exhaustive
test enumerates every score vector over a tied level set, every resident set
and every capacity of a small layer.
"""

import itertools
import random
import unittest

import torch

from sglang.srt.layers.moe.expert_residency import decide_residency_on_device
from test_expert_residency_batched import _ReferencePolicy


def _reference(scores, resident, capacity, margin, sigmas):
    policy = _ReferencePolicy(
        len(scores), capacity, decay=1.0, promotion_margin=margin,
        decay_tokens=None, promotion_sigmas=sigmas, initial_scores=scores,
    )
    return policy.decide(resident)


def device_rows(scores, residents, capacities, margin, sigmas, max_promotions, device):
    """Run the device decision over rows of scores, resident lists and capacities."""
    num_experts = len(scores[0])
    resident_mask = torch.zeros(len(scores), num_experts, dtype=torch.bool)
    for row, resident in enumerate(residents):
        resident_mask[row, list(resident)] = True
    return decide_residency_on_device(
        torch.tensor(scores, dtype=torch.float32, device=device),
        resident_mask.to(device),
        torch.tensor(capacities, dtype=torch.int64, device=device),
        promotion_margin=margin,
        promotion_sigmas=sigmas,
        max_promotions=max_promotions,
    )


def mismatches(scores, residents, capacities, output, margin, sigmas, max_promotions):
    """Rows where the device decision differs from the reference, prefix-truncated to ``max_promotions``.

    Promotions compare in rank order; evictions compare worst first, the order
    the reference's victim search removes them in.
    """
    promotions = output.promotions.cpu().tolist()
    promotion_counts = output.promotion_counts.cpu().tolist()
    evictions = output.evictions.cpu().tolist()
    eviction_counts = output.eviction_counts.cpu().tolist()
    failed = []
    for row, (score, resident, capacity) in enumerate(zip(scores, residents, capacities)):
        expected = _reference(score, resident, capacity, margin, sigmas)
        worst_first = sorted(expected.evictions, key=lambda expert: (score[expert], -expert))
        kept = min(len(expected.promotions), max_promotions)
        fill = len(expected.promotions) - len(expected.evictions)
        expected_promotions = list(expected.promotions[:kept])
        expected_evictions = worst_first[: max(0, kept - fill)]
        actual_promotions = promotions[row][: promotion_counts[row]]
        actual_evictions = evictions[row][: eviction_counts[row]]
        if actual_promotions != expected_promotions or actual_evictions != expected_evictions:
            failed.append((row, score, sorted(resident), capacity, actual_promotions, expected_promotions, actual_evictions, expected_evictions))
    return failed


class TestDeviceResidencyDecisions(unittest.TestCase):
    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_exhaustive_small_layer_matches_reference_prefix_form(self):
        num_experts = 5
        levels = (0.0, 0.5, 1.0, 2.0)
        rows = []
        for score in itertools.product(levels, repeat=num_experts):
            for capacity in range(num_experts + 1):
                for size in range(capacity + 1):
                    for resident in itertools.combinations(range(num_experts), size):
                        rows.append((list(score), resident, capacity))
        scores, residents, capacities = map(list, zip(*rows))
        for margin, sigmas in ((0.0, 0.0), (0.5, 0.0), (1.5, 0.5), (0.0, 1.0)):
            for device in self.devices:
                output = device_rows(scores, residents, capacities, margin, sigmas, num_experts, device)
                failed = mismatches(scores, residents, capacities, output, margin, sigmas, num_experts)
                with self.subTest(margin=margin, sigmas=sigmas, device=device, rows=len(rows)):
                    self.assertEqual(failed[:3], [])

    def test_randomized_production_sized_layers_match_reference(self):
        generator = random.Random(3)
        num_experts = 512
        for device in self.devices:
            scores, residents, capacities = [], [], []
            for _ in range(48):
                capacity = generator.randrange(80, 130)
                score = [float(generator.randrange(40)) * 0.25 for _ in range(num_experts)]
                resident = generator.sample(range(num_experts), generator.choice((capacity, capacity - generator.randrange(4))))
                scores.append(score)
                residents.append(resident)
                capacities.append(capacity)
            for margin, sigmas in ((2.0, 0.0), (0.0, 0.0), (1.0, 1.5)):
                output = device_rows(scores, residents, capacities, margin, sigmas, num_experts, device)
                self.assertEqual(mismatches(scores, residents, capacities, output, margin, sigmas, num_experts)[:2], [])

    def test_capped_promotions_are_the_reference_prefix(self):
        generator = random.Random(5)
        num_experts = 64
        for device in self.devices:
            scores, residents, capacities = [], [], []
            for _ in range(64):
                capacity = generator.randrange(0, 40)
                scores.append([float(generator.randrange(6)) for _ in range(num_experts)])
                residents.append(generator.sample(range(num_experts), generator.randrange(capacity + 1)))
                capacities.append(capacity)
            for max_promotions in (1, 3, 8):
                output = device_rows(scores, residents, capacities, 0.0, 0.0, max_promotions, device)
                self.assertEqual(tuple(output.promotions.shape), (64, max_promotions))
                self.assertLessEqual(int(output.promotion_counts.max()), max_promotions)
                self.assertEqual(mismatches(scores, residents, capacities, output, 0.0, 0.0, max_promotions)[:2], [])
                needed = output.needed_promotions.cpu().tolist()
                truncated = 0
                for row, (score, resident, capacity) in enumerate(zip(scores, residents, capacities)):
                    uncapped = len(_reference(score, resident, capacity, 0.0, 0.0).promotions)
                    self.assertEqual(min(needed[row], max_promotions + 1), min(uncapped, max_promotions + 1), f"row {row}")
                    self.assertEqual(needed[row] > max_promotions, uncapped > max_promotions, f"row {row}")
                    truncated += uncapped > max_promotions
                self.assertGreater(truncated, 0, f"no truncated rows at cap {max_promotions}")

    def test_inactive_layers_decide_nothing(self):
        scores = [[0.0, 5.0, 4.0, 1.0], [3.0, 0.0, 9.0, 0.0]]
        output = decide_residency_on_device(
            torch.tensor(scores),
            torch.tensor([[True, False, False, False], [False, True, False, False]]),
            torch.tensor([1, 1]),
            promotion_margin=0.0,
            promotion_sigmas=0.0,
            max_promotions=2,
            active=torch.tensor([False, True]),
        )
        self.assertEqual(output.promotion_counts.tolist(), [0, 1])
        self.assertEqual(output.eviction_counts.tolist(), [0, 1])
        self.assertEqual(output.promotions[1, 0].item(), 2)
        self.assertEqual(output.evictions[1, 0].item(), 1)

    def test_decision_runs_without_host_synchronization(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        scores = torch.rand(48, 512, device="cuda") * 10
        resident = torch.rand(48, 512, device="cuda") < 0.2
        capacity = resident.sum(1)
        active = torch.ones(48, dtype=torch.bool, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            decide_residency_on_device(
                scores, resident, capacity, promotion_margin=2.0, promotion_sigmas=0.0,
                max_promotions=16, active=active,
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_mismatch_helper_fails_on_perturbed_device_output(self):
        scores = [[3.0, 3.0, 1.0, 0.0, 2.0], [1.0, 4.0, 4.0, 0.0, 0.0]]
        residents = [(2, 3), (0,)]
        capacities = [2, 2]
        output = device_rows(scores, residents, capacities, 0.0, 0.0, 5, "cpu")
        self.assertEqual(mismatches(scores, residents, capacities, output, 0.0, 0.0, 5), [])
        for field, row, column, value in (
            ("promotions", 0, 0, 1),
            ("evictions", 0, 0, 2),
            ("promotion_counts", 1, None, 0),
            ("eviction_counts", 0, None, 1),
        ):
            perturbed = output._replace(**{field: getattr(output, field).clone()})
            target = getattr(perturbed, field)
            if column is None:
                target[row] = value
            else:
                target[row, column] = value
            with self.subTest(field=field):
                self.assertNotEqual(mismatches(scores, residents, capacities, perturbed, 0.0, 0.0, 5), [])


if __name__ == "__main__":
    unittest.main()
