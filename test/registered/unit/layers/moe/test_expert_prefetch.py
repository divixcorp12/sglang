import unittest
from contextlib import nullcontext
from unittest.mock import patch

from sglang.srt.layers.moe.expert_prefetch import (
    ExpertPrefetchCoordinator,
    SparseNextLayerPolicy,
)


class _Event:
    def __init__(self):
        self.recorded = 0

    def record(self, stream):
        self.recorded += 1

    def wait(self, stream):
        return None


class TestExpertPrefetch(unittest.TestCase):
    def test_policy_uses_affinity_then_popularity_and_excludes_hot_rows(self):
        policy = SparseNextLayerPolicy(max_candidates=2)

        candidates = policy.predict(
            routed_experts=(1, 3),
            popularity={2: 5.0, 4: 4.0, 5: 1.0},
            affinity={(1, 4): 8.0, (3, 2): 3.0, (3, 5): 9.0},
            resident_experts={5},
        )

        self.assertEqual(candidates, (4, 2))

    def test_policy_fills_remaining_capacity_from_popularity_after_affinity(self):
        policy = SparseNextLayerPolicy(max_candidates=3)

        candidates = policy.predict(
            routed_experts=(1,),
            popularity={2: 30.0, 3: 20.0, 4: 10.0},
            affinity={(1, 4): 1.0, (1, 3): 2.0},
            resident_experts=set(),
        )

        self.assertEqual(candidates, (3, 4, 2))

    def test_disabled_coordinator_never_submits_speculative_transfer(self):
        coordinator = ExpertPrefetchCoordinator(enabled=False)
        submitted = []

        launched = coordinator.launch(
            (4, 2), protected_slots={1}, submit=submitted.extend
        )

        self.assertFalse(launched)
        self.assertEqual(submitted, [])
        self.assertEqual(coordinator.snapshot_stats()["predicted_experts"], 0)

    def test_failed_submission_releases_protected_slots(self):
        coordinator = ExpertPrefetchCoordinator(
            enabled=True, copy_stream=object(), ready_event=_Event()
        )

        with (
            patch("torch.cuda.stream", return_value=nullcontext()),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            coordinator.launch(
                (4,),
                protected_slots={1},
                submit=lambda _: (_ for _ in ()).throw(RuntimeError("copy failed")),
            )

        self.assertEqual(coordinator.protected_slots, frozenset())

    def test_coordinator_records_event_protects_slots_and_corrects_sync(self):
        event = _Event()
        coordinator = ExpertPrefetchCoordinator(enabled=True, ready_event=event)
        submitted = []
        corrected = []

        with patch("torch.cuda.stream", return_value=nullcontext()):
            self.assertTrue(
                coordinator.launch(
                    (4, 2),
                    protected_slots={1, 3},
                    submit=lambda experts: submitted.extend(experts),
                )
            )
        coordinator.synchronous_correction((2, 6), corrected.extend)

        self.assertEqual(submitted, [4, 2])
        self.assertEqual(event.recorded, 1)
        self.assertEqual(coordinator.protected_slots, frozenset())
        self.assertEqual(corrected, [2, 6])
        stats = coordinator.snapshot_stats()
        self.assertEqual((stats["useful_experts"], stats["wasted_experts"]), (1, 1))
        self.assertEqual(stats["synchronous_corrections"], 1)
        self.assertEqual(stats["event_wait_enqueues"], 1)
        self.assertNotIn("exposed_wait_ns", stats)


if __name__ == "__main__":
    unittest.main()
