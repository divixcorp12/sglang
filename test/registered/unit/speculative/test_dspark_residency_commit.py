"""CPU tests for DSpark's residency-clock commit hook.

``DSparkWorkerV2`` cannot be constructed on CPU (it needs a real target
worker/model runner), so this tests the extracted module function
``commit_accept_to_hot_cache`` directly, per the task-7 brief. It mirrors
``EagleWorkerV2.on_verify_complete_cpu``
(python/sglang/srt/speculative/eagle_worker_v2.py:1638-1651): the call count
must be exactly one ``on_speculative_commit`` call per invocation (one per
real verify step), and the reported value must be
``sum(num_correct_drafts_per_req) + len(num_correct_drafts_per_req) *
GenerationBatchResult.num_non_draft_tokens_per_req`` -- i.e. accepted tokens
including each request's bonus token.
"""

import unittest

from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    commit_accept_to_hot_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeHotCacheManager:
    """Records ``on_speculative_commit`` calls without touching the GPU."""

    def __init__(self):
        self.commit_calls: list[int] = []

    def on_speculative_commit(self, accepted_tokens: int) -> None:
        self.commit_calls.append(accepted_tokens)


class TestCommitAcceptedToHotCache(CustomTestCase):
    def test_no_manager_is_a_no_op(self):
        # Must not raise when the model builds no expert streamers.
        commit_accept_to_hot_cache(None, [3, 1, 4])

    def test_single_request_reports_accepted_plus_bonus(self):
        manager = _FakeHotCacheManager()
        commit_accept_to_hot_cache(manager, [3])
        self.assertEqual(
            manager.commit_calls,
            [3 + GenerationBatchResult.num_non_draft_tokens_per_req],
        )

    def test_batch_sums_correct_drafts_and_adds_one_bonus_per_request(self):
        # Mirrors eagle_worker_v2.py:1649-1652 exactly: sum(...) + len(...) *
        # num_non_draft_tokens_per_req (the bonus token count per request).
        manager = _FakeHotCacheManager()
        num_correct_drafts_per_req = [3, 0, 2, 5]
        commit_accept_to_hot_cache(manager, num_correct_drafts_per_req)
        expected = sum(num_correct_drafts_per_req) + len(
            num_correct_drafts_per_req
        ) * GenerationBatchResult.num_non_draft_tokens_per_req
        self.assertEqual(manager.commit_calls, [expected])

    def test_exactly_one_commit_call_per_invocation(self):
        # The residency clock's ResidencyBoundaryClock.commit pops exactly one
        # outstanding provisional verify per call
        # (expert_residency_clock.py:113), so the call COUNT must match the
        # number of real verify steps observed, one call per step -- never
        # zero, never more than one.
        manager = _FakeHotCacheManager()
        for step_accepts in ([1, 2], [0, 0, 3], [4]):
            commit_accept_to_hot_cache(manager, step_accepts)
        self.assertEqual(len(manager.commit_calls), 3)

    def test_empty_batch_still_calls_commit_once_with_zero(self):
        # An edge case that should not crash: zero requests means the sum and
        # per-request bonus are both zero, but the hook still fires (matches
        # the batch-result processor always calling on_verify_complete_cpu
        # once per resolved batch, even a degenerate empty one).
        manager = _FakeHotCacheManager()
        commit_accept_to_hot_cache(manager, [])
        self.assertEqual(manager.commit_calls, [0])


if __name__ == "__main__":
    unittest.main()
