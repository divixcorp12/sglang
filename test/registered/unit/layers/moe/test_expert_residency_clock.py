"""CPU tests for speculative residency boundaries in the expert hot cache."""

import random
import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.srt.layers.moe.expert_residency_clock import (
    ForwardKind,
    ResidencyBoundaryClock,
    classify_forward,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _draft_input():
    return SimpleNamespace(is_draft_input=lambda: True)


def _verify_input(draft_token_num):
    return SimpleNamespace(
        is_draft_input=lambda: False, draft_token_num=draft_token_num
    )


def _batch(mode, *, tokens=0, batch_size=1, spec_info=None):
    return SimpleNamespace(
        forward_mode=mode,
        extend_num_tokens=tokens,
        batch_size=batch_size,
        spec_info=spec_info,
    )


def _draft_decode():
    return _batch(ForwardMode.DECODE, spec_info=_draft_input())


def _draft_extend():
    return _batch(ForwardMode.DRAFT_EXTEND_V2, tokens=4, spec_info=_draft_input())


def _verify(draft_token_num=4):
    return _batch(ForwardMode.TARGET_VERIFY, spec_info=_verify_input(draft_token_num))


def _observe(clock, batch):
    return clock.observe(*classify_forward(batch))


def _clock_state(clock):
    return (
        clock.forwards,
        clock.tokens_since_boundary,
        clock.decode_forwards_since_boundary,
        len(clock._provisional),
    )


class _LegacyBoundaryClock:
    """Boundary accounting of the hot cache observer before speculative awareness."""

    def __init__(self, update_prefill_tokens, update_decode_forwards, dynamic):
        self.update_prefill_tokens = update_prefill_tokens
        self.update_decode_forwards = update_decode_forwards
        self.dynamic = dynamic
        self.forwards = 0
        self.tokens = 0
        self.decode_forwards = 0

    def observe(self, forward_batch):
        self.forwards += 1
        mode = forward_batch.forward_mode
        prefill = mode.is_extend_without_speculative()
        extend_tokens = forward_batch.extend_num_tokens or 0
        self.tokens += extend_tokens if prefill else forward_batch.batch_size
        if not prefill and not mode.is_idle():
            self.decode_forwards += 1
        qualifying = self.dynamic and (
            (prefill and extend_tokens >= self.update_prefill_tokens)
            or (
                not prefill
                and self.update_decode_forwards > 0
                and self.decode_forwards >= self.update_decode_forwards
            )
        )
        if not qualifying:
            return None
        tokens, self.tokens, self.decode_forwards = self.tokens, 0, 0
        return tokens


class TestResidencyBoundaryClock(unittest.TestCase):
    def test_verify_forwards_reach_the_decode_forward_cadence(self):
        clock = ResidencyBoundaryClock(16, 2)

        self.assertIsNone(_observe(clock, _verify()))
        clock.commit(3)
        self.assertIsNotNone(_observe(clock, _verify()))
        self.assertEqual(clock.decode_forwards_since_boundary, 0)

    def test_draft_decode_leaves_every_counter_unchanged(self):
        clock = ResidencyBoundaryClock(16, 1)
        _observe(clock, _verify())
        before = _clock_state(clock)

        self.assertEqual(classify_forward(_draft_decode()), (ForwardKind.DRAFT, 0))
        for _ in range(8):
            self.assertIsNone(_observe(clock, _draft_decode()))

        self.assertEqual(_clock_state(clock), before)

    def test_draft_extend_v2_is_ignored(self):
        clock = ResidencyBoundaryClock(1, 1)
        extend_without_spec_info = _batch(ForwardMode.DRAFT_EXTEND_V2, tokens=40)

        for batch in (_draft_extend(), extend_without_spec_info):
            self.assertEqual(classify_forward(batch), (ForwardKind.DRAFT, 0))
            self.assertIsNone(_observe(clock, batch))

        self.assertEqual(_clock_state(clock), (0, 0, 0, 0))

    def test_draft_prefill_extend_and_draft_idle_are_ignored(self):
        clock = ResidencyBoundaryClock(1, 1)
        draft_prefill = _batch(ForwardMode.EXTEND, tokens=40, spec_info=_draft_input())
        draft_idle = _batch(ForwardMode.IDLE, batch_size=0, spec_info=_draft_input())

        for batch in (draft_prefill, draft_idle):
            self.assertEqual(classify_forward(batch), (ForwardKind.DRAFT, 0))
            self.assertIsNone(_observe(clock, batch))

        self.assertEqual(_clock_state(clock), (0, 0, 0, 0))
        target_prefill = _batch(ForwardMode.EXTEND, tokens=40)
        self.assertEqual(classify_forward(target_prefill), (ForwardKind.PREFILL, 40))

    def test_overlap_ordered_commits_never_advance_by_negative_tokens(self):
        clock = ResidencyBoundaryClock(16, 1)
        verify = _batch(
            ForwardMode.TARGET_VERIFY, batch_size=8, spec_info=_verify_input(4)
        )
        boundaries = [_observe(clock, verify), _observe(clock, verify)]
        clock.commit(8)
        self.assertGreaterEqual(clock.tokens_since_boundary, 0)
        clock.commit(8)
        self.assertGreaterEqual(clock.tokens_since_boundary, 0)
        boundaries.append(_observe(clock, verify))

        self.assertEqual(boundaries[:2], [0, 32])
        self.assertTrue(all(tokens >= 0 for tokens in boundaries), boundaries)

    def test_dropped_uncommitted_verify_warns_once(self):
        clock = ResidencyBoundaryClock(16, 0)
        with self.assertLogs(
            "sglang.srt.layers.moe.expert_residency_clock", level="WARNING"
        ) as logs:
            for _ in range(40):
                _observe(clock, _verify())

        self.assertEqual(len(logs.records), 1)

    def test_manager_ignores_draft_forwards_before_touching_state(self):
        bare = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
        # Every manager polls its trace telemetry first; that is not residency state.
        bare._trace_telemetry = {}
        counts = {"global_physical_count": object()}
        draft_prefill = _batch(ForwardMode.EXTEND, tokens=40, spec_info=_draft_input())

        for batch in (_draft_decode(), _draft_extend(), draft_prefill):
            self.assertIsNone(bare.on_expert_distribution(batch, counts))
        with self.assertRaises(AttributeError):
            bare.on_expert_distribution(_verify(), counts)

    def test_commit_replaces_provisional_verify_tokens_before_the_next_advance(self):
        clock = ResidencyBoundaryClock(16, 1)

        self.assertEqual(_observe(clock, _verify(4)), 0)
        clock.commit(3)
        self.assertEqual(_observe(clock, _verify(4)), 3)

        manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
        manager._boundary_clock = clock
        manager.on_speculative_commit(2)
        self.assertEqual(_observe(clock, _verify(4)), 2)

    def test_worker_commits_accept_lengths_including_non_draft_tokens(self):
        from unittest.mock import patch

        from sglang.srt.managers.scheduler import GenerationBatchResult
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

        commits = []
        worker = SimpleNamespace(
            adaptive_controller=None,
            _target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    expert_hot_cache_manager=SimpleNamespace(
                        on_speculative_commit=commits.append
                    )
                )
            ),
        )
        EAGLEWorkerV2.on_verify_complete_cpu(worker, [2, 0, 3], batch_size=3)
        with patch.object(GenerationBatchResult, "num_non_draft_tokens_per_req", 2):
            EAGLEWorkerV2.on_verify_complete_cpu(worker, [2, 0, 3], batch_size=3)

        self.assertEqual(commits, [8, 11])

    def test_verify_counts_drafted_tokens_per_request(self):
        batch = _batch(
            ForwardMode.TARGET_VERIFY, batch_size=3, spec_info=_verify_input(4)
        )
        self.assertEqual(classify_forward(batch), (ForwardKind.VERIFY, 12))

    def test_minimum_residence_counts_only_target_forwards(self):
        clock = ResidencyBoundaryClock(16, 0)
        for _ in range(5):
            for _ in range(3):
                _observe(clock, _draft_decode())
            _observe(clock, _verify())
            _observe(clock, _draft_extend())
            clock.commit(3)

        self.assertEqual(clock.forwards, 5)

    def test_boundary_cadence_per_committed_token_ignores_num_steps(self):
        schedules = {}
        for num_steps in (2, 3, 5):
            clock = ResidencyBoundaryClock(16, 2)
            boundaries = []
            for cycle in range(12):
                for _ in range(num_steps):
                    self.assertIsNone(_observe(clock, _draft_decode()))
                boundary = _observe(clock, _verify(num_steps + 1))
                if boundary is not None:
                    boundaries.append((cycle, boundary))
                self.assertIsNone(_observe(clock, _draft_extend()))
                clock.commit(3)
            schedules[num_steps] = boundaries
            self.assertEqual(
                sum(tokens for _, tokens in boundaries) + clock.tokens_since_boundary,
                12 * 3,
            )

        expected = [(1, 3)] + [(cycle, 6) for cycle in range(3, 12, 2)]
        self.assertEqual(schedules, {steps: expected for steps in (2, 3, 5)})

    def test_non_speculative_boundaries_match_the_legacy_observer(self):
        for seed in range(40):
            rng = random.Random(seed)
            prefill_tokens = rng.randint(1, 32)
            decode_forwards = rng.choice((0, 1, 2, 3, 8))
            dynamic = rng.random() < 0.8
            clock = ResidencyBoundaryClock(
                prefill_tokens, decode_forwards, enabled=dynamic
            )
            legacy = _LegacyBoundaryClock(prefill_tokens, decode_forwards, dynamic)
            for step in range(300):
                choice = rng.random()
                if choice < 0.15:
                    batch = _batch(ForwardMode.EXTEND, tokens=rng.randint(0, 40))
                elif choice < 0.25:
                    batch = _batch(ForwardMode.IDLE, batch_size=0)
                else:
                    batch = _batch(ForwardMode.DECODE, batch_size=rng.randint(1, 3))
                with self.subTest(seed=seed, step=step):
                    self.assertEqual(_observe(clock, batch), legacy.observe(batch))
                    self.assertEqual(clock.forwards, legacy.forwards)


if __name__ == "__main__":
    unittest.main()
