"""DSpark's confidence-head verify cut: the w* rule, and the startup T(w) calibration that feeds it (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
    CALIBRATION_VERIFIES,
    ConfidenceCut,
    cut_verify_lens,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GAMMA = 5
NAN = float("nan")


class FakeEvent:
    """A CUDA event stand-in: an end event carries the elapsed ms of its verify and completes when the test says."""

    def __init__(self, timings):
        self.timings = timings
        self.ms = None
        self.ready = False

    def record(self):
        self.ms = self.timings.next_ms()

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        return end.ms


class FakeTimings:
    def __init__(self, ms_for_width):
        self.ms_for_width = ms_for_width
        self.width = None
        self.events = []

    def next_ms(self):
        return self.ms_for_width(self.width)

    def new_event(self):
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def complete_all(self):
        for event in self.events:
            event.ready = True


def run_verify(cut, timings, confidence):
    lens = cut.verify_lens(confidence)
    timings.width = int(lens[0])
    cut.begin_verify()
    cut.end_verify()
    return int(lens[0])


def test_the_cut_verifies_every_draft_at_flat_cost_and_only_the_anchor_when_width_is_expensive():
    confident = torch.full((1, GAMMA), 0.95)
    flat = torch.ones(GAMMA + 1)
    assert cut_verify_lens(confidence=confident, cost_ms=flat).tolist() == [GAMMA + 1]
    # E(6) / E(1) is at most 6, so a T(6) / T(1) of 10 makes the anchor alone the best rate.
    steep = torch.tensor([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert cut_verify_lens(confidence=confident, cost_ms=steep).tolist() == [1]


def test_the_cut_picks_the_best_expected_tokens_per_ms_and_ends_the_prefix_at_a_nan():
    confidence = torch.tensor(
        [
            [0.9, 0.5, 0.2, 0.9, 0.9],  # E = 1, 1.9, 2.35, 2.44, ...; at T = 1, 1.2, 1.4, ... w = 3 wins
            [0.9, NAN, 0.99, 0.99, 0.99],  # E stops growing after draft 1: w = 2
            [NAN, 0.99, 0.99, 0.99, 0.99],  # the anchor alone
            [0.0, 0.0, 0.0, 0.0, 0.0],  # a tie at E = 1 goes to the narrowest verify
        ]
    )
    cost = torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    lens = cut_verify_lens(confidence=confidence, cost_ms=cost)
    assert lens.dtype == torch.int64
    assert lens.tolist() == [3, 2, 1, 1]


def calibrate(ms_for_width):
    timings = FakeTimings(ms_for_width)
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    for _ in range(CALIBRATION_VERIFIES):
        run_verify(cut, timings, confident)
    timings.complete_all()
    cut.verify_lens(confident)
    return cut


def test_calibration_forces_widths_round_robin_then_freezes_an_affine_fit_over_every_sample():
    timings = FakeTimings(None)

    def ms_for_width(width):
        # +-3 ms noise, alternating per verify round so each width's noise sums to zero; the record() of a verify's
        # end event is the 2nd of its two events.
        verify_idx = (len(timings.events) - 1) // 2
        noise = 3.0 if (verify_idx // (GAMMA + 1)) % 2 == 0 else -3.0
        return 100.0 + 20.0 * (width - 1) + noise

    timings.ms_for_width = ms_for_width
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    widths = [run_verify(cut, timings, confident) for _ in range(CALIBRATION_VERIFIES)]
    assert widths == [i % (GAMMA + 1) + 1 for i in range(CALIBRATION_VERIFIES)]
    assert not cut.calibrated

    # The timings are still in flight: no host wait, every row verified, no table yet.
    assert run_verify(cut, timings, confident) == GAMMA + 1
    assert not cut.calibrated
    timings.complete_all()
    cut.verify_lens(confident)
    assert cut.calibrated and not cut.disabled
    assert cut.cost_ms() == pytest.approx([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    # Calibrated: the cut, not a forced width, and no further timing. At c = 0.99, E(6) / T(6) = 5.85 / 200 is the
    # best rate; at c = 0.01, E(2) / T(2) = 1.01 / 120 loses to the anchor's 1 / 100.
    num_events = len(timings.events)
    assert run_verify(cut, timings, confident) == GAMMA + 1
    assert run_verify(cut, timings, torch.full((1, GAMMA), 0.01)) == 1
    assert len(timings.events) == num_events


def test_a_negative_fitted_slope_clamps_to_a_flat_table_at_the_mean():
    cut = calibrate(lambda width: 200.0 - 10.0 * (width - 1))
    assert cut.calibrated
    assert cut.cost_ms() == pytest.approx([175.0] * (GAMMA + 1))


def test_calibration_drains_only_completed_timings_in_order():
    timings = FakeTimings(lambda width: float(width))
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    for _ in range(CALIBRATION_VERIFIES):
        run_verify(cut, timings, confident)
    # All but the last end event completed: the table must wait for the last sample.
    for event in timings.events[:-1]:
        event.ready = True
    cut.verify_lens(confident)
    assert not cut.calibrated
    timings.events[-1].ready = True
    cut.verify_lens(confident)
    assert cut.cost_ms() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_a_width_without_a_valid_time_disables_the_cut_and_verifies_every_row():
    timings = FakeTimings(lambda width: NAN if width == 4 else 1.0)
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.1)
    for _ in range(CALIBRATION_VERIFIES):
        run_verify(cut, timings, confident)
    timings.complete_all()
    assert cut.verify_lens(confident).tolist() == [GAMMA + 1]
    assert cut.disabled and not cut.calibrated and cut.cost_ms() is None
    num_events = len(timings.events)
    assert run_verify(cut, timings, confident) == GAMMA + 1
    assert len(timings.events) == num_events


def test_a_non_positive_fitted_anchor_time_disables_the_cut():
    cut = calibrate(lambda width: 2.0 * (width - 1) - 1.0)
    assert cut.disabled and cut.cost_ms() is None


def test_the_cut_applies_to_greedy_single_request_batches_only():
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=FakeTimings(lambda w: 1.0).new_event)
    assert cut.applies(bs=1, sampling_info=None)
    assert cut.applies(bs=1, sampling_info=SimpleNamespace(is_all_greedy=True))
    assert not cut.applies(bs=2, sampling_info=SimpleNamespace(is_all_greedy=True))
    assert not cut.applies(bs=1, sampling_info=SimpleNamespace(is_all_greedy=False))


def test_from_envs_builds_the_cut_only_when_the_flag_is_on_and_no_sts_data_is_collected():
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(False):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu") is None
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        assert isinstance(ConfidenceCut.from_envs(gamma=GAMMA, device="cpu"), ConfidenceCut)
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override("/x/sts"):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu") is None
