"""DSpark's confidence-head verify cut: the w* rule, and the T(w) calibration that feeds it once the server is truly
serving (CPU)."""

import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components import dspark_confidence_cut as cc
from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
    CALIBRATION_VERIFIES,
    COPY_ENGINE_ARMED,
    COPY_ENGINE_OFF,
    COPY_ENGINE_PENDING,
    CalibrationGate,
    ConfidenceCut,
    cut_verify_lens,
    is_serving_rid,
    ram_miss_copy_engine_state,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GAMMA = 5
NAN = float("nan")
OPEN = CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_ARMED, serving_request=True)


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


def run_verify(cut, timings, confidence, gate=OPEN):
    """One eligible verify as the worker runs it; None when the gate held it back (every row verified, untimed)."""
    if not cut.calibrated and not cut.admit_calibration(gate):
        return None
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


def test_calibration_takes_96_verifies_16_per_width_and_logs_the_fit_with_per_width_medians(caplog):
    assert CALIBRATION_VERIFIES == 96
    timings = FakeTimings(lambda width: 100.0 + 20.0 * (width - 1))
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    widths = [run_verify(cut, timings, confident) for _ in range(CALIBRATION_VERIFIES)]
    assert [widths.count(width) for width in range(1, GAMMA + 2)] == [16] * (GAMMA + 1)
    assert len(timings.events) == 2 * CALIBRATION_VERIFIES
    timings.complete_all()
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        cut.verify_lens(confident)
    assert cut.calibrated
    lines = [r.getMessage() for r in caplog.records if "T(w) ms" in r.getMessage()]
    assert len(lines) == 1
    assert "a=100.0000, b=20.0000/row, r2=1.0000" in lines[0]
    assert "per-width median ms [100.0, 120.0, 140.0, 160.0, 180.0, 200.0]" in lines[0]
    assert "counts [16, 16, 16, 16, 16, 16]" in lines[0]


@pytest.mark.parametrize(
    "closed",
    [
        CalibrationGate(engine_ready=False, copy_engine=COPY_ENGINE_OFF, serving_request=True),
        CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_PENDING, serving_request=True),
        CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_ARMED, serving_request=False),
    ],
    ids=["engine-init", "copy-engine-pending", "warm-up-request"],
)
def test_no_verify_is_forced_or_timed_before_the_gate_opens_then_calibration_starts_at_width_1(closed, caplog):
    timings = FakeTimings(lambda width: float(width))
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    assert not closed.open
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        assert [run_verify(cut, timings, confident, gate=closed) for _ in range(7)] == [None] * 7
        assert timings.events == [] and not cut.calibrated
        assert not any("calibration starts" in r.getMessage() for r in caplog.records)
        # The held verifies used none of the round robin: the first admitted verify forces width 1.
        assert run_verify(cut, timings, confident) == 1
    starts = [r.getMessage() for r in caplog.records if "calibration starts" in r.getMessage()]
    assert starts == [
        "dspark confidence cut: T(w) calibration starts at eligible verify 8, after 7 held back (engine init and "
        "startup warm-up done True, RAM-miss copy engine armed, serving request True): 96 verifies, widths 1..6 "
        "round robin"
    ]
    assert len(timings.events) == 2


def test_a_health_check_mid_calibration_is_held_back_and_logs_no_second_start(caplog):
    timings = FakeTimings(lambda width: 100.0 + 20.0 * (width - 1))
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=timings.new_event)
    confident = torch.full((1, GAMMA), 0.99)
    health_check = CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_OFF, serving_request=False)
    widths = []
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        for i in range(CALIBRATION_VERIFIES + 3):
            widths.append(run_verify(cut, timings, confident, gate=health_check if i in (2, 50, 51) else OPEN))
    assert [w for w in widths if w is None] == [None] * 3
    assert [w for w in widths if w is not None] == [i % (GAMMA + 1) + 1 for i in range(CALIBRATION_VERIFIES)]
    assert len(timings.events) == 2 * CALIBRATION_VERIFIES
    assert sum("calibration starts" in r.getMessage() for r in caplog.records) == 1
    timings.complete_all()
    cut.verify_lens(confident)
    assert cut.cost_ms() == pytest.approx([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])


def _calibrate_with_draft(*, draft_cost, draft_ms=30.0):
    """96 forced verifies as the worker runs them, each step's draft opened by begin_draft and closed by the timed
    verify's start event. The fake clock: begin_draft at t, the verify start at t + draft_ms, its end at
    t + draft_ms + T(w)."""
    clock = {"t": 0.0}

    class Clocked:
        def __init__(self):
            self.t = None

        def record(self):
            self.t = clock["t"]

        def query(self):
            return True

        def elapsed_time(self, end):
            return end.t - self.t

    cut = ConfidenceCut(gamma=GAMMA, device="cpu", new_event=Clocked, draft_cost=draft_cost)
    confident = torch.full((1, GAMMA), 0.99)
    for _ in range(CALIBRATION_VERIFIES):
        cut.begin_draft()
        clock["t"] += draft_ms
        assert cut.admit_calibration(OPEN)
        width = int(cut.verify_lens(confident)[0])
        cut.begin_verify()
        clock["t"] += 100.0 + 20.0 * (width - 1)
        cut.end_verify()
        clock["t"] += 1.0
    cut.begin_draft()
    cut.verify_lens(confident)
    return cut


def test_the_draft_ms_is_timed_on_the_calibration_verifies_and_logged_whether_or_not_it_is_in_the_cost(caplog):
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        off = _calibrate_with_draft(draft_cost=False)
    assert off.calibrated and off.draft_ms() == pytest.approx(30.0)
    assert off.cost_ms() == pytest.approx([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    assert off.verify_cost_ms() == off.cost_ms()
    line = next(r.getMessage() for r in caplog.records if "T(w) ms" in r.getMessage())
    assert "draft ms 30.0000 (median; mean 30.0000, n 96), not in the cost" in line
    # Calibrated: begin_draft records nothing more.
    last = off._draft_start_event
    off.begin_draft()
    assert off._draft_start_event is last


def test_with_draft_cost_on_the_rule_divides_by_draft_plus_verify_ms(caplog):
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        on = _calibrate_with_draft(draft_cost=True, draft_ms=200.0)
    assert on.verify_cost_ms() == pytest.approx([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    assert on.cost_ms() == pytest.approx([300.0, 320.0, 340.0, 360.0, 380.0, 400.0])
    line = next(r.getMessage() for r in caplog.records if "T(w) ms" in r.getMessage())
    assert "in the cost: [300.0, 320.0, 340.0, 360.0, 380.0, 400.0]" in line
    # c = 0.15: verify-only, E(2) / T(2) = 1.15 / 120 loses to 1 / 100; with the draft's 200 ms, 1.15 / 320 beats
    # 1 / 300.
    doubtful = torch.full((1, GAMMA), 0.15)
    assert int(on.verify_lens(doubtful)[0]) > 1
    assert int(_calibrate_with_draft(draft_cost=False, draft_ms=200.0).verify_lens(doubtful)[0]) == 1
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST.override(True):
            assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu").draft_cost
        assert not ConfidenceCut.from_envs(gamma=GAMMA, device="cpu").draft_cost


def test_a_calibration_without_a_timed_draft_logs_it_and_keeps_t_w_alone():
    cut = calibrate(lambda width: 100.0 + 20.0 * (width - 1))
    assert cut.draft_ms() is None and cut.cost_ms() == pytest.approx([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])


def test_serving_rids_exclude_warm_up_and_health_check_requests():
    assert is_serving_rid("0a1b2c")
    assert not is_serving_rid("WARMUP_0a1b2c_0")
    assert not is_serving_rid("HEALTH_CHECK_0a1b2c")
    assert not is_serving_rid(None)


def test_the_copy_engine_state_reads_the_ram_miss_service_without_importing_it(monkeypatch):
    name = "sglang.srt.layers.moe.expert_engine.service"

    def with_service(service):
        module = SimpleNamespace(ExpertEngineService=SimpleNamespace(_instance=service))
        monkeypatch.setitem(sys.modules, name, module)
        return ram_miss_copy_engine_state()

    monkeypatch.delitem(sys.modules, name, raising=False)
    assert ram_miss_copy_engine_state() == COPY_ENGINE_OFF
    assert name not in sys.modules
    assert with_service(None) == COPY_ENGINE_OFF
    assert with_service(SimpleNamespace(copy_engine=False, copy_engine_armed=False)) == COPY_ENGINE_OFF
    assert with_service(SimpleNamespace(copy_engine=True, copy_engine_armed=False)) == COPY_ENGINE_PENDING
    assert with_service(SimpleNamespace(copy_engine=True, copy_engine_armed=True)) == COPY_ENGINE_ARMED


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


def test_from_envs_preloads_the_cut_accept_kernels_on_cuda_only_when_the_cut_is_built(monkeypatch):
    """A cut verify's accept kernels first run at T(w) calibration, after the warm-ups, so the worker loads them at
    init. Mutant: the preload is dropped, runs with the cut off or in STS collect mode, or on a CPU device -- red."""
    from sglang.kernels.ops.speculative.dspark import dspark_accept

    calls = []
    monkeypatch.setattr(dspark_accept, "preload_cut_accept_kernels", lambda **kwargs: calls.append(kwargs))
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(False):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cuda") is None
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override("/x/sts"):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cuda") is None
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu") is not None
        assert calls == []
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cuda") is not None
    assert calls == [{"device": "cuda", "verify_num_draft_tokens": GAMMA + 1}]


def _calibrated_cut(live_fit_interval, truth, timings=None):
    """A cut whose forced calibration measured ``truth(width)`` ms, with the live refit set as asked."""
    timings = timings or FakeTimings(lambda width: truth[0](width))
    cut = ConfidenceCut(
        gamma=GAMMA, device="cpu", new_event=timings.new_event, live_fit_interval=live_fit_interval
    )
    confident = torch.full((1, GAMMA), 0.99)
    for _ in range(CALIBRATION_VERIFIES):
        run_verify(cut, timings, confident)
    timings.complete_all()
    cut.verify_lens(confident)
    assert cut.calibrated
    return cut, timings


def _serve(cut, timings, count, seed=0):
    """``count`` calibrated verifies at assorted confidences, their timings completed before the next one."""
    generator = torch.Generator().manual_seed(seed)
    widths = []
    for _ in range(count):
        timings.complete_all()
        confidence = torch.rand((1, GAMMA), generator=generator)
        widths.append(run_verify(cut, timings, confidence))
    timings.complete_all()
    cut.verify_lens(torch.full((1, GAMMA), 0.5))
    return widths


def test_the_live_refit_follows_the_server_out_of_a_slow_calibration():
    """The calibration saw a slow phase (100 + 40 (w - 1) ms); the server then ran at 40 + 10 (w - 1). The frozen table
    keeps the slow numbers, the live one reaches the new ones. Mutant: the refit never installs, or reads the wrong
    width -- red."""
    slow = lambda width: 100.0 + 40.0 * (width - 1)  # noqa: E731
    fast = lambda width: 40.0 + 10.0 * (width - 1)  # noqa: E731
    live, live_timings = _calibrated_cut(16, [slow])
    frozen, frozen_timings = _calibrated_cut(0, [slow])
    assert live.cost_ms() == pytest.approx(frozen.cost_ms())
    live_timings.ms_for_width = fast
    frozen_timings.ms_for_width = fast
    widths = _serve(live, live_timings, 240)
    _serve(frozen, frozen_timings, 240)
    assert len(set(widths)) >= 3
    assert live.cost_ms() == pytest.approx([40.0 + 10.0 * i for i in range(6)], rel=0.02)
    assert frozen.cost_ms() == pytest.approx([100.0 + 40.0 * i for i in range(6)])


def test_the_live_refit_reads_only_completed_timings_and_forgets_a_phase_after_the_window():
    cut, timings = _calibrated_cut(8, [lambda width: 40.0 + 10.0 * (width - 1)])
    before = list(cut.cost_ms())
    # Verifies whose end events have not completed are not read: no refit however many are issued.
    generator = torch.Generator().manual_seed(1)
    for _ in range(40):
        run_verify(cut, timings, torch.rand((1, GAMMA), generator=generator))
    assert cut.cost_ms() == before
    timings.complete_all()
    # A slow phase, then a fast one: after a window of fast verifies the slow ones are gone.
    timings.ms_for_width = lambda width: 90.0 + 30.0 * (width - 1)
    _serve(cut, timings, 200, seed=2)
    assert cut.cost_ms()[0] == pytest.approx(90.0, rel=0.05)
    timings.ms_for_width = lambda width: 40.0 + 10.0 * (width - 1)
    _serve(cut, timings, 400, seed=3)
    assert cut.cost_ms() == pytest.approx([40.0 + 10.0 * i for i in range(6)], rel=0.03)


def test_a_full_ring_of_unread_timings_skips_timing_instead_of_overwriting_a_slot():
    from sglang.srt.speculative.dspark_components.dspark_confidence_cut import LIVE_RING_SLOTS

    cut, timings = _calibrated_cut(8, [lambda width: 40.0 + 10.0 * (width - 1)])
    generator = torch.Generator().manual_seed(4)
    for _ in range(3 * LIVE_RING_SLOTS):
        run_verify(cut, timings, torch.rand((1, GAMMA), generator=generator))
    assert len(cut._live_pending) == LIVE_RING_SLOTS
    timings.complete_all()
    cut.verify_lens(torch.full((1, GAMMA), 0.5))
    assert len(cut._live_pending) == 0 and len(cut._live_window) == LIVE_RING_SLOTS


def test_the_live_refit_keeps_the_table_when_the_window_holds_one_width():
    cut, timings = _calibrated_cut(8, [lambda width: 100.0 + 20.0 * (width - 1)])
    before = list(cut.cost_ms())
    for _ in range(100):
        timings.complete_all()
        run_verify(cut, timings, torch.full((1, GAMMA), 0.99))
    timings.complete_all()
    cut.verify_lens(torch.full((1, GAMMA), 0.99))
    assert cut.cost_ms() == before


def test_with_the_live_refit_off_a_calibrated_cut_times_nothing_more():
    cut, timings = _calibrated_cut(0, [lambda width: 100.0 + 20.0 * (width - 1)])
    count = len(timings.events)
    _serve(cut, timings, 20)
    assert len(timings.events) == count


def test_the_live_refit_keeps_the_draft_ms_in_the_cost_with_draft_cost_on():
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", draft_cost=True, live_fit_interval=8)
    cut._draft_ms = 7.0
    cut._install_cost(cc.AffineCostFit(base_ms=50.0, per_row_ms=10.0, r2=1.0))
    assert cut.cost_ms() == pytest.approx([57.0 + 10.0 * i for i in range(6)])
    assert cut.verify_cost_ms() == pytest.approx([50.0 + 10.0 * i for i in range(6)])


def test_from_envs_reads_the_live_fit_interval():
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")._live_fit_interval == 0
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL.override(32):
            assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")._live_fit_interval == 32
