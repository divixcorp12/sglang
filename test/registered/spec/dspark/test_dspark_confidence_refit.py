"""The confidence cut's online refit (SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL): the per-position slope fit, the
lazy outcome ring, the refit cadence and its guard (CPU)."""

import concurrent.futures
import logging

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_confidence_refit as rf
from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
    ConfidenceCut,
    cut_verify_lens,
    stop_rule_verify_lens,
)
from sglang.test import dspark_cut_distribution as dist
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GAMMA = 5


class _Now:
    """An executor that runs the refit at submit, so the next verify picks it up."""

    def submit(self, fn, **kwargs):
        future = concurrent.futures.Future()
        future.set_result(fn(**kwargs))
        return future


class _Event:
    def __init__(self, ready=True):
        self.ready = ready

    def record(self):
        pass

    def query(self):
        return self.ready


def _synthetic(n, true_slopes, *, seed=0, widths=None):
    """Verifies whose draft k is accepted, given 1..k-1 were, with probability sigmoid(true_slope_k * logit_k)."""
    g = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn((n, GAMMA), generator=g, dtype=torch.float64)
    p = torch.sigmoid(logits * torch.tensor(true_slopes, dtype=torch.float64))
    accept = torch.rand((n, GAMMA), generator=g, dtype=torch.float64) < p
    taken = torch.cumprod(accept.to(torch.int64), dim=1).sum(dim=1)
    if widths is None:
        widths = torch.full((n,), GAMMA + 1, dtype=torch.int64)
    return logits, widths, taken


def test_the_slope_fit_recovers_each_positions_slope_from_the_verifies_that_ran_it():
    true = [0.5, 1.0, 2.0, 0.7, 1.5]
    logits, widths, taken = _synthetic(40_000, true)
    fit = rf.fit_slopes(logits=logits, widths=widths, taken=taken)
    assert fit.slopes == pytest.approx(true, rel=0.1)
    assert all(after <= before for before, after in zip(fit.nll_before, fit.nll_after))


def test_positions_past_the_width_or_with_too_few_samples_keep_the_offline_slope():
    true = [0.5, 0.5, 0.5, 0.5, 0.5]
    # Every verify cut at width 2: positions 1 and 2 ran, 3..5 never did (their outcomes read dead rows).
    logits, widths, taken = _synthetic(5_000, true, widths=torch.full((5_000,), 2))
    fit = rf.fit_slopes(logits=logits, widths=widths, taken=taken)
    assert fit.slopes[:2] == pytest.approx([0.5, 0.5], rel=0.15)
    assert fit.slopes[2:] == [1.0, 1.0, 1.0] and fit.samples[2:] == [0, 0, 0]
    # A taken count past the width (the dead rows' accepts) must not leak into position 3's labels.
    fake_taken = torch.full((5_000,), GAMMA)
    assert rf.fit_slopes(logits=logits, widths=widths, taken=fake_taken).samples[2:] == [0, 0, 0]


def _refit(interval=100, events=None):
    return rf.OnlineRefit(
        gamma=GAMMA, device="cpu", interval=interval, executor=_Now(),
        new_event=(lambda: _Event()) if events is None else events,
    )


def _observe_all(refit, logits, widths, taken, *, sampled):
    for x, w, m in zip(logits, widths, taken):
        w = int(w)
        capped = min(int(m), w - 1)
        refit.observe(
            confidence=torch.sigmoid(x.float()).view(1, -1),
            verify_lens=torch.tensor([w]),
            correct_len=torch.tensor([capped], dtype=torch.int32),
            cap_trim_lens=torch.tensor([int(m) - capped], dtype=torch.int32),
            sampled=sampled,
        )


def test_the_refit_runs_every_interval_samples_per_mode_and_logs_its_parameters(caplog):
    true = [2.0, 2.0, 2.0, 2.0, 2.0]  # the offline confidences are under-confident
    logits, widths, taken = _synthetic(1_000, true, seed=3)
    refit = _refit(interval=500)
    with caplog.at_level(logging.INFO, logger=rf.__name__):
        _observe_all(refit, logits, widths, taken, sampled=True)
        refit.observe(  # one more verify drains the last sample and picks the second refit up
            confidence=torch.full((1, GAMMA), 0.5), verify_lens=torch.tensor([1]),
            correct_len=torch.tensor([0]), cap_trim_lens=torch.tensor([0]), sampled=False,
        )
    lines = [r.getMessage() for r in caplog.records if "online refit" in r.getMessage()]
    assert len(lines) == 2 and "online refit 2 (sampled) over the last 1000 verifies" in lines[1]
    assert "slopes [" in lines[1] and "samples per position [1000," in lines[1] and "log loss" in lines[1]
    slopes = refit._modes["sampled"].slopes
    assert slopes.tolist() == pytest.approx(true, rel=0.25)
    assert refit._modes["greedy"].slopes is None  # greedy verifies have their own buffer, still short
    # The guard: the refit raises a confidence where it says more, and never lowers one.
    c = torch.tensor([[0.9, 0.2, 0.5, 0.99, 0.01]])
    out = refit.apply(c, sampled=True)
    assert torch.all(out >= c) and out[0, 0] > c[0, 0] and out[0, 1] == c[0, 1]
    assert refit.apply(c, sampled=False) is c


def test_a_sample_whose_slot_is_still_in_flight_is_dropped_never_waited_on():
    pending = []

    def new_event():
        pending.append(_Event(ready=False))
        return pending[-1]

    refit = _refit(interval=10**9, events=new_event)
    logits, widths, taken = _synthetic(rf.RING_SLOTS + 5, [1.0] * GAMMA)
    _observe_all(refit, logits, widths, taken, sampled=False)
    assert len(pending) == rf.RING_SLOTS and refit.dropped_ct == 5
    assert len(refit._modes["greedy"].buffer) == 0
    for event in pending:
        event.ready = True
    _observe_all(refit, logits[:1], widths[:1], taken[:1], sampled=False)
    assert len(refit._modes["greedy"].buffer) == rf.RING_SLOTS


def test_the_cut_reads_the_refit_through_its_guard_so_it_never_cuts_narrower_than_offline():
    cost = torch.tensor([96.5 + 18.3 * w for w in range(GAMMA + 1)])
    refit = _refit()
    refit._modes["sampled"].slopes = torch.tensor([0.3, 3.0, 1.0, 0.5, 2.0])
    cut = ConfidenceCut(gamma=GAMMA, device="cpu", cut_sampled=True, refit=refit)
    confidence = torch.rand((20_000, GAMMA), generator=torch.Generator().manual_seed(4))
    offline = stop_rule_verify_lens(confidence=confidence, cost_ms=cost)
    refitted = stop_rule_verify_lens(confidence=cut.cut_confidence(confidence, sampled=True), cost_ms=cost)
    assert torch.all(refitted >= offline) and torch.any(refitted > offline)
    assert torch.equal(cut_verify_lens(confidence=cut.cut_confidence(confidence, sampled=True), cost_ms=cost), refitted)
    assert cut.cut_confidence(confidence, sampled=False) is confidence


def test_the_refitted_cut_keeps_the_sampled_output_law_the_targets():
    cut = dist.calibrated_cut(gamma=GAMMA, base_ms=96.5, per_row_ms=18.3, cut_sampled=True, refit=_refit())
    cut.refit._modes["sampled"].slopes = torch.tensor([0.3, 3.0, 1.0, 0.5, 2.0])
    toy = dist.MarkovToy(vocab=4, gamma=GAMMA, temperature=0.6, seed=0)
    g = torch.Generator().manual_seed(30)
    sequences = dist.generate(
        toy, n=300_000, length=5,
        verify_lens_fn=lambda c, d: cut.verify_lens(cut.cut_confidence(c, sampled=True), sampled=True),
        accept_fn=lambda **kw: dist.reference_accept(**kw, generator=g), generator=g,
    )
    assert dist.sequence_pvalue(toy, sequences) > 1e-3


def test_from_envs_builds_the_refit_only_with_a_positive_interval():
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu").refit is None
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL.override(256):
            cut = ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")
    assert isinstance(cut.refit, rf.OnlineRefit) and cut.refit.interval == 256
    # Without a refit, recording an outcome is a no-op.
    ConfidenceCut(gamma=GAMMA, device="cpu").observe_outcome(
        confidence=torch.full((1, GAMMA), 0.5), verify_lens=torch.tensor([2]), correct_len=torch.tensor([1]),
        cap_trim_lens=torch.tensor([0]), sampled=True,
    )
