"""The confidence cut on sampled verifies (SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED): the rule's stopping form, and that the
cut leaves sampled decoding's output law exact (CPU, with a torch model of AcceptSampling; the real kernels run in
test/manual/dsv41/test_dspark_confidence_cut_gpu.py)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
    ConfidenceCut,
    cut_verify_lens,
    stop_rule_verify_lens,
)
from sglang.test import dspark_cut_distribution as dist
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

GAMMA = 5
# The served fit's shape (warmup-calib-check2-20261017: a = 96.5, b = 18.3 ms/row).
BASE_MS, PER_ROW_MS = 96.5, 18.3
VOCAB, LENGTH, TEMPERATURE = 4, 5, 0.6
NUM_SEQUENCES = 300_000
PASS_P = 1e-3


def _affine(base, per_row):
    return torch.tensor([base + per_row * w for w in range(GAMMA + 1)])


def test_the_stopping_rule_picks_the_argmaxs_width_for_any_affine_cost():
    generator = torch.Generator().manual_seed(1)
    confidence = torch.rand((20_000, GAMMA), generator=generator)
    # Exact ties and dead prefixes: zeros and NaNs, as the head can emit.
    confidence[::7, 2] = 0.0
    confidence[::11, 0] = float("nan")
    confidence[::13] = 1.0
    for base, per_row in [(96.5, 18.3), (271.0, 4.7), (100.0, 0.0), (1.0, 10.0), (226.6, 16.8)]:
        cost = _affine(base, per_row)
        assert torch.equal(
            stop_rule_verify_lens(confidence=confidence, cost_ms=cost),
            cut_verify_lens(confidence=confidence, cost_ms=cost),
        ), (base, per_row)


def test_the_stopping_rule_admits_slot_k_from_confidences_before_it_only():
    """w >= k must not read c_k or anything after it (c_k carries draft k - 1; the slot to admit is draft k). Mutant:
    the argmax over a cost table that rises then falls (non-affine), which looks ahead -- red on the argmax only."""
    generator = torch.Generator().manual_seed(2)
    cost = _affine(BASE_MS, PER_ROW_MS)
    confidence = torch.rand((5_000, GAMMA), generator=generator)
    lens = stop_rule_verify_lens(confidence=confidence, cost_ms=cost)
    for k in range(1, GAMMA + 2):
        perturbed = confidence.clone()
        perturbed[:, k - 1 :] = torch.rand((5_000, GAMMA - k + 1), generator=generator)
        again = stop_rule_verify_lens(confidence=perturbed, cost_ms=cost)
        assert torch.equal(again >= k, lens >= k), k
    # The look-ahead the stopping form rules out: with a cost that is not affine, the argmax's w >= 3 can depend on c_3.
    # Rates: low [1, 1.9, 1.36, 1.36, 1.36, 1.36]; high [1, 1.9, 1.36, 1.76, 2.17, 2.57].
    bumpy = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    low = torch.tensor([[0.9, 0.9, 0.0, 0.0, 0.0]])
    high = torch.tensor([[0.9, 0.9, 1.0, 1.0, 1.0]])
    assert int(cut_verify_lens(confidence=low, cost_ms=bumpy)) == 2
    assert int(cut_verify_lens(confidence=high, cost_ms=bumpy)) == GAMMA + 1
    assert int(stop_rule_verify_lens(confidence=low, cost_ms=bumpy)) == 2
    assert int(stop_rule_verify_lens(confidence=high, cost_ms=bumpy)) == 2


def test_sampled_verifies_take_the_stopping_form_and_greedy_ones_the_argmax(monkeypatch):
    cut = dist.calibrated_cut(gamma=GAMMA, base_ms=BASE_MS, per_row_ms=PER_ROW_MS, cut_sampled=True)
    from sglang.srt.speculative.dspark_components import dspark_confidence_cut as cc

    calls = []
    monkeypatch.setattr(cc, "stop_rule_verify_lens", lambda **kw: calls.append("stop") or torch.tensor([2]))
    monkeypatch.setattr(cc, "cut_verify_lens", lambda **kw: calls.append("argmax") or torch.tensor([3]))
    confidence = torch.full((1, GAMMA), 0.5)
    assert cut.verify_lens(confidence, sampled=True).tolist() == [2]
    assert cut.verify_lens(confidence).tolist() == [3]
    assert calls == ["stop", "argmax"]


def test_the_sampled_flag_makes_the_cut_apply_to_sampled_single_request_batches():
    sampling = SimpleNamespace(is_all_greedy=False)
    greedy = SimpleNamespace(is_all_greedy=True)
    off = ConfidenceCut(gamma=GAMMA, device="cpu")
    on = ConfidenceCut(gamma=GAMMA, device="cpu", cut_sampled=True)
    assert not off.applies(bs=1, sampling_info=sampling) and off.applies(bs=1, sampling_info=greedy)
    assert on.applies(bs=1, sampling_info=sampling) and on.applies(bs=1, sampling_info=greedy)
    assert on.applies(bs=1, sampling_info=None)
    assert not on.applies(bs=2, sampling_info=sampling)
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.override(False):
            assert not ConfidenceCut.from_envs(gamma=GAMMA, device="cpu").cut_sampled
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.override(True):
            assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu").cut_sampled


def _decode(verify_lens_fn, *, seed):
    toy = dist.MarkovToy(vocab=VOCAB, gamma=GAMMA, temperature=TEMPERATURE, seed=0)
    generator = torch.Generator().manual_seed(seed)
    widths = []
    sequences = dist.generate(
        toy,
        n=NUM_SEQUENCES,
        length=LENGTH,
        verify_lens_fn=verify_lens_fn,
        accept_fn=lambda **kw: dist.reference_accept(**kw, generator=generator),
        generator=generator,
        widths_out=widths,
    )
    return toy, sequences, torch.cat(widths)


def _full_width(confidence, drafts):
    return torch.full((confidence.shape[0],), GAMMA + 1, dtype=torch.int64)


@pytest.fixture(scope="module")
def sampled_cut():
    return dist.calibrated_cut(gamma=GAMMA, base_ms=BASE_MS, per_row_ms=PER_ROW_MS, cut_sampled=True)


def test_the_output_law_is_the_targets_with_every_row_verified():
    toy, sequences, _ = _decode(_full_width, seed=10)
    assert dist.sequence_pvalue(toy, sequences) > PASS_P


def test_the_cut_keeps_the_output_law_the_targets_under_sampling(sampled_cut):
    toy, sequences, widths = _decode(lambda c, d: sampled_cut.verify_lens(c, sampled=True), seed=11)
    # Not vacuous: the cut picked several widths, most of them short of every row.
    counts = torch.bincount(widths, minlength=GAMMA + 2)
    assert (counts[1:] > 0).all() and widths.float().mean() < GAMMA, counts.tolist()
    assert dist.sequence_pvalue(toy, sequences) > PASS_P


def test_a_cut_that_reads_the_admitted_slots_own_draft_shifts_the_output_law(sampled_cut):
    """The distribution test's power, on the bug the stopping form rules out: admitting slot k from draft k's token."""
    mutant = dist.token_at_slot_mutant(lambda c, d: sampled_cut.verify_lens(c, sampled=True), token=1)
    toy, sequences, _ = _decode(mutant, seed=12)
    assert dist.sequence_pvalue(toy, sequences) < 1e-6


def test_the_goodness_of_fit_test_rejects_a_one_percent_shift_at_this_sample_size():
    toy = dist.MarkovToy(vocab=VOCAB, gamma=GAMMA, temperature=TEMPERATURE, seed=0)
    exact = toy.exact_sequence_probs(LENGTH)
    top2 = torch.topk(exact, 2).indices
    shifted = exact.clone()
    shifted[top2[0]] -= 0.01
    shifted[top2[1]] += 0.01  # total variation 0.01
    generator = torch.Generator().manual_seed(13)
    counts = torch.bincount(
        torch.multinomial(shifted.float(), NUM_SEQUENCES, replacement=True, generator=generator),
        minlength=exact.numel(),
    )
    assert dist.chi_square_pvalue(counts=counts, probs=exact) < 1e-3
