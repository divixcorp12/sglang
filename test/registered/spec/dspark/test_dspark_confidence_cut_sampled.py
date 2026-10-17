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


def test_re_tempering_a_head_output_gives_the_sampled_files_survival():
    from sglang.srt.speculative.dspark_components.dspark_confidence_cut import rescale_confidence

    raw = torch.randn((64, GAMMA)) * 4
    head_t = torch.tensor([0.89, 0.79, 1.26, 0.89, 0.63])
    sampled_t = torch.tensor([1.3, 0.5, 2.0, 1.0, 0.9])
    got = rescale_confidence(torch.sigmoid(raw / head_t), head_t / sampled_t)
    assert torch.allclose(got, torch.sigmoid(raw / sampled_t), atol=1e-5)
    edge = torch.tensor([[0.0, 1.0, float("nan"), 0.5, 0.5]])
    out = rescale_confidence(edge, torch.full((GAMMA,), 2.0))
    assert out[0, 0] == 0.0 and out[0, 1] == 1.0 and torch.isnan(out[0, 2]) and out[0, 3] == 0.5


def _sts_file(path, temperatures):
    from sglang.srt.speculative.dspark_components.dspark_sts import DSparkStsCalibration

    path.write_text(DSparkStsCalibration(temperatures=temperatures).to_json())
    return str(path)


def test_a_sampled_sts_file_re_tempers_sampled_verifies_only(tmp_path, monkeypatch):
    from sglang.srt.speculative.dspark_components import dspark_confidence_cut as cc

    head = _sts_file(tmp_path / "head.json", [1.0, 2.0, 1.0, 1.0, 1.0])
    sampled = _sts_file(tmp_path / "sampled.json", [2.0, 1.0, 1.0, 1.0, 4.0])
    rescale = cc.sampled_sts_rescale(gamma=GAMMA, head_sts_path=head, sampled_sts_path=sampled, device="cpu")
    assert rescale.tolist() == [0.5, 2.0, 1.0, 1.0, 0.25]
    assert cc.sampled_sts_rescale(gamma=GAMMA, head_sts_path=None, sampled_sts_path=sampled, device="cpu").tolist() == [
        0.5, 1.0, 1.0, 1.0, 0.25
    ]
    with pytest.raises(ValueError):
        cc.sampled_sts_rescale(gamma=4, head_sts_path=head, sampled_sts_path=sampled, device="cpu")

    cut = ConfidenceCut(gamma=GAMMA, device="cpu", cut_sampled=True, sampled_rescale=rescale)
    confidence = torch.full((1, GAMMA), 0.8)
    assert cut.cut_confidence(confidence, sampled=False) is confidence
    assert torch.allclose(cut.cut_confidence(confidence, sampled=True), cc.rescale_confidence(confidence, rescale))

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_spec", lambda: SimpleNamespace(speculative_dspark_confidence_sts_path=head)
    )
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True), envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED_STS_PATH.override(sampled):
            with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.override(False), pytest.raises(ValueError):
                ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")
            with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.override(True):
                built = ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")
        assert built._sampled_rescale.tolist() == rescale.tolist()
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.override(True):
            assert ConfidenceCut.from_envs(gamma=GAMMA, device="cpu")._sampled_rescale is None


def test_the_draft_accept_rates_are_min_one_p_over_q_for_sampled_rows_and_matches_for_greedy_ones():
    from sglang.srt.speculative.dspark_components.dspark_observability import draft_accept_rates

    gamma, vocab = 2, 3
    target_logits = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]] * 2))
    corrected = torch.log(torch.tensor([[[0.25, 0.5, 0.25], [0.1, 0.2, 0.7]]] * 2))
    candidates = torch.tensor([[9, 1, 2]] * 2)
    block = SimpleNamespace(
        corrected_logits=corrected, temperatures=torch.ones(2), greedy_mask=torch.tensor([False, True])
    )
    sampling = SimpleNamespace(
        is_all_greedy=False, need_top_k_sampling=False, need_top_p_sampling=False, temperatures=torch.ones(2, 1)
    )
    rates = draft_accept_rates(
        candidates=candidates, target_logits=target_logits, draft_block=block, sampling_info=sampling,
        verify_num_draft_tokens=gamma + 1,
    )
    # Row 0 sampled: p/q = 0.3/0.5 and min(1, 0.8/0.7). Row 1 greedy: draft 1 (token 1) is not row 0's argmax (0);
    # draft 2 (token 2) is row 1's argmax.
    assert torch.allclose(rates, torch.tensor([[0.6, 1.0], [0.0, 1.0]]), atol=1e-5)
    greedy_only = draft_accept_rates(
        candidates=candidates, target_logits=target_logits, draft_block=block, sampling_info=None,
        verify_num_draft_tokens=gamma + 1,
    )
    assert greedy_only.tolist() == [[0.0, 1.0], [0.0, 1.0]]


def test_the_collection_records_the_drafts_own_step_probability_top1_and_entropy(tmp_path):
    import math

    from sglang.srt.speculative.dspark_components.dspark_observability import draft_step_features
    from sglang.srt.speculative.dspark_components.dspark_sts import StsDataRecorder

    # Temperature 0.5 turns logits log([0.25, 0.5, 0.25]) into q = [1/6, 2/3, 1/6].
    corrected = torch.log(torch.tensor([[[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]]))
    block = SimpleNamespace(corrected_logits=corrected, temperatures=torch.tensor([0.5]))
    features = draft_step_features(draft_tokens=torch.tensor([[1, 0]]), draft_block=block)
    q = torch.tensor([1 / 6, 2 / 3, 1 / 6])
    assert torch.allclose(features["draft_q"], torch.tensor([[2 / 3, 1 / 6]]), atol=1e-6)
    assert torch.allclose(features["draft_top1"], torch.tensor([[2 / 3, 2 / 3]]), atol=1e-6)
    entropy = -float((q * q.log()).sum())
    assert torch.allclose(features["draft_entropy"], torch.full((1, 2), entropy), atol=1e-5) and entropy < math.log(3)
    assert draft_step_features(draft_tokens=torch.tensor([[1, 0]]),
                               draft_block=SimpleNamespace(corrected_logits=None)) is None

    recorder = StsDataRecorder(path_stem=str(tmp_path / "s"), gamma=2, flush_every=10)
    recorder.record(confidence_raw=torch.zeros(1, 2), num_correct_drafts=torch.tensor([1]), draft_features=features)
    recorder.record(confidence_raw=torch.zeros(1, 2), num_correct_drafts=torch.tensor([1]))
    recorder.flush()
    shard = torch.load(tmp_path / "s.0.pt")
    assert torch.allclose(shard["draft_q"][0], features["draft_q"][0]) and torch.isnan(shard["draft_q"][1]).all()


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


def test_a_confidence_built_from_draft_ks_own_drawn_probability_keeps_the_output_law(sampled_cut):
    """A draft-probability predictor's confidence for draft k reads d_k itself. The stopping rule feeds c_k only to
    the admission of slots k+1 and later, which may read d_1..d_k, so it stays exact. Mutant: c_k reads d_{k+1}
    (shift 1), so admitting slot k+1 reads its own draft -- red."""
    toy, sequences, widths = _decode(dist.drawn_q_cut(sampled_cut), seed=14)
    assert (torch.bincount(widths, minlength=GAMMA + 2)[1:] > 0).sum() >= 4
    assert dist.sequence_pvalue(toy, sequences) > PASS_P
    toy, sequences, _ = _decode(dist.drawn_q_cut(sampled_cut, shift=1), seed=15)
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
