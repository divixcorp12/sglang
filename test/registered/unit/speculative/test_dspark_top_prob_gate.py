"""DSpark's top-probability verify gate: the cut rule, where it applies, the eager proposal's probabilities, and the
capped accept (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import sample_draft_block
from sglang.srt.speculative.dspark_components.dspark_top_prob_gate import (
    TopProbGate,
    gate_verify_lens,
)
from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GAMMA, VOCAB = 5, 13
NAN = float("nan")


def test_a_verify_keeps_the_prefix_of_drafts_whose_running_product_clears_the_threshold():
    probs = torch.tensor(
        [
            [0.9, 0.5, 0.95, 1.0, 1.0],  # products 0.9 0.45 0.4275 ...: all kept
            [0.9, 0.4, 1.0, 1.0, 1.0],  # 0.36 < 0.4 at draft 2
            [0.3, 1.0, 1.0, 1.0, 1.0],  # the anchor alone
            [0.5, 0.8, 1.0, 1.0, 1.0],  # exactly 0.4 at draft 2 is kept
            [0.9, NAN, 1.0, 1.0, 1.0],  # a NaN probability ends the prefix
            # 0.99 * 0.4 < 0.4 at draft 2, then a top probability above 1 (rounding) lifts the product back over the
            # threshold: the cut stays at draft 2.
            [0.99, 0.4, 1.02, 1.0, 1.0],
        ]
    )
    assert gate_verify_lens(top_probs=probs, threshold=0.4).tolist() == [6, 2, 1, 3, 2, 2]


def test_the_gate_is_off_at_zero_and_refuses_an_out_of_range_threshold():
    with envs.SGLANG_DSPARK_TOP_PROB_GATE.override(0.0):
        assert TopProbGate.from_envs() is None
    with envs.SGLANG_DSPARK_TOP_PROB_GATE.override(0.4):
        assert TopProbGate.from_envs() == TopProbGate(threshold=0.4)
    for bad in (-0.1, 1.5):
        with envs.SGLANG_DSPARK_TOP_PROB_GATE.override(bad), pytest.raises(ValueError):
            TopProbGate.from_envs()


def test_the_gate_applies_to_greedy_single_request_batches_only():
    gate = TopProbGate(threshold=0.4)
    greedy = SimpleNamespace(is_all_greedy=True)
    sampling = SimpleNamespace(is_all_greedy=False)
    assert gate.applies(bs=1, sampling_info=None)
    assert gate.applies(bs=1, sampling_info=greedy)
    # bs > 1: a request's dead rows are not a suffix of the verify's flat rows.
    assert not gate.applies(bs=2, sampling_info=greedy)
    assert not gate.applies(bs=1, sampling_info=sampling)


@torch.no_grad()
def test_the_eager_greedy_proposal_records_each_steps_top_probability():
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=3)
    base = torch.randn(1, GAMMA, VOCAB) * 3
    anchor = torch.tensor([4])
    sync = SimpleNamespace(sync=lambda site, tensor: tensor)
    kwargs = dict(
        base_logits=base, anchor_tokens=anchor, draft_hidden=None, markov_head=head, device="cpu", tp_sync=sync
    )
    block = sample_draft_block(sampling_info=None, record_top_prob=True, **kwargs)
    want = torch.softmax(block.corrected_logits.float(), dim=-1).amax(dim=-1)
    assert torch.allclose(block.top_probs, want)
    unrecorded = sample_draft_block(sampling_info=None, **kwargs)
    assert unrecorded.top_probs is None
    assert torch.equal(unrecorded.draft_tokens, block.draft_tokens)


def test_a_capped_greedy_accept_never_accepts_past_the_cut():
    drafts = [7, 3, 9, 2, 5]
    candidates = torch.tensor([[1] + drafts])
    # The target agrees with every draft and predicts 11 after the last.
    predict = drafts + [11]
    logits = torch.full((GAMMA + 1, VOCAB + 4), -10.0)
    logits[torch.arange(GAMMA + 1), torch.tensor(predict)] = 10.0
    block = SimpleNamespace(greedy_mask=torch.tensor([True]))

    def accept(cutoff):
        return accept_draft_tokens(
            candidates=candidates, target_logits=logits, draft_block=block, sampling_info=None,
            draft_input=None, gamma=GAMMA, verify_num_draft_tokens=GAMMA + 1, cutoff_verify_lens=cutoff,
        )

    correct_len, bonus, _ = accept(None)
    assert (int(correct_len), int(bonus)) == (5, 11)
    # Three rows verified: the anchor and drafts 1-2. Draft 3 matched, but it was cut; the bonus is the target's
    # prediction after draft 2, which is draft 3's token.
    correct_len, bonus, _ = accept(torch.tensor([3]))
    assert (int(correct_len), int(bonus)) == (2, 9)
    correct_len, bonus, _ = accept(torch.tensor([1]))
    assert (int(correct_len), int(bonus)) == (0, 7)
