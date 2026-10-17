"""The DSpark worker's side of the confidence-head verify cut (``SGLANG_DSPARK_CONFIDENCE_CUT``): where the cut sets a
verify's lengths, its launch refusals, the folded proposal's same-step confidence, and the cut's observability (CPU).

`DSparkWorkerV2` cannot be constructed on CPU, so its methods run unbound on stand-ins carrying the attributes they
read, as in test_dspark_draft_graph_init_idempotent.py.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as dw
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    ConfidenceCutStats,
    expect_accept_lengths,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GAMMA, VOCAB = 5, 13
GREEDY = SimpleNamespace(is_all_greedy=True)
SAMPLING = SimpleNamespace(is_all_greedy=False)


class _FakeCut:
    def __init__(self, *, lens, disabled=False, calibrated=True):
        self._lens = lens
        self.disabled = disabled
        self.calibrated = calibrated
        self.seen = []

    def applies(self, *, bs, sampling_info):
        return bs == 1 and (sampling_info is None or sampling_info.is_all_greedy)

    def verify_lens(self, confidence):
        self.seen.append(confidence)
        return self._lens


def _cut_lens(cut, *, bs=1, sampling_info=GREEDY, confidence=None):
    worker = SimpleNamespace(_confidence_cut=cut)
    if confidence is None:
        confidence = torch.full((bs, GAMMA), 0.5)
    return dw.DSparkWorkerV2._cut_verify_lens(
        worker, bs=bs, sampling_info=sampling_info, confidence=confidence
    )


def test_the_cut_sets_a_greedy_single_request_verify_from_this_steps_confidence():
    lens = torch.tensor([3])
    cut = _FakeCut(lens=lens)
    confidence = torch.full((1, GAMMA), 0.7)
    assert _cut_lens(cut, confidence=confidence) is lens
    assert cut.seen == [confidence]


def test_every_draft_is_verified_where_the_cut_does_not_apply():
    lens = torch.tensor([3])
    assert _cut_lens(None) is None  # the flag off
    assert _cut_lens(_FakeCut(lens=lens), bs=2) is None
    assert _cut_lens(_FakeCut(lens=lens), sampling_info=SAMPLING) is None
    # T(w) had a width without a valid sample: the cut stays off for the server's life.
    disabled = _FakeCut(lens=lens, disabled=True)
    assert _cut_lens(disabled) is None and disabled.seen == []
    worker = SimpleNamespace(_confidence_cut=_FakeCut(lens=lens))
    assert dw.DSparkWorkerV2._cut_verify_lens(worker, bs=1, sampling_info=GREEDY, confidence=None) is None


def test_the_cut_refuses_the_top_probability_gate_and_a_draft_without_a_confidence_head():
    attached = []
    worker = SimpleNamespace(
        _top_prob_gate=SimpleNamespace(threshold=0.4),
        draft_model=SimpleNamespace(confidence_head=object()),
        _init_verify_dead_rows=lambda **kwargs: attached.append(kwargs) or "dead rows",
    )
    with pytest.raises(ValueError, match="set only one"):
        dw.DSparkWorkerV2._init_confidence_cut(worker)
    worker._top_prob_gate = None
    worker.draft_model = SimpleNamespace(confidence_head=None)
    with pytest.raises(ValueError, match="confidence head"):
        dw.DSparkWorkerV2._init_confidence_cut(worker)
    assert attached == []
    worker.draft_model = SimpleNamespace(confidence_head=object())
    assert dw.DSparkWorkerV2._init_confidence_cut(worker) == "dead rows"
    assert attached[0]["flag"] == "SGLANG_DSPARK_CONFIDENCE_CUT"


def test_the_dead_row_refusals_name_the_cut_that_asked(monkeypatch):
    monkeypatch.setattr(dw, "get_parallel", lambda: SimpleNamespace(enable_dp_attention=False))
    monkeypatch.setattr(dw, "attach_verify_dead_rows", lambda model, dead_rows: 3)
    worker = SimpleNamespace(
        _verify_planner=SimpleNamespace(mode_value="compact"),
        ps=SimpleNamespace(tp_size=1, tp_rank=0),
        verify_num_draft_tokens=GAMMA + 1,
        device="cpu",
        model_runner=SimpleNamespace(model=None),
    )
    with pytest.raises(ValueError, match=r"^SGLANG_DSPARK_CONFIDENCE_CUT is unsupported with SGLANG_RAGGED_VERIFY_MODE"):
        dw.DSparkWorkerV2._init_verify_dead_rows(worker, flag="SGLANG_DSPARK_CONFIDENCE_CUT", description="cut")
    worker._verify_planner = SimpleNamespace(mode_value="static")
    dead_rows = dw.DSparkWorkerV2._init_verify_dead_rows(worker, flag="SGLANG_DSPARK_CONFIDENCE_CUT", description="cut")
    assert isinstance(dead_rows, dw.VerifyDeadRows)


class _ToyDraft:
    """compute_base_logits returns the logits and the confidence tap the dsv4 draft's compute_confidence reads."""

    sample_from_anchor = True

    def __init__(self):
        torch.manual_seed(0)
        self.markov_head = VanillaMarkov(vocab_size=VOCAB, markov_rank=3)
        self.proj = torch.randn(4, VOCAB)

    def compute_base_logits(self, hidden):
        return hidden @ self.proj, hidden[:, :1] * 0.1


@torch.no_grad()
def test_the_folded_greedy_proposal_writes_this_steps_confidence_for_the_static_branch():
    calls = []

    def confidence_fn(*, draft_hidden, anchor_tokens, draft_tokens, confidence_tap):
        calls.append(draft_tokens.clone())
        return torch.sigmoid(confidence_tap.view(-1, GAMMA) + draft_tokens.float())

    sampler = DsparkDraftSampler(
        model=_ToyDraft(), gamma=GAMMA, max_bs=2, device="cpu",
        tp_sync=SimpleNamespace(sync=lambda site, tensor: tensor),
        confidence_fn=confidence_fn, folded_sampling=False,
    )
    hidden = torch.randn(GAMMA, 4)
    sampler(hidden, torch.tensor([4, 0, 0, 0, 0]))
    draft_tokens = sampler.out[:GAMMA].view(1, GAMMA)
    assert torch.equal(calls[0], draft_tokens)
    want = torch.sigmoid((hidden[:, :1] * 0.1).view(1, GAMMA) + draft_tokens.float())
    # The proposer hands confidence_out[:bs] to the worker as proposal.confidence: the same step's, not a lagged one.
    assert torch.allclose(sampler.confidence_out[:1], want)


def test_the_expected_accept_length_sums_the_survival_of_the_verified_drafts():
    confidence = torch.tensor([[0.5, 0.5, 1.0, 1.0, 1.0]] * 4)
    lens = torch.tensor([1, 2, 3, 6])
    # S = 0.5, 0.25, 0.25, 0.25, 0.25: E(1) = 1, E(2) = 1.5, E(3) = 1.75, E(6) = 2.5.
    assert expect_accept_lengths(confidence=confidence, verify_lens=lens).tolist() == [1.0, 1.5, 1.75, 2.5]


def test_the_cut_stats_report_each_log_intervals_verifies_and_reset():
    stats = ConfidenceCutStats(verify_num_draft_tokens=GAMMA + 1, device="cpu")
    assert stats.log_suffix() is None
    confidence = torch.tensor([[0.5, 0.5, 1.0, 1.0, 1.0]])
    stats.observe(confidence=confidence, verify_lens=torch.tensor([2]), commit_lens=torch.tensor([2]))
    stats.observe(confidence=confidence, verify_lens=torch.tensor([6]), commit_lens=torch.tensor([3]))
    assert stats.log_suffix() == (
        "conf cut: verifies 2, w* 4.00, E(w*) 2.00, accept len 2.50, cut rows 4 (total 4)"
    )
    assert stats.log_suffix() is None
    stats.observe(confidence=confidence, verify_lens=torch.tensor([1]), commit_lens=torch.tensor([1]))
    assert stats.log_suffix() == (
        "conf cut: verifies 1, w* 1.00, E(w*) 1.00, accept len 1.00, cut rows 5 (total 9)"
    )
