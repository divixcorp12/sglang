"""The DSpark worker's side of the confidence-head verify cut (``SGLANG_DSPARK_CONFIDENCE_CUT``): where the cut sets a
verify's lengths, its launch refusals, the folded proposal's same-step confidence, and the cut's observability (CPU).

`DSparkWorkerV2` cannot be constructed on CPU, so its methods run unbound on stand-ins carrying the attributes they
read, as in test_dspark_draft_graph_init_idempotent.py.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models.dspark import DSparkConfidenceHead, VanillaMarkov
from sglang.srt.speculative.dspark_components import dspark_confidence_cut as dw_cut
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as dw
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    ConfidenceCutStats,
    DsparkStepObservers,
    expect_accept_lengths,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GAMMA, VOCAB = 5, 13
GREEDY = SimpleNamespace(is_all_greedy=True)
SAMPLING = SimpleNamespace(is_all_greedy=False)


class _FakeCut:
    def __init__(self, *, lens, disabled=False, calibrated=True, admit=True, cut_sampled=False):
        self._lens = lens
        self.disabled = disabled
        self.calibrated = calibrated
        self._admit = admit
        self.cut_sampled = cut_sampled
        self.seen = []
        self.sampled = []
        self.gates = []

    applies = dw_cut.ConfidenceCut.applies

    def admit_calibration(self, gate):
        self.gates.append(gate)
        return self._admit

    def verify_lens(self, confidence, *, sampled=False):
        self.seen.append(confidence)
        self.sampled.append(sampled)
        return self._lens


SERVING = [SimpleNamespace(rid="0a1b2c")]


def _cut_lens(cut, *, bs=1, sampling_info=GREEDY, confidence=None, reqs=SERVING):
    worker = SimpleNamespace(_confidence_cut=cut, _confidence_cut_gate=dw.DSparkWorkerV2._confidence_cut_gate)
    if confidence is None:
        confidence = torch.full((bs, GAMMA), 0.5)
    return dw.DSparkWorkerV2._cut_verify_lens(
        worker, bs=bs, sampling_info=sampling_info, confidence=confidence, reqs=reqs
    )


def test_the_cut_sets_a_greedy_single_request_verify_from_this_steps_confidence():
    lens = torch.tensor([3])
    cut = _FakeCut(lens=lens)
    confidence = torch.full((1, GAMMA), 0.7)
    assert _cut_lens(cut, confidence=confidence) is lens
    assert cut.seen == [confidence] and cut.sampled == [False]


def test_with_sampled_cutting_on_a_sampled_single_request_verify_is_cut_by_the_stopping_rule_and_calibrates():
    """SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED: a sampled bs-1 verify reaches the gate and the cut, asking for the rule's
    stopping form; a greedy one still takes the argmax. Mutant: the worker passes sampled=False, or the gate is skipped
    for sampled batches -- red."""
    lens = torch.tensor([2])
    cut = _FakeCut(lens=lens, calibrated=False, admit=True, cut_sampled=True)
    assert _cut_lens(cut, sampling_info=SAMPLING) is lens
    assert len(cut.gates) == 1 and cut.sampled == [True]
    assert _cut_lens(cut, sampling_info=GREEDY) is lens and cut.sampled == [True, False]
    assert _cut_lens(cut, sampling_info=SAMPLING, bs=2) is None


def test_every_draft_is_verified_where_the_cut_does_not_apply():
    lens = torch.tensor([3])
    assert _cut_lens(None) is None  # the flag off
    assert _cut_lens(_FakeCut(lens=lens), bs=2) is None
    assert _cut_lens(_FakeCut(lens=lens), sampling_info=SAMPLING) is None
    # T(w) had a width without a valid sample: the cut stays off for the server's life.
    disabled = _FakeCut(lens=lens, disabled=True)
    assert _cut_lens(disabled) is None and disabled.seen == []
    worker = SimpleNamespace(_confidence_cut=_FakeCut(lens=lens))
    assert dw.DSparkWorkerV2._cut_verify_lens(
        worker, bs=1, sampling_info=GREEDY, confidence=None, reqs=SERVING
    ) is None


def test_before_t_w_freezes_the_gate_decides_and_a_held_verify_covers_every_row():
    lens = torch.tensor([3])
    held = _FakeCut(lens=lens, calibrated=False, admit=False)
    assert _cut_lens(held) is None and held.seen == [] and len(held.gates) == 1
    admitted = _FakeCut(lens=lens, calibrated=False, admit=True)
    assert _cut_lens(admitted) is lens and len(admitted.gates) == 1
    # Frozen: the gate is not read again.
    frozen = _FakeCut(lens=lens, calibrated=True, admit=False)
    assert _cut_lens(frozen) is lens and frozen.gates == []


def test_the_gate_reads_engine_init_the_copy_engine_and_the_batchs_rids(monkeypatch):
    monkeypatch.setattr(dw.triton_load_watch, "_serving_started", False)
    monkeypatch.setattr(dw, "ram_miss_copy_engine_state", lambda: dw_cut.COPY_ENGINE_PENDING)
    gate = dw.DSparkWorkerV2._confidence_cut_gate(SERVING)
    assert gate == dw.CalibrationGate(engine_ready=False, copy_engine="pending", serving_request=True)
    assert not gate.open
    monkeypatch.setattr(dw.triton_load_watch, "_serving_started", True)
    assert not dw.DSparkWorkerV2._confidence_cut_gate(SERVING).open
    monkeypatch.setattr(dw, "ram_miss_copy_engine_state", lambda: dw_cut.COPY_ENGINE_ARMED)
    assert dw.DSparkWorkerV2._confidence_cut_gate(SERVING).open
    from sglang.srt.entrypoints.warmup import warmup_rid

    for warm_up_rid in (warmup_rid(), f"{warmup_rid()}_0", "HEALTH_CHECK_0a1b2c"):
        gate = dw.DSparkWorkerV2._confidence_cut_gate([SimpleNamespace(rid=warm_up_rid)])
        assert not gate.serving_request and not gate.open


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


def test_sts_collection_builds_no_cut_so_every_row_is_verified(monkeypatch):
    built = []
    monkeypatch.setattr(
        dw.ConfidenceCut, "from_envs", classmethod(lambda cls, *, gamma, device: built.append(gamma) or "cut")
    )
    worker = SimpleNamespace(gamma=GAMMA, device="cpu", ps=SimpleNamespace(tp_rank=0), verify_num_draft_tokens=GAMMA + 1)
    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True):
        with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override("/tmp/sts-collect/shard"):
            assert dw.DSparkWorkerV2._build_confidence_cut(worker) is None
        assert built == []
        with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(""):
            assert dw.DSparkWorkerV2._build_confidence_cut(worker) == "cut"
        assert built == [GAMMA]


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


@torch.no_grad()
def test_static_sts_collection_records_the_folded_proposals_pre_sts_logits():
    model = _ToyDraft()
    model.confidence_head = DSparkConfidenceHead(hidden_size=1, markov_rank=0, with_markov=False)
    model.confidence_head.sts_temperatures = torch.tensor(2.0)

    def confidence_fn(*, draft_hidden, anchor_tokens, draft_tokens, confidence_tap):
        raw = model.confidence_head(confidence_tap.view(-1, GAMMA, 1))
        return model.confidence_head.apply_sts(raw)

    def sampler(record_confidence_raw):
        return DsparkDraftSampler(
            model=model, gamma=GAMMA, max_bs=2, device="cpu",
            tp_sync=SimpleNamespace(sync=lambda site, tensor: tensor),
            confidence_fn=confidence_fn, folded_sampling=False, record_confidence_raw=record_confidence_raw,
        )

    assert sampler(False).confidence_raw_out is None
    recording = sampler(True)
    hidden = torch.randn(GAMMA, 4)
    recording(hidden, torch.tensor([4, 0, 0, 0, 0]))
    want_raw = model.confidence_head(hidden[:, :1].view(1, GAMMA, 1) * 0.1)
    assert torch.allclose(recording.confidence_raw_out[:1], want_raw)
    assert torch.allclose(recording.confidence_out[:1], torch.sigmoid(want_raw / 2.0))


def test_a_folded_verify_feeds_sts_collection_only_with_recorded_logits():
    recorded = []
    observers = SimpleNamespace(
        _verify_markers=None,
        _planner=SimpleNamespace(carries_confidence=True, is_compact_mode=False, last_confidence_raw=None),
        _maybe_record_sts_collect=lambda **kwargs: recorded.append(kwargs["confidence_raw"]),
        _confidence_probe=SimpleNamespace(maybe_observe=lambda **kwargs: None),
        _block_accept_recorder=None,
        _confidence_cut_stats=None,
        _info_dumper=SimpleNamespace(enabled=False),
    )
    kwargs = dict(
        forward_ct=0, reqs=[], bs=1, verify_ids_2d=None, target_logits=None, layout=None, confidence=None,
        prefix_lens=None, draft_tokens=None, draft_block=None, sampling_info=None, correct_len=None,
        cap_trim_lens=None, bonus=None, commit_lens=None, verify_token_budget=None, req_pool_indices=None,
        verify_tier_num_tokens=-1, dp_tier_num_tokens=None,
    )
    raw = torch.zeros(1, GAMMA)
    DsparkStepObservers.observe_verify_step(observers, proposal_folded=True, **kwargs)
    assert recorded == []  # as before: the head's last raw tensor is not this folded step's
    DsparkStepObservers.observe_verify_step(observers, proposal_folded=True, folded_confidence_raw=raw, **kwargs)
    DsparkStepObservers.observe_verify_step(observers, proposal_folded=False, **kwargs)
    assert recorded[0] is raw and recorded[1] is None  # the eager step reads the planner's own raw


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
    stats.observe(confidence=confidence, verify_lens=torch.tensor([6]), commit_lens=torch.tensor([3]), sampled=True)
    assert stats.log_suffix() == (
        "conf cut: verifies 2, w* 4.00, E(w*) 2.00, accept len 2.50, cut rows 4 (total 4), "
        "widths [0, 1, 0, 0, 0, 1], sampled 1"
    )
    assert stats.log_suffix() is None
    stats.observe(confidence=confidence, verify_lens=torch.tensor([1]), commit_lens=torch.tensor([1]))
    assert stats.log_suffix() == (
        "conf cut: verifies 1, w* 1.00, E(w*) 1.00, accept len 1.00, cut rows 5 (total 9), "
        "widths [1, 0, 0, 0, 0, 0], sampled 0"
    )
