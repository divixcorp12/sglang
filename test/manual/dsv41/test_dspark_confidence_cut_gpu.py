"""DSpark's confidence-head verify cut on the GPU: the folded greedy proposal's confidence replayed in a captured draft
graph (the same step's, with no host read), the startup T(w) calibration over the static branch's begin/end events,
and the static epilogue's accept capped at the cut's width.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

GAMMA, VOCAB, HIDDEN = 5, 4096, 64
TOKENS = GAMMA + 1
SINGLE_RANK = SimpleNamespace(world_size=1)
CALIBRATION_VERIFIES = 96


class _ToyDraft:
    sample_from_anchor = True

    def __init__(self):
        from sglang.srt.models.dspark import VanillaMarkov

        generator = torch.Generator(device="cpu").manual_seed(0)
        self.markov_head = VanillaMarkov(vocab_size=VOCAB, markov_rank=8).cuda()
        self.proj = torch.randn((HIDDEN, VOCAB), generator=generator).cuda()
        self.head = torch.randn((HIDDEN,), generator=generator).cuda()

    def compute_base_logits(self, hidden):
        return hidden @ self.proj, hidden

    def confidence(self, *, draft_hidden, anchor_tokens, draft_tokens, confidence_tap):
        del draft_hidden, anchor_tokens
        bs = draft_tokens.shape[0]
        return torch.sigmoid(confidence_tap.view(bs, GAMMA, HIDDEN) @ self.head + draft_tokens.float() * 1e-4)


@torch.no_grad()
def test_a_captured_folded_proposal_replays_this_steps_confidence():
    from sglang.srt.speculative.dspark_components.dspark_draft_sampler import DsparkDraftSampler

    model = _ToyDraft()
    sampler = DsparkDraftSampler(
        model=model, gamma=GAMMA, max_bs=1, device="cuda",
        tp_sync=SimpleNamespace(sync=lambda site, tensor: tensor),
        confidence_fn=model.confidence, folded_sampling=False,
    )
    hidden = torch.randn((GAMMA, HIDDEN), device="cuda")
    input_ids = torch.tensor([7] + [0] * (GAMMA - 1), device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        sampler(hidden, input_ids)  # warm-up outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # a host read inside the hook would fail the capture
        sampler(hidden, input_ids)

    for seed in (1, 2):
        hidden.copy_(torch.randn((GAMMA, HIDDEN), device="cuda", generator=torch.Generator("cuda").manual_seed(seed)))
        graph.replay()
        torch.cuda.synchronize()
        draft_tokens = sampler.out[:GAMMA].view(1, GAMMA).clone()
        want = model.confidence(
            draft_hidden=None, anchor_tokens=None, draft_tokens=draft_tokens, confidence_tap=hidden
        )
        assert torch.allclose(sampler.confidence_out[:1], want, rtol=1e-4, atol=1e-6), seed


def _cut():
    from sglang.srt.environ import envs
    from sglang.srt.speculative.dspark_components.dspark_confidence_cut import ConfidenceCut

    with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True):
        cut = ConfidenceCut.from_envs(gamma=GAMMA, device=torch.device("cuda"))
    assert cut is not None
    return cut


def test_calibration_times_forced_widths_through_the_verify_events_then_cuts_by_confidence():
    from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
        COPY_ENGINE_ARMED,
        COPY_ENGINE_PENDING,
        CalibrationGate,
    )

    cut = _cut()
    work, scratch = torch.randn((2048, 2048), device="cuda"), torch.empty((2048, 2048), device="cuda")
    confident = torch.full((1, GAMMA), 0.99, device="cuda")
    # Warm-up: the copy engine not armed yet, then a warm-up request. Held back, so no event is recorded.
    for gate in (
        CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_PENDING, serving_request=True),
        CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_ARMED, serving_request=False),
    ):
        assert not cut.admit_calibration(gate)
    assert cut._pending == [] and cut._start_event is None
    serving = CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_ARMED, serving_request=True)
    widths = []
    for _ in range(CALIBRATION_VERIFIES):
        assert cut.applies(bs=1, sampling_info=None) and not cut.calibrated
        assert cut.admit_calibration(serving)
        lens = cut.verify_lens(confident)
        assert lens.device.type == "cuda" and lens.dtype == torch.int64 and lens.shape == (1,)
        width = int(lens.item())  # the test's own read; the worker never reads it
        widths.append(width)
        # What the static branch brackets: set_live_rows, the graph replay and its lease-gate wait.
        cut.begin_verify()
        for _ in range(width):  # verify time grows with the width, as the live rows' expert reads do
            torch.mm(work, work, out=scratch)
        cut.end_verify()
    assert sorted(widths) == sorted(list(range(1, TOKENS + 1)) * (CALIBRATION_VERIFIES // TOKENS))

    # T(w) freezes at the first verify after the last end event has completed (read lazily, never waited on), so the
    # test lets the 96th verify's work finish before the 97th.
    torch.cuda.synchronize()
    assert cut.admit_calibration(serving)
    lens = cut.verify_lens(confident)
    assert cut.calibrated and not cut.disabled
    cost = cut.cost_ms()
    assert len(cost) == TOKENS and all(c > 0 for c in cost)
    assert all(a <= b for a, b in zip(cost, cost[1:]))
    assert 1 <= int(lens.item()) <= TOKENS

    # A draft the head is sure is wrong is not worth a row past the anchor's.
    doubtful = torch.tensor([[0.01] * GAMMA], device="cuda")
    assert int(cut.verify_lens(doubtful).item()) == 1
    assert not cut.applies(bs=2, sampling_info=None)
    assert not cut.applies(bs=1, sampling_info=SimpleNamespace(is_all_greedy=False))


def test_the_static_epilogue_stops_the_accept_at_the_cuts_width():
    from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync

    epilogue = DsparkVerifyEpilogue(
        max_bs=1, verify_num_draft_tokens=TOKENS, device="cuda", tp_sync=SpecTpSync(SINGLE_RANK), static_cutoff=True
    )
    drafts = [7, 3, 9, 2, 5]
    predict = drafts + [11]  # the target agrees with every draft
    logits = torch.full((TOKENS, VOCAB), -10.0, device="cuda")
    logits[torch.arange(TOKENS), torch.tensor(predict)] = 10.0
    batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([1] + drafts, device="cuda"),
        seq_lens=torch.tensor([100], device="cuda"),
    )

    def accept(cut_lens):
        epilogue.begin_gated_static_step(cut_lens, armed=False)
        epilogue._static_epilogue(SimpleNamespace(next_token_logits=logits), batch)
        out = epilogue.read_accept(1)
        return int(out.correct_len), int(out.bonus), int(out.commit_lens)

    # A width of 2: the anchor and draft 1. Draft 2 matched but was cut, so the bonus is its token.
    assert accept(torch.tensor([2], device="cuda", dtype=torch.int64)) == (1, 3, 2)
    assert accept(torch.tensor([1], device="cuda", dtype=torch.int64)) == (0, 7, 1)
    assert accept(torch.tensor([TOKENS], device="cuda", dtype=torch.int64)) == (5, 11, 6)
