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
        cut.begin_draft()
        torch.mm(work, work, out=scratch)  # the draft: one matmul, at every width
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
    # The draft's matmul, timed between the draft's event and the verify's start; one matmul is about T(w)'s slope,
    # so it is well under T(2) and not in the cost with the flag off.
    assert 0 < cut.draft_ms() < cost[1] and cut.verify_cost_ms() == cost
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


def test_the_cut_accept_preload_covers_the_variants_a_cut_verify_runs():
    """preload_cut_accept_kernels loads the capped greedy accept's kernels with the dtypes a cut verify passes, so the
    first cut verify (accept_greedy_triton at bs 1, int64 candidates, float32 logits, int64 verify lens) device-loads
    neither. Mutant: the preload passes another dtype or shape -- the accept then loads a second variant, red."""
    import triton.knobs as knobs

    from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_greedy_triton, preload_cut_accept_kernels

    names = ("_cap_correct_len_kernel", "_gather_row_bonus_kernel")
    loads = []

    def hook(module, function, name, metadata_group, hash):
        loads.append(name)

    knobs.runtime.kernel_load_start_hook.add(hook)
    try:
        preload_cut_accept_kernels(device="cuda", verify_num_draft_tokens=TOKENS)
        torch.cuda.synchronize()
        preloaded = [n for n in loads if n in names]
        loads.clear()
        for width in range(1, TOKENS + 1):
            accept_greedy_triton(
                candidates=torch.zeros((1, TOKENS), dtype=torch.int64, device="cuda"),
                target_logits=torch.randn((TOKENS, VOCAB), dtype=torch.float32, device="cuda"),
                verify_num_draft_tokens=TOKENS,
                cutoff_verify_lens=torch.tensor([width], dtype=torch.int64, device="cuda"),
                fused_argmax=True,
            )
        torch.cuda.synchronize()
    finally:
        knobs.runtime.kernel_load_start_hook.remove(hook)
    print("preloaded", preloaded)
    assert [n for n in loads if n in names] == []


def test_the_sampled_cut_draft_timing_and_online_refit_add_no_device_sync():
    """A calibrated sampled cut verify as the worker runs it -- begin_draft, re-tempered and refitted confidences, the
    stopping rule, the outcome's ring copy, the refit's pick-up -- under torch's sync debug mode set to error: any
    synchronizing call raises. Mutant: a .item()/.tolist() on a device tensor, or a pageable copy, in that path -- red."""
    import concurrent.futures

    from sglang.srt.speculative.dspark_components.dspark_confidence_refit import RING_SLOTS, OnlineRefit
    from sglang.test import dspark_cut_distribution as dist

    class Now:
        def submit(self, fn, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(**kwargs))
            return future

    refit = OnlineRefit(gamma=GAMMA, device="cuda", interval=32, executor=Now())
    cut = dist.calibrated_cut(
        gamma=GAMMA, base_ms=96.5, per_row_ms=18.3, device="cuda", cut_sampled=True, draft_cost=True,
        sampled_rescale=torch.tensor([1.1, 0.9, 1.0, 1.2, 0.8], device="cuda"), refit=refit,
    )
    generator = torch.Generator(device="cuda").manual_seed(5)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3 * RING_SLOTS):
            cut.begin_draft()
            confidence = torch.rand((1, GAMMA), device="cuda", generator=generator)
            cut_confidence = cut.cut_confidence(confidence, sampled=True)
            lens = cut.verify_lens(cut_confidence, sampled=True)
            correct_len = torch.minimum(lens - 1, torch.randint(0, GAMMA + 1, (1,), device="cuda", generator=generator))
            cut.observe_outcome(
                confidence=confidence, verify_lens=lens, correct_len=correct_len.to(torch.int32),
                cap_trim_lens=torch.zeros(1, dtype=torch.int32, device="cuda"), sampled=True,
            )
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert refit._modes["sampled"].refit_ct >= 1 and refit._modes["sampled"].slopes.device.type == "cuda"
    assert len(refit._modes["sampled"].buffer) > RING_SLOTS


# The sampled cut's distribution test on the real accept: AcceptSampling (temperature softmax, the chain rejection
# sampling kernel, CapCorrectLen and the bonus gather) with widths from the real rule (ConfidenceCut.verify_lens,
# sampled). sglang.test.dspark_cut_distribution has the toy; its CPU twin is test_dspark_confidence_cut_sampled.py.
DIST_VOCAB, DIST_LENGTH, DIST_TEMPERATURE = 4, 5, 0.6
DIST_SEQUENCES = 400_000


def _real_sampling_accept(temperature):
    from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptSampling

    def accept(*, candidates, target_probs, draft_probs, verify_lens):
        m, rows, vocab = target_probs.shape
        sampling_info = SimpleNamespace(
            temperatures=torch.full((m, 1), temperature, device="cuda"),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        )
        # Logits whose temperature softmax is the toy's row exactly.
        target_logits = (temperature * torch.log(target_probs.clamp_min(1e-30))).reshape(m * rows, vocab)
        correct_len, bonus, _ = AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=None,
            gamma=rows - 1,
            verify_num_draft_tokens=rows,
            cutoff_verify_lens=verify_lens,
        )
        return correct_len, bonus

    return accept


def _decode_on_gpu(verify_lens_fn, *, seed):
    from sglang.test import dspark_cut_distribution as dist

    toy = dist.MarkovToy(vocab=DIST_VOCAB, gamma=GAMMA, temperature=DIST_TEMPERATURE, seed=0)
    torch.manual_seed(seed)  # AcceptSampling's coins come from the default CUDA generator
    widths = []
    sequences = dist.generate(
        toy,
        n=DIST_SEQUENCES,
        length=DIST_LENGTH,
        verify_lens_fn=verify_lens_fn,
        accept_fn=_real_sampling_accept(DIST_TEMPERATURE),
        generator=torch.Generator(device="cuda").manual_seed(seed),
        device="cuda",
        widths_out=widths,
    )
    torch.cuda.synchronize()
    return toy, sequences, torch.cat(widths)


@pytest.fixture(scope="module")
def sampled_cut_on_gpu():
    from sglang.test import dspark_cut_distribution as dist

    return dist.calibrated_cut(gamma=GAMMA, base_ms=96.5, per_row_ms=18.3, device="cuda", cut_sampled=True)


def test_the_real_sampling_accept_keeps_the_targets_law_with_every_row_and_under_the_cut(sampled_cut_on_gpu):
    """The chi-square p-value of the first 5 tokens of 400k sequences against the exact target law, with every row
    verified and with the cut; the module's 1%-shift check shows the test's power at this size."""
    from sglang.test import dspark_cut_distribution as dist

    full = lambda c, d: torch.full((c.shape[0],), TOKENS, dtype=torch.int64, device="cuda")
    toy, sequences, _ = _decode_on_gpu(full, seed=20)
    p_full = dist.sequence_pvalue(toy, sequences)
    toy, sequences, widths = _decode_on_gpu(lambda c, d: sampled_cut_on_gpu.verify_lens(c, sampled=True), seed=21)
    p_cut = dist.sequence_pvalue(toy, sequences)
    counts = torch.bincount(widths, minlength=TOKENS + 1)
    print(f"p every row {p_full:.4g}, p cut {p_cut:.4g}, widths {counts[1:].tolist()}")
    assert (counts[1:] > 0).all() and widths.float().mean() < GAMMA
    assert p_full > 1e-3 and p_cut > 1e-3


def test_the_real_sampling_accept_shows_the_bias_of_a_cut_that_reads_the_admitted_slots_draft(sampled_cut_on_gpu):
    from sglang.test import dspark_cut_distribution as dist

    mutant = dist.token_at_slot_mutant(lambda c, d: sampled_cut_on_gpu.verify_lens(c, sampled=True), token=1)
    toy, sequences, _ = _decode_on_gpu(mutant, seed=22)
    p_mutant = dist.sequence_pvalue(toy, sequences)
    print(f"p mutant {p_mutant:.4g}")
    assert p_mutant < 1e-6


def test_the_one_percent_shift_is_detected_at_the_gpu_tests_sample_size():
    from sglang.test import dspark_cut_distribution as dist

    toy = dist.MarkovToy(vocab=DIST_VOCAB, gamma=GAMMA, temperature=DIST_TEMPERATURE, seed=0)
    exact = toy.exact_sequence_probs(DIST_LENGTH)
    top2 = torch.topk(exact, 2).indices
    shifted = exact.clone()
    shifted[top2[0]] -= 0.01
    shifted[top2[1]] += 0.01
    counts = torch.bincount(
        torch.multinomial(shifted.float(), DIST_SEQUENCES, replacement=True,
                          generator=torch.Generator().manual_seed(23)),
        minlength=exact.numel(),
    )
    assert dist.chi_square_pvalue(counts=counts, probs=exact) < 1e-3
