"""The GPU scorer's kernels (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design, "The scoring kernels") on a real
GPU: their candidates against the host scorer's torch reference (gate_reference), the oversize and lapped slots, and the
seq word stored last. Task 4 adds a captured post with its scoring replayed through the production row backend.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import random

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import EXPERTS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.srt.layers.moe.ram_prefetch import SpecScoreRow  # noqa: E402
from sglang.test.dsv41_ram_prefetch_fixtures import exact_gate_case, gate_reference, lead_gate_case  # noqa: E402

KEEP = es.CAND_MAX  # a reference per_layer that lists every candidate the GPU writes
BF16 = torch.bfloat16


def _candidates(x, w, bias, *, top_k, per_token, top_k_only=False, hot=(), mapped=(), seq=1, tokens_max=None,
                page=None, scores=None, module=None):
    """Scores `x` (bf16 [tokens, hidden]) against (w, bias) for target row 1 of a two-row map, with `hot` VRAM-resident
    and `mapped` mapped in RAM there, and returns record `seq`'s slot."""
    experts, tokens = w.shape[0], x.shape[0]
    tokens_max = tokens_max or max(1, tokens)
    if scores is None:
        scores = torch.empty((tokens_max, experts), dtype=torch.float32, device="cuda")
    page = es.new_candidate_page(pin=True) if page is None else page
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device="cuda")
    state[es.STATE_WORDS["posted"]] = seq
    hot_slots = torch.full((max(1, len(hot)),), -1, dtype=torch.int64, device="cuda")
    if hot:
        hot_slots[: len(hot)] = torch.tensor(list(hot), dtype=torch.int64)
    ram_slot = torch.full((2, experts), -1, dtype=torch.int32, device="cuda")
    for slot, e in enumerate(mapped):
        ram_slot[1, e] = slot
    if 1 <= tokens <= tokens_max:
        es.run_spec_score(x.cuda(), w.cuda(), bias.cuda(), scores, module=module)
    es.run_spec_select(
        scores, tokens, top_k=top_k, per_token=per_token, top_k_only=top_k_only, hot_slots=hot_slots,
        hot_capacity=len(hot), ram_slot=ram_slot, target=1, state=state, candidates=page, module=module,
    )
    torch.cuda.synchronize()
    return es.read_candidates(page, seq)


def _skip(experts, hot=(), mapped=()):
    return [e in hot or e in mapped for e in range(experts)]


COMBOS = [(6, 1, 1), (6, 1, 3), (6, 2, 4), (6, 3, 8), (2, 1, 2), (1, 2, 5)]


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k_only", [False, True])
@pytest.mark.parametrize("top_k, per_token, per_layer", COMBOS)
@pytest.mark.parametrize("seed", range(8))
def test_the_gpu_ranks_exactly_as_the_host_reference(seed, top_k, per_token, per_layer, top_k_only, w_dtype):
    """Experts, ranks and margins bit for bit, hot and mapped experts skipped; the first per_layer are the CPU scorer's
    per_layer. Mutant: the margin to the (top_k - 1)-th score -- red."""
    x, w, bias = exact_gate_case(seed)
    order = torch.randperm(16, generator=torch.Generator().manual_seed(100 + seed)).tolist()
    hot, mapped = order[:2], order[2:4]
    got = _candidates(x.to(BF16), w.to(w_dtype), bias, top_k=top_k, per_token=per_token, top_k_only=top_k_only,
                      hot=hot, mapped=mapped)
    kw = dict(top_k=top_k, per_token=per_token, top_k_only=top_k_only, meta=True)
    skip = _skip(16, hot, mapped)
    assert got.status == "ready" and got.flags == 0
    assert list(got.picks) == gate_reference(x, w, bias, skip, per_layer=KEEP, **kw)
    assert list(got.picks[:per_layer]) == gate_reference(x, w, bias, skip, per_layer=per_layer, **kw)


@pytest.mark.parametrize("per_token", [1, 2, 12])
@pytest.mark.parametrize("seed", range(4))
def test_tied_scores_and_margins_go_to_the_lower_id(seed, per_token):
    """Experts e and e + 16 share a gate row and there is no bias. Mutants: either id tie-break dropped -- red."""
    x, w, _ = exact_gate_case(seed, tokens=6)
    w = torch.cat([w, w])
    bias = torch.zeros(32)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=per_token)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 32, top_k=6, per_token=per_token, per_layer=KEEP,
                                             meta=True)


@pytest.mark.parametrize("seed", range(4))
def test_negative_inputs_rank_as_the_reference(seed):
    x, w, bias = lead_gate_case(seed)
    assert (x < 0).any()
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=2)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=2, per_layer=KEEP, meta=True)


def test_an_infinite_input_ranks_as_the_reference():
    """Review Focus 4. x[:, 2] = inf: an expert with w > 0 there scores +inf, one with w == 0 scores NaN (inf * 0) and
    ranks last."""
    x, w, bias = lead_gate_case(1)
    x[:, 2] = float("inf")
    w[:, 2] = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.bfloat16)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=12)
    want = gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=12, per_layer=KEEP, meta=True)
    assert list(got.picks) == want and all(e % 4 != 0 for e, _, _ in got.picks)


@pytest.mark.parametrize("seed", range(4))
def test_nan_scores_rank_last_in_id_order(seed):
    """Review Focus 4. A NaN bias on half the experts. Mutant: store the NaN score unmapped -- red."""
    x, w, bias = exact_gate_case(seed, hidden=16)
    bias[torch.randperm(16, generator=torch.Generator().manual_seed(200 + seed))[:8]] = float("nan")
    got = _candidates(x.to(BF16), w, bias, top_k=3, per_token=12)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=3, per_token=12, per_layer=KEEP,
                                             meta=True)


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
def test_on_random_data_the_top_candidate_agrees_with_torch_in_99_percent_of_records(w_dtype):
    """DSV4's shape (384 experts, hidden 7168), 1-6 tokens: the softplus branch and fp32 sums in another order may
    move a near-tie, so agreement, not identity."""
    g = torch.Generator().manual_seed(7)
    experts, hidden, trials = 384, 7168, 200
    w = (torch.randn((experts, hidden), generator=g) * 0.02).to(w_dtype)
    scores = torch.empty((6, experts), dtype=torch.float32, device="cuda")
    page, agree = es.new_candidate_page(pin=True), 0
    for trial in range(trials):
        x = torch.randn((1 + trial % 6, hidden), generator=g).to(BF16)
        bias = torch.randn(experts, generator=g) * 0.1
        got = _candidates(x, w, bias, top_k=6, per_token=1, seq=trial + 1, tokens_max=6, page=page, scores=scores)
        want = gate_reference(x, w, bias, [False] * experts, top_k=6, per_token=1, per_layer=1)
        agree += got.picks[0][0] == want[0]
    assert agree >= 0.99 * trials


@pytest.mark.parametrize("tokens", [0, 3])
def test_a_record_outside_one_to_tokens_max_tokens_publishes_count_zero(tokens):
    """Prefill (more tokens than the rows hold) or a post without input: count 0, the oversize flag, nothing scored."""
    x, w, bias = exact_gate_case(0, tokens=max(tokens, 1))
    got = _candidates(x[:tokens].to(BF16), w, bias, top_k=6, per_token=1, tokens_max=2)
    assert (got.status, got.count, got.flags, got.picks) == ("ready", 0, es.CAND_FLAG_OVERSIZE, ())


def test_a_record_with_fewer_tokens_ranks_only_its_own_rows_of_the_scratch():
    """Review Focus 3. A three-token record fills the scratch, then a one-token record rewrites row 0 only. Mutant:
    rank tokens_max rows -- red."""
    x3, w, bias = exact_gate_case(1, tokens=3)
    x1 = exact_gate_case(2, tokens=1)[0]
    kw = dict(top_k=6, per_token=2, per_layer=KEEP, meta=True)
    want = gate_reference(x1, w, bias, [False] * 16, **kw)
    stale = gate_reference(torch.cat([x1, x3[1:]]), w, bias, [False] * 16, **kw)
    assert want != stale, "pick seeds whose stale rows change the ranking"
    scores = torch.empty((3, 16), dtype=torch.float32, device="cuda")
    page = es.new_candidate_page(pin=True)
    _candidates(x3.to(BF16), w, bias, top_k=6, per_token=2, seq=1, tokens_max=3, page=page, scores=scores)
    got = _candidates(x1.to(BF16), w, bias, top_k=6, per_token=2, seq=2, tokens_max=3, page=page, scores=scores)
    assert list(got.picks) == want


def test_seventeen_records_lap_the_first_slot_and_the_seqlock_shows_it():
    x, w, bias = exact_gate_case(0, tokens=1)
    page = es.new_candidate_page(pin=True)
    scores = torch.empty((1, 16), dtype=torch.float32, device="cuda")
    for seq in range(1, 18):
        _candidates(x.to(BF16), w, bias, top_k=6, per_token=1, seq=seq, page=page, scores=scores)
    assert es.read_candidates(page, 1).status == "lapped"
    assert es.read_candidates(page, 17).status == "ready" and es.read_candidates(page, 2).status == "ready"
    assert es.read_candidates(page, 18).status == "not_yet"


def test_the_seq_word_is_stored_last():
    """With the closing store compiled out (EXL3_RAM_MISS_TEST_SPEC_NO_SEQ), a slot that already held this record's
    seq reads as open, its payload written: the kernel opens the slot (seq 0, a release fence) before any payload store,
    and only the final release publishes it. Mutant: drop the opening store -- red (the stale seq reads ready)."""
    module = es.device_module_with_hooks(["EXL3_RAM_MISS_TEST_SPEC_NO_SEQ"])
    x, w, bias = exact_gate_case(0, tokens=1)
    page = es.new_candidate_page(pin=True)
    off = es.candidate_offset(5)
    page[off : off + 4] = torch.tensor([5], dtype=torch.int32).view(torch.uint8)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=1, seq=5, page=page, module=module)
    assert got.status == "not_yet"
    assert int(page[off : off + 4].view(torch.int32)[0]) == 0 and int(page[off + 4 : off + 6].view(torch.int16)[0]) == 1


def _step(c, experts, row=0):
    c.plan(experts, row)
    c.gather(row)
    snapshot = c.snapshot(row)
    torch.cuda.synchronize()
    c.check(experts, snapshot, row)


def test_a_captured_post_and_its_scoring_replay_the_reference_candidates(tmp_path):
    """The production row backend's post with its scoring, captured once, replayed with new inputs, hot sets and
    plans twice round the 16-slot ring: each replay's slot holds the reference's candidates for that replay's x past
    its hot slots and row 1's device map. Mutant: score a copy of x taken at capture -- red."""
    c = Chain(tmp_path)
    try:
        x16, w16, bias = exact_gate_case(0, tokens=1, experts=EXPERTS)
        x = x16.to(BF16).cuda()
        hot = torch.full((4,), -1, dtype=torch.int64, device="cuda")
        page = es.new_candidate_page(pin=True)
        c.dev.enable_spec_scorer(page, top_k=TOP_K, per_token=2, top_k_only=False)
        backend = c.backends[0]
        backend.spec_expected = True
        backend.spec_score = SpecScoreRow(w16.cuda(), bias.cuda(), 1, hot, 4)
        backend.cpu_input = (x, None)  # the scoring reads x alone: this rig's rows have no CPU experts
        _step(c, [0, 1], 0)  # loads both kernels and scores eagerly
        _step(c, [2, 3], 1)
        _step(c, [2, 3], 1)  # applies row 1's delta: 2 and 3 mapped in the device map
        assert c.handled()
        mapped = {e for e, slot in enumerate(c.device_map(1)) if slot >= 0}
        assert {2, 3} <= mapped
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            c.gather(0)
        rng = random.Random(11)
        for i in range(2 * es.CAND_RECORDS + 1):
            xi = exact_gate_case(100 + i, tokens=1, experts=EXPERTS)[0]
            hot_i = rng.sample(range(EXPERTS), 4)
            with torch.cuda.stream(stream):
                x.copy_(xi.to(BF16))
                hot.copy_(torch.tensor(hot_i, dtype=torch.int64))
                c.plan(rng.sample(range(EXPERTS), rng.randint(1, TOP_K)), 0)
                graph.replay()
            stream.synchronize()
            seq = int(c.dev.stats()["posted"]) & 0xFFFFFFFF
            skip = [e in hot_i or e in mapped for e in range(EXPERTS)]
            want = gate_reference(xi, w16, bias, skip, top_k=TOP_K, per_token=2, per_layer=KEEP, meta=True)
            got = es.read_candidates(page, seq)
            assert got.status == "ready" and list(got.picks) == want, i
        assert c.handled()
    finally:
        c.close()


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [60, 62, 64, 1032])
def test_hidden_sizes_off_the_vector_width_rank_as_the_reference(hidden, w_dtype):
    """The score kernel's vector loads cover hidden // width chunks and a scalar loop the rest; 62 is off both widths."""
    x, w, bias = exact_gate_case(3, hidden=hidden)
    got = _candidates(x.to(BF16), w.to(w_dtype), bias, top_k=6, per_token=2)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=2, per_layer=KEEP, meta=True)


def test_an_input_off_16_byte_alignment_ranks_as_the_reference():
    """x two bytes into its allocation: the kernel must not take its 16-byte vector path."""
    x, w, bias = exact_gate_case(4)
    flat = torch.empty(x.numel() + 1, dtype=BF16, device="cuda")
    shifted = flat[1:].view(x.shape)
    shifted.copy_(x.to(BF16))
    assert shifted.data_ptr() % 16 != 0
    got = _candidates(shifted, w, bias, top_k=6, per_token=2)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=2, per_layer=KEEP, meta=True)


@pytest.mark.parametrize("tokens", [9, 17])
def test_records_wider_than_one_token_tile_rank_as_the_reference(tokens):
    """More tokens than the kernel accumulates per pass over a gate row: each pass reuses the block's partial sums."""
    x, w, bias = exact_gate_case(5, tokens=tokens)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=1, tokens_max=tokens)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=1, per_layer=KEEP, meta=True)
