"""spec_head's serving launch on a real GPU: it publishes the offline record's read into the post's candidate-page slot,
seq stored last, with the slot read from the device state so a captured graph follows ``state[posted]``.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.kernels.ops.moe import spec_head  # noqa: E402

DEV = "cuda"
E, H, PCA, DIM, ROWS, TOP_K = 64, 128, 16, 64, 2, 6
AGG_HIDDEN, BLOCK_FEATURES = 32, 10
RECORD_HEAD = slice(4, 16)  # count, flags, then entry 0: expert, rank, pad, margin
POSTED = es.STATE_WORDS["posted"]


def _weights(seed: int = 0) -> spec_head.SpecHeadWeights:
    g = torch.Generator().manual_seed(seed)

    def mat(*shape):
        return (torch.randn(shape, generator=g) * 0.1).bfloat16().to(DEV)

    def vec(*shape):
        return (torch.randn(shape, generator=g) * 0.1).float().to(DEV)

    return spec_head.SpecHeadWeights(
        w_pca=mat(PCA, H), w_in_pca=mat(DIM, PCA), b_in=vec(DIM), w_in_sparse=mat(4, E, DIM),
        row_bias=vec(ROWS, DIM), pos_bias=torch.empty(0, dtype=torch.float32, device=DEV),
        w_r1=mat(DIM, DIM), b_r1=vec(DIM), w_r2=mat(DIM, DIM), b_r2=vec(DIM),
        w_gate=mat(DIM, PCA), b_gate=vec(DIM), w_out=mat(E, DIM), b_out=vec(E),
        agg_w1=mat(AGG_HIDDEN, BLOCK_FEATURES), agg_row_bias=vec(ROWS, AGG_HIDDEN), agg_w2=mat(AGG_HIDDEN),
        agg_b2=vec(1), rows=ROWS,
    )  # fmt: skip


class _Inputs:
    def __init__(self, tokens: int, seed: int = 1) -> None:
        g = torch.Generator().manual_seed(seed)
        self.tokens = tokens
        self.x = torch.randn((tokens, H), generator=g).bfloat16().to(DEV)
        self.scores = torch.randn((spec_head.HEAD_MAX_TOKENS, E), generator=g).to(DEV)
        self.src_ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(tokens)]).to(DEV)
        self.src_w = torch.rand((tokens, TOP_K), generator=g).to(DEV)
        self.positions = torch.arange(spec_head.HEAD_MAX_TOKENS, dtype=torch.int64, device=DEV)[:tokens]
        self.hot_slots = torch.full((4,), -1, dtype=torch.int64, device=DEV)
        self.ram_slot = torch.full((2, E), -1, dtype=torch.int32, device=DEV)
        self.threshold = torch.zeros(1, dtype=torch.float32, device=DEV)

    def common(self, weights, workspace) -> dict:
        return dict(
            weights=weights, workspace=workspace, scores=self.scores, src_ids=self.src_ids, src_w=self.src_w,
            positions=self.positions, row=1, hot_slots=self.hot_slots, hot_capacity=4, ram_slot=self.ram_slot,
            target=1, threshold=self.threshold,
        )  # fmt: skip


def _state(posted: int) -> torch.Tensor:
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device=DEV)
    state[POSTED] = posted
    return state


def _slot(page: torch.Tensor, seq: int) -> bytes:
    off = es.candidate_offset(seq)
    return bytes(page[off : off + es.CAND_STRIDE].tolist())


def _serve(c: _Inputs, weights, workspace, *, x, tokens, state, page, picks: int = 1) -> None:
    spec_head.run_spec_head_serving(
        x=x, tokens=tokens, state=state, candidates=page, picks=picks, **c.common(weights, workspace)
    )


def test_two_picks_publish_the_two_best_eligible_blocks_in_order():
    """picks=2 publishes the best two blocks past the RAM-mapped and VRAM-hot experts and at or above the threshold,
    best first, rank = order. Mutant: the second round re-picks the first block, skips no taken block, ignores the
    skips or the threshold, or publishes count 2 when only one block clears it -- red."""
    weights, c = _weights(), _Inputs(4)
    page = es.new_candidate_page(pin=True)
    ws = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    _serve(c, weights, ws, x=c.x, tokens=4, state=_state(1), page=page)
    torch.cuda.synchronize()
    scores = ws.block_scores.cpu()
    order = sorted(range(E), key=lambda e: (-scores[e].item(), e))
    c.ram_slot[1, order[0]] = 0  # mapped in RAM on the target row
    c.hot_slots[0] = order[1]  # VRAM hot
    c.threshold.fill_(scores[order[3]].item())  # the fourth best sits exactly at the threshold: read
    _serve(c, weights, ws, x=c.x, tokens=4, state=_state(2), page=page, picks=2)
    torch.cuda.synchronize()
    got = es.read_candidates(page, 2)
    assert got.count == 2
    assert [(e, r) for e, r, _ in got.picks] == [(order[2], 0), (order[3], 1)]
    assert [m for _, _, m in got.picks] == [scores[order[2]].item(), scores[order[3]].item()]
    above = torch.nextafter(scores[order[3]], torch.tensor(float("inf"))).item()
    c.threshold.fill_(above)  # now only the third best clears it
    _serve(c, weights, ws, x=c.x, tokens=4, state=_state(3), page=page, picks=2)
    torch.cuda.synchronize()
    got = es.read_candidates(page, 3)
    assert got.count == 1 and got.picks[0][0] == order[2]


@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_serving_slot_equals_the_offline_record(tokens):
    """The slot's count, flags, expert, rank and margin bytes are run_spec_head's record's. Mutant: the margin written
    from the wrong score (a token logit, or the score of a different expert) -- red."""
    weights, c = _weights(), _Inputs(tokens)
    offline = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    spec_head.run_spec_head(x=c.x, **c.common(weights, offline))
    page, posted = es.new_candidate_page(pin=True), 3
    serving = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    _serve(c, weights, serving, x=c.x, tokens=tokens, state=_state(posted), page=page)
    torch.cuda.synchronize()
    want = bytes(offline.record.cpu().tolist())
    got = _slot(page, posted)
    assert want[4] == 1  # threshold 0: a pick, so the margin bytes are compared
    assert got[RECORD_HEAD] == want[RECORD_HEAD]
    block = spec_head.read_record(offline)[2]
    assert es.read_candidates(page, posted).picks[0][2] == pytest.approx(block, abs=0)


@pytest.mark.parametrize("posted", [1, 16, 17, 35])
def test_seq_is_posted_and_lands_in_the_ring_slot(posted):
    """The seq word is state[posted] and only slot (posted - 1) % kCandRecords is written, across a ring wrap.
    Mutant: slot taken from a host count or from posted % kCandRecords -- red."""
    weights, c = _weights(), _Inputs(3)
    page = es.new_candidate_page(pin=True)
    ws = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    _serve(c, weights, ws, x=c.x, tokens=3, state=_state(posted), page=page)
    torch.cuda.synchronize()
    written = [s for s in range(es.CAND_RECORDS) if any(page[s * es.CAND_STRIDE : (s + 1) * es.CAND_STRIDE].tolist())]
    assert written == [(posted - 1) % es.CAND_RECORDS]
    assert es.read_candidates(page, posted).status == "ready"
    assert int.from_bytes(_slot(page, posted)[:4], "little") == posted


@pytest.mark.parametrize("case", ["no_x", "zero_tokens", "nine_tokens"])
def test_a_record_without_a_live_x_publishes_count_zero_oversize(case):
    """No x or tokens outside 1..8 give count 0 with kCandFlagOversize, as select does for an oversize record, and
    overwrite the slot's older payload. Mutant: the empty launch skipping the flag or the stale count -- red."""
    weights, c = _weights(), _Inputs(4)
    page = es.new_candidate_page(pin=True)
    ws = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    _serve(c, weights, ws, x=c.x, tokens=4, state=_state(2), page=page)  # an older pick in the slot
    torch.cuda.synchronize()
    assert es.read_candidates(page, 2).count == 1
    x, tokens = {"no_x": (None, 4), "zero_tokens": (c.x, 0), "nine_tokens": (c.x, 9)}[case]
    _serve(c, weights, ws, x=x, tokens=tokens, state=_state(2), page=page)
    torch.cuda.synchronize()
    got = es.read_candidates(page, 2)
    assert (got.status, got.count, got.flags, got.picks) == ("ready", 0, es.CAND_FLAG_OVERSIZE, ())


def test_graph_replay_follows_state_posted():
    """A captured launch reads its slot from the device state: replay after state[posted] changes publishes into the
    new slot. Mutant: the slot or seq taken as a host argument at capture time -- red."""
    weights, c = _weights(), _Inputs(4)
    page = es.new_candidate_page(pin=True)
    ws = spec_head.new_workspace(device=DEV, pca_rank=PCA, hidden=DIM, experts=E)
    state = _state(1)
    _serve(c, weights, ws, x=c.x, tokens=4, state=state, page=page)  # warm: the JIT build precedes capture
    torch.cuda.synchronize()
    want = _slot(page, 1)[RECORD_HEAD]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _serve(c, weights, ws, x=c.x, tokens=4, state=state, page=page)
    for posted in (5, 18):
        state[POSTED] = posted
        graph.replay()
        torch.cuda.synchronize()
        assert es.read_candidates(page, posted).status == "ready"
        assert _slot(page, posted)[RECORD_HEAD] == want
