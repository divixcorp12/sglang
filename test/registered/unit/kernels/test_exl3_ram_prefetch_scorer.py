"""The speculative thread's gate scorer (host/gate_scorer.h) against a torch reference of the Phase 0 replay's ranking
(analysis/dsv41-drive/prefetch-replay/verify_replay.py gate_choice and issue's budget): sqrt(softplus(W x)) + b per live
token, each token's top 12 walked in score order past the skipped experts for per_token picks, each pick's margin its
score less the token's top_k-th, the union by margin (ties by id), the first per_layer (CPU)."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEPTH = 12


def reference(x, w, bias, skip, *, top_k, per_token, per_layer, top_k_only=False, meta=False):
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    scores = torch.where(scores.isnan(), torch.tensor(float("-inf")), scores)  # a NaN score ranks below every other
    best, ranks = {}, {}
    experts = w.shape[0]
    for m in range(x.shape[0]):
        s = scores[m]
        order = sorted(range(experts), key=lambda e: (-float(s[e]), e))[: min(DEPTH, experts)]
        walk = order[:top_k] if top_k_only else order
        kth = s[order[top_k - 1]]
        picked = 0
        for rank, e in enumerate(walk):
            if picked >= per_token:
                break
            if skip[e]:
                continue
            # an fp32 subtraction, as the host's; equal scores (also two infinities) give 0, -inf stays above unpicked
            margin = 0.0 if s[e] == kth else max(float(s[e] - kth), -3.4028234663852886e38)
            # an expert's rank is its position in the token that gave its best margin, the lowest on a tie
            if e not in best or margin > best[e] or (margin == best[e] and rank < ranks[e]):
                ranks[e] = rank
            best[e] = max(best.get(e, float("-inf")), margin)
            picked += 1
    chosen = [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:per_layer]
    return [(e, ranks[e], best[e]) for e in chosen] if meta else chosen


def _exact_case(seed, tokens=3, experts=16, hidden=64):
    """Small non-negative integers with a constant 21 term: every logit is an integer above 20, exact in fp32 in any
    summation order, so softplus is the identity and host and torch produce the same fp32 scores bit for bit."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 21
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "top_k, per_token, per_layer", [(6, 1, 1), (6, 1, 3), (6, 2, 4), (6, 3, 8), (2, 1, 2), (1, 2, 5)]
)
def test_the_host_ranks_exactly_as_the_replays_reference(seed, top_k, per_token, per_layer):
    """Mutant: the margin to the (top_k - 1)-th score -- red (the cross-token union order changes)."""
    x, w, bias = _exact_case(seed)
    g = torch.Generator().manual_seed(100 + seed)
    skip = (torch.rand(w.shape[0], generator=g) < 0.25).tolist()
    kw = dict(top_k=top_k, per_token=per_token, per_layer=per_layer)
    assert es.score_gate(x, w, bias, skip, **kw) == reference(x, w, bias, skip, **kw)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("per_token", [1, 2, 12])
def test_tied_scores_and_margins_go_to_the_lower_id(seed, per_token):
    """Experts e and e + 16 share a gate row and there is no bias, so they tie in every token's ranking and in the
    union's margins. Mutants: either sort's id tie-break dropped -- red."""
    x, w, _ = _exact_case(seed, tokens=6, experts=16)
    w = torch.cat([w, w])
    bias = torch.zeros(32)
    skip = [False] * 32
    kw = dict(top_k=6, per_token=per_token, per_layer=8)
    assert es.score_gate(x, w, bias, skip, **kw) == reference(x, w, bias, skip, **kw)


@pytest.mark.parametrize("seed", range(4))
def test_rows_are_read_at_their_stride_not_the_gates_width(seed):
    """x's rows carry padding past the gate's width, as a record's staged rows do: the host reads `hidden` values from
    each row's start. Mutant: the stride taken as 2 * hidden -- red."""
    x, w, bias = _exact_case(seed)
    padded = torch.cat([x, torch.full((x.shape[0], 8), 3.0, dtype=torch.float16)], dim=1)
    kw = dict(top_k=6, per_token=2, per_layer=4)
    assert es.score_gate(padded, w, bias, [False] * 16, **kw) == reference(x, w, bias, [False] * 16, **kw)


@pytest.mark.parametrize("seed", range(4))
def test_a_width_off_the_16_lane_blocks_ranks_as_the_reference(seed):
    """hidden = 70 runs the dot product's scalar tail past its 16-lane blocks."""
    x, w, bias = _exact_case(seed, hidden=70)
    kw = dict(top_k=6, per_token=2, per_layer=4)
    assert es.score_gate(x, w, bias, [False] * 16, **kw) == reference(x, w, bias, [False] * 16, **kw)


def test_skipped_experts_are_passed_over_not_counted():
    x, w, bias = _exact_case(0, tokens=1)
    first = reference(x, w, bias, [False] * 16, top_k=6, per_token=1, per_layer=1)
    skip = [e in first for e in range(16)]
    kw = dict(top_k=6, per_token=1, per_layer=1)
    chosen = es.score_gate(x, w, bias, skip, **kw)
    assert chosen == reference(x, w, bias, skip, **kw)
    assert chosen not in ([], first)


def _separated(x, w, bias, top_k, per_token):
    """True when each token's top-12 scores, and the margins of every token's per_token picks, are at least 1e-4
    apart: the log1p/exp branch may differ from torch's by an ulp, which must not reorder anything."""
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    top = scores.sort(dim=-1, descending=True).values[:, :DEPTH]
    margins = (top[:, :per_token] - top[:, top_k - 1 : top_k]).flatten().sort().values
    return bool((top[:, :-1] - top[:, 1:]).min() > 1e-4) and bool((margins[1:] - margins[:-1]).min() > 1e-4)


def test_the_softplus_branch_ranks_as_the_reference():
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-2, 3, (3, 16), generator=g).to(torch.float16)
        w = torch.randint(-2, 3, (16, 16), generator=g).to(torch.bfloat16)
        bias = torch.randperm(16, generator=g).float() * 1e-2
        if _separated(x, w, bias, 6, 2):
            break
    else:
        pytest.fail("no well-separated case in 200 seeds")
    kw = dict(top_k=6, per_token=2, per_layer=4)
    assert es.score_gate(x, w, bias, [False] * 16, **kw) == reference(x, w, bias, [False] * 16, **kw)


@pytest.mark.parametrize(
    "kw, why",
    [
        (dict(top_k=0, per_token=1, per_layer=1), "top_k"),
        (dict(top_k=13, per_token=1, per_layer=1), "top_k"),
        (dict(top_k=6, per_token=0, per_layer=1), "per_token"),
        (dict(top_k=6, per_token=13, per_layer=1), "per_token"),
        (dict(top_k=6, per_token=1, per_layer=0), "per_layer"),
        (dict(top_k=6, per_token=1, per_layer=9), "per_layer"),
    ],
)
def test_a_choice_outside_the_hosts_bounds_is_refused(kw, why):
    x, w, bias = _exact_case(0)
    with pytest.raises(Exception, match=why):
        es.score_gate(x, w, bias, [False] * 16, **kw)


def _lead_case(seed, tokens=3, experts=16, hidden=16):
    """_exact_case with a lead term of 128, so x may go negative while every logit stays an exact integer above 20."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-2, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 128
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias


KW = dict(top_k=6, per_token=2, per_layer=4)


@pytest.mark.parametrize("seed", range(4))
def test_negative_fp16_inputs_rank_as_the_reference(seed):
    x, w, bias = _lead_case(seed)
    assert (x < 0).any()
    assert es.score_gate(x, w, bias, [False] * 16, **KW) == reference(x, w, bias, [False] * 16, **KW)


@pytest.mark.parametrize("seed", range(4))
def test_subnormal_fp16_inputs_are_not_flushed(seed):
    """x's second column is 2^-24 * a (fp16 subnormals) against w = 2^30 * b: terms of 64 * a * b that reorder the
    experts if the host flushed or mis-scaled a subnormal."""
    x, w, bias = _lead_case(seed)
    g = torch.Generator().manual_seed(50 + seed)
    a = torch.randint(1, 4, (x.shape[0],), generator=g)
    b = torch.randint(0, 4, (w.shape[0],), generator=g)
    x[:, 1] = (a.float() * 2.0**-24).to(torch.float16)
    w[:, 1] = (b.float() * 2.0**30).to(torch.bfloat16)
    assert 0 < float(x[0, 1]) < 2.0**-14
    assert es.score_gate(x, w, bias, [False] * 16, **KW) == reference(x, w, bias, [False] * 16, **KW)


def test_an_infinite_input_ranks_as_the_reference_with_nan_scores_last():
    """x[:, 2] = inf: an expert with w > 0 there scores +inf, one with w == 0 scores NaN (inf * 0) and ranks last."""
    x, w, bias = _lead_case(1)
    x[:, 2] = float("inf")
    w[:, 2] = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.bfloat16)
    chosen = es.score_gate(x, w, bias, [False] * 16, top_k=6, per_token=12, per_layer=8)
    assert chosen == reference(x, w, bias, [False] * 16, top_k=6, per_token=12, per_layer=8)
    assert all(e % 4 != 0 for e in chosen)  # the 4 NaN experts never beat the 12 infinite ones


@pytest.mark.parametrize("seed", range(4))
def test_nan_scores_rank_last_in_id_order_whatever_the_comparison_sort_does(seed):
    """A NaN bias on half the experts, ids shuffled: NaN breaks the sort's strict weak ordering unless it is mapped."""
    x, w, bias = _exact_case(seed, hidden=16)
    g = torch.Generator().manual_seed(200 + seed)
    bias[torch.randperm(16, generator=g)[:8]] = float("nan")
    kw = dict(top_k=3, per_token=12, per_layer=8)
    chosen = es.score_gate(x, w, bias, [False] * 16, **kw)
    assert chosen == reference(x, w, bias, [False] * 16, **kw)
    assert chosen == es.score_gate(x, w, bias, [False] * 16, **kw)


def _raw(x, w, bias, skip, *, top_k=6, per_token=2, per_layer=4):
    """The FFI as a direct caller sees it: x, w as raw bytes, no wrapper to even the strides."""
    out = torch.full((per_layer,), -1, dtype=torch.int64)
    meta = torch.zeros((per_layer, 2), dtype=torch.float32)
    return es._host_module("exl3", None).expert_stream_score_gate(
        x, w, bias, torch.tensor(skip, dtype=torch.uint8), top_k, per_token, per_layer, 0, out, meta
    )


def _bytes(x, w):
    return x.contiguous().view(torch.uint8), w.contiguous().view(torch.uint8)


def test_the_raw_ffi_scores_what_the_wrapper_scores():
    x, w, bias = _exact_case(0)
    xb, wb = _bytes(x, w)
    assert _raw(xb, wb, bias, [False] * 16) == len(es.score_gate(x, w, bias, [False] * 16, **KW))


def test_an_odd_x_row_stride_is_refused():
    x, w, bias = _exact_case(0)
    xb, wb = _bytes(x, w)
    odd = torch.cat([xb, torch.zeros(xb.shape[0], 1, dtype=torch.uint8)], dim=1)
    with pytest.raises(Exception, match="x rows are fp16"):
        _raw(odd, wb, bias, [False] * 16)


def test_x_rows_shorter_than_w_are_refused():
    x, w, bias = _exact_case(0)
    xb, wb = _bytes(x, w)
    with pytest.raises(Exception, match="x rows are shorter"):
        _raw(xb[:, : xb.shape[1] - 2].contiguous(), wb, bias, [False] * 16)


def test_odd_w_rows_are_refused():
    x, w, bias = _exact_case(0)
    xb, wb = _bytes(x, w)
    with pytest.raises(Exception, match="w rows are bf16"):
        _raw(xb, torch.cat([wb, torch.zeros(wb.shape[0], 1, dtype=torch.uint8)], dim=1), bias, [False] * 16)


def test_a_bias_or_skip_of_the_wrong_length_is_refused():
    x, w, bias = _exact_case(0)
    xb, wb = _bytes(x, w)
    with pytest.raises(Exception):
        _raw(xb, wb, bias[:8].contiguous(), [False] * 16)
    with pytest.raises(Exception):
        _raw(xb, wb, bias, [False] * 8)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("top_k, per_token, per_layer", [(6, 1, 1), (6, 2, 4), (6, 3, 8), (2, 1, 2)])
def test_top_k_only_walks_each_tokens_predicted_top_k_alone(seed, top_k, per_token, per_layer):
    """A pick past the token's top_k is one the gate predicts it will not route. Mutant (gate_scorer.h): walk the
    full depth under top_k_only -- red."""
    x, w, bias = _exact_case(seed)
    g = torch.Generator().manual_seed(200 + seed)
    skip = (torch.rand(w.shape[0], generator=g) < 0.5).tolist()
    kw = dict(top_k=top_k, per_token=per_token, per_layer=per_layer, top_k_only=True)
    assert es.score_gate(x, w, bias, skip, **kw) == reference(x, w, bias, skip, **kw)


def test_top_k_only_picks_nothing_when_every_predicted_expert_is_skipped():
    """One token, its top 6 all skipped: the full walk still picks rank 6, top_k_only picks nothing."""
    x, w, bias = _exact_case(0, tokens=1)
    s = reference(x, w, bias, [False] * 16, top_k=6, per_token=6, per_layer=8)
    skip = [e in s[:6] for e in range(16)]
    kw = dict(top_k=6, per_token=1, per_layer=1)
    assert es.score_gate(x, w, bias, skip, **kw) == reference(x, w, bias, skip, **kw) != []
    assert es.score_gate(x, w, bias, skip, top_k_only=True, **kw) == []


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("top_k_only", [False, True])
def test_each_pick_reports_its_rank_and_margin(seed, top_k_only):
    """The rank and margin the speculative thread's spec_submit event carries. Mutants: the rank of the last token
    that picked the expert rather than of its best margin, or a margin off by the kth score -- red."""
    x, w, bias = _exact_case(seed, tokens=4)
    g = torch.Generator().manual_seed(300 + seed)
    skip = (torch.rand(w.shape[0], generator=g) < 0.4).tolist()
    kw = dict(top_k=6, per_token=3, per_layer=8, top_k_only=top_k_only)
    got = es.score_gate(x, w, bias, skip, return_meta=True, **kw)
    assert got == reference(x, w, bias, skip, meta=True, **kw)
    assert all(rank < 6 for _, rank, _ in got) or not top_k_only
