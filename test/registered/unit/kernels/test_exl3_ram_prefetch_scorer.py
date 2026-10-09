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


def reference(x, w, bias, skip, *, top_k, per_token, per_layer):
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    best = {}
    experts = w.shape[0]
    for m in range(x.shape[0]):
        s = scores[m]
        order = sorted(range(experts), key=lambda e: (-float(s[e]), e))[: min(DEPTH, experts)]
        kth = s[order[top_k - 1]]
        picked = 0
        for e in order:
            if picked >= per_token:
                break
            if skip[e]:
                continue
            margin = float(s[e] - kth)  # an fp32 subtraction, as the host's
            best[e] = max(best.get(e, float("-inf")), margin)
            picked += 1
    return [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:per_layer]


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
