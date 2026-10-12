"""r1_gate: full gate recompute, native margins and top-12 parity on small synthetic gates."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_gate  # noqa: E402
from b1_rows import GATE_DEPTH, Group  # noqa: E402

E, H, N = 384, 16, 5


def _gate(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, H, generator=g)
    W = torch.randn(E, H, generator=g) * 0.3
    bias = torch.randn(E, generator=g) * 0.1
    return x, W, bias


def _group(scores: torch.Tensor, *, has_bias: bool = True) -> Group:
    top = torch.topk(scores, GATE_DEPTH, dim=-1)
    n = len(scores)
    return Group(
        session="s", split="dev", block="b", seq=0, rid="r", warm_start=False, flagged=False, horizon=1,
        source_row=0, source_layer=1, source_stratum="a", gate_has_bias=has_bias, target_row=1, target_layer=2,
        target_stratum="b", tokens=np.arange(n, dtype=np.int16), x_offset=0,
        source_ids=np.zeros((n, 6), np.int16), source_weights=np.zeros((n, 6), np.float32),
        target_ids=np.zeros((n, 6), np.int16), target_kinds=np.zeros((n, 6), np.uint8),
        ram_mask=np.zeros(6, np.uint64), hot=np.zeros(0, np.int16),
        gate_ids=top.indices.numpy().astype(np.int16), gate_scores=top.values.numpy().astype(np.float32),
    )


def test_scores_match_formula_and_margins_zero_at_sixth():
    x, W, bias = _gate()
    s = r1_gate.full_gate_scores(x, W=W, bias=bias)
    assert s.shape == (N, E)
    assert torch.allclose(s, torch.nn.functional.softplus(x @ W.T).sqrt() + bias)
    m = r1_gate.native_margins(s)
    order = torch.argsort(s, dim=-1, descending=True)
    assert torch.all(m.gather(1, order[:, 5:6]) == 0)
    assert torch.all(m.gather(1, order[:, :5]) >= 0)


def test_parity_exact():
    x, W, bias = _gate()
    s = r1_gate.full_gate_scores(x, W=W, bias=bias)
    p = r1_gate.top12_parity(_group(s), scores=s)
    assert (p.tokens, p.exact, p.tie_only, p.mismatched) == (N, N, 0, 0)


def test_parity_tie_at_12th_and_real_mismatch():
    x, W, bias = _gate()
    s = r1_gate.full_gate_scores(x, W=W, bias=bias).clone()
    order = torch.argsort(s[0], descending=True)
    s[0, order[12]] = s[0, order[11]]  # exact tie; lower id wins in the recompute
    g = _group(s)
    stored = g.gate_ids.copy()
    lo, hi = sorted((int(order[11]), int(order[12])))
    stored[0, np.nonzero(stored[0] == order[11])[0][0]] = hi if lo == int(order[11]) else lo
    g = Group(**{**{f: getattr(g, f) for f in g.__struct_fields__}, "gate_ids": stored})
    p = r1_gate.top12_parity(g, scores=s)
    assert p.tokens == N and p.mismatched == 0 and p.tie_only + p.exact == N

    bad = g.gate_ids.copy()
    bad[1, 0] = int(torch.argsort(s[1])[0])  # the lowest-scoring expert
    g2 = Group(**{**{f: getattr(g, f) for f in g.__struct_fields__}, "gate_ids": bad})
    assert r1_gate.top12_parity(g2, scores=s).mismatched == 1


def test_no_bias_group_has_no_tokens():
    x, W, bias = _gate()
    s = r1_gate.full_gate_scores(x, W=W, bias=bias)
    assert r1_gate.top12_parity(_group(s, has_bias=False), scores=s).tokens == 0
