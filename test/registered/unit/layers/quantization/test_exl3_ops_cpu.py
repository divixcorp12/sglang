"""exl3_moe_loop against the reference Expert math, with a dense stand-in linear."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_moe_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, INTER, EXPERTS, TOPK, TOKENS, LIMIT = 32, 16, 5, 2, 7, 10.0


def _fake(in_f, out_f, seed):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(in_f, out_f, generator=g)
    # The loop only hands Exl3Tensors to `linear`; the fake linear reads `dense`.
    t = Exl3Tensors(
        trellis=torch.zeros(in_f // 16, out_f // 16, 48, dtype=torch.int16),
        suh=torch.ones(in_f, dtype=torch.float16),
        svh=torch.ones(out_f, dtype=torch.float16),
        mul1=True,
    )
    return t, dense


def test_loop_matches_reference_expert_math():
    torch.manual_seed(0)
    dense = {}
    w13, w2 = [], []
    for e in range(EXPERTS):
        (g, gd), (u, ud), (d, dd) = (
            _fake(HIDDEN, INTER, 3 * e),
            _fake(HIDDEN, INTER, 3 * e + 1),
            _fake(INTER, HIDDEN, 3 * e + 2),
        )
        dense[id(g)], dense[id(u)], dense[id(d)] = gd, ud, dd
        w13.append((g, u))
        w2.append(d)

    def linear(x, t, out_dtype=None):
        return (x.float() @ dense[id(t)]).to(out_dtype or x.dtype)

    x = torch.randn(TOKENS, HIDDEN)
    topk_ids = torch.tensor([[0, 3], [3, 1], [4, 4], [2, 0], [1, 1], [0, 0], [3, 2]])
    topk_weights = torch.rand(TOKENS, TOPK)

    got = exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, LIMIT, linear=linear)

    want = torch.zeros(TOKENS, HIDDEN)
    for t in range(TOKENS):
        for k in range(TOPK):
            e = int(topk_ids[t, k])
            gate = (x[t] @ dense[id(w13[e][0])]).clamp(max=LIMIT)
            up = (x[t] @ dense[id(w13[e][1])]).clamp(-LIMIT, LIMIT)
            h = F.silu(gate) * up * topk_weights[t, k]
            want[t] += h @ dense[id(w2[e])]
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-4)


def test_loop_without_limit_does_not_clamp():
    x = torch.full((1, HIDDEN), 3.0)
    t_g, gd = _fake(HIDDEN, INTER, 1)
    t_u, ud = _fake(HIDDEN, INTER, 2)
    t_d, dd = _fake(INTER, HIDDEN, 3)
    table = {id(t_g): gd * 100, id(t_u): ud * 100, id(t_d): dd}

    def linear(x, t, out_dtype=None):
        return (x.float() @ table[id(t)]).to(out_dtype or x.dtype)

    got = exl3_moe_loop(
        x, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), [(t_g, t_u)], [t_d], None, linear=linear
    )
    gate, up = x @ table[id(t_g)], x @ table[id(t_u)]
    assert torch.allclose(got, (F.silu(gate) * up) @ dd, rtol=1e-5)


def test_loop_zero_limit_does_not_clamp():
    # Matches the reference and LazyExpert: swiglu_limit=0.0 means "no limit",
    # same as None, not "clamp to zero" (exl3_ops.py exl3_moe_loop).
    x = torch.full((1, HIDDEN), 3.0)
    t_g, gd = _fake(HIDDEN, INTER, 1)
    t_u, ud = _fake(HIDDEN, INTER, 2)
    t_d, dd = _fake(INTER, HIDDEN, 3)
    table = {id(t_g): gd * 100, id(t_u): ud * 100, id(t_d): dd}

    def linear(x, t, out_dtype=None):
        return (x.float() @ table[id(t)]).to(out_dtype or x.dtype)

    got = exl3_moe_loop(
        x, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), [(t_g, t_u)], [t_d], 0.0, linear=linear
    )
    gate, up = x @ table[id(t_g)], x @ table[id(t_u)]
    assert torch.allclose(got, (F.silu(gate) * up) @ dd, rtol=1e-5)


def test_tensors_shape_properties():
    t = Exl3Tensors(
        trellis=torch.zeros(320, 144, 48, dtype=torch.int16),
        suh=torch.ones(5120, dtype=torch.float16),
        svh=torch.ones(2304, dtype=torch.float16),
        mul1=True,
    )
    assert (t.in_features, t.out_features, t.bits) == (5120, 2304, 3)


def test_tensors_reject_wrong_dtype():
    with pytest.raises(ValueError, match="trellis"):
        Exl3Tensors(
            trellis=torch.zeros(1, 8, 48, dtype=torch.int32),
            suh=torch.ones(16, dtype=torch.float16),
            svh=torch.ones(128, dtype=torch.float16),
            mul1=True,
        )


def test_moe_loop_raises_under_cuda_graph_capture(monkeypatch):
    # exl3_moe_loop makes host syncs (.tolist(), torch.where) and only works
    # eagerly; under capture it must fail loudly instead of hitting an opaque
    # CUDA "operation not permitted when stream is capturing" error.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    t, _ = _fake(HIDDEN, INTER, 0)
    x = torch.randn(1, HIDDEN)
    topk_ids = torch.zeros(1, 1, dtype=torch.int64)
    topk_weights = torch.ones(1, 1)
    with pytest.raises(RuntimeError, match="disable-cuda-graph"):
        exl3_moe_loop(x, topk_weights, topk_ids, [(t, t)], [t], LIMIT, linear=lambda x, t, out_dtype=None: x)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
