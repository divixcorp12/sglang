"""exl3_moe_loop with real kernels against a dense per-expert computation."""

import os

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.exl3_ops import (
    exl3_dense_weight,
    exl3_moe_loop,
    random_exl3_tensors,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)


@pytest.mark.parametrize("tokens", [1, 6, 200])
def test_loop_matches_dense_experts(tokens):
    experts, hidden, inter, topk, limit = 8, 5120, 2304, 6, 10.0
    w13 = [
        (
            random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e),
            random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e + 1),
        )
        for e in range(experts)
    ]
    w2 = [random_exl3_tensors(inter, hidden, 3, device="cuda", seed=3 * e + 2) for e in range(experts)]
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    topk_ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)])
    topk_weights = torch.rand(tokens, topk, device="cuda")
    got = exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, limit).float()

    want = torch.zeros(tokens, hidden, device="cuda")
    for e in range(experts):
        tok, slot = torch.where(topk_ids == e)
        if tok.numel() == 0:
            continue
        xe = x[tok].half().float()
        gate = (xe @ exl3_dense_weight(w13[e][0]).float()).clamp(max=limit)
        up = (xe @ exl3_dense_weight(w13[e][1]).float()).clamp(-limit, limit)
        h = (F.silu(gate) * up * topk_weights[tok, slot, None]).to(torch.bfloat16).half().float()
        want.index_add_(0, tok, h @ exl3_dense_weight(w2[e]).float())
    assert float((got - want).norm() / want.norm()) < 2e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
