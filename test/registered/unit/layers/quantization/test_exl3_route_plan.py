"""Exl3RoutePlan: a layer's routes grouped by expert on the host, identical to the per-expert torch.where loop."""

import pytest
import torch

from sglang.srt.layers.quantization import exl3_ops
from sglang.srt.layers.quantization.exl3_ops import Exl3RoutePlan, random_exl3_tensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, INTER = 64, 32


def _fake_linear(x, t, out_dtype=None):
    """A CPU stand-in for exl3_linear: each output row depends on its input row and on its expert's tensors."""
    scale = 1.0 + t.trellis.float().mean() / 32768 + t.suh.float().mean()
    y = x.float().sum(-1, keepdim=True) * t.svh.float() * scale
    return y.to(out_dtype or x.dtype)


def _weights(num_experts):
    def t(i, o, seed):
        return random_exl3_tensors(i, o, 3, device="cpu", seed=seed)

    w13 = [(t(HIDDEN, INTER, 3 * e), t(HIDDEN, INTER, 3 * e + 1)) for e in range(num_experts)]
    w2 = [t(INTER, HIDDEN, 3 * e + 2) for e in range(num_experts)]
    return w13, w2


CASES = {
    "distinct": torch.tensor([[5, 0, 3], [3, 1, 5], [0, 4, 1], [5, 3, 4]], dtype=torch.int32),
    "repeated": torch.tensor([[2, 2, 0], [1, 2, 2], [0, 0, 0]], dtype=torch.int32),
    "dropped": torch.tensor([[-1, 3, -1], [3, -1, 1], [-1, -1, -1]], dtype=torch.int32),
    "all_dropped": torch.full((2, 3), -1, dtype=torch.int32),
    "int64": torch.tensor([[4, 1, -1], [0, 4, 2]], dtype=torch.int64),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_plan_matches_torch_where_row_major(case):
    topk_ids = CASES[case]
    plan = Exl3RoutePlan.from_topk(topk_ids)
    flat = topk_ids.reshape(-1)
    assert plan.experts == sorted(set(flat[flat >= 0].tolist()))
    assert plan.source_ids.dtype == topk_ids.dtype and plan.source_ids.tolist() == plan.experts
    for expert in plan.experts:
        want_token, want_slot = torch.where(topk_ids == expert)
        token, slot = plan.routes_of(expert)
        assert token.dtype == want_token.dtype == torch.int64
        assert torch.equal(token, want_token) and torch.equal(slot, want_slot)


def test_planned_accumulate_is_bitwise_the_where_loop_over_many_experts():
    """80 experts, more than one 64-expert gather chunk, each chunk accumulated in turn."""
    num_experts, tokens, topk = 80, 50, 6
    g = torch.Generator().manual_seed(1)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=g)[:topk] for _ in range(tokens)]).to(torch.int32)
    topk_ids[3, 2] = -1
    topk_ids[7, 1] = topk_ids[7, 0]  # one token routes an expert twice
    x = torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16)
    topk_weights = torch.rand(tokens, topk, generator=g)
    w13, w2 = _weights(num_experts)
    experts = sorted(set(topk_ids[topk_ids >= 0].tolist()))
    assert len(experts) > 64
    want = torch.zeros(tokens, HIDDEN)
    got = torch.zeros(tokens, HIDDEN)
    plan = Exl3RoutePlan.from_topk(topk_ids)
    for start in range(0, len(experts), 64):
        chunk = experts[start : start + 64]
        exl3_ops.exl3_moe_accumulate(want, x, topk_weights, topk_ids, w13, w2, 10.0, chunk, _fake_linear)
        exl3_ops.exl3_moe_accumulate_planned(got, x, topk_weights, plan, w13, w2, 10.0, chunk, _fake_linear)
    assert torch.equal(got, want)


def _device_rows(slots, hits):
    """_gather_cached's own row_of_source computation (the hit_rows branch in expert_stream.py), on CPU tensors."""
    slots_t, hit_mask = torch.tensor(slots), torch.tensor(hits)
    if bool(hit_mask.all()):
        return slots_t.tolist()
    order = torch.cat(((~hit_mask).nonzero().flatten(), hit_mask.nonzero().flatten()))
    row_of_source = torch.empty_like(order)
    row_of_source[order] = torch.arange(len(slots))
    return row_of_source.tolist()


@pytest.mark.parametrize(
    "slots",
    [[2, 0, 1], [-1, -1, -1], [-1, 4, -1, 0, -1], [7, -1]],
    ids=["all_hit", "all_miss", "mixed", "hit_first"],
)
def test_host_row_of_source_matches_the_device_order(slots):
    from sglang.srt.layers.moe.expert_stream import host_row_of_source

    hits = [slot >= 0 for slot in slots]
    assert host_row_of_source(slots, hits) == _device_rows(slots, hits)
