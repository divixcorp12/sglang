"""DraftResidentMoe on the GPU: the draft's resident experts through Exl3FusedMoE, a sink for every other route,
captured in a graph with no host read.

Random valid EXL3 experts (no checkpoint read) in the [expert, part, ...] slab layout of a DSpark draft stage's
parameters. Bars:
  * parity: run() over all resident experts matches a dense fp32 reference, rel <= 2e-2, for M in (1, 2, 5, 6, 10);
    10 is the most a call holds at top-k 6 (64 routes, exl3_route_tables.cuh), and 11 is refused;
  * the sink: with resident {0, 2, 5}, routes to 1, 7 (not resident), -1 and 9 (out of range) add nothing, so run()
    matches exl3_moe_accumulate over experts [0, 2, 5]; with layer fusion on and off;
  * capture: a graph captured at M = 5 replays rewritten ids, weights and inputs bitwise equal to an eager run;
  * no host read: an eager run() after prepare() passes torch.cuda.set_sync_debug_mode("error").
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.exl3.ops import exl3_dense_weight, exl3_moe_accumulate, random_exl3_tensors

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT = 8, 5120, 2304, 6, 10.0
REL = 2e-2


@pytest.fixture(scope="module")
def layer():
    """A draft stage's parameters ([expert, part, ...] slabs) on the host, as _attach_cpu_draft sees them, plus the
    per-expert views (on the GPU) for the references."""
    w13 = [
        (
            random_exl3_tensors(HIDDEN, INTER, 3, device="cuda", seed=3 * e),
            random_exl3_tensors(HIDDEN, INTER, 3, device="cuda", seed=3 * e + 1),
        )
        for e in range(EXPERTS)
    ]
    w2 = [random_exl3_tensors(INTER, HIDDEN, 3, device="cuda", seed=3 * e + 2) for e in range(EXPERTS)]
    slab = {}
    for kind in ("trellis", "suh", "svh"):
        slab[f"w13_{kind}"] = torch.stack([torch.stack([getattr(t, kind) for t in pair]) for pair in w13]).cpu()
        slab[f"w2_{kind}"] = torch.stack([getattr(t, kind)[None] for t in w2]).cpu()
    return SimpleNamespace(top_k=TOP_K, exl3_w13=w13, exl3_w2=w2, **slab)


def _moe(layer, resident, *, layer_fusion=True):
    from sglang.srt.layers.quantization.exl3.draft_moe import DraftResidentMoe

    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(layer_fusion):
        moe = DraftResidentMoe(layer, resident, EXPERTS, torch.device("cuda"))
        moe.prepare()
    return moe


def _inputs(gen, m, pool):
    ids = torch.stack([torch.tensor(pool)[torch.randperm(len(pool), generator=gen)[:TOP_K]] for _ in range(m)])
    weights = torch.softmax(torch.randn(m, TOP_K, generator=gen), -1)
    x = torch.randn(m, HIDDEN, generator=gen) * 0.05
    return x.cuda().to(torch.bfloat16), ids.cuda(), weights.cuda()


def _dense(layer, x, ids, weights, experts):
    want = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    for e in experts:
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        xe = x[tok].half().float()
        gate = (xe @ exl3_dense_weight(layer.exl3_w13[e][0]).float()).clamp(max=LIMIT)
        up = (xe @ exl3_dense_weight(layer.exl3_w13[e][1]).float()).clamp(-LIMIT, LIMIT)
        h = (F.silu(gate) * up * weights[tok, slot, None]).half().float()
        want.index_add_(0, tok, h @ exl3_dense_weight(layer.exl3_w2[e]).float())
    return want


def _rel(got, want):
    return float((got.float() - want).norm() / want.norm())


@pytest.mark.parametrize("m", [1, 2, 5, 6, 10])
def test_all_resident_matches_the_dense_experts(layer, m):
    moe = _moe(layer, range(EXPERTS))
    assert moe.tokens == 10
    gen = torch.Generator().manual_seed(100 + m)
    x, ids, weights = _inputs(gen, m, list(range(EXPERTS)))
    got = moe.run(x, ids, weights, LIMIT)
    assert got.dtype == torch.float32 and got.shape == (m, HIDDEN)
    assert _rel(got, _dense(layer, x, ids, weights, range(EXPERTS))) <= REL


def test_a_call_past_the_stages_tokens_is_refused(layer):
    moe = _moe(layer, range(EXPERTS))
    x, ids, weights = _inputs(torch.Generator().manual_seed(5), 11, list(range(EXPERTS)))
    with pytest.raises(ValueError, match="1-10 tokens"):
        moe.run(x, ids, weights, LIMIT)


@pytest.mark.parametrize("layer_fusion", [True, False])
@pytest.mark.parametrize("m", [1, 6])
def test_the_sink_drops_every_route_that_is_not_resident(layer, layer_fusion, m):
    resident = [0, 2, 5]
    moe = _moe(layer, resident, layer_fusion=layer_fusion)
    assert moe.slots == 3
    gen = torch.Generator().manual_seed(7 + m)
    x, ids, weights = _inputs(gen, m, [0, 1, 2, 5, 7, -1, 9])
    got = moe.run(x, ids, weights, LIMIT)
    want = torch.zeros(m, HIDDEN, dtype=torch.float32, device="cuda")
    exl3_moe_accumulate(want, x, weights, ids, layer.exl3_w13, layer.exl3_w2, LIMIT, experts=resident)
    assert want.norm() > 0
    assert _rel(got, want) <= REL


def test_a_captured_run_replays_what_eager_computes(layer):
    moe = _moe(layer, [0, 2, 5])
    gen = torch.Generator().manual_seed(11)
    pool = [0, 1, 2, 5, 7, -1]
    x_s, ids_s, w_s = _inputs(gen, 5, pool)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        moe.run(x_s, ids_s, w_s, LIMIT)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_s = moe.run(x_s, ids_s, w_s, LIMIT)
    for _ in range(4):
        x_n, ids_n, w_n = _inputs(gen, 5, pool)
        x_s.copy_(x_n), ids_s.copy_(ids_n), w_s.copy_(w_n)
        graph.replay()
        replayed = out_s.clone()
        assert torch.equal(replayed, moe.run(x_s, ids_s, w_s, LIMIT).clone())


def test_an_eager_run_reads_nothing_from_the_device(layer):
    moe = _moe(layer, [0, 2, 5])
    x, ids, weights = _inputs(torch.Generator().manual_seed(3), 4, [0, 1, 2, 5, 7, -1, 9])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        moe.run(x, ids, weights, LIMIT)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
