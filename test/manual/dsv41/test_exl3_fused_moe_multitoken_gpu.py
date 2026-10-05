"""D2-1 gate: the in-graph EXL3 MoE over M tokens that share slots, on real layer rows (GPU).

16 real experts of one layer fill 16 slots. Every M in (1, 2, 4, 6) draws 8 route sets of top-6 slots per token,
so tokens share slots. Bars, per token:
  * rel(fused) <= 1.2e-2 and rel(fused) <= 2 * rel(exl3_moe_loop) + 1e-3, against the probe's fp32 reference;
  * the layer-fusion and torch-chain route tables give the same output bitwise, and a second eager run repeats it;
  * a graph captured at M replays rewritten routes and inputs bitwise equal to an eager run;
  * one object serving M = 6 then M = 1 gives what a fresh object gives at M = 1.
Reports eager and replay microseconds per M to DSV41_MULTITOKEN_OUT (JSON) when set.
Env: DSV41_EXL3_DIR, DSV41_PROBE_LAYER (as the probe).
"""

import json
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_moe_probe_gpu import (  # noqa: E402
    ACT_LIMIT,
    LAYER,
    REL_BOUND,
    _load_slots,
    _reference,
    _rel,
    _views,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXPERTS = list(range(0, 384, 24))  # 16 experts -> 16 slots
TOP_K = 6
TOKENS = 6
WIDTHS = (1, 2, 4, 6)
ROUTE_SETS = 8
OUT = os.environ.get("DSV41_MULTITOKEN_OUT")


@pytest.fixture(scope="module")
def rows():
    import test_exl3_moe_probe_gpu as probe

    device = torch.device("cuda", torch.cuda.current_device())
    saved, probe.EXPERTS = probe.EXPERTS, EXPERTS  # _load_slots reads the module global
    try:
        tensors = _load_slots(device)
    finally:
        probe.EXPERTS = saved  # the probe test may run later in the same session
    return tensors, [_views(tensors, s) for s in range(len(EXPERTS))]


def _fused(tensors, layer_fusion):
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3.fused_moe import Exl3FusedMoE

    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(layer_fusion):
        return Exl3FusedMoE(
            tensors,
            len(EXPERTS),
            hidden=tensors["w13_suh"].shape[-1],
            inter=tensors["w2_suh"].shape[-1],
            top_k=TOP_K,
            device=tensors["w13_suh"].device,
            tokens=TOKENS,
        )


def _inputs(gen, m, hidden, device):
    remap = torch.cat([torch.randperm(len(EXPERTS), generator=gen)[:TOP_K] for _ in range(m)]).to(device)
    weights = torch.softmax(torch.randn(m, TOP_K, generator=gen), -1).reshape(-1).to(device)
    x16 = (torch.randn((m, hidden), generator=gen) * 0.5).to(device, torch.float16)
    return x16, weights, remap


def _run(fused, x16, weights, remap):
    keep = torch.ones(1, device=x16.device)
    return fused.run(x16, weights, remap if fused.layer_fusion else remap.long(), keep, ACT_LIMIT).clone()


def test_multitoken_fused_moe(rows):
    from sglang.srt.layers.quantization.exl3.ops import exl3_moe_loop

    tensors, views = rows
    hidden = tensors["w13_suh"].shape[-1]
    device = tensors["w13_suh"].device
    w13, w2 = [v[0] for v in views], [v[1] for v in views]
    chain, fusion = _fused(tensors, False), _fused(tensors, True)
    gen = torch.Generator().manual_seed(4321)
    report = {"layer": LAYER, "experts": EXPERTS, "widths": {}}
    failures = []
    for m in WIDTHS:
        entry = {"route_sets": []}
        for _ in range(ROUTE_SETS):
            x16, weights, remap = _inputs(gen, m, hidden, device)
            a = _run(chain, x16, weights, remap)
            b = _run(fusion, x16, weights, remap)
            again = _run(fusion, x16, weights, remap)
            loop = exl3_moe_loop(x16, weights.view(m, TOP_K), remap.view(m, TOP_K), w13, w2, ACT_LIMIT).float()
            if not torch.equal(a, b):
                failures.append(f"M={m}: layer fusion != torch chain")
            if not torch.equal(b, again):
                failures.append(f"M={m}: eager rerun differs")
            for t in range(m):
                ref = _reference(x16[t : t + 1], weights.view(m, TOP_K)[t], remap.view(m, TOP_K)[t], views)
                rf, rl = _rel(a[t : t + 1], ref), _rel(loop[t : t + 1], ref)
                entry["route_sets"].append({"token": t, "rel_fused": rf, "rel_loop": rl})
                if not (rf <= REL_BOUND and rf <= 2 * rl + 1e-3):
                    failures.append(f"M={m} token {t}: rel {rf:.4g} (loop {rl:.4g})")
        # Capture at M over static inputs, replay after rewriting them in place.
        x_s, w_s, r_s = _inputs(gen, m, hidden, device)
        keep = torch.ones(1, device=device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_s = fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        for _ in range(4):
            x_n, w_n, r_n = _inputs(gen, m, hidden, device)
            x_s.copy_(x_n), w_s.copy_(w_n), r_s.copy_(r_n)
            graph.replay()
            replayed = out_s.clone()
            if not torch.equal(replayed, _run(fusion, x_s, w_s, r_s)):
                failures.append(f"M={m}: replay != eager")
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        torch.cuda.synchronize()
        entry["eager_us"] = (time.perf_counter() - started) * 1e4
        started = time.perf_counter()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        entry["replay_us"] = (time.perf_counter() - started) * 1e4
        entry["max_rel_fused"] = max(r["rel_fused"] for r in entry["route_sets"])
        report["widths"][str(m)] = entry
    # A wider run leaves nothing behind for a narrower one.
    x16, weights, remap = _inputs(gen, 1, hidden, device)
    _run(fusion, *_inputs(gen, TOKENS, hidden, device))
    if not torch.equal(_run(fusion, x16, weights, remap), _run(_fused(tensors, True), x16, weights, remap)):
        failures.append("M=1 after M=6 differs from a fresh object")
    report["failures"] = failures
    if OUT:
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)
    print(
        json.dumps(
            {m: (e["max_rel_fused"], round(e["eager_us"], 1), round(e["replay_us"], 1)) for m, e in report["widths"].items()}
        )
    )
    assert not failures, failures
