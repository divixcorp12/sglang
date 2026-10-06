"""The DSpark hybrid draft path (Exl3MoEMethod._apply_draft: the draft channel's post, DraftResidentMoe, finish) matches
exl3_moe_loop on the GPU, at the target's 3 and the draft's 4 bits; a captured call replays what eager computes; an eager
call reads nothing from the device.

Run on divix01 with the GPU lock, EXL3_MOE_CPU_PIN=0, SGLANG_DSV41_CPU_EXPERTS=1 and SGLANG_EXL3_SRC set (the
optimized CPU kernel's build):
  flock .../cc-gpu.lock taskset -c 6-17,32-63 python -m pytest test/manual/dsv41/test_dspark_hybrid_draft_gpu.py -s
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SGLANG_EXL3_SRC"),
    reason="needs CUDA and SGLANG_EXL3_SRC",
)

E_ROUTED, SHARED, HIDDEN, INTER, TOPK = 8, 1, 512, 256, 3
LIMIT = 10.0


def _layer(bits, resident, monkeypatch, registry, tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft, draft_resident
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    path = tmp_path / "resident.json"
    draft_resident.write_resident_set(str(path), {0: resident}, n=len(resident), source="")
    draft_resident._cache.clear()
    cfg = {"quant_method": "exl3", "version": "1.4.2", "bits": float(bits), "head_bits": 6, "codebook": "mul1"}
    e = E_ROUTED + SHARED
    layer = nn.Module()
    layer.layer_id = 0
    layer.num_experts = e
    layer.num_fused_shared_experts = SHARED
    layer.moe_runner_config = SimpleNamespace(swiglu_limit=LIMIT)
    layer.top_k = TOPK + SHARED
    method = Exl3MoEMethod(Exl3Config.from_config(cfg), streamed=False, draft=True, cpu_draft=True)
    method.create_weights(layer, e, HIDDEN, INTER, torch.bfloat16)
    g = torch.Generator().manual_seed(bits)
    for expert in range(e):
        for shard, (in_f, out_f), prefix in (
            ("w1", (HIDDEN, INTER), "w13"),
            ("w3", (HIDDEN, INTER), "w13"),
            ("w2", (INTER, HIDDEN), "w2"),
        ):
            tensors = {
                "trellis": torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 16 * bits), generator=g, dtype=torch.int16),
                "suh": (torch.randint(0, 2, (in_f,), generator=g) * 2 - 1).half(),
                "svh": (torch.randint(0, 2, (out_f,), generator=g) * 2 - 1).half(),
                "mul1": torch.tensor(1, dtype=torch.int32),
            }
            for name, tensor in tensors.items():
                param = getattr(layer, f"{prefix}_{name}")
                param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=expert)
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        method.process_weights_after_loading(layer)
    return layer, method


def _routes(pattern, g, rows):
    if pattern == "shared":
        routed = torch.randperm(E_ROUTED, generator=g)[:TOPK].repeat(rows, 1)
    else:
        routed = torch.stack([torch.randperm(E_ROUTED, generator=g)[:TOPK] for _ in range(rows)])
    shared = torch.full((rows, 1), E_ROUTED, dtype=torch.int64)
    return torch.cat([routed, shared], dim=1)


def _cores():
    return sorted(os.sched_getaffinity(0) & set(range(6, 18))) or sorted(os.sched_getaffinity(0))[:4]


class _Model(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.stage = layer


def _prepared(bits, resident, monkeypatch, tmp_path):
    """A prepared hybrid stage on a fresh registry; the caller closes the registry."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.draft import DraftCpuExpertsRegistry
    from sglang.srt.layers.quantization.exl3.draft_moe import prepare_dspark_draft_graph

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    registry = DraftCpuExpertsRegistry()
    layer, method = _layer(bits, resident, monkeypatch, registry, tmp_path)
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(",".join(map(str, _cores()))):
        assert prepare_dspark_draft_graph(_Model(layer)) == 1
    return layer, method, registry


def _inputs(g, pattern, rows):
    x = (torch.randn(rows, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).cuda()
    ids = _routes(pattern, g, rows).cuda()
    weights = torch.rand(rows, TOPK + 1, generator=g).cuda()
    return x, ids, weights


@pytest.mark.parametrize("m", [1, 5, 16, 40])
@pytest.mark.parametrize("resident", [[], [1, 5], list(range(E_ROUTED))], ids=["all-cpu", "hybrid", "all-gpu"])
@pytest.mark.parametrize("pattern", ["independent", "shared"])
@pytest.mark.parametrize("bits", [3, 4])
def test_hybrid_draft_matches_the_gpu_loop(bits, pattern, resident, m, monkeypatch, tmp_path):
    from dataclasses import replace

    from sglang.srt.layers.quantization.exl3.ops import exl3_moe_loop

    layer, method, registry = _prepared(bits, resident, monkeypatch, tmp_path)
    g = torch.Generator().manual_seed(100 + bits)
    x, ids, weights = _inputs(g, pattern, m)

    def gpu(t):
        return replace(t, trellis=t.trellis.cuda(), suh=t.suh.cuda(), svh=t.svh.cuda())

    w13 = [tuple(gpu(t) for t in pair) for pair in layer.exl3_w13]
    w2 = [gpu(t) for t in layer.exl3_w2]
    try:
        got = method._apply_draft(layer, x, weights, ids, LIMIT).float()
        torch.cuda.synchronize()
        jobs = registry.runtime().stats()["jobs"]
    finally:
        registry.close()
    ref = exl3_moe_loop(x, weights, ids, w13, w2, LIMIT).float()
    rel = float((got - ref).norm() / ref.norm())
    chunks = -(-m // layer.exl3_draft_moe.tokens)
    print(f"bits={bits} pattern={pattern} resident={len(resident)} m={m} rel_l2={rel:.4f} jobs={jobs}")
    assert torch.isfinite(got).all()
    # Measured 0.003-0.014 (fused-vs-loop rounding, worst at m=1; the CPU share adds none): 0.02 still fails a dropped
    # low-weight route, which 0.05 would pass.
    assert rel < 0.02
    assert jobs == (0 if len(resident) == E_ROUTED else chunks)


def test_the_captured_hybrid_draft_matches_eager(monkeypatch, tmp_path):
    layer, method, registry = _prepared(4, [1, 5], monkeypatch, tmp_path)
    g = torch.Generator().manual_seed(7)
    try:
        x_s, ids_s, w_s = _inputs(g, "independent", 5)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            method._apply_draft(layer, x_s, w_s, ids_s, LIMIT)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_s = method._apply_draft(layer, x_s, w_s, ids_s, LIMIT)
        runtime = registry.runtime()
        on_cpu = runtime.on_cpu[layer.exl3_cpu_draft_key]
        for replay in range(5):
            x_n, ids_n, w_n = _inputs(g, "independent" if replay % 2 else "shared", 5)
            if replay == 3:
                ids_n[:, :TOPK] = torch.tensor([1, 5, -1], device="cuda")  # resident and -1 only: nothing posted
            x_s.copy_(x_n), ids_s.copy_(ids_n), w_s.copy_(w_n)
            torch.cuda.synchronize()
            before = runtime.stats()["jobs"]
            graph.replay()
            torch.cuda.synchronize()
            posted = bool((runtime.cpu_slots(layer.exl3_cpu_draft_key, ids_n.cpu()) >= 0).any())
            assert runtime.stats()["jobs"] - before == int(posted), replay
            replayed = out_s.clone()
            eager = method._apply_draft(layer, x_s, w_s, ids_s, LIMIT)
            torch.cuda.synchronize()
            assert torch.equal(replayed, eager), replay
        assert on_cpu.any()
    finally:
        registry.close()


def test_the_draft_path_reads_no_host(monkeypatch, tmp_path):
    layer, method, registry = _prepared(4, [1, 5], monkeypatch, tmp_path)
    try:
        x, ids, w = _inputs(torch.Generator().manual_seed(3), "independent", 5)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            method._apply_draft(layer, x, w, ids, LIMIT)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    finally:
        registry.close()
