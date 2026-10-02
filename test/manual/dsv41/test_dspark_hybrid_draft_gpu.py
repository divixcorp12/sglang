"""The DSpark hybrid draft path matches exl3_moe_loop on the GPU, at the target's 3 and the draft's 4 bits.

Run on divix01 with the GPU lock, EXL3_MOE_CPU_PIN=0 and SGLANG_EXL3_SRC set (the CPU kernel's build):
  flock .../cc-gpu.lock taskset -c 18-29,32-63 python -m pytest test/manual/dsv41/test_dspark_hybrid_draft_gpu.py -s
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

E_ROUTED, SHARED, HIDDEN, INTER, ROWS, TOPK = 8, 1, 512, 256, 5, 3
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
    method = Exl3MoEMethod(Exl3Config.from_config(cfg), streamed=False, cpu_draft=True)
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


def _routes(pattern, g):
    if pattern == "shared":
        routed = torch.randperm(E_ROUTED, generator=g)[:TOPK].repeat(ROWS, 1)
    else:
        routed = torch.stack([torch.randperm(E_ROUTED, generator=g)[:TOPK] for _ in range(ROWS)])
    shared = torch.full((ROWS, 1), E_ROUTED, dtype=torch.int64)
    return torch.cat([routed, shared], dim=1)


@pytest.mark.parametrize("resident", [[], [1, 5], list(range(E_ROUTED))], ids=["all-cpu", "hybrid", "all-gpu"])
@pytest.mark.parametrize("pattern", ["independent", "shared"])
@pytest.mark.parametrize("bits", [3, 4])
def test_hybrid_draft_matches_the_gpu_loop(bits, pattern, resident, monkeypatch, tmp_path):
    from dataclasses import replace

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.draft import DraftCpuExpertsRegistry
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_loop

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    cores = sorted(os.sched_getaffinity(0) & set(range(18, 30))) or sorted(os.sched_getaffinity(0))[:4]
    registry = DraftCpuExpertsRegistry()
    layer, method = _layer(bits, resident, monkeypatch, registry, tmp_path)
    g = torch.Generator().manual_seed(100 + bits)
    x = (torch.randn(ROWS, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).cuda()
    ids = _routes(pattern, g).cuda()
    weights = torch.rand(ROWS, TOPK + 1, generator=g).cuda()

    def gpu(t):
        return replace(t, trellis=t.trellis.cuda(), suh=t.suh.cuda(), svh=t.svh.cuda())

    w13 = [tuple(gpu(t) for t in pair) for pair in layer.exl3_w13]
    w2 = [gpu(t) for t in layer.exl3_w2]
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(",".join(map(str, cores))):
        try:
            got = method._apply_cpu_draft(layer, x, weights, ids, LIMIT).float()
            stats = registry.runtime().stats
            skips = stats.skips
        finally:
            registry.close()
    ref = exl3_moe_loop(x, weights, ids, w13, w2, LIMIT).float()
    rel = float((got - ref).norm() / ref.norm())
    print(f"bits={bits} pattern={pattern} resident={len(resident)} rel_l2={rel:.4f} skips={skips}")
    assert torch.isfinite(got).all()
    assert rel < 0.05
    assert skips == (1 if len(resident) == E_ROUTED else 0)
