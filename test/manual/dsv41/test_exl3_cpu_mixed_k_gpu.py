"""The EXL3 CPU expert kernel on layers whose gate/up (w13) and down (w2) bit widths differ from DSV4.1's 3/3, against
exl3_moe_loop on the GPU: a K2 layer and layers with K(w13) = 3 and K(w2) = 2 or 4, as the mixed-K checkpoint holds
them. Each runs the generic plan (the plan counter), with every projection decoded at its own K.

Needs CUDA, SGLANG_EXL3_SRC and the optimized CPU kernel (SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1). Short GPU work:
it shares the card. Run on divix01 under taskset -c 0-51,56-107 with OMP_NUM_THREADS set.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SGLANG_EXL3_SRC"),
    reason="needs CUDA and SGLANG_EXL3_SRC",
)

HIDDEN, INTER, CAP, TOPK, LIMIT, THREADS = 512, 256, 6, 3, 10.0, 4


def _trait():
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the kernel under test is the optimized one: set SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
    return Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT)


def _experts(bits_w13, bits_w2, seed):
    """CAP experts of random codes and signs: (gate, up, down) Exl3Tensors per expert, mul1 as DSV4.1 ships."""
    from sglang.srt.layers.quantization.exl3.ops import Exl3Tensors

    g = torch.Generator().manual_seed(seed)

    def linear(in_f, out_f, bits):
        return Exl3Tensors(
            trellis=torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 16 * bits), generator=g, dtype=torch.int16),
            suh=(torch.randint(0, 2, (in_f,), generator=g) * 2 - 1).half(),
            svh=(torch.randint(0, 2, (out_f,), generator=g) * 2 - 1).half(),
            mul1=True,
        )

    return [
        (linear(HIDDEN, INTER, bits_w13), linear(HIDDEN, INTER, bits_w13), linear(INTER, HIDDEN, bits_w2))
        for _ in range(CAP)
    ]


def _slabs(experts):
    """The pinned tier's six slabs over the experts, expert e in slot e: w13 rows hold gate then up."""

    def stack(get):
        return torch.stack([get(e) for e in experts]).contiguous()

    return {
        "w13_trellis": stack(lambda e: torch.stack([e[0].trellis, e[1].trellis])),
        "w13_suh": stack(lambda e: torch.stack([e[0].suh, e[1].suh])),
        "w13_svh": stack(lambda e: torch.stack([e[0].svh, e[1].svh])),
        "w2_trellis": stack(lambda e: e[2].trellis),
        "w2_suh": stack(lambda e: e[2].suh),
        "w2_svh": stack(lambda e: e[2].svh),
    }


def _reference(experts, x, ids, weights):
    from dataclasses import replace

    from sglang.srt.layers.quantization.exl3.ops import exl3_moe_loop

    def gpu(t):
        return replace(t, trellis=t.trellis.cuda(), suh=t.suh.cuda(), svh=t.svh.cuda())

    w13 = [(gpu(gate), gpu(up)) for gate, up, _ in experts]
    w2 = [gpu(down) for _, _, down in experts]
    return exl3_moe_loop(x.cuda(), weights.cuda(), ids.cuda(), w13, w2, LIMIT).float().cpu()


@pytest.mark.parametrize("bits_w13, bits_w2", [(2, 2), (3, 2), (3, 4)], ids=["k2", "w13k3-w2k2", "w13k3-w2k4"])
@pytest.mark.parametrize("rows", [1, 5])
def test_a_layer_of_any_k_matches_the_gpu_loop_on_the_generic_plan(monkeypatch, bits_w13, bits_w2, rows):
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
    trait = _trait()
    experts = _experts(bits_w13, bits_w2, seed=10 * bits_w13 + bits_w2)
    spec = trait.layer_spec(_slabs(experts), CAP)
    assert spec.plan.startswith("EXL3 CPU plan generic (")
    layer = es.kernel_layer(trait.kernel_address(), spec, variant="instr")
    g = torch.Generator().manual_seed(rows)
    x = (torch.randn(rows, HIDDEN, generator=g) * 0.5).half()
    ids = torch.stack([torch.randperm(CAP, generator=g)[:TOPK] for _ in range(rows)])
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g), -1).half()
    try:
        before = trait._op("plan_calls")()
        got = torch.full((rows, HIDDEN), float("nan"))
        status, why = es.kernel_forward(
            layer, x, ids.to(torch.int32), weights.float(), got, threads=THREADS, variant="instr"
        )
        after = trait._op("plan_calls")()
    finally:
        es.kernel_drop(layer, variant="instr")
    assert (status, why) == (0, "")
    assert (after[0] - before[0], after[1] - before[1]) == (0, 1), "one forward, on the generic plan"
    want = _reference(experts, x, ids, weights)
    rel = float((got - want).norm() / want.norm())
    print(f"K w13 {bits_w13} K w2 {bits_w2} rows {rows}: relative error {rel:.4f}")
    assert torch.isfinite(got).all()
    # The same bound as test_dspark_hybrid_draft_gpu's CPU share against this loop; measured 0.0075-0.014 here.
    assert rel < 0.02
