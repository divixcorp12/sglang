"""The EXL3 CPU kernel's DSV4.1 plan on calls whose routes share an expert (csrc/exl3/optimized/forward_plan.hpp).

Exl3Quant::dispatch groups a call's routes by expert into chunks of up to CHUNK_M tokens (sglang_exl3_cpu::chunk_m:
MAX_M / 2, so 2 by default and 4 in a SGLANG_EXL3_CPU_MAX_M=8 build). Each case runs one call on a DeepSeek
V4.1-shaped layer:
  - unswizzled (native), which Dsv41Shape::accepts may take on AVX-512BW;
  - swizzled, which it always refuses, so it runs the generic plan;
  - token by token, through the native layer.
The three must agree bit for bit. The plan counter (sglang_exl3_cpu::plan_calls) shows which plan each call took. The
chunk-size cases are generated from the build's CHUNK_M, so a larger MAX_M widens them with no edit here.

Needs SGLANG_EXL3_SRC, SGLANG_DSV41_CPU_EXPERTS=1 and the bw tier (EXL3_MOE_CPU_MAX_ISA=bw on a host above it). Run on
divix01 under taskset -c 0-63.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

H, I, CAP, LIMIT, THREADS = 5120, 2304, 12, 10.0, 4
DEFAULT_CHUNK_M = 2  # CHUNK_M at the source's default MAX_M (math.hpp's EXL3_MOE_CPU_MAX_M)


def _swizzle(t):
    """The swizzled trellis layout over a slab's last three dims [k/16, n/16, 48] (as test_exl3_cpu_act_quant)."""
    tk, tn, ps = t.shape[-3:]
    return t.reshape(-1, tk, tn // 8, 8, ps).permute(0, 2, 1, 3, 4).contiguous().view(t.shape)


def _slabs(seed):
    """The pinned tier's six slabs, CAP slots of random 3-bit codes and signs."""
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, H // 16, I // 16, 48),
        "w13_suh": signs(CAP, 2, H),
        "w13_svh": signs(CAP, 2, I),
        "w2_trellis": codes(CAP, 1, I // 16, H // 16, 48),
        "w2_suh": signs(CAP, 1, I),
        "w2_svh": signs(CAP, 1, H),
    }


def _random_routes(seed, rows, k):
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(CAP, generator=g)[:k].tolist() for _ in range(rows)]


# (name, routes [rows][k]; -1 is no route). The comment gives the chunks dispatch makes at CHUNK_M = 2, in expert order.
CASES = [
    ("one-token", [[4, 0, 5]]),  # 1, 1, 1
    ("two-tokens-one-expert", [[3], [3]]),  # 2: a single-expert call (grouped on, wide off)
    ("two-tokens-three-experts", [[4, 0, 5], [0, 4, 5]]),  # 2, 2, 2
    ("three-tokens-one-expert", [[2], [2], [2]]),  # 2 then 1, one expert
    ("mixed-sizes", [[0, 1, 2], [1, 3, 5]]),  # 1, 2, 1, 1, 1
    ("one-token-twice", [[1, 1, 2]]),  # 2 (the same token twice), 1
    ("dead-routes", [[1, -1, 2], [1, 2, -1]]),  # 2, 2; token 1's route to expert 1 weighs 0
    ("draft-2x3", _random_routes(2, 2, 3)),
    ("draft-4x3", _random_routes(4, 4, 3)),
    ("draft-8x3", _random_routes(8, 8, 3)),
    ("draft-16x3", _random_routes(16, 16, 3)),
    ("prefill-64x6", _random_routes(64, 64, 6)),
]
ZERO_WEIGHT = {"dead-routes": (1, 0)}


def _shares_an_expert(routes):
    live = [s for row in routes for s in row if s >= 0]
    return len(live) != len(set(live))


def _expected_plan(routes):
    """The plan a call on the native layer takes: the DSV4.1 plan unless two of its routes share an expert."""
    return "generic" if _shares_an_expert(routes) else "dsv41"


def _chunk_m():
    return int(torch.ops.sglang_exl3_cpu.chunk_m())


def _inputs(name, routes):
    g = torch.Generator().manual_seed(sum(map(ord, name)))
    rows, k = len(routes), len(routes[0])
    x = (torch.randn(rows, H, generator=g) * 4.0).half()
    weights = torch.rand(rows, k, generator=g) + 0.05
    if name in ZERO_WEIGHT:
        weights[ZERO_WEIGHT[name]] = 0.0
    return x, torch.tensor(routes, dtype=torch.int32), weights


@pytest.fixture(scope="module")
def layers():
    """{"native": handle, "swizzled": handle} over the same experts."""
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the kernel under test is the optimized one: set SGLANG_DSV41_CPU_EXPERTS=1")
    ext = exl3_ext()
    if not ext.exl3_moe_cpu_has_avx512_bw():
        pytest.skip("the DSV4.1 plan needs AVX-512BW")
    slabs = _slabs(20261006)
    handles = {}
    for layout in ("native", "swizzled"):
        swizzled = layout == "swizzled"
        trait = Exl3CpuQuantTrait(ext, act_limit=LIMIT, swizzled=swizzled)
        held = {n: _swizzle(t) if swizzled and n.endswith("_trellis") else t for n, t in slabs.items()}
        handles[layout] = es.kernel_layer(trait.kernel_address(), trait.layer_spec(held, CAP), variant="instr")
    yield handles
    for handle in handles.values():
        es.kernel_drop(handle, variant="instr")


def _forward(layer, x, slots, weights, threads=THREADS):
    """One call, unpinned; returns (out, the plan it took) from the plan counter's step."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    before = torch.ops.sglang_exl3_cpu.plan_calls()
    out = torch.full((x.shape[0], H), float("nan"))
    status, why = es.kernel_forward(layer, x, slots, weights, out, threads=threads, variant="instr")
    assert (status, why) == (0, "")
    after = torch.ops.sglang_exl3_cpu.plan_calls()
    step = (after[0] - before[0], after[1] - before[1])
    assert step in ((1, 0), (0, 1)), step
    return out, "dsv41" if step == (1, 0) else "generic"


def _check(layers, name, routes):
    """One call on both layers and token by token: the same bits everywhere, and the plan _expected_plan names."""
    x, slots, weights = _inputs(name, routes)
    got, plan = _forward(layers["native"], x, slots, weights)
    want, generic = _forward(layers["swizzled"], x, slots, weights)
    assert (plan, generic) == (_expected_plan(routes), "generic"), name
    assert torch.isfinite(got).all(), name
    assert torch.equal(got, want), name
    singles = torch.cat(
        [_forward(layers["native"], x[t : t + 1], slots[t : t + 1], weights[t : t + 1])[0] for t in range(len(routes))]
    )
    assert torch.equal(got, singles), name


def test_the_build_reports_its_chunk_size(layers):
    """CHUNK_M is half of MAX_M: SGLANG_EXL3_CPU_MAX_M's half when the build sets it, else the source default's."""
    max_m = int(os.environ.get("SGLANG_EXL3_CPU_MAX_M") or 0)
    assert _chunk_m() == (max_m // 2 if max_m else DEFAULT_CHUNK_M)


@pytest.mark.parametrize("name,routes", CASES, ids=[c[0] for c in CASES])
def test_dsv41_plan_matches_the_generic_plan_and_one_token_runs(layers, name, routes):
    _check(layers, name, routes)


def test_every_chunk_size_matches_the_generic_plan(layers):
    """For m = 1..CHUNK_M, read from the build: m tokens on one expert and m tokens sharing three experts (chunks of
    exactly m), then CHUNK_M + 1 tokens on one expert (a full chunk, then a chunk of one)."""
    chunk_m = _chunk_m()
    for m in range(1, chunk_m + 1):
        _check(layers, f"m{m}-one-expert", [[3]] * m)
        _check(layers, f"m{m}-three-experts", [[4, 0, 5]] * m)
    _check(layers, f"m{chunk_m + 1}-overflow", [[2]] * (chunk_m + 1))


@pytest.mark.parametrize("threads", [1, 3, 16])
def test_dsv41_plan_bits_do_not_depend_on_the_team(layers, threads):
    routes = dict(CASES)["draft-16x3"]
    x, slots, weights = _inputs("draft-16x3", routes)
    want, _ = _forward(layers["swizzled"], x, slots, weights)
    got, plan = _forward(layers["native"], x, slots, weights, threads=threads)
    assert plan == _expected_plan(routes)
    assert torch.equal(got, want)
