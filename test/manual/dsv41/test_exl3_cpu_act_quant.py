"""The EXL3 CPU MoE kernel under the build's activation quantization options (SGLANG_EXL3_CPU_ACT_*).

Builds whatever flavor the environment selects, then checks it on random DSV4.1-shaped experts against the scalar
tier (fp32 activations, which ignores the options):
  - the swizzled layout gives the same bits as the native one (the k-block sub-view addresses both);
  - a 5-token batch gives each token the same bits as running it alone (chunks are capped for the residual rows);
  - the relative error against the scalar tier is under the flavor's bound.

Needs SGLANG_EXL3_SRC and an AVX2-or-better CPU, and builds the expert-stream host module's instrumented variant for its
test exports (kernel_layer, kernel_forward; CXX the kernels' GCC 15, as run_exl3_cpu_forward_checks.sh sets it). Each
ISA tier runs in its own process, since the tier is fixed at the first kernel call. Run on divix01 under taskset, e.g.
  SGLANG_EXL3_CPU_ACT_RESIDUAL=1 SGLANG_EXL3_CPU_ACT_BLOCK=128 \\
    taskset -c 18-25 python -m pytest test/manual/dsv41/test_exl3_cpu_act_quant.py -q
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

H, I, KB, E, TOKENS, TOPK = 5120, 2304, 3, 8, 5, 6
SWIGLU_LIMIT = 10.0


def _swizzle(t):
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)


def _weights(torch):
    g = torch.Generator().manual_seed(20260929)

    def trellis(k, n):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * KB), dtype=torch.int16, generator=g)

    def signs(n, scale):
        s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
        return (s * scale * (1 + 0.25 * torch.randn(n, generator=g))).half()

    w = {k: [] for k in ("gt", "gs", "gv", "ut", "us", "uv", "dt", "ds", "dv")}
    for _ in range(E):
        w["gt"].append(trellis(H, I)); w["gs"].append(signs(H, 0.015)); w["gv"].append(signs(I, 1.0))
        w["ut"].append(trellis(H, I)); w["us"].append(signs(H, 0.015)); w["uv"].append(signs(I, 1.0))
        w["dt"].append(trellis(I, H)); w["ds"].append(signs(I, 0.015)); w["dv"].append(signs(H, 1.0))
    x = (torch.randn(TOKENS, H, generator=g) * 2).half()
    sel = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(TOKENS)]).long()
    rw = torch.rand(TOKENS, TOPK, generator=g) + 0.1
    return w, x, sel, (rw / rw.sum(-1, keepdim=True)).half()


def _slabs(torch, w, tr):
    """The experts as the pinned tier holds them, expert e in slot e: w13 rows hold gate then up, w2 rows hold down."""

    def stack(*parts):
        return torch.stack([torch.stack([p[e] for p in parts]) for e in range(E)]).contiguous()

    return {
        "w13_trellis": stack([tr(t) for t in w["gt"]], [tr(t) for t in w["ut"]]),
        "w13_suh": stack(w["gs"], w["us"]),
        "w13_svh": stack(w["gv"], w["uv"]),
        "w2_trellis": stack([tr(t) for t in w["dt"]]),
        "w2_suh": stack(w["ds"]),
        "w2_svh": stack(w["dv"]),
    }


def _run(tier, out_path):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = tier
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    import torch

    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.ext import exl3_ext
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    e = exl3_ext()
    w, x, sel, rw = _weights(torch)
    threads = min(8, len(os.sched_getaffinity(0)))
    res = {"bw": bool(e.exl3_moe_cpu_has_avx512_bw())}

    def forward(layer, rows):
        out = torch.zeros(x[rows].shape[0], H)
        status, why = es.kernel_forward(layer, x[rows], sel[rows], rw[rows], out, threads=threads, variant="instr")
        assert status == 0, why
        return out

    for layout in ("native", "swizzled") if tier != "scalar" else ("native",):
        swz = layout == "swizzled"
        trait = Exl3CpuQuantTrait(e, act_limit=SWIGLU_LIMIT, swizzled=swz)
        spec = trait.layer_spec(_slabs(torch, w, _swizzle if swz else (lambda t: t)), E)
        layer = es.kernel_layer(trait.kernel_address(), spec, variant="instr")
        try:
            batch = forward(layer, slice(0, TOKENS))
            single = torch.cat([forward(layer, slice(t, t + 1)) for t in range(TOKENS)])
        finally:
            es.kernel_drop(layer, variant="instr")
        res[layout] = {"batch": batch, "single": single}
    torch.save(res, out_path)


def _tier(tier, tmp_path):
    import torch

    out = tmp_path / f"{tier}.pt"
    subprocess.run([sys.executable, __file__, tier, str(out)], check=True)
    return torch.load(out)


def _bound():
    from sglang.srt.environ import envs

    residual = envs.SGLANG_EXL3_CPU_ACT_RESIDUAL.get()
    block = envs.SGLANG_EXL3_CPU_ACT_BLOCK.get()
    # Upstream measures ~1.4-1.9% on real DSV4.1 rows (DSV41_REFERENCE §28, the P0 plan results); a second int8 pass
    # leaves ~1/127 of the first pass's error, and per-block scales a fraction of it.
    return 0.006 if residual else (0.02 if block else 0.03)


@pytest.fixture(scope="module")
def tiers(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cpu_act")
    return {tier: _tier(tier, tmp) for tier in ("bw", "scalar")}


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_swizzled_layout_matches_native(tiers):
    import torch

    bw = tiers["bw"]
    assert bw["bw"], "the bw tier did not detect AVX-512BW"
    assert torch.equal(bw["swizzled"]["batch"], bw["native"]["batch"])


def test_batch_matches_single_token_runs(tiers):
    import torch

    for layout in ("native", "swizzled"):
        run = tiers["bw"][layout]
        assert torch.equal(run["batch"], run["single"]), layout


def test_error_against_fp32_activations_is_bounded(tiers):
    import torch

    ref = tiers["scalar"]["native"]["batch"]
    got = tiers["bw"]["native"]["batch"]
    assert torch.isfinite(got).all()
    err = _rel(got, ref)
    print(f"rel L2 vs scalar tier: {err:.5f} (bound {_bound()})")
    assert err < _bound()


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2])
