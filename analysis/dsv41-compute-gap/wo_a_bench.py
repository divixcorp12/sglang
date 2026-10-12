"""E1 (handoff C2a): TP1 BS1 wo_a, einsum fallback vs the standalone Triton GEMV.

Shape: x [1, 8, 4096] bf16, weight [8, 1024, 4096] bf16 per layer (64 MiB). Forty distinct layer
weights (2.5 GiB) are cycled so no call is served from the 96 MiB L2. Each candidate is captured as
one CUDA graph running all 40 layers back to back and timed by replay; per-call time is replay/40.

Usage: python wo_a_bench.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.wo_a import wo_a_bf16_gemv

LAYERS, G, R, D = 40, 8, 1024, 4096


@triton.jit
def _gemv_bn(X, W, Y, R: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, D)
    x = tl.load(X + group * D + columns).to(tl.float32)
    w = tl.load(W + (group * R + rows[:, None]) * D + columns[None, :]).to(tl.float32)
    tl.store(Y + group * R + rows, tl.sum(w * x[None, :], axis=1))


def gemv_bn(x, w, out, bn, warps):
    _gemv_bn[(R // bn, G)](x, w, out, R, D, bn, num_warps=warps, enable_fp_fusion=False)
    return out


def time_graph(fn, trials=30, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(trials):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        per_call.append(a.elapsed_time(b) * 1000 / reps / LAYERS)
    return {
        "median_us": statistics.median(per_call),
        "min_us": min(per_call),
        "max_us": max(per_call),
        "stdev_us": statistics.pstdev(per_call),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    ws = [torch.randn(G, R, D, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(LAYERS)]
    xs = [torch.randn(1, G, D, device=dev, dtype=torch.bfloat16) for _ in range(LAYERS)]
    outs = [torch.empty(1, G, R, device=dev, dtype=torch.bfloat16) for _ in range(LAYERS)]

    def einsum_all():
        for l in range(LAYERS):
            outs[l].copy_(torch.einsum("tgd,grd->tgr", xs[l], ws[l]))

    def einsum_out_all():
        # What production actually runs: the einsum result is consumed directly.
        for l in range(LAYERS):
            torch.einsum("tgd,grd->tgr", xs[l], ws[l])

    def gemv_all():
        for l in range(LAYERS):
            wo_a_bf16_gemv(xs[l], ws[l])

    cands = {"einsum": einsum_out_all, "einsum+copy": einsum_all, "triton_bn1_w4(stock)": gemv_all}
    for bn in (1, 2, 4):
        for warps in (2, 4, 8):
            name = f"triton_bn{bn}_w{warps}"
            if name == "triton_bn1_w4":
                continue
            cands[name] = (lambda bn=bn, warps=warps: [gemv_bn(xs[l], ws[l], outs[l], bn, warps) for l in range(LAYERS)])

    bytes_per_call = G * R * D * 2
    res = {"shape": [G, R, D], "layers": LAYERS, "device": torch.cuda.get_device_name(), "candidates": {}}
    # Interleave twice to expose drift.
    for rnd in range(2):
        for name, fn in cands.items():
            t = time_graph(fn)
            t["TBps"] = bytes_per_call / (t["median_us"] * 1e-6) / 1e12
            res["candidates"].setdefault(name, []).append(t)
            print(f"round{rnd} {name:24s} {t['median_us']:7.2f} us/call  {t['TBps']:.3f} TB/s", flush=True)

    # Numerics on realistic-magnitude inputs, reference = the production einsum.
    ref = torch.einsum("tgd,grd->tgr", xs[0], ws[0])
    fp32 = torch.einsum("tgd,grd->tgr", xs[0].float(), ws[0].float())
    num = {}
    for name, fn1 in {
        "stock": lambda: wo_a_bf16_gemv(xs[0], ws[0]),
        **{f"bn{bn}_w{w}": (lambda bn=bn, w=w: gemv_bn(xs[0], ws[0], torch.empty_like(ref), bn, w).clone()) for bn in (1, 2, 4) for w in (2, 4, 8)},
    }.items():
        y = fn1()
        num[name] = {
            "bitwise_equal_to_einsum": bool(torch.equal(y, ref)),
            "frac_elems_differ": float((y != ref).float().mean()),
            "max_abs_vs_einsum": float((y.float() - ref.float()).abs().max()),
            "max_abs_vs_fp32": float((y.float() - fp32).abs().max()),
        }
    num["einsum_max_abs_vs_fp32"] = float((ref.float() - fp32).abs().max())
    res["numerics"] = num
    print(json.dumps(num, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
