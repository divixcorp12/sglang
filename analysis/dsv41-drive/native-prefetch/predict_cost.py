"""Gate 1: the per-layer cost of the h=1 native-gate predict step inside a captured CUDA graph.

Per predicting layer (T-1 = 0..38): one ``tiny_gemm_bf16`` of x_{T-1} [1, 5120] against layer T's gate
[384, 5120] bf16 into fp32 logits, then top-6. Two top-6 variants are timed:

- ``fused_gate``: ``moe_fused_gate(..., "sqrtsoftplus", sqrtsoftplus_log1p=True)``, the router's own kernel
  (the spec's formulation);
- ``gemm_only``: the GEMV alone, the floor for any predict whose ranking is folded into another kernel.

Each layer has its own gate weight and bias, 40 of each, so the 157 MB of gates exceed the 96 MiB L2 and the
GEMV reads HBM as it does in decode (CLAUDE.md: size the working set past L2). The graph holds 39 predicts
back to back; the cost per predict is (graph time - empty graph time) / 39. A filler kernel between predicts
(``--filler``) mimics the decode stream, where each predict follows other work.

Output: one JSON line per variant with p50/p10/p90 of the per-predict cost in microseconds and the per-token
total (39 predicts).
"""

from __future__ import annotations

import argparse
import json

import torch

from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

LAYERS, EXPERTS, HIDDEN, TOPK = 40, 384, 5120, 6


def build(variant: str, x, W, bias, logits, filler_src, filler_dst):
    def body():
        for t in range(1, LAYERS):
            if filler_src is not None:
                filler_dst.copy_(filler_src)
            tiny_gemm_bf16(x, W[t], logits[t], max_m=16)
            if variant == "fused_gate":
                moe_fused_gate(logits[t], bias[t], TOPK, scoring_func="sqrtsoftplus", renormalize=True,
                               sqrtsoftplus_log1p=True)

    return body


def time_graph(body, reps: int) -> list[float]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        body()
    torch.cuda.synchronize()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def quantiles(values: list[float]) -> dict:
    t = torch.tensor(values)
    return {q: float(torch.quantile(t, p)) for q, p in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--filler", action="store_true", help="a 4 MiB device copy before each predict")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    x = torch.randn(1, HIDDEN, device=dev).to(torch.bfloat16)
    W = torch.randn(LAYERS, EXPERTS, HIDDEN, device=dev).mul_(0.02).to(torch.bfloat16)
    bias = torch.randn(LAYERS, EXPERTS, device=dev).mul_(0.1).to(torch.bfloat16)
    logits = torch.empty(LAYERS, 1, EXPERTS, device=dev, dtype=torch.float32)
    filler_src = filler_dst = None
    if args.filler:
        filler_src = torch.empty(4 << 20, dtype=torch.uint8, device=dev)
        filler_dst = torch.empty_like(filler_src)
    gpu = torch.cuda.get_device_name()
    base = None
    if args.filler:
        base = time_graph(lambda: [filler_dst.copy_(filler_src) for _ in range(1, LAYERS)], args.reps)
    else:
        base = time_graph(lambda: None, args.reps)
    base50 = quantiles(base)["p50"]
    for variant in ("gemm_only", "fused_gate"):
        times = time_graph(build(variant, x, W, bias, logits, filler_src, filler_dst), args.reps)
        q = quantiles(times)
        per = {k: (v - base50) / (LAYERS - 1) for k, v in q.items()}
        print(json.dumps({
            "gpu": gpu, "variant": variant, "filler": args.filler, "reps": args.reps,
            "graph_us": q, "empty_graph_p50_us": base50, "per_predict_us": per,
            "per_token_ms_p50": per["p50"] * (LAYERS - 1) / 1000.0,
        }), flush=True)


if __name__ == "__main__":
    main()
