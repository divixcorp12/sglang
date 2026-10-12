"""B1 scorer cost: bf16 bytes and CUDA-graph median time for one layer (C6, S1.2).

Times one forward (PCA projection, feature build, MLP, embeddings) on 6 tokens, replayed from a CUDA graph with
random inputs; the head runs in fp32 here, bytes are reported at bf16 as head_bytes_bf16 counts them.
Inputs: a head saved by b1_head.save_head. Run under cc-gpu.lock (cores 32-63) so the timing is not shared.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python b1_cost.py HEAD_PATH --out cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training import llapor  # noqa: E402

X_DIM = 5120
TOP_K = 6
BUDGET_MS = 0.1  # the 0.1 ms scorer budget of Design S1.2


def _make_step(model: b1_head.B1Head, *, tokens: int, device: str):
    """Static inputs and a no-grad closure running the full scorer once on them."""
    cfg = model.cfg
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn((tokens, X_DIM), generator=gen).to(device)
    ids = torch.randint(0, cfg.num_experts, (tokens, TOP_K), generator=gen).to(device)
    weights = torch.rand((tokens, TOP_K), generator=gen).to(device)
    gate = torch.rand((tokens, 2 * cfg.num_experts), generator=gen).to(device)
    zero = torch.zeros(tokens, dtype=torch.int64, device=device)
    y = torch.zeros((tokens, cfg.num_experts), device=device)

    def step() -> torch.Tensor:
        u = llapor.encode_features(x, ids, weights, pca=model.pca, num_experts=cfg.num_experts)
        if cfg.variant == "E5":
            u = torch.cat([u, gate], dim=-1)
        batch = b1_head.HeadBatch(u=u, source_layer=zero, target_layer=zero, stratum=zero, y=y, hard_neg=y)
        return torch.sigmoid(model(batch))

    return step


def _replay_times_ms(graph: torch.cuda.CUDAGraph, *, runs: int) -> np.ndarray:
    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times[i] = start.elapsed_time(end)
    return times


def measure_cost(model: b1_head.B1Head, *, tokens: int = 6, runs: int = 1000, warmup: int = 50,
                 device: str = "cuda") -> dict:
    """bf16 bytes plus median and p90 replay time of one scorer forward over ``tokens`` tokens."""
    model = model.to(device).eval()
    step = _make_step(model, tokens=tokens, device=device)
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(warmup, 3)):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        times = _replay_times_ms(graph, runs=runs)
    median_ms = float(np.median(times))
    return {
        "bytes_bf16": int(b1_head.head_bytes_bf16(model)),
        "median_ms": median_ms,
        "p90_ms": float(np.percentile(times, 90)),
        "runs": runs,
        "tokens": tokens,
        "within_0_1ms": median_ms <= BUDGET_MS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("head_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args(argv)
    model, sha = b1_head.load_head(args.head_path)
    result = measure_cost(model, tokens=args.tokens, runs=args.runs, warmup=args.warmup)
    result["head_sha256"] = sha
    result["gpu"] = torch.cuda.get_device_name()
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
