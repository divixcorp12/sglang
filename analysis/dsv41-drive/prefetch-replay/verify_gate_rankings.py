"""Per-token next-layer gate rankings of a DSpark verify capture, for verify_replay.py.

For every verify forward (step), target layer T, live token m and horizon h = 0..H, the top ``depth`` experts of
gate_T(x[step, T-h, m]) by biased score sqrt(softplus(W x)) + b, the score the model routes on. h = 0 is the
self-check against the captured top-k ids; h >= 1 is the predictor: layer T's gate on the same token's layer T-h
input. There is no cross-step source (the next verify's tokens come from a draft that has not run), so targets
T < h are invalid. Padding tokens (m >= tokens[step]) rank -1.

Output (npz): ``order`` int16 [steps, layers, M, H+1, depth], ``score`` float32 [same], ``valid`` bool
[steps, layers, H+1], ``tokens`` int64 [steps], ``seq`` int64 [steps], ``h0_route_match``.

Usage: verify_gate_rankings.py TRACE_JSONL ROUTER_PREFIX MODEL_DIR --out rank.npz [--horizons 2] [--depth 12]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "router-capture"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
import router_score  # noqa: E402
import tier_sim  # noqa: E402
from gate_rankings import load_gates  # noqa: E402


def verify_steps(loaded: dict) -> list[dict]:
    return [f for f in loaded["forwards"] if f.get("verify")]


def rank_verify(loaded: dict, capture, W: torch.Tensor, bias: torch.Tensor, *, horizons: int, depth: int) -> dict:
    steps = verify_steps(loaded)
    records = np.asarray([capture.record_of_seq[f["seq"]] for f in steps], dtype=np.int64)
    tokens = capture.tokens[records]
    n, layers, M = len(steps), len(loaded["layer_ids"]), int(capture.x.shape[2])
    H = horizons
    order = np.full((n, layers, M, H + 1, depth), -1, dtype=np.int16)
    score = np.zeros((n, layers, M, H + 1, depth), dtype=np.float32)
    valid = np.zeros((n, layers, H + 1), dtype=bool)
    live = np.arange(M)[None, :] < tokens[:, None]  # [steps, M]
    for src in range(layers):
        x = torch.from_numpy(router_score.bf16_to_f32(np.asarray(capture.x[records, src]))).reshape(n * M, -1)
        for h in range(H + 1):
            target = src + h
            if target >= layers:
                continue
            biased = torch.nn.functional.softplus(x @ W[target].T).sqrt() + bias[target]
            top = torch.topk(biased, depth, dim=-1)
            o = top.indices.numpy().astype(np.int16).reshape(n, M, depth)
            sc = top.values.numpy().astype(np.float32).reshape(n, M, depth)
            o[~live] = -1
            sc[~live] = 0.0
            order[:, target, :, h] = o
            score[:, target, :, h] = sc
            valid[:, target, h] = True
    ids = capture.ids[records]  # [steps, layers, M, topk]
    topk = ids.shape[-1]
    matches = [
        set(order[s, t, m, 0, :topk].tolist()) == set(ids[s, t, m].tolist())
        for s in range(n) for t in range(layers) for m in range(int(tokens[s]))
    ]
    return {
        "order": order, "score": score, "valid": valid, "tokens": tokens.astype(np.int64),
        "seq": np.asarray([f["seq"] for f in steps], dtype=np.int64),
        "h0_route_match": float(np.mean(matches)) if matches else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("router_prefix")
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", type=int, default=2)
    p.add_argument("--depth", type=int, default=12)
    a = p.parse_args()
    capture = router_score.load_capture(a.router_prefix)
    loaded = tier_sim.load_forwards(a.trace)
    if loaded["layer_ids"] != capture.header["layer_ids"]:
        raise ValueError("route log and router capture disagree on the layers")
    W, bias = load_gates(a.model, capture.header["layer_ids"])
    out = rank_verify(loaded, capture, W, bias, horizons=a.horizons, depth=a.depth)
    print(f"verify steps {len(out['seq'])}; h=0 self-check: route sets reproduced {out['h0_route_match']:.6f}")
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
