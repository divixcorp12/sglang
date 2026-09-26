"""Native-gate rankings for the pinned-tier prefetch replay: layer T's gate on layer T-h's router input.

For every decode step of a router capture (router_score.py's inputs) and every target layer T, writes the top
``depth`` experts of gate_T(x_src) by biased score, for h = 0..H:

- h >= 1: x_src is the same token's layer T-h input, or for T < h the same request's previous token at layer
  40+T-h (prefetch_sim.source_of). Missing sources are marked invalid.
- h = 0: layer T's gate on its own input. Ranks 1..6 are the routes; ranks 7.. are the near misses that the
  next-token predictor uses for step s+1.

Output (npz): ``order[steps, layers, H+1, depth]`` int16, ``valid[steps, layers, H+1]``, ``seq[steps]`` (the
graph forward seq of each decode step, for joining to the route log), and ``h0_route_match`` (the self-check).

Usage: gate_rankings.py STAGES_JSONL ROUTER_PREFIX MODEL_DIR --out rank.npz [--horizons 4] [--depth 12]
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
import prefetch_sim  # noqa: E402
import router_score  # noqa: E402
import tier_sim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stages")
    p.add_argument("router_prefix")
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", type=int, default=4)
    p.add_argument("--depth", type=int, default=12)
    a = p.parse_args()

    capture = router_score.load_capture(a.router_prefix)
    # Gates first: safetensors maps whole shard files, which must not coexist with the loaded trace under ulimit -v.
    W, bias = router_score.load_gates(a.model, capture.header["layer_ids"])
    loaded = tier_sim.load_forwards(a.stages)
    stream = prefetch_sim.decode_stream(loaded)
    if stream.layers != capture.header["layer_ids"]:
        raise ValueError("route log and router capture disagree on the layers")
    records = router_score.decode_records(loaded, capture)
    seqs = np.asarray([f["seq"] for f in loaded["forwards"] if f["phase"] == "decode"], dtype=np.int64)
    del loaded
    steps, layers = len(stream.rids), len(stream.layers)
    H, depth = a.horizons, a.depth
    order = np.zeros((steps, layers, H + 1, depth), dtype=np.int16)
    valid = np.zeros((steps, layers, H + 1), dtype=bool)

    # x per source layer, loaded once per layer (fp32 [steps, hidden], ~126 MB).
    for src_layer in range(layers):
        x = torch.from_numpy(router_score.bf16_to_f32(np.asarray(capture.x[records, src_layer])))
        for h in range(H + 1):
            # Targets whose source is this layer at horizon h: same step T = src + h, or next step T = src + h - 40.
            for target, same_step in ((src_layer + h, True), (src_layer + h - layers, False)):
                if not (0 <= target < layers) or (h == 0 and not same_step):
                    continue
                biased = torch.nn.functional.softplus(x @ W[target].T).sqrt() + bias[target]
                top = torch.topk(biased, depth, dim=-1).indices.numpy().astype(np.int16)
                if same_step:
                    order[:, target, h] = top
                    valid[:, target, h] = True
                else:
                    # Step s's target draws on step s-1's source, when s continues the same request.
                    ok = stream.continuous.copy()
                    order[1:, target, h][ok[1:]] = top[:-1][ok[1:]]
                    valid[1:, target, h] = ok[1:]
        print(f"layer {src_layer} done", flush=True)

    routes = stream.routes
    match = np.mean([
        set(order[s, t, 0, :6].tolist()) == set(routes[s, t].tolist()) for s in range(steps) for t in range(layers)
    ])
    print(f"h=0 self-check: route sets reproduced {match:.6f}")
    np.savez_compressed(a.out, order=order, valid=valid, seq=seqs, h0_route_match=match)


if __name__ == "__main__":
    main()
