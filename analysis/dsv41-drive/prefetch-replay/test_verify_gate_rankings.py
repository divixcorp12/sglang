"""verify_gate_rankings.py: per-token next-layer rankings of a verify capture. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".." / "router-capture"))

import router_score  # noqa: E402
from verify_gate_rankings import rank_verify  # noqa: E402

LAYERS, EXPERTS, HIDDEN, TOPK, M = 3, 8, 4, 2, 3


def _gates(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(LAYERS, EXPERTS, HIDDEN, generator=g), torch.randn(LAYERS, EXPERTS, generator=g)


def _biased(W, bias, layer, x):
    return torch.nn.functional.softplus(x @ W[layer].T).sqrt() + bias[layer]


def _capture(tmp_path, W, bias, tokens_per_step):
    """A capture whose ids are each token's true top-k of its own layer, so h=0 reproduces them."""
    prefix = str(tmp_path / "router")
    g = torch.Generator().manual_seed(1)
    steps = len(tokens_per_step)
    x = torch.randn(steps, LAYERS, M, HIDDEN, generator=g).to(torch.bfloat16)
    ids = torch.full((steps, LAYERS, M, TOPK), -1, dtype=torch.int32)
    w = torch.zeros(steps, LAYERS, M, TOPK)
    for s in range(steps):
        for layer in range(LAYERS):
            top = torch.topk(_biased(W, bias, layer, x[s, layer].float()), TOPK, dim=-1)
            ids[s, layer], w[s, layer] = top.indices.int(), top.values
    with open(prefix + ".json", "w") as f:
        json.dump({"schema": 2, "layer_ids": list(range(LAYERS)), "tokens": M, "hidden": HIDDEN, "topk": TOPK}, f)
    x.view(torch.int16).numpy().astype(np.uint16).tofile(prefix + ".x.bin")
    ids.numpy().tofile(prefix + ".ids.bin")
    w.numpy().tofile(prefix + ".w.bin")
    np.stack([np.arange(steps), np.arange(steps) + 10, np.asarray(tokens_per_step)], axis=1).astype(np.int64).tofile(prefix + ".seq.bin")
    forwards = [{"kind": "graph", "seq": s, "phase": "target_verify", "verify": True, "tokens": int(t),
                 "router": s, "rids": ["a"], "routes": {layer: sorted(set(ids[s, layer, :t].flatten().tolist())) for layer in range(LAYERS)},
                 "hot": None, "misses": {}} for s, t in enumerate(tokens_per_step)]
    return {"layer_ids": list(range(LAYERS)), "forwards": forwards}, router_score.load_capture(prefix), x


def test_h0_reproduces_each_live_tokens_routes_and_h1_scores_the_next_layer_on_this_layers_input(tmp_path):
    W, bias = _gates()
    loaded, cap, x = _capture(tmp_path, W, bias, [3, 2])
    out = rank_verify(loaded, cap, W, bias, horizons=1, depth=TOPK)
    assert out["order"].shape == (2, LAYERS, M, 2, TOPK) and out["tokens"].tolist() == [3, 2]
    assert out["h0_route_match"] == 1.0
    want = torch.topk(_biased(W, bias, 1, x[0, 0, 1].float()), TOPK).indices.tolist()
    assert out["order"][0, 1, 1, 1].tolist() == want  # step 0, target layer 1, token 1, h=1: gate_1 on layer 0's x
    assert out["valid"][0, 0, 1] == False and out["valid"][0, 1, 1] == True  # noqa: E712


def test_live_tokens_only_are_ranked(tmp_path):
    """Padding rows past the forward's live count must not rank: their routes are garbage."""
    W, bias = _gates()
    loaded, cap, _ = _capture(tmp_path, W, bias, [2])
    out = rank_verify(loaded, cap, W, bias, horizons=1, depth=TOPK)
    assert (out["order"][0, :, 2] == -1).all()
    live = out["order"][0, :, :2].transpose(0, 2, 1, 3)  # [layers, H+1, live tokens, depth]
    assert (live[out["valid"][0]] >= 0).all()  # target 0 at h=1 has no source layer and stays -1
