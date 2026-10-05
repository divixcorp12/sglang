#!/usr/bin/env python3
"""DSpark verify in the decode graph, offline: the per-layer expert union of a multi-token verify, and its tok/s.

A verify of ``width`` tokens routes every token through each layer, and the layer's gather moves the union of their
experts. ``window_forwards`` turns a one-token decode trace (``tier_sim.load_forwards``) into verify forwards: each
window is ``width`` consecutive decode tokens of one request, and the next window starts ``stride`` tokens later
(the accept length). The true next tokens' routes stand in for the draft tokens' (teacher forcing). A window never
spans a request or an eager forward. Residency decays per forward, so a verify decays the insert scores as one token
(the simulator's ``graph_forward`` counts one); that is a small bias toward stickier residency.

``project`` replays the windows through ``cpu_expert_sim.replay_nm`` with CPU experts off (v1, DSV41_REFERENCE.md
§33.3), and ``baseline`` is plain decode with CPU hits on the same trace and cost model (§31.2).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _rid(forward: dict) -> tuple:
    return tuple(forward.get("rids") or ())


def _merge(window: list[dict]) -> dict:
    first = window[0]
    routes = {layer: list(dict.fromkeys(e for f in window for e in f["routes"][layer])) for layer in first["routes"]}
    misses = {layer: sum(f["misses"].get(layer, 0) for f in window) for layer in first["misses"]}
    return {**first, "tokens": len(window), "routes": routes, "misses": misses}


def window_forwards(loaded: dict, width: int, stride: int) -> dict:
    """``loaded`` with every run of one request's decode forwards replaced by its verify windows."""
    if width < 1 or not 1 <= stride <= width:
        raise ValueError(f"want width >= 1 and stride in 1..width, got width {width}, stride {stride}")
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        out.extend(_merge(run[start : start + width]) for start in range(0, len(run), stride))
        run.clear()

    for forward in loaded["forwards"]:
        decode = forward["kind"] == "graph" and forward["phase"] == "decode"
        if decode and run and _rid(forward) == _rid(run[0]):
            run.append(forward)
            continue
        flush()
        if decode:
            run.append(forward)
        else:
            out.append(forward)
    flush()
    return {**loaded, "forwards": out}


def union_stats(loaded: dict) -> dict:
    """Distinct experts per decode forward (a verify, once windowed) and layer."""
    sizes = np.array(
        [
            [len(set(r)) for r in f["routes"].values()]
            for f in loaded["forwards"]
            if f["kind"] == "graph" and f["phase"] == "decode"
        ],
        dtype=np.int64,
    )
    flat = sizes.reshape(-1)
    return {
        "verifies": int(sizes.shape[0]),
        "mean": float(flat.mean()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": int(flat.max()),
        "per_layer_mean": [float(x) for x in sizes.mean(axis=0)],
    }


def shrink_hot(loaded: dict, slots: int) -> dict:
    """``loaded`` with ``slots`` fewer hot slots per layer: the VRAM a resident draft takes from the target."""
    if slots == 0:
        return loaded
    capacity = {layer: c - slots for layer, c in loaded["hot_capacity"].items()}
    if min(capacity.values()) < 1:
        raise ValueError(f"taking {slots} slots leaves a layer with {min(capacity.values())}")
    return {**loaded, "hot_capacity": capacity}
