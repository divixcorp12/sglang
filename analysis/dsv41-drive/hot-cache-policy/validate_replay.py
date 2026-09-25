"""Does tier_sim's DIRECT replay reproduce the run? Per decode step and layer, simulated against logged misses.

Three measured references, from the same trace: the route log's per-forward plan counts (``misses``), the
graph_step lines' routed-miss register deltas (what G has always been read from), and, where logged, the
hot set each layer held at each forward's start. The replay starts from the framework's unseeded startup
residency (experts 0..capacity-1), or with ``--initial-from-log`` from the first logged hot set (the
warmup and capture gathers insert dummy routes the log cannot see).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "dsv41"))
import tier_sim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("--initial-from-log", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    loaded = tier_sim.load_forwards(args.trace)
    initial = None
    if args.initial_from_log:
        first = next(
            f for f in loaded["forwards"]
            if f["kind"] == "graph" and f["phase"] != "capture" and f.get("hot") is not None
        )
        initial = {layer: list(experts) for layer, experts in first["hot"].items()}
    out = tier_sim.replay_direct(loaded, initial=initial)
    steps = out["per_forward"]
    step_equal = sum(1 for _, sim, measured in steps if sim == measured)
    cell_diff = [
        (seq, layer, sim[layer], measured[layer])
        for seq, sim, measured in steps
        for layer in sim
        if sim[layer] != measured[layer]
    ]
    graph_steps = [c for c in tier_sim.load_trace(args.trace) if c.get("kind") == "graph_step"]
    registers = {
        "lines": len(graph_steps),
        "steps": sum(c.get("steps", 1) for c in graph_steps),
        "vram_miss": sum(c["vram_miss"] for c in graph_steps),
    }
    phases = {}
    for forward in loaded["forwards"]:
        key = f"{forward['kind']}:{forward['phase']}"
        phases[key] = phases.get(key, 0) + 1
    summary = {
        "trace": args.trace,
        "run": loaded["run"],
        "schema": loaded["schema"],
        "dropped": loaded["dropped"],
        "forwards_by_kind_phase": phases,
        "requests": len({f["rids"][0] for f in loaded["forwards"] if f["rids"]}),
        "decode_tokens": out["decode_tokens"],
        "sim_G": round(out["G"], 3),
        "measured_G_route_log": round(out["measured_G"], 3),
        "measured_G_graph_step_registers": round(registers["vram_miss"] / registers["steps"], 3) if registers["steps"] else None,
        "graph_step_registers": registers,
        "steps_equal": step_equal,
        "layer_steps_different": len(cell_diff),
        "first_differences": cell_diff[:10],
        "sum_abs_difference": sum(abs(a - b) for _, _, a, b in cell_diff),
        "hot_checked": out["hot_checked"],
        "hot_mismatched": out["hot_mismatched"],
        "first_hot_mismatch": out["first_hot_mismatch"],
        "truncated": out["truncated"],
        "initial": "first logged hot set" if initial else "framework unseeded allocation",
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
