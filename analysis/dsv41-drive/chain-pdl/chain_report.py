#!/usr/bin/env python3
"""The real-chain PDL probe (chain_pdl.py JSONL): savings per scenario, and the stamp ring's edge gaps. Pure Python.

    python3 chain_report.py chain.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

KERNELS = {1: "post", 2: "W1", 3: "A", 4: "S", 5: "CW", 6: "F"}  # lease_device.cuh TestPdlEntry ids
GATE_US_PER_LAYER = 2.0  # plan Decision table: Task 7/8 saving that justifies the next step
SHIP_SHARE = 0.01  # 1% of untraced ms/token to put a ship decision to the user
STEP_US = 66800.0
LAYERS = 40


def replays(stamps: list[list[int]]) -> list[list[dict]]:
    """Stamps [kernel | block << 8, entry, waited, exit] in entry order, split into replays at each post (kernel 1),
    consecutive blocks of one kernel merged into one launch (entry min, waited and exit max)."""
    out: list[list[dict]] = []
    for word, entry, waited, exit_ in sorted(stamps, key=lambda s: s[1]):
        kernel = word & 0xFF
        if kernel == 1 or not out:
            out.append([])
        cur = out[-1]
        if cur and cur[-1]["kernel"] == kernel and kernel != 1 and entry <= cur[-1]["exit"]:
            last = cur[-1]
            last.update(entry=min(last["entry"], entry), waited=max(last["waited"], waited),
                        exit=max(last["exit"], exit_), blocks=last["blocks"] + 1)
        else:
            cur.append({"kernel": kernel, "entry": entry, "waited": waited, "exit": exit_, "blocks": 1})
    return out


def _labels(replay: list[dict]) -> list[str]:
    labels, a = [], 0
    for launch in replay:
        name = KERNELS.get(launch["kernel"], str(launch["kernel"]))
        if name == "A":
            a += 1
            name = f"A{a}"
        labels.append(name)
    return labels


def edges(reps: list[list[dict]]) -> dict[str, list[dict]]:
    """Per edge between consecutive stamped launches: the successor's entry, and the start of its body (after its
    PDL wait), each relative to the predecessor's exit (ns; negative = before the predecessor finished)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for replay in reps:
        labels = _labels(replay)
        for i in range(1, len(replay)):
            pred, succ = replay[i - 1], replay[i]
            out[f"{labels[i - 1]}->{labels[i]}"].append({"entry_after_exit": succ["entry"] - pred["exit"],
                                                         "body_after_exit": succ["waited"] - pred["exit"]})
    return dict(out)


def edge_summary(reps: list[list[dict]]) -> dict[str, dict]:
    return {name: {"n": len(v),
                   "entry_after_exit_p50": statistics.median(x["entry_after_exit"] for x in v),
                   "body_after_exit_p50": statistics.median(x["body_after_exit"] for x in v)}
            for name, v in edges(reps).items()}


def prologue(reps: list[list[dict]]) -> dict[str, dict]:
    """Per kernel, entry-to-body time (waited - entry) of the launches that entered after their predecessor exited, so
    the PDL wait returned at once: the kernel's pre-wait prologue plus the wait instruction. A replay's first launch
    has no stamped predecessor and always counts."""
    got: dict[str, list[int]] = defaultdict(list)
    for replay in reps:
        labels = _labels(replay)
        for i, launch in enumerate(replay):
            if i == 0 or launch["entry"] >= replay[i - 1]["exit"]:
                got[labels[i]].append(launch["waited"] - launch["entry"])
    return {name: {"n": len(v), "p50_ns": statistics.median(v)} for name, v in got.items()}


def savings(records: list[dict]) -> list[dict]:
    """Per scenario and mode, the median replay over its records (one per round) against off's median."""
    times: dict[tuple, list[float]] = {}
    for r in records:
        if r.get("kind") == "chain":
            times.setdefault((r["scenario"], r["mode"]), []).append(r["replay_us_p50"])
    rows = []
    for (scenario, mode), ts in times.items():
        if mode == "off" or (scenario, "off") not in times:
            continue
        layer = statistics.median(times[(scenario, "off")]) - statistics.median(ts)  # one replay = one layer
        step = layer * LAYERS
        rows.append({"scenario": scenario, "mode": mode, "rounds": len(ts), "per_layer_us": layer,
                     "per_step_us": step, "step_share": step / STEP_US, "gate": layer >= GATE_US_PER_LAYER,
                     "ship": step / STEP_US >= SHIP_SHARE})
    return rows


def main() -> int:
    records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    print("| scenario | mode | saving per layer (us) | per 40-layer step (us) | share of 66.8 ms | >= 2 us gate | >= 1% ship bar |")
    print("|---|---|---:|---:|---:|---|---|")
    for r in savings(records):
        print(f"| {r['scenario']} | {r['mode']} | {r['per_layer_us']:.2f} | {r['per_step_us']:.1f} | "
              f"{100 * r['step_share']:.3f}% | {'yes' if r['gate'] else 'no'} | {'yes' if r['ship'] else 'no'} |")
    for r in records:
        if r.get("kind") == "stamps":
            print(f"\nstamps {r['scenario']} / {r['mode']}:")
            for name, e in r["edges"].items():
                print(f"  {name}: n {e['n']}, entry after exit p50 {e['entry_after_exit_p50']} ns, "
                      f"body after exit p50 {e['body_after_exit_p50']} ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
