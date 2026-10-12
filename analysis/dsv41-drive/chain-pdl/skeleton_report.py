#!/usr/bin/env python3
"""PDL saving per layer and per step against mode 0 of the same work, from skeleton.py JSONL (pure Python).

    python3 skeleton_report.py skeleton.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import sys

GATE_US_PER_LAYER = 2.0  # plan Decision table: Task 7 saving >= 2 us per layer to run Task 8
STEP_US = 66800.0  # untraced ms/token on divix01, in us


def savings(records: list[dict]) -> list[dict]:
    """One row per PDL record (mode 1 or 2): its saving against mode 0 at the same work_ns and pre-wait spin."""
    base = {(r["work_ns"], r.get("pre_ns", 0)): r for r in records if r["mode"] == 0}
    rows = []
    for r in records:
        key = (r["work_ns"], r.get("pre_ns", 0))
        if r["mode"] == 0 or key not in base:
            continue
        step = base[key]["replay_us_p50"] - r["replay_us_p50"]
        layer = step / r["layers"]
        rows.append({"work_ns": key[0], "pre_ns": key[1], "mode": r["mode"], "per_layer_us": layer,
                     "per_step_us": step, "step_share": step / STEP_US, "gate": layer >= GATE_US_PER_LAYER})
    return sorted(rows, key=lambda x: (x["work_ns"], x["pre_ns"], x["mode"]))


def main() -> int:
    records = [json.loads(line) for path in sys.argv[1:] for line in open(path) if line.strip()]
    print("| work_ns | pre-wait ns | mode | saving per layer (us) | per step (us) | share of 66.8 ms | >= 2 us gate |")
    print("|---:|---:|---|---:|---:|---:|---|")
    for r in savings(records):
        print(f"| {r['work_ns']} | {r['pre_ns']} | {r['mode']} | {r['per_layer_us']:.3f} | {r['per_step_us']:.1f} | "
              f"{100 * r['step_share']:.3f}% | {'yes' if r['gate'] else 'no'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
