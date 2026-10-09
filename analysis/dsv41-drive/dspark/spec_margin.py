"""Hit rate of the RAM prefetch's speculative reads by the gate rank and margin of each pick.

Reads the InstrBuild job-trace files of a served prefetch run (``SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX``) and joins
each ``spec_submit`` (row = target row, seq = source record seq, a = expert, gen = the pick's rank << 32 | its
margin's float bits) to its ``spec_land`` (same key) and ``spec_use`` (row = target row, a = expert, c = the
entry's source seq). A read is used when a forced CPU miss swapped it in.

Timing (host CLOCK_MONOTONIC, per used read): submit->land is the read with its queueing; submit->use is the lead
the prediction had over the demand; land->use is the slack, 0 for a promoted read whose demand waited for the landing
(spec_use is stamped after the wait, so a promoted read's true need came earlier than its use event says).

Usage: spec_margin.py 'PREFIX.*.jsonl' [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import struct
import sys

MARGIN_EDGES = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)


def unpack_gen(gen: int) -> tuple[int, float]:
    """(rank, margin) of a spec_submit's gen field."""
    return gen >> 32, struct.unpack("<f", struct.pack("<I", gen & 0xFFFFFFFF))[0]


def load(paths: list[str]) -> tuple[list[dict], int]:
    events, dropped = [], 0
    for path in paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if "dropped" in record:
                    dropped += int(record["dropped"])
                elif record.get("event", "").startswith("spec_"):
                    events.append(record)
    return events, dropped


def join(events: list[dict]) -> list[dict]:
    """One dict per submitted read: rank, margin, group, landed, used."""
    picks = {}
    for e in events:
        if e["event"] == "spec_submit":
            rank, margin = unpack_gen(int(e["gen"]))
            picks[(e["row"], e["seq"], e["a"])] = {"rank": rank, "margin": margin, "group": e["group"],
                                                   "landed": False, "used": False, "submit_ns": e["ns"],
                                                   "land_ns": None, "use_ns": None}
    for e in events:
        if e["event"] == "spec_land":
            pick = picks.get((e["row"], e["seq"], e["a"]))
            if pick is not None:
                pick["landed"] = True
                pick["land_ns"] = e["ns"]
        elif e["event"] == "spec_use":
            pick = picks.get((e["row"], e["c"], e["a"]))
            if pick is not None:
                pick["used"] = True
                pick["use_ns"] = e["ns"]
    return list(picks.values())


def _row(label: str, picks: list[dict]) -> dict:
    landed = sum(p["landed"] for p in picks)
    used = sum(p["used"] for p in picks)
    return {"bin": label, "submitted": len(picks), "landed": landed, "used": used,
            "precision": round(used / landed, 3) if landed else None}


def bins(picks: list[dict], top_k: int) -> dict:
    by_rank = [_row(str(r), [p for p in picks if p["rank"] == r]) for r in sorted({p["rank"] for p in picks})]
    edges = (-float("inf"), *MARGIN_EDGES, float("inf"))
    by_margin = [_row(f"[{lo:g}, {hi:g})", [p for p in picks if lo <= p["margin"] < hi])
                 for lo, hi in zip(edges, edges[1:])]
    inside = [p for p in picks if p["rank"] < top_k]
    outside = [p for p in picks if p["rank"] >= top_k]
    return {"all": _row("all", picks), "inside_top_k": _row(f"rank < {top_k}", inside),
            "outside_top_k": _row(f"rank >= {top_k}", outside), "by_rank": by_rank, "by_margin": by_margin}


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda f: round(v[min(len(v) - 1, int(f * len(v)))], 3)
    return {"n": len(v), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "min": round(v[0], 3)}


def timing(picks: list[dict]) -> dict:
    """Milliseconds: the read, the lead and the slack of every used read, and the read of every landed one."""
    landed = [p for p in picks if p["land_ns"] is not None]
    used = [p for p in landed if p["use_ns"] is not None]
    ms = lambda a, b: (b - a) / 1e6
    return {
        "read_ms (landed)": _quantiles([ms(p["submit_ns"], p["land_ns"]) for p in landed]),
        "lead_ms submit->use (used)": _quantiles([ms(p["submit_ns"], p["use_ns"]) for p in used]),
        "slack_ms land->use (used)": _quantiles([ms(p["land_ns"], p["use_ns"]) for p in used]),
    }


def table(rows: list[dict]) -> str:
    out = ["| bin | submitted | landed | used | precision |", "|---|---:|---:|---:|---:|"]
    out += [f"| {r['bin']} | {r['submitted']} | {r['landed']} | {r['used']} | {r['precision']} |" for r in rows]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--json")
    a = p.parse_args()
    paths = sorted(glob.glob(a.pattern))
    if not paths:
        print(f"no files match {a.pattern}", file=sys.stderr)
        return 2
    events, dropped = load(paths)
    if dropped:
        print(f"refusing: {dropped} events dropped (trace buffer overflow)", file=sys.stderr)
        return 1
    picks = join(events)
    if not picks:
        print("no spec_submit events", file=sys.stderr)
        return 1
    result = bins(picks, a.top_k)
    result["timing"] = timing(picks)
    print(table([result["all"], result["inside_top_k"], result["outside_top_k"]]))
    print()
    print(table(result["by_rank"]))
    print()
    print(table(result["by_margin"]))
    print()
    for name, q in result["timing"].items():
        print(name, json.dumps(q))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**result, "files": paths}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
