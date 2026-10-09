"""Hit rate of the RAM prefetch's speculative reads by the gate rank and margin of each pick.

Reads the InstrBuild job-trace files of a served prefetch run (``SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX``) and joins
each ``spec_submit`` (row = target row, seq = source record seq, a = expert, gen = the pick's rank << 32 | its
margin's float bits) to its ``spec_land`` (same key) and ``spec_use`` (row = target row, a = expert, c = the
entry's source seq). A read is used when a forced CPU miss swapped it in.

Timing (host CLOCK_MONOTONIC, per used read): submit->land is the read with its queueing; submit->use is the lead
the prediction had over the demand; land->use is the slack, 0 for a promoted read whose demand waited for the landing
(spec_use is stamped after the wait, so a promoted read's true need came earlier than its use event says).

Admission sweep (the expert-prediction handoff's E6/E7 on recorded reads): the run is split at the largest time
gap in its middle third (a session boundary), rules are fit on the first part and scored on the second. A per-layer
rule admits a (target layer, bin) cell when its used rate, shrunk toward the pooled rate of its bin by ``--alpha``
pseudo-reads, reaches the floor. Only issued reads are recorded, so a rule can only be scored as a subset of them,
and dropping a read is scored as losing its use: the cache effects of reads not made are not modelled.

``--fit 'OTHER.*.jsonl'`` fits the rules on another run instead and scores every read of this one.

Usage: spec_margin.py 'PREFIX.*.jsonl' [--json OUT] [--alpha 20] [--fit 'OTHER.*.jsonl']
"""

from __future__ import annotations

import argparse
import bisect
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
            picks[(e["row"], e["seq"], e["a"])] = {"layer": e["row"], "rank": rank, "margin": margin,
                                                   "group": e["group"],
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
            "outside_top_k": _row(f"rank >= {top_k}", outside), "by_rank": by_rank, "by_margin": by_margin,
            "by_layer": [_row(str(t), [p for p in picks if p["layer"] == t]) for t in sorted({p["layer"] for p in picks})],
            "by_stratum": [_row(s, [p for p in picks if stratum(p["layer"]) == s])
                           for s in sorted({stratum(p["layer"]) for p in picks})]}


FLOORS = (0.3, 0.4, 0.5, 0.6)


def stratum(layer: int) -> str:
    """The handoff's architecture strata, by target layer: Engram sits at 1 and 14, the encoder ends at 19, the
    bounded prefill replay starts at 21, DSpark taps 37-39."""
    if layer in (1, 14):
        return "enters engram"
    if layer in (2, 15):
        return "leaves engram"
    if layer == 20:
        return "19->20"
    if layer == 21:
        return "replay start"
    if layer >= 37:
        return "late"
    return "interior"


def margin_bin(p: dict) -> int:
    return bisect.bisect_right(MARGIN_EDGES, p["margin"])


def rank_margin(p: dict) -> tuple[int, int]:
    return min(p["rank"], 3), margin_bin(p)


def session_split(picks: list[dict]) -> int:
    """The submit time that opens the second part: the later side of the largest gap in the middle third."""
    t = sorted(p["submit_ns"] for p in picks)
    lo, hi = int(0.35 * len(t)), int(0.65 * len(t))
    i = max(range(lo, hi), key=lambda j: t[j + 1] - t[j])
    return t[i + 1]


def cell_rates(train: list[dict], key, alpha: float) -> dict:
    """Used rate per (layer, key) shrunk toward the pooled rate of its key; pooled rates under (None, key)."""
    pooled, cells = {}, {}
    for p in train:
        if not p["landed"]:
            continue
        for k, table in (((None, key(p)), pooled), ((p["layer"], key(p)), cells)):
            n, u = table.get(k, (0, 0))
            table[k] = (n + 1, u + p["used"])
    rates = {k: u / n for k, (n, u) in pooled.items()}
    for (layer, k), (n, u) in cells.items():
        rates[(layer, k)] = (u + alpha * rates[(None, k)]) / (n + alpha)
    return rates


def per_layer_rule(train: list[dict], key, floor: float, alpha: float):
    rates = cell_rates(train, key, alpha)
    return lambda p: rates.get((p["layer"], key(p)), rates.get((None, key(p)), 0.0)) >= floor


def pooled_rule(train: list[dict], key, floor: float):
    rates = cell_rates(train, key, 0)
    return lambda p: rates.get((None, key(p)), 0.0) >= floor


def margin_floors(train: list[dict], floor: float, alpha: float) -> dict:
    """Per target layer, the minimum margin to admit: the lowest edge of the run of admitted margin bins that starts
    at the top bin, so the live rule is one threshold per layer. A layer whose top bin fails admits nothing (inf).
    A bin with no reads anywhere carries no evidence and does not end the run."""
    rates = cell_rates(train, margin_bin, alpha)
    lows = (-float("inf"), *MARGIN_EDGES)
    floors = {}
    for layer in sorted({p["layer"] for p in train}):
        edge = float("inf")
        for b in reversed(range(len(lows))):
            rate = rates.get((layer, b), rates.get((None, b)))
            if rate is None:
                continue
            if rate < floor:
                break
            edge = lows[b]
        floors[layer] = edge
    return floors


def floor_rule(floors: dict):
    return lambda p: p["margin"] >= floors.get(p["layer"], -float("inf"))


def evaluate(label: str, rule, test: list[dict]) -> dict:
    landed = [p for p in test if p["landed"]]
    kept = [p for p in landed if rule(p)]
    used_all = sum(p["used"] for p in landed)
    used = sum(p["used"] for p in kept)
    return {"bin": label, "kept": len(kept), "kept_share": round(len(kept) / len(landed), 3) if landed else None,
            "used": used, "precision": round(used / len(kept), 3) if kept else None,
            "used_kept_share": round(used / used_all, 3) if used_all else None,
            "wasted": len(kept) - used}


def sweep(picks: list[dict], alpha: float, train: list[dict] | None = None) -> dict:
    """Fit on the first sessions and score the rest, or, given ``train`` from another run, score every pick."""
    if train is None:
        split = session_split(picks)
        train = [p for p in picks if p["submit_ns"] < split]
        test = [p for p in picks if p["submit_ns"] >= split]
    else:
        test = picks
    floors = {}
    rows = [evaluate("all (no rule)", lambda p: True, test)]
    rows += [evaluate(f"rank < {k}", lambda p, k=k: p["rank"] < k, test) for k in (1, 2, 3, 4)]
    keys = (("rank", lambda p: p["rank"]), ("margin", margin_bin), ("rank x margin", rank_margin))
    for floor in FLOORS:
        for name, key in keys:
            rows.append(evaluate(f"pooled {name} >= {floor}", pooled_rule(train, key, floor), test))
            rows.append(evaluate(f"per-layer {name} >= {floor}", per_layer_rule(train, key, floor, alpha), test))
        floors[floor] = margin_floors(train, floor, alpha)
        rows.append(evaluate(f"per-layer min margin >= {floor}", floor_rule(floors[floor]), test))
    return {"train": len(train), "test": len(test), "alpha": alpha, "rows": rows,
            "margin_floors": {str(f): {str(t): v for t, v in fl.items()} for f, fl in floors.items()}}


def sweep_table(rows: list[dict]) -> str:
    out = ["| rule | kept | kept share | used | precision | used kept share | wasted |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    out += [f"| {r['bin']} | {r['kept']} | {r['kept_share']} | {r['used']} | {r['precision']} | "
            f"{r['used_kept_share']} | {r['wasted']} |" for r in rows]
    return "\n".join(out)


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
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--fit", help="fit the admission rules on this run's trace instead of this run's first sessions")
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
    train = None
    if a.fit:
        fit_paths = sorted(glob.glob(a.fit))
        fit_events, fit_dropped = load(fit_paths) if fit_paths else ([], 0)
        if not fit_paths or fit_dropped:
            print(f"refusing --fit {a.fit}: {'no files' if not fit_paths else f'{fit_dropped} events dropped'}",
                  file=sys.stderr)
            return 1
        train = join(fit_events)
    result["sweep"] = sweep(picks, a.alpha, train)
    print(table([result["all"], result["inside_top_k"], result["outside_top_k"]]))
    print()
    print(table(result["by_rank"]))
    print()
    print(table(result["by_margin"]))
    print()
    print(table(result["by_stratum"]))
    print()
    print(table(result["by_layer"]))
    print()
    for name, q in result["timing"].items():
        print(name, json.dumps(q))
    sw = result["sweep"]
    print(f"\nadmission sweep: fit on {sw['train']} reads, scored on {sw['test']}, alpha {sw['alpha']:g}")
    print(sweep_table(sw["rows"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**result, "files": paths}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
