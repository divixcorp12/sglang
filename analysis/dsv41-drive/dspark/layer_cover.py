"""How completely the RAM prefetch covers each layer's forced misses, and what a covered read is worth.

Reads a served run's InstrBuild job trace (the ``job.prefix`` of ``SGLANG_EXPERT_STREAM_TRACE``). A record is one (row, gen), as in
``layer_misses.py``: ``copy_submit.b`` summed over both groups is its forced CPU misses, each ``spec_use`` one miss
the pool covered, and the wait is the first submit to ``gate_open``. A used read is *late* (promoted: its demand
waited for the landing) when its ``spec_land`` is less than ``--late-us`` before its ``spec_use``; ``spec_use`` is
stamped after that wait, so a promoted read's landing sits just before it. At 100 us the late share matches the
server's ``spec_promoted / spec_used`` within 2-3 points on the 2026-10-09 captures.

Reports, over rows 1 and up of complete forwards: misses per layer (0, 1, 2, 3, 4+); of the layers with a miss, the
share the pool covered all (in time, or with a read still in flight), some or none; the reads' timing; and the
**step value of each covered read**: the mean wait of a layer with k + 1 reads left minus one with k, within (row,
misses) cells. If clearing a layer were what pays, the k = 0 -> 1 step would exceed the others.

The instrumented build's times are not throughput numbers; compare them only with another instrumented run.

Usage: layer_cover.py 'PREFIX.*.jsonl' [--late-us 100] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys

from layer_misses import forwards


def load(paths: list[str]) -> list[dict]:
    """One record per (row, gen) with a submit and a gate: misses, wait and its used reads' submit/land/use times."""
    recs = collections.defaultdict(lambda: {"miss": 0, "sub": None, "gate": None, "uses": []})
    submits, lands, dropped = {}, {}, 0
    for path in paths:
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if "dropped" in e:
                    dropped += int(e["dropped"])
                    continue
                kind = e.get("event")
                if kind == "copy_submit":
                    r = recs[(e["row"], e["gen"])]
                    r["miss"] += e["b"]
                    r["sub"] = e["ns"] if r["sub"] is None else min(r["sub"], e["ns"])
                elif kind == "gate_open":
                    recs[(e["row"], e["gen"])]["gate"] = e["ns"]
                elif kind == "spec_use":  # row = target row, a = expert, c = the read's source seq
                    recs[(e["row"], e["gen"])]["uses"].append((e["a"], e["c"], e["ns"]))
                elif kind == "spec_submit":  # row = target row, seq = source seq, a = expert
                    submits[(e["row"], e["seq"], e["a"])] = e["ns"]
                elif kind == "spec_land":
                    lands[(e["row"], e["seq"], e["a"])] = e["ns"]
    if dropped:
        raise ValueError(f"{dropped} events dropped (trace buffer overflow)")
    out = []
    for (row, gen), r in recs.items():
        if r["sub"] is None or r["gate"] is None:
            continue
        uses = [{"submit": submits.get((row, c, a)), "land": lands.get((row, c, a)), "use": ns}
                for a, c, ns in r["uses"]]
        out.append({"row": row, "gen": gen, "miss": r["miss"], "used": len(uses), "left": r["miss"] - len(uses),
                    "uses": uses, "sub_ns": r["sub"], "gate_ns": r["gate"], "ms": (r["gate"] - r["sub"]) / 1e6})
    return sorted(out, key=lambda r: r["gen"])


def late(use: dict, late_us: float) -> bool:
    return use["land"] is not None and use["use"] - use["land"] < late_us * 1e3


def _share(n: int, d: int):
    return round(n / d, 3) if d else None


def _q(values: list[float]):
    if not values:
        return None
    v = sorted(values)
    return [round(v[min(len(v) - 1, int(p * len(v)))], 3) for p in (0.1, 0.5, 0.9)]


def _bucket(m: int) -> str:
    return str(m) if m < 4 else "4+"


def coverage(layers: list[dict], late_us: float) -> dict:
    """Of the layers with a miss: all covered (none late / some late), some, none; overall and by misses."""
    def split(sub):
        full = [r for r in sub if r["left"] <= 0]
        n_late = [sum(late(u, late_us) for u in r["uses"]) for r in full]
        return {"layers": len(sub), "all": _share(len(full), len(sub)),
                "all_in_time": _share(sum(k == 0 for k in n_late), len(sub)),
                "all_some_late": _share(sum(k > 0 for k in n_late), len(sub)),
                "some": _share(sum(0 < r["used"] < r["miss"] for r in sub), len(sub)),
                "none": _share(sum(r["used"] == 0 for r in sub), len(sub))}
    hit = [r for r in layers if r["miss"] > 0]
    by = collections.defaultdict(list)
    for r in hit:
        by[_bucket(r["miss"])].append(r)
    return {**split(hit), "by_misses": {k: split(v) for k, v in sorted(by.items())}}


def timing(layers: list[dict], late_us: float) -> dict:
    uses = [u for r in layers for u in r["uses"] if u["land"] is not None and u["submit"] is not None]
    lates = [u for u in uses if late(u, late_us)]
    return {"used_reads": len(uses), "late_share": _share(len(lates), len(uses)),
            "read_ms_q10_50_90": _q([(u["land"] - u["submit"]) / 1e6 for u in uses]),
            "late_read_ms_q10_50_90": _q([(u["land"] - u["submit"]) / 1e6 for u in lates]),
            "lead_ms_q10_50_90": _q([(u["use"] - u["submit"]) / 1e6 for u in uses]),
            "slack_ms_q10_50_90": _q([(u["use"] - u["land"]) / 1e6 for u in uses])}


def read_steps(layers: list[dict], max_misses: int = 4) -> dict:
    """Mean wait at k + 1 reads left minus at k, within (row, misses) cells, weighted by the smaller cell."""
    cells = collections.defaultdict(list)
    for r in layers:
        cells[(r["row"], r["miss"], r["left"])].append(r["ms"])
    rows = sorted({r["row"] for r in layers})
    out = {}
    for m in range(1, max_misses + 1):
        for k in range(m):
            total = weight = 0.0
            for row in rows:
                a, b = cells.get((row, m, k)), cells.get((row, m, k + 1))
                if a and b:
                    n = min(len(a), len(b))
                    total += (statistics.mean(b) - statistics.mean(a)) * n
                    weight += n
            if weight:
                out[f"{m} misses, left {k}->{k + 1}"] = {"ms": round(total / weight, 3), "weight": int(weight)}
    return out


def summarize(recs: list[dict], late_us: float) -> dict:
    fwd = forwards(recs, max(r["row"] for r in recs) + 1)
    layers = [r for f in fwd for r in f if r["row"] > 0]
    hist = collections.Counter(_bucket(r["miss"]) for r in layers)
    zero = [r["ms"] for r in layers if r["miss"] == 0]
    out = {"forwards": len(fwd), "layers_rows_1_up": len(layers),
           "miss_hist": {k: hist[k] for k in sorted(hist)},
           "misses_per_forward": round(sum(r["miss"] for f in fwd for r in f) / max(len(fwd), 1), 2),
           "covered_per_forward": round(sum(r["used"] for f in fwd for r in f) / max(len(fwd), 1), 2),
           "zero_miss_wait_ms_median": round(statistics.median(zero), 3) if zero else None,
           "coverage": coverage(layers, late_us), "read_steps": read_steps(layers)}
    if any(r["used"] for r in layers):
        out["timing"] = timing(layers, late_us)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern")
    p.add_argument("--late-us", type=float, default=100.0)
    p.add_argument("--json")
    a = p.parse_args()
    paths = sorted(glob.glob(a.pattern))
    if not paths:
        print(f"no files match {a.pattern}", file=sys.stderr)
        return 2
    try:
        result = summarize(load(paths), a.late_us)
    except ValueError as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**result, "files": paths}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
