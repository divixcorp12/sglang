"""Per layer record of a served run's InstrBuild job trace: its forced misses, how many the RAM prefetch's pool
covered, and how long the layer waited.

A record is one (row, gen): ``copy_submit`` per NUMA group carries the group's forced CPU misses (``b`` = late_cpu,
every lane whose row was not in RAM, read from NVMe or swapped in from the prefetch pool), each ``spec_use`` of the
record is one miss the pool covered, and ``gate_open`` is the record done. The layer's wait is the first group's
submit to the gate. A forward is rows 0..n-1 once each at consecutive gens.

The cost of a remaining NVMe read is the slope of the wait on the remaining reads within (row, misses) strata, rows
1 and up (row 0 has no source layer, so the prefetch never covers it): it compares a layer with itself at the same
miss count, so how much a layer computes does not enter. ``--compare`` matches another run's records to this one's by
(row, misses), among records the pool did not cover, and reports how much slower the other run's are: a cost the
prefetch puts on every layer rather than on its own reads.

The instrumented build's times are not throughput numbers; compare them only with another instrumented run.

Usage: layer_misses.py 'PREFIX.*.jsonl' [--compare 'OTHER.*.jsonl'] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys


def load(paths: list[str]) -> list[dict]:
    recs = collections.defaultdict(lambda: {"miss": 0, "used": 0, "sub": None, "gate": None})
    dropped = 0
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
                elif kind == "spec_use":
                    recs[(e["row"], e["gen"])]["used"] += 1
    if dropped:
        raise ValueError(f"{dropped} events dropped (trace buffer overflow)")
    out = []
    for (row, gen), r in recs.items():
        if r["sub"] is None or r["gate"] is None:
            continue
        out.append({"row": row, "gen": gen, "miss": r["miss"], "used": r["used"], "left": r["miss"] - r["used"],
                    "sub_ns": r["sub"], "gate_ns": r["gate"], "ms": (r["gate"] - r["sub"]) / 1e6})
    return sorted(out, key=lambda r: r["gen"])


def forwards(recs: list[dict], rows: int) -> list[list[dict]]:
    out, cur = [], []
    for r in sorted(recs, key=lambda r: r["gen"]):
        if r["row"] == 0:
            cur = [r]
        elif cur and r["row"] == cur[-1]["row"] + 1 and r["gen"] == cur[-1]["gen"] + 1:
            cur.append(r)
        else:
            cur = []
        if len(cur) == rows:
            out.append(cur)
            cur = []
    return out


def within_slope(recs: list[dict], key, x: str, min_n: int = 5):
    """Least-squares slope of ``ms`` on ``x`` with a separate intercept per ``key`` stratum; None without spread."""
    strata = collections.defaultdict(list)
    for r in recs:
        strata[key(r)].append(r)
    num = den = 0.0
    for s in strata.values():
        if len(s) < min_n:
            continue
        mx = statistics.mean(r[x] for r in s)
        my = statistics.mean(r["ms"] for r in s)
        num += sum((r[x] - mx) * (r["ms"] - my) for r in s)
        den += sum((r[x] - mx) ** 2 for r in s)
    return num / den if den else None


def _r(v, n=3):
    return None if v is None else round(v, n)


def summarize(recs: list[dict]) -> dict:
    rows = max(r["row"] for r in recs) + 1
    fwd = forwards(recs, rows)
    later = [r for r in recs if r["row"] > 0]
    out = {"records": len(recs), "rows": rows, "forwards": len(fwd)}
    if fwd:
        wall = [(f[-1]["gate_ns"] - f[0]["sub_ns"]) / 1e6 for f in fwd]
        gate = [sum(r["ms"] for r in f) for f in fwd]
        out["per_forward"] = {
            "wall_ms_median": _r(statistics.median(wall), 1),
            "gate_ms_median": _r(statistics.median(gate), 1),
            "gate_share_median": _r(statistics.median(g / w for g, w in zip(gate, wall))),
            "misses": _r(statistics.mean(sum(r["miss"] for r in f) for f in fwd), 1),
            "covered": _r(statistics.mean(sum(r["used"] for r in f) for f in fwd), 1),
            "nvme_left": _r(statistics.mean(sum(r["left"] for r in f) for f in fwd), 1),
            "nvme_left_rows_1_up": _r(statistics.mean(sum(r["left"] for r in f if r["row"] > 0) for f in fwd), 1),
        }
    per_read = within_slope(later, key=lambda r: (r["row"], r["miss"]), x="left")
    uncovered = [r for r in later if r["used"] == 0]
    out["cost_ms"] = {
        "per_nvme_read_left": _r(per_read),
        "per_nvme_read_left_by_misses": {
            str(m): _r(within_slope([r for r in later if r["miss"] == m], key=lambda r: r["row"], x="left"))
            for m in (1, 2, 3, 4)},
        "per_miss_uncovered": _r(within_slope(uncovered, key=lambda r: r["row"], x="miss")),
        "zero_miss_layer_median": _r(statistics.median(r["ms"] for r in later if r["miss"] == 0)
                                     if any(r["miss"] == 0 for r in later) else None),
    }
    if fwd and per_read is not None:
        out["ceiling_ms_per_forward"] = _r(per_read * out["per_forward"]["nvme_left_rows_1_up"], 1)
    by_row = collections.defaultdict(list)
    for r in recs:
        by_row[r["row"]].append(r)
    out["by_row"] = []
    for row in sorted(by_row):
        s = by_row[row]
        hit = [r for r in s if r["miss"] > 0]
        out["by_row"].append({
            "row": row, "misses_per_record": _r(statistics.mean(r["miss"] for r in s), 2),
            "cleared_share": _r(sum(r["left"] == 0 for r in hit) / len(hit), 2) if hit else None,
            "ms_median": _r(statistics.median(r["ms"] for r in s), 2)})
    return out


def compare(base: list[dict], other: list[dict]) -> dict:
    """How much slower ``other``'s uncovered records are than ``base``'s of the same (row, misses), weighted by
    ``other``'s count per cell."""
    cells = collections.defaultdict(lambda: ([], []))
    for side, recs in ((0, base), (1, other)):
        for r in recs:
            if r["used"] == 0:
                cells[(r["row"], r["miss"])][side].append(r["ms"])
    total = weight = 0.0
    by_m = collections.defaultdict(lambda: [0.0, 0])
    for (row, miss), (b, o) in cells.items():
        if not b or not o:
            continue
        diff = statistics.mean(o) - statistics.mean(b)
        total += diff * len(o)
        weight += len(o)
        by_m[miss][0] += diff * len(o)
        by_m[miss][1] += len(o)
    return {
        "matched_records": int(weight),
        "ms_per_record_slower": _r(total / weight) if weight else None,
        "by_misses": {str(m): {"records": n, "ms_slower": _r(t / n)} for m, (t, n) in sorted(by_m.items()) if m <= 8},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern")
    p.add_argument("--compare", help="another run's trace, matched against this one")
    p.add_argument("--json")
    a = p.parse_args()
    paths = sorted(glob.glob(a.pattern))
    if not paths:
        print(f"no files match {a.pattern}", file=sys.stderr)
        return 2
    try:
        recs = load(paths)
        result = summarize(recs)
        if a.compare:
            other = sorted(glob.glob(a.compare))
            if not other:
                print(f"no files match {a.compare}", file=sys.stderr)
                return 2
            result["compare"] = compare(recs, load(other))
    except ValueError as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "by_row"}, indent=1))
    print("row  misses/rec  cleared  ms")
    for r in result["by_row"]:
        print(f"{r['row']:3d}  {r['misses_per_record']:10}  {r['cleared_share']!s:7}  {r['ms_median']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**result, "files": paths}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
