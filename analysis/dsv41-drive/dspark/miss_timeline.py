"""Per record and NUMA group of an InstrBuild job trace: when its NVMe-read CPU misses landed, when the group's CPU
expert thread ran them, and what held them up.

The service submits a record's CPU-hit job (part 0) first, then its copy job (``copy_submit``: seq = cpu_seq, a =
late_seq, b = late_cpu, c = cpu_mask), then one part-1 job per batch of CPU misses whose rows landed together
(ram_tier.h, submit_landed_cpu_misses), so a part-1 ``cpu_submit`` is a landing time. The record's jobs are the group
engine's sequences cpu_seq..late_seq. One thread per group runs every job in order, and DSpark draft jobs
(``draft_start``/``draft_end``) on the same thread between them.

Per record: the first and last landing after the submit; how many batches the misses landed in (one batch for several
misses: they landed together, so computing them as they land gained nothing); how long the first miss job waited
after its landing and how much of that the record's own CPU-hit job caused; how long the thread was idle between the
submit and the first landing (room for more CPU hits at no cost to the misses); the compute after the last landing;
and whether the miss chain or the DMA finished last.

The instrumented build's times are not throughput numbers; compare them only with another instrumented run.

Usage: miss_timeline.py DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys


def _events(path):
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if "event" in e:
                yield e
            elif "dropped" in e and int(e["dropped"]):
                raise ValueError(f"{path}: {e['dropped']} events dropped (trace buffer overflow)")


def busy_within(intervals, t0, t1) -> float:
    """The length of the union of `intervals` inside [t0, t1)."""
    total, end = 0.0, t0
    for a, b in sorted(intervals):
        a, b = max(a, end), min(b, t1)
        if b > a:
            total += b - a
            end = b
    return total


def _load_engine(path):
    jobs = collections.defaultdict(dict)
    busy, draft_open = [], None
    for e in _events(path):
        kind = e["event"]
        if kind in ("cpu_submit", "cpu_start", "cpu_end"):
            j = jobs[e["seq"]]
            j[kind[4:]] = e["ns"]
            j["part"], j["k"] = e["a"], e["b"]
        elif kind == "draft_start":
            draft_open = e["ns"]
        elif kind == "draft_end" and draft_open is not None:
            busy.append((draft_open, e["ns"]))
            draft_open = None
    for j in jobs.values():
        if "start" in j and "end" in j:
            busy.append((j["start"], j["end"]))
    return jobs, busy


def records(out_dir: str) -> list[dict]:
    engines = {}
    for path in glob.glob(os.path.join(out_dir, "events.*.exl3-cpu-exp*.jsonl")):
        g = int(re.search(r"exl3-cpu-exp(\d+)\.", os.path.basename(path)).group(1))
        engines[g] = _load_engine(path)
    subs, dma, done = {}, {}, {}
    for path in glob.glob(os.path.join(out_dir, "events.*.exl3-copy-eng*.jsonl")):
        for e in _events(path):
            key = (e["gen"], e["group"])
            if e["event"] == "copy_submit":
                subs[key] = e
            elif e["event"] == "copy_dma_observed":
                dma[key] = e["ns"]
            elif e["event"] == "group_done":
                done[key] = e["ns"]
    ms = lambda ns: ns / 1e6
    out = []
    for key, s in subs.items():
        late, g = s["b"], s["group"]
        if late == 0 or g not in engines or key not in done:
            continue
        jobs, busy = engines[g]
        mine = [jobs.get(q) for q in range(s["seq"], s["a"] + 1)]
        mine = [j for j in mine if j]
        hit = [j for j in mine if j["part"] == 0]
        miss = sorted((j for j in mine if j["part"] == 1), key=lambda j: j.get("submit", 0))
        if sum(j["k"] for j in miss) != late or any(not {"submit", "start", "end"} <= j.keys() for j in mine):
            continue
        t0 = min([s["ns"]] + [j["submit"] for j in hit])
        land0, land1 = miss[0]["submit"], miss[-1]["submit"]
        hit_end = hit[0]["end"] if hit else None
        chain_end = max(j["end"] for j in miss)
        out.append({
            "gen": key[0], "group": g, "row": s["row"], "misses": late, "hits": bin(s["c"]).count("1"),
            "batches": len(miss),
            "land_first_ms": ms(land0 - t0), "land_last_ms": ms(land1 - t0),
            "first_wait_ms": ms(miss[0]["start"] - land0),
            "hit_blocked_ms": ms(max(0, min(hit_end, miss[0]["start"]) - land0)) if hit_end else 0.0,
            "idle_before_land_ms": ms((land0 - t0) - busy_within(busy, t0, land0)),
            "tail_ms": ms(chain_end - land1),
            "miss_ms_per_lane": ms(sum(j["end"] - j["start"] for j in miss)) / late,
            "hit_ms": ms(hit[0]["end"] - hit[0]["start"]) if hit else None,
            "last": "cpu" if key not in dma or chain_end >= dma[key] else "dma",
            "done_ms": ms(done[key] - t0),
        })
    return sorted(out, key=lambda r: (r["gen"], r["group"]))


def summarize(recs: list[dict]) -> dict:
    if not recs:
        raise ValueError("no record with CPU misses joined its jobs")
    med = lambda xs: round(statistics.median(xs), 3) if xs else None
    multi = [r for r in recs if r["misses"] >= 2]
    side = collections.Counter(r["last"] for r in recs)
    by_m = collections.defaultdict(list)
    for r in recs:
        by_m[min(r["misses"], 8)].append(r)
    keys = ("land_first_ms", "land_last_ms", "first_wait_ms", "hit_blocked_ms", "idle_before_land_ms", "tail_ms",
            "done_ms")
    return {
        "records": len(recs),
        "median": {k: med([r[k] for r in recs]) for k in keys + ("miss_ms_per_lane",)},
        "multi_miss_one_batch_share": round(sum(r["batches"] == 1 for r in multi) / len(multi), 3) if multi else None,
        "landing_spread_ms_median": med([r["land_last_ms"] - r["land_first_ms"] for r in multi]),
        "hit_blocked_share": round(sum(r["hit_blocked_ms"] > 0 for r in recs) / len(recs), 3),
        "hit_blocked_ms_mean": round(statistics.mean(r["hit_blocked_ms"] for r in recs), 3),
        "last_side": {k: round(v / len(recs), 3) for k, v in sorted(side.items())},
        "by_misses": {str(m): {"records": len(rs), **{k: med([r[k] for r in rs]) for k in keys},
                               "batches": med([r["batches"] for r in rs])}
                      for m, rs in sorted(by_m.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--json")
    a = p.parse_args()
    try:
        result = summarize(records(a.out_dir))
    except ValueError as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
