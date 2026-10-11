"""A traced run's forced misses split by verified position: how many only a rejected draft position needed.

Inputs, all written by spec_margin_capture.py into OUT_DIR:

* the InstrBuild job trace (``events.*.jsonl``): each forward's records (layer_misses.py) and its ``miss_expert``
  events, one per forced miss (row, gen, a = the expert);
* the stage trace's route log (``stages.jsonl``, ``graph_routes`` lines): each graph forward's phase, rids, live
  token count, per-layer routed experts and its record in the router capture;
* the router capture (``router.json`` / ``router.ids.bin``): every graph forward's top-k ids per layer and token row;
* the accept log (``verify-accept.*.jsonl``): each request's k-th verify and its correct drafts.

The route log has every graph forward in order; the job trace has those it completed, in order, so it can miss some.
``layer_misses.match_forwards`` joins each job forward to the route line whose routes hold every expert it missed,
and a capture where under 95% of the job forwards with a miss find theirs is refused (a dropped forward costs only
itself; a job trace from another run or in another order joins almost none). A verify of a request's k-th line has its correct drafts c: positions 0..c are used (the bonus
token's included), the rest ran on a rejected draft. A miss is ``accept`` when a used position routed to it and
``reject_only`` otherwise.

Usage: verify_split.py OUT_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_misses  # noqa: E402


def split_layer(misses: set, ids_layer, num_correct_drafts: int, live: int) -> dict:
    used = set()
    for p in range(min(num_correct_drafts + 1, live)):
        used.update(int(e) for e in ids_layer[p] if e >= 0)
    accept = len(misses & used)
    return {"misses": len(misses), "accept": accept, "reject_only": len(misses) - accept}


def verify_forwards(lines: list[dict], accepts: dict) -> list[dict]:
    """The route log's target_verify forwards, each with its request's k-th accept count (None when unlogged)."""
    seen = collections.Counter()
    out = []
    for line in lines:
        if line.get("phase") != "target_verify":
            continue
        rid = (line.get("rids") or [None])[0]
        k = seen[rid]
        seen[rid] += 1
        out.append({"seq": line["seq"], "rid": rid, "k": k, "router": line.get("router"),
                    "live": int(line.get("forward_tokens") or 0), "num_correct_drafts": accepts.get((rid, k))})
    return out


def router_ids(prefix: str) -> np.ndarray:
    with open(prefix + ".json") as f:
        h = json.load(f)
    ids = np.fromfile(prefix + ".ids.bin", dtype=np.int32)
    return ids.reshape(-1, len(h["layer_ids"]), h["tokens"], h["topk"])


def _job_forwards(paths: list[str]) -> list[list[set]]:
    recs = layer_misses.load(paths)
    missed = collections.defaultdict(set)
    for path in paths:
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("event") == "miss_expert":
                    missed[(e["row"], e["gen"])].add(e["a"])
    rows = max(r["row"] for r in recs) + 1
    return [[missed.get((r["row"], r["gen"]), set()) for r in fwd] for fwd in layer_misses.forwards(recs, rows)]


def analyze(out_dir: str) -> dict:
    job = _job_forwards(sorted(glob.glob(os.path.join(out_dir, "events.*.jsonl"))))
    with open(os.path.join(out_dir, "stages.jsonl")) as f:
        lines = [x for x in map(json.loads, f) if x.get("kind") == "graph_routes"]
    if any(x.get("dropped_before") for x in lines):
        raise ValueError("the route log dropped entries")
    accepts = {}
    for path in glob.glob(os.path.join(out_dir, "verify-accept.*.jsonl")):
        with open(path) as f:
            for e in map(json.loads, f):
                accepts[(e["rid"], e["k"])] = e["num_correct_drafts"]
    routes = [[set(r) for r in x["routes"]] for x in lines]
    pairs = layer_misses.match_forwards(job, routes)
    joined = {i for i, _ in pairs}
    missing = [i for i, fwd in enumerate(job) if any(fwd)]
    share = sum(i in joined for i in missing) / max(1, len(missing))
    if share < 0.95:
        raise ValueError(f"the job trace and the route log do not join ({share:.3f} of {len(missing)} job forwards "
                         "with a miss found their route line)")
    verifies = {v["seq"]: v for v in verify_forwards(lines, accepts)}
    ids = router_ids(os.path.join(out_dir, "router"))
    per, by_c = [], collections.defaultdict(list)
    for i, li in pairs:
        fwd, v = job[i], verifies.get(lines[li]["seq"])
        if v is None or v["num_correct_drafts"] is None or v["router"] is None:
            continue
        tot = collections.Counter()
        for row, missed in enumerate(fwd):
            tot.update(split_layer(missed, ids[v["router"], row], v["num_correct_drafts"], v["live"]))
        per.append(dict(tot))
        by_c[v["num_correct_drafts"]].append(dict(tot))
    if not per:
        raise ValueError("no verify forward joined its accept count")
    mean = lambda xs, k: round(statistics.mean(x.get(k, 0) for x in xs), 2)
    return {
        "alignment": {"joined": len(pairs), "job_forwards": len(job), "route_lines": len(lines),
                      "joined_with_a_miss": round(share, 4)},
        "verifies": len(per),
        "per_verify": {k: mean(per, k) for k in ("misses", "accept", "reject_only")},
        "reject_only_share": round(sum(x.get("reject_only", 0) for x in per) / max(1, sum(x["misses"] for x in per)), 3),
        "by_correct_drafts": {str(c): {"verifies": len(xs), **{k: mean(xs, k) for k in ("misses", "reject_only")}}
                              for c, xs in sorted(by_c.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--json")
    a = p.parse_args()
    try:
        result = analyze(a.out_dir)
    except (ValueError, FileNotFoundError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
