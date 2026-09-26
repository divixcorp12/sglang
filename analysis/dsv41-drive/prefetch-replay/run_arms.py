"""Run a list of pinned_prefetch_replay arms in parallel over one loaded trace.

Usage: run_arms.py TRACE RANKS ARMS_FILE OUT_JSONL [--workers N]

ARMS_FILE: one arm per line, ``name: <pinned_prefetch_replay options>`` (``#`` comments). The trace is loaded
once and shared with the forked workers.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shlex
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
import pinned_prefetch_replay as ppr  # noqa: E402
from tier_sim import load_forwards  # noqa: E402

LOADED = None
RANKS = None


def work(item):
    name, argv = item
    a = ppr.parse(argv)
    out = ppr.run_one(a, LOADED, RANKS)
    out["name"] = name
    return out


def main() -> None:
    global LOADED, RANKS
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("ranks")
    p.add_argument("arms")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    items = []
    for line in open(a.arms):
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, opts = line.split(":", 1)
        items.append((name.strip(), [a.trace, "--ranks", a.ranks] + shlex.split(opts)))
    LOADED = load_forwards(a.trace)
    RANKS = dict(np.load(a.ranks))
    with multiprocessing.get_context("fork").Pool(a.workers) as pool, open(a.out, "w") as f:
        for out in pool.imap_unordered(work, items):
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(out["name"], f"{out['ram_misses_per_token']:.2f}", f"{out['exposed_ms_per_token']:.2f}", flush=True)


if __name__ == "__main__":
    main()
