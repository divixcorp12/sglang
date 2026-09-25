#!/usr/bin/env python3
"""Host time spent in cudaGraphLaunch, per call, from graph-mode nsys smoke traces.

Usage: launch_durations.py NAME=trace.sqlite [NAME=trace.sqlite ...] [--json OUT]

Reads CUPTI_ACTIVITY_KIND_RUNTIME (host API calls), not the graph-trace table, which can describe the warm-up
rather than the capture (CLAUDE.md). Every cudaGraphLaunch in these smokes is a decode step's segment launch:
prefill runs eager. Also reports cudaStreamSynchronize / cudaEventSynchronize, where the host waits instead once
the launch no longer blocks. CPU only; run under ``taskset -c 0-63``.
"""

import argparse
import json
import sqlite3
import statistics


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summary(ms):
    return {
        "n": len(ms),
        "p50": statistics.median(ms) if ms else float("nan"),
        "p90": pct(ms, 0.9),
        "max": max(ms) if ms else float("nan"),
        "sum": sum(ms),
    }


def calls(db, prefix):
    rows = db.execute(
        "SELECT r.end - r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
        "WHERE s.value LIKE ?",
        (prefix + "%",),
    ).fetchall()
    return [d / 1e6 for (d,) in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for arm in a.arms:
        name, path = arm.split("=", 1)
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        out[name] = {
            api: summary(calls(db, api))
            for api in ("cudaGraphLaunch", "cudaStreamSynchronize", "cudaEventSynchronize", "cudaMemcpyAsync")
        }
        db.close()
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
