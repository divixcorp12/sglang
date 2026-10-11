"""The expert drives' busy time and bytes during a served run's decode steps: how much idle drive time a prefetch
could move reads into.

``sample OUT`` appends one JSON line per tick with each device's ``/sys/block/<dev>/stat`` (sectors read, io_ticks =
ms with any I/O in flight, the in-flight count) stamped with CLOCK_MONOTONIC, the job trace's clock; SIGTERM stops it.
``analyze SAMPLES 'EVENTS.*.jsonl'`` takes the decode steps (layer_misses.py's complete forwards) as windows and
reports per drive its bytes and busy share, and for the set of drives the time none was busy: bounded from io_ticks
(at least the interval less every drive's busy time, at most less the busiest one's), and estimated from the samples
whose in-flight counts were all zero.

``drive-load SERVER_LOG`` needs no sampler: it reads the last ``exl3 RAM miss drive load {json}`` line a host writes at
stop (the reader's own per-root accounting, host/drive_load.h, which polled reads leave out of /sys's in_flight) and
reports per root, over the host's lifetime, the share of time demand reads, speculative reads, and both at once were in
flight, the bytes each kind landed, and the sub-reads the dynamic root choice (SGLANG_MOE_EXPERT_MIRROR_DYNAMIC) moved
away from and to it.

Usage: drive_busy.py sample OUT [--devices nvme0n1,nvme1n1,nvme2n1] [--interval-ms 5]
       drive_busy.py analyze SAMPLES 'EVENTS.*.jsonl' [--json OUT]
       drive_busy.py drive-load SERVER_LOG [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_misses  # noqa: E402

DEVICES = "nvme0n1,nvme1n1,nvme2n1"  # the three expert-row mirrors (arm_env.EXPERT_MIRROR_DIRS)


def parse_stat(line: str) -> dict:
    f = line.split()
    return {"read_bytes": int(f[2]) * 512, "busy_ms": int(f[9]), "in_flight": int(f[8])}


def sample(out: str, devices: list[str], interval_ms: float) -> None:
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    files = {d: open(f"/sys/block/{d}/stat") for d in devices}
    step = int(interval_ms * 1e6)
    nxt = time.monotonic_ns()
    with open(out, "w") as f:
        while not stop:
            row = {}
            for d, fh in files.items():
                fh.seek(0)
                row[d] = parse_stat(fh.read())
            f.write(json.dumps({"ns": time.monotonic_ns(), "dev": row}) + "\n")
            nxt += step
            delay = nxt - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            else:
                nxt = time.monotonic_ns()


def _pairs(samples, t0, t1):
    """(overlap fraction, overlap ms, earlier sample, later sample) for each interval meeting [t0, t1)."""
    for a, b in zip(samples, samples[1:]):
        lo, hi = max(a["ns"], t0), min(b["ns"], t1)
        if hi > lo and b["ns"] > a["ns"]:
            yield (hi - lo) / (b["ns"] - a["ns"]), (hi - lo) / 1e6, a, b


def window(samples: list[dict], t0: int, t1: int) -> dict:
    out = {}
    for frac, _, a, b in _pairs(samples, t0, t1):
        for d in b["dev"]:
            o = out.setdefault(d, {"read_bytes": 0.0, "busy_ms": 0.0})
            o["read_bytes"] += frac * (b["dev"][d]["read_bytes"] - a["dev"][d]["read_bytes"])
            o["busy_ms"] += frac * (b["dev"][d]["busy_ms"] - a["dev"][d]["busy_ms"])
    return out


def all_idle_bounds(samples: list[dict], t0: int, t1: int) -> tuple[float, float]:
    lo = hi = 0.0
    for frac, ms, a, b in _pairs(samples, t0, t1):
        busy = [min(ms, frac * (b["dev"][d]["busy_ms"] - a["dev"][d]["busy_ms"])) for d in b["dev"]]
        lo += max(0.0, ms - sum(busy))
        hi += ms - max(busy)
    return lo, hi


def analyze(samples: list[dict], recs: list[dict]) -> dict:
    rows = max(r["row"] for r in recs) + 1
    fwd = layer_misses.forwards(recs, rows)
    if not fwd:
        raise ValueError("no complete forward in the trace")
    wins = [(f[0]["sub_ns"], f[-1]["gate_ns"]) for f in fwd]
    if samples[0]["ns"] > wins[0][0] or samples[-1]["ns"] < wins[-1][1]:
        raise ValueError("the samples do not cover the trace's decode steps (another clock, or a late sampler)")
    total_ms = sum((t1 - t0) / 1e6 for t0, t1 in wins)
    devs = sorted(samples[0]["dev"])
    agg = {d: {"read_bytes": 0.0, "busy_ms": 0.0} for d in devs}
    lo = hi = 0.0
    per_fwd = []
    for t0, t1 in wins:
        w = window(samples, t0, t1)
        for d in devs:
            agg[d]["read_bytes"] += w[d]["read_bytes"]
            agg[d]["busy_ms"] += w[d]["busy_ms"]
        a, b = all_idle_bounds(samples, t0, t1)
        lo, hi = lo + a, hi + b
        per_fwd.append(sum(w[d]["read_bytes"] for d in devs) / 1e6)
    inside = [s for s in samples if any(t0 <= s["ns"] < t1 for t0, t1 in wins)]
    idle = sum(all(s["dev"][d]["in_flight"] == 0 for d in devs) for s in inside)
    return {
        "forwards": len(fwd),
        "decode_s": round(total_ms / 1e3, 2),
        "forward_ms_median": round(statistics.median((t1 - t0) / 1e6 for t0, t1 in wins), 1),
        "read_mb_per_forward_median": round(statistics.median(per_fwd), 1),
        "drives": {d: {"gb_per_s": round(agg[d]["read_bytes"] / total_ms / 1e6, 3),
                       "busy_share": round(agg[d]["busy_ms"] / total_ms, 3)} for d in devs},
        "total_gb_per_s": round(sum(agg[d]["read_bytes"] for d in devs) / total_ms / 1e6, 3),
        "all_idle_share_bounds": [round(lo / total_ms, 3), round(hi / total_ms, 3)],
        "all_idle_share_sampled": round(idle / len(inside), 3) if inside else None,
        "samples_in_decode": len(inside),
    }


DRIVE_LOAD_MARKER = "exl3 RAM miss drive load "


def drive_load_report(text: str) -> dict:
    """The last drive-load line of a server log, per root: busy shares by kind and their overlap over the host's
    lifetime, and bytes by kind. Raises ValueError when the log has no such line."""
    lines = [line for line in text.splitlines() if DRIVE_LOAD_MARKER in line]
    if not lines:
        raise ValueError(f"no '{DRIVE_LOAD_MARKER.strip()}' line in the log")
    load = json.loads(lines[-1].split(DRIVE_LOAD_MARKER, 1)[1])
    elapsed = load["elapsed_ns"]
    if elapsed <= 0:
        raise ValueError("the drive load has no elapsed time")
    total = sum(r["demand_bytes"] + r["spec_bytes"] for r in load["roots"]) or 1
    roots = []
    for q, r in enumerate(load["roots"]):
        roots.append({
            "root": q,
            "demand_busy_share": round(r["demand_busy_ns"] / elapsed, 4),
            "spec_busy_share": round(r["spec_busy_ns"] / elapsed, 4),
            "overlap_share": round(r["overlap_ns"] / elapsed, 4),
            # Of the time this root served demand, the share it also had a speculative read in flight.
            "overlap_of_demand": round(r["overlap_ns"] / r["demand_busy_ns"], 4) if r["demand_busy_ns"] else None,
            "demand_gb": round(r["demand_bytes"] / 1e9, 3),
            "spec_gb": round(r["spec_bytes"] / 1e9, 3),
            "bytes_share": round((r["demand_bytes"] + r["spec_bytes"]) / total, 4),
            "in_flight_at_log": r["demand_reads"] + r["spec_reads"],
            # Sub-reads SGLANG_MOE_EXPERT_MIRROR_DYNAMIC sent away from / to this root (0 in older logs).
            "moved_from": r.get("moved_from", 0),
            "moved_to": r.get("moved_to", 0),
        })
    return {"lines": len(lines), "elapsed_s": round(elapsed / 1e9, 3), "clock_reads": load["clock_reads"],
            "roots": roots}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--devices", default=DEVICES)
    s.add_argument("--interval-ms", type=float, default=5.0)
    an = sub.add_parser("analyze")
    an.add_argument("samples")
    an.add_argument("events")
    an.add_argument("--json")
    dl = sub.add_parser("drive-load")
    dl.add_argument("log")
    dl.add_argument("--json")
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.out, a.devices.split(","), a.interval_ms)
        return 0
    if a.cmd == "drive-load":
        with open(a.log, errors="replace") as f:
            try:
                result = drive_load_report(f.read())
            except ValueError as error:
                print(f"refusing: {error}", file=sys.stderr)
                return 1
        print(json.dumps(result, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(result, f, indent=1)
        return 0
    with open(a.samples) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    try:
        result = analyze(samples, layer_misses.load(sorted(glob.glob(a.events))))
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
