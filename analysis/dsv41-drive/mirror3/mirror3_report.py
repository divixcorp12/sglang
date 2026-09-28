"""The 3-root piece-streaming arm against its 2-root reference (plan 2026-09-28-mirror3-piece-stream Task 6).

ms/token per turn (1000 / decode_tokens_per_sec), byte identity of every turn's reasoning and content, and each mirror
drive's read split over the arm's timed window: server_ready to the last session's boundary sample, both on
CLOCK_MONOTONIC (boundary-samples.jsonl's "monotonic", nvme_sampler.py's "mono"). The window includes each session's
prefill as well as its decode.

Usage: mirror3_report.py <ref_run_dir> <ref_diskstats.jsonl> <new_run_dir> <new_diskstats.jsonl> <device>...
"""

import json
import statistics
import sys
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def timed_window(run_dir):
    rows = load_jsonl(Path(run_dir) / "boundary-samples.jsonl")
    start = next(r["monotonic"] for r in rows if r["label"] == "server_ready")
    ends = [r["monotonic"] for r in rows if r["label"].startswith("session_")]
    if not ends:
        raise ValueError(f"{run_dir}: no timed session finished")
    return start, max(ends)


def drive_split(samples, start, end, devices):
    inside = [s for s in samples if start <= s["mono"] <= end]
    if len(inside) < 2:
        raise ValueError(f"fewer than two diskstats samples in [{start}, {end}]")
    a, b = inside[0], inside[-1]
    dt = b["mono"] - a["mono"]
    drives = {}
    for d in devices:
        reads = b[d]["reads"] - a[d]["reads"]
        read_bytes = (b[d]["sectors_read"] - a[d]["sectors_read"]) * 512
        drives[d] = {
            "read_MB": read_bytes / 1e6,
            "MBps": read_bytes / 1e6 / dt,
            "iops": reads / dt,
            "req_kB": read_bytes / 1e3 / reads if reads else 0.0,
            "util_pct": 100 * (b[d]["io_ticks_ms"] - a[d]["io_ticks_ms"]) / (dt * 1e3),
        }
    total = sum(v["read_MB"] for v in drives.values())
    for v in drives.values():
        v["share_pct"] = 100 * v["read_MB"] / total if total else 0.0
    return {"seconds": dt, "drives": drives}


def _turns(run_dir):
    return {f'{r["session_id"]}/t{r["turn"]}': r for r in load_jsonl(Path(run_dir) / "results.jsonl") if "error" not in r}


def ms_per_token(run_dir):
    per_turn = {k: 1000.0 / r["decode_tokens_per_sec"] for k, r in _turns(run_dir).items() if r.get("decode_tokens_per_sec")}
    return per_turn, statistics.median(per_turn.values())


def identity(ref_dir, new_dir):
    ref, new = _turns(ref_dir), _turns(new_dir)
    if ref.keys() != new.keys():
        raise ValueError(f"the arms ran different turns: {sorted(ref)} vs {sorted(new)}")
    return {k: (ref[k].get("reasoning"), ref[k].get("content")) == (new[k].get("reasoning"), new[k].get("content")) for k in sorted(ref)}


def arm(run_dir, diskstats, devices):
    per_turn, median = ms_per_token(run_dir)
    start, end = timed_window(run_dir)
    return {"run_dir": str(run_dir), "ms_per_token": per_turn, "median_ms_per_token": median,
            "disk": drive_split(load_jsonl(diskstats), start, end, devices)}


def main(argv):
    ref_dir, ref_disk, new_dir, new_disk, *devices = argv
    out = {"ref": arm(ref_dir, ref_disk, devices), "new": arm(new_dir, new_disk, devices),
           "identical": identity(ref_dir, new_dir)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
