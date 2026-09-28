"""The 3-root piece-streaming arm against its 2-root reference (plan 2026-09-28-mirror3-piece-stream Task 6).

ms/token per turn (1000 / decode_tokens_per_sec), byte identity of every turn's reasoning and content, and each mirror
drive's read split over the arm's timed window: server_ready to the last session's boundary sample, both on
CLOCK_MONOTONIC (boundary-samples.jsonl's "monotonic", nvme_sampler.py's "mono"). The window includes each session's
prefill as well as its decode.

Also, per arm: pooled ms/token (total decode seconds over total decode tokens, per session and over the arm), TTFT per
turn and session, the SM clock over the timed window (the driver's nvidia-smi CSV, whose timestamps are the sampling
host's local time, so run this on divix01) and at each session's start and end (run_arm.sh's clocks.jsonl), and the
RAM-miss service's counters. Those come from the two lines the server writes to server.log at shutdown ("exl3 RAM miss
thread counters {...}" and "exl3 expert stream: {...}"); they are cumulative over the server's life, so they include
warm-up rounds (counted here) and prefill, not only the timed window. No trace is enabled to get them.

Usage: mirror3_report.py <ref_run_dir> <ref_diskstats.jsonl> <ref_clocks.csv> <new_run_dir> <new_diskstats.jsonl>
       <new_clocks.csv> <device>...
"""

import datetime
import json
import re
import statistics
import sys
from pathlib import Path

RAM_MISS_KEYS = (
    "served", "rows_read", "read_errors", "copy_errors", "copy_fallbacks", "overruns", "late_after_fatal",
    "piece_stream_refused", "piece_publish_refused",
)
STREAM_KEYS = ("vram_misses", "ram_misses", "read_ms")


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


def decode(run_dir):
    """Pooled ms/token and TTFT. A turn decodes completion_tokens - 1 tokens in (completion_tokens - 1) /
    decode_tokens_per_sec seconds (run_capture_sessions.py), so pooling is sum(seconds) / sum(tokens)."""
    sessions = {}
    for key, r in _turns(run_dir).items():
        s = sessions.setdefault(r["session_id"], {"decode_tokens": 0, "decode_s": 0.0, "ttft_s": {}})
        s["ttft_s"][key] = r.get("ttft")
        if r.get("decode_tokens_per_sec"):
            tokens = r["completion_tokens"] - 1
            s["decode_tokens"] += tokens
            s["decode_s"] += tokens / r["decode_tokens_per_sec"]
    for s in sessions.values():
        s["pooled_ms_per_token"] = 1000 * s["decode_s"] / s["decode_tokens"] if s["decode_tokens"] else None
    tokens = sum(s["decode_tokens"] for s in sessions.values())
    seconds = sum(s["decode_s"] for s in sessions.values())
    ttfts = [t for s in sessions.values() for t in s["ttft_s"].values() if t is not None]
    return {"sessions": sessions, "pooled_ms_per_token": 1000 * seconds / tokens if tokens else None,
            "decode_tokens": tokens, "decode_s": seconds,
            "median_ttft_s": statistics.median(ttfts) if ttfts else None, "total_ttft_s": sum(ttfts)}


def timed_window_utc(run_dir):
    rows = load_jsonl(Path(run_dir) / "boundary-samples.jsonl")
    start = next(r["utc"] for r in rows if r["label"] == "server_ready")
    end = max((r for r in rows if r["label"].startswith("session_")), key=lambda r: r["monotonic"])["utc"]
    return datetime.datetime.fromisoformat(start), datetime.datetime.fromisoformat(end)


def _mhz(field):
    return int(field.strip().split()[0])


def clock_summary(clocks_csv, start, end):
    """clocks.sm over [start, end] (aware datetimes) from `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm
    --format=csv -lms ...`, whose timestamps are naive local time on the sampling host."""
    sm, max_sm = [], set()
    for line in Path(clocks_csv).read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 3:
            continue
        t = datetime.datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S.%f").astimezone()
        if start <= t <= end:
            sm.append(_mhz(parts[1]))
            max_sm.add(_mhz(parts[2]))
    if not sm:
        raise ValueError(f"{clocks_csv}: no clock sample in [{start}, {end}]")
    return {"samples": len(sm), "sm_min_mhz": min(sm), "sm_median_mhz": statistics.median(sm),
            "sm_max_mhz": max(sm), "max_sm_mhz": sorted(max_sm)}


def session_clocks(run_dir):
    path = Path(run_dir) / "clocks.jsonl"
    return {r["session_id"]: [r["clock_sm_start_mhz"], r["clock_sm_end_mhz"]] for r in load_jsonl(path)} if path.exists() else None


def _last_json(text, marker):
    found = re.findall(re.escape(marker) + r"\s*(\{.*\})\s*$", text, flags=re.MULTILINE)
    return json.loads(found[-1]) if found else None


def ram_miss(run_dir):
    run_dir = Path(run_dir)
    log = (run_dir / "server.log").read_text(errors="replace")
    thread = _last_json(log, "exl3 RAM miss thread counters")
    stream = _last_json(log, "exl3 expert stream:")
    return {
        "scope": "server lifetime (warm-up, prefill and the timed set)",
        "warmup_rounds": len(list(run_dir.glob("results-warmup-*.jsonl"))),
        "thread": {k: thread.get(k) for k in RAM_MISS_KEYS} if thread else None,
        "stream": {k: stream.get(k) for k in STREAM_KEYS} if stream else None,
    }


def arm(run_dir, diskstats, clocks_csv, devices):
    per_turn, median = ms_per_token(run_dir)
    start, end = timed_window(run_dir)
    return {"run_dir": str(run_dir), "ms_per_token": per_turn, "median_ms_per_token": median,
            "decode": decode(run_dir),
            "clocks": {"timed_window": clock_summary(clocks_csv, *timed_window_utc(run_dir)),
                       "session_start_end_mhz": session_clocks(run_dir)},
            "ram_miss": ram_miss(run_dir),
            "disk": drive_split(load_jsonl(diskstats), start, end, devices)}


def main(argv):
    ref_dir, ref_disk, ref_clocks, new_dir, new_disk, new_clocks, *devices = argv
    out = {"ref": arm(ref_dir, ref_disk, ref_clocks, devices), "new": arm(new_dir, new_disk, new_clocks, devices),
           "identical": identity(ref_dir, new_dir)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
