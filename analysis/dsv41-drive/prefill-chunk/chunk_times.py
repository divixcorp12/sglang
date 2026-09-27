"""Summarise one chunk_smoke.sh run: TTFT, peak VRAM, and per-chunk prefill time of the long prompt.

Usage: chunk_times.py <run dir>

Chunk times come from server.log's "Prefill batch" lines (a line is logged when a chunk is scheduled, so a
chunk's time is the gap to the next line). Expert movement comes from stages.jsonl, split into layers that see
the whole chunk and the decoder-replay layers that see only its trailing window.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

run = sys.argv[1]


def stamp(text: str, fmt: str) -> float:
    return datetime.strptime(text, fmt).timestamp()


phases = {}
for line in open(f"{run}/phases.txt"):
    day, clock, name = line.split()
    phases[name] = stamp(f"{day} {clock}", "%Y/%m/%d %H:%M:%S.%f")

long = json.load(open(f"{run}/long.json"))
print(f"prompt {long['prompt_tokens']}  ttft {long['ttft_s']} s  "
      f"prefill {long['prompt_tokens'] / long['ttft_s']:.1f} tok/s  decode {long['decode_ms_per_token']} ms/token")

peak_idle, peak_long = 0, 0
for line in open(f"{run}/vram.csv"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2 or not parts[1].isdigit():
        continue
    t = stamp(parts[0], "%Y/%m/%d %H:%M:%S.%f")
    if phases.get("healthy", 1e18) <= t < phases.get("long_start", 0):
        peak_idle = max(peak_idle, int(parts[1]))
    if phases.get("long_start", 1e18) <= t <= phases.get("long_end", 0):
        peak_long = max(peak_long, int(parts[1]))
print(f"vram MiB: peak before long {peak_idle}, peak during long {peak_long}, "
      f"long adds {peak_long - peak_idle}; OOM retries {open(f'{run}/retries.txt').read().strip()}")

line_re = re.compile(r"^\[(\S+ \S+)\] Prefill batch.*#new-token: (\d+), #cached-token: (\d+)")
starts = []
for line in open(f"{run}/server.log"):
    m = line_re.match(line)
    if m:
        t = stamp(m.group(1), "%Y-%m-%d %H:%M:%S")
        if t >= int(phases["long_start"]):
            starts.append((t, int(m.group(2))))
chunk_s = [(starts[i + 1][0] - t, n) for i, (t, n) in enumerate(starts[:-1])]
full = [s for s, n in chunk_s if n > 1]
if full:
    print(f"chunks {len(full)}: median {statistics.median(full):.0f} s, first {full[0]:.0f} s, last {full[-1]:.0f} s "
          f"(1 s log resolution)")

# stages.jsonl: per (forward, layer) records; keep the long prompt's extend forwards.
fw = defaultdict(lambda: defaultdict(lambda: {"n": 0, "tokens": 0, "experts": 0, "vram": 0, "ram": 0, "read": 0.0}))
t_first = {}
for line in open(f"{run}/stages.jsonl"):
    r = json.loads(line)
    if r.get("phase") != "extend" or "forward" not in r or r["t"] is None:
        continue
    group = "whole" if r["layer"] <= 20 else "tail"
    g = fw[r["forward"]][group]
    g["n"] += 1
    g["tokens"] = max(g["tokens"], r["tokens"])
    g["experts"] += len(r["experts"])
    g["vram"] += r["vram_miss"]
    g["ram"] += r["ram_miss"]
    g["read"] += r["read_ms"]
long_fw = [k for k in sorted(fw) if fw[k]["whole"]["tokens"] > 256]
print("forwards: fwd | layers 0-20: tokens experts/layer vram_miss/layer ram_miss/layer | "
      "layers 21-39: tokens experts/layer | read_s")
for k in long_fw:
    w, t = fw[k]["whole"], fw[k]["tail"]
    nw, nt = max(w["n"], 1), max(t["n"], 1)
    print(f"{k:5d} | {w['tokens']:6d} {w['experts'] / nw:6.1f} {w['vram'] / nw:6.1f} {w['ram'] / nw:6.1f} | "
          f"{t['tokens']:5d} {t['experts'] / nt:6.1f} | {(w['read'] + t['read']) / 1e3:6.1f}")
