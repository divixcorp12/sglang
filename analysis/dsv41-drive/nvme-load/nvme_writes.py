"""Per-second writes (MB/s) and queue depth for nvme3n1 and nvme1n1 over the whole sampler window."""

import datetime
import json
import sys

samples = [json.loads(line) for line in open(sys.argv[1])]
by_sec = {}
for s in samples:
    by_sec.setdefault(int(s["wall"]), s)
secs = sorted(by_sec)
for a, b in zip(secs, secs[1:]):
    x, y = by_sec[a], by_sec[b]
    dt = y["wall"] - x["wall"]
    row = []
    for d in ("nvme3n1", "nvme1n1"):
        w = (y[d]["sectors_written"] - x[d]["sectors_written"]) * 512 / 1e6 / dt
        r = (y[d]["sectors_read"] - x[d]["sectors_read"]) * 512 / 1e6 / dt
        qd = (y[d]["weighted_ms"] - x[d]["weighted_ms"]) / (dt * 1e3)
        row.append(f"{d} r {r:6.0f} w {w:5.0f} qd {qd:7.1f}")
    if any(" w     0" not in c for c in row):
        print(datetime.datetime.fromtimestamp(a).strftime("%H:%M:%S"), " | ".join(row))
