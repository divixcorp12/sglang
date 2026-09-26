"""Prefill GPU idle time against NVMe activity, in 100 ms bins.

Usage: prefill_idle_vs_nvme.py MAIN_SQLITE NVME_DISKSTATS_JSONL PREFILL_START_S DECODE_START_S
(trace-relative seconds, as compare_arms.py prints them). The trace's UTC session start maps it to diskstats wall time.
"""

import json
import sqlite3
import sys

db_path, ds_path, p0_s, d0_s = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
utc0 = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0] / 1e9
p0, d0 = int(p0_s * 1e9), int(d0_s * 1e9)
iv = db.execute(
    "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where start < ? and end > ? "
    "union all select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where start < ? and end > ?",
    (d0, p0, d0, p0),
).fetchall()
BIN = 100_000_000
nbins = (d0 - p0) // BIN
busy = [0] * nbins
merged, cur = [], None
for a, b in sorted(iv):
    a, b = max(a, p0), min(b, d0)
    if cur and a <= cur[1]:
        cur[1] = max(cur[1], b)
    else:
        if cur:
            merged.append(cur)
        cur = [a, b]
merged.append(cur)
for a, b in merged:
    t = a
    while t < b:
        k = (t - p0) // BIN
        if k >= nbins:
            break
        edge = p0 + (k + 1) * BIN
        busy[k] += min(b, edge) - t
        t = min(b, edge)

samples = [json.loads(line) for line in open(ds_path)]
walls = [s["wall"] for s in samples]


def nvme_util(t_wall):
    import bisect

    i = bisect.bisect_left(walls, t_wall)
    j = bisect.bisect_left(walls, t_wall + 0.1)
    if j >= len(samples) or i >= j:
        return None
    x, y = samples[i], samples[j]
    dt = y["wall"] - x["wall"]
    return max((y[d]["io_ticks_ms"] - x[d]["io_ticks_ms"]) / (dt * 1e3) for d in ("nvme0n1", "nvme3n1"))


groups = {"NVMe busy >=50%": [0, 0.0], "NVMe busy 10-50%": [0, 0.0], "NVMe busy <10%": [0, 0.0]}
for k in range(nbins):
    u = nvme_util(utc0 + (p0 + k * BIN) / 1e9)
    if u is None:
        continue
    g = "NVMe busy >=50%" if u >= 0.5 else "NVMe busy 10-50%" if u >= 0.1 else "NVMe busy <10%"
    groups[g][0] += 1
    groups[g][1] += (BIN - busy[k]) / 1e6
total_idle = sum(v[1] for v in groups.values())
print(f"prefill {p0_s:.3f}-{d0_s:.3f} s: GPU idle {total_idle / 1e3:.2f} s over {nbins} bins")
for g, (n, idle_ms) in groups.items():
    print(f"  {g:18s}: {n:4d} bins ({n / 10:.1f} s), GPU idle {idle_ms / 1e3:.2f} s ({idle_ms / (n * 100) * 100 if n else 0:.0f}% of those bins)")
