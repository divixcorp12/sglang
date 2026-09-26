"""Read rate of each mirror while it is busy: 100 ms intervals by utilisation, over a wall-clock window.

Usage: nvme_busy_rate.py nvme_diskstats.jsonl YYYY-MM-DD HH:MM:SS.s HH:MM:SS.s
"""

import datetime
import json
import sys

path, day, a, b = sys.argv[1:5]


def wall(hms):
    return datetime.datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S.%f").timestamp()


t0, t1 = wall(a), wall(b)
samples = [json.loads(line) for line in open(path)]
samples = [s for s in samples if t0 <= s["wall"] <= t1]
for dev in ("nvme0n1", "nvme3n1"):
    buckets = {}
    total_mb = total_busy = 0.0
    for x, y in zip(samples, samples[1:]):
        dt = y["wall"] - x["wall"]
        util = (y[dev]["io_ticks_ms"] - x[dev]["io_ticks_ms"]) / (dt * 1e3)
        mb = (y[dev]["sectors_read"] - x[dev]["sectors_read"]) * 512 / 1e6
        total_mb += mb
        total_busy += util * dt
        key = min(int(util * 10), 9)
        e = buckets.setdefault(key, [0, 0.0, 0.0])
        e[0] += 1
        e[1] += mb
        e[2] += dt
    print(f"{dev}: {total_mb / total_busy / 1e3:.2f} GB/s per busy second "
          f"({total_mb:.0f} MB over {total_busy:.2f} busy s of {samples[-1]['wall'] - samples[0]['wall']:.1f} s)")
    for key in sorted(buckets):
        n, mb, dt = buckets[key]
        print(f"   util {key * 10:3d}-{key * 10 + 9:3d}%: {n:4d} intervals, {mb / dt / 1e3:5.2f} GB/s")
