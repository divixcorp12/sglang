"""NVMe load from nvme_diskstats.jsonl: a 0.5 s timeline over the timed sessions, then per-phase aggregates.

Per device and interval: read MB/s, read IOPS, average request size, average queue depth (d weighted_ms / d t), and
%util (d io_ticks / d t). Phases are wall-clock windows passed as name=HH:MM:SS.s-HH:MM:SS.s.
"""

import datetime
import json
import sys

DEVICES = ("nvme0n1", "nvme3n1", "nvme2n1", "nvme1n1")
path = sys.argv[1]
day = sys.argv[2]  # YYYY-MM-DD
start_s, end_s = sys.argv[3], sys.argv[4]
phases = [a.split("=", 1) for a in sys.argv[5:]]


def wall(hms):
    return datetime.datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S.%f").timestamp()


samples = [json.loads(line) for line in open(path)]


def stats(a, b):
    dt = b["wall"] - a["wall"]
    out = {}
    for d in DEVICES:
        x, y = a[d], b[d]
        reads = y["reads"] - x["reads"]
        mb = (y["sectors_read"] - x["sectors_read"]) * 512 / 1e6
        out[d] = {
            "MBps": mb / dt,
            "iops": reads / dt,
            "req_kB": (mb * 1e3 / reads) if reads else 0.0,
            "qd": (y["weighted_ms"] - x["weighted_ms"]) / (dt * 1e3),
            "util": 100 * (y["io_ticks_ms"] - x["io_ticks_ms"]) / (dt * 1e3),
            "wMBps": (y["sectors_written"] - x["sectors_written"]) * 512 / 1e6 / dt,
        }
    return out


def window(t0, t1):
    inside = [s for s in samples if t0 <= s["wall"] <= t1]
    return stats(inside[0], inside[-1]) if len(inside) >= 2 else None


t0, t1 = wall(start_s), wall(end_s)
print("time        " + "".join(f"| {d:^34}" for d in DEVICES))
print("            " + "".join(f"| {'MB/s':>7} {'IOPS':>6} {'kB':>5} {'qd':>5} {'util':>5} " for d in DEVICES))
t = t0
while t < t1:
    w = window(t, t + 0.5)
    if w:
        label = datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:10]
        row = "".join(
            f"| {w[d]['MBps']:7.0f} {w[d]['iops']:6.0f} {w[d]['req_kB']:5.0f} {w[d]['qd']:5.1f} {w[d]['util']:5.0f} "
            for d in DEVICES
        )
        print(f"{label:12}{row}")
    t += 0.5
print()
for name, span in phases:
    a, b = span.split("-")
    w = window(wall(a), wall(b))
    print(f"== {name} ({a}-{b})")
    for d in DEVICES:
        s = w[d]
        print(
            f"   {d}: {s['MBps']:7.0f} MB/s read, {s['iops']:6.0f} IOPS, {s['req_kB']:4.0f} kB/req, "
            f"qd {s['qd']:5.2f}, util {s['util']:4.0f}%, write {s['wMBps']:5.0f} MB/s"
        )
