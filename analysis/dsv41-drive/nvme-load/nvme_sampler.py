"""Samples /proc/diskstats for the given NVMe namespaces every 100 ms until the arm driver logs DRIVER DONE.

Usage: nvme_sampler.py <out.jsonl> <done_log> [device ...]  (default: nvme0n1 nvme1n1 nvme2n1 nvme3n1)

One JSON line per sample: wall, CLOCK_MONOTONIC and CLOCK_MONOTONIC_RAW times (to line up with the nsys report), and
each device's raw cumulative counters; utilisation and throughput come from deltas afterwards.
"""

import json
import sys
import time

FIELDS = (
    "reads", "reads_merged", "sectors_read", "ms_reading", "writes", "writes_merged", "sectors_written",
    "ms_writing", "in_flight", "io_ticks_ms", "weighted_ms",
)
out_path, done_log = sys.argv[1], sys.argv[2]
DEVICES = tuple(sys.argv[3:]) or ("nvme0n1", "nvme1n1", "nvme2n1", "nvme3n1")
deadline = time.monotonic() + 4 * 3600
with open(out_path, "a") as out:
    n = 0
    while time.monotonic() < deadline:
        sample = {
            "wall": time.time(),
            "mono": time.clock_gettime(time.CLOCK_MONOTONIC),
            "mono_raw": time.clock_gettime(time.CLOCK_MONOTONIC_RAW),
        }
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if parts[2] in DEVICES:
                    sample[parts[2]] = dict(zip(FIELDS, map(int, parts[3:14])))
        missing = [d for d in DEVICES if d not in sample]
        if missing:
            sys.exit(f"not in /proc/diskstats: {missing}")
        out.write(json.dumps(sample) + "\n")
        n += 1
        if n % 50 == 0:
            out.flush()
            with open(done_log) as f:
                if "DRIVER DONE" in f.read():
                    break
        time.sleep(0.1)
