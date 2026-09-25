"""Peak VRAM per phase from a hot-cache-size smoke run (smoke.sh's vram.csv + phases.txt).

Usage: vram_peaks.py DIR [DIR ...]

vram.csv is nvidia-smi --query-gpu=timestamp,memory.used at -lms 50 (MiB); phases.txt holds the driver's phase
markers on the same wall clock. Per phase: max memory.used and its time. "idle" is memory.used at the healthy
marker (after startup and graph capture, before any request). Headroom = total - peak.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

TOTAL_MIB = 32607  # nvidia-smi memory.total on divix01's RTX 5090


def ts(s: str) -> float:
    return datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def main() -> int:
    for d in sys.argv[1:]:
        samples = []
        for line in open(os.path.join(d, "vram.csv")):
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                samples.append((ts(parts[0]), int(parts[1])))
            except ValueError:
                continue
        marks = [(ts(l[:23]), l[24:].strip()) for l in open(os.path.join(d, "phases.txt")) if l.strip()]
        print(f"== {d}: {len(samples)} samples, median gap "
              f"{sorted(b[0] - a[0] for a, b in zip(samples, samples[1:]))[len(samples) // 2] * 1e3:.0f} ms")
        at = {name: t for t, name in marks}
        if "healthy" in at:
            idle = [m for t, m in samples if at["healthy"] - 1 <= t <= at["healthy"]]
            print(f"  idle at healthy: {idle[-1] if idle else 'n/a'} MiB")
        bounds = marks + [(float("inf"), "end")]
        for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
            win = [(t, m) for t, m in samples if t0 <= t < t1]
            if not win:
                continue
            tpk, pk = max(win, key=lambda x: x[1])
            print(f"  {name:12s} {t1 - t0 if t1 != float('inf') else 0:7.1f} s  peak {pk:6d} MiB "
                  f"(+{tpk - t0:6.1f} s)  headroom {TOTAL_MIB - pk:6d} MiB")
        pk = max(m for _, m in samples)
        print(f"  overall peak {pk} MiB = {pk / 1024:.2f} GiB; headroom {TOTAL_MIB - pk} MiB = "
              f"{(TOTAL_MIB - pk) / 1024:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
