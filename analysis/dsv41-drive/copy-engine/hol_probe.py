#!/usr/bin/env python3
"""How many blocked streams does it take before a fresh stream's work waits behind them?

N default-priority streams each queue a ~--sleep-ms kernel and a kernel that waits for it (a blocked queue head).
Then, on fresh streams created afterwards, it times: a tiny kernel at default priority, a tiny kernel at the greatest
priority, and a pinned H2D copy at the greatest priority (the copy thread's stream). A time near the sleep means that
work waited behind a blocked stream.

    python hol_probe.py --out hol.json [--counts 1,8,16,31,32,33,48,64] [--sleep-ms 300]
"""

import argparse
import json
import time

import torch


def _cycles_per_ms() -> float:
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(int(5e7))
    torch.cuda.synchronize()
    return 5e7 / ((time.perf_counter() - t0) * 1e3)


def _time(stream, work) -> float:
    done = torch.cuda.Event()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        work()
        done.record(stream)
    while not done.query():
        pass
    return round((time.perf_counter() - t0) * 1e3, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", default="1,8,16,31,32,33,48,64")
    ap.add_argument("--sleep-ms", type=float, default=300.0)
    a = ap.parse_args()
    torch.cuda.init()
    least, greatest = torch.cuda.Stream.priority_range()
    cycles = int(_cycles_per_ms() * a.sleep_ms)
    src = torch.empty(16 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    x = torch.zeros(1, device="cuda")
    scratch = torch.zeros(256, device="cuda")
    rows = []
    for n in [int(c) for c in a.counts.split(",")]:
        blockers = [torch.cuda.Stream() for _ in range(n)]
        default_fresh = torch.cuda.Stream()
        high_fresh = torch.cuda.Stream(priority=greatest)
        high_copy = torch.cuda.Stream(priority=greatest)
        torch.cuda.synchronize()
        for i, blocker in enumerate(blockers):
            with torch.cuda.stream(blocker):
                torch.cuda._sleep(cycles)
                scratch[i : i + 1].add_(1)
        time.sleep(0.002)
        row = {
            "blocked_streams": n,
            "default_kernel_ms": _time(default_fresh, lambda: x.add_(1)),
            "greatest_kernel_ms": _time(high_fresh, lambda: x.add_(1)),
            "greatest_h2d_ms": _time(high_copy, lambda: dst.copy_(src, non_blocking=True)),
        }
        torch.cuda.synchronize()
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        json.dump({"priority_range": [least, greatest], "sleep_ms": a.sleep_ms, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
