#!/usr/bin/env python3
"""Does a copy on another stream wait behind a blocked stream? (hardware-queue aliasing, CUDA_DEVICE_MAX_CONNECTIONS)

The copy wait CW spins in the decode graph until the copy thread's cuMemcpyAsync completes on the copy thread's own
stream. CUDA maps streams onto CUDA_DEVICE_MAX_CONNECTIONS hardware queues; two streams that share a queue are
serialised, so a copy queued behind the graph's CW (and the kernels after it) cannot start until CW ends: a deadlock
only the device deadline breaks.

For each of --streams freshly created streams B: on stream A, a ~--sleep-ms kernel (torch.cuda._sleep) followed by a
second kernel that must wait for it (the blocked head of A's queue, as F waits for CW); then a pinned H2D copy and an
event on B, polled from the host. B is "aliased" when its copy completes only after A's sleep, not in ~copy time.

    python alias_probe.py --out alias.json [--streams 40] [--sleep-ms 200]
Run it with CUDA_DEVICE_MAX_CONNECTIONS set as the server sets it (8) and at other values.
"""

import argparse
import json
import os
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=40)
    ap.add_argument("--sleep-ms", type=float, default=200.0)
    ap.add_argument("--priority", type=int, default=0, help="priority of the probed streams B (0 default, -1 high)")
    a = ap.parse_args()
    torch.cuda.init()
    a_stream = torch.cuda.Stream()
    src = torch.empty(64 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    x = torch.zeros(1, device="cuda")
    # Calibrate _sleep's cycles per ms.
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(int(1e8))
    torch.cuda.synchronize()
    cycles_per_ms = 1e8 / ((time.perf_counter() - t0) * 1e3)
    sleep_cycles = int(cycles_per_ms * a.sleep_ms)
    results = []
    for k in range(a.streams):
        b_stream = torch.cuda.Stream(priority=a.priority)
        # The copy alone, for its own time.
        with torch.cuda.stream(b_stream):
            dst.copy_(src, non_blocking=True)
        b_stream.synchronize()
        with torch.cuda.stream(a_stream):
            torch.cuda._sleep(sleep_cycles)
            x.add_(1)  # waits for the sleep: the blocked head of A's queue
        time.sleep(0.002)  # the sleep is running
        done = torch.cuda.Event()
        t0 = time.perf_counter()
        with torch.cuda.stream(b_stream):
            dst.copy_(src, non_blocking=True)
            done.record(b_stream)
        while not done.query():
            pass
        copy_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        row = {"stream": k, "copy_ms": round(copy_ms, 2), "aliased": copy_ms > a.sleep_ms / 2}
        results.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "priority": a.priority,
        "sleep_ms": a.sleep_ms,
        "aliased": [r["stream"] for r in results if r["aliased"]],
        "results": results,
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("aliased streams:", summary["aliased"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
