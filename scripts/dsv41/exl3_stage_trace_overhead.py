"""Host CPU cost of the RAM-miss stage trace: trace off vs on, interleaved, per request.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. Native service time per request (post, serve, publish) with
cache-resident buffered reads from a tmpfs checkpoint: the CPU cost of the trace, which is the part that
does not scale with drive latency. It says nothing about O_DIRECT reads, a real drive, the graph path or
any GPU. Quote it only with that label and with the "conditions" block this script writes.

It refuses to produce numbers unless --box-clear is given: a measurement on a contended box is worse
than none. Without the flag it only proves the harness runs (--smoke) and prints no timings.

Run:  taskset -c 0-63 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=<wt>/python \
      python scripts/dsv41/exl3_stage_trace_overhead.py --box-clear --out result.json
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.kernels.ops.moe.exl3_ram_miss import STAGE_FIELDS
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

EXPERTS, CAPACITY = 64, 32


def _busy_foreign(limit=6):
    """Processes using CPU that are not this one: what a reader must see next to the number."""
    out = subprocess.run(
        ["ps", "-eo", "pid,pcpu,psr,comm", "--sort=-pcpu"], capture_output=True, text=True
    ).stdout.splitlines()[1 : limit + 1]
    return [line.split(None, 3) for line in out if int(line.split()[0]) != os.getpid()]


def _conditions():
    def head(path):
        try:
            return subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ""

    import sglang

    tree = str(Path(sglang.__file__).resolve().parents[2])
    return {
        "load1": os.getloadavg()[0],
        "affinity": sorted(os.sched_getaffinity(0)),
        "busy_foreign": _busy_foreign(),
        "tree": tree,
        "tree_head": head(tree),
        "stage_record_words": len(STAGE_FIELDS),
        "stage_record_bytes": len(STAGE_FIELDS) * 8,
        "torch": torch.__version__,
        "kind": "host CPU cost per request; cache-resident buffered reads (tmpfs); no O_DIRECT, no drive, no GPU",
    }


def _run_delay_ns():
    """Time this thread has spent runnable but not running: another process held the core. Zero over
    a block means nobody competed for it, which no load average can say."""
    with open("/proc/thread-self/schedstat") as f:
        return int(f.read().split()[1])


def _core_ticks(core):
    """(busy, total) jiffies of one core from /proc/stat, for the SMT sibling's activity."""
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith(f"cpu{core} "):
                fields = [int(x) for x in line.split()[1:]]
                return sum(fields) - fields[3] - fields[4], sum(fields)
    return 0, 0


def _siblings(core):
    try:
        with open(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list") as f:
            return [int(x) for x in f.read().replace("-", ",").split(",") if x.strip() and int(x) != core]
    except OSError:
        return []


class Arm:
    def __init__(self, root, name, trace, ring):
        root.mkdir()
        self.name, self.trace = name, trace
        self.s = ram_miss_setup(root, capacity=CAPACITY, experts=EXPERTS)
        self.page = new_page(pin=False)
        self.host = Exl3RamMissHost(
            self.s.tables, page=self.page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), direct=False
        )
        if trace:
            self.host.enable_trace(capacity=ring)
        self.cursor = 0

    def block(self, rows_per_request, requests):
        """Wall time of `requests` requests, each needing a window of experts that was evicted since."""
        # 0 rows: one resident expert asked for again, so the request reads nothing.
        windows = EXPERTS // max(1, rows_per_request)
        clocks0 = self.host.trace_clock_reads()
        delay0 = _run_delay_ns()
        t0 = time.perf_counter_ns()
        for _ in range(requests):
            w = self.cursor % windows
            self.cursor += 1
            ids = [w * rows_per_request + j for j in range(rows_per_request)] if rows_per_request else [0]
            seq = sim_post(self.page, 1, need=ids, protect=ids)
            self.host.pump()
            sim_wait(self.page, seq, timeout_s=5.0)
        elapsed = time.perf_counter_ns() - t0
        self.run_delay_ns = _run_delay_ns() - delay0
        self.clock_reads_per_request = (self.host.trace_clock_reads() - clocks0) / requests
        rows = 0
        if self.trace:
            rows = sum(r["rows"] for r in self.host.drain_trace())  # outside the timed span
        return elapsed / requests, rows / requests

    def close(self):
        self.host.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-clear", action="store_true", help="the lead said the box is quiet")
    ap.add_argument("--smoke", action="store_true", help="tiny run, no timings printed")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 3, 8], help="rows read per request, at most 8")
    ap.add_argument("--ring", type=int, default=8192, help="trace ring slots; 8192 is the service's default")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not (args.box_clear or args.smoke):
        raise SystemExit("refusing to measure: pass --box-clear once the box is confirmed clear")
    if args.smoke:
        args.reps, args.requests, args.rows = 2, 20, [1, 3]
    if max(args.rows) > 8:
        raise SystemExit("a request carries at most 8 ids (kMaxIds); more would silently read fewer rows")

    base = Path(tempfile.mkdtemp(prefix="stage_trace_overhead_", dir="/dev/shm"))
    arms = {"off": Arm(base / "off", "off", False, args.ring), "on": Arm(base / "on", "on", True, args.ring)}
    conditions = {"before": _conditions()}
    result = {"conditions": conditions, "by_rows": {}}
    try:
        # Touch every slot of the ring once before measuring: its first pass pays a page fault per ~2
        # records, which the steady state (the ring wraps) does not.
        for _ in range(args.ring // args.requests + 2):
            arms["on"].block(0, args.requests)
        for rows in args.rows:
            for arm in arms.values():
                arm.block(rows, args.requests)  # warmup: page cache, branch predictors
            samples = {"off": [], "on": []}
            served_rows = []
            clocks = []
            loads = []
            delays = {"off": [], "on": []}
            core = min(os.sched_getaffinity(0))
            sibling_before = [(c, _core_ticks(c)) for c in _siblings(core)]
            for rep in range(args.reps):
                order = ["off", "on"] if rep % 2 == 0 else ["on", "off"]  # alternate who goes first
                for name in order:
                    ns, rr = arms[name].block(rows, args.requests)
                    samples[name].append(ns)
                    delays[name].append(arms[name].run_delay_ns)
                    if name == "on":
                        served_rows.append(rr)
                        clocks.append(arms["on"].clock_reads_per_request)
                loads.append(os.getloadavg()[0])
            paired = [on - off for on, off in zip(samples["on"], samples["off"])]
            result["by_rows"][rows] = {
                "rows_read_per_request": statistics.mean(served_rows),
                "clock_reads_per_request": statistics.mean(clocks),
                "off_ns_per_request": samples["off"],
                "on_ns_per_request": samples["on"],
                "delta_ns_median": statistics.median(paired),
                "delta_ns_min": min(paired),
                "delta_ns_max": max(paired),
                "off_ns_median": statistics.median(samples["off"]),
                "load1_per_rep": loads,
                "run_delay_ns_per_block": delays,
                "core": core,
                "sibling_busy_fraction": {
                    c: (
                        (_core_ticks(c)[0] - before[0]) / max(1, _core_ticks(c)[1] - before[1])
                    )
                    for c, before in sibling_before
                },
            }
    finally:
        for arm in arms.values():
            arm.close()
        shutil.rmtree(base, ignore_errors=True)
    conditions["after"] = _conditions()
    if args.smoke:
        print("smoke ok: arms built, blocks ran, trace drained (no timings reported)")
        return
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    for rows, r in result["by_rows"].items():
        print(
            f"rows={rows:>2} read/request={r['rows_read_per_request']:.1f} clocks/request={r['clock_reads_per_request']:.0f} off={r['off_ns_median']:.0f} ns "
            f"delta(on-off) median={r['delta_ns_median']:.0f} range=[{r['delta_ns_min']:.0f},{r['delta_ns_max']:.0f}] ns/request"
        )
    print("conditions:", json.dumps({k: conditions["before"][k] for k in ("load1", "affinity", "busy_foreign", "kind")}))


if __name__ == "__main__":
    main()
