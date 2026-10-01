"""The RAM-miss service's cost to serve one demand record, as its stage trace times it: from the moment the service
sees the post (observed) to the moment it has finished the record (done). Records carry SM-hit lanes only, with a GPU
hot record, so the service reads the record and its hot record and touches the slots: the read path, nothing else.

Each post is served before the next, and the service idles --gap-us between them, so every record is found from an
idle poll: production's decode steady state. --perf counts the service thread's user-mode events with perf stat -t.

Run on divix01 under cc-gpu.lock from test/manual/dsv41, PYTHONPATH at the tree under test (the plan's Global
Constraints have the command).
"""

import argparse
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lease_chain_rig import TOP_K, Chain  # noqa: E402

import sglang  # noqa: E402
from sglang.kernels.ops.moe import expert_stream_transport  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

PERF_EVENTS = (
    "instructions:u",
    "br_misp_retired.all_branches:u",
    "machine_clears.memory_ordering:u",
    "mem_load_retired.l3_hit:u",
    "mem_load_l3_hit_retired.xsnp_none:u",
    "mem_load_l3_miss_retired.local_dram:u",
    "mem_load_l3_miss_retired.remote_dram:u",
)


def service_tid() -> int:
    for task in Path("/proc/self/task").iterdir():
        if (task / "comm").read_text().strip().endswith("ram-miss"):
            return int(task.name)
    raise RuntimeError("no RAM-miss service thread in this process")


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--gap-us", type=float, default=100.0)
    parser.add_argument("--cpu-core", type=int, default=-1)
    parser.add_argument("--busy-poll", action="store_true")
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args()
    print("sglang:", sglang.__file__)
    print(f"pause: {expert_stream_transport.pause_ns(variant='instr'):.1f} ns")
    with tempfile.TemporaryDirectory() as tmp:
        c = Chain(Path(tmp), start=False, gpu_hot=True)
        try:
            row = 0
            experts = list(range(TOP_K))
            c.host.enable_trace(capacity=args.posts + args.warmup + 64)
            busy = {"busy_poll": True} if args.busy_poll else {}
            c.host.start_thread(cpu_core=args.cpu_core, fatal_wait_s=60.0, **busy)
            print("service core:", c.host.counters()["spin_cpu"], "busy_poll:", args.busy_poll)
            c.plan(experts, row)
            backend, plan = c.backends[row], c.plans[row]
            hot_slots = torch.arange(TOP_K, dtype=torch.int64, device="cuda")
            backend.hot_slots, backend.hot_capacity = hot_slots, TOP_K
            # One production gather of all misses: the tier reads the experts into RAM and its delta maps them on the
            # device, so every post below is SM hits the host and the device agree on.
            c.gather(row)
            torch.cuda.synchronize()
            assert c.handled(timeout_s=10.0), "the service did not serve the warming gather"
            backend._stage_planned(plan)

            def post_and_wait() -> None:
                c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, hot_slots=hot_slots,
                           hot_capacity=TOP_K)
                torch.cuda.synchronize()
                assert c.handled(timeout_s=5.0), "the service did not finish the record"
                time.sleep(args.gap_us * 1e-6)

            for _ in range(args.warmup):
                post_and_wait()
            assert c.kinds(TOP_K) == [LaneKind.HIT_SM] * TOP_K, c.kinds(TOP_K)
            c.host.drain_trace()
            overruns = c.host.counters()["overruns"]
            perf = perf_out = None
            if args.perf:
                perf_out = Path(tmp) / "perf.csv"
                perf = subprocess.Popen(["perf", "stat", "-x,", "-o", str(perf_out), "-t", str(service_tid()),
                                         "-e", ",".join(PERF_EVENTS)])
                time.sleep(0.5)
            for _ in range(args.posts):
                post_and_wait()
            if perf is not None:
                perf.send_signal(signal.SIGINT)
                perf.wait()
            spans = [r["done"] - r["observed"] for r in c.host.drain_trace()]
            assert len(spans) == args.posts, (len(spans), args.posts)
            print(f"span ns over {len(spans)}: p10 {percentile(spans, 0.1)} median {int(statistics.median(spans))} "
                  f"p90 {percentile(spans, 0.9)} p99 {percentile(spans, 0.99)}")
            print("overruns in the window:", c.host.counters()["overruns"] - overruns)
            if perf_out is not None:
                for line in perf_out.read_text().splitlines():
                    fields = line.split(",")
                    if line.startswith("#") or len(fields) < 3 or not fields[0].replace(".", "").isdigit():
                        continue
                    print(f"perf {fields[2]}: {float(fields[0]) / args.posts:.2f} per post")
        finally:
            c.close()


if __name__ == "__main__":
    main()
