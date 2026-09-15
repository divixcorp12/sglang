"""Doorbell vs in-graph expert-row copy latency on production-shaped rows (throwaway measurement).

Rows: 1024 x 2,764,808 B arena-registered like production (cc-pcie-bench Setup, imported).
For N in {1, 3, 10, 30} random rows/slots per iteration:
  ingraph      captured graph of copy_expert_row_segments_gpu, sync -> replay -> sync
  doorbell     captured graph of post+wait, sync -> replay -> sync (end to end, GPU blocked in wait)
  post_done    eager post, post launch -> copy completion observed by the thread (thread clock)
  reaction     eager post: GPU event after post observed by a Python spin -> thread saw the head
  wait_only    captured graph of wait for an already-completed request (waiter + no-op fallback launch)
Every iteration's destination rows are checked byte-exact.
"""

import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/models/slang/nvfp4-work/cc-pcie-bench")

from bench import TENSORS, Setup  # noqa: E402

from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu  # noqa: E402
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier  # noqa: E402

DEV = torch.device("cuda:0")
GIB = 1024**3


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summary(samples):
    return {
        "n": len(samples),
        "p50": statistics.median(samples),
        "p90": percentile(samples, 0.9),
        "min": min(samples),
        "max": max(samples),
    }


class Bench:
    def __init__(self, args):
        self.args = args
        self.setup = Setup(args.rows, "arena", seed=0)
        self.generator = torch.Generator().manual_seed(5)
        self.out = open(args.out, "a")

    def emit(self, record):
        record.update({"tag": self.args.tag})
        print(json.dumps(record), flush=True)
        self.out.write(json.dumps(record) + "\n")
        self.out.flush()

    def pick(self, count):
        rows = torch.randperm(self.args.rows, generator=self.generator)[:count].tolist()
        slots = torch.randperm(self.args.rows, generator=self.generator)[:count].tolist()
        return rows, slots

    def plan_tensors(self, count):
        return (
            torch.zeros(count, dtype=torch.int64, device=DEV),
            torch.zeros(count, dtype=torch.int32, device=DEV),
            torch.full((1,), count, dtype=torch.int32, device=DEV),
        )

    def load(self, plan, rows, slots):
        plan[0].copy_(torch.tensor(rows, dtype=torch.int64), non_blocking=False)
        plan[1].copy_(torch.tensor(slots, dtype=torch.int32), non_blocking=False)
        torch.cuda.synchronize(DEV)

    def verify(self, rows, slots):
        torch.cuda.synchronize(DEV)
        for name, _, _ in TENSORS:
            source = self.setup.src[name].view(self.setup.rows, -1)
            got = self.setup.dst[name].view(self.setup.rows, -1)[torch.tensor(slots, device=DEV)].cpu()
            if not torch.equal(got.view(torch.uint8), source[torch.tensor(rows)].view(torch.uint8)):
                return False
        return True

    def capture(self, body):
        side = torch.cuda.Stream(DEV)
        with torch.cuda.stream(side):
            for _ in range(3):
                body()
        torch.cuda.current_stream(DEV).wait_stream(side)
        torch.cuda.synchronize(DEV)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize(DEV)
        return graph

    def timed_replays(self, graph, plan, count):
        samples, exact = [], True
        for index in range(self.args.warmup + self.args.iters):
            rows, slots = self.pick(count)
            self.load(plan, rows, slots)
            started = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(DEV)
            elapsed = (time.perf_counter() - started) * 1e3
            exact = exact and self.verify(rows, slots)
            if index >= self.args.warmup:
                samples.append(elapsed)
        return samples, exact

    def run_ingraph(self, count):
        plan = self.plan_tensors(count)
        graph = self.capture(lambda: copy_expert_row_segments_gpu(self.setup.segments, *plan))
        samples, exact = self.timed_replays(graph, plan, count)
        nbytes = count * 2_764_808
        self.emit({"method": "ingraph", "count": count, "ms": summary(samples), "exact": exact,
                   "gib_s_p50": nbytes / GIB / (statistics.median(samples) / 1e3)})

    def run_doorbell(self, count, prefer_overlap):
        label = {"prefer_overlap": prefer_overlap}
        plan = self.plan_tensors(count)
        with ExpertDoorbellCopier(self.setup.segments, count, cpu_core=self.args.spin_core,
                                  prefer_overlap=prefer_overlap,
                                  timeout_polls=self.args.timeout_polls) as copier:
            def body():
                copier.post(*plan, tag=0)
                copier.wait(tag=0)

            graph = self.capture(body)
            base = copier.stats()
            samples, exact = self.timed_replays(graph, plan, count)
            requests = copier.trace()[-self.args.iters:]
            stats = copier.stats()
            copy_ms = [(r["complete_ns"] - r["seen_ns"]) / 1e6 for r in requests if r["status"] == "serviced"]
            nbytes = count * 2_764_808
            self.emit({"method": "doorbell_graph", "count": count, **label, "ms": summary(samples), "exact": exact,
                       "thread_seen_to_copy_complete_ms": summary(copy_ms),
                       "copy_gib_s_p50": nbytes / GIB / (statistics.median(copy_ms) / 1e3),
                       "timeouts": stats["timeouts"] - base["timeouts"], "serviced": stats["serviced"] - base["serviced"],
                       "spin_cpu": stats["spin_cpu"]})

            post_done, reaction, launch_seen, exact_eager = [], [], [], True
            for index in range(self.args.warmup + self.args.iters):
                rows, slots = self.pick(count)
                self.load(plan, rows, slots)
                event = torch.cuda.Event()
                launched = now_ns()
                copier.post(*plan, tag=0)
                event.record()
                while not event.query():
                    pass
                gpu_posted = now_ns()
                request = None
                while request is None or request["complete_ns"] == 0:
                    trace = copier.trace()
                    request = trace[-1] if trace and trace[-1]["seq"] == copier.stats()["posted"] else None
                copier.wait(tag=0)
                exact_eager = exact_eager and self.verify(rows, slots)
                if index >= self.args.warmup:
                    post_done.append((request["complete_ns"] - launched) / 1e6)
                    reaction.append((request["seen_ns"] - gpu_posted) / 1e3)
                    launch_seen.append((request["seen_ns"] - launched) / 1e3)
            self.emit({"method": "doorbell_eager", "count": count, **label, "exact": exact_eager,
                       "post_launch_to_copy_complete_ms": summary(post_done),
                       "gpu_post_event_observed_to_thread_seen_us": summary(reaction),
                       "post_launch_to_thread_seen_us": summary(launch_seen),
                       "timeouts": copier.stats()["timeouts"] - stats["timeouts"]})

            wait_graph = self.capture(lambda: copier.wait(tag=0))
            wait_samples = []
            for index in range(self.args.warmup + self.args.iters):
                copier.post(*plan, tag=0)
                torch.cuda.synchronize(DEV)
                while copier.stats()["done"] != copier.stats()["posted"]:
                    pass
                started = time.perf_counter()
                wait_graph.replay()
                torch.cuda.synchronize(DEV)
                if index >= self.args.warmup:
                    wait_samples.append((time.perf_counter() - started) * 1e3)
            self.emit({"method": "wait_only_completed", "count": count, **label, "ms": summary(wait_samples),
                       "timeouts": copier.stats()["timeouts"] - stats["timeouts"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 3, 10, 30])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--spin-core", type=int, default=71)
    parser.add_argument("--timeout-polls", type=int, default=20_000)
    parser.add_argument("--prefer-overlap", nargs="+", type=int, default=[1, 0])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--out", default="/data/models/slang/nvfp4-work/cc-doorbell/results/results.jsonl")
    args = parser.parse_args()
    bench = Bench(args)
    try:
        for count in args.counts:
            bench.run_ingraph(count)
            for prefer_overlap in args.prefer_overlap:
                bench.run_doorbell(count, prefer_overlap == 1)
    finally:
        bench.setup.close()


if __name__ == "__main__":
    main()
