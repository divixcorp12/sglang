"""How long does a CUDA graph launch hold a host-to-device copy queued after it launched?

Throwaway probe for the doorbell prototype. Every time is a CLOCK_MONOTONIC host time at which a CUDA event,
or a doorbell request's completion, was first observed by a spin loop.

(a) One compute-only graph (no doorbell kernel, no acquire, no poll), host-issued side-stream copies:
    long_copy_before     copy queued, then the graph replayed (control)
    long_copy_after      graph replayed, copy queued immediately after replay() returned
    long_copy_after_half graph replayed, copy queued half a replay later
    eager_copy_after     the same compute launched eagerly, copy queued after
(b) K short per-layer graphs replayed back to back, host-issued copies:
    layers_copy_after_first  replay 0, queue copy, replay 1..K-1
    layers_copy_after_all    replay 0..K-1, then queue copy
    layers_gate_after_first  replay 0, queue copy, host waits for the copy, replay 1..K-1
(c) The doorbell between K per-layer graphs; the thread is quiesced across every capture. Each case sets:
    post      "graph": graph j posts tag j+1; "eager": graph j is compute only and the host launches the
              post for tag j+1 eagerly right after replaying graph j
    waits     "graph": graph j > 0 starts with wait(tag j); "eager": every wait is launched after the token
    gated     host waits for the thread to enqueue request j+1 before launching replay j+1
    head_store / poll_mode / prefer_overlap as in ExpertDoorbellCopier
    The thread's copy arm comes from --copy-api, --src-access-order and --torch-stream. Every plan maps a source
    row to the same destination slot, so the bytes after a token do not depend on which copy (thread or
    fallback) landed last; after each token the whole destination is compared with that image.
"""

import argparse
import json
import statistics
import time

import torch

DEV = torch.device("cuda:0")
ROW_BYTES = 2_764_808
SOURCE_ROWS = 32
DOORBELL_SLOTS = 64

DOORBELL_CASES = (
    {"name": "doorbell_layers", "post": "graph", "waits": "graph"},
    {"name": "doorbell_layers_gated", "post": "graph", "waits": "graph", "gated": True},
    {"name": "doorbell_post_only_layers", "post": "graph", "waits": "eager"},
    {"name": "doorbell_post_only_layers_no_overlap_flag", "post": "graph", "waits": "eager", "prefer_overlap": False},
    {"name": "doorbell_layers_volatile_head", "post": "graph", "waits": "graph", "head_store": "volatile"},
    {"name": "doorbell_layers_volatile_head_nc_poll", "post": "graph", "waits": "graph", "head_store": "volatile",
     "poll_mode": "noncoherent"},
    {"name": "compute_graphs_eager_post", "post": "eager", "waits": "eager"},
    {"name": "compute_graphs_eager_post_no_overlap_flag", "post": "eager", "waits": "eager", "prefer_overlap": False},
)


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def emit(record, out):
    line = json.dumps(record)
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


def summary(samples):
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50": round(statistics.median(ordered), 3),
        "p90": round(ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def observe(events):
    """Host time at which each event is first seen complete."""
    times = [None] * len(events)
    deadline = time.perf_counter() + 60.0
    while any(t is None for t in times) and time.perf_counter() < deadline:
        for index, event in enumerate(events):
            if times[index] is None and event.query():
                times[index] = now_ns()
    return times


def busy_wait(nanoseconds):
    end = now_ns() + nanoseconds
    while now_ns() < end:
        pass


def main_event():
    event = torch.cuda.Event()
    event.record()
    return event


def drain(copier, timeout_s=30.0):
    """Wait until the thread has handled every posted request and its copies completed."""
    torch.cuda.synchronize(DEV)
    posted = copier.stats()["posted"]
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        trace = copier.trace()
        last = trace[-1] if trace else None
        if posted == 0 or (last and last["seq"] == posted and (last["status"] != "serviced" or last["complete_ns"])):
            return True
    return False


class Probe:
    def __init__(self, args):
        self.args = args
        self.out = open(args.out, "a")
        copy_bytes = args.rows * ROW_BYTES
        self.source = torch.randint(0, 256, (SOURCE_ROWS * ROW_BYTES,), dtype=torch.uint8).pin_memory()
        self.destination = torch.zeros(SOURCE_ROWS * ROW_BYTES, dtype=torch.uint8, device=DEV)
        self.copy_source = self.source[:copy_bytes]
        self.copy_destination = self.destination[:copy_bytes]
        self.side = torch.cuda.Stream(DEV)
        self.matrix = torch.randn((args.size, args.size), device=DEV)
        self.product = torch.empty_like(self.matrix)
        self.long_graph = self.capture(lambda: self.compute(args.long_matmuls))
        self.layer_graphs = [self.capture(lambda: self.compute(args.layer_matmuls)) for _ in range(args.layers)]

    def compute(self, matmuls):
        for _ in range(matmuls):
            torch.matmul(self.matrix, self.matrix, out=self.product)

    def capture(self, body, before_capture=None, after_capture=None):
        with torch.cuda.stream(self.side):
            for _ in range(2):
                body()
        torch.cuda.current_stream(DEV).wait_stream(self.side)
        torch.cuda.synchronize(DEV)
        if before_capture is not None:
            before_capture()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                body()
            torch.cuda.synchronize(DEV)
        finally:
            if after_capture is not None:
                after_capture()
        return graph

    def queue_copy(self):
        issued = now_ns()
        with torch.cuda.stream(self.side):
            self.copy_destination.copy_(self.copy_source, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return issued, event

    def exact(self):
        torch.cuda.synchronize(DEV)
        return torch.equal(self.copy_destination[:: 1 << 16].cpu(), self.copy_source[:: 1 << 16])

    def idle_copy_ms(self):
        samples = []
        for _ in range(self.args.iters):
            self.copy_destination.zero_()
            torch.cuda.synchronize(DEV)
            issued, event = self.queue_copy()
            (done,) = observe([event])
            samples.append((done - issued) / 1e6)
        return samples

    def replay_ms(self, graph):
        samples = []
        for _ in range(self.args.iters):
            torch.cuda.synchronize(DEV)
            launched = now_ns()
            graph.replay()
            (end,) = observe([main_event()])
            samples.append((end - launched) / 1e6)
        return samples

    def run_long(self, name, idle_ms, long_ms):
        issue_after_ns = int(long_ms * 1e6 / 2)
        copy_ms, beyond_idle_ms, after_replay_end_ms, replay_ms_samples, exact = [], [], [], [], True
        for _ in range(self.args.iters):
            self.copy_destination.zero_()
            torch.cuda.synchronize(DEV)
            launched = now_ns()
            if name == "long_copy_before":
                issued, copy_event = self.queue_copy()
                self.long_graph.replay()
            elif name == "eager_copy_after":
                self.compute(self.args.long_matmuls)
                issued, copy_event = self.queue_copy()
            else:
                self.long_graph.replay()
                if name == "long_copy_after_half":
                    busy_wait(issue_after_ns)
                issued, copy_event = self.queue_copy()
            end_event = main_event()
            done, end = observe([copy_event, end_event])
            exact = exact and self.exact()
            copy_ms.append((done - issued) / 1e6)
            beyond_idle_ms.append((done - issued) / 1e6 - idle_ms)
            after_replay_end_ms.append((done - end) / 1e6)
            replay_ms_samples.append((end - launched) / 1e6)
        emit({"probe": name, "rows": self.args.rows, "idle_copy_ms_p50": round(idle_ms, 3),
              "compute_ms": summary(replay_ms_samples), "queue_to_copy_done_ms": summary(copy_ms),
              "hold_beyond_idle_ms": summary(beyond_idle_ms),
              "copy_done_minus_compute_end_ms": summary(after_replay_end_ms), "exact": exact}, self.out)

    def run_layers(self, name, idle_ms):
        copy_ms, minus_first_end, minus_last_launched_end, total_ms, exact = [], [], [], [], True
        for _ in range(self.args.iters):
            self.copy_destination.zero_()
            torch.cuda.synchronize(DEV)
            launched = now_ns()
            ends = []
            if name == "layers_copy_after_all":
                for graph in self.layer_graphs:
                    graph.replay()
                    ends.append(main_event())
                issued, copy_event = self.queue_copy()
                last_launched = len(self.layer_graphs) - 1
            else:
                self.layer_graphs[0].replay()
                ends.append(main_event())
                issued, copy_event = self.queue_copy()
                last_launched = 0
                if name == "layers_gate_after_first":
                    observe([copy_event])
                for graph in self.layer_graphs[1:]:
                    graph.replay()
                    ends.append(main_event())
            times = observe([copy_event, *ends])
            done, end_times = times[0], times[1:]
            exact = exact and self.exact()
            copy_ms.append((done - issued) / 1e6)
            minus_first_end.append((done - end_times[0]) / 1e6)
            minus_last_launched_end.append((done - end_times[last_launched]) / 1e6)
            total_ms.append((end_times[-1] - launched) / 1e6)
        first_layer_ms = self.replay_ms(self.layer_graphs[0])
        emit({"probe": name, "rows": self.args.rows, "layers": self.args.layers,
              "idle_copy_ms_p50": round(idle_ms, 3), "layer_replay_ms": summary(first_layer_ms),
              "all_layers_ms": summary(total_ms), "queue_to_copy_done_ms": summary(copy_ms),
              "copy_done_minus_layer0_end_ms": summary(minus_first_end),
              "copy_done_minus_last_launched_layer_end_ms": summary(minus_last_launched_end),
              "exact": exact}, self.out)

    def run_doorbell(self, case):
        from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
        from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

        name = case["name"]
        post_in_graph = case["post"] == "graph"
        waits_in_graph = case["waits"] == "graph"
        gated = case.get("gated", False)
        settings = {
            "head_store": case.get("head_store", "release"),
            "poll_mode": case.get("poll_mode", "acquire"),
            "prefer_overlap": case.get("prefer_overlap", True),
        }
        arm = {
            "copy_api": self.args.copy_api,
            "src_access_order": self.args.src_access_order,
            "torch_stream": self.args.torch_stream,
        }
        rows, layers = self.args.rows, self.args.layers
        source = self.source.view(SOURCE_ROWS, ROW_BYTES)
        destination = torch.zeros((DOORBELL_SLOTS, ROW_BYTES), dtype=torch.uint8, device=DEV)
        segments = expert_row_segments([(source, destination)])
        stream = torch.cuda.Stream(DEV) if self.args.torch_stream else None
        with ExpertDoorbellCopier(segments, rows, max_tags=layers + 1, cpu_core=self.args.spin_core,
                                  timeout_polls=self.args.timeout_polls, copy_api=arm["copy_api"],
                                  src_access_order=arm["src_access_order"], stream=stream, **settings) as copier:
            plans = []
            generator = torch.Generator().manual_seed(7)
            slot_of = torch.randperm(DOORBELL_SLOTS, generator=generator)[:SOURCE_ROWS]
            written = set()
            for _ in range(layers + 1):
                picked = torch.randperm(SOURCE_ROWS, generator=generator)[:rows]
                plans.append((picked.to(DEV), slot_of[picked].to(DEV, torch.int32),
                              torch.tensor([rows], dtype=torch.int32, device=DEV)))
            for plan_rows, _, _ in plans[1:]:
                written.update(plan_rows.tolist())
            expected = torch.zeros_like(destination)
            written_rows = torch.tensor(sorted(written))
            expected[slot_of[written_rows].to(DEV)] = source[written_rows].to(DEV)

            def post(index):
                copier.post(*plans[index + 1], tag=index + 1)

            def layer(index):
                if waits_in_graph and index > 0:
                    copier.wait(tag=index)
                self.compute(self.args.layer_matmuls)
                if post_in_graph:
                    post(index)

            try:
                graphs = [
                    self.capture(lambda index=index: layer(index), copier.quiesce, copier.resume)
                    for index in range(layers)
                ]
            except Exception as error:
                emit({"probe": name, "rows": rows, "layers": layers, **arm, "capture_error": repr(error)[:300],
                      "stats": copier.stats()}, self.out)
                return
            for tag in range(1, layers + 1):
                copier.wait(tag=tag)
            torch.cuda.synchronize(DEV)
            timeouts, elapsed, holds = [], [], []
            exact, drained = True, True
            start_stats = copier.stats()
            for _ in range(self.args.iters):
                drained = drain(copier) and drained
                destination.zero_()
                torch.cuda.synchronize(DEV)
                before = copier.stats()
                launched = now_ns()
                ends = []
                for index, graph in enumerate(graphs):
                    graph.replay()
                    ends.append(main_event())
                    if not post_in_graph:
                        post(index)
                    if gated:
                        seq = before["posted"] + index + 1
                        deadline = time.perf_counter() + 5.0
                        while time.perf_counter() < deadline:
                            trace = copier.trace()
                            if trace and trace[-1]["seq"] >= seq and trace[-1]["enqueued_ns"]:
                                break
                if waits_in_graph:
                    copier.wait(tag=layers)
                else:
                    for tag in range(1, layers + 1):
                        copier.wait(tag=tag)
                end_times = observe(ends)
                torch.cuda.synchronize(DEV)
                finished = now_ns()
                after = copier.stats()
                trace = {entry["seq"]: entry for entry in copier.trace()}
                for index in range(layers):
                    request = trace.get(before["posted"] + index + 1)
                    if request and request["complete_ns"]:
                        holds.append((request["complete_ns"] - end_times[index]) / 1e6)
                timeouts.append(after["timeouts"] - before["timeouts"])
                elapsed.append((finished - launched) / 1e6)
                drained = drain(copier) and drained
                exact = exact and torch.equal(destination, expected)
            destination[0, 0] ^= 0xFF
            flip_detected = not torch.equal(destination, expected)
            end_stats = copier.stats()
            emit({"probe": name, "rows": rows, "layers": layers, "post": case["post"], "waits": case["waits"],
                  "gated": gated, **settings, **arm, "timeouts_per_token": summary(timeouts),
                  "waits_per_token": layers, "token_ms": summary(elapsed),
                  "request_complete_minus_posting_layer_end_ms": summary(holds) if holds else None,
                  "exact": exact, "exact_check_detects_flip": flip_detected, "drained": drained,
                  "serviced": end_stats["serviced"] - start_stats["serviced"],
                  "copy_errors": end_stats["copy_errors"] - start_stats["copy_errors"],
                  "last_copy_error": end_stats["last_copy_error"],
                  "configured": {key: end_stats[key] for key in ("copy_api", "src_access_order", "external_stream")}},
                 self.out)

    def run(self):
        idle = self.idle_copy_ms()
        idle_ms = statistics.median(idle)
        emit({"probe": "idle_copy", "rows": self.args.rows, "copy_ms": summary(idle), "exact": self.exact()}, self.out)
        if not self.args.doorbell_only:
            long_ms = statistics.median(self.replay_ms(self.long_graph))
            for name in ("long_copy_before", "long_copy_after", "long_copy_after_half", "eager_copy_after"):
                self.run_long(name, idle_ms, long_ms)
            for name in ("layers_copy_after_first", "layers_copy_after_all", "layers_gate_after_first"):
                self.run_layers(name, idle_ms)
        if not self.args.skip_doorbell:
            for case in DOORBELL_CASES:
                if self.args.cases and case["name"] not in self.args.cases:
                    continue
                self.run_doorbell(case)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--long-matmuls", type=int, default=16)
    parser.add_argument("--layer-matmuls", type=int, default=3)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--spin-core", type=int, default=71)
    parser.add_argument("--timeout-polls", type=int, default=20_000)
    parser.add_argument("--skip-doorbell", action="store_true")
    parser.add_argument("--doorbell-only", action="store_true")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--copy-api", choices=["batch", "per_segment"], default="batch")
    parser.add_argument("--src-access-order", choices=["stream", "during_call", "any"], default="stream")
    parser.add_argument("--torch-stream", action="store_true")
    parser.add_argument("--out", default="probe_hold.jsonl")
    Probe(parser.parse_args()).run()


if __name__ == "__main__":
    main()
