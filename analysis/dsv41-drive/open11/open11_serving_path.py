"""OPEN 11, re-taken through the real serving backend (LEASE_PROTOCOL.md 20.1 step 5).

`open11_arming_cost.py` measured the kernels plus the real C++ service through a hand-written step. This
measures the same all-hit cost through `Exl3RamMissService` and `Exl3RamMissRowBackend` driven by the real switch,
`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES`. One process per switch value (the service is a singleton and reads the switch
once). Every planned expert is resident in the pinned tier, so the request is all-hit; `rows_read` during timing must
be 0 and is reported.

Four views. eager is the old harness's method (one post, synchronize, wall clock) through the real backend; the
other three are CUDA graph replays (serving replays a graph, unlike the eager 8 us figure):
  layer  : one `Exl3MoEMethod._apply_graph` (the whole in-graph MoE layer) per replay
  chain  : 40 backend.post calls in one graph, each followed by a GPU spin of SPACING_MS (a stand-in for a layer's
           compute, so a request's acknowledgement has time to retire before the ring slot is reused)
  packed : the same 40 posts back to back (spacing 0): the worst case, where the service's retirement of a lane
           can be waited on by the next request that reuses its ring slot

Run under gpu-run.sh, taskset -c 0-63, OMP_NUM_THREADS=1, SGLANG_EXL3_SRC set, PYTHONPATH at the tree under test.
    python open11_serving_path.py --lease 0|1 [--reps 300]
"""
import argparse
import json
import statistics as S
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "test" / "manual" / "dsv41"))
import test_exl3_ram_miss_graph_gpu as harness  # noqa: E402

ROUTE = [3, 5, 7, 4, 0, 1]  # 0 and 1 are VRAM-hot; 3, 5, 7, 4 are four RAM lanes
LAYERS_IN_CHAIN = 40


def pct(xs, q):
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def timed(graph, reps, warmup):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50": S.median(ts), "p90": pct(ts, 0.9), "min": min(ts)}


def spin_cycles_per_ms():
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 20
    return int(1_000_000 * 1e-3 / per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lease", type=int, required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--spacing-ms", type=float, default=1.5)
    args = ap.parse_args()
    torch.set_num_threads(1)

    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    with tempfile.TemporaryDirectory() as tmp:
        layer, streamer, service, checks = harness._layers(Path(tmp), lease=bool(args.lease))
        out = {"lease": bool(args.lease), "service_lease_mode": service.lease_mode, "route": ROUTE}
        try:
            x = torch.zeros((1, harness.HIDDEN), device="cuda", dtype=torch.bfloat16)
            weights = torch.full((1, harness.TOP_K), 1.0 / harness.TOP_K, device="cuda")
            ids = torch.tensor([ROUTE], device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
            for _ in range(3):  # prime: the first replay reads the misses into RAM
                graph.replay()
                torch.cuda.synchronize()
                time.sleep(0.05)
            rows_before = service.host.counters()["rows_read"]
            out["layer"] = timed(graph, args.reps, 30)
            assert streamer.row_backend.keep.item() == 1.0
            out["layer"]["rows_read_during_timing"] = service.host.counters()["rows_read"] - rows_before

            # chain and packed: backend.post directly, static plan of the four RAM lanes
            backend = streamer.row_backend
            lanes = [3, 5, 7, 4]
            cap = backend.host_rows.numel()
            expert_ids = torch.zeros(cap, dtype=torch.int64, device="cuda")
            expert_ids[: len(lanes)] = torch.tensor(lanes)
            plan = ExpertRowPlan(
                expert_ids=expert_ids,
                slots=torch.arange(cap, dtype=torch.int32, device="cuda"),
                count=torch.tensor([len(lanes)], dtype=torch.int32, device="cuda"),
            )
            backend.routes.fill_(-1)
            backend.routes[: len(lanes)] = expert_ids[: len(lanes)]
            tag = next(iter(backend.segments))
            # eager: the old harness's method (one post, then a synchronize, wall clock) but through the real backend
            for _ in range(30):
                backend.post(tag, plan)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                backend.post(tag, plan)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            out["eager"] = {"p50": S.median(ts), "p90": pct(ts, 0.9), "min": min(ts)}
            cycles = int(spin_cycles_per_ms() * args.spacing_ms)
            for name, spacing in (("chain", cycles), ("packed", 0)):
                backend.post(tag, plan)  # warm
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(LAYERS_IN_CHAIN):
                        backend.post(tag, plan)
                        if spacing:
                            torch.cuda._sleep(spacing)
                before = service.host.counters()["rows_read"]
                out[name] = timed(g, max(50, args.reps // 3), 5)
                out[name]["rows_read_during_timing"] = service.host.counters()["rows_read"] - before
                out[name]["kept"] = backend.keep.item()
            out["spacing_ms"] = args.spacing_ms
            counters = service.host.counters()
            out["counters"] = {k: counters[k] for k in ("leases_granted", "leases_acked", "leases_voided", "lease_double_signal", "deferred", "deferred_reuse", "touch_only", "served")}
            out["fatal"] = service.host.fatal_seq()
        finally:
            service.shutdown()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
