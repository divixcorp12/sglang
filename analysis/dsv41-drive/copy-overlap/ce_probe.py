#!/usr/bin/env python3
"""Probe (ii): can a host service thread drive copy-engine copies that a replaying decode graph waits on?

One captured graph mimics a decode step: LAYERS x (FILLER tiny kernels, post, wait). The post kernel publishes a
sequence number in mapped host memory; a C++ service thread polls it, issues cudaMemcpyAsync for ROWS pinned rows
(six real EXL3 segments each) on its own stream, then a 4-byte copy of the sequence into the device done word. The
wait kernel spins on that word (5 s timeout latches `fail` instead of hanging). The main thread replays the graph
back to back without syncing, so it blocks in cudaGraphLaunch the way decode does.

Reports: stalls (timeouts), the host time blocked in each replay, and per request the device post -> done time
against the copy-engine time for the same bytes, and the host poll/issue latency.

    PYTHONPATH=<repo>/python python ce_probe.py --repo <repo> --out probe.json [--rows 2]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SEGMENTS = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)
ROW_BYTES = sum(SEGMENTS)
SRC_ROWS = 96
LAYERS = 40
FILLER = 140


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--ahead", type=int, default=0, help="0: replay back to back; k: wait for replay i-k after launching replay i")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--host-nodes", action="store_true", help="add host nodes at layers 1 and 14, as the Engram callbacks")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.jit.utils.compile.loader import load_jit
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    mod = load_jit(
        "ce_probe_copy_overlap",
        cuda_files=[str(Path(__file__).resolve().parent / "ce_probe.cuh")],
        cuda_wrappers=[(n, n) for n in ("ce_probe_start", "ce_probe_stop", "ce_probe_post", "ce_probe_wait", "ce_probe_filler", "ce_probe_host_node")],
    )
    dev = torch.device("cuda")
    src = [allocate_host_slab(SRC_ROWS, (b,), torch.uint8, register=True, placement=((0, SRC_ROWS * b),)) for b in SEGMENTS]
    for s in src:
        s.fill_(3)
    dst = [torch.zeros((max(a.rows, 1), b), dtype=torch.uint8, device=dev) for b in SEGMENTS]
    post_word = torch.zeros(1, dtype=torch.int32).pin_memory()
    gen_ring = torch.zeros(1024, dtype=torch.int32).pin_memory()
    done_word = torch.zeros(1, dtype=torch.int32, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    fail = torch.zeros(1, dtype=torch.int32, device=dev)
    cap = LAYERS * (a.replays + 8)
    stamps = torch.zeros(2 * cap, dtype=torch.int64, device=dev)
    x = torch.ones(32, dtype=torch.float32, device=dev)

    # Copy-engine time for one request's bytes, alone: the yardstick for the added latency.
    ce = []
    for _ in range(12):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for r in range(a.rows):
            for s, d in zip(src, dst):
                d[r].copy_(s[r], non_blocking=True)
        e1.record()
        torch.cuda.synchronize()
        ce.append(e0.elapsed_time(e1) * 1e3)
    ce_us = statistics.median(ce[2:])

    ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64)
    mod.ce_probe_start(post_word, done_word, gen_ring, ptr(src), ptr(dst), torch.tensor(SEGMENTS, dtype=torch.int64),
                       SRC_ROWS, a.rows, max(a.rows, 1), dev.index or 0)

    def step():
        for layer in range(LAYERS):
            if a.host_nodes and layer in (1, 14):
                mod.ce_probe_host_node(x)
            for _ in range(FILLER):
                mod.ce_probe_filler(x)
            mod.ce_probe_post(post_word, counter, stamps)
            mod.ce_probe_wait(done_word, counter, stamps, fail, a.timeout_ms * 1_000_000)

    step()  # warm the path eagerly, with the service live
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            step()
    torch.cuda.synchronize()
    stamps.zero_()
    base_seq = int(counter.item())
    launch_ms = []
    t0 = time.perf_counter()
    pending = []
    for _ in range(a.replays):
        t = time.perf_counter()
        graph.replay()
        launch_ms.append((time.perf_counter() - t) * 1e3)
        if a.ahead:
            pending.append(torch.cuda.Event())
            pending[-1].record(torch.cuda.current_stream())
            if len(pending) > a.ahead:
                pending.pop(0).synchronize()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    out = torch.zeros(2 + 3 * 20000, dtype=torch.int64)
    n = mod.ce_probe_stop(out)
    served, error = int(out[0]), int(out[1])
    host = out[2 : 2 + 3 * n].view(n, 3).tolist()
    st = stamps.view(-1, 2).cpu().tolist()
    seqs = range(base_seq + 1, base_seq + a.replays * LAYERS + 1)
    dev_us = [(st[(q - 1) % cap][1] - st[(q - 1) % cap][0]) / 1e3 for q in seqs]
    # host records are in service order: the last replays*LAYERS belong to the replays
    rec_host = host[-a.replays * LAYERS :]
    poll_us = [(h[0] - st[(q - 1) % cap][0]) / 1e3 for h, q in zip(rec_host, seqs)]
    issue_us = [(h[1] - h[0]) / 1e3 for h in rec_host]
    api_us = [h[2] / 1e3 for h in rec_host]
    res = {
        "rows_per_request": a.rows,
        "host_nodes": a.host_nodes,
        "ahead": a.ahead,
        "requests": len(dev_us),
        "served_total": served,
        "service_error": error,
        "timeouts": int(fail.item()),
        "ce_alone_us": ce_us,
        "post_to_done_us": {"p50": statistics.median(dev_us), "p90": pct(dev_us, 0.9), "max": max(dev_us)},
        "added_us_p50": statistics.median(dev_us) - ce_us,
        "post_to_host_seen_us": {"p50": statistics.median(poll_us), "p90": pct(poll_us, 0.9), "max": max(poll_us)},
        "host_seen_to_issued_us": {"p50": statistics.median(issue_us), "p90": pct(issue_us, 0.9), "max": max(issue_us)},
        "api_us_per_request_p50": statistics.median(api_us),
        "replay_call_ms": {"p50": statistics.median(launch_ms), "max": max(launch_ms), "sum": sum(launch_ms)},
        "wall_ms": wall_ms,
        "step_ms": wall_ms / a.replays,
    }
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if res["timeouts"] == 0 and error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
