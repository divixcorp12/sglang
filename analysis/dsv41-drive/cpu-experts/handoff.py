"""Per-layer GPU->CPU->GPU handoff latency, 40-layer CUDA graph, optional EXL3 CPU expert compute (see handoff_ext.cu).

usage: handoff.py build | handoff.py run WORKER_CPU THREADS
"""
import json, os, statistics, sys

import bench  # weights(), emit(), EXL, HERE

HERE, EXL = bench.HERE, bench.EXL
LAYERS = 40


def ext():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return load(
        name="exl3handoff",
        sources=[os.path.join(EXL, "cpu", "moe_mul1.cpp"), os.path.join(HERE, "handoff_ext.cu")],
        extra_include_paths=[EXL],
        extra_cflags=["-O3", "-Ofast"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=os.path.join(HERE, "build_handoff"),
        verbose=False,
    )


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run(worker_cpu, threads):
    if worker_cpu >= 0:
        # The exllamav3 pool is spawned from the worker and inherits its affinity: 12 spinning threads on one core
        # livelock (divix01, 2026-09-29). Leave the worker floating inside the process's taskset.
        raise SystemExit("WORKER_CPU must be -1")
    import torch
    torch.set_num_threads(min(16, len(os.sched_getaffinity(0))))
    e = ext()
    e.init_shm()
    n_experts = 384
    g = torch.Generator().manual_seed(1)
    w = bench.weights(n_experts, g)
    handle = bench.make(e, w, 1)
    x = torch.randn(5120, device="cuda").half()
    out = torch.zeros(5120, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    graphs = {}
    for mode in (0, 1, 2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for l in range(LAYERS):
                e.layer(x, out, l, mode)
        graphs[mode] = graph
    torch.cuda.synchronize()

    def replay(mode, n):
        times = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graphs[mode].replay(); b.record()
            times.append((a, b))
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a, b in times]

    replay(0, 20)
    base = replay(0, 200)
    base_ms = statistics.median(base)
    bench.emit({"mode": "handoff", "config": "baseline (40 consume kernels, no handoff)", "graph_ms_p50": base_ms})
    names = {1: "kernel", 2: "memop"}
    for mode in (1, 2):
        for topk in (0, 1, 2, 4):
            e.start_worker(handle if topk else -1, topk, threads, worker_cpu, n_experts)
            try:
                replay(mode, 10)
                ms = replay(mode, 100 if topk else 300)
            finally:
                work_us = e.stop_worker()
            work_us = work_us[10 * LAYERS:]  # drop warm-up layers
            per_layer_us = [(t - base_ms) * 1e3 / LAYERS for t in ms]
            rec = {"mode": "handoff", "config": f"{names[mode]}/topk{topk}", "threads": threads,
                   "worker_cpu": worker_cpu, "aborted": int(e.aborted()),
                   "graph_ms_p10": pct(ms, 0.1), "graph_ms_p50": statistics.median(ms), "graph_ms_p90": pct(ms, 0.9),
                   "per_layer_us_p50": statistics.median(per_layer_us), "per_layer_us_p90": pct(per_layer_us, 0.9),
                   "cpu_work_us_p50": statistics.median(work_us), "cpu_work_us_p90": pct(work_us, 0.9),
                   "overhead_us_p50": statistics.median(per_layer_us) - statistics.median(work_us),
                   "out_finite": bool(torch.isfinite(out).all())}
            bench.emit(rec)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        ext(); print("built")
    else:
        run(int(sys.argv[2]), int(sys.argv[3]))
