#!/usr/bin/env python3
"""The harness G_MEASUREMENT_PREREG.md section 4 asks for and never got: the per-stage cost g of a stage triple.

Output is the JSONL that the frozen g_measurement/g_analysis.py reads (schema in its docstring), one line per
(process, variant, N) cell. Judge it with `python g_measurement/g_analysis.py results.jsonl`; `summary` below prints the
slopes even when a gate fails, for a PRELIMINARY look, and says so.

A stage triple is W_s, C_s, A_s, three dependent kernel nodes in one captured graph, N of them in series:
  C_s  the PRODUCTION copy_expert_row_segments_gpu_kernel (grid 8 x 256), fed a one-segment table of a 4 KiB row and a
       device count word: 0 for an empty triple, 1 for an active one. It moves nothing significant, so L2 does not enter g.
  W_s, A_s  stand-ins (g_kernels.cu). Empty: W_s takes the empty-range exit (go = 0, no host word read), A_s acknowledges
       nothing. Active(p): W_s does p serial system-scope acquire loads of a mapped host page (registered p = 4), A_s does a
       __threadfence_system() and a release store to a distinct mapped line.
Variants: empty | active_p1 | active_p4 | active_p6 | control20 (empty plus a 20 us spin in W_s: the positive control, the
slope must rise by 20 +- 1 us) | empty_base8k (8,000 filler nodes underneath, then the empty triples).
Sweep N in {0,40,80,160,320,640} (base8k: {0,160,640}); R = 50 replays per event-timed batch, 20 batches per cell; cell
order randomised within a process; one process per invocation, run 3 (run_g.sh does).

Empty and active are measured separately on purpose: the plan's 160 extra triples are 85.10 empty tail stages plus
74.5 active ones, and an empty triple is three launches and no host read.

    gpu-run.sh python g_harness.py run --repo <tree> --out DIR --process 0     (one process; repeat for 1, 2)
    python g_harness.py summary DIR/results.jsonl                              (CPU, preliminary look)
    python g_harness.py build                                                  (CPU: compile g_kernels.so, no GPU)

The interpreter trap: PYTHONPATH must be <repo>/python; this refuses if sglang resolves anywhere else.
"""
import argparse, ctypes, hashlib, json, os, random, statistics as S, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c_measurement"))
sys.path.insert(0, str(HERE.parent / "g_measurement"))

VARIANTS = ("empty", "active_p1", "active_p4", "active_p6", "control20", "empty_base8k")
NS_FULL = (0, 40, 80, 160, 320, 640)
NS_BASE = (0, 160, 640)
P_OF = {"empty": 0, "control20": 0, "empty_base8k": 0, "active_p1": 1, "active_p4": 4, "active_p6": 6}
BASE_NODES = 8000
ACK_LINES = 1024
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.2/bin/nvcc")
SEED = 20260921


def build_lib(build_dir):
    src = HERE / "g_kernels.cu"
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    so = Path(build_dir) / ("g_kernels_%s.so" % tag)
    if not so.exists():
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        subprocess.check_call([NVCC, "-O2", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_120", "-o", str(so), str(src)])
    return so


def node_count(g):
    from cuda.bindings import runtime as cudart
    graph = cudart.cudaGraph_t(g.raw_cuda_graph())
    err, _, n = cudart.cudaGraphGetNodes(graph, None)
    if err != cudart.cudaError_t.cudaSuccess: raise RuntimeError("cudaGraphGetNodes: %s" % err)
    return int(n)


def run(a):
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo / "python"))
    import torch
    import sglang
    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit("INTERPRETER TRAP: sglang imported from %s, not under %s; set PYTHONPATH=%s/python" % (sglang.__file__, repo, repo))
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu, expert_row_segments
    from sglang.kernels.ops.moe.exl3_ram_miss import new_page
    import c_harness as H

    lib = ctypes.CDLL(str(build_lib(a.build_dir)))
    for name, argt in (("launch_w", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]),
                       ("launch_a", [ctypes.c_void_p] * 3), ("launch_nop", [ctypes.c_void_p])):
        getattr(lib, name).argtypes = argt; getattr(lib, name).restype = ctypes.c_int

    torch.cuda.init(); dev = torch.device("cuda", torch.cuda.current_device()); torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    stream = torch.cuda.Stream(device=dev)
    page = new_page(pin=True)                                   # the request page, as the service allocates it
    words = page[:64 * 8].view(torch.int32)
    words[::16] = 1                                             # 8 poll lines, all "ready"
    acks = torch.zeros(ACK_LINES * 64, dtype=torch.uint8, pin_memory=True)   # a distinct mapped line per ack
    src = torch.zeros((4, 4096), dtype=torch.uint8).pin_memory()
    dst = torch.zeros((4, 4096), dtype=torch.uint8, device=dev)
    seg = expert_row_segments([(src, dst)])                     # one segment, one 4 KiB row
    rows = torch.zeros(1, dtype=torch.int64, device=dev); slots = torch.zeros(1, dtype=torch.int32, device=dev)
    go = torch.zeros(1, dtype=torch.int32, device=dev)

    def chain(variant, N):
        p = P_OF[variant]; spin = 20_000 if variant == "control20" else 0
        s = stream.cuda_stream
        def chk(rc):
            if rc != 0: raise RuntimeError("kernel launch failed: cudaError %d" % rc)
        if variant == "empty_base8k":
            for _ in range(BASE_NODES): chk(lib.launch_nop(s))
        for i in range(N):
            chk(lib.launch_w(s, page.data_ptr(), p, go.data_ptr(), spin))
            copy_expert_row_segments_gpu(seg, rows, slots, go)
            chk(lib.launch_a(s, go.data_ptr(), acks.data_ptr() + 64 * (i % ACK_LINES)))

    def capture(variant, N):
        with torch.cuda.stream(stream):
            chain(variant, min(N, 2))                           # eager warm-up: JIT, first-touch, context
        stream.synchronize()
        g = torch.cuda.CUDAGraph(keep_graph=True)              # keep_graph: the raw graph is needed to count nodes
        with torch.cuda.graph(g, stream=stream):
            chain(variant, N)
        nodes = node_count(g); g.instantiate()
        return g, nodes

    meta = {"script": "g_harness.py", "process": a.process, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": str(repo),
            "repo_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "g_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "g_kernels_sha256": hashlib.sha256((HERE / "g_kernels.cu").read_bytes()).hexdigest(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(dev),
            "sglang_file": sglang.__file__, "R": a.R, "batches": a.batches, "keepalive": a.keepalive,
            "page_numa_note": "pinned by torch (cudaHostAlloc); NUMA node not bound, see numa_maps below",
            "box_start": {"loadavg": open("/proc/loadavg").read().split()[:3]}}
    try:
        meta["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        meta["apps_start"] = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception as e: meta["nvidia_smi_error"] = str(e)   # noqa: E701
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    variants = [v for v in VARIANTS if not a.variants or v in a.variants.split(",")]
    ns_full = tuple(int(x) for x in a.ns.split(",")) if a.ns else NS_FULL
    cells = [(v, N) for v in variants for N in (tuple(n for n in ns_full if n in NS_BASE) if v == "empty_base8k" else ns_full)]
    random.Random(SEED + a.process).shuffle(cells)

    # graph correctness first: one active graph must actually poll, ack and copy (a wrong-but-fast graph is the failure to fear)
    go.zero_(); acks.zero_()
    gtest, nn = capture("active_p4", 3)
    with torch.cuda.stream(stream): gtest.replay()
    stream.synchronize()
    assert nn == 9, "active_p4 N=3 has %d nodes, expected 9" % nn
    assert int(go.item()) == 1 and int(acks[:64 * 3].view(torch.int32)[::16].sum()) == 3, "the active chain did not poll and acknowledge"
    gtest2, nn2 = capture("empty", 3)
    acks.zero_()
    with torch.cuda.stream(stream): gtest2.replay()
    stream.synchronize()
    assert nn2 == 9 and int(go.item()) == 0 and int(acks.sum()) == 0, "the empty chain touched host memory or set go"
    meta["self_check"] = "active_p4 N=3: 9 nodes, go=1, 3 acks written; empty N=3: 9 nodes, go=0, no ack"
    del gtest, gtest2

    smi = H.SmiSampler(os.getpid()); smi.start(); time.sleep(2.0)
    stop = threading.Event()
    if a.keepalive:                                             # optional: small H2D copies keep the link out of Gen1 during empty cells
        side = torch.cuda.Stream(device=dev); bsrc = torch.zeros(1 << 16, dtype=torch.uint8).pin_memory(); bdst = torch.zeros(1 << 16, dtype=torch.uint8, device=dev)
        def ka():
            while not stop.is_set():
                with torch.cuda.stream(side): bdst.copy_(bsrc, non_blocking=True)
                time.sleep(0.002)
        threading.Thread(target=ka, daemon=True).start()
    # warm-up: 2 s of load on the stream so the link and clocks are up
    gw, _ = capture("active_p4", 160)
    t_end = time.monotonic() + 2.0
    with torch.cuda.stream(stream):
        while time.monotonic() < t_end: gw.replay(); stream.synchronize()
    del gw

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    resf = open(out / "results.jsonl", "a")
    for ci, (variant, N) in enumerate(cells):
        g, nodes = capture(variant, N) if N > 0 else capture_empty(torch, stream)
        with torch.cuda.stream(stream):
            for _ in range(3): g.replay()
        stream.synchronize()
        batches, host_launch = [], []
        t0 = time.monotonic()
        with torch.cuda.stream(stream):
            for _ in range(a.batches):
                e0.record(stream)
                h0 = time.perf_counter()
                for _ in range(a.R): g.replay()
                host_launch.append((time.perf_counter() - h0) * 1e3 / a.R)
                e1.record(stream); e1.synchronize()
                batches.append(e0.elapsed_time(e1))
        t1 = time.monotonic()
        cond = smi.window(t0, t1) or {"link_gen_start": None, "link_gen_end": None, "pstate_start": None, "sm_mhz_min": 0, "sm_mhz_max": 0, "other_gpu_procs": None}
        rec = {"process": a.process, "variant": variant, "N": N, "R": a.R, "batch_ms": batches, "nodes": nodes,
               "link_gen_start": cond["link_gen_start"], "link_gen_end": cond["link_gen_end"], "sm_mhz_min": cond["sm_mhz_min"],
               "sm_mhz_max": cond["sm_mhz_max"], "other_gpu_procs": cond["other_gpu_procs"],
               "pstate_start": cond["pstate_start"], "host_launch_ms_per_replay": host_launch, "wall_s": t1 - t0,
               "loadavg1": float(open("/proc/loadavg").read().split()[0])}
        resf.write(json.dumps(rec) + "\n"); resf.flush()
        print("cell %3d/%d %-13s N=%3d nodes=%5d  per-replay p50 %.4f ms  link %s/%s P%s SM %s-%s MHz apps=%s" % (
            ci + 1, len(cells), variant, N, nodes, S.median(batches) / a.R, cond["link_gen_start"], cond["link_gen_end"], cond["pstate_start"],
            cond["sm_mhz_min"], cond["sm_mhz_max"], cond["other_gpu_procs"]), flush=True)
        del g
    stop.set(); smi.stop()
    meta["box_end"] = {"loadavg": open("/proc/loadavg").read().split()[:3]}; meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / ("meta_process%d.json" % a.process)).write_text(json.dumps(meta, indent=1, default=str))
    return 0


def capture_empty(torch, stream):
    """N = 0: a graph with no nodes (what replay costs by itself)."""
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=stream):
        pass
    nodes = node_count(g); g.instantiate()
    return g, nodes


def summary(path):
    import g_analysis as GA
    lines = [json.loads(l) for l in open(path)]
    r = GA.analyse(lines)
    print("PRELIMINARY LOOK. The frozen verdict below is meaningful only if it is not INVALID and the run was the full registered sweep.")
    print("frozen g_analysis verdict:", r["verdict"])
    for gate in r["gates"]: print("  gate:", gate)
    print("slopes per process (us per triple, OLS on per-N medians, [bootstrap 95%], linearity diff N<=160 vs N>=160):")
    for v, procs in r["slopes"].items():
        for p, x in sorted(procs.items()):
            print("  %-13s process %s: %7.2f [%7.2f, %7.2f]   lin %s" % (v, p, x[0], x[1], x[2], "n/a" if x[3] is None else "%.3f" % x[3]))
    med = {v: S.median(x[0] for x in procs.values()) for v, procs in r["slopes"].items()}
    print("median across processes:", {k: round(v, 2) for k, v in med.items()})
    if "empty" in med and "active_p4" in med:
        ge, ga = med["empty"], med["active_p4"]
        GX = (GA.E_EMPTY * ge + (GA.E_B + GA.E_C + GA.E_D) * ga) / 1000; GH = (GA.E_EMPTY * ge + (GA.E_B + GA.E_D) * ga) / 1000
        print("g_e = %.2f us, g_a(p=4) = %.2f us  ->  G_X (all exposed) = %.3f ms/step, G_H (class C hidden) = %.3f ms/step   vs G* = %.3f ms" % (ge, ga, GX, GH, GA.G_STAR_MS))
        print("  = 85.10 empty x g_e + (48.93 + 4.65 [+ 20.86 in G_X]) active x g_a; crossings: uniform g* = 6.98 (all exposed) / 8.03 us (C hidden)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--repo", required=True); r.add_argument("--out", required=True)
    r.add_argument("--process", type=int, default=0); r.add_argument("--R", type=int, default=50); r.add_argument("--batches", type=int, default=20)
    r.add_argument("--variants", default="", help="comma list (default: all six)"); r.add_argument("--ns", default="", help="comma list (default: 0,40,80,160,320,640)")
    r.add_argument("--keepalive", action="store_true", help="tiny H2D copies during cells so the link stays out of Gen1 (recorded; changes what the empty cells run beside)")
    r.add_argument("--build-dir", default=str(Path.home() / ".cache" / "g_kernels"))
    b = sub.add_parser("build"); b.add_argument("--build-dir", default=str(Path.home() / ".cache" / "g_kernels"))
    s = sub.add_parser("summary"); s.add_argument("results")
    ns = p.parse_args()
    if ns.cmd == "build": print(build_lib(ns.build_dir)); return 0
    return run(ns) if ns.cmd == "run" else summary(ns.results)


if __name__ == "__main__":
    sys.exit(main())
