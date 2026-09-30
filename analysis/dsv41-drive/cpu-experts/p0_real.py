"""P0 of docs/superpowers/plans/2026-09-29-dsv41-cpu-experts.md: exllamav3's CPU MoE kernel on real DSV4.1 rows.

The kernel is the fork's own exl3_ext() build (exllamav3 02aef45, cpu/moe_mul1.cpp), called on rows laid out as
the pinned tier's slabs (ExpertPinnedHostCache.tensors: w13_* [N, 2, ...] with gate at part 0 and up at part 1,
w2_* [N, ...]), with activation 0 and act_limit = swiglu_limit, which is what the graph decode path passes
(exl3_fused_moe.py, ACT_SILU).

Run from the repo root with PYTHONPATH=$PWD/python. Modes:
  accuracy LAYERS PER_LAYER TAG
      The CPU kernel (bw and scalar ISA tiers) and the GPU exl3_linear path, each against the fp32 reconstruct
      reference (exl3_linear_reference), on PER_LAYER real experts of each of LAYERS (comma list). Needs the GPU.
  perf LAYER N THREADS TOPKS LAYOUT TAG
      Cold throughput over N real experts of LAYER held as tier slabs. Calls rotate so no expert repeats within
      N/topk calls. LAYOUT is native (the tier's bytes as they are) or swizzled (a band-contiguous repacked copy).
  _cpu TIER CASES OUT
      Internal: one ISA tier's CPU outputs for a cases file, in a fresh process (the tier is fixed at first use).

Results append to p0_results.jsonl beside this file.
"""
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")  # the pool otherwise pins to the first cores and ignores taskset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

for key in ("SGLANG_EXL3_SRC", "SGLANG_EXL3_BUILD_DIR"):
    os.environ.setdefault(key, arm_env.base_env()[key])

H, I, KB = 5120, 2304, 3
SWIGLU_LIMIT = 10.0
MUL1 = -2082680531  # 0x83DCD12D as int32, the codebook constant the CPU kernel hard-codes
RESULTS = os.path.join(HERE, "p0_results.jsonl")


def flavor():
    """The CPU kernel build this process uses: "" is upstream's, else exl3_ext.build_flavor's suffix."""
    from sglang.srt.layers.quantization.exl3_ext import build_flavor, cpu_act_defines

    return build_flavor(cpu_act_defines()) or "upstream"


def emit(rec):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "flavor": flavor(), **rec}
    line = json.dumps(rec)
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def ext():
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    return exl3_ext()


def check_affinity(threads):
    cores = len(os.sched_getaffinity(0))
    if cores < 2:
        # The pool's workers inherit the affinity: many spinning threads on one core livelock (DSV41_REFERENCE §28).
        raise SystemExit(f"refusing to run the CPU pool on {cores} core")
    if threads > cores:
        raise SystemExit(f"{threads} threads on {cores} cores")


def load_slabs(layer, experts):
    """Real rows of ``experts`` in ``layer``, as the pinned tier's six slab tensors."""
    import torch
    from safetensors import safe_open

    root = arm_env.EXPERT_DIR
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    n = len(experts)
    i16, f16 = torch.int16, torch.float16
    slabs = {
        "w13_trellis": torch.empty((n, 2, H // 16, I // 16, 16 * KB), dtype=i16),
        "w13_suh": torch.empty((n, 2, H), dtype=f16),
        "w13_svh": torch.empty((n, 2, I), dtype=f16),
        "w2_trellis": torch.empty((n, I // 16, H // 16, 16 * KB), dtype=i16),
        "w2_suh": torch.empty((n, I), dtype=f16),
        "w2_svh": torch.empty((n, H), dtype=f16),
    }
    parts = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", None)}
    files = {}
    for i, e in enumerate(experts):
        for w, (prefix, part) in parts.items():
            base = f"layers.{layer}.ffn.experts.{e}.{w}"
            f = files.setdefault(index[base + ".trellis"], safe_open(os.path.join(root, index[base + ".trellis"]), "pt"))
            mul1 = int(f.get_tensor(base + ".mul1"))
            if mul1 != MUL1:
                raise SystemExit(f"{base}.mul1 = {mul1}, the CPU kernel only decodes {MUL1}")
            for kind in ("trellis", "suh", "svh"):
                slab = slabs[f"{prefix}_{kind}"][i]
                (slab if part is None else slab[part]).copy_(f.get_tensor(f"{base}.{kind}"))
    return slabs


def swizzle(t):
    """exllamav3's band-contiguous trellis layout (moe_mul1.h, MoeCpuMatrix.swz)."""
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)


def make_layer(e, slabs, swizzled):
    """Register a slab set with the CPU kernel: per-expert views, gate = w13 part 0, up = w13 part 1."""
    n = slabs["w2_trellis"].shape[0]
    tr = (lambda t: swizzle(t)) if swizzled else (lambda t: t)
    lists = [
        [tr(slabs["w13_trellis"][i, 0]) for i in range(n)],
        [slabs["w13_suh"][i, 0] for i in range(n)],
        [slabs["w13_svh"][i, 0] for i in range(n)],
        [tr(slabs["w13_trellis"][i, 1]) for i in range(n)],
        [slabs["w13_suh"][i, 1] for i in range(n)],
        [slabs["w13_svh"][i, 1] for i in range(n)],
        [tr(slabs["w2_trellis"][i]) for i in range(n)],
        [slabs["w2_suh"][i] for i in range(n)],
        [slabs["w2_svh"][i] for i in range(n)],
    ]
    return e.exl3_moe_cpu_make_layer(*lists, [], [], [], 0, SWIGLU_LIMIT, 1 if swizzled else 0)


def make_cases(layers, per_layer, seed=20260929):
    """Per layer: PER_LAYER random experts; per scale, 8 single-expert calls and 8 top-6 calls."""
    import torch

    g = torch.Generator().manual_seed(seed)
    cases = []
    for layer in layers:
        experts = sorted(torch.randperm(384, generator=g)[:per_layer].tolist())
        for scale in (1.0, 8.0):  # 8x drives |gate|, |up| past the limit, so the clamp is exercised
            for topk in (1, 6):
                for _ in range(8):
                    x = (torch.randn(1, H, generator=g) * scale).half()
                    local = torch.randperm(per_layer, generator=g)[:topk]
                    w = torch.rand(topk, generator=g) + 0.1
                    cases.append({"layer": layer, "experts": experts, "scale": scale, "topk": topk, "x": x,
                                  "sel": local.view(1, topk).long(), "w": (w / w.sum()).view(1, topk).half()})
    return cases


def cpu_outputs(tier, cases_path, out_path):
    import torch

    os.environ["EXL3_MOE_CPU_MAX_ISA"] = tier
    e = ext()
    cases = torch.load(cases_path)
    threads = min(8, len(os.sched_getaffinity(0)))
    check_affinity(threads)
    outs, handles = [], {}
    for c in cases:
        if c["layer"] not in handles:
            slabs = load_slabs(c["layer"], c["experts"])
            handles[c["layer"]] = (make_layer(e, slabs, False), slabs)
        out = torch.zeros(1, H)
        e.exl3_moe_cpu_forward(handles[c["layer"]][0], c["x"], c["sel"], c["w"], out, threads)
        outs.append(out)
    torch.save({"outs": outs, "bw": bool(e.exl3_moe_cpu_has_avx512_bw())}, out_path)


def gpu_outputs(cases):
    """Per case: (production exl3_linear path, fp32 reconstruct reference), both fp32 [1, H]."""
    import torch
    from sglang.srt.layers.quantization.exl3_ops import (
        Exl3Tensors,
        exl3_linear,
        exl3_linear_reference,
        exl3_moe_accumulate,
    )

    dev = torch.device("cuda")
    gpu, ref, cached = [], [], {}
    for c in cases:
        if c["layer"] not in cached:
            s = {k: v.to(dev) for k, v in load_slabs(c["layer"], c["experts"]).items()}
            n = s["w2_trellis"].shape[0]
            w13 = [tuple(Exl3Tensors(s["w13_trellis"][i, p], s["w13_suh"][i, p], s["w13_svh"][i, p], True)
                         for p in (0, 1)) for i in range(n)]
            w2 = [Exl3Tensors(s["w2_trellis"][i], s["w2_suh"][i], s["w2_svh"][i], True) for i in range(n)]
            cached[c["layer"]] = (w13, w2)
        w13, w2 = cached[c["layer"]]
        x, sel, w = c["x"].to(dev), c["sel"].to(dev), c["w"].to(dev)
        experts = sorted(set(sel.flatten().tolist()))
        pair = []
        for linear in (exl3_linear, lambda xe, t, _dtype=None: exl3_linear_reference(xe, t)):
            out = torch.zeros(1, H, dtype=torch.float32, device=dev)
            exl3_moe_accumulate(out, x, w, sel, w13, w2, SWIGLU_LIMIT, experts, linear)
            pair.append(out.cpu())
        gpu.append(pair[0])
        ref.append(pair[1])
    return gpu, ref


def accuracy(layers, per_layer, tag):
    import torch

    cases = make_cases(layers, per_layer)
    cases_path = os.path.join(HERE, f"p0_cases_{tag}.pt")
    torch.save(cases, cases_path)
    cpu = {}
    for tier in ("bw", "scalar"):
        out_path = os.path.join(HERE, f"p0_cpu_{tier}_{tag}.pt")
        subprocess.run([sys.executable, __file__, "_cpu", tier, cases_path, out_path], check=True)
        cpu[tier] = torch.load(out_path)
    if not cpu["bw"]["bw"]:
        raise SystemExit("the bw run did not detect AVX-512BW")
    gpu, ref = gpu_outputs(cases)

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    groups = {}
    for i, c in enumerate(cases):
        key = (c["layer"], c["scale"], c["topk"])
        g = groups.setdefault(key, {"cpu_bw": [], "cpu_scalar": [], "gpu": [], "bw_vs_gpu": []})
        g["cpu_bw"].append(rel(cpu["bw"]["outs"][i], ref[i]))
        g["cpu_scalar"].append(rel(cpu["scalar"]["outs"][i], ref[i]))
        g["gpu"].append(rel(gpu[i], ref[i]))
        g["bw_vs_gpu"].append(rel(cpu["bw"]["outs"][i], gpu[i]))
        if not torch.isfinite(cpu["bw"]["outs"][i]).all():
            raise SystemExit(f"non-finite CPU output, case {i}")
    for (layer, scale, topk), g in sorted(groups.items()):
        emit({"mode": "accuracy", "tag": tag, "layer": layer, "scale": scale, "topk": topk, "cases": len(g["gpu"]),
              **{f"rel_l2_{k}_p50": statistics.median(v) for k, v in g.items()},
              **{f"rel_l2_{k}_max": max(v) for k, v in g.items()}})
    for path in [cases_path] + [os.path.join(HERE, f"p0_cpu_{t}_{tag}.pt") for t in ("bw", "scalar")]:
        os.remove(path)


def perf(layer, n, thread_list, topk_list, layout, tag):
    import torch

    torch.set_num_threads(4)
    e = ext()
    t0 = time.time()
    slabs = load_slabs(layer, list(range(n)))
    h = make_layer(e, slabs, layout == "swizzled")
    load_s = time.time() - t0
    expert_bytes = sum(t[0].numel() * t.element_size() for t in slabs.values())
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, H, generator=g).half()
    order = torch.randperm(n, generator=g).tolist()
    cursor = 0
    for threads in thread_list:
        check_affinity(threads)
        for topk in topk_list:
            w = torch.full((1, topk), 1.0 / topk).half()
            out = torch.zeros(1, H)

            def call():
                nonlocal cursor
                sel = torch.tensor([[order[(cursor + j) % n] for j in range(topk)]], dtype=torch.int64)
                cursor = (cursor + topk) % n
                t = time.perf_counter()
                e.exl3_moe_cpu_forward(h, x, sel, w, out, threads)
                return time.perf_counter() - t

            for _ in range(6):
                call()
            samples = sorted(call() for _ in range(max(40, 2 * n // topk)))
            med = statistics.median(samples)
            emit({"mode": "perf", "tag": tag, "layer": layer, "layout": layout, "threads": threads, "topk": topk,
                  "experts": n, "expert_bytes": expert_bytes, "calls": len(samples),
                  "cores": sorted(os.sched_getaffinity(0)),
                  "ms_p10": 1e3 * samples[len(samples) // 10], "ms_p50": 1e3 * med,
                  "ms_p90": 1e3 * samples[9 * len(samples) // 10], "ms_per_expert_p50": 1e3 * med / topk,
                  "eff_GBps_p50": topk * expert_bytes / med / 1e9, "finite": bool(torch.isfinite(out).all()),
                  "load_s": round(load_s, 1)})
    e.exl3_moe_cpu_free_layer(h)


def ints(s):
    return [int(v) for v in s.split(",")]


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "accuracy":
        accuracy(ints(args[0]), int(args[1]), args[2])
    elif mode == "perf":
        if args[4] not in ("native", "swizzled"):
            raise SystemExit("LAYOUT is native or swizzled")
        perf(int(args[0]), int(args[1]), ints(args[2]), ints(args[3]), args[4], args[5])
    elif mode == "_cpu":
        cpu_outputs(*args)
    else:
        raise SystemExit(__doc__)
