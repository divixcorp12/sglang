"""EXL3 mul1 CPU expert GEMV on divix01, at DSV4.1 geometry (hidden 5120, intermediate 2304, 3 bpw).

Needs exllamav3 cloned beside this file (git clone https://github.com/turboderp-org/exllamav3; checkout 02aef45):
the kernel is its cpu/moe_mul1.cpp. Run with EXL3_MOE_CPU_PIN=0, because the kernel's pool pins its workers to the
machine's first physical cores and ignores taskset.

Modes:
  build                      compile the extension
  tiers                      bw (AVX-512BW) tier vs scalar fp32 reference on DSV4.1 shapes, 2 experts, 1 thread
  perf  E THREADS TOPKS TAG  cold expert throughput: E experts registered, each call routes TOPK distinct experts,
                             rotated so no expert repeats within E/TOPK calls (13.3 MB each, L3 24.75 MB)
  bw    GIB THREADS TAG      streaming-read ceiling over a GIB buffer
"""
import json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXL = os.path.join(HERE, "exllamav3", "exllamav3", "exllamav3_ext")
H, I, KB = 5120, 2304, 3


def ext():
    import torch
    from torch.utils.cpp_extension import load
    return load(
        name="exl3cpubench",
        sources=[os.path.join(EXL, "cpu", "moe_mul1.cpp"), os.path.join(HERE, "bench_ext.cpp")],
        extra_include_paths=[EXL],
        extra_cflags=["-O3", "-Ofast"],
        build_directory=os.path.join(HERE, "build"),
        verbose=False,
    )


def emit(rec):
    rec["host_cpus"] = os.sched_getaffinity(0).__len__()
    line = json.dumps(rec)
    print(line, flush=True)
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(line + "\n")


def weights(n_experts, gen, swizzled_random=True):
    import torch
    lists = {k: [] for k in ("gt", "gs", "gv", "ut", "us", "uv", "dt", "ds", "dv")}

    def trellis(k, n):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * KB), dtype=torch.int16, generator=gen)

    def signs(n, scale, spread):
        s = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
        return (s * (scale + spread * torch.randn(n, generator=gen))).half().contiguous()

    for _ in range(n_experts):
        lists["gt"].append(trellis(H, I)); lists["gs"].append(signs(H, 0.015, 0.004)); lists["gv"].append(signs(I, 1.0, 0.1))
        lists["ut"].append(trellis(H, I)); lists["us"].append(signs(H, 0.015, 0.004)); lists["uv"].append(signs(I, 1.0, 0.1))
        lists["dt"].append(trellis(I, H)); lists["ds"].append(signs(I, 0.015, 0.004)); lists["dv"].append(signs(H, 1.0, 0.1))
    return lists


def make(e, w, swz):
    return e.make_layer(w["gt"], w["gs"], w["gv"], w["ut"], w["us"], w["uv"], w["dt"], w["ds"], w["dv"],
                        [], [], [], 0, 0.0, swz)


def swizzle(ts):
    out = []
    for t in ts:
        tk, tn, ps = t.shape
        out.append(t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps))
    return out


def tier_worker(tier, path):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = tier
    import torch
    e = ext()
    g = torch.Generator().manual_seed(20260929)
    w = weights(2, g)
    x = torch.randn(1, H, generator=g).half()
    sel = torch.tensor([[0, 1]], dtype=torch.int64)
    rw = torch.tensor([[0.6, 0.4]]).half()
    res = {}
    h = make(e, w, 0)
    out = torch.zeros(1, H); e.forward(h, x, sel, rw, out, 1); res["native"] = out.clone()
    e.free_layer(h)
    if tier != "scalar":
        ws = dict(w); ws["gt"], ws["ut"], ws["dt"] = swizzle(w["gt"]), swizzle(w["ut"]), swizzle(w["dt"])
        h = make(e, ws, 1)
        out = torch.zeros(1, H); e.forward(h, x, sel, rw, out, 1); res["swizzled"] = out.clone()
        e.free_layer(h)
    res["bw_flag"] = e.has_avx512_bw()
    torch.save(res, path)


def tiers():
    import torch
    outs = {}
    for tier in ("bw", "scalar"):
        path = os.path.join(HERE, f"tier_{tier}.pt")
        subprocess.run([sys.executable, __file__, "_tier", tier, path], check=True)
        outs[tier] = torch.load(path)
    ref = outs["scalar"]["native"]
    def rel(a): return float((a - ref).norm() / ref.norm())
    emit({"mode": "tiers", "bw_detected": bool(outs["bw"]["bw_flag"]),
          "rel_l2_bw_native_vs_scalar": rel(outs["bw"]["native"]),
          "rel_l2_bw_swizzled_vs_scalar": rel(outs["bw"]["swizzled"]),
          "rel_l2_bw_swizzled_vs_native": float((outs["bw"]["swizzled"] - outs["bw"]["native"]).norm()
                                                / outs["bw"]["native"].norm()),
          "ref_norm": float(ref.norm()), "finite": bool(torch.isfinite(outs["bw"]["native"]).all())})


def perf(n_experts, thread_list, topk_list, tag):
    import torch
    torch.set_num_threads(min(16, len(os.sched_getaffinity(0))))
    e = ext()
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    w = weights(n_experts, g)
    h = make(e, w, 1)
    expert_bytes = sum(t.numel() * t.element_size() for k in w for t in [w[k][0]])
    gen_s = time.time() - t0
    x = torch.randn(1, H).half()
    order = torch.randperm(n_experts, generator=g).tolist()
    cursor = 0
    for threads in thread_list:
        for topk in topk_list:
            rw = torch.full((1, topk), 1.0 / topk).half()
            out = torch.zeros(1, H)
            def call():
                nonlocal cursor
                ids = [order[(cursor + j) % n_experts] for j in range(topk)]
                cursor = (cursor + topk) % n_experts
                sel = torch.tensor([ids], dtype=torch.int64)
                t = time.perf_counter(); e.forward(h, x, sel, rw, out, threads); return time.perf_counter() - t
            for _ in range(6):
                call()
            samples = sorted(call() for _ in range(max(40, 2 * n_experts // topk)))
            med = statistics.median(samples)
            emit({"mode": "perf", "tag": tag, "threads": threads, "topk": topk, "experts_registered": n_experts,
                  "expert_bytes": expert_bytes, "calls": len(samples),
                  "ms_p10": 1e3 * samples[len(samples) // 10], "ms_p50": 1e3 * med,
                  "ms_p90": 1e3 * samples[9 * len(samples) // 10],
                  "ms_per_expert_p50": 1e3 * med / topk, "eff_GBps_p50": topk * expert_bytes / med / 1e9,
                  "finite": bool(torch.isfinite(out).all()), "gen_s": round(gen_s, 1)})


def bw(gib, thread_list, tag):
    import torch
    e = ext()
    buf = torch.empty(int(gib * (1 << 30)), dtype=torch.uint8)
    buf.fill_(1)
    for threads in thread_list:
        e.read_seconds(buf, threads)
        s = sorted(e.read_seconds(buf, threads) for _ in range(5))
        emit({"mode": "bw", "tag": tag, "threads": threads, "gib": gib,
              "GBps_p50": buf.numel() / s[2] / 1e9, "GBps_best": buf.numel() / s[0] / 1e9})


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        e = ext(); print("built; avx512_bw", e.has_avx512_bw(), "vnni", e.has_avx512_vnni())
    elif mode == "tiers":
        tiers()
    elif mode == "_tier":
        tier_worker(sys.argv[2], sys.argv[3])
    elif mode == "perf":
        perf(int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], [int(v) for v in sys.argv[4].split(",")],
             sys.argv[5])
    elif mode == "bw":
        bw(float(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], sys.argv[4])
