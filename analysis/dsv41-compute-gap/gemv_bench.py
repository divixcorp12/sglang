"""E2 (handoff C2b/R1): BS1 dense EXL3 GEMVs at the DSV4.1 attention and shared-expert shapes.

Real 5-bpw weights of all 40 layers (~3.2 GB, far past L2) are loaded from the checkpoint. Each kind is
captured as one CUDA graph of 40 calls (one per layer, as decode issues them) through ``exl3_gemm`` exactly
as ``exl3_gemm_bs1`` calls it; graphs are replayed round-robin and timed with CUDA events.

One process per extension build (the extension reads its env knobs once):

    python gemv_bench.py --variant prod  --out prod.json --save prod.pt   # the production build
    python gemv_bench.py --variant gemv_d2 --out d2.json --save d2.pt     # an exl3_variant build
    python gemv_bench.py --compare prod.pt d2.pt                          # bitwise comparison
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

MODEL = "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
LAYERS = 40
KINDS = {
    # name: (checkpoint tensor, input features)
    "wq_a": ("attn.wq_a", 5120),
    "wq_b": ("attn.wq_b", 1280),
    "wkv": ("attn.wkv", 5120),
    "wo_b": ("attn.wo_b", 8192),
    "sh_w1": ("ffn.shared_experts.w1", 5120),
    "sh_w3": ("ffn.shared_experts.w3", 5120),
    "sh_w2": ("ffn.shared_experts.w2", 2304),
}


def load_weights(device):
    from safetensors import safe_open

    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for layer in range(LAYERS):
        for base, _ in KINDS.values():
            for part in ("trellis", "suh", "svh"):
                key = f"layers.{layer}.{base}.{part}"
                by_file.setdefault(index[key], []).append(key)
    raw = {}
    for fname, keys in by_file.items():
        with safe_open(os.path.join(MODEL, fname), framework="pt", device="cpu") as f:
            for k in keys:
                raw[k] = f.get_tensor(k).to(device)
    return {
        kind: [
            tuple(raw[f"layers.{layer}.{base}.{p}"] for p in ("trellis", "suh", "svh")) for layer in range(LAYERS)
        ]
        for kind, (base, _) in KINDS.items()
    }


def get_ext(variant: str):
    if variant == "prod":
        from sglang.srt.layers.quantization.exl3_ext import exl3_ext

        return exl3_ext(), {"variant": "prod", "build_dir": os.environ.get("SGLANG_EXL3_BUILD_DIR")}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import exl3_variant

    return exl3_variant.load_variant(variant), exl3_variant.provenance(variant)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--out")
    ap.add_argument("--save")
    ap.add_argument("--replays", type=int, default=400)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()

    if args.compare:
        a, b = (torch.load(p) for p in args.compare)
        res = {}
        for kind in a:
            ya, yb = a[kind].float(), b[kind].float()
            res[kind] = {
                "bitwise_equal": bool(torch.equal(a[kind], b[kind])),
                "frac_differ": float((ya != yb).float().mean()),
                "max_abs": float((ya - yb).abs().max()),
                "max_rel_to_rms": float((ya - yb).abs().max() / ya.pow(2).mean().sqrt()),
            }
        print(json.dumps(res, indent=1))
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    ext, prov = get_ext(args.variant)
    weights = load_weights(device)
    gen = torch.Generator(device="cpu").manual_seed(7)
    xs = {k: [torch.randn((1, fin), generator=gen).to(device, torch.float16) for _ in range(LAYERS)] for k, (_, fin) in KINDS.items()}
    ys = {k: [torch.empty((1, weights[k][0][2].shape[0]), dtype=torch.float16, device=device) for _ in range(LAYERS)] for k in KINDS}
    scratch = {k: [torch.empty_like(x) for x in xs[k]] for k in KINDS}

    def run(kind):
        for layer in range(LAYERS):
            tr, suh, svh = weights[kind][layer]
            ext.exl3_gemm(xs[kind][layer], tr, ys[kind][layer], suh, scratch[kind][layer], svh, -1, False, True, 0)

    graphs = {k: capture(lambda k=k: run(k)) for k in KINDS}
    graphs["attn_4"] = capture(lambda: [run(k) for k in ("wq_a", "wq_b", "wkv", "wo_b")])
    graphs["shared_3"] = capture(lambda: [run(k) for k in ("sh_w1", "sh_w3", "sh_w2")])

    times = {k: [] for k in graphs}
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    names = list(graphs)
    for rep in range(args.replays + 10):
        for i in range(len(names)):
            n = names[(i + rep) % len(names)]
            a.record()
            graphs[n].replay()
            b.record()
            b.synchronize()
            if rep >= 10:
                times[n].append(a.elapsed_time(b) * 1e3 / LAYERS)

    report = {"provenance": prov, "replays": args.replays, "device": torch.cuda.get_device_name(), "kinds": {}}
    for n, ts in times.items():
        ts.sort()
        med = statistics.median(ts)
        entry = {"median_us_per_layer": med, "p10": ts[len(ts) // 10], "p90": ts[9 * len(ts) // 10]}
        if n in KINDS:
            nbytes = weights[n][0][0].numel() * 2
            entry["trellis_bytes"] = nbytes
            entry["implied_TBps"] = nbytes / (med * 1e-6) / 1e12
        report["kinds"][n] = entry
        print(f"{n:9s} {med:8.3f} us/layer  p10 {entry['p10']:.3f} p90 {entry['p90']:.3f}"
              + (f"  {entry['implied_TBps']:.3f} TB/s" if "implied_TBps" in entry else ""), flush=True)

    # One eager pass on the fixed inputs for bitwise comparison between builds.
    for k in KINDS:
        run(k)
    torch.cuda.synchronize()
    if args.save:
        torch.save({k: torch.cat(ys[k]).cpu() for k in KINDS}, args.save)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
