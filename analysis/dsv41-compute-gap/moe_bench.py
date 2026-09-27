"""E3 (handoff C3a/R2): the production in-graph routed MoE (``Exl3FusedMoE.run``) per extension build and knob.

Reuses resident-first's 40-layer real-row setup (``split_launch_bench.build_layers``: ~3.2 GB touched per token)
and its ``one`` arm, which is the production call. The group width (EXL3_MOE_GROUP_WIDTH, patched builds only)
and EXL3_MOE_TILE_N are read once per process, so each configuration is its own process:

    EXL3_MOE_GROUP_WIDTH=16 python moe_bench.py --variant knobs --out w16.json --save w16.pt
    python moe_bench.py --compare prod.pt w16.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _split_bench():
    path = os.path.join(HERE, "..", "dsv41-drive", "resident-first", "split_launch_bench.py")
    spec = importlib.util.spec_from_file_location("split_launch_bench", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--out")
    ap.add_argument("--save")
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()

    if args.compare:
        a, b = (torch.load(p) for p in args.compare)
        ya, yb = a.float(), b.float()
        print(json.dumps({
            "bitwise_equal": bool(torch.equal(a, b)),
            "frac_differ": float((ya != yb).float().mean()),
            "max_abs": float((ya - yb).abs().max()),
            "max_rel_to_rms": float((ya - yb).abs().max() / ya.pow(2).mean().sqrt()),
        }, indent=1))
        return

    import sglang.srt.layers.quantization.exl3_fused_moe  # noqa: F401  (bind before routing exl3_ext)

    if args.variant == "prod":
        prov = {"variant": "prod", "build_dir": os.environ.get("SGLANG_EXL3_BUILD_DIR")}
    else:
        sys.path.insert(0, HERE)
        import exl3_variant

        exl3_variant.use_variant(args.variant)
        prov = exl3_variant.provenance(args.variant)
    prov["EXL3_MOE_GROUP_WIDTH"] = os.environ.get("EXL3_MOE_GROUP_WIDTH")
    prov["EXL3_MOE_TILE_N"] = os.environ.get("EXL3_MOE_TILE_N")

    sb = _split_bench()
    par = sb._parity_module()
    device = torch.device("cuda", torch.cuda.current_device())
    real = par.load_slot_rows(device)
    row_bytes = sum(t[0].numel() * t.element_size() for t in real.values())
    layers = sb.build_layers(par, real, device, torch.Generator().manual_seed(2026))
    del real
    assert layers[0].fused.ext.__name__ != "sglang_exl3_ext" or args.variant == "prod", layers[0].fused.ext.__name__

    graph = sb.capture(lambda L: sb.one(par, L), layers)
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for rep in range(args.replays + 10):
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        if rep >= 10:
            ts.append(a.elapsed_time(b) * 1e3 / sb.LAYERS)
    ts.sort()
    med = statistics.median(ts)
    touched = sb.TOP_K * row_bytes
    report = {
        "provenance": prov,
        "ext_module": layers[0].fused.ext.__name__,
        "median_us_per_layer": med,
        "p10": ts[len(ts) // 10],
        "p90": ts[9 * len(ts) // 10],
        "implied_TBps": touched / (med * 1e-6) / 1e12,
    }
    print(json.dumps(report), flush=True)

    outs = []
    for L in layers:
        outs.append(L.fused.run(L.x, L.weights, L.remap, L.keep, par.ACT_LIMIT).clone())
    torch.cuda.synchronize()
    if args.save:
        torch.save(torch.cat(outs).cpu(), args.save)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
