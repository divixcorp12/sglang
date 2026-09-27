#!/usr/bin/env python3
"""Upper bound on what PDL saves per layer on the lease chain: a skeleton with production launch shapes.

The chain post -> W1 -> C1 -> A1 -> S -> A2 -> CW -> F (exl3_ram_miss.py post(); shapes from the launchers:
1x32, 1x32, 8x256, 1x8, 8x256, 1x8, 1x256, 1x32) is captured 40 times in one CUDA graph, each layer separated by
a non-PDL 'moe' stage (the real graph has attention and MoE between chains). Time per replay with CUDA events.

    PYTHONPATH=<repo>/python python skeleton.py --repo <repo> --out skeleton.jsonl
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAIN = (("post", 1, 32), ("w1", 1, 32), ("c1", 8, 256), ("a1", 1, 8), ("s", 8, 256), ("a2", 1, 8), ("cw", 1, 256),
         ("f", 1, 32))
MOE = ("moe", 1, 256)
LAYERS = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--work-ns", default="0,2000", help="comma-separated per-stage body spins")
    ap.add_argument("--pre-ns", default="0", help="comma-separated per-stage spins before the PDL wait (prologue)")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.jit.utils.compile.loader import load_jit

    mod = load_jit("chain_pdl_skeleton", cuda_files=[str(HERE / "skeleton.cuh")],
                   cuda_wrappers=[("skel_stage", "skel_stage")])
    stages = []
    for _ in range(LAYERS):
        stages += [(name, g, b, True) for name, g, b in CHAIN] + [(*MOE, False)]
    out = open(a.out, "a")
    for work_ns, pre_ns in [(w, p) for w in map(int, a.work_ns.split(",")) for p in map(int, a.pre_ns.split(","))]:
        for mode in (0, 1, 2):
            words = torch.zeros(len(stages), dtype=torch.int64, device="cuda")
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for i, (_, g, b, pdl) in enumerate(stages):  # warm: loads the module's kernels before capture
                    mod.skel_stage(words, i, g, b, work_ns, mode if pdl else 0, pre_ns if pdl else 0)
            torch.cuda.synchronize()
            words.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for i, (_, g, b, pdl) in enumerate(stages):
                    mod.skel_stage(words, i, g, b, work_ns, mode if pdl else 0, pre_ns if pdl else 0)
            torch.cuda.synchronize()
            words.zero_()
            times = []
            for r in range(a.replays + 20):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graph.replay()
                e1.record()
                torch.cuda.synchronize()
                if r >= 20:
                    times.append(e0.elapsed_time(e1) * 1e3)
            got = set(words.cpu().tolist())
            if got != {a.replays + 20}:
                raise SystemExit(f"mode {mode}: stage words {sorted(got)[:5]}, not all {a.replays + 20}: an edge lost "
                                 "its ordering")
            us = statistics.median(times)
            rec = {"kind": "skeleton", "mode": mode, "work_ns": work_ns, "pre_ns": pre_ns, "layers": LAYERS,
                   "replay_us_p50": round(us, 2), "per_layer_us_p50": round(us / LAYERS, 3)}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
