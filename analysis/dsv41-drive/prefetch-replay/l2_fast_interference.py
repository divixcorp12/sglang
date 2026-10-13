"""Side-stream interference of the fused L+2 learned path (T0, T3) on a fixed main-stream bf16 GEMM load.

Four arms replay the same main-stream CUDA graph of bf16 GEMMs and time only the main stream with events: no side work,
the native scorer (spec_score + spec_select, 38 rotating slots), and the fused learned path of T0 and of T3
(spec_score + spec_head, 38 rotating slots). The side graph is launched on a second stream at the start of each main
replay. The main-load shapes are CLI parameters; how representative they are of one decode layer is a hypothesis.

Inputs: an R1 run dir (stages/l2/result.json names the L2_T0 and L2_T3 checkpoints, verified by sha256).
Output: --out interference.json {label, scope, main_load, width, runs, gpu, streams, main_alone, native_side,
        native_slowdown_vs_none, models: {L2_T0|L2_T3: {checkpoint_sha256, fused_side, slowdown_vs_native,
        slowdown_vs_none}}}.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python l2_fast_interference.py
       --run-dir RUN_DIR --out /mnt/nvme1/nvfp4-work/f123-l2-fast-DATE/interference.json --runs 1000 --width 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_cap1_scorers  # noqa: E402
import l2_fast_common as common  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
from b1_rows import TOPK  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.kernels.ops.moe import spec_head  # noqa: E402

MODELS = ("L2_T0", "L2_T3")
L2_BYTES = 96 * 2**20
EXPERTS = spec_head.HEAD_EXPERTS
# Arbitrary; one decode layer's dense work (attention projections plus shared expert), no evidence it is exact.
DEFAULT_SHAPES = "5120x1536,1536x8192,8192x5120,5120x4096,2048x5120"
SCOPE = ("main stream: one CUDA graph of bf16 GEMMs per weight copy, replayed round-robin; side stream: one CUDA graph "
         "per layer slot launched at the start of each main replay; only the main stream is timed (CUDA events); "
         "RTX 5090, graph replay")
MAIN_LABEL = f"{common.HYPOTHESIS}: shapes are a guess at one decode layer's dense work"


class MainLoad(msgspec.Struct):
    """The main-stream graphs (one per weight copy) and their size."""

    graphs: list[torch.cuda.CUDAGraph]
    shapes: list[list[int]]
    bytes_per_copy: int
    copies: int
    keep: list  # weights and activations the graphs read; alive as long as the graphs


def parse_shapes(text: str) -> list[tuple[int, int]]:
    """'KxN,KxN' to [(K, N)]."""
    out = []
    for part in text.split(","):
        k, n = part.lower().split("x")
        out.append((int(k), int(n)))
    return out


def copies_for(*, bytes_per_copy: int, l2_bytes: int = L2_BYTES) -> int:
    """Weight copies so one pass over them exceeds 2x L2; at least 2."""
    return max(2, math.ceil(2 * l2_bytes / bytes_per_copy))


def _capture(step: Callable[[], object]) -> torch.cuda.CUDAGraph:
    """Warm ``step`` on a side stream (JIT builds, allocator), then capture it into a CUDA graph."""
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def build_main_load(*, shapes: list[tuple[int, int]], tokens: int, device: str, seed: int = 0) -> MainLoad:
    """One graph of ``tokens``-row bf16 GEMMs per weight copy; the copies together exceed 2x L2."""
    per_copy = sum(k * n * 2 for k, n in shapes)
    copies = copies_for(bytes_per_copy=per_copy)
    g = torch.Generator(device=device).manual_seed(seed)
    keep, graphs = [], []
    for _ in range(copies):
        pairs = []
        for k, n in shapes:
            w = (torch.randn((k, n), generator=g, device=device) * 0.02).to(torch.bfloat16)
            a = torch.randn((tokens, k), generator=g, device=device).to(torch.bfloat16)
            pairs.append((a, w, torch.empty((tokens, n), dtype=torch.bfloat16, device=device)))
        keep.append(pairs)

        def step(pairs=pairs) -> None:
            for a, w, out in pairs:
                torch.mm(a, w, out=out)

        graphs.append(_capture(step))
    return MainLoad(graphs=graphs, shapes=[[k, n] for k, n in shapes], bytes_per_copy=per_copy, copies=copies,
                    keep=keep)


class SideInputs(msgspec.Struct):
    """Per-slot serving inputs: gate weights are per layer, x and scores are fresh per slot."""

    gates: list[torch.Tensor]  # bf16 [E, X_DIM]
    bias: torch.Tensor
    xs: list[torch.Tensor]  # bf16 [width, X_DIM]
    scores: list[torch.Tensor]  # fp32 [HEAD_MAX_TOKENS, E]
    ids: torch.Tensor  # int64 [width, TOPK]
    weights: torch.Tensor  # fp32 [width, TOPK]
    positions: torch.Tensor  # int64 [width]


def build_side_inputs(*, slots: int, width: int, device: str, seed: int = 1) -> SideInputs:
    g = torch.Generator(device=device).manual_seed(seed)
    dim = r1_cost.X_DIM
    gates = [(torch.randn((EXPERTS, dim), generator=g, device=device) * 0.02).to(torch.bfloat16) for _ in range(slots)]
    xs = [torch.randn((width, dim), generator=g, device=device).to(torch.bfloat16) for _ in range(slots)]
    scores = [torch.zeros((spec_head.HEAD_MAX_TOKENS, EXPERTS), dtype=torch.float32, device=device)
              for _ in range(slots)]
    ids = torch.stack([torch.randperm(EXPERTS, generator=g, device=device)[:TOPK] for _ in range(width)])
    return SideInputs(gates=gates, bias=torch.randn(EXPERTS, generator=g, device=device) * 0.1, xs=xs, scores=scores,
                      ids=ids, weights=torch.rand((width, TOPK), generator=g, device=device),
                      positions=torch.arange(width, dtype=torch.int64, device=device))


def native_graphs(inputs: SideInputs, *, width: int, device: str) -> list[torch.cuda.CUDAGraph]:
    """One graph per slot: spec_score then spec_select, as serving's native scorer."""
    page = es.new_candidate_page(pin=True)
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device=device)
    state[es.STATE_WORDS["posted"]] = 1
    hot = torch.full((16,), -1, dtype=torch.int64, device=device)
    ram_slot = torch.full((2, EXPERTS), -1, dtype=torch.int32, device=device)
    graphs = []
    for i in range(len(inputs.gates)):
        def step(i=i) -> None:
            es.run_spec_score(inputs.xs[i], inputs.gates[i], inputs.bias, inputs.scores[i])
            es.run_spec_select(inputs.scores[i], width, top_k=TOPK, per_token=1, top_k_only=False, hot_slots=hot,
                               hot_capacity=16, ram_slot=ram_slot, target=1, state=state, candidates=page)

        graphs.append(_capture(step))
    return graphs


def fused_graphs(inputs: SideInputs, *, head: l2_fast_kernels.KernelHead, device: str) -> list[torch.cuda.CUDAGraph]:
    """One graph per slot: spec_score then the fused learned path (its tail selects)."""
    skip = torch.zeros(EXPERTS, dtype=torch.bool, device=device)
    threshold = torch.full((1,), 0.5, dtype=torch.float32, device=device)  # arbitrary; cost does not depend on it
    graphs = []
    for i in range(len(inputs.gates)):
        def step(i=i) -> None:
            es.run_spec_score(inputs.xs[i], inputs.gates[i], inputs.bias, inputs.scores[i])
            head.launch(scores=inputs.scores[i], x=inputs.xs[i], src_ids=inputs.ids, src_w=inputs.weights,
                        positions=inputs.positions, row=0, skip=skip, threshold=threshold)

        graphs.append(_capture(step))
    return graphs


def _check_side(side: torch.cuda.Stream) -> None:
    if side == torch.cuda.current_stream():
        raise ValueError("the side stream must differ from the main stream")


def time_main(main: list[torch.cuda.CUDAGraph], side_graphs: list[torch.cuda.CUDAGraph] | None, *,
              side: torch.cuda.Stream, runs: int, warmup: int) -> dict:
    """Median/p90/p99 ms of a main replay with ``side_graphs`` (or nothing) launched alongside, on ``side``."""
    _check_side(side)
    mains = torch.cuda.current_stream()

    def one(i: int, start: torch.cuda.Event | None, stop: torch.cuda.Event | None) -> None:
        gate = torch.cuda.Event()
        gate.record(mains)
        if side_graphs is not None:
            side.wait_event(gate)
            with torch.cuda.stream(side):
                side_graphs[i % len(side_graphs)].replay()
        if start is not None:
            start.record(mains)
        main[i % len(main)].replay()
        if stop is not None:
            stop.record(mains)

    for i in range(warmup):
        one(i, None, None)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for i in range(runs):
        one(i, starts[i], stops[i])
    torch.cuda.synchronize()
    ms = np.array([a.elapsed_time(b) for a, b in zip(starts, stops)], dtype=np.float64)
    return {"median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
            "p99_ms": float(np.percentile(ms, 99)), "runs": runs, "graphs": len(main)}


def serving_row(model: r1_models.R1Model) -> tuple[l2_fast_torch.FoldRow, ...]:
    """One fold row from the checkpoint's vocabulary; the kernels' cost does not depend on which row."""
    vocab = model.head.vocab
    return (l2_fast_torch.FoldRow(source_layer=vocab.layers[0], target_layer=vocab.layers[-1],
                                  stratum=vocab.strata[0]),)


def load_heads(*, checkpoints: dict[str, tuple[str, str]]) -> dict[str, tuple[l2_fast_torch.FoldedHead, str]]:
    """{scorer: (folded head, verified checkpoint sha256)} for T0 and T3."""
    out = {}
    for name in MODELS:
        path, sha = checkpoints[name]
        actual = l2_cap1_scorers.verify_ckpt(path, sha=sha)
        model = r1_models.load_r1(path)[0]
        out[name] = (l2_fast_torch.fold(model, rows=serving_row(model)), actual)
    return out


def measure(*, heads: dict[str, tuple[l2_fast_torch.FoldedHead, str]], shapes: list[tuple[int, int]],
            main_tokens: int, width: int, slots: int, runs: int, warmup: int = 50, device: str = "cuda") -> dict:
    """The four arms; the fused arms are built and timed one head at a time."""
    main = build_main_load(shapes=shapes, tokens=main_tokens, device=device)
    inputs = build_side_inputs(slots=slots, width=width, device=device)
    side = torch.cuda.Stream()
    streams = {"main_stream_ptr": int(torch.cuda.current_stream().cuda_stream), "side_stream_ptr": int(side.cuda_stream)}
    alone = time_main(main.graphs, None, side=side, runs=runs, warmup=warmup)
    native = time_main(main.graphs, native_graphs(inputs, width=width, device=device), side=side, runs=runs,
                       warmup=warmup)
    models = {}
    for name, (folded, sha) in heads.items():
        head = l2_fast_kernels.KernelHead(folded, device=device)
        fused = time_main(main.graphs, fused_graphs(inputs, head=head, device=device), side=side, runs=runs,
                          warmup=warmup)
        models[name] = {"checkpoint_sha256": sha, "fused_side": fused,
                        "slowdown_vs_native": fused["median_ms"] / native["median_ms"] - 1.0,
                        "slowdown_vs_none": fused["median_ms"] / alone["median_ms"] - 1.0}
    return {
        "label": common.OBSERVED, "scope": SCOPE,
        "main_load": {"label": MAIN_LABEL, "shapes": main.shapes, "tokens": main_tokens,
                      "bytes": main.bytes_per_copy, "copies": main.copies, "rotated_bytes": main.bytes_per_copy * main.copies},
        "width": width, "runs": runs, "slots": slots, "gpu": torch.cuda.get_device_name(), "streams": streams,
        "main_alone": alone, "native_side": native,
        "native_slowdown_vs_none": native["median_ms"] / alone["median_ms"] - 1.0, "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--width", type=int, default=6, help="live tokens of the side scorer")
    ap.add_argument("--main-tokens", type=int, default=6)
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="main-load GEMMs as KxN,KxN (hypothesis, see SCOPE)")
    ap.add_argument("--slots", type=int, default=common.SLOTS)
    args = ap.parse_args(argv)
    out = common.guard_out(args.out, run_dir=args.run_dir)
    with open(os.path.join(args.run_dir, "stages", "l2", "result.json")) as f:
        checkpoints = l2_cap1_scorers.learned_ckpts(json.load(f))
    result = measure(heads=load_heads(checkpoints=checkpoints), shapes=parse_shapes(args.shapes),
                     main_tokens=args.main_tokens, width=args.width, slots=args.slots, runs=args.runs)
    common.write_json(out, result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
