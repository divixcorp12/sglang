"""R1 complete offline scorer cost under one CUDA graph (D13, D17): gate through top-2 at 6 live tokens.

The timed step is: fp32 gate scores from source x, E5 feature build (PCA, route mask, gate score and margin), the R1
head with its block aggregator, sigmoid, the skip (hot/RAM) filter, the threshold and the capped top-k. It is timed
on the plan's intended path (fp32 gate and features, bf16 head and aggregator) and on the all-fp32 reference path,
each beside a head-only graph (head plus aggregator plus sigmoid on precomputed features). Dtypes come from the
plan's CostSpec. read_set_diff measures how often the capped read set differs between the two paths.

Inputs: an R1 checkpoint (r1_models.save_r1), the target layer's gate (W, bias) and a run_plan.json.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python r1_cost.py CKPT --plan run_plan.json
       --model-path MODEL --layer 20 --out cost.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_policy  # noqa: E402
from b1_rows import GATE_DEPTH, TOPK  # noqa: E402
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training import llapor  # noqa: E402

X_DIM = 5120
_SHORT = {"bfloat16": "bf16", "float32": "fp32"}


def select_torch(block_scores: torch.Tensor, *, skip: torch.Tensor, threshold: torch.Tensor, cap: int) -> torch.Tensor:
    """Up to ``cap`` expert ids scoring >= threshold, best first, ties by ascending id; -1 pads. Graph-capturable."""
    vals = torch.where(skip, torch.full_like(block_scores, float("-inf")), block_scores)
    by_val = torch.sort(vals, descending=True, stable=True).indices
    # A second stable sort puts skipped rows last without disturbing the (-val, id) order.
    keep_first = torch.sort(skip[by_val].to(torch.uint8), stable=True).indices
    order = by_val[keep_first][:cap]
    ok = ~skip[order] & (block_scores[order] >= threshold)
    return torch.where(ok, order, torch.full_like(order, -1))


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def _gate_features(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, *, ref: r1_models.R1Model,
                   W: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """f32 [n, F]: PCA | mask | route (| gate score | gate margin for E5), as b1_head.make_batch builds them."""
    e = ref.head.cfg.num_experts
    scores = r1_gate.full_gate_scores(x, W=W, bias=bias)
    u = llapor.encode_features(x, ids, weights, pca=ref.head.pca, num_experts=e)
    if ref.head.cfg.variant != "E5":
        return u
    top = torch.topk(scores, GATE_DEPTH, dim=-1)
    in_top = torch.zeros_like(scores).scatter_(1, top.indices, 1.0)
    margin = scores - top.values[:, TOPK - 1:TOPK]
    return torch.cat([u, scores * in_top, margin * in_top], dim=-1)


def _static_batch(*, tokens: int, dtype: torch.dtype, device: str, experts: int) -> GroupBatch:
    """One full group of device tensors (as_tensor is a no-op on them, so capture does no host copies)."""
    zero_i = torch.zeros(1, dtype=torch.int64, device=device)
    return GroupBatch(
        live=torch.ones((1, tokens), dtype=torch.bool, device=device),
        position=torch.arange(tokens, device=device).unsqueeze(0),
        width=torch.full((1,), float(tokens), dtype=dtype, device=device),
        source_layer=zero_i, target_layer=zero_i, stratum=zero_i,
        miss=torch.zeros((1, experts), device=device), available=torch.ones((1, experts), device=device),
        route=torch.zeros((1, tokens, experts), device=device),
        token_y=torch.zeros((1, tokens, experts), device=device),
        hard_neg=torch.zeros((1, tokens, experts), device=device))


class _Static(msgspec.Struct):
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    skip: torch.Tensor
    threshold: torch.Tensor


def _make_static(*, tokens: int, experts: int, device: str) -> _Static:
    gen = torch.Generator(device="cpu").manual_seed(0)
    skip = torch.rand(experts, generator=gen) < 0.3
    return _Static(x=torch.randn((tokens, X_DIM), generator=gen).to(device),
                   ids=torch.randint(0, experts, (tokens, TOPK), generator=gen).to(device),
                   weights=torch.rand((tokens, TOPK), generator=gen).to(device),
                   skip=skip.to(device), threshold=torch.zeros((), device=device))


def _capture(step) -> torch.cuda.CUDAGraph:
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    return graph


def _time_graph(step, *, runs: int, warmup: int) -> dict:
    graph = _capture(step)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = b1_cost._replay_times_ms(graph, runs=runs)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "p99_ms": float(np.percentile(times, 99)), "runs": runs}


def _bytes(model: torch.nn.Module, *, itemsize: int) -> dict:
    return {"param_bytes": itemsize * sum(p.numel() for p in model.parameters()),
            "buffer_bytes": itemsize * sum(b.numel() for b in model.buffers())}


def _path_cost(model: r1_models.R1Model, *, ref: r1_models.R1Model, W: torch.Tensor, bias: torch.Tensor,
               st: _Static, tokens: int, dtype: torch.dtype, cap: int, runs: int, warmup: int, device: str) -> dict:
    batch = _static_batch(tokens=tokens, dtype=dtype, device=device, experts=ref.head.cfg.num_experts)

    def head_scores(u: torch.Tensor) -> torch.Tensor:
        inp = ModelInput(batch=batch, u=u.to(dtype).unsqueeze(0), x=None, s_native=None)
        return torch.sigmoid(model(inp)[0].float())

    with torch.no_grad():
        u_fixed = _gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias)

    def head_step() -> torch.Tensor:
        return head_scores(u_fixed)

    def complete_step() -> torch.Tensor:
        u = _gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias)
        return select_torch(head_scores(u), skip=st.skip, threshold=st.threshold, cap=cap)

    return {"complete": _time_graph(complete_step, runs=runs, warmup=warmup),
            "head_only": _time_graph(head_step, runs=runs, warmup=warmup),
            **_bytes(model, itemsize=torch.finfo(dtype).bits // 8)}


def _check_cost_spec(cost: r1_plan.CostSpec) -> None:
    if cost.head_dtype != cost.aggregator_dtype:
        raise ValueError("head and aggregator share one module, so their dtypes must match")
    if cost.gate_dtype != "float32":
        raise ValueError("the gate and feature build are measured in float32 only")


def measure_complete(model: r1_models.R1Model, *, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec,
                     tokens: int = 6, runs: int = 1000, warmup: int = 50, device: str = "cuda") -> dict:
    """CUDA-graph timings of the complete scorer on the intended dtype path and the reference path, plus bytes."""
    _check_cost_spec(cost)
    if tokens != cost.live_tokens:
        raise ValueError(f"tokens {tokens} != the plan's live_tokens {cost.live_tokens}")
    ref = copy.deepcopy(model).float().to(device).eval()
    W, bias = W.float().to(device), bias.float().to(device)
    st = _make_static(tokens=tokens, experts=ref.head.cfg.num_experts, device=device)
    out: dict = {"tokens": tokens, "gate_dtype": cost.gate_dtype, "gate_bytes": 4 * (W.numel() + bias.numel())}
    for name in (cost.head_dtype, cost.reference_dtype):
        dtype = _dtype(name)
        path_model = ref if dtype == torch.float32 else copy.deepcopy(ref).to(dtype).eval()
        out[_SHORT[name]] = _path_cost(path_model, ref=ref, W=W, bias=bias, st=st, tokens=tokens, dtype=dtype,
                                       cap=2, runs=runs, warmup=warmup, device=device)
    low = out[_SHORT[cost.head_dtype]]
    out["param_bytes"], out["buffer_bytes"] = low["param_bytes"], low["buffer_bytes"]
    out["gate_ms"] = low["complete"]["median_ms"]
    out["selection_matches_policy"] = _selection_matches_static(ref, W=W, bias=bias, st=st, device=device)
    return out


def _selection_matches_static(ref: r1_models.R1Model, *, W: torch.Tensor, bias: torch.Tensor, st: _Static,
                              device: str) -> bool:
    tokens = st.x.shape[0]
    batch = _static_batch(tokens=tokens, dtype=torch.float32, device=device, experts=ref.head.cfg.num_experts)
    with torch.no_grad():
        u = _gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias).unsqueeze(0)
        scores = torch.sigmoid(ref(ModelInput(batch=batch, u=u, x=None, s_native=None)).float())
    return selection_mismatches(scores, skips=st.skip.unsqueeze(0), threshold=float(st.threshold), cap=2) == 0


def selection_mismatches(block_scores: torch.Tensor, *, skips: torch.Tensor, threshold: float, cap: int) -> int:
    """Groups [G,E] where select_torch's read list differs from r1_policy.candidates + select (order included)."""
    thr = torch.tensor(threshold, dtype=block_scores.dtype, device=block_scores.device)
    bad = 0
    for row, skip in zip(block_scores, skips, strict=True):
        got = select_torch(row, skip=skip.bool(), threshold=thr, cap=cap)
        got = got[got >= 0].cpu().numpy()
        c = r1_policy.candidates(row.float().cpu().numpy(), skip=skip.bool().cpu().numpy())
        want = r1_policy.select(c, threshold=threshold, cap=cap)
        bad += int(not np.array_equal(got, want))
    return bad


def _cast_input(inp: ModelInput, *, dtype: torch.dtype, device) -> ModelInput:
    batch = msgspec.structs.replace(inp.batch, width=torch.as_tensor(inp.batch.width, dtype=dtype, device=device))
    return ModelInput(batch=batch, u=inp.u.to(device=device, dtype=dtype), x=None, s_native=None)


def read_set_diff(model: r1_models.R1Model, *, inputs: list[ModelInput], skips: list, threshold: float,
                  cap: int = 2) -> float:
    """Fraction of groups whose capped read set differs between the bf16 head and the fp32 head."""
    ref = copy.deepcopy(model).float().eval()
    low = copy.deepcopy(ref).to(torch.bfloat16).eval()
    device = ref.device
    thr = torch.tensor(threshold, device=device)
    differ = total = 0
    with torch.no_grad():
        for inp, skip in zip(inputs, skips, strict=True):
            sk = torch.as_tensor(skip, device=device).bool()
            p32 = torch.sigmoid(ref(_cast_input(inp, dtype=torch.float32, device=device)).float())
            p16 = torch.sigmoid(low(_cast_input(inp, dtype=torch.bfloat16, device=device)).float())
            for a, b, s in zip(p32, p16, sk, strict=True):
                ra = select_torch(a, skip=s, threshold=thr, cap=cap)
                rb = select_torch(b, skip=s, threshold=thr, cap=cap)
                differ += int(set(ra[ra >= 0].tolist()) != set(rb[rb >= 0].tolist()))
                total += 1
    if total == 0:
        raise ValueError("no groups to compare")
    return differ / total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layer", type=int, required=True, help="target layer id whose gate is timed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args(argv)
    plan, plan_sha = r1_plan.load_plan(args.plan)
    model, sha, _ = r1_models.load_r1(args.checkpoint)
    W, bias = r1_gate.load_target_gates(args.model_path, layer_ids=[args.layer])[args.layer]
    result = measure_complete(model, W=W, bias=bias, cost=plan.cost, tokens=plan.cost.live_tokens, runs=args.runs,
                              warmup=args.warmup)
    result.update({"checkpoint_sha256": sha, "plan_sha256": plan_sha, "layer": args.layer,
                   "gpu": torch.cuda.get_device_name()})
    tmp = f"{args.out}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
