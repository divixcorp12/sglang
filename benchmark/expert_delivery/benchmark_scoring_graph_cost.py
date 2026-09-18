#!/usr/bin/env python3
"""Per-decode-step CUDA-graph replay cost of LLaPor prefetch scoring, by component.

Loads the real LLaPor checkpoints, builds BS1 feature buffers for every MoE
layer, and captures each component for all targets into its own CUDA graph,
the way the decode graph records the FeatureStore ``after_write`` hook. Every
sample is ``synchronize; replay; synchronize`` wall time; the report is the
median over ``--replays`` samples plus the CUDA kernel count of one eager step.

Components:
  a  taps          FeatureStore copies of router_input, topk_ids, topk_weights (48 layers)
  b0 pca           (x - mean) @ projection
  b  dense_input   zeros x2, scatter x2, weight cast, cat
  c  mlp           predictor MLP, float, sigmoid
  d  bank16 / top1 width-16 reference bank write vs JIT top-1 selector
  e  recall        BudgetRecall.observe
Composites run the real ``PrefetchScoring`` hook for the C arm (pull off, recall
off, calibration off) with the width-16 reference bank and with the default
selection for that configuration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def _time_graph(fn, *, replays: int, pool) -> dict:
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[len(samples) // 10],
        "p90_ms": samples[(len(samples) * 9) // 10],
    }


def _kernel_count(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630")
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--residents", type=int, default=160)
    parser.add_argument("--only", default="", help="comma list of component names")
    args = parser.parse_args()

    from sglang.kernels.ops.moe.expert_prefetch_top1 import select_prefetch_top1_cuda
    from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
    from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
    from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
    from sglang.srt.layers.moe.expert_prediction.serving.runtime import PrefetchScoring
    from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_dir = Path(args.model_dir)
    first_pair = sorted((model_dir / "llapor").glob("pair-*"))[0]
    hidden = torch.load(first_pair / "pca.pt", weights_only=True)["mean"].shape[0]
    experts, top_k, layers = 512, 10, 48
    specs = [MoeLayerSpec(layer_id=i, num_experts=experts, top_k=top_k, hidden_size=hidden) for i in range(layers)]
    checkpoints = load_prefetch_checkpoints(model_dir, predictor="llapor", specs=specs)
    targets = sorted(checkpoints)
    source = {t: checkpoints[t].source_layer for t in targets}

    generator = torch.Generator().manual_seed(0)
    hidden_in = [torch.randn(1, hidden, generator=generator).to(device, dtype) for _ in range(layers)]
    ids_in = [torch.randperm(experts, generator=generator)[:top_k].to(torch.int32).unsqueeze(0).to(device) for _ in range(layers)]
    weights_in = [torch.softmax(torch.randn(1, top_k, generator=generator), -1).to(device) for _ in range(layers)]
    expert_to_slot = {}
    for layer in range(layers):
        mapping = torch.full((experts,), -1, dtype=torch.int64)
        mapping[torch.randperm(experts, generator=generator)[: args.residents]] = torch.arange(args.residents)
        expert_to_slot[layer] = mapping.to(device)
    hot_caches = {layer: SimpleNamespace(expert_to_slot=expert_to_slot[layer]) for layer in range(layers)}

    features = PrefetchScoring.features_for("llapor")
    store = FeatureStore(specs=specs, features=features, max_rows=1, device=device, hidden_dtype=dtype)

    def taps():
        for layer in range(layers):
            store.write(layer, RouteFeature.ROUTER_INPUT, hidden_in[layer])
            store.write(layer, RouteFeature.TOPK_IDS, ids_in[layer])
            store.write(layer, RouteFeature.TOPK_WEIGHTS, weights_in[layer])

    taps()
    torch.cuda.synchronize()
    view = {
        (layer, feature): store.view(layer, feature, 1) for layer in range(layers) for feature in features
    }
    dense = {
        t: LlaporScorer(checkpoints[t], num_experts=experts, dtype=dtype, device=device) for t in targets
    }

    def inputs(t):
        s = source[t]
        return (
            view[(s, RouteFeature.ROUTER_INPUT)],
            view[(s, RouteFeature.TOPK_IDS)],
            view[(s, RouteFeature.TOPK_WEIGHTS)],
        )

    with torch.no_grad():
        h_buf = {t: ((inputs(t)[0].to(dtype) - dense[t].mean) @ dense[t].projection) for t in targets}
        u_buf = {}
        for t in targets:
            _, ids, w = inputs(t)
            mask = torch.zeros((1, experts), dtype=dtype, device=device)
            route = torch.zeros_like(mask)
            mask.scatter_(1, ids.long(), 1.0)
            route.scatter_(1, ids.long(), w.to(dtype))
            u_buf[t] = torch.cat((h_buf[t], mask, route), dim=-1)
        score_buf = {t: dense[t](*inputs(t)) for t in targets}

    bank16 = PrefetchCandidateBank(layer_ids=targets, width=16, device=device)
    top1_out = {
        t: (
            torch.full((1,), -1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.bool, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        for t in targets
    }
    recall = BudgetRecall(layer_ids=targets, budget=2, device=device)

    def pca():
        for t in targets:
            x = inputs(t)[0]
            (x.to(dtype) - dense[t].mean) @ dense[t].projection

    def dense_input():
        for t in targets:
            _, ids, w = inputs(t)
            h = h_buf[t]
            idl = ids.long()
            mask = torch.zeros((h.shape[0], experts), dtype=h.dtype, device=h.device)
            route = torch.zeros_like(mask)
            mask.scatter_(1, idl, 1.0)
            route.scatter_(1, idl, w.to(h.dtype))
            torch.cat((h, mask, route), dim=-1)

    def mlp():
        for t in targets:
            torch.sigmoid(dense[t].model(u_buf[t]).float())

    def scorer_full():
        for t in targets:
            dense[t](*inputs(t))

    def bank_write16():
        for t in targets:
            bank16.write(t, score_buf[t], expert_to_slot=expert_to_slot[t])

    def top1():
        for t in targets:
            select_prefetch_top1_cuda(score_buf[t], expert_to_slot[t], *top1_out[t])

    def recall_observe():
        for t in targets:
            recall.observe(
                target_layer=t,
                candidate_ids=bank16.ids_for(t),
                candidate_valid=bank16.valid_for(t),
                topk_ids=view[(t, RouteFeature.TOPK_IDS)],
                expert_to_slot=expert_to_slot[t],
            )

    def c_arm_hook(scoring):
        def step():
            store.after_write = scoring._on_write
            try:
                taps()
            finally:
                store.after_write = None

        return step

    def scoring_for(**kwargs):
        scoring = PrefetchScoring.from_checkpoints(
            predictor="llapor", checkpoints=checkpoints, specs=specs, store=store, hot_caches=hot_caches,
            width=16, budget=2, tau=0.95, dtype=dtype, device=device, pull_mode="off",
            shadow_recall=False, calibration=False, **kwargs,
        )
        store.after_write = None
        return scoring

    components = {
        "empty": lambda: store.view(0, RouteFeature.TOPK_IDS, 1).add_(0),
        "a_taps": taps,
        "b0_pca": pca,
        "b_dense_input": dense_input,
        "c_mlp": mlp,
        "bc_scorer_full": scorer_full,
        "d_bank16": bank_write16,
        "d_top1": top1,
        "e_recall": recall_observe,
        "C_arm_total_ref_bank16": c_arm_hook(scoring_for(fused_top1=False)),
        "C_arm_total_default": c_arm_hook(scoring_for()),
    }
    only = {name for name in args.only.split(",") if name}
    pool = torch.cuda.graph_pool_handle()
    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "targets": len(targets),
        "hidden": hidden,
        "groups": {g: sum(1 for c in checkpoints.values() if c.group == g) for g in ("outer", "middle")},
        "replays": args.replays,
    }
    print(json.dumps({"meta": meta}), flush=True)
    with torch.no_grad():
        for name, fn in components.items():
            if only and name not in only:
                continue
            result = _time_graph(fn, replays=args.replays, pool=pool)
            result["kernels"] = _kernel_count(fn)
            result["name"] = name
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
