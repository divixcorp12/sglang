#!/usr/bin/env python3
"""Execute real LLaPor/APEX scoring and candidate selection on captured features.

``--captured-features`` is a ``torch.save`` dictionary with ``layer_specs``
(``layer -> {num_experts, top_k, hidden_size}``) and ``features`` (`target ->
captured tensors`).  It intentionally does not synthesize scores: each timing
invokes the production checkpoint scorer followed by ``PrefetchCandidateBank``.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=("llapor", "apex"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--captured-features", required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
    from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
    from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer

    capture = torch.load(args.captured_features, map_location="cuda", weights_only=True)
    specs = [MoeLayerSpec(layer_id=int(layer), **shape) for layer, shape in capture["layer_specs"].items()]
    checkpoints = load_prefetch_checkpoints(Path(args.model_dir), predictor=args.predictor, specs=specs)
    by_layer = {spec.layer_id: spec for spec in specs}
    bank = PrefetchCandidateBank(layer_ids=list(checkpoints), width=args.width, device=torch.device("cuda"))
    for target, checkpoint in checkpoints.items():
        feature = capture["features"][str(target)]
        scorer = (LlaporScorer(checkpoint, num_experts=by_layer[target].num_experts, dtype=torch.bfloat16, device=torch.device("cuda")) if args.predictor == "llapor" else ApexScorer(checkpoint, num_experts=by_layer[target].num_experts, tau=args.tau, dtype=torch.bfloat16, device=torch.device("cuda")))
        residency = torch.full((by_layer[target].num_experts,), -1, dtype=torch.int32, device="cuda")
        def score_and_select():
            scores = (scorer(feature["router_input"], feature["topk_ids"], feature["topk_weights"]) if args.predictor == "llapor" else scorer(feature["pre_mixer"]))
            bank.write(target, scores, expert_to_slot=residency)
        for _ in range(10): score_and_select()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        for start, end in zip(starts, ends):
            start.record(); score_and_select(); end.record()
        torch.cuda.synchronize()
        print(json.dumps({"predictor": args.predictor, "target_layer": target, "scorer_plus_selection_ms": statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)), "width": args.width}, sort_keys=True))


if __name__ == "__main__":
    main()
