#!/usr/bin/env python3
"""Execute real LLaPor/APEX scoring and candidate selection on captured features.

``--capture-dir`` is the canonical CAPTURE=1 directory (header.json,
manifest.jsonl, safetensors shards). It intentionally does not synthesize
scores: each timing invokes the production checkpoint scorer then selection.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch


def load_captured_features(capture_dir: Path, *, source_layer: int, target_layer: int) -> dict[str, torch.Tensor]:
    """Load one representative canonical CAPTURE=1 row with LLaPor source mapping."""
    from sglang.srt.layers.moe.expert_prediction.capture_reader import load_shard, read_manifest
    from sglang.srt.layers.moe.expert_prediction.capture_schema import feature_key
    from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
    entries = read_manifest(capture_dir)
    if not entries:
        raise ValueError("capture has no completed safetensors shard")
    tensors = load_shard(capture_dir, entries[0]["shard"]).tensors
    def get(layer: int, feature: RouteFeature) -> torch.Tensor:
        return tensors[feature_key(layer, feature)][:1].cuda()
    return {
        "router_input": get(source_layer, RouteFeature.ROUTER_INPUT),
        "topk_ids": get(source_layer, RouteFeature.TOPK_IDS),
        "topk_weights": get(source_layer, RouteFeature.TOPK_WEIGHTS),
        "pre_mixer": get(target_layer, RouteFeature.PRE_MIXER),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=("llapor", "apex"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture-dir", required=True)
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

    capture_dir = Path(args.capture_dir)
    header = json.loads((capture_dir / "header.json").read_text())
    specs = [MoeLayerSpec(**shape) for shape in header["layers"]]
    checkpoints = load_prefetch_checkpoints(Path(args.model_dir), predictor=args.predictor, specs=specs)
    by_layer = {spec.layer_id: spec for spec in specs}
    bank = PrefetchCandidateBank(layer_ids=list(checkpoints), width=args.width, device=torch.device("cuda"))
    for target, checkpoint in checkpoints.items():
        source = checkpoint.source_layer if args.predictor == "llapor" else target
        feature = load_captured_features(capture_dir, source_layer=source, target_layer=target)
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
