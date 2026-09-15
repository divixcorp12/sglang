"""Serving bf16 scorers vs training fp32 forwards on real checkpoints and dev decode rows (CPU)."""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_prediction.training import llapor
from sglang.srt.layers.moe.expert_prediction.training.dataset import load_layer_rows, load_session_splits, split_mask
from sglang.srt.layers.moe.expert_prediction.training.metrics import recall_at_budget

ROWS = 4096


def _specs(capture_dir: Path) -> list[MoeLayerSpec]:
    header = json.loads((capture_dir / "capture.json").read_text())
    return [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", default="5,20,44")
    args = parser.parse_args()
    specs = _specs(args.capture_dir)
    splits = load_session_splits(args.sessions)
    llapor_models = load_prefetch_checkpoints(args.model_dir, predictor="llapor", specs=specs)
    apex_models = load_prefetch_checkpoints(args.model_dir, predictor="apex", specs=specs)
    experts = specs[0].num_experts
    report = {}
    cpu = torch.device("cpu")
    for target in (int(layer) for layer in args.layers.split(",")):
        pair = llapor_models[target]
        rows = load_layer_rows(
            args.capture_dir, splits, layer_id=pair.source_layer,
            features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS, RouteFeature.PRE_MIXER),
            next_layer_topk=target,
        )
        keep = (split_mask(rows, splits, "dev") & rows.is_decode).nonzero().flatten()[:ROWS]
        router_input = rows.features["router_input"][keep]
        topk_ids, topk_weights = rows.features["topk_ids"][keep], rows.features["topk_weights"][keep]
        labels = rows.features["next_topk_ids"][keep]
        with torch.no_grad():
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=pair.pca, num_experts=experts)
            full = torch.sigmoid(pair.model(u))
            half = LlaporScorer(pair, num_experts=experts, dtype=torch.bfloat16, device=cpu)(
                router_input.to(torch.bfloat16), topk_ids, topk_weights
            )
        report[f"llapor_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
        same = load_layer_rows(args.capture_dir, splits, layer_id=target,
                               features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS))
        keep = (split_mask(same, splits, "dev") & same.is_decode).nonzero().flatten()[:ROWS]
        layer = apex_models[target]
        with torch.no_grad():
            full = torch.softmax(layer.ranker(same.features["pre_mixer"][keep]), dim=-1)
            half = ApexScorer(layer, num_experts=experts, tau=1.0, dtype=torch.bfloat16, device=cpu)(
                same.features["pre_mixer"][keep].to(torch.bfloat16)
            )
        labels = same.features["topk_ids"][keep]
        report[f"apex_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
    print(json.dumps(report, indent=2))
    worst = max(abs(v["recall16_fp32"] - v["recall16_bf16"]) for v in report.values())
    raise SystemExit(0 if worst <= 0.005 else 1)


if __name__ == "__main__":
    main()
