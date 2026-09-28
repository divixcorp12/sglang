"""Offline prefetch gate: recall of non-resident native experts within a budget (CPU).

Reports budget recall per layer, split by forward.kind and mixer kind, with an oracle
(perfect recall/precision) upper bound. For LLaPor's next-layer target it also prices an
in-graph side-stream copy: the rows that land inside the inter-layer window and the
in-graph copy time their hits avoid.
"""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
    IN_GRAPH_FIXED_MS,
    IN_GRAPH_ROW_MS,
    budget_hits,
    oracle_hits,
    side_stream_ready_rows,
)
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import (
    load_prefetch_checkpoints,
)
from sglang.srt.layers.moe.expert_prediction.serving.scorers import (
    ApexScorer,
    LlaporScorer,
)
from sglang.srt.layers.moe.expert_prediction.training.dataset import (
    load_layer_rows,
    load_session_splits,
    split_mask,
)

# E28 p50 windows (copy(L) end -> copy(L+1) start) by the target layer's mixer kind.
WINDOW_MS = {"full_attention": 0.290, "linear_attention": 0.267, "unknown": 0.267}
BATCH = 4096
FORWARD_KINDS = ("decode", "prefill")


def _mixer_kinds(model_config: Path, layer_ids) -> dict[int, str]:
    config = json.loads(model_config.read_text())
    config = config.get("text_config", config)
    layer_types = config.get("layer_types") or []
    return {
        layer: layer_types[layer] if layer < len(layer_types) else "unknown"
        for layer in layer_ids
    }


def _popularity_scores(rows, splits, num_experts: int) -> torch.Tensor:
    """Static per-layer expert frequency prior from the train split, broadcast to every row."""
    native_train = rows.features["topk_ids"][split_mask(rows, splits, "train")]
    counts = torch.zeros(num_experts, dtype=torch.float32)
    counts.index_add_(0, native_train.reshape(-1), torch.ones(native_train.numel()))
    return counts.unsqueeze(0).expand(rows.features["topk_ids"].shape[0], -1)


def _scores(predictor, checkpoint, rows, experts):
    cpu = torch.device("cpu")
    if predictor == "llapor":
        scorer = LlaporScorer(
            checkpoint, num_experts=experts, dtype=torch.float32, device=cpu
        )
        inputs = (
            rows.features["router_input"],
            rows.features["topk_ids"],
            rows.features["topk_weights"],
        )
    else:
        scorer = ApexScorer(
            checkpoint, num_experts=experts, tau=1.0, dtype=torch.float32, device=cpu
        )
        inputs = (rows.features["pre_mixer"],)
    with torch.no_grad():
        return torch.cat(
            [
                scorer(*(x[i : i + BATCH] for x in inputs))
                for i in range(0, inputs[0].shape[0], BATCH)
            ]
        )


def _recall(*, scores, native, resident, budgets, is_oracle):
    """Per-budget entries: misses, hits and budget recall."""
    out = {}
    for budget in budgets:
        if is_oracle:
            missed = (~resident.gather(1, native.long())).sum(dim=1)
            hits = oracle_hits(missed, budget)
        else:
            missed, hits = budget_hits(scores, native, resident, budget)
        out[f"b{budget}"] = {
            "rows": int(missed.numel()),
            "missed_per_token": float(missed.double().mean()),
            "hits_per_token": float(hits.double().mean()),
            "budget_recall": float(hits.sum() / missed.sum().clamp(min=1)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--predictor", choices=("llapor", "apex", "oracle", "popularity"), required=True
    )
    parser.add_argument("--budgets", default="1,2,3,4,6,8,10,16,32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    header = json.loads((args.capture_dir / "capture.json").read_text())
    specs = [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]
    splits = load_session_splits(args.sessions)
    kinds = _mixer_kinds(args.model_config, [spec.layer_id for spec in specs])
    budgets = [int(b) for b in args.budgets.split(",")]
    experts = specs[0].num_experts

    is_oracle = args.predictor == "oracle"
    is_popularity = args.predictor == "popularity"
    # Popularity, like the oracle, is not tied to a checkpoint set: it scores every target layer
    # from a static train-split frequency prior instead of a loaded model.
    models = (
        None
        if is_oracle or is_popularity
        else load_prefetch_checkpoints(
            args.model_dir, predictor=args.predictor, specs=specs
        )
    )
    # The oracle is priced under every target layer, since it isn't tied to a checkpoint set.
    layer_ids = (
        sorted(spec.layer_id for spec in specs)
        if is_oracle or is_popularity
        else sorted(models)
    )

    per_layer = {}
    for target in layer_ids:
        checkpoint = None if is_oracle or is_popularity else models[target]
        if is_oracle or is_popularity or args.predictor == "apex":
            source_layer = target
            rows = load_layer_rows(
                args.capture_dir,
                splits,
                layer_id=target,
                features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS),
                residency_layer=target,
            )
            native_all = rows.features["topk_ids"]
        else:
            source_layer = checkpoint.source_layer
            rows = load_layer_rows(
                args.capture_dir,
                splits,
                layer_id=checkpoint.source_layer,
                features=(
                    RouteFeature.ROUTER_INPUT,
                    RouteFeature.TOPK_IDS,
                    RouteFeature.TOPK_WEIGHTS,
                ),
                next_layer_topk=target,
                residency_layer=target,
            )
            native_all = rows.features["next_topk_ids"]
        if is_oracle:
            scores_all = None
        elif is_popularity:
            scores_all = _popularity_scores(rows, splits, experts)
        else:
            scores_all = _scores(args.predictor, checkpoint, rows, experts)
        window = WINDOW_MS[kinds[target]]
        per_layer[target] = {"mixer": kinds[target], "window_ms": window}
        for split in ("dev", "shifted_test"):
            for kind in FORWARD_KINDS:
                keep = (
                    (
                        split_mask(rows, splits, split)
                        & (rows.is_decode == (kind == "decode"))
                    )
                    .nonzero()
                    .flatten()
                )
                if keep.numel() == 0:
                    continue
                native = native_all[keep]
                resident = rows.resident[keep]
                scores = None if is_oracle else scores_all[keep]
                recall = _recall(
                    scores=scores,
                    native=native,
                    resident=resident,
                    budgets=budgets,
                    is_oracle=is_oracle,
                )
                for key, entry in recall.items():
                    per_layer[target][f"{split}_{kind}_{key}"] = entry
                if not is_oracle:
                    # In-graph side-stream copy, LLaPor's next-layer target only.
                    if args.predictor == "llapor":
                        source_entry_key = f"{split}_{kind}_b{budgets[0]}"
                        source_entry = per_layer.get(source_layer, {}).get(
                            source_entry_key, {}
                        )
                        source_copy_ms = (
                            IN_GRAPH_FIXED_MS
                            + source_entry.get("missed_per_token", 0.0)
                            * IN_GRAPH_ROW_MS
                        )
                        for budget in budgets:
                            entry = per_layer[target][f"{split}_{kind}_b{budget}"]
                            for label, side_window in (
                                ("gap", window),
                                ("overlap", window + source_copy_ms),
                            ):
                                ready = side_stream_ready_rows(
                                    budget=budget, window_ms=side_window
                                )
                                side_hits = (
                                    budget_hits(scores, native, resident, ready)[1]
                                    if ready
                                    else torch.zeros(scores.shape[0])
                                )
                                entry[f"side_ready_rows_{label}"] = ready
                                entry[f"side_saving_ms_{label}"] = float(
                                    side_hits.double().mean() * IN_GRAPH_ROW_MS
                                )
        print(json.dumps({"layer": target, "mixer": kinds[target]}), flush=True)

    totals = {}
    for split in ("dev", "shifted_test"):
        for kind in FORWARD_KINDS:
            for budget in budgets:
                key = f"{split}_{kind}_b{budget}"
                entries = [v[key] for v in per_layer.values() if key in v]
                if not entries:
                    continue
                total_missed = sum(e["missed_per_token"] for e in entries)
                total = {
                    "budget_recall": sum(e["hits_per_token"] for e in entries)
                    / max(total_missed, 1e-9)
                }
                if args.predictor == "llapor":
                    for label in ("gap", "overlap"):
                        total[f"side_saving_ms_per_token_{label}"] = sum(
                            e[f"side_saving_ms_{label}"] for e in entries
                        )
                totals[key] = total

    # Also split by mixer kind (linear vs full attention), decode rows only.
    by_mixer = {}
    for mixer in ("linear_attention", "full_attention"):
        layers_of_kind = [v for v in per_layer.values() if v["mixer"] == mixer]
        for budget in budgets:
            key = f"shifted_test_decode_b{budget}"
            entries = [v[key] for v in layers_of_kind if key in v]
            if not entries:
                continue
            total_missed = sum(e["missed_per_token"] for e in entries)
            by_mixer.setdefault(mixer, {})[f"b{budget}"] = {
                "budget_recall": sum(e["hits_per_token"] for e in entries)
                / max(total_missed, 1e-9),
            }

    args.out.write_text(
        json.dumps(
            {
                "predictor": args.predictor,
                "totals": totals,
                "by_mixer_kind": by_mixer,
                "layers": per_layer,
            },
            indent=2,
        )
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
