#!/usr/bin/env python
"""Fit LLaPor next-layer predictors, one adjacent layer pair (L -> L+1) at a
time, resumable per pair. See docs/superpowers/plans/2026-09-14-llapor-gpu-only.md
sections 4-5 and the offline-training write-up for the exact contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.training import llapor, metrics
from sglang.srt.layers.moe.expert_prediction.training.dataset import (
    load_layer_rows,
    load_session_splits,
    split_mask,
)
from sglang.srt.layers.moe.expert_prediction.training.pca import fit_pca

DONE_NAME = "DONE"
_LR = {"outer": 1e-3, "middle": 2e-3}
_WD = {"outer": 1e-4, "middle": 1e-3}
_BUDGETS = (10, 12, 16, 24, 32)


def _sha256_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def _load_capture_header(capture_dir: Path) -> dict:
    return json.loads((capture_dir / "capture.json").read_text())


def _lr_at_epoch(epoch: int, base_lr: float, max_epochs: int, warmup_epochs: int = 5) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())


def _evaluate(model, u, next_topk_ids, num_experts, source_topk_ids) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(u)
    out = {}
    for budget in _BUDGETS:
        candidates = metrics.topk_candidates(logits, budget)
        out[f"llapor_recall@{budget}"] = metrics.recall_at_budget(candidates, next_topk_ids)
        same_cand = metrics.same_expert_baseline_candidates(source_topk_ids, budget, num_experts)
        out[f"same_expert_recall@{budget}"] = metrics.recall_at_budget(same_cand, next_topk_ids)
        pop_cand = metrics.popularity_baseline_candidates(next_topk_ids, num_experts, budget)
        out[f"popularity_recall@{budget}"] = metrics.recall_at_budget(pop_cand, next_topk_ids)
    model.train()
    return out


def train_pair(
    *,
    capture_dir: Path,
    splits,
    source_layer: int,
    num_experts: int,
    device: torch.device,
    out_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    group = llapor.layer_group(source_layer)
    rank = llapor.pca_rank_for_group(group)

    t_load = time.time()
    rows = load_layer_rows(
        capture_dir,
        splits,
        layer_id=source_layer,
        features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS),
        next_layer_topk=source_layer + 1,
    )
    train_mask = split_mask(rows, splits, "train")
    dev_mask = split_mask(rows, splits, "dev")
    test_mask = split_mask(rows, splits, "shifted_test")
    is_decode = rows.is_decode

    router_input = rows.features["router_input"].to(device)
    topk_ids = rows.features["topk_ids"].to(device)
    topk_weights = rows.features["topk_weights"].to(device)
    next_topk_ids = rows.features["next_topk_ids"].to(device)
    load_seconds = time.time() - t_load

    t_train = time.time()
    pca = fit_pca(router_input[train_mask.to(device)], rank=rank)
    q = llapor.expert_frequency_weights(next_topk_ids[train_mask.to(device)], num_experts).to(device)

    model = llapor.build_predictor(group, pca_rank=rank, num_experts=num_experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR[group], weight_decay=_WD[group])

    train_idx = train_mask.nonzero(as_tuple=True)[0].to(device)
    best_metric, best_state, epochs_without_improve = -1.0, None, 0
    history = []
    for epoch in range(max_epochs):
        t_epoch = time.time()
        lr = _lr_at_epoch(epoch, _LR[group], max_epochs)
        for group_param in optimizer.param_groups:
            group_param["lr"] = lr
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        epoch_loss = 0.0
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            u = llapor.encode_features(
                router_input[idx], topk_ids[idx], topk_weights[idx], pca=pca, num_experts=num_experts
            )
            y = llapor.multihot_labels(next_topk_ids[idx], num_experts)
            logits = model(u)
            loss = llapor.llapor_loss(logits, y, group=group, q=q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * idx.numel()
        epoch_loss /= perm.numel()

        dev_idx = dev_mask.nonzero(as_tuple=True)[0].to(device)
        u_dev = llapor.encode_features(
            router_input[dev_idx], topk_ids[dev_idx], topk_weights[dev_idx], pca=pca, num_experts=num_experts
        )
        dev_metrics = _evaluate(model, u_dev, next_topk_ids[dev_idx], num_experts, topk_ids[dev_idx])
        dev_metrics["train_loss"] = epoch_loss
        history.append({"epoch": epoch, **dev_metrics})
        print(
            f"layer {source_layer} ({group}): epoch {epoch} loss={epoch_loss:.4f} "
            f"dev_recall@16={dev_metrics['llapor_recall@16']:.4f} in {time.time() - t_epoch:.1f}s",
            flush=True,
        )
        selection_metric = dev_metrics["llapor_recall@16"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    model.load_state_dict(best_state)
    train_seconds = time.time() - t_train

    t_eval = time.time()
    report = {
        "group": group,
        "pca_rank": rank,
        "epochs_trained": len(history),
        "history": history,
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
    }
    for name, mask in (("dev", dev_mask), ("shifted_test", test_mask)):
        mask_dev = mask.to(device)
        for phase_name, phase_mask in (
            ("decode", mask_dev & is_decode.to(device)),
            ("prefill", mask_dev & ~is_decode.to(device)),
        ):
            idx = phase_mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            u = llapor.encode_features(
                router_input[idx], topk_ids[idx], topk_weights[idx], pca=pca, num_experts=num_experts
            )
            report[f"{name}_{phase_name}"] = _evaluate(
                model, u, next_topk_ids[idx], num_experts, topk_ids[idx]
            )
    report["eval_seconds"] = time.time() - t_eval

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    torch.save({"mean": pca.mean, "components": pca.components}, out_dir / "pca.pt")
    manifest = {
        "architecture": {"group": group, "pca_rank": rank, "num_experts": num_experts},
        "pca_stats": {"explained_variance": pca.explained_variance.tolist()},
        "grouping": {"source_layer": source_layer, "target_layer": source_layer + 1, "group": group},
        "split_session_hashes": {
            split_name: hashlib.sha256(
                json.dumps(sorted(sid for sid, s in splits.split_of_session.items() if s == split_name)).encode()
            ).hexdigest()
            for split_name in ("train", "dev", "shifted_test")
        },
        "capture_dir": str(capture_dir),
        "seed": seed,
        "metrics": report,
        "tensor_checksums": {"model": _sha256_state_dict(best_state)},
        "created_ns": time.time_ns(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / DONE_NAME).write_text("ok")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="source layers; default all")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    header = _load_capture_header(args.capture_dir)
    num_experts = header["layers"][0]["num_experts"]
    num_layers = len(header["layers"])
    layers = args.layers if args.layers is not None else list(range(num_layers - 1))
    splits = load_session_splits(args.sessions)
    device = torch.device(args.device)

    for source_layer in layers:
        out_dir = args.out_dir / "llapor" / f"pair-{source_layer:02d}"
        if (out_dir / DONE_NAME).exists():
            print(f"skip layer {source_layer}: already done")
            continue
        t0 = time.time()
        report = train_pair(
            capture_dir=args.capture_dir,
            splits=splits,
            source_layer=source_layer,
            num_experts=num_experts,
            device=device,
            out_dir=out_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        print(
            f"layer {source_layer}: {report['epochs_trained']} epochs, "
            f"load={report['load_seconds']:.1f}s train={report['train_seconds']:.1f}s "
            f"eval={report['eval_seconds']:.1f}s total={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
