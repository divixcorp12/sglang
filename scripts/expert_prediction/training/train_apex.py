#!/usr/bin/env python
"""Fit APEX same-layer rankers and ordinal CDF calibration, one layer at a
time, resumable per layer. See docs/superpowers/plans/2026-09-14-apex-gpu-only.md
section 4 and the offline-training write-up for the exact contract.

Teacher distribution: softmax(router_input @ gate.weight.T), matching this
model's TopK scoring_func="softmax" (no grouped-topk/correction bias); see
scripts/expert_prediction/check-capture-gate-topk.py for the same gate lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.training import apex, metrics
from sglang.srt.layers.moe.expert_prediction.training.dataset import (
    apex_subset_mask,
    carve_apex_train_subsets,
    load_layer_rows,
    load_session_splits,
    split_mask,
)

DONE_NAME = "DONE"
_LR = 1e-3
_WD = 1e-4
_TAUS = (0.90, 0.95, 0.99)


def _sha256_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def _load_capture_header(capture_dir: Path) -> dict:
    return json.loads((capture_dir / "capture.json").read_text())


def _load_gate_weight(model_dir: Path, layer_id: int) -> torch.Tensor:
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    pattern = re.compile(rf"(^|\.)layers\.{layer_id}\.mlp\.gate\.weight$")
    keys = [key for key in weight_map if pattern.search(key) and not key.startswith("mtp.")]
    if len(keys) != 1:
        raise ValueError(f"layer {layer_id}: gate key not unique: {keys}")
    with safe_open(str(model_dir / weight_map[keys[0]]), framework="pt") as handle:
        return handle.get_tensor(keys[0])


def _lr_at_epoch(epoch: int, base_lr: float, max_epochs: int, warmup_epochs: int = 5) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())


def _train_ranker(
    *, router_input, gate_weight, train_idx, dev_idx, num_experts, device, max_epochs, patience, batch_size, seed
):
    torch.manual_seed(seed)
    model = apex.Ranker(hidden_size=router_input.shape[1], num_experts=num_experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WD)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(max_epochs):
        lr = _lr_at_epoch(epoch, _LR, max_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            teacher = apex.teacher_probabilities(router_input[idx], gate_weight)
            loss = apex.ranker_kl_loss(model(router_input[idx]), teacher)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            teacher_dev = apex.teacher_probabilities(router_input[dev_idx], gate_weight)
            dev_loss = float(apex.ranker_kl_loss(model(router_input[dev_idx]), teacher_dev))
        if dev_loss < best_loss:
            best_loss, best_state, stale = dev_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_loss


def _fit_cdf(*, model, router_input, topk_ids, cdf_idx, num_experts, top_k, device, max_epochs, batch_size, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        logits = model(router_input[cdf_idx])
        delta_star = apex.oracle_delta(logits, topk_ids[cdf_idx])
    num_depths = num_experts - top_k + 1
    cdf = apex.OrdinalCDF(hidden_size=router_input.shape[1], num_depths=num_depths).to(device)
    optimizer = torch.optim.AdamW(cdf.parameters(), lr=1e-3, weight_decay=0.0)
    x = router_input[cdf_idx]
    for _epoch in range(max_epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            loss = apex.cdf_loss(cdf(x[idx]), delta_star[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return cdf, num_depths


def _calibrate(*, model, cdf, router_input, topk_ids, calib_idx, num_depths):
    with torch.no_grad():
        rank_logits = model(router_input[calib_idx])
        delta_star = apex.oracle_delta(rank_logits, topk_ids[calib_idx])
        cdf_logits = cdf(router_input[calib_idx])
    result = {}
    for tau in _TAUS:
        chosen_depth = apex.select_depth(cdf_logits, tau, num_depths - 1)
        result[str(tau)] = {
            "empirical_full_set_coverage": metrics.full_set_coverage(chosen_depth, delta_star),
            "mean_requested_depth": float(chosen_depth.float().mean()),
        }
    return result


def train_layer(
    *,
    capture_dir: Path,
    model_dir: Path,
    splits,
    layer_id: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    out_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> dict:
    rows = load_layer_rows(
        capture_dir,
        splits,
        layer_id=layer_id,
        features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS),
    )
    subset_of_session = carve_apex_train_subsets(splits, seed=seed)
    is_decode = rows.is_decode
    router_input = rows.features["pre_mixer"].to(device)
    topk_ids = rows.features["topk_ids"].to(device)

    dev_mask = split_mask(rows, splits, "dev")
    test_mask = split_mask(rows, splits, "shifted_test")
    train_subset = apex_subset_mask(rows, splits, subset_of_session, "ranker_train")
    cdf_subset = apex_subset_mask(rows, splits, subset_of_session, "cdf_fit")
    calib_subset = apex_subset_mask(rows, splits, subset_of_session, "calibration")

    gate_weight = _load_gate_weight(model_dir, layer_id).to(device)

    train_idx = train_subset.nonzero(as_tuple=True)[0].to(device)
    dev_idx = dev_mask.nonzero(as_tuple=True)[0].to(device)
    model, dev_kl = _train_ranker(
        router_input=router_input,
        gate_weight=gate_weight,
        train_idx=train_idx,
        dev_idx=dev_idx,
        num_experts=num_experts,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
    )

    cdf_idx = cdf_subset.nonzero(as_tuple=True)[0].to(device)
    cdf, num_depths = _fit_cdf(
        model=model,
        router_input=router_input,
        topk_ids=topk_ids,
        cdf_idx=cdf_idx,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        max_epochs=20,
        batch_size=batch_size,
        seed=seed,
    )

    calib_idx = calib_subset.nonzero(as_tuple=True)[0].to(device)
    calibration = _calibrate(
        model=model, cdf=cdf, router_input=router_input, topk_ids=topk_ids,
        calib_idx=calib_idx, num_depths=num_depths,
    )

    report = {"dev_kl": dev_kl, "calibration": calibration, "coverage": {}}
    for name, mask in (("dev", dev_mask), ("shifted_test", test_mask)):
        for phase_name, phase_mask in (
            ("decode", mask & is_decode), ("prefill", mask & ~is_decode),
        ):
            idx = phase_mask.nonzero(as_tuple=True)[0].to(device)
            if idx.numel() == 0:
                continue
            with torch.no_grad():
                rank_logits = model(router_input[idx])
            report["coverage"][f"{name}_{phase_name}"] = {
                f"top10_coverage@{depth}": metrics.recall_at_budget(
                    metrics.topk_candidates(rank_logits, depth), topk_ids[idx]
                )
                for depth in (10, 16, 24, 32)
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    ranker_state = model.state_dict()
    torch.save(ranker_state, out_dir / "ranker.pt")
    torch.save(cdf.state_dict(), out_dir / "cdf.pt")
    manifest = {
        "architecture": {"hidden_size": router_input.shape[1], "num_experts": num_experts, "top_k": top_k},
        "layer_id": layer_id,
        "gate_transform": "softmax(router_input @ gate.weight.T)",
        "split_session_hashes": {
            split_name: hashlib.sha256(
                json.dumps(sorted(sid for sid, s in splits.split_of_session.items() if s == split_name)).encode()
            ).hexdigest()
            for split_name in ("train", "dev", "shifted_test")
        },
        "capture_dir": str(capture_dir),
        "model_dir": str(model_dir),
        "seed": seed,
        "metrics": report,
        "tensor_checksums": {"ranker": _sha256_state_dict(ranker_state)},
        "created_ns": time.time_ns(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / DONE_NAME).write_text("ok")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    header = _load_capture_header(args.capture_dir)
    num_experts = header["layers"][0]["num_experts"]
    top_k = header["layers"][0]["top_k"]
    num_layers = len(header["layers"])
    layers = args.layers if args.layers is not None else list(range(num_layers))
    splits = load_session_splits(args.sessions)
    device = torch.device(args.device)

    for layer_id in layers:
        out_dir = args.out_dir / "apex" / f"layer-{layer_id:02d}"
        if (out_dir / DONE_NAME).exists():
            print(f"skip layer {layer_id}: already done")
            continue
        t0 = time.time()
        report = train_layer(
            capture_dir=args.capture_dir,
            model_dir=args.model_dir,
            splits=splits,
            layer_id=layer_id,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            out_dir=out_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"layer {layer_id}: dev_kl={report['dev_kl']:.4f} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
