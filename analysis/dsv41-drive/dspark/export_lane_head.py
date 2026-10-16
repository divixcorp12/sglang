"""Export a trained next-verify head (next_verify_head.py train, variant B) and its PCA as one lane-head file.

The training standardized the dense features (``fm``, ``fs``) and the PCA coordinates (``zs``); serving computes raw
features, so the scaling is folded here: ``dense_w = w / fs``, ``bias = dense_b - sum(w * fm / fs)``, ``z_w = a / zs``,
``expert_u = U / zs``. A feature constant in training (``fs`` = 1e9 after next_verify_head.constant_safe) folds to
weight 0. The PCA projection is stored in bf16 (26 MB for 40 layers at rank 64); everything else fp32.

Usage: export_lane_head.py --head HEAD_DIR --data DATA_DIR --out FILE [--layer-ids 0,1,...,39]
  HEAD_DIR holds L<LL>.npz and summary.json (variant B), DATA_DIR holds pca.npz and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from sglang.srt.layers.moe.lane_head import (  # noqa: E402
    FEATURES,
    LANE_HEAD_FORMAT_VERSION,
    LaneHeadMeta,
    read_lane_head_file,
    write_lane_head_file,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer-ids", default=",".join(str(i) for i in range(40)))
    ap.add_argument("--decay", type=float, default=0.98)
    args = ap.parse_args()
    with open(os.path.join(args.head, "summary.json")) as f:
        summary = json.load(f)
    if summary["variant"] != "B":
        raise SystemExit(f"the lane head needs variant B (per-expert directions), {args.head} is {summary['variant']}")
    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)
    if tuple(manifest["dense"]) != FEATURES:
        raise SystemExit(f"dense features {manifest['dense']} differ from lane_head.FEATURES {FEATURES}")
    pca = np.load(os.path.join(args.data, "pca.npz"))
    mean, proj = pca["mean"], pca["proj"]  # [L, H], [L, H, R]
    layer_ids = [int(i) for i in args.layer_ids.split(",")]
    layers, hidden, rank = proj.shape
    if layers != len(layer_ids):
        raise SystemExit(f"PCA covers {layers} layers, --layer-ids names {len(layer_ids)}")
    experts = None
    expert_bias, expert_u, bias, dense_w, z_w = [], [], [], [], []
    constant: dict[str, float] = {}
    for L in range(layers):
        a = np.load(os.path.join(args.head, f"L{L:02d}.npz"))
        fm, fs, zs = a["fm"].astype(np.float32), a["fs"].astype(np.float32), a["zs"].astype(np.float32)
        # a feature next_verify_head.constant_safe zeroed (deviation ~0 in training): serving feeds its mean
        for j in np.flatnonzero(fs >= 1e8).tolist():
            constant.setdefault(FEATURES[j], float(fm[j]))
            if abs(constant[FEATURES[j]] - float(fm[j])) > 1e-6:
                raise SystemExit(f"layer {L}: constant feature {FEATURES[j]} has value {fm[j]}, other layers {constant[FEATURES[j]]}")
        b_e = a["state_bias.weight"].reshape(-1).astype(np.float32)
        w = a["state_dense.weight"].reshape(-1).astype(np.float32)
        d_b = float(a["state_dense.bias"].reshape(-1)[0])
        z_a = a["state_z.weight"].reshape(-1).astype(np.float32)
        U = a["state_u.weight"].astype(np.float32)
        experts = experts or b_e.shape[0]
        if b_e.shape[0] != experts or U.shape != (experts, rank) or w.shape[0] != len(FEATURES):
            raise SystemExit(f"layer {L}: unexpected shapes bias {b_e.shape} U {U.shape} dense {w.shape}")
        expert_bias.append(b_e)
        dense_w.append(w / fs)
        bias.append(np.float32(d_b - float(np.sum(w * fm / fs))))
        z_w.append(z_a / zs)
        expert_u.append(U / zs[None, :])
    tensors = {
        "pca_mean": torch.from_numpy(np.ascontiguousarray(mean, dtype=np.float32)),
        "pca_proj": torch.from_numpy(np.ascontiguousarray(proj, dtype=np.float32)).to(torch.bfloat16),
        "expert_bias": torch.from_numpy(np.stack(expert_bias)),
        "expert_u": torch.from_numpy(np.stack(expert_u)),
        "bias": torch.from_numpy(np.asarray(bias, dtype=np.float32)),
        "dense_w": torch.from_numpy(np.stack(dense_w)),
        "z_w": torch.from_numpy(np.stack(z_w)),
    }
    meta = LaneHeadMeta(
        format_version=LANE_HEAD_FORMAT_VERSION, layer_ids=tuple(layer_ids), hidden=int(hidden), rank=int(rank),
        experts=int(experts), features=FEATURES, score_decay=args.decay, constant_features=constant,
        training={"variant": summary["variant"], "epochs": summary["epochs"], "head_dir": os.path.abspath(args.head),
                  "data_dir": os.path.abspath(args.data), "train_shards": sorted(manifest["train"]),
                  "mean_dev_logloss": float(np.mean([v["dev_logloss"] for v in summary["layers"].values()]))},
    )
    write_lane_head_file(args.out, tensors=tensors, meta=meta)
    loaded = read_lane_head_file(args.out, layer_ids=layer_ids, hidden=hidden, experts=experts)
    print(f"{args.out}: {loaded.device_bytes / 2**20:.1f} MiB of weights, layers {layers}, rank {rank}, experts {experts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
