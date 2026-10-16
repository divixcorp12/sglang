"""Export a trained next-verify head (next_verify_head.py train, variant B) and its PCA as one lane-head file.

The file keeps the training's standardization unfolded (``feature_mean``/``feature_scale``/``z_scale``), so serving
computes ``(f - fm) / fs`` exactly as ``next_verify_head.predict``. ``feature_scale`` is ``constant_safe(fs)``: a
feature constant in training (the draft odds in the B0 corpus) gets scale 1e9, and the metadata records the value
training saw (its mean), which serving feeds so the term is exactly zero. The PCA projection is bf16 by default (26 MB
for 40 layers at rank 64); ``--proj-dtype fp32`` keeps the training's own projection (52 MB).

Usage: export_lane_head.py --head HEAD_DIR --data DATA_DIR --out FILE [--proj-dtype bf16|fp32] [--layer-ids 0,...,39]
  HEAD_DIR holds L<LL>.npz and summary.json (variant B), DATA_DIR holds pca.npz and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from sglang.srt.layers.moe.lane_head import (  # noqa: E402
    CONSTANT_SCALE,
    FEATURES,
    LANE_HEAD_FORMAT_VERSION,
    PROJ_DTYPES,
    LaneHeadMeta,
    read_lane_head_file,
    write_lane_head_file,
)

# next_verify_head.constant_safe's threshold on the saved deviation.
_CONSTANT_STD = 1e-4


def constant_safe(fs: np.ndarray) -> np.ndarray:
    return np.where(fs < _CONSTANT_STD, CONSTANT_SCALE, fs).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proj-dtype", default="bf16", choices=sorted(PROJ_DTYPES))
    ap.add_argument("--layer-ids", default=",".join(str(i) for i in range(40)))
    ap.add_argument("--decay", type=float, default=0.98, help="next_verify_head.DECAY")
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
    if layers != len(layer_ids) or int(manifest["pca_rank"]) != rank:
        raise SystemExit(f"PCA covers {layers} layers at rank {rank}, --layer-ids names {len(layer_ids)}")
    experts = None
    cols = {k: [] for k in ("feature_mean", "feature_scale", "z_scale", "expert_bias", "dense_w", "dense_b", "z_w",
                            "expert_u")}
    constant: dict[str, float] = {}
    for L in range(layers):
        a = np.load(os.path.join(args.head, f"L{L:02d}.npz"))
        fm, fs, zs = a["fm"].astype(np.float32), constant_safe(a["fs"]), a["zs"].astype(np.float32)
        for j in np.flatnonzero(fs >= CONSTANT_SCALE * 0.5).tolist():
            value = float(fm[j])
            if constant.setdefault(FEATURES[j], value) != value:
                raise SystemExit(f"layer {L}: constant feature {FEATURES[j]} has value {value}, others {constant[FEATURES[j]]}")
        b_e = a["state_bias.weight"].reshape(-1).astype(np.float32)
        U = a["state_u.weight"].astype(np.float32)
        experts = experts or b_e.shape[0]
        w = a["state_dense.weight"].reshape(-1).astype(np.float32)
        if b_e.shape[0] != experts or U.shape != (experts, rank) or w.shape[0] != len(FEATURES):
            raise SystemExit(f"layer {L}: unexpected shapes bias {b_e.shape} U {U.shape} dense {w.shape}")
        cols["feature_mean"].append(fm)
        cols["feature_scale"].append(fs)
        cols["z_scale"].append(zs)
        cols["expert_bias"].append(b_e)
        cols["dense_w"].append(w)
        cols["dense_b"].append(a["state_dense.bias"].reshape(-1).astype(np.float32)[:1])
        cols["z_w"].append(a["state_z.weight"].reshape(-1).astype(np.float32))
        cols["expert_u"].append(U)
    if "odds" not in constant:
        raise SystemExit("the draft odds vary in the training data; the gather cannot serve them")
    tensors = {
        "pca_mean": torch.from_numpy(np.ascontiguousarray(mean, dtype=np.float32)),
        "pca_proj": torch.from_numpy(np.ascontiguousarray(proj, dtype=np.float32)).to(PROJ_DTYPES[args.proj_dtype]),
    }
    for name, parts in cols.items():
        tensors[name] = torch.from_numpy(np.concatenate(parts) if name == "dense_b" else np.stack(parts))
    try:
        commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = LaneHeadMeta(
        format_version=LANE_HEAD_FORMAT_VERSION, layer_ids=tuple(layer_ids), hidden=int(hidden), rank=int(rank),
        experts=int(experts), features=FEATURES, pca_proj_dtype=args.proj_dtype, score_decay=args.decay,
        constant_features=constant,
        training={"variant": summary["variant"], "epochs": summary["epochs"], "head_dir": os.path.abspath(args.head),
                  "data_dir": os.path.abspath(args.data), "train_shards": sorted(manifest["train"]),
                  "export_commit": commit,
                  "mean_dev_logloss": float(np.mean([v["dev_logloss"] for v in summary["layers"].values()]))},
    )
    write_lane_head_file(args.out, tensors=tensors, meta=meta)
    loaded = read_lane_head_file(args.out, layer_ids=layer_ids, hidden=hidden, experts=experts)
    print(f"{args.out}: {loaded.weight_bytes / 2**20:.1f} MiB of weights, layers {layers}, rank {rank}, "
          f"experts {experts}, pca_proj {args.proj_dtype}, constant {constant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
