"""Check that gate weights times captured router input reproduce the captured top-k ids."""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.capture_reader import load_shard, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    header = json.loads((args.capture_dir / "capture.json").read_text())
    shards = [
        load_shard(args.capture_dir, entry["shard"]) for entry in read_manifest(args.capture_dir)
    ]
    layer_keys = {key for shard in shards for key in shard.tensors if key.startswith("layer.")}
    tensors = {
        key: torch.cat([shard.tensors[key] for shard in shards if key in shard.tensors])
        for key in layer_keys
    }
    weight_map = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    results = {}
    for layer in header["layers"]:
        layer_id, top_k = layer["layer_id"], layer["top_k"]
        pattern = re.compile(rf"(^|\.)layers\.{layer_id}\.mlp\.gate\.weight$")
        # MTP checkpoints also carry mtp.layers.N.mlp.gate.weight, which is not a decoder layer.
        keys = [key for key in weight_map if pattern.search(key) and not key.startswith("mtp.")]
        if len(keys) != 1:
            results[layer_id] = f"gate key not unique: {keys}"
            continue
        with safe_open(str(args.model_dir / weight_map[keys[0]]), framework="pt") as handle:
            weight = handle.get_tensor(keys[0])
        if not weight.is_floating_point():
            results[layer_id] = f"gate {keys[0]} is {weight.dtype}"
            continue
        router_input = tensors[f"layer.{layer_id}.router_input"].float()
        captured = tensors[f"layer.{layer_id}.topk_ids"].long()
        predicted = torch.topk(router_input @ weight.float().T, top_k, dim=-1).indices
        agreement = (predicted.unsqueeze(-1) == captured.unsqueeze(-2)).any(-1).float().mean()
        results[layer_id] = round(float(agreement), 5)
    print(json.dumps({"gate_key_example": keys, "agreement": results}))


if __name__ == "__main__":
    main()
