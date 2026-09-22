"""Build a DSpark draft-only checkpoint dir from the DSV4.1 EXL3 export.

The truncated target dir drops every ``mtp.*`` tensor; the draft loader
(``DeepseekV4ForCausalLMDSpark``) reads only those, so it gets a dir holding just
them, with the target's config and ``num_nextn_predict_layers`` restored from the
export's ``text_config``. Reads only the shards that hold ``mtp.*`` keys.
"""

import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def make_draft_dir(src: str, target_dir: str, out: str) -> int:
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("mtp."):
            by_shard.setdefault(shard, []).append(key)
    tensors = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(src, shard), "pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    src_cfg = json.load(open(os.path.join(src, "config.json")))
    nextn = src_cfg.get("text_config", src_cfg)["num_nextn_predict_layers"]
    cfg = json.load(open(os.path.join(target_dir, "config.json")))
    if "text_config" in cfg:
        cfg["text_config"]["num_nextn_predict_layers"] = nextn
    else:
        cfg["num_nextn_predict_layers"] = nextn
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump({"weight_map": {k: "model.safetensors" for k in sorted(tensors)}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    return len(tensors)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(make_draft_dir(a.src, a.target_dir, a.out), "tensors")
