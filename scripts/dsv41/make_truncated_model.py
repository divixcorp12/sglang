"""Build a truncated DeepSeek V4.1 EXL3 model dir: the first N decoder layers + head.

The dir holds an edited config.json, a filtered index and symlinks to the
original shards and tokenizer files; the loader skips the other layers' tensors.
"""

import argparse
import copy
import json
import os

# Real text_config key names (Task 7 Step 1, verified against
# /mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/config.json on divix01).
_KEYS = {
    "layers": "num_hidden_layers",
    "ratios": "compress_ratios",
    "kv_sources": "kv_source_layer_ids",
    "index_sources": "index_source_layer_ids",
    "candidate": "candidate_source_layer_id",
    "engram_ids": "engram_layer_ids",
    "engram_rows": "engram_num_embeddings",
    "nextn": "num_nextn_predict_layers",
}
_DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
_COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "quantization_config.json",
    "generation_config.json",
)


def truncate_text_config(text_config: dict, n_layers: int) -> dict:
    t = copy.deepcopy(text_config)
    t[_KEYS["layers"]] = n_layers
    t[_KEYS["ratios"]] = t[_KEYS["ratios"]][:n_layers]
    t[_KEYS["kv_sources"]] = [i for i in t[_KEYS["kv_sources"]] if i < n_layers]
    t[_KEYS["index_sources"]] = [i for i in t[_KEYS["index_sources"]] if i < n_layers]
    if t[_KEYS["candidate"]] >= n_layers:
        t[_KEYS["candidate"]] = -1
    kept = [(i, r) for i, r in zip(t[_KEYS["engram_ids"]], t[_KEYS["engram_rows"]]) if i < n_layers]
    t[_KEYS["engram_ids"]] = [i for i, _ in kept]
    t[_KEYS["engram_rows"]] = [r for _, r in kept]
    t[_KEYS["nextn"]] = 0
    return t


def select_weight_map(weight_map: dict, n_layers: int) -> dict:
    out = {}
    for name, shard in weight_map.items():
        if name.startswith(_DROP_PREFIXES):
            continue
        if name.startswith("layers.") and int(name.split(".")[1]) >= n_layers:
            continue
        out[name] = shard
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--engram-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    config["text_config"] = truncate_text_config(config["text_config"], args.n_layers)
    config["engram_table_dir"] = args.engram_dir
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    index["weight_map"] = select_weight_map(index["weight_map"], args.n_layers)
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    for name in sorted(set(index["weight_map"].values())) + [
        n for n in _COPY_FILES if os.path.exists(os.path.join(args.src, n))
    ]:
        link = os.path.join(args.dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(args.src, name), link)
    print(f"{args.dst}: {len(index['weight_map'])} tensors in {len(set(index['weight_map'].values()))} shards")


if __name__ == "__main__":
    main()
