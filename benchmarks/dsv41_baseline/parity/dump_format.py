"""On-disk format of a parity dump, shared by dump_logits.py (writer) and compare.py (reader). NumPy only.

A dump is a directory holding `logits.npz` and `manifest.json`. For each prompt id `p`, logits.npz has:
  p.tokens      int32 [N]      the teacher-forced continuation token ids
  p.chosen_lp   float32 [N]    logprob of tokens[j] given the prompt and tokens[:j]
  p.top_ids     int32 [N, K]   top-K ids of the distribution that predicts tokens[j], best first, -1 padded
  p.top_lp      float32 [N, K] their logprobs, -inf padded
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TOP_K = 20
NPZ_NAME = "logits.npz"
MANIFEST_NAME = "manifest.json"
FIELDS = ("tokens", "chosen_lp", "top_ids", "top_lp")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dump(*, out_dir: Path, manifest: dict, prompts: dict[str, dict[str, np.ndarray]]) -> None:
    """prompts maps prompt id to a dict with the FIELDS arrays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prompt_id, fields in prompts.items():
        for name in FIELDS:
            arrays[f"{prompt_id}.{name}"] = fields[name]
    np.savez_compressed(out_dir / NPZ_NAME, **arrays)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_dump(path: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    prompts: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path / NPZ_NAME) as archive:
        for key in archive.files:
            prompt_id, name = key.rsplit(".", 1)
            prompts.setdefault(prompt_id, {})[name] = archive[key]
    for prompt_id, fields in prompts.items():
        missing = [name for name in FIELDS if name not in fields]
        if missing:
            raise ValueError(f"{path}: prompt {prompt_id} lacks {missing}")
    return manifest, prompts
