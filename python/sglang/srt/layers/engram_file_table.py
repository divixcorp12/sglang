"""A read-only Engram hash table served from its safetensors shard via np.memmap.

The layer-1 table alone is 101.5 GB, more than the host RAM budget, and a batch
touches 24 rows per token. Rows are gathered on the CPU (the page cache does the
caching), copied to the device and dequantized there. Eager only.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from sglang.srt.layers.moe.exl3_expert_layout import read_safetensors_header


class EngramFileTable:
    def __init__(
        self, path: str, weight_key: str, scale_key: str, num_embeddings: int, dim: int, block: int = 32
    ):
        data_start, header = read_safetensors_header(path)
        w, s = header[weight_key], header[scale_key]
        if (w["dtype"], s["dtype"]) != ("F8_E4M3", "F8_E8M0"):
            raise ValueError(f"{path}: expected F8_E4M3/F8_E8M0, got {w['dtype']}/{s['dtype']}")
        if w["shape"] != [num_embeddings, dim] or s["shape"] != [num_embeddings, dim // block]:
            raise ValueError(
                f"{path}: rows/shape {w['shape']} {s['shape']} != [{num_embeddings}, {dim}]"
            )
        self.dim, self.block = dim, block
        self.weight = np.memmap(path, np.uint8, "r", data_start + w["data_offsets"][0], (num_embeddings, dim))
        self.scale = np.memmap(
            path, np.uint8, "r", data_start + s["data_offsets"][0], (num_embeddings, dim // block)
        )

    @classmethod
    def open(cls, table_dir: str, layer_id: int, num_embeddings: int, dim: int) -> "EngramFileTable":
        weight_key = f"layers.{layer_id}.engram.embed.weight"
        for path in sorted(glob.glob(os.path.join(table_dir, "*.safetensors"))):
            _, header = read_safetensors_header(path)
            if weight_key in header:
                scale_key = f"layers.{layer_id}.engram.embed.scale"
                return cls(path, weight_key, scale_key, num_embeddings, dim)
        raise FileNotFoundError(f"{weight_key} not found in {table_dir}")

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).cpu().numpy()
        weight = torch.from_numpy(self.weight[flat]).to(indices.device).view(torch.float8_e4m3fn)
        scale = torch.from_numpy(self.scale[flat]).to(indices.device).view(torch.float8_e8m0fnu)
        values = weight.float().unflatten(-1, (-1, self.block)) * scale.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).reshape(*indices.shape, self.dim)
