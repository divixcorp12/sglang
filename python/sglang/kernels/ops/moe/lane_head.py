"""JIT wrapper for the gather-time lane head (srt/layers/moe/lane_head.py): PCA of the gate input, then one plan key
per routed expert."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_WEIGHT_DTYPES = (torch.float32, torch.bfloat16)


@cache_once
def _jit_lane_head_module(id_dtype: torch.dtype, weight_dtype: torch.dtype) -> Module:
    if id_dtype not in _ID_DTYPES:
        raise RuntimeError(f"lane head: route ids must be int32 or int64, got {id_dtype}")
    if weight_dtype not in _WEIGHT_DTYPES:
        raise RuntimeError(f"lane head: routing weights must be float32 or bfloat16, got {weight_dtype}")
    args = make_cpp_args(id_dtype, weight_dtype, is_arch_support_pdl())
    return load_jit(
        "lane_head",
        *args,
        cuda_files=["moe/lane_head.cuh"],
        cuda_wrappers=[("lane_head_keys", f"lane_head_keys<{args}>")],
    )


def lane_head_keys(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x: torch.Tensor,
    pca_mean: torch.Tensor,
    pca_proj: torch.Tensor,
    expert_bias: torch.Tensor,
    expert_u: torch.Tensor,
    bias: torch.Tensor,
    dense_w: torch.Tensor,
    z_w: torch.Tensor,
    insert_scores: torch.Tensor,
    last_routed: torch.Tensor,
    expert_to_slot: torch.Tensor,
    forward: torch.Tensor,
    z: torch.Tensor,
    keys: torch.Tensor,
    launches: torch.Tensor,
    odds: float = 1.0,
) -> None:
    """Write ``keys[e]`` for every expert ``topk_ids`` routes, on the current stream. ``expert_to_slot`` (int64
    [experts], -1 for a nonresident) gives the resident feature; ``odds`` is the draft-odds feature's value, the one
    training saw (lane_head.LaneHeadMeta.odds). Tensor invariants are checked in the C++ launcher; see lane_head.cuh."""
    module = _jit_lane_head_module(topk_ids.dtype, topk_weights.dtype)
    module.lane_head_keys(
        topk_ids, topk_weights, x, pca_mean, pca_proj, expert_bias, expert_u, bias, dense_w, z_w, insert_scores,
        last_routed, expert_to_slot, forward, z, keys, launches, float(odds),
    )
