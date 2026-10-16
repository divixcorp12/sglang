"""JIT wrapper for the gather-time lane head (srt/layers/moe/lane_head.py): PCA of the gate input, then one plan key
per routed expert and the layer's per-request state update."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_WEIGHT_DTYPES = (torch.float32, torch.bfloat16)
_PROJ_DTYPES = (torch.bfloat16, torch.float32)


@cache_once
def _jit_lane_head_module(id_dtype: torch.dtype, weight_dtype: torch.dtype, proj_dtype: torch.dtype) -> Module:
    if id_dtype not in _ID_DTYPES:
        raise RuntimeError(f"lane head: route ids must be int32 or int64, got {id_dtype}")
    if weight_dtype not in _WEIGHT_DTYPES:
        raise RuntimeError(f"lane head: routing weights must be float32 or bfloat16, got {weight_dtype}")
    if proj_dtype not in _PROJ_DTYPES:
        raise RuntimeError(f"lane head: the PCA projection must be bfloat16 or float32, got {proj_dtype}")
    args = make_cpp_args(id_dtype, weight_dtype, proj_dtype)
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
    dead: Optional[torch.Tensor],
    pca_mean: torch.Tensor,
    pca_proj: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    z_scale: torch.Tensor,
    expert_bias: torch.Tensor,
    dense_w: torch.Tensor,
    dense_b: torch.Tensor,
    z_w: torch.Tensor,
    expert_u: torch.Tensor,
    expert_to_slot: torch.Tensor,
    score: torch.Tensor,
    last: torch.Tensor,
    verify_ct: torch.Tensor,
    z_partial: torch.Tensor,
    decay_pow: torch.Tensor,
    keys: torch.Tensor,
    odds: float,
) -> None:
    """Write ``keys[e]`` for every expert the live rows of ``topk_ids`` route, then advance ``score``, ``last`` and
    ``verify_ct``, on the current stream. ``dead`` (bool, a row per token, or None) marks a verify's dead rows;
    ``expert_to_slot`` (int64 [experts], -1 nonresident) gives the resident feature; ``odds`` is the draft-odds
    feature's training value (lane_head.LaneHeadMeta.odds). Tensor invariants are checked in the C++ launcher."""
    module = _jit_lane_head_module(topk_ids.dtype, topk_weights.dtype, pca_proj.dtype)
    module.lane_head_keys(
        topk_ids,
        topk_weights,
        x,
        dead,
        pca_mean,
        pca_proj,
        feature_mean,
        feature_scale,
        z_scale,
        expert_bias,
        dense_w,
        dense_b,
        z_w,
        expert_u,
        expert_to_slot,
        score,
        last,
        verify_ct,
        z_partial,
        decay_pow,
        keys,
        float(odds),
    )
