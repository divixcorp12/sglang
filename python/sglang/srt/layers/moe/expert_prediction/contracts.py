"""Model-independent contracts shared by MoE expert predictors."""

from __future__ import annotations

from enum import Enum

import msgspec
import torch


class RouteFeature(str, Enum):
    """A per-token routing tensor a predictor may read after a forward."""

    ROUTER_INPUT = "router_input"
    ROUTER_LOGITS = "router_logits"
    TOPK_IDS = "topk_ids"
    TOPK_WEIGHTS = "topk_weights"
    PRE_MIXER = "pre_mixer"


class MoeLayerSpec(msgspec.Struct, frozen=True):
    """One tapped MoE layer; ``num_experts`` and ``top_k`` exclude fused shared experts."""

    layer_id: int
    num_experts: int
    top_k: int
    hidden_size: int


def feature_width(feature: RouteFeature, spec: MoeLayerSpec) -> int:
    if feature in (RouteFeature.ROUTER_INPUT, RouteFeature.PRE_MIXER):
        return spec.hidden_size
    if feature is RouteFeature.ROUTER_LOGITS:
        return spec.num_experts
    return spec.top_k


def feature_dtype(feature: RouteFeature, hidden_dtype: torch.dtype) -> torch.dtype:
    if feature is RouteFeature.TOPK_IDS:
        return torch.int64
    if feature in (RouteFeature.ROUTER_LOGITS, RouteFeature.TOPK_WEIGHTS):
        return torch.float32
    return hidden_dtype
