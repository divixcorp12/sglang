"""LLaPor: predict layer L+1's selected experts from layer L's routing features.

Architecture and loss follow docs/superpowers/plans/2026-09-14-llapor-gpu-only.md
sections 4-5 (outer/middle groups, PCA + mask/route features, frequency-weighted
focal loss for outer layers).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats
from sglang.srt.layers.moe.expert_prediction.training.pca import encode as pca_encode

OUTER_SOURCE_LAYERS = frozenset(range(0, 8)) | frozenset(range(39, 47))


def layer_group(source_layer: int) -> str:
    return "outer" if source_layer in OUTER_SOURCE_LAYERS else "middle"


def pca_rank_for_group(group: str) -> int:
    return 256 if group == "outer" else 512


def encode_features(
    router_input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    pca: PCAStats,
    num_experts: int,
) -> torch.Tensor:
    """u = concat(h, mask, route), mask/route dense over the E experts."""
    h = pca_encode(router_input, pca)
    mask = torch.zeros(
        (router_input.shape[0], num_experts), dtype=torch.float32, device=router_input.device
    )
    mask.scatter_(1, topk_ids, 1.0)
    route = torch.zeros_like(mask)
    route.scatter_(1, topk_ids, topk_weights.float())
    return torch.cat([h, mask, route], dim=-1)


def multihot_labels(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    y = torch.zeros((topk_ids.shape[0], num_experts), dtype=torch.float32, device=topk_ids.device)
    y.scatter_(1, topk_ids, 1.0)
    return y


class OuterPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_experts: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        v = self.dropout(F.gelu(self.fc1(u)))
        return self.fc2(v)


class MiddlePredictor(nn.Module):
    def __init__(
        self, feature_dim: int, pca_dim: int, num_experts: int, hidden: int = 128, dropout: float = 0.1
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.fc_in = nn.Linear(feature_dim, hidden)
        self.fc_r1 = nn.Linear(hidden, hidden)
        self.fc_r2 = nn.Linear(hidden, hidden)
        self.fc_gate = nn.Linear(pca_dim, hidden)
        self.fc_out = nn.Linear(hidden, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = u[:, : self.pca_dim]
        v = F.gelu(self.fc_in(u))
        r = self.fc_r2(self.dropout(F.gelu(self.fc_r1(v))))
        g = torch.sigmoid(self.fc_gate(h))
        return self.fc_out(self.dropout(v + g * r))


def build_predictor(group: str, *, pca_rank: int, num_experts: int) -> nn.Module:
    feature_dim = pca_rank + 2 * num_experts
    if group == "outer":
        return OuterPredictor(feature_dim, num_experts)
    return MiddlePredictor(feature_dim, pca_rank, num_experts)


def expert_frequency_weights(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """q_e = clip(mean(f) / max(f_e, 1e-4), 0.1, 10), renormalized to mean 1."""
    counts = torch.zeros(num_experts, dtype=torch.float64, device=topk_ids.device)
    counts.scatter_add_(
        0,
        topk_ids.reshape(-1).long(),
        torch.ones(topk_ids.numel(), dtype=torch.float64, device=topk_ids.device),
    )
    freq = counts / topk_ids.shape[0]
    q = freq.mean() / freq.clamp_min(1e-4)
    q = q.clamp(0.1, 10.0)
    q = q / q.mean()
    return q.float()


def llapor_loss(logits: torch.Tensor, y: torch.Tensor, *, group: str, q: torch.Tensor) -> torch.Tensor:
    b = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if group != "outer":
        return b.mean()
    p = torch.sigmoid(logits)
    pt = y * p + (1 - y) * (1 - p)
    focal = ((1 - pt) ** 2) * b
    return (q * b).mean() + focal.mean()
