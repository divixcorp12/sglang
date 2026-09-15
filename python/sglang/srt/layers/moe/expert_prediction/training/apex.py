"""APEX: same-layer expert ranker distilled from the teacher router, plus an
ordinal CDF over extra candidate depth.

Follows docs/superpowers/plans/2026-09-14-apex-gpu-only.md section 4. The
teacher distribution is softmax(router_input @ gate.weight.T), matching this
model's TopK scoring_func="softmax" with no grouped-topk/correction bias
(see sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock and
sglang.srt.layers.moe.topk.TopK).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ranker(nn.Module):
    """Linear-softmax ranker on pre_mixer, one independent instance per layer."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def teacher_probabilities(router_input: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """softmax(router_input @ gate_weight.T), FP32."""
    logits = router_input.float() @ gate_weight.float().T
    return F.softmax(logits, dim=-1)


def ranker_kl_loss(rank_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(rank_logits.float(), dim=-1), teacher_probs.float(), reduction="batchmean"
    )


def oracle_delta(logits: torch.Tensor, native_ids: torch.Tensor) -> torch.Tensor:
    """One-based worst rank among native_ids minus K; ties break by expert ID."""
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(1, logits.shape[-1] + 1, device=logits.device)
    ranks.scatter_(1, order, positions.expand_as(order))
    k = native_ids.shape[1]
    return ranks.gather(1, native_ids).amax(dim=1) - k


class OrdinalCDF(nn.Module):
    """p_d(x) = sigmoid(theta_d - w^T x); theta ordered via a softplus cumsum."""

    def __init__(self, hidden_size: int, num_depths: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, 1, bias=False)
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.raw_increments = nn.Parameter(torch.zeros(num_depths - 1))

    def thresholds(self) -> torch.Tensor:
        increments = F.softplus(self.raw_increments)
        return torch.cat([self.theta0, self.theta0 + torch.cumsum(increments, dim=0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.thresholds()
        score = self.w(x.float())
        return theta.unsqueeze(0) - score


def cdf_loss(cdf_logits: torch.Tensor, delta_star: torch.Tensor) -> torch.Tensor:
    depths = torch.arange(cdf_logits.shape[1], device=cdf_logits.device)
    labels = (depths.unsqueeze(0) >= delta_star.unsqueeze(1)).float()
    return F.binary_cross_entropy_with_logits(cdf_logits.float(), labels)


def select_depth(cdf_logits: torch.Tensor, tau: float, max_depth: int) -> torch.Tensor:
    """First depth d with sigmoid(cdf_logits[:, d]) >= tau, else max_depth (=E-K)."""
    meets = torch.sigmoid(cdf_logits) >= tau
    any_meets = meets.any(dim=1)
    first = meets.float().argmax(dim=1)
    return torch.where(any_meets, first, torch.full_like(first, max_depth))
