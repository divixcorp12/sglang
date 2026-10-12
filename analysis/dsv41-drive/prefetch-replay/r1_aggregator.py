"""R1 block aggregator and the block / route / B1 token losses.

Inputs: per-token expert logits [G,6,E] of a group batch (padding rows have live=False) and its
GroupBatch labels. Output: one block logit per expert per group, and scalar losses.
Padding never reaches a feature, a pool or a loss (R7).

Usage: imported by r1_models, r1_challengers and r1_train.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
from r1_groups import GroupBatch  # noqa: E402

# anchor logit, anchor present, draft logit, draft present, max, mean, std, sum, noisy-or, live count
N_BLOCK_FEATURES = 10


def block_features(token_logits: torch.Tensor, *, live: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Per-expert pooled features over live positions only; position 0 is the anchor, the rest are drafts."""
    liv = live.to(token_logits.device).bool().unsqueeze(-1)  # [G,6,1]
    pos = position.to(token_logits.device).unsqueeze(-1)
    zero = torch.zeros_like(token_logits)
    lg = torch.where(liv, token_logits, zero)
    is_anchor = liv & (pos == 0)
    is_draft = liv & (pos != 0)
    anchor = torch.where(is_anchor, token_logits, zero).sum(dim=1)
    anchor_present = is_anchor.any(dim=1).to(token_logits.dtype).expand_as(anchor)
    neg_inf = torch.full_like(token_logits, float("-inf"))
    draft_present = is_draft.any(dim=1)
    draft = torch.where(is_draft, token_logits, neg_inf).amax(dim=1)
    draft = torch.where(draft_present, draft, torch.zeros_like(draft))
    count = liv.to(token_logits.dtype).sum(dim=1)  # [G,1]
    safe = count.clamp_min(1.0)
    total = lg.sum(dim=1)
    mean = total / safe
    var = (torch.where(liv, token_logits - mean.unsqueeze(1), zero) ** 2).sum(dim=1) / safe
    peak = torch.where(liv, token_logits, neg_inf).amax(dim=1)
    peak = torch.where(count > 0, peak, torch.zeros_like(peak))
    # noisy-or of sigmoid(l) in logit space: 1 - prod(1 - p) = 1 - exp(-sum softplus(l))
    soft = torch.where(liv, F.softplus(token_logits), zero).sum(dim=1)
    noisy = -torch.expm1(-soft)
    feats = [anchor, anchor_present, draft, draft_present.to(token_logits.dtype), peak, mean,
             var.clamp_min(0.0).sqrt(), total, noisy, count.expand_as(anchor)]
    return torch.stack(feats, dim=-1)


class BlockAggregator(nn.Module):
    """Shared Linear-GELU-Linear over block features plus target-layer and stratum embeddings."""

    def __init__(self, *, n_layers: int, n_strata: int, emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.layer_emb = nn.Embedding(n_layers, emb_dim)
        self.stratum_emb = nn.Embedding(n_strata, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(N_BLOCK_FEATURES + 2 * emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, token_logits: torch.Tensor, *, batch: GroupBatch) -> torch.Tensor:
        dev = token_logits.device
        feats = block_features(token_logits, live=torch.as_tensor(batch.live, device=dev),
                               position=torch.as_tensor(batch.position, device=dev))
        e = feats.shape[1]
        ctx = torch.cat([self.layer_emb(torch.as_tensor(batch.target_layer, device=dev)),
                         self.stratum_emb(torch.as_tensor(batch.stratum, device=dev))], dim=-1)
        ctx = ctx.unsqueeze(1).expand(-1, e, -1)
        return self.mlp(torch.cat([feats, ctx], dim=-1)).squeeze(-1)


def block_loss(logits: torch.Tensor, *, miss: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """D4: per-group BCE averaged over available experts, then over groups that have any."""
    avail = available.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, miss.to(logits.dtype), reduction="none")
    n = avail.sum(dim=1)
    per_group = (bce * avail).sum(dim=1) / n.clamp_min(1.0)
    has = (n > 0).to(logits.dtype)
    return (per_group * has).sum() / has.sum().clamp_min(1.0)


def _masked_token_mean(bce: torch.Tensor, *, live: torch.Tensor) -> torch.Tensor:
    liv = live.to(bce.dtype)
    per_token = bce.mean(dim=-1)
    return (per_token * liv).sum() / liv.sum().clamp_min(1.0)


def route_loss(token_logits: torch.Tensor, *, route: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Token route BCE over live rows only."""
    bce = F.binary_cross_entropy_with_logits(token_logits, route.to(token_logits.dtype), reduction="none")
    return _masked_token_mean(bce, live=live)


def b1_token_loss(token_logits: torch.Tensor, *, token_y: torch.Tensor, hard_neg: torch.Tensor,
                  live: torch.Tensor, hard_neg_weight: float) -> torch.Tensor:
    """B1's original loss (b1_head.head_loss), restricted to live rows."""
    bce = F.binary_cross_entropy_with_logits(token_logits, token_y.to(token_logits.dtype), reduction="none")
    weight = 1.0 + (hard_neg_weight - 1.0) * hard_neg.to(token_logits.dtype)
    return _masked_token_mean(bce * weight, live=live)
