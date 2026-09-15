"""Evaluation metrics and baselines shared by LLaPor and APEX training."""

from __future__ import annotations

import torch


def topk_candidates(logits: torch.Tensor, budget: int) -> torch.Tensor:
    return torch.topk(logits, budget, dim=-1).indices


def recall_at_budget(candidates: torch.Tensor, native_ids: torch.Tensor) -> float:
    """Mean fraction of each row's native_ids present in candidates."""
    hit = (candidates.unsqueeze(1) == native_ids.unsqueeze(2)).any(dim=2)
    return float(hit.float().mean(dim=1).mean())


def same_expert_baseline_candidates(
    source_topk_ids: torch.Tensor, budget: int, num_experts: int
) -> torch.Tensor:
    """Baseline: the current layer's own selected experts, padded by ascending ID
    (the pad ids are a fixed deterministic fill, not excluded per row)."""
    rows, k = source_topk_ids.shape
    if budget <= k:
        return source_topk_ids[:, :budget]
    pad = torch.arange(budget - k, device=source_topk_ids.device).unsqueeze(0).expand(rows, -1)
    return torch.cat([source_topk_ids, pad], dim=1)


def popularity_baseline_candidates(
    topk_ids: torch.Tensor, num_experts: int, budget: int
) -> torch.Tensor:
    """Global top-`budget` most frequently selected experts, identical for every row."""
    counts = torch.zeros(num_experts, dtype=torch.float64)
    counts.scatter_add_(
        0, topk_ids.reshape(-1).long(), torch.ones(topk_ids.numel(), dtype=torch.float64)
    )
    top = torch.topk(counts, budget).indices
    return top.unsqueeze(0).expand(topk_ids.shape[0], -1)


def full_set_coverage(chosen_depth: torch.Tensor, delta_star: torch.Tensor) -> float:
    """Fraction of rows whose chosen depth covers every native id."""
    return float((chosen_depth >= delta_star).float().mean())
