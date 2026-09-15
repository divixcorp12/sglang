"""Train-only PCA fitting and projection for LLaPor routing features."""

from __future__ import annotations

import msgspec
import torch


class PCAStats(msgspec.Struct, frozen=True):
    mean: torch.Tensor  # [D] fp32
    components: torch.Tensor  # [rank, D] fp32, row i is the i-th principal axis
    explained_variance: torch.Tensor  # [rank] fp32


def fit_pca(x: torch.Tensor, rank: int) -> PCAStats:
    """Fit centering and top-`rank` components via randomized SVD, FP32 accumulation."""
    x = x.float()
    mean = x.mean(dim=0)
    centered = x - mean
    oversample = min(rank + 10, centered.shape[1])
    _, singular_values, v = torch.pca_lowrank(centered, q=oversample, center=False)
    components = v[:, :rank].T.contiguous()
    explained_variance = (singular_values[:rank] ** 2) / max(centered.shape[0] - 1, 1)
    return PCAStats(mean=mean, components=components, explained_variance=explained_variance)


def encode(x: torch.Tensor, pca: PCAStats) -> torch.Tensor:
    """h = (a - mu) @ P.T."""
    return (x.float() - pca.mean) @ pca.components.T


def reconstruct(h: torch.Tensor, pca: PCAStats) -> torch.Tensor:
    """Inverse of `encode`, up to the truncated components' residual."""
    return h @ pca.components + pca.mean
