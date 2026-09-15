"""Static-shape prefetch scorers; every op in ``forward`` can be recorded in a decode CUDA graph."""

from __future__ import annotations

import copy

import torch

from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint, LlaporCheckpoint
from sglang.srt.layers.moe.expert_prediction.training import apex


class LlaporScorer(torch.nn.Module):
    """Target-layer expert activation probabilities from one source layer's routing features."""

    def __init__(self, checkpoint: LlaporCheckpoint, *, num_experts: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("mean", checkpoint.pca.mean.to(device=device, dtype=dtype))
        self.register_buffer("projection", checkpoint.pca.components.T.contiguous().to(device=device, dtype=dtype))
        self.model = copy.deepcopy(checkpoint.model).to(device=device, dtype=dtype).eval()

    def forward(self, router_input: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        h = (router_input.to(self.mean.dtype) - self.mean) @ self.projection
        ids = topk_ids.long()
        mask = torch.zeros((h.shape[0], self.num_experts), dtype=h.dtype, device=h.device)
        route = torch.zeros_like(mask)
        mask.scatter_(1, ids, 1.0)
        route.scatter_(1, ids, topk_weights.to(h.dtype))
        return torch.sigmoid(self.model(torch.cat((h, mask, route), dim=-1)).float())


class ApexScorer(torch.nn.Module):
    """Same-layer softmax probabilities, kept only for ranks below ``top_k + depth(tau)``."""

    def __init__(
        self, checkpoint: ApexCheckpoint, *, num_experts: int, tau: float, dtype: torch.dtype, device: torch.device
    ):
        super().__init__()
        self.top_k = checkpoint.top_k
        self.tau = tau
        self.ranker = copy.deepcopy(checkpoint.ranker.linear).to(device=device, dtype=dtype)
        self.depth_projection = copy.deepcopy(checkpoint.cdf.w).to(device=device, dtype=dtype)
        self.register_buffer("thresholds", checkpoint.cdf.thresholds().detach().float().to(device))
        self.register_buffer("positions", torch.arange(num_experts, device=device))

    def forward(self, pre_mixer: torch.Tensor) -> torch.Tensor:
        x = pre_mixer.to(self.ranker.weight.dtype)
        probabilities = torch.softmax(self.ranker(x).float(), dim=-1)
        cdf_logits = self.thresholds.unsqueeze(0) - self.depth_projection(x).float()
        depth = apex.select_depth(cdf_logits, self.tau, self.thresholds.numel() - 1)
        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order).scatter_(1, order, self.positions.expand_as(order))
        return probabilities * (rank < (self.top_k + depth).unsqueeze(1))
