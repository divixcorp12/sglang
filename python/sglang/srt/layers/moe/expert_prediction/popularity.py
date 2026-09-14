"""Same-layer baseline: rank each layer's experts by decayed route counts."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

# Arbitrary; roughly a 100-forward memory.
_DECAY = 0.99


def accumulate_routes(counts: torch.Tensor, ids: torch.Tensor, *, decay: float) -> None:
    """Decay ``counts`` then add one per in-range routed id, without host syncs."""
    num_experts = counts.numel()
    valid = (ids >= 0) & (ids < num_experts)
    counts.mul_(decay)
    counts.index_add_(
        0, ids.clamp(0, num_experts - 1).reshape(-1), valid.reshape(-1).to(counts.dtype)
    )


class PopularityPredictor(ExpertPredictor):
    name = "popularity"
    target_offset = 0
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        super().__init__(specs=specs, device=device, max_candidates=max_candidates)
        self._counts = {
            layer_id: torch.zeros(spec.num_experts, dtype=torch.float32, device=device)
            for layer_id, spec in self.specs.items()
        }

    @property
    def state_nbytes(self) -> int:
        return sum(counts.numel() * counts.element_size() for counts in self._counts.values())

    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        counts = self._counts[target_layer]
        ranked = torch.topk(counts, min(self.max_candidates, counts.numel())).indices
        return pad_candidates(ranked.unsqueeze(0).expand(rows, -1), self.max_candidates)

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        for layer_id, counts in self._counts.items():
            accumulate_routes(
                counts, store.view(layer_id, RouteFeature.TOPK_IDS, rows), decay=_DECAY
            )
