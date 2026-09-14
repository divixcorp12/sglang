"""Next-layer baseline: decayed source-to-target route co-occurrence, then popularity."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.popularity import accumulate_routes

_DECAY = 0.99
# Arbitrary; scales sum-normalized popularity below one co-occurrence so it
# mostly orders targets without co-occurrence evidence.
_POPULARITY_WEIGHT = 1e-3


class AffinityPredictor(ExpertPredictor):
    name = "affinity"
    target_offset = 1
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        super().__init__(specs=specs, device=device, max_candidates=max_candidates)
        self._pairs = tuple(zip(self.layer_ids, self.layer_ids[1:]))
        self._transitions = {
            source: torch.zeros(
                (self.specs[source].num_experts, self.specs[target].num_experts),
                dtype=torch.float32,
                device=device,
            )
            for source, target in self._pairs
        }
        self._popularity = {
            layer_id: torch.zeros(spec.num_experts, dtype=torch.float32, device=device)
            for layer_id, spec in self.specs.items()
        }

    @property
    def state_nbytes(self) -> int:
        tensors = [*self._transitions.values(), *self._popularity.values()]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        transitions = self._transitions[source_layer]
        ids = store.view(source_layer, RouteFeature.TOPK_IDS, rows)
        num_sources = transitions.shape[0]
        valid = ((ids >= 0) & (ids < num_sources)).unsqueeze(-1).to(transitions.dtype)
        scores = (transitions[ids.clamp(0, num_sources - 1)] * valid).sum(dim=1)
        popularity = self._popularity[target_layer]
        scores = scores + _POPULARITY_WEIGHT * popularity / (popularity.sum() + 1.0)
        ranked = torch.topk(scores, min(self.max_candidates, scores.shape[1]), dim=1).indices
        return pad_candidates(ranked, self.max_candidates)

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        for source, target in self._pairs:
            self._observe_pair(
                transitions=self._transitions[source],
                source_ids=store.view(source, RouteFeature.TOPK_IDS, rows),
                target_ids=store.view(target, RouteFeature.TOPK_IDS, rows),
            )
        for layer_id, popularity in self._popularity.items():
            accumulate_routes(
                popularity, store.view(layer_id, RouteFeature.TOPK_IDS, rows), decay=_DECAY
            )

    @staticmethod
    def _observe_pair(
        *, transitions: torch.Tensor, source_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> None:
        num_sources, num_targets = transitions.shape
        source_valid = (source_ids >= 0) & (source_ids < num_sources)
        target_valid = (target_ids >= 0) & (target_ids < num_targets)
        pair_index = source_ids.clamp(0, num_sources - 1).unsqueeze(2) * num_targets + (
            target_ids.clamp(0, num_targets - 1).unsqueeze(1)
        )
        pair_weight = (source_valid.unsqueeze(2) & target_valid.unsqueeze(1)).to(
            transitions.dtype
        )
        transitions.mul_(_DECAY)
        transitions.view(-1).index_add_(0, pair_index.reshape(-1), pair_weight.reshape(-1))
