"""Interface every MoE expert predictor implements."""

from __future__ import annotations

import abc
from typing import ClassVar, Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore


class ExpertPredictor(abc.ABC):
    """Rank expert candidates for tapped MoE layers from a completed forward's features.

    ``target_offset`` 0 predicts the source layer's own routes and 1 the next
    tapped MoE layer. Within a forward every ``predict`` runs before
    ``observe``, and both must stay device-only.
    """

    name: ClassVar[str]
    target_offset: ClassVar[int]
    required_features: ClassVar[frozenset[RouteFeature]]

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.specs = {spec.layer_id: spec for spec in specs}
        self.layer_ids = tuple(sorted(self.specs))
        self.device = device
        self.max_candidates = max_candidates

    @abc.abstractmethod
    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        """Ranked expert ids for ``target_layer``: int64 ``[rows, max_candidates]``, -1 padded."""

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        """Update online state from this forward's features after it was scored."""

    @property
    def state_nbytes(self) -> int:
        return 0


def pad_candidates(ranked: torch.Tensor, width: int) -> torch.Tensor:
    if ranked.shape[1] >= width:
        return ranked[:, :width]
    return torch.nn.functional.pad(ranked, (0, width - ranked.shape[1]), value=-1)
