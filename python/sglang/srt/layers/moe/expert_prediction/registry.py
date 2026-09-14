"""Name-to-class registry used to build the predictors named in the env var."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.affinity import AffinityPredictor
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.popularity import PopularityPredictor

_PREDICTORS: dict[str, type[ExpertPredictor]] = {
    cls.name: cls for cls in (AffinityPredictor, PopularityPredictor)
}


def register_predictor(cls: type[ExpertPredictor]) -> type[ExpertPredictor]:
    if cls.name in _PREDICTORS:
        raise ValueError(f"expert predictor {cls.name} is already registered")
    _PREDICTORS[cls.name] = cls
    return cls


def registered_predictor_names() -> tuple[str, ...]:
    return tuple(sorted(_PREDICTORS))


def build_predictors(
    names: Sequence[str],
    *,
    specs: Sequence[MoeLayerSpec],
    device: torch.device,
    max_candidates: int,
) -> tuple[ExpertPredictor, ...]:
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate expert predictors in {list(names)}")
    unknown = [name for name in names if name not in _PREDICTORS]
    if unknown:
        raise ValueError(
            f"unknown expert predictors {unknown}; registered: "
            f"{list(registered_predictor_names())}"
        )
    return tuple(
        _PREDICTORS[name](specs=specs, device=device, max_candidates=max_candidates)
        for name in names
    )
