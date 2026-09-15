"""Live expert prefetch scoring: per-target-layer candidates written inside the decode graph from tap writes.

Every per-token op runs from ``FeatureStore.after_write`` during eager forwards and graph
capture, so replay executes recorded kernels only. Phase B hands ``bank`` rows to the
shared copy layer; this module never touches the copy path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer

logger = logging.getLogger(__name__)

_FEATURES = {
    "llapor": frozenset({RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS}),
    "apex": frozenset({RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS}),
}
# The last feature each tap writes for a layer; scoring fires on it so the others are fresh.
_SCORE_TRIGGER = {"llapor": RouteFeature.TOPK_WEIGHTS, "apex": RouteFeature.PRE_MIXER}


class PrefetchScoring:
    """Owns scorers, the candidate bank Phase B reads, and shadow budget-recall counters."""

    @staticmethod
    def features_for(predictor: str) -> frozenset[RouteFeature]:
        return _FEATURES[predictor]

    @classmethod
    def build(cls, *, predictor: str, model_dir: Path, specs: Sequence[MoeLayerSpec], **kwargs) -> "PrefetchScoring":
        checkpoints = load_prefetch_checkpoints(model_dir, predictor=predictor, specs=specs)
        return cls.from_checkpoints(predictor=predictor, checkpoints=checkpoints, specs=specs, **kwargs)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        predictor: str,
        checkpoints: Mapping[int, Any],
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        hot_caches: Mapping[int, Any],
        width: int,
        budget: int,
        tau: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PrefetchScoring":
        by_layer = {spec.layer_id: spec for spec in specs}
        if width > min(spec.num_experts for spec in specs):
            raise ValueError("SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES exceeds the expert count")
        missing_caches = sorted(set(checkpoints) - set(hot_caches))
        if missing_caches:
            raise ValueError(f"expert prefetch scoring needs hot caches for layers {missing_caches}")
        if predictor == "llapor":
            scorers = {target: LlaporScorer(c, num_experts=by_layer[target].num_experts, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {c.source_layer: target for target, c in checkpoints.items()}
            next_target = dict(source_of)
        else:
            scorers = {target: ApexScorer(c, num_experts=by_layer[target].num_experts, tau=tau, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {target: target for target in checkpoints}
            next_target = {}
        scoring = cls(
            predictor=predictor, scorers=scorers, source_of=source_of, next_target=next_target, store=store,
            hot_caches=hot_caches,
            bank=PrefetchCandidateBank(layer_ids=list(checkpoints), width=width, device=device),
            recall=BudgetRecall(layer_ids=list(checkpoints), budget=budget, device=device),
        )
        store.after_write = scoring._on_write
        logger.info("MoE expert prefetch scoring: predictor=%s targets=%d width=%d budget=%d state_bytes=%d",
                    predictor, len(checkpoints), width, budget, scoring.state_nbytes)
        return scoring

    def __init__(self, *, predictor, scorers, source_of, next_target, store, hot_caches, bank, recall) -> None:
        self.predictor = predictor
        self.targets = sorted(scorers)
        # Source layer -> target layer for a next-layer predictor; empty for same-layer predictors.
        self.next_target = next_target
        self._scorers = scorers
        self._source_of = source_of
        self._store = store
        self._hot_caches = dict(hot_caches)
        self.bank = bank
        self.recall = recall

    @property
    def required_features(self) -> frozenset[RouteFeature]:
        return _FEATURES[self.predictor]

    @property
    def state_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self._scorers.values() for t in (*s.parameters(), *s.buffers()))

    def _on_write(self, layer_id: int, feature: RouteFeature, rows: int) -> None:
        if feature is RouteFeature.TOPK_IDS and layer_id in self._scorers:
            self.recall.observe(
                target_layer=layer_id, candidate_ids=self.bank.ids_for(layer_id),
                topk_ids=self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                expert_to_slot=self._hot_caches[layer_id].expert_to_slot,
            )
        if feature is not _SCORE_TRIGGER[self.predictor] or layer_id not in self._source_of:
            return
        target = self._source_of[layer_id]
        with torch.no_grad():
            if self.predictor == "llapor":
                scores = self._scorers[target](
                    self._store.view(layer_id, RouteFeature.ROUTER_INPUT, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_WEIGHTS, rows),
                )
            else:
                scores = self._scorers[target](self._store.view(layer_id, RouteFeature.PRE_MIXER, rows))
        self.bank.write(target, scores)

    def metrics_record(self) -> dict:
        """Host read of the device counters; call only at metric log intervals."""
        layers = {str(layer): {"missed_routes": missed, "covered_routes": covered}
                  for layer, (missed, covered) in self.recall.snapshot().items()}
        missed = sum(v["missed_routes"] for v in layers.values())
        covered = sum(v["covered_routes"] for v in layers.values())
        return {"predictor": self.predictor, "budget": self.recall.budget,
                "budget_recall": covered / missed if missed else 0.0, "layers": layers}
