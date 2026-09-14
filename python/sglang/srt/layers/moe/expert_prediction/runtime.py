"""Tap MoE routes during forwards and shadow-score expert predictors after each one."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.adapters import install_pre_mixer_taps
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.metrics import ShadowMetrics, score_candidates
from sglang.srt.layers.moe.expert_prediction.registry import build_predictors
from sglang.srt.layers.moe.expert_prediction.taps import (
    RouteTaps,
    TappedMoeLayer,
    discover_moe_layers,
)
from sglang.srt.layers.moe.expert_residency_clock import ForwardKind, classify_forward

logger = logging.getLogger(__name__)

_SCORED_KINDS = frozenset({ForwardKind.DECODE, ForwardKind.VERIFY})


def layer_pairs(layer_ids: Sequence[int], offset: int) -> tuple[tuple[int, int], ...]:
    """``(source, target)`` pairs where target is ``offset`` tapped MoE layers after source."""
    return tuple(zip(layer_ids, layer_ids[offset:]))


class ExpertPredictionRuntime:
    """Shadow-score registered predictors against native routes.

    Scoring enqueues device work behind the forward on the current stream and
    never synchronizes; only the periodic metrics record reads the device.
    """

    def __init__(
        self,
        *,
        layers: Sequence[TappedMoeLayer],
        store: FeatureStore,
        taps: RouteTaps,
        pre_mixer_removers: Sequence[Callable[[], None]],
        predictors: Sequence[ExpertPredictor],
        metrics: ShadowMetrics,
        hot_caches: Mapping[int, Any],
        log_interval: int,
        metrics_path: Path | None,
    ) -> None:
        if log_interval < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL must be positive")
        self.store = store
        self.metrics = metrics
        self.forwards = 0
        self._specs = {layer.spec.layer_id: layer.spec for layer in layers}
        self._layer_ids = tuple(sorted(self._specs))
        self._taps = taps
        self._pre_mixer_removers = list(pre_mixer_removers)
        self._predictors = tuple(predictors)
        self._hot_caches = dict(hot_caches)
        self._log_interval = log_interval
        self._metrics_path = metrics_path
        self._reported_unsupported = False

    @classmethod
    def from_env(
        cls,
        *,
        model: nn.Module,
        gpu_id: int,
        hidden_dtype: torch.dtype,
        decode_max_bs: int,
        tokens_per_request: int,
        tp_size: int,
        moe_ep_size: int,
        attn_dp_size: int | None,
        expert_hot_cache_manager: Any | None,
    ) -> "ExpertPredictionRuntime":
        if tp_size > 1 or moe_ep_size > 1 or (attn_dp_size or 1) > 1:
            raise ValueError(
                "SGLANG_MOE_EXPERT_PREDICTOR supports a single GPU; got "
                f"tp={tp_size} moe_ep={moe_ep_size} attn_dp={attn_dp_size}"
            )
        max_rows = (
            envs.SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS.get()
            or decode_max_bs * tokens_per_request
        )
        if max_rows < 1:
            raise ValueError(
                "SGLANG_MOE_EXPERT_PREDICTOR needs decode CUDA graphs or "
                "SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS to size its tap buffers"
            )
        metrics_file = envs.SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE.get()
        return cls.build(
            model=model,
            predictor_names=envs.SGLANG_MOE_EXPERT_PREDICTOR.get(),
            device=torch.device("cuda", gpu_id),
            hidden_dtype=hidden_dtype,
            max_rows=max_rows,
            max_candidates=envs.SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES.get(),
            hot_caches=(
                {} if expert_hot_cache_manager is None else expert_hot_cache_manager.caches
            ),
            log_interval=envs.SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL.get(),
            metrics_path=Path(metrics_file) if metrics_file else None,
        )

    @classmethod
    def build(
        cls,
        *,
        model: nn.Module,
        predictor_names: Sequence[str],
        device: torch.device,
        hidden_dtype: torch.dtype,
        max_rows: int,
        max_candidates: int,
        hot_caches: Mapping[int, Any],
        log_interval: int,
        metrics_path: Path | None,
        topk_type: type | None = None,
        experts_type: type | None = None,
    ) -> "ExpertPredictionRuntime":
        layers = discover_moe_layers(model, topk_type=topk_type, experts_type=experts_type)
        specs = [layer.spec for layer in layers]
        predictors = build_predictors(
            predictor_names, specs=specs, device=device, max_candidates=max_candidates
        )
        features = {RouteFeature.TOPK_IDS}.union(
            *(predictor.required_features for predictor in predictors)
        )
        store = FeatureStore(
            specs=specs,
            features=features,
            max_rows=max_rows,
            device=device,
            hidden_dtype=hidden_dtype,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        pre_mixer_removers = (
            install_pre_mixer_taps(model=model, layers=layers, store=store)
            if RouteFeature.PRE_MIXER in features
            else []
        )
        logger.info(
            "MoE expert prediction shadow mode: predictors=%s layers=%d max_rows=%d "
            "tap_bytes=%d predictor_state_bytes=%d",
            ",".join(predictor_names),
            len(layers),
            max_rows,
            store.nbytes,
            sum(predictor.state_nbytes for predictor in predictors),
        )
        return cls(
            layers=layers,
            store=store,
            taps=taps,
            pre_mixer_removers=pre_mixer_removers,
            predictors=predictors,
            metrics=ShadowMetrics(
                predictor_names=predictor_names,
                layer_ids=[spec.layer_id for spec in specs],
                device=device,
            ),
            hot_caches=hot_caches,
            log_interval=log_interval,
            metrics_path=metrics_path,
        )

    def on_forward_end(self, forward_batch: Any) -> None:
        rows = self._scored_rows(forward_batch)
        if rows == 0:
            return
        for index, predictor in enumerate(self._predictors):
            self._score(predictor_index=index, predictor=predictor, rows=rows)
            predictor.observe(store=self.store, rows=rows)
        self.forwards += 1
        if self._metrics_path is not None and self.forwards % self._log_interval == 0:
            self.metrics.append_jsonl(self._metrics_path, forwards=self.forwards)

    def close(self) -> None:
        self._taps.remove()
        for remove in self._pre_mixer_removers:
            remove()
        self._pre_mixer_removers = []

    def _scored_rows(self, forward_batch: Any) -> int:
        """Rows this forward's taps hold for scoring, or 0 when it is not scored."""
        if self._taps.unsupported_layers:
            if not self._reported_unsupported:
                self._reported_unsupported = True
                logger.warning(
                    "MoE expert prediction disabled: layers %s lack standard top-k outputs",
                    sorted(self._taps.unsupported_layers),
                )
            return 0
        kind, _ = classify_forward(forward_batch)
        if kind not in _SCORED_KINDS:
            return 0
        rows = forward_batch.input_ids.shape[0]
        return rows if rows <= self.store.max_rows else 0

    def _score(self, *, predictor_index: int, predictor: ExpertPredictor, rows: int) -> None:
        for source, target in layer_pairs(self._layer_ids, predictor.target_offset):
            spec = self._specs[target]
            cache = self._hot_caches.get(target)
            candidates = predictor.predict(
                source_layer=source, target_layer=target, store=self.store, rows=rows
            )
            counts = score_candidates(
                candidates=candidates,
                actual=self.store.view(target, RouteFeature.TOPK_IDS, rows),
                top_k=spec.top_k,
                num_experts=spec.num_experts,
                resident=None if cache is None else cache.expert_to_slot >= 0,
            )
            self.metrics.add(
                predictor_index=predictor_index, target_layer=target, counts=counts
            )
