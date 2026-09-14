"""Fixed-address per-layer buffers that route taps fill and predictors read."""

from __future__ import annotations

from typing import Iterable

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)

# Top-k tensors may carry fused shared-expert columns after the routed ones.
_PREFIX_FEATURES = frozenset({RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS})


class FeatureStore:
    """Preallocated ``[max_rows, width]`` buffers keyed by ``(layer_id, feature)``.

    Addresses never change, so copies recorded during CUDA graph capture keep
    writing the same storage on replay.
    """

    def __init__(
        self,
        *,
        specs: Iterable[MoeLayerSpec],
        features: Iterable[RouteFeature],
        max_rows: int,
        device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> None:
        if max_rows < 1:
            raise ValueError("feature store needs at least one row")
        self.max_rows = max_rows
        wanted = frozenset(features)
        self._buffers: dict[tuple[int, RouteFeature], torch.Tensor] = {
            (spec.layer_id, feature): torch.zeros(
                (max_rows, feature_width(feature, spec)),
                dtype=feature_dtype(feature, hidden_dtype),
                device=device,
            )
            for spec in specs
            for feature in wanted
        }

    @property
    def nbytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size() for buffer in self._buffers.values()
        )

    def holds(self, layer_id: int, feature: RouteFeature) -> bool:
        return (layer_id, feature) in self._buffers

    def write(self, layer_id: int, feature: RouteFeature, source: torch.Tensor) -> None:
        """Copy ``source`` rows in; unstored features and batches above ``max_rows`` are skipped."""
        buffer = self._buffers.get((layer_id, feature))
        rows = source.shape[0]
        if buffer is None or rows == 0 or rows > self.max_rows:
            return
        width = buffer.shape[1]
        flat = source.reshape(rows, -1)
        prefix_ok = feature in _PREFIX_FEATURES and flat.shape[1] > width
        if flat.shape[1] != width and not prefix_ok:
            raise ValueError(
                f"layer {layer_id} {feature.value} has width {flat.shape[1]}, "
                f"expected {width}"
            )
        with torch.no_grad():
            buffer[:rows].copy_(flat[:, :width])

    def view(self, layer_id: int, feature: RouteFeature, rows: int) -> torch.Tensor:
        return self._buffers[(layer_id, feature)][:rows]
