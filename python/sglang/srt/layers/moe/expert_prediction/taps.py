"""Discover MoE blocks by their TopK and FusedMoE children and tap their routes."""

from __future__ import annotations

import logging
from typing import Sequence

import msgspec
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)


class TappedMoeLayer(msgspec.Struct, frozen=True):
    spec: MoeLayerSpec
    topk: nn.Module
    block: nn.Module


def discover_moe_layers(
    model: nn.Module,
    *,
    topk_type: type | None = None,
    experts_type: type | None = None,
) -> tuple[TappedMoeLayer, ...]:
    """Pair every TopK with its sibling FusedMoE; layer ids come from the FusedMoE."""
    if topk_type is None or experts_type is None:
        # Deferred: fused_moe_triton.layer pulls in the quantization stack.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopK

        topk_type = topk_type or TopK
        experts_type = experts_type or FusedMoE
    layers: dict[int, TappedMoeLayer] = {}
    for block in model.modules():
        children = list(block.children())
        topks = [child for child in children if isinstance(child, topk_type)]
        experts = [child for child in children if isinstance(child, experts_type)]
        if not topks or not experts:
            continue
        if len(topks) != 1 or len(experts) != 1:
            raise ValueError(
                f"{type(block).__name__} holds {len(topks)} TopK and "
                f"{len(experts)} FusedMoE children; expected one of each"
            )
        spec = _layer_spec(topk=topks[0], moe=experts[0])
        if spec.layer_id in layers:
            raise ValueError(f"two MoE blocks claim layer_id {spec.layer_id}")
        layers[spec.layer_id] = TappedMoeLayer(spec=spec, topk=topks[0], block=block)
    if not layers:
        raise ValueError("model has no TopK + FusedMoE blocks to tap")
    return tuple(layers[layer_id] for layer_id in sorted(layers))


def _layer_spec(*, topk: nn.Module, moe: nn.Module) -> MoeLayerSpec:
    # Qwen2MoeSparseMoeBlock builds TopK without fused shared experts while its
    # FusedMoE counts the shared slots and appends shared ids after TopK returns.
    return MoeLayerSpec(
        layer_id=moe.layer_id,
        num_experts=moe.num_experts - moe.num_fused_shared_experts,
        top_k=topk.topk_config.top_k - topk.topk_config.num_fused_shared_experts,
        hidden_size=moe.hidden_size,
    )


class RouteTaps:
    """Copy each TopK call's tensors into a FeatureStore with device-only ops.

    Hooks run in eager forwards and while a CUDA graph is captured; replay
    re-executes the recorded copies without calling Python.
    """

    def __init__(self, store: FeatureStore) -> None:
        self._store = store
        self._handles: list = []
        self.unsupported_layers: set[int] = set()

    def install(self, layers: Sequence[TappedMoeLayer]) -> None:
        for layer in layers:
            self._handles.append(
                layer.topk.register_forward_hook(
                    self._hook(layer.spec.layer_id), with_kwargs=True
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _hook(self, layer_id: int):
        store = self._store

        def hook(module, args, kwargs, output):
            if not isinstance(output, StandardTopKOutput):
                if layer_id not in self.unsupported_layers:
                    self.unsupported_layers.add(layer_id)
                    logger.warning(
                        "MoE expert prediction cannot tap layer %d: TopK returned %s",
                        layer_id,
                        type(output).__name__,
                    )
                return
            hidden_states = args[0] if args else kwargs["hidden_states"]
            try:
                store.write(layer_id, RouteFeature.ROUTER_INPUT, hidden_states)
                if output.router_logits is not None:
                    store.write(layer_id, RouteFeature.ROUTER_LOGITS, output.router_logits)
                store.write(layer_id, RouteFeature.TOPK_IDS, output.topk_ids)
                store.write(layer_id, RouteFeature.TOPK_WEIGHTS, output.topk_weights)
            except ValueError as error:
                if layer_id not in self.unsupported_layers:
                    self.unsupported_layers.add(layer_id)
                    logger.warning(
                        "MoE expert prediction cannot tap layer %d: %s", layer_id, error
                    )

        return hook
