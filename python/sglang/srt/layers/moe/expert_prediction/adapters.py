"""Per-architecture taps for the pre-mixer feature; the default suits plain decoder stacks."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import TappedMoeLayer

PreMixerInstaller = Callable[..., list[Callable[[], None]]]


def decoder_layers_by_moe_layer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer]
) -> dict[int, nn.Module]:
    """Map each tapped layer id to the outermost ``nn.ModuleList`` element containing its block."""
    block_ids = {id(layer.block): layer.spec.layer_id for layer in layers}
    found: dict[int, nn.Module] = {}
    for module in model.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for element in module:
            for sub in element.modules():
                layer_id = block_ids.get(id(sub))
                if layer_id is not None and layer_id not in found:
                    found[layer_id] = element
    missing = sorted({layer.spec.layer_id for layer in layers} - set(found))
    if missing:
        raise ValueError(f"no decoder layer contains MoE layers {missing}")
    return found


def install_decoder_input_pre_mixer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    """Tap each decoder layer's ``hidden_states`` input (kwarg, else first 2-D float arg of hidden width)."""
    widths = {layer.spec.layer_id: layer.spec.hidden_size for layer in layers}
    architecture = type(model).__name__
    removers = []
    for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items():

        def hook(module, args, kwargs, layer_id=layer_id, width=widths[layer_id]):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = next(
                    (
                        arg
                        for arg in args
                        if isinstance(arg, torch.Tensor)
                        and arg.dim() == 2
                        and arg.is_floating_point()
                        and arg.shape[-1] == width
                    ),
                    None,
                )
            if hidden is None:
                raise ValueError(
                    f"decoder layer {layer_id} has no hidden-state input of width "
                    f"{width}; register a pre-mixer adapter for {architecture}"
                )
            store.write(layer_id, RouteFeature.PRE_MIXER, hidden)

        handle = decoder.register_forward_pre_hook(hook, with_kwargs=True)
        removers.append(handle.remove)
    return removers


def install_hyper_connection_pre_mixer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    """Tap ``attn_hyper_connection.mix(...)[0]``, the tensor the attention or linear mixer consumes."""
    removers = []
    for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items():
        owner = decoder.attn_hyper_connection
        original = owner.mix

        def mix(hyper_input, original=original, layer_id=layer_id):
            result = original(hyper_input)
            store.write(layer_id, RouteFeature.PRE_MIXER, result[0])
            return result

        owner.__dict__["mix"] = mix
        removers.append(lambda owner=owner: owner.__dict__.pop("mix", None))
    return removers


_PRE_MIXER_ADAPTERS: dict[str, PreMixerInstaller] = {
    "Qwen4ExpForConditionalGeneration": install_hyper_connection_pre_mixer,
}


def register_pre_mixer_adapter(*, architecture: str, installer: PreMixerInstaller) -> None:
    if architecture in _PRE_MIXER_ADAPTERS:
        raise ValueError(f"pre-mixer adapter already registered for {architecture}")
    _PRE_MIXER_ADAPTERS[architecture] = installer


def install_pre_mixer_taps(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    installer = _PRE_MIXER_ADAPTERS.get(
        type(model).__name__, install_decoder_input_pre_mixer
    )
    return installer(model=model, layers=layers, store=store)
