"""Expert format registry and the ``ExpertFormat`` protocol.

A format says what one quantization format's streamed expert rows are, how its rows are read, which GPU
kernel applies them over hot-cache slots and which CPU kernel (if any) computes them. Engine code asks a
format for a capability (``engine_served``, ``supports_cpu_experts``, ...) instead of comparing its name.

The registry maps a format name, and the quantization method names that resolve to it, to a format class
that is imported on first use. Class-level members (``key``, ``transport_build``, the ``supports_*``
flags, ``cpu_trait``) are the format's facts; one instance serves one layer.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence

import msgspec

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertQuantTrait
    from sglang.srt.layers.moe.expert_format import ExpertTensorSpec
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource


class SlotApplyArgs(msgspec.Struct, frozen=True):
    """What ``ExpertFormat.moe_apply`` takes beside the activations: the layer's slot buffers and the gather's side
    channels.

    ``streamer`` owns the hot-cache slot buffers the kernel reads. ``keep`` is the gather's drop flag (the layer's
    routed output is dropped at 0), ``swiglu_limit`` the activation clamp, and ``cpu`` the CPU lanes' output
    addresses when CPU experts serve part of the routes (None otherwise).
    """

    streamer: Any
    keep: Optional[torch.Tensor]
    swiglu_limit: float
    cpu: Optional[tuple] = None


class ExpertFormat(Protocol):
    """What the expert engine and streamer need to know about one quantization format's expert rows.

    ``tensor_specs`` returns one spec per streamed tensor, in streamer order.
    ``source`` returns the dense ``[experts, ...]`` tensor of a name at call time
    (the host arena rebinds layer tensors after startup), or None when the format
    has no dense source and only its row source can read rows.
    ``default_row_source`` builds the row source for a knob kind (see
    ``SGLANG_EXPERT_STREAM_ROW_SOURCE``) and raises for kinds it does not know.
    ``file_source_bytes_per_expert`` returns the file bytes one expert row reads, or
    None; None keeps eager gathers out of the pinned host tier and reports file
    counters as unknown.
    ``cpu_trait`` returns the CPU kernel's quant trait for ``source`` ("draft" or "target"), or None when
    the format has none; ``supports_cpu_experts`` says which. ``prepare_host_rows`` preprocesses one tensor of
    rows before it is staged (a scale interleave, say), and returns it unchanged for a format with nothing to
    do. ``moe_apply`` runs the format's GPU kernel over expert ids already remapped to hot-cache slot ids.

    Attributes:
        key: Format name, used in error messages and as the registry name.
        transport_build: The format half of the transport's ``(format, router)`` build key, or None for a
            format the native engine does not serve.
        engine_served: Whether the native engine service serves this format's rows (leased gathers, DIRECT
            residency) instead of dense host tensors.
        supports_graph_gather: Whether dense host sources can serve graph gathers.
        supports_host_arena: Whether ``SGLANG_EXPERT_STREAM_ENABLE_HOST_ARENA`` applies.
        supports_transfer_executor: Whether one expert's rows move as the transfer executor's fixed bundle of
            row copies.
        supports_cpu_experts: Whether a real CPU kernel computes this format (False: a stub or none).
        supports_draft: Whether the speculative draft's CPU experts run this format.
        supports_dead_rows: Whether a speculative verify's dead rows can be masked out of this format's apply.
        max_gather_rows: Cap on rows one eager gather stages, or None.
        inclusive_pinned_tier: True when the format's ``is_pinned`` protects its
            hot-cache experts, so the pinned tier holds every hot row (see
            :func:`inclusive_hot_slot_limit`).
    """

    key: str
    transport_build: Optional[str]
    engine_served: bool
    supports_graph_gather: bool
    supports_host_arena: bool
    supports_transfer_executor: bool
    supports_cpu_experts: bool
    supports_draft: bool
    supports_dead_rows: bool
    max_gather_rows: Optional[int]
    inclusive_pinned_tier: bool

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]: ...

    def num_experts(self, layer: torch.nn.Module) -> int: ...

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]: ...

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional[ExpertRowSource]: ...

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional[ExpertRowSource]
    ) -> Optional[int]: ...

    def pinned_tier_options(self, layer: torch.nn.Module) -> Mapping[str, Any]:
        """Keyword arguments for this layer's ``ExpertPinnedHostCache``, e.g.
        ``is_pinned``."""
        ...

    def layer_dims(self, layer: torch.nn.Module) -> tuple[int, int]:
        """``(experts, hidden)`` of a layer the format streams, as its quantization method created it."""
        ...

    def build_streamer(self, layer: torch.nn.Module) -> Any:
        """The ``ExpertStreamer`` of a layer whose routed experts stay on disk."""
        ...

    def cpu_trait(self, *, source: str) -> Optional[CpuExpertQuantTrait]: ...

    def prepare_host_rows(
        self, layer: torch.nn.Module, name: str, tensor: torch.Tensor
    ) -> torch.Tensor: ...

    def moe_apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        slot_ids: torch.Tensor,
        weights: torch.Tensor,
        slot_tensors: SlotApplyArgs,
    ) -> torch.Tensor: ...


class _FormatEntry(msgspec.Struct, frozen=True):
    """A registered format: where its class lives and which quantization methods resolve to it."""

    target: str
    quant_methods: tuple[str, ...]


# Formats are imported on first use, so the registry loads without torch. A format adds its entry here
# together with its ``formats/<name>.py``.
_FORMATS: dict[str, _FormatEntry] = {
    "exl3": _FormatEntry(
        target="sglang.srt.layers.moe.expert_engine.formats.exl3:Exl3ExpertFormat",
        quant_methods=("exl3",),
    ),
    # ModelOpt NVFP4 checkpoints: the quant method name is shared with every NVFP4 model (Qwen too), so a name match
    # says what the experts' format is, not that the engine serves the model: that is the model profile's call.
    "nvfp4": _FormatEntry(
        target="sglang.srt.layers.moe.expert_engine.formats.nvfp4:Nvfp4ExpertFormat",
        quant_methods=("modelopt_fp4",),
    ),
    # MXFP4 experts sit in an FP8 checkpoint (quant_method "fp8"), so they are told apart by ``store_dtype``
    # (_STORE_DTYPE_FORMATS), not by quant method name.
    "mxfp4": _FormatEntry(
        target="sglang.srt.layers.moe.expert_engine.formats.mxfp4:Mxfp4ExpertFormat",
        quant_methods=(),
    ),
}

# ``quantization_config.store_dtype`` -> format name, for a format whose checkpoint's quant method names another scheme.
_STORE_DTYPE_FORMATS: dict[str, str] = {"mxfp4": "mxfp4"}


def registered_formats() -> tuple[str, ...]:
    """The registered format names, sorted."""
    return tuple(sorted(_FORMATS))


def get_format(name: str) -> type[ExpertFormat]:
    """The format class registered under ``name``; ValueError naming the registered formats otherwise."""
    entry = _FORMATS.get(name)
    if entry is None:
        raise ValueError(f"no expert format {name!r}; registered: {list(registered_formats())}")
    module_name, _, class_name = entry.target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def format_for_quantization_config(quantization_config: Any) -> Optional[type[ExpertFormat]]:
    """The format a quantization config resolves to, or None when it names none.

    ``quantization_config`` is a ``QuantizationConfig`` (named by ``get_name()``) or a checkpoint's
    ``quantization_config`` mapping (named by ``quant_method``); None resolves to None. A ``store_dtype`` that
    names a format (MXFP4 inside an FP8 checkpoint) wins over the quant method, and a ModelOpt mapping
    (``quant_method`` "modelopt") whose ``quant_algo`` is an NVFP4 one resolves like ``modelopt_fp4``.
    """
    if quantization_config is None:
        return None
    if isinstance(quantization_config, Mapping):
        method = quantization_config.get("quant_method")
        store_dtype = quantization_config.get("store_dtype")
        if method == "modelopt" and "NVFP4" in str(quantization_config.get("quant_algo", "")):
            method = "modelopt_fp4"
    else:
        method = quantization_config.get_name()
        store_dtype = getattr(quantization_config, "store_dtype", None)
    by_store_dtype = _STORE_DTYPE_FORMATS.get(str(store_dtype).lower()) if store_dtype is not None else None
    if by_store_dtype is not None:
        return get_format(by_store_dtype)
    for name, entry in _FORMATS.items():
        if method in entry.quant_methods:
            return get_format(name)
    return None


def format_for_quant_method(quant_method: Any) -> Optional[type[ExpertFormat]]:
    """The format a MoE quant method streams, or None when it is not an expert-engine quant method.

    A quant method names its format with ``expert_format_name``; one that does not (every dense method)
    has None.
    """
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

    if not isinstance(quant_method, FusedMoEMethodBase):
        return None
    name = quant_method.expert_format_name
    return None if name is None else get_format(name)
