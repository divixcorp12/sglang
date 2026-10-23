"""Expert tensor formats: the row schema and dense sources an expert streamer reads.

A format tells :class:`~sglang.srt.layers.moe.expert_stream.ExpertStreamer` which
tensors one expert row holds (:class:`ExpertTensorSpec`), where each tensor's dense
``[experts, ...]`` source lives if it has one, and which row source fills host rows.

This file holds:

* :class:`ExpertTensorSpec`, one streamed tensor's per-expert row.
* :class:`DenseLayerFormat`, the NVFP4 format: every tensor is a layer attribute.
  (The ``ExpertFormat`` protocol and the EXL3 format live in
  ``expert_engine/formats``.)
* Helpers that find a layer's streamer, resolve the row source kind
  (``SGLANG_EXPERT_STREAM_ROW_SOURCE``) and check graph-gather and pinned-tier support.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import torch

from sglang.srt.layers.moe.expert_engine.config import ExpertStreamConfig, stream_config
from sglang.srt.layers.moe.expert_transfer import NVFP4_TRANSFER_TENSOR_COUNT

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

logger = logging.getLogger(__name__)
_WARNED_WITHOUT_TIER_OPTIONS: set[str] = set()

# The attribute a quantization method sets on a MoE layer to attach its streamer.
# The name predates other formats; every format uses it so discovery has one home.
STREAMER_ATTRIBUTE = "_nvfp4_expert_streamer"
# Set by the NVFP4 method when every host tensor is a verified expert-file view.
FILE_SOURCE_BYTES_ATTRIBUTE = "_nvfp4_file_source_bytes_per_expert"
# Row source kinds every format accepts; a format may define more.
GENERIC_ROW_SOURCE_KINDS = ("auto", "files", "tensor")


@dataclass(frozen=True)
class ExpertTensorSpec:
    """One streamed tensor's per-expert row: ``row_shape`` elements of ``dtype``.

    ``residence`` is where rows come from: ``host`` rows are read into host
    memory (the pinned tier, pinned staging, or a registered arena), and
    ``device`` rows are indexed from a CUDA source. Immutable; construction
    validates the name, the dimensions, the dtype and the residence.
    """

    name: str
    row_shape: tuple[int, ...]
    dtype: torch.dtype
    residence: Literal["host", "device"]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_shape", tuple(int(dimension) for dimension in self.row_shape)
        )
        if not self.name:
            raise ValueError("expert tensor spec needs a name")
        if any(dimension < 0 for dimension in self.row_shape):
            raise ValueError(
                f"expert tensor spec {self.name!r} has a negative dimension"
            )
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(
                f"expert tensor spec {self.name!r} dtype must be a torch.dtype"
            )
        if self.residence not in ("host", "device"):
            raise ValueError(
                f"expert tensor spec {self.name!r} residence must be 'host' or 'device'"
            )

    @property
    def row_bytes(self) -> int:
        """Bytes in one expert's row."""
        return math.prod(self.row_shape) * self.dtype.itemsize


class DenseLayerFormat:
    """Every streamed tensor is a dense ``[experts, ...]`` attribute of the layer.

    This is the NVFP4 format and the behaviour of every streamer created without a
    format: the dense tensors may live on CPU or CUDA, and the pinned host tier
    gets no extra options. The tensors must be contiguous, non-empty and agree on
    the expert count; ``num_experts`` checks this. The native engine does not serve it
    yet, so it has no transport build, no CPU kernel and no draft.
    """

    key = "dense"
    transport_build: Optional[str] = None
    engine_served = False
    supports_graph_gather = True
    supports_host_arena = True
    supports_cpu_experts = False
    supports_draft = False
    supports_dead_rows = False
    max_gather_rows: Optional[int] = None
    inclusive_pinned_tier = False

    def __init__(self, tensor_names: Iterable[str]):
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")
        # The transfer executor submits one expert's rows as a fixed bundle of NVFP4_TRANSFER_TENSOR_COUNT copies.
        self.supports_transfer_executor = (
            len(self.tensor_names) == NVFP4_TRANSFER_TENSOR_COUNT
        )

    def layer_dims(self, layer: torch.nn.Module) -> tuple[int, int]:
        raise NotImplementedError(
            "a dense format is attached by its quantization method, not by the expert engine"
        )

    def build_streamer(self, layer: torch.nn.Module) -> Any:
        raise NotImplementedError(
            "a dense format is attached by its quantization method, not by the expert engine"
        )

    def cpu_trait(self, *, source: str, ext: Any = None) -> None:
        return None

    def prepare_host_rows(
        self, layer: torch.nn.Module, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor

    def moe_apply(self, layer, x, slot_ids, weights, slot_tensors) -> torch.Tensor:
        raise NotImplementedError(
            "a dense format applies its experts through its quantization method, not over slot buffers"
        )

    @staticmethod
    def _dense(layer: torch.nn.Module, name: str) -> torch.Tensor:
        """The layer attribute ``name`` as a plain tensor (unwrapping parameters)."""
        value = getattr(layer, name)
        return value.data if isinstance(value, torch.nn.Parameter) else value

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return self._dense(layer, name)

    def num_experts(self, layer: torch.nn.Module) -> int:
        """The shared expert count; raises if a tensor is missing or malformed."""
        expert_count = None
        for name in self.tensor_names:
            if not hasattr(layer, name):
                raise ValueError(f"expert source tensor {name!r} is missing")
            tensor = self._dense(layer, name)
            if tensor.ndim == 0:
                raise ValueError(
                    f"expert source tensor {name!r} has no expert dimension"
                )
            if tensor.shape[0] == 0:
                raise ValueError(f"expert source tensor {name!r} has no expert rows")
            if not tensor.is_contiguous():
                raise ValueError(f"expert source tensor {name!r} must be contiguous")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"expert source tensor {name!r} uses unsupported device {tensor.device}"
                )
            if expert_count is None:
                expert_count = tensor.shape[0]
            elif tensor.shape[0] != expert_count:
                raise ValueError(
                    f"expert count mismatch for {name!r}: {tensor.shape[0]} != {expert_count}"
                )
        assert expert_count is not None
        return expert_count

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        self.num_experts(layer)
        specs = []
        for name in self.tensor_names:
            tensor = self._dense(layer, name)
            specs.append(
                ExpertTensorSpec(
                    name,
                    tuple(tensor.shape[1:]),
                    tensor.dtype,
                    "host" if tensor.device.type == "cpu" else "device",
                )
            )
        return tuple(specs)

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional[ExpertRowSource]:
        # Imported here: the reader pulls in sglang.srt.model_loader, whose package
        # import reaches modelopt_quant, which imports expert_stream.
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader
        from sglang.srt.model_loader.file_row_reader import validate_file_reader_mode

        names = tuple(spec.name for spec in specs)
        if kind == "auto":
            return ExpertFileRowReader.from_layer(layer, names)
        if kind == "files":
            mode = validate_file_reader_mode(stream_config().tiers.file_reader)
            if mode == "mmap":
                raise ValueError(
                    "SGLANG_EXPERT_STREAM_ROW_SOURCE=files needs "
                    "SGLANG_EXPERT_STREAM_FILE_READER=uring or uring_direct"
                )
            return ExpertFileRowReader.from_layer(layer, names, mode=mode)
        if kind == "tensor":
            return None
        raise ValueError(
            f"expert format {self.key!r} has no row source kind {kind!r}; "
            f"choose from {GENERIC_ROW_SOURCE_KINDS}"
        )

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional[ExpertRowSource]
    ) -> Optional[int]:
        """The bytes the NVFP4 method verified, or None.

        Only that attribute enables file attribution and the eager pinned tier.
        """
        return getattr(layer, FILE_SOURCE_BYTES_ATTRIBUTE, None)

    def pinned_tier_options(self, layer: torch.nn.Module) -> Mapping[str, Any]:
        """No options: the dense format builds a default pinned tier."""
        return {}


def expert_streamer_of(module: torch.nn.Module) -> Optional[ExpertStreamer]:
    """The expert streamer attached to ``module``, or None."""
    return getattr(module, STREAMER_ATTRIBUTE, None)


def iter_expert_streamers(model: torch.nn.Module) -> Iterator[ExpertStreamer]:
    """Every attached expert streamer of ``model``, in module order."""
    for module in model.modules():
        streamer = expert_streamer_of(module)
        if streamer is not None:
            yield streamer


def resolve_row_source_kind(config: Optional[ExpertStreamConfig] = None) -> str:
    """The row source kind ``SGLANG_EXPERT_STREAM_ROW_SOURCE`` selects (default ``auto``); ``config`` defaults to
    ``stream_config()``.

    Raises if the variable is blank.
    """
    kind = (stream_config() if config is None else config).tiers.row_source.strip()
    if not kind:
        raise ValueError("SGLANG_EXPERT_STREAM_ROW_SOURCE must name a row source kind")
    return kind


def graph_source_kind_of(expert_format: Any) -> str:
    """Where a format's graph gathers read host rows.

    ``"dense"`` means ``[experts, ...]`` layer tensors or the host arena, indexed by
    expert id; ``"pinned_tier"`` means the layer's pinned host tier, indexed by
    pinned slot.
    """
    return getattr(expert_format, "graph_source_kind", "dense")


def graph_gather_needs_host_arena(model: torch.nn.Module) -> bool:
    """Whether ``model``'s graph gathers need ``SGLANG_EXPERT_STREAM_ENABLE_HOST_ARENA``.

    True unless every attached streamer's format serves them from its pinned host
    tier.
    """
    return any(
        graph_source_kind_of(streamer.format) != "pinned_tier"
        for streamer in iter_expert_streamers(model)
    )


def require_graph_gather_support(
    streamers: Iterable[ExpertStreamer],
    *,
    pinned_tier_ok: bool = False,
    engine_direct_ok: bool = False,
) -> None:
    """Raise unless every streamer's format can serve sync-free graph gathers.

    Dense formats need dense, GPU-readable host sources frozen at startup, which
    spec-only tensors lack. A ``pinned_tier`` format serves graph gathers from its
    pinned host tier instead; only the plain graph gather supports that
    (``pinned_tier_ok``), not the GPU residency update, whose copies index host rows
    by expert id. ``engine_direct_ok`` admits an ``engine_served`` format on the same terms.
    """
    for streamer in streamers:
        expert_format = streamer.format
        key = expert_format.key
        if (
            pinned_tier_ok or (engine_direct_ok and expert_format.engine_served)
        ) and graph_source_kind_of(expert_format) == "pinned_tier":
            if streamer.pinned_host_cache is None:
                raise ValueError(
                    f"expert format {key!r} of layer {streamer.layer_id} serves graph "
                    "gathers from its pinned host tier; set SGLANG_EXPERT_STREAM_PINNED_HOST_MB"
                )
            continue
        unsupported = (
            not expert_format.supports_graph_gather or streamer.has_spec_only_tensors
        )
        if unsupported:
            raise ValueError(
                f"expert format {key!r} of layer {streamer.layer_id} does not support "
                "graph gather; unset SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER and "
                "SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE"
            )


def pinned_tier_options_of(
    expert_format: Any, layer: torch.nn.Module
) -> Mapping[str, Any]:
    """``expert_format.pinned_tier_options(layer)``, or no options without the hook.

    The protocol requires the hook; a format missing it gets a default pinned
    tier (no ``is_pinned`` filter) and one warning, instead of failing startup.
    """
    hook = getattr(expert_format, "pinned_tier_options", None)
    if hook is None:
        key = str(getattr(expert_format, "key", type(expert_format).__name__))
        if key not in _WARNED_WITHOUT_TIER_OPTIONS:
            _WARNED_WITHOUT_TIER_OPTIONS.add(key)
            logger.warning(
                "expert format %r has no pinned_tier_options; its pinned host tier "
                "gets default options (no is_pinned filter)",
                key,
            )
        return {}
    return hook(layer)


def inclusive_hot_slot_limit(streamer: ExpertStreamer) -> Optional[int]:
    """The most hot-cache slots a layer may hold when its pinned tier is inclusive.

    An inclusive pinned tier keeps every hot expert in host memory too,
    protected from eviction, and an eager gather needs room for up to
    ``max_gather_rows`` more rows beside them; so the layer may hold at most
    ``pinned rows - max_gather_rows`` hot slots (never below 0). None when the
    format does not set ``inclusive_pinned_tier`` or the layer has no pinned tier.

    Raises if the format sets ``inclusive_pinned_tier`` but has no positive
    ``max_gather_rows``: without a cap, the hot cache may pin the whole tier,
    driving ``evictable_rows()`` to 0 and raising "every pinned host slot
    holds a protected expert" at the first miss instead of at startup.
    """
    expert_format = streamer.format
    if not getattr(expert_format, "inclusive_pinned_tier", False):
        return None
    max_gather_rows = expert_format.max_gather_rows
    if not max_gather_rows or max_gather_rows <= 0:
        key = getattr(expert_format, "key", type(expert_format).__name__)
        raise ValueError(
            f"expert format {key!r} sets inclusive_pinned_tier but has no positive "
            "max_gather_rows; an inclusive pinned tier needs a cap so eager gathers "
            "always have room beside the pinned rows"
        )
    cache = streamer.pinned_host_cache
    if cache is None or not cache.capacity:
        return None
    return max(cache.capacity - max_gather_rows, 0)
