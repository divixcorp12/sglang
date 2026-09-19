"""Expert tensor formats: the row schema and dense sources an expert streamer reads.

A format tells :class:`~sglang.srt.layers.moe.expert_stream.ExpertStreamer`
which tensors one expert row holds (:class:`ExpertTensorSpec`), where each
tensor's dense ``[experts, ...]`` source lives if it has one, and which row
source fills host rows. :class:`DenseLayerFormat` is the behaviour every
streamer had before formats existed: every tensor is a layer attribute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
)

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

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
    ``device`` rows are indexed from a CUDA source.
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
            raise ValueError(f"expert tensor spec {self.name!r} has a negative dimension")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"expert tensor spec {self.name!r} dtype must be a torch.dtype")
        if self.residence not in ("host", "device"):
            raise ValueError(
                f"expert tensor spec {self.name!r} residence must be 'host' or 'device'"
            )

    @property
    def row_bytes(self) -> int:
        return math.prod(self.row_shape) * self.dtype.itemsize


class ExpertFormat(Protocol):
    """What an expert streamer needs to know about one layer's expert tensors.

    ``tensor_specs`` returns one spec per streamed tensor, in streamer order.
    ``source`` returns the dense ``[experts, ...]`` tensor of a name at call
    time (the host arena rebinds layer tensors after startup), or None when
    the format has no dense source and only its row source can read rows.
    ``default_row_source`` builds the row source for a knob kind (see
    ``SGLANG_MOE_EXPERT_ROW_SOURCE``) and raises for kinds it does not know.
    ``file_source_bytes_per_expert`` returns the file bytes one expert row
    reads, or None; None keeps eager gathers out of the pinned host tier and
    reports file counters as unknown.
    """

    key: str
    supports_graph_gather: bool
    supports_host_arena: bool
    max_gather_rows: Optional[int]

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]: ...

    def num_experts(self, layer: torch.nn.Module) -> int: ...

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]: ...

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional["ExpertRowSource"]: ...

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]: ...


class DenseLayerFormat:
    """Every streamed tensor is a dense ``[experts, ...]`` attribute of the layer.

    This is the NVFP4 format and the behaviour of every streamer created
    without a format.
    """

    key = "dense"
    supports_graph_gather = True
    supports_host_arena = True
    max_gather_rows: Optional[int] = None

    def __init__(self, tensor_names: Iterable[str]):
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")

    @staticmethod
    def _dense(layer: torch.nn.Module, name: str) -> torch.Tensor:
        value = getattr(layer, name)
        return value.data if isinstance(value, torch.nn.Parameter) else value

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return self._dense(layer, name)

    def num_experts(self, layer: torch.nn.Module) -> int:
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
    ) -> Optional["ExpertRowSource"]:
        # Imported here: the reader pulls in sglang.srt.model_loader, whose
        # package import reaches modelopt_quant, which imports expert_stream.
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader
        from sglang.srt.model_loader.file_row_reader import validate_file_reader_mode

        names = tuple(spec.name for spec in specs)
        if kind == "auto":
            return ExpertFileRowReader.from_layer(layer, names)
        if kind == "files":
            mode = validate_file_reader_mode(envs.SGLANG_MOE_EXPERT_FILE_READER.get())
            if mode == "mmap":
                raise ValueError(
                    "SGLANG_MOE_EXPERT_ROW_SOURCE=files needs "
                    "SGLANG_MOE_EXPERT_FILE_READER=uring or uring_direct"
                )
            return ExpertFileRowReader.from_layer(layer, names, mode=mode)
        if kind == "tensor":
            return None
        raise ValueError(
            f"expert format {self.key!r} has no row source kind {kind!r}; "
            f"choose from {GENERIC_ROW_SOURCE_KINDS}"
        )

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]:
        # Exactly the pre-format gate: only the NVFP4 method's verified
        # attribute enables file attribution and the eager pinned tier.
        return getattr(layer, FILE_SOURCE_BYTES_ATTRIBUTE, None)


def expert_streamer_of(module: torch.nn.Module) -> Optional["ExpertStreamer"]:
    """The expert streamer attached to ``module``, or None."""
    return getattr(module, STREAMER_ATTRIBUTE, None)


def iter_expert_streamers(model: torch.nn.Module) -> Iterator["ExpertStreamer"]:
    """Every attached expert streamer of ``model``, in module order."""
    for module in model.modules():
        streamer = expert_streamer_of(module)
        if streamer is not None:
            yield streamer


def resolve_row_source_kind() -> str:
    """The row source kind ``SGLANG_MOE_EXPERT_ROW_SOURCE`` selects (default ``auto``)."""
    kind = envs.SGLANG_MOE_EXPERT_ROW_SOURCE.get().strip()
    if not kind:
        raise ValueError("SGLANG_MOE_EXPERT_ROW_SOURCE must name a row source kind")
    return kind
