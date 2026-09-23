"""The EXL3 routed-expert format plugin of the expert streaming framework.

An EXL3 expert row on disk is 12 tensors back to back (w1/w2/w3 x
suh/svh/mul1/trellis). The framework streams six per-name tensors instead:
w1 and w3 stack into the ``w13_*`` rows as parts 0 and 1, w2 is part 0 of the
``w2_*`` rows, and the three ``mul1`` scalars are dropped (the codebook is
always mul1, so ``Exl3Tensors`` takes ``mul1=True``). ``segment_map`` says
where every streamed byte sits in the on-disk row, so a row source can split
one superset read into the six per-name rows.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.expert_format import ExpertTensorSpec, expert_streamer_of

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource

EXL3_STREAMED_NAMES = (
    "w13_trellis",
    "w13_suh",
    "w13_svh",
    "w2_trellis",
    "w2_suh",
    "w2_svh",
)
# Rows one eager gather stages: 64 x 13.3 MB = 852 MB of VRAM staging.
EXL3_MAX_GATHER_ROWS = 64
_DTYPES = {"I16": torch.int16, "F16": torch.float16}
# On-disk linear -> (streamed prefix, part of the row's leading dimension).
_PARTS = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}
_KINDS = ("trellis", "suh", "svh")


@dataclass(frozen=True)
class RowSegment:
    """``nbytes`` at ``src_offset`` of the on-disk row fill part ``part`` of
    streamed tensor ``name``'s row, at byte ``dst_offset`` of that row."""

    name: str
    part: int
    dst_offset: int
    src_offset: int
    nbytes: int


def _row_schema(
    layout: Exl3ExpertLayout,
) -> tuple[tuple[ExpertTensorSpec, ...], tuple[RowSegment, ...]]:
    spans = {span.name: span for span in layout.tensors}
    expected = {f"{w}.{kind}" for w in _PARTS for kind in _KINDS + ("mul1",)}
    if set(spans) != expected:
        raise ValueError(
            f"exl3 expert rows hold {sorted(spans)}, expected {sorted(expected)}"
        )
    for w in _PARTS:
        mul1 = spans[f"{w}.mul1"]
        if mul1.dtype != "I32" or tuple(mul1.shape) != ():
            raise ValueError(f"exl3 {mul1.name} is {mul1.dtype} {mul1.shape}, expected I32 []")
    specs: dict[str, ExpertTensorSpec] = {}
    segments = []
    for w, (prefix, part) in _PARTS.items():
        for kind in _KINDS:
            span = spans[f"{w}.{kind}"]
            dtype = _DTYPES.get(span.dtype)
            shape = tuple(span.shape)
            if dtype is None or math.prod(shape) * dtype.itemsize != span.nbytes:
                raise ValueError(
                    f"exl3 expert tensor {span.name}: unexpected {span.dtype} "
                    f"{shape} ({span.nbytes} bytes)"
                )
            name = f"{prefix}_{kind}"
            parts = 2 if prefix == "w13" else 1
            spec = ExpertTensorSpec(name, (parts,) + shape, dtype, "host")
            previous = specs.setdefault(name, spec)
            if previous != spec:
                raise ValueError(
                    f"exl3 {name}: w1 and w3 disagree "
                    f"({previous.row_shape} vs {spec.row_shape})"
                )
            segments.append(
                RowSegment(name, part, part * span.nbytes, span.rel_offset, span.nbytes)
            )
    return (
        tuple(specs[name] for name in EXL3_STREAMED_NAMES),
        tuple(sorted(segments, key=lambda segment: segment.src_offset)),
    )


class Exl3ExpertFormat:
    """``ExpertFormat`` for one layer of an EXL3 checkpoint's routed experts.

    Spec-only: there is no dense ``[experts, ...]`` source, so every host row
    comes from the row source. ``direct`` picks O_DIRECT shard reads; None
    takes it from ``SGLANG_MOE_EXPERT_FILE_READER``.
    """

    key = "exl3"
    supports_graph_gather = False
    supports_host_arena = False
    # Graph gathers read missed rows from the pinned host tier by pinned slot
    # (PinnedTierRowBackend); there is no dense [experts, ...] host source.
    graph_source_kind = "pinned_tier"
    max_gather_rows: Optional[int] = EXL3_MAX_GATHER_ROWS
    names = EXL3_STREAMED_NAMES

    def __init__(
        self,
        layout: Exl3ExpertLayout,
        layer_id: int,
        *,
        direct: Optional[bool] = None,
        source_root: Optional[str] = None,
    ) -> None:
        if not 0 <= layer_id < layout.num_layers:
            raise ValueError(
                f"exl3 layer {layer_id} is outside the checkpoint's "
                f"{layout.num_layers} layers"
            )
        self.layout = layout
        self.layer_id = layer_id
        self.direct = direct
        self.source_root = source_root
        self._specs, self._segments = _row_schema(layout)

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        return self._specs

    def num_experts(self, layer: torch.nn.Module) -> int:
        return self.layout.num_experts

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return None

    def segment_map(self) -> tuple[RowSegment, ...]:
        return self._segments

    # Inclusive hierarchy (DSV41_REFERENCE §9.1): every expert the layer's hot
    # cache holds keeps its pinned host row, so a VRAM eviction never costs a
    # shard read. For a format that sets this, the framework clamps each layer's
    # hot slots to its pinned rows minus max_gather_rows.
    inclusive_pinned_tier = True

    def pinned_tier_options(self, layer: torch.nn.Module) -> dict:
        """Keyword arguments for this layer's pinned host tier: ``is_pinned``.

        An expert is pinned while the layer's hot cache holds it, or is loading
        it, into a slot. The callable looks the hot cache up at eviction time:
        the pinned tier is built before the hot cache, and residency changes.
        """

        def is_pinned(expert_id: int) -> bool:
            # Called O(pinned rows) times per miss chunk; each call scans the
            # layer's ~32 hot slots in C. That is a few ms per 40-layer forward,
            # accepted for eager 3a; a framework-side resident set is a 3b item.
            streamer = expert_streamer_of(layer)
            hot = None if streamer is None else streamer.hot_cache
            if hot is not None:
                from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService

                service = Exl3RamMissService.get()
                if service.gpu_hot_enabled:
                    return expert_id in service.hot_experts(self.layer_id)
            return hot is not None and expert_id in hot.slot_to_expert

        options = {"is_pinned": is_pinned}
        if envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.get():
            # Option C: the C++ RAM-miss thread owns this tier's slots (plan D12).
            # ExpertPinnedHostCache binds the tier's capacity into the table.
            from sglang.srt.layers.moe.exl3_ram_miss import (
                Exl3RamMissService,
                NativePinnedSlotTable,
            )

            options["slot_table"] = NativePinnedSlotTable(
                Exl3RamMissService.get(), self.layer_id, lambda: expert_streamer_of(layer)
            )
        return options

    def attach_hot_cache_manager(self, manager, streamer) -> None:
        """Option C hooks (fail-stop check, residency pushes, the RAM-miss row backend),
        for a layer whose pinned tier runs on the native slot table."""
        from sglang.srt.layers.moe.exl3_ram_miss import (
            Exl3RamMissService,
            NativePinnedSlotTable,
        )

        tier = getattr(streamer, "pinned_host_cache", None)
        if not isinstance(getattr(tier, "_lru", None), NativePinnedSlotTable):
            return
        Exl3RamMissService.get().attach(manager, streamer)

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional["ExpertRowSource"]:
        """``auto`` and ``shards`` read the original EXL3 shards, or, when
        ``SGLANG_MOE_EXPERT_MIRROR_DIRS`` is set, the mirrored copies of them."""
        if kind in ("auto", "shards"):
            mirror = exl3_mirror_config()
            if mirror is not None:
                return self._mirror_row_source(*mirror)
            # Imported here: the source pulls in sglang.srt.model_loader, whose
            # package import reaches the quantization methods that import this module.
            from sglang.srt.layers.moe.exl3_shard_row_source import (
                Exl3ShardRowSource,
            )

            return Exl3ShardRowSource.for_layer(
                self.layout, self.layer_id, self._segments, direct=self._resolve_direct()
            )
        raise ValueError(
            f"expert format {self.key!r} has no row source kind {kind!r}; "
            "choose from ('auto', 'shards')"
        )

    def _mirror_row_source(
        self, roots: tuple[str, ...], weights: tuple[float, ...]
    ) -> "ExpertRowSource":
        self._check_mirror_roots(roots)
        from sglang.srt.layers.moe.exl3_mirror_row_source import Exl3MirrorRowSource

        return Exl3MirrorRowSource.for_mirrored_layer(
            self.layout,
            self.layer_id,
            self._segments,
            direct=self._resolve_direct(),
            roots=roots,
            policy=StaticSplitPolicy(weights),
            source_root=self.source_root,
        )

    def _check_mirror_roots(self, roots: tuple[str, ...]) -> None:
        """What the eager and the native reader both need of the roots, beyond parsing them."""
        if self.source_root is None:
            raise ValueError(
                "SGLANG_MOE_EXPERT_MIRROR_DIRS needs the format built with "
                "source_root, the checkpoint directory the layout was read from "
                "(SGLANG_DSV41_EXPERT_DIR), to find each root's copy of a shard"
            )
        for root in roots:
            if os.path.realpath(root) == os.path.realpath(self.source_root):
                raise ValueError(
                    f"{_MIRROR_DIRS}: {root!r} is the checkpoint directory "
                    f"{self.source_root!r} itself, not a mirror of it"
                )

    def mirror_table_args(self) -> dict:
        """The mirror keyword arguments of ``exl3_ram_miss_tables`` (the native reader's tables):
        empty without ``SGLANG_MOE_EXPERT_MIRROR_DIRS``, else the same validated roots and split
        policy ``default_row_source`` builds its mirror source from."""
        mirror = exl3_mirror_config()
        if mirror is None:
            return {}
        roots, weights = mirror
        self._check_mirror_roots(roots)
        return dict(roots=roots, policy=StaticSplitPolicy(weights), source_root=self.source_root)

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]:
        # Non-None turns on the eager pinned host tier and the file counters.
        return None if row_source is None else row_source.file_bytes_per_expert

    def _resolve_direct(self) -> bool:
        if self.direct is not None:
            return self.direct
        from sglang.srt.model_loader.file_row_reader import validate_file_reader_mode

        mode = validate_file_reader_mode(envs.SGLANG_MOE_EXPERT_FILE_READER.get())
        if mode == "mmap":
            raise ValueError(
                "the exl3 shard row source reads with io_uring; set "
                "SGLANG_MOE_EXPERT_FILE_READER=uring_direct (or uring for buffered reads)"
            )
        return mode == "uring_direct"


_MIRROR_DIRS = "SGLANG_MOE_EXPERT_MIRROR_DIRS"
_MIRROR_WEIGHTS = "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS"


def parse_mirror_roots(value: str) -> tuple[str, ...]:
    """The mirror roots in an ``os.pathsep``-separated ``SGLANG_MOE_EXPERT_MIRROR_DIRS``.

    Each must be a readable directory. An empty entry (``a::b``, a trailing
    separator) is refused rather than dropped: it changes the root count the
    weights are matched against, and is far likelier a typo than intent.
    """
    entries = value.split(os.pathsep)
    if any(not entry for entry in entries):
        raise ValueError(
            f"{_MIRROR_DIRS}={value!r} has an empty entry; separate roots with "
            f"{os.pathsep!r} and no others"
        )
    for root in entries:
        if not os.path.isabs(root):
            raise ValueError(
                f"{_MIRROR_DIRS}: {root!r} is a relative path, which would mean "
                "a different directory in every working directory; give an absolute one"
            )
        if not (os.path.isdir(root) and os.access(root, os.R_OK | os.X_OK)):
            raise ValueError(f"{_MIRROR_DIRS}: {root!r} is not a readable directory")
    real = [os.path.realpath(root) for root in entries]
    for i, path in enumerate(real):
        if path in real[:i]:
            raise ValueError(
                f"{_MIRROR_DIRS}: {entries[i]!r} is the same directory as "
                f"{entries[real.index(path)]!r}; two entries for one drive would "
                "look like a two-drive mirror and read from one"
            )
    return tuple(entries)


def parse_mirror_weights(value: str, num_roots: int) -> tuple[float, ...]:
    """The split weights, one per root; empty means equal weights."""
    if not value.strip():
        return (1.0,) * num_roots
    weights = []
    for token in value.split(":"):
        try:
            weight = float(token)
        except ValueError:
            raise ValueError(
                f"{_MIRROR_WEIGHTS}={value!r}: {token!r} is not a number; give "
                "colon-separated non-negative numbers such as 3:1"
            ) from None
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{_MIRROR_WEIGHTS}={value!r}: {token!r} is not a finite "
                "non-negative number"
            )
        weights.append(weight)
    if len(weights) != num_roots:
        raise ValueError(
            f"{_MIRROR_WEIGHTS} lists {len(weights)} weights but {_MIRROR_DIRS} "
            f"lists {num_roots} roots"
        )
    if not any(weights):
        raise ValueError(f"{_MIRROR_WEIGHTS}={value!r}: every weight is zero")
    return tuple(weights)


def exl3_mirror_config() -> Optional[tuple[tuple[str, ...], tuple[float, ...]]]:
    """``(roots, weights)`` from the environment, or None when mirroring is off."""
    dirs = envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.get()
    weights = envs.SGLANG_MOE_EXPERT_MIRROR_WEIGHTS.get()
    if not dirs:
        if weights.strip():
            raise ValueError(
                f"{_MIRROR_WEIGHTS} is set but {_MIRROR_DIRS} is not; the weights "
                "would be silently ignored"
            )
        return None
    roots = parse_mirror_roots(dirs)
    return roots, parse_mirror_weights(weights, len(roots))


logger = logging.getLogger(__name__)
_WARNED_WITHOUT_PINNED_TIER = False


def prefetch_enabled() -> bool:
    """Option F advisories (``SGLANG_DSV41_ENABLE_EXPERT_PREFETCH``)."""
    return envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.get()


@functools.lru_cache(maxsize=4)
def exl3_expert_layout_for(expert_dir: str) -> Exl3ExpertLayout:
    """The checkpoint's expert layout, read once per directory (headers only)."""
    return build_exl3_expert_layout(expert_dir)


def build_exl3_expert_streamer(layer: torch.nn.Module, expert_dir: Optional[str] = None):
    """An ``ExpertStreamer`` for an EXL3 MoE layer whose routed experts stay on disk.

    ``expert_dir`` defaults to ``SGLANG_DSV41_EXPERT_DIR``. The row source comes
    from ``SGLANG_MOE_EXPERT_ROW_SOURCE`` (``auto`` means the shards).
    """
    # Imported here: expert_stream imports Triton kernels and the model loader.
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

    global _WARNED_WITHOUT_PINNED_TIER
    expert_dir = expert_dir or envs.SGLANG_DSV41_EXPERT_DIR.get()
    if not expert_dir:
        raise ValueError("SGLANG_DSV41_EXPERT_STREAM needs SGLANG_DSV41_EXPERT_DIR")
    if not envs.SGLANG_MOE_PINNED_HOST_MB.get() and not _WARNED_WITHOUT_PINNED_TIER:
        # SGLANG_DSV41_EXPERT_RAM_GIB is retired; an old launch script setting it
        # would otherwise run with no RAM tier and no sign of it.
        _WARNED_WITHOUT_PINNED_TIER = True
        logger.warning(
            "EXL3 expert streaming without SGLANG_MOE_PINNED_HOST_MB: there is no "
            "host RAM tier, so every VRAM miss reads the shards"
        )
    layout = exl3_expert_layout_for(os.path.realpath(expert_dir))
    if layout.num_experts != layer.exl3_num_experts:
        raise ValueError(
            f"exl3 streaming: the layer has {layer.exl3_num_experts} experts, the "
            f"checkpoint {layout.num_experts}; launch with disable_shared_experts_fusion=True"
        )
    fmt = Exl3ExpertFormat(
        layout, layer.layer_id, source_root=os.path.realpath(expert_dir)
    )
    specs = {spec.name: spec.row_shape for spec in fmt.tensor_specs(layer)}
    hidden, inter = layer.exl3_hidden // 16, layer.exl3_inter // 16
    if specs["w13_trellis"][:3] != (2, hidden, inter) or specs["w2_trellis"][:3] != (1, inter, hidden):
        raise ValueError(
            f"exl3 streaming: expert trellis rows {specs['w13_trellis']} / "
            f"{specs['w2_trellis']} do not match the layer "
            f"({layer.exl3_hidden} -> {layer.exl3_inter})"
        )
    return ExpertStreamer(layer, fmt.names, layer_id=layer.layer_id, format=fmt)
