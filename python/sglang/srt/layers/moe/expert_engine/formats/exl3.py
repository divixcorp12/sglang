"""The EXL3 routed-expert format plugin of the expert streaming framework.

An EXL3 expert row on disk is 12 tensors back to back (w1/w2/w3 x
suh/svh/mul1/trellis). The framework streams six per-name tensors instead: w1 and
w3 stack into the ``w13_*`` rows as parts 0 and 1, w2 is part 0 of the ``w2_*``
rows, and the three ``mul1`` scalars are dropped (the codebook is always mul1, so
``Exl3Tensors`` takes ``mul1=True``).

This file holds:

* :class:`RowSegment` and ``_row_schema``, which derive the streamed tensor specs
  and the byte map from an on-disk layout.
* :class:`Exl3ExpertFormat`, the ``ExpertFormat`` of one layer.
* The parsers of ``SGLANG_EXPERT_STREAM_MIRROR``'s ``dirs`` and ``weights``.
* :func:`build_exl3_expert_streamer`, the entry point a quantization method calls.

See DSV41_REFERENCE.md section 9.1 for the pinned-tier budget these formats use.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    Exl3RowGeometry,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.expert_engine.config import ExpertStreamConfig, stream_config
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
    """One contiguous byte run of the on-disk row and where it lands.

    ``nbytes`` at ``src_offset`` of the on-disk row fill part ``part`` of streamed
    tensor ``name``'s row, at byte ``dst_offset`` of that row. The full set (see
    ``Exl3ExpertFormat.segment_map``) lets a row source split one superset read
    into the six per-name rows.
    """

    name: str
    part: int
    dst_offset: int
    src_offset: int
    nbytes: int


def _row_schema(
    geometry: Exl3RowGeometry,
) -> tuple[tuple[ExpertTensorSpec, ...], tuple[RowSegment, ...]]:
    """Validate one layer's row ``geometry`` and derive the streamed specs and the byte map.

    Returns the six specs in ``EXL3_STREAMED_NAMES`` order and the segments sorted
    by on-disk offset. Raises ``ValueError`` if the row is not the expected 12
    tensors or w1 and w3 disagree (they share the w13 slabs); w2's K may differ.
    """
    spans = {span.name: span for span in geometry.tensors}
    expected = {f"{w}.{kind}" for w in _PARTS for kind in _KINDS + ("mul1",)}
    if set(spans) != expected:
        raise ValueError(
            f"exl3 expert rows hold {sorted(spans)}, expected {sorted(expected)}"
        )
    for w in _PARTS:
        mul1 = spans[f"{w}.mul1"]
        if mul1.dtype != "I32" or tuple(mul1.shape) != ():
            raise ValueError(
                f"exl3 {mul1.name} is {mul1.dtype} {mul1.shape}, expected I32 []"
            )
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


def layer_segments(layout: Exl3ExpertLayout, layer: int) -> tuple[RowSegment, ...]:
    """Layer ``layer``'s segment map: where each streamed byte of its on-disk row sits."""
    return _row_schema(layout.geometry(layer))[1]


class Exl3ExpertFormat:
    """``ExpertFormat`` for one layer of an EXL3 checkpoint's routed experts.

    Spec-only: there is no dense ``[experts, ...]`` source, so every host row comes
    from the row source. ``direct`` picks O_DIRECT shard reads; None takes it from
    ``SGLANG_EXPERT_STREAM_FILE_READER``. ``source_root`` is the checkpoint directory
    the layout was read from; mirrored reads need it to find each root's copy of a
    shard.

    Graph gathers read missed rows from the pinned host tier by pinned slot
    (``PinnedTierRowBackend``), since there is no dense host source to index.

    The pinned tier is inclusive (DSV41_REFERENCE.md section 9.1): every expert the
    layer's hot cache holds keeps its pinned host row, so a VRAM eviction never
    costs a shard read. For a format that sets ``inclusive_pinned_tier``, the
    framework clamps each layer's hot slots to its pinned rows minus
    ``max_gather_rows``.
    """

    key = "exl3"
    transport_build = "exl3"
    engine_served = True
    supports_graph_gather = False
    supports_host_arena = False
    # The six streamed names are exactly the transfer executor's bundle of row copies.
    supports_transfer_executor = True
    supports_cpu_experts = True
    supports_draft = True
    supports_dead_rows = True
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
        self._specs, self._segments = _row_schema(layout.geometry(layer_id))

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        return self._specs

    def num_experts(self, layer: torch.nn.Module) -> int:
        return self.layout.num_experts

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return None

    def segment_map(self) -> tuple[RowSegment, ...]:
        """Where every streamed byte sits in the on-disk row, by source offset."""
        return self._segments

    inclusive_pinned_tier = True

    def pinned_tier_options(self, layer: torch.nn.Module) -> dict:
        """Keyword arguments for this layer's pinned host tier.

        Always ``is_pinned``: an expert is pinned while the layer's hot cache holds
        it, or is loading it, into a slot. The callable looks the hot cache up at
        eviction time, because the pinned tier is built before the hot cache and
        residency changes. With ``SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER`` it also returns
        ``slot_table`` (and ``row_fills`` under
        ``SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS``).
        """
        copy = stream_config().copy

        def is_pinned(expert_id: int) -> bool:
            # Called O(pinned rows) times per miss chunk; each call scans the
            # layer's ~32 hot slots in C. That costs a few ms per 40-layer eager
            # forward, accepted until the framework keeps a resident set.
            streamer = expert_streamer_of(layer)
            hot = None if streamer is None else streamer.hot_cache
            if hot is not None:
                from sglang.srt.layers.moe.expert_engine.service import (
                    ExpertEngineService,
                )

                service = ExpertEngineService.get()
                if service.gpu_hot_enabled:
                    return expert_id in service.hot_experts(self.layer_id)
            return hot is not None and expert_id in hot.slot_to_expert

        options = {"is_pinned": is_pinned}
        if copy.graph_gather:
            # The C++ RAM-miss thread owns this tier's slots;
            # ExpertPinnedHostCache binds the tier's capacity into the table.
            from sglang.srt.layers.moe.expert_engine.row_backend import (
                NativePinnedSlotTable,
            )
            from sglang.srt.layers.moe.expert_engine.service import (
                ExpertEngineService,
            )

            options["slot_table"] = NativePinnedSlotTable(
                ExpertEngineService.get(),
                self.layer_id,
                lambda: expert_streamer_of(layer),
            )
            if copy.prefill.fills:
                options["row_fills"] = options["slot_table"]
        elif copy.prefill.fills:
            raise RuntimeError(
                "exl3: SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS needs SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER: the "
                "fills are the RAM-miss service's reads, and only option C runs that service"
            )
        return options

    def plan_graph_gather(self, streamer, rows: int, routes: int = 0) -> None:
        """Tell the RAM-miss service the layer's graph gather misses up to ``rows`` ids (of ``routes`` routes).

        Called before the first tier fill, while the service reserves its staging
        slots. A no-op unless the tier runs on the native slot table.
        """
        from sglang.srt.layers.moe.expert_engine.row_backend import (
            NativePinnedSlotTable,
        )
        from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService

        tier = getattr(streamer, "pinned_host_cache", None)
        if isinstance(getattr(tier, "_lru", None), NativePinnedSlotTable):
            service = ExpertEngineService.get()
            service.plan_gather_width(rows, routes)
            victims = service.config.stream.copy.graph_gather_victim_lanes
            if victims and victims < rows:
                service.plan_staging_width(victims)

    def attach_hot_cache_manager(self, manager, streamer) -> None:
        """Attach the RAM-miss service (fail-stop check, residency pushes, row backend).

        A no-op unless the layer's pinned tier runs on the native slot table.
        """
        from sglang.srt.layers.moe.expert_engine.row_backend import (
            NativePinnedSlotTable,
        )
        from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService

        tier = getattr(streamer, "pinned_host_cache", None)
        if not isinstance(getattr(tier, "_lru", None), NativePinnedSlotTable):
            return
        ExpertEngineService.get().attach(manager, streamer)

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional[ExpertRowSource]:
        """Build the row source for ``kind`` (``auto`` or ``shards``).

        Reads the original EXL3 shards, or, when ``SGLANG_EXPERT_STREAM_MIRROR``'s ``dirs``
        is set, the mirrored copies of them. Raises for any other kind.
        """
        if kind in ("auto", "shards"):
            mirror = exl3_mirror_config()
            if mirror is not None:
                return self._mirror_row_source(*mirror)
            # Imported here: the source pulls in sglang.srt.model_loader, whose
            # package import reaches the quantization methods that import this
            # module.
            from sglang.srt.layers.moe.exl3_shard_row_source import (
                Exl3ShardRowSource,
            )

            return Exl3ShardRowSource.for_layer(
                self.layout,
                self.layer_id,
                self._segments,
                direct=self._resolve_direct(),
            )
        raise ValueError(
            f"expert format {self.key!r} has no row source kind {kind!r}; "
            "choose from ('auto', 'shards')"
        )

    def _mirror_row_source(
        self, roots: tuple[str, ...], weights: tuple[float, ...]
    ) -> ExpertRowSource:
        """The mirrored shard source, splitting reads across ``roots`` by ``weights``."""
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
        """Check what the eager and the native reader both need of the roots.

        These are the conditions beyond parsing: a ``source_root`` to locate each
        shard's copy by, and no root that is the checkpoint directory itself.
        """
        if self.source_root is None:
            raise ValueError(
                f"{_MIRROR_DIRS} needs the format built with "
                "source_root, the checkpoint directory the layout was read from "
                "(SGLANG_EXPERT_STREAM_EXPERT_DIR), to find each root's copy of a shard"
            )
        for root in roots:
            if os.path.realpath(root) == os.path.realpath(self.source_root):
                raise ValueError(
                    f"{_MIRROR_DIRS}: {root!r} is the checkpoint directory "
                    f"{self.source_root!r} itself, not a mirror of it"
                )

    def mirror_table_args(self) -> dict:
        """The mirror keyword arguments of ``exl3_ram_miss_tables``.

        Empty without ``SGLANG_EXPERT_STREAM_MIRROR``'s ``dirs``; otherwise the same
        validated roots and split policy ``default_row_source`` builds its mirror
        source from.
        """
        mirror = exl3_mirror_config()
        if mirror is None:
            return {}
        roots, weights = mirror
        self._check_mirror_roots(roots)
        return dict(
            roots=roots, policy=StaticSplitPolicy(weights), source_root=self.source_root
        )

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional[ExpertRowSource]
    ) -> Optional[int]:
        """The file bytes one expert row reads, or None without a row source.

        A non-None value turns on the eager pinned host tier and the file counters.
        """
        return None if row_source is None else row_source.file_bytes_per_expert

    def engine_tables(self, streamers: dict):
        """The flat tables the native reader runs from, over every streamed layer's pinned slabs and row images.

        ``streamers`` maps a layer id to its streamer. The mirror roots and direct-read setting are the ones the
        eager row source uses, validated by the same code.
        """
        from sglang.srt.layers.moe.expert_engine.formats.exl3_tables import (
            exl3_ram_miss_tables,
            open_service_row_images,
        )

        mirrors = self.mirror_table_args()
        direct = self._resolve_direct()
        return exl3_ram_miss_tables(
            self.layout,
            self.segment_map(),
            {
                layer_id: s.pinned_host_cache.tensors
                for layer_id, s in streamers.items()
            },
            **mirrors,
            row_images=open_service_row_images(self.layout, mirrors, direct, streamers),
        )

    @staticmethod
    def layer_dims(layer: torch.nn.Module) -> tuple[int, int]:
        return layer.exl3_num_experts, layer.exl3_hidden

    @staticmethod
    def build_streamer(layer: torch.nn.Module):
        return build_exl3_expert_streamer(layer)

    @staticmethod
    def cpu_trait(*, source: str, ext=None):
        """The EXL3 CPU kernel's quant trait for ``source``'s layers ("draft" or "target").

        Row weighting (SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT) is the one per-source choice.
        """
        from sglang.srt.layers.moe.cpu_experts.assignment import row_weighted_sources
        from sglang.srt.layers.quantization.exl3.ext import exl3_ext
        from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

        if ext is None:
            ext = exl3_ext()
        row_weighted = source in row_weighted_sources()
        path = getattr(ext, "__file__", None)
        logger.info(
            "CPU experts: the %s's EXL3 layers use %s tile assignment (extension %s, sha256 %s)",
            source,
            "row-weighted" if row_weighted else "even",
            path,
            _short_sha256(path),
        )
        return Exl3CpuQuantTrait(ext, act_limit=None, row_weighted=row_weighted)

    def prepare_host_rows(
        self, layer: torch.nn.Module, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        """EXL3 rows are staged as the checkpoint stores them."""
        return tensor

    def moe_apply(self, layer, x, slot_ids, weights, slot_tensors) -> torch.Tensor:
        """The fused EXL3 MoE over the hot-cache slots the gather filled (fp32 output).

        ``slot_ids`` is the router's int32 remap of this batch's expert ids to slot ids. Layer fusion takes it as it
        is and writes the int64 copy inside its one kernel; the unfused kernel reads int64.
        """
        from sglang.srt.layers.quantization.exl3.fused_moe import exl3_fused_moe_for

        fused = exl3_fused_moe_for(layer, slot_tensors.streamer)
        flat_ids = slot_ids.reshape(-1)
        return fused.run(
            x,
            weights.reshape(-1),
            flat_ids if fused.layer_fusion else flat_ids.long(),
            slot_tensors.keep,
            slot_tensors.swiglu_limit,
            cpu=slot_tensors.cpu,
        )

    def _resolve_direct(self) -> bool:
        """Whether shard reads use O_DIRECT, from ``direct`` or the file-reader mode."""
        if self.direct is not None:
            return self.direct
        from sglang.srt.model_loader.file_row_reader import validate_file_reader_mode

        mode = validate_file_reader_mode(stream_config().tiers.file_reader)
        if mode == "mmap":
            raise ValueError(
                "the exl3 shard row source reads with io_uring; set "
                "SGLANG_EXPERT_STREAM_FILE_READER=uring_direct (or uring for buffered reads)"
            )
        return mode == "uring_direct"


# The env members these parsers name in their errors (SGLANG_EXPERT_STREAM_MIRROR is one JSON object).
_MIRROR_DIRS = "SGLANG_EXPERT_STREAM_MIRROR dirs"
_MIRROR_WEIGHTS = "SGLANG_EXPERT_STREAM_MIRROR weights"


def parse_mirror_roots(value: str) -> tuple[str, ...]:
    """Parse the ``os.pathsep``-separated ``SGLANG_EXPERT_STREAM_MIRROR`` ``dirs`` roots.

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


def parse_mirror_weights(value: str, num_roots: int, name: str = _MIRROR_WEIGHTS) -> tuple[float, ...]:
    """Parse ``SGLANG_EXPERT_STREAM_MIRROR``'s ``weights`` (or another list in its format, ``name``): one weight per
    root.

    Colon-separated, non-negative and not all zero; empty means equal weights.
    """
    if not value.strip():
        return (1.0,) * num_roots
    weights = []
    for token in value.split(":"):
        try:
            weight = float(token)
        except ValueError:
            raise ValueError(
                f"{name}={value!r}: {token!r} is not a number; give "
                "colon-separated non-negative numbers such as 3:1"
            ) from None
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{name}={value!r}: {token!r} is not a finite "
                "non-negative number"
            )
        weights.append(weight)
    if len(weights) != num_roots:
        raise ValueError(
            f"{name} lists {len(weights)} weights but {_MIRROR_DIRS} "
            f"lists {num_roots} roots"
        )
    if not any(weights):
        raise ValueError(f"{name}={value!r}: every weight is zero")
    return tuple(weights)


def two_stage_span_weights(
    config: Optional[ExpertStreamConfig] = None,
) -> tuple[Optional[tuple[float, ...]], Optional[tuple[float, ...]]]:
    """``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``'s ``w13_weights`` and ``w2_weights``, each one weight per mirror root
    (one root without a mirror), or None when unset (the span keeps the mirror's shares). ``config`` defaults to
    ``stream_config()``.

    Raises for a list the mirror cannot take: another root count, all zero, or a share for a root the mirror weighs 0
    (that root reads none of a row, so the tables give it nothing to read from).
    """
    config = stream_config() if config is None else config
    mirror = exl3_mirror_config(config)
    roots, mirror_weights = (len(mirror[0]), mirror[1]) if mirror is not None else (1, (1.0,))
    two_stage = config.cpu_experts.two_stage
    out = []
    for name, value in (
        ("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE w13_weights", two_stage.w13_weights),
        ("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE w2_weights", two_stage.w2_weights),
    ):
        if not value.strip():
            out.append(None)
            continue
        weights = parse_mirror_weights(value, roots, name)
        for root, (weight, base) in enumerate(zip(weights, mirror_weights)):
            if weight > 0 and base == 0:
                raise ValueError(f"{name} gives root {root} a share, but {_MIRROR_WEIGHTS} gives it 0")
        out.append(weights)
    return out[0], out[1]


def exl3_mirror_config(
    config: Optional[ExpertStreamConfig] = None,
) -> Optional[tuple[tuple[str, ...], tuple[float, ...]]]:
    """``(roots, weights)`` from ``config.tiers.mirror`` (``stream_config()`` when None), or None when mirroring is off.

    Raises if weights are set without roots, since they would be silently ignored.
    """
    mirror = (stream_config() if config is None else config).tiers.mirror
    dirs = mirror.dirs
    weights = mirror.weights
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


def _short_sha256(path: Optional[str]) -> str:
    """First 12 hex digits of the file's sha256, so a log line names the exact shared object that served an arm."""
    if not path:
        return "?"
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return "?"


@functools.lru_cache(maxsize=4)
def exl3_expert_layout_for(expert_dir: str) -> Exl3ExpertLayout:
    """The checkpoint's expert layout, read once per directory (headers only)."""
    return build_exl3_expert_layout(expert_dir)


def build_exl3_expert_streamer(
    layer: torch.nn.Module, expert_dir: Optional[str] = None
):
    """An ``ExpertStreamer`` for an EXL3 MoE layer whose routed experts stay on disk.

    ``expert_dir`` defaults to ``SGLANG_EXPERT_STREAM_EXPERT_DIR``. The row source comes
    from ``SGLANG_EXPERT_STREAM_ROW_SOURCE`` (``auto`` means the shards). Raises if the
    checkpoint's expert count or trellis shapes do not match the layer.
    """
    # Imported here: expert_stream imports Triton kernels and the model loader.
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

    global _WARNED_WITHOUT_PINNED_TIER
    tiers = stream_config().tiers
    expert_dir = expert_dir or tiers.expert_dir
    if not expert_dir:
        raise ValueError("SGLANG_EXPERT_STREAM_ENABLE needs SGLANG_EXPERT_STREAM_EXPERT_DIR")
    if not tiers.pinned_host_mb and not _WARNED_WITHOUT_PINNED_TIER:
        _WARNED_WITHOUT_PINNED_TIER = True
        logger.warning(
            "EXL3 expert streaming without SGLANG_EXPERT_STREAM_PINNED_HOST_MB: there is no "
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
    if specs["w13_trellis"][:3] != (2, hidden, inter) or specs["w2_trellis"][:3] != (
        1,
        inter,
        hidden,
    ):
        raise ValueError(
            f"exl3 streaming: expert trellis rows {specs['w13_trellis']} / "
            f"{specs['w2_trellis']} do not match the layer "
            f"({layer.exl3_hidden} -> {layer.exl3_inter})"
        )
    return ExpertStreamer(layer, fmt.names, layer_id=layer.layer_id, format=fmt)
