"""RAM-miss service for EXL3 streamed experts.

A native service thread fills the pinned RAM tier's slots: it reads the missing experts'
row images with O_DIRECT straight into the layers' pinned slabs, with no Python on the
path. Device kernels in the decode graph exchange records with that thread over the
slot-map protocol (``analysis/dsv41-drive/LEASE_PROTOCOL.md``).

The pieces, in file order:

- ``Exl3RamMissTables`` and ``exl3_ram_miss_tables``: flatten a checkpoint's row images
  and each layer's pinned slabs into the int64 tensors the native thread reads.
- Copy-engine helpers: the small-copy table, and the checks that refuse an unsafe
  launch configuration.
- ``NativePinnedSlotTable``: a layer's ``PinnedSlotTable`` over the native bookkeeping.
- ``Exl3RamMissRowBackend``: a layer's gather, the lease chain of device kernels.
- ``Exl3RamMissService``: the process-wide singleton that starts the thread, owns the
  device side and the hooks, runs per-batch upkeep and shuts everything down safely.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
from sglang.kernels.ops.moe.expert_lease_block import WireLayout, wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    ExpertStreamDevice,
    ExpertStreamHost,
    host_layout,
    new_hot_page,
    new_page,
    stream_segment_map,
)
from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.threading_config import ThreadingConfig
from sglang.srt.layers.moe.exl3_expert_format import (
    EXL3_MAX_GATHER_ROWS,
    EXL3_STREAMED_NAMES,
    RowSegment,
)
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_read_split import SplitPolicy, StaticSplitPolicy
from sglang.srt.layers.moe.exl3_row_image import RowImageSet
from sglang.srt.layers.moe.exl3_stream_trace import GraphRouteLog
from sglang.srt.layers.moe.expert_host_tier import quarantine_host_slabs
from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
from sglang.srt.layers.moe.host_numa import group_ranges, slot_nodes

logger = logging.getLogger(__name__)
# NVTX ranges around the eager host-use pause's stream sync and thread pause.
_SYNC_WAIT_NVTX = os.environ.get("SGLANG_DSV41_SYNC_WAIT_NVTX") == "1"


def _quarantine_service_at_exit(service: weakref.ref[Exl3RamMissService]) -> None:
    """Exit hook: quarantine the tiers, never free them.

    No device barrier is attempted in an exit handler, so nothing can prove a GPU
    reader has finished.
    """
    live = service()
    if live is not None:
        live.shutdown(at_exit=True)


@dataclass(frozen=True)
class Exl3RamMissTables:
    """The flat tables the native RAM-miss reader runs from.

    Dimensions: ``L`` streamed layers (ascending layer id is row order), ``E`` experts
    per layer, ``F`` files, ``P`` parts (mirror roots) per row, ``S`` segments.

    Files and extents:

    - ``paths``, ``source_paths``, ``file_sizes`` (int64 ``[F]``): per file, its path,
      the source shard it copies (itself with no mirror roots) and its size.
    - ``extents`` (int64 ``[L, E, P, 4]``): file index, aligned offset, aligned length
      and destination offset in the row's slot. Part ``p`` of a row is served by root
      ``p``. Parts sum to the row's aligned length, and a zero-length part means this
      root serves none of the row (no read is issued).
    - ``starts`` (int64 ``[L, E]``): where the row starts inside its aligned superset.
    - ``parts``: ``P``, mirror roots per row (1 with no roots).

    Destination:

    - ``segments`` (int64 ``[S, 4]``): name index, destination offset, source offset,
      bytes.
    - ``slabs`` (int64 ``[L, 6]``): slab base addresses in ``EXL3_STREAMED_NAMES``
      order. ``row_bytes`` (int64 ``[6]``) is each slab's row size and ``capacity``
      (int64 ``[L]``) each layer's row count.
    - ``slot_bytes``: bytes of one row slot, the image size.

    ``row_images`` marks the only supported source: the files are row-image layer
    files (see ``exl3_row_image``), a row's parts tile its image (``starts`` is 0 and
    the extents' destination is the position in the image), the segments are the six
    identity segments, and the reader reads straight into the slabs (its direct mode).

    ``keepalive`` holds every slab tensor whose address is in ``slabs``: the native
    reader writes through those raw addresses, so the tables must keep the slabs alive.
    """

    layer_ids: list[int]
    paths: list[str]
    source_paths: list[str]
    file_sizes: torch.Tensor
    extents: torch.Tensor
    starts: torch.Tensor
    parts: int
    segments: torch.Tensor
    slabs: torch.Tensor
    row_bytes: torch.Tensor
    capacity: torch.Tensor
    slot_bytes: int
    row_images: bool = False
    keepalive: tuple = field(default=(), repr=False, compare=False)


def exl3_ram_miss_tables(
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    roots: Sequence[str] = (),
    policy: Optional[SplitPolicy] = None,
    source_root: Optional[str] = None,
    row_images: Optional[RowImageSet] = None,
) -> Exl3RamMissTables:
    """Tables for the streamed layers in ``slabs_by_layer`` (ascending id is row order).

    ``row_images`` is an ``open_row_images`` set over ``roots``, split across them as
    ``policy`` splits the mirrors; see ``_row_image_tables``. ``source_root`` is
    accepted only so ``mirror_table_args()`` can be passed whole.

    Row images are the only supported source: the native reader reads them with
    O_DIRECT straight into the slabs, and shard tables would need a bounce-and-pack
    path that no longer exists. A call without ``row_images`` raises ``ValueError``
    naming the converter.
    """
    # The images carry their own paths, so the mirrors' source root names nothing here.
    del source_root
    if row_images is None:
        raise ValueError(
            "the RAM-miss reader reads row images only: build them for this checkpoint with "
            "scripts/dsv41/build_row_images.py and pass the opened set (row_images=...); shard tables were refused"
        )
    return _row_image_tables(
        layout, segments, slabs_by_layer, row_images, roots=roots, policy=policy
    )


def _slab_table(
    layer_ids, slabs_by_layer, row_bytes, names
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every streamed layer's slab base addresses (int64 ``[L, 6]``) and row counts.

    Raises ``ValueError`` for a slab the native reader could not write through its raw
    address: non-contiguous, not on the CPU, or with a row size other than
    ``row_bytes``.
    """
    slabs = torch.empty((len(layer_ids), len(EXL3_STREAMED_NAMES)), dtype=torch.int64)
    capacity = torch.empty(len(layer_ids), dtype=torch.int64)
    for row, layer_id in enumerate(layer_ids):
        tensors = slabs_by_layer[layer_id]
        rows = {int(tensors[name].shape[0]) for name in EXL3_STREAMED_NAMES}
        if len(rows) != 1:
            raise ValueError(
                f"layer {layer_id}: pinned slabs disagree on their row count {rows}"
            )
        capacity[row] = rows.pop()
        for name, index in names.items():
            slab = tensors[name]
            if not slab.is_contiguous() or slab.device.type != "cpu":
                raise ValueError(
                    f"layer {layer_id} {name}: slab must be a contiguous CPU tensor"
                )
            per_row = slab.numel() * slab.element_size() // max(int(capacity[row]), 1)
            if per_row != int(row_bytes[index]):
                raise ValueError(
                    f"layer {layer_id} {name}: slab rows hold {per_row} B, expected {int(row_bytes[index])}"
                )
            slabs[row, index] = slab.data_ptr()
    return slabs, capacity


def _row_image_tables(
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    images: RowImageSet,
    *,
    roots: Sequence[str],
    policy: Optional[SplitPolicy],
) -> Exl3RamMissTables:
    """The native reader's tables over row images.

    File ``row * parts + p`` is root ``p``'s image file of streamed row ``row``. Expert
    ``e``'s image is split across the roots as ``policy`` splits its page-rounded
    ``row_stride`` (the mirrors' split), each part clipped at ``image_bytes``. The last
    part ends exactly at the image's end because the padding after it is never read: a
    ``readv`` of it would run past the last slab row.

    Part ``p`` is ``(file, e * row_stride + start, bytes, start)``, with ``start`` its
    position in the image. ``starts`` is 0 and the segments are the identity (name
    ``n``'s slab row is image bytes ``[name_offsets[n], + row_bytes[n])``), so piece
    ``j`` of the reader's geometry is exactly sub-read ``j``. Every length and offset
    is a multiple of 512, as the image layout's names are; the reader re-checks that at
    open against O_DIRECT's alignment.
    """
    parts = len(images.roots)
    if tuple(roots) and tuple(roots) != tuple(images.roots):
        raise ValueError(
            f"row images are on {images.roots}, the tables were asked for roots {tuple(roots)}"
        )
    if policy is None:
        if parts != 1:
            raise ValueError(
                "row images on several roots need the split policy the mirrors use"
            )
        policy = StaticSplitPolicy((1.0,))
    image = images.layout
    if images.num_experts != layout.num_experts:
        raise ValueError(
            f"row images hold {images.num_experts} experts per layer, the layout {layout.num_experts}"
        )
    if image.segments != tuple(segments):
        raise ValueError(
            "row images were opened for a different segment map than the tables are built from"
        )
    split = policy.plan(image.row_stride)
    if len(split.part_bytes) != parts:
        raise ValueError(
            f"split policy plans {len(split.part_bytes)} parts for {parts} row-image roots"
        )
    clipped = [
        (min(start, image.image_bytes), min(start + nbytes, image.image_bytes))
        for start, nbytes in zip(split.starts, split.part_bytes)
    ]
    layer_ids = sorted(slabs_by_layer)
    missing = [layer for layer in layer_ids if layer not in images.paths]
    if missing:
        raise ValueError(f"row images were not opened for streamed layers {missing}")
    extents = torch.empty(
        (len(layer_ids), layout.num_experts, parts, 4), dtype=torch.int64
    )
    experts = torch.arange(layout.num_experts, dtype=torch.int64) * image.row_stride
    for row in range(len(layer_ids)):
        for part, (lo, hi) in enumerate(clipped):
            extents[row, :, part, 0] = row * parts + part
            extents[row, :, part, 1] = experts + lo
            extents[row, :, part, 2] = hi - lo
            extents[row, :, part, 3] = lo
    paths = [images.paths[layer][part] for layer in layer_ids for part in range(parts)]
    # Every root's file of a layer copies root 0's; the reader's open-time size check
    # names that source.
    copied = [images.paths[layer][0] for layer in layer_ids for _ in range(parts)]
    file_bytes = layout.num_experts * image.row_stride
    names = {name: index for index, name in enumerate(EXL3_STREAMED_NAMES)}
    segment_table = torch.tensor(
        [
            [n, 0, image.name_offsets[n], image.row_bytes[n]]
            for n in range(len(EXL3_STREAMED_NAMES))
        ],
        dtype=torch.int64,
    )
    slabs, capacity = _slab_table(layer_ids, slabs_by_layer, image.row_bytes, names)
    return Exl3RamMissTables(
        layer_ids=layer_ids,
        paths=paths,
        source_paths=copied,
        file_sizes=torch.full((len(paths),), file_bytes, dtype=torch.int64),
        extents=extents,
        starts=torch.zeros((len(layer_ids), layout.num_experts), dtype=torch.int64),
        parts=parts,
        segments=segment_table,
        slabs=slabs,
        row_bytes=torch.tensor(image.row_bytes, dtype=torch.int64),
        capacity=capacity,
        slot_bytes=image.image_bytes,
        row_images=True,
        keepalive=tuple(
            slabs_by_layer[layer_id][name]
            for layer_id in layer_ids
            for name in EXL3_STREAMED_NAMES
        ),
    )


def sm_copy_mask(names: Sequence[str], lanes: int = 8) -> int:
    """Bitmask of the copy-table entries the copy wait copies itself.

    With ``SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES`` the copy wait moves the host
    layout's small tensors, wherever they sit in ``names``.
    """
    layout_names, small = host_layout(lanes=lanes)
    small_names = {name for i, name in enumerate(layout_names) if small >> i & 1}
    return sum(1 << i for i, name in enumerate(names) if name in small_names)


def sm_copy_table(segments, sm_mask: int) -> torch.Tensor:
    """The rows of ``segments.table`` (the device copy table) that ``sm_mask`` names.

    Raises ``ValueError`` unless every source, destination and row size is 16-byte
    aligned.
    """
    rows = [i for i in range(segments.table.shape[0]) if sm_mask >> i & 1]
    table = segments.table[rows].contiguous()
    for source, destination, row_bytes in table.tolist():
        # The copy wait falls back to 1-byte loads off 16-byte alignment: refuse the
        # slow path rather than take it silently.
        if (source | destination | row_bytes) % 16:
            raise ValueError(
                f"CW reads 16-byte units: source {source:#x}, destination {destination:#x} and rows of "
                f"{row_bytes} B are not all 16-byte aligned"
            )
    return table


def check_sm_small_copies(cfg: Dsv41Config) -> None:
    """Refuse ``SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES`` without the copy engine.

    It moves part of the copy engine's work into the copy wait, which only the
    copy-engine chain runs.
    """
    if cfg.enable_ram_miss_sm_small_copies and not cfg.enable_ram_miss_copy_engine:
        raise RuntimeError(
            "exl3 RAM miss: SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES needs SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"
        )


def open_service_row_images(
    layout, segments, mirrors: dict, direct: bool, streamers
) -> RowImageSet:
    """Open the row images the service reads with; they are its only row source.

    The images live beside the mirrored shards, one set per mirror root, and are split
    across the roots exactly as the mirrors are (``mirrors`` is
    ``mirror_table_args()``). Raises ``RuntimeError`` without mirror dirs (there is
    nowhere to read from) and without O_DIRECT reads (the point is a ``readv`` straight
    into the pinned slabs). ``open_row_images`` raises unless every root holds a
    complete set matching this checkpoint for every streamed layer, and names the
    converter (``scripts/dsv41/build_row_images.py``).

    A direct read overwrites its victim slot from submission, so only the lease check
    in victim selection keeps a slot that a GPU copy may still be reading from being
    chosen.
    """
    if not mirrors:
        raise RuntimeError(
            "exl3 RAM miss: the RAM-miss service reads row images and needs SGLANG_MOE_EXPERT_MIRROR_DIRS (roots "
            "holding images built by scripts/dsv41/build_row_images.py)"
        )
    if not direct:
        raise RuntimeError(
            "exl3 RAM miss: the RAM-miss service reads row images and needs SGLANG_MOE_EXPERT_FILE_READER=uring_direct"
        )
    from sglang.srt.layers.moe.exl3_row_image import open_row_images

    return open_row_images(
        mirrors["roots"], layout, segments, mirrors["source_root"], list(streamers)
    )


class NativePinnedSlotTable:
    """``PinnedSlotTable`` over the C++ service's bookkeeping for one streamed layer.

    Created at pinned-tier construction (from ``pinned_tier_options``) and registered
    with the service; the service itself starts on first use, once every layer's slabs
    exist. ``capacity`` is bound later by ``ExpertPinnedHostCache.__init__`` to the
    tier's row count.

    Host use nests: ``_depth`` counts this table's nesting, and the service counts the
    process-wide pause apart. Only the outermost level of a table pushes its hot set
    and refreshes the device slot map.
    """

    def __init__(
        self,
        service: Exl3RamMissService,
        layer_id: int,
        streamer_of: Callable[[], object],
    ):
        self.service = service
        self.layer_id = layer_id
        self.capacity: Optional[int] = None
        self.streamer_of = streamer_of
        self._seen_version = -1
        self._slots: dict[int, int] = {}
        self._slots_version = -1
        self._depth = 0
        service.register(layer_id, self)

    def bind_capacity(self, capacity: int) -> None:
        self.capacity = int(capacity)

    @property
    def reserved_rows(self) -> int:
        """The row's staging slots: the service's lanes own them, so ``assign`` never hands one out."""
        return self.service.staging_for(self.capacity)

    @property
    def _row(self) -> int:
        self.service.ensure_started()
        return self.service.row_of(self.layer_id)

    @property
    def slot_to_expert(self) -> list[int]:
        """The READY slots' experts, -1 elsewhere: the inverse of the published map."""
        slots = [-1] * self.capacity
        for expert, slot in enumerate(self.service.host.mapping(self._row)):
            if slot >= 0:
                slots[slot] = expert
        return slots

    @property
    def expert_to_slot(self) -> dict[int, int]:
        """Resident experts and their slots, rebuilt when the C++ map version moves.

        Promotions read this once per promoted row. Every change of membership bumps
        the version. Callers must not mutate the returned dict.
        """
        row = self._row
        version = self.service.host.version()
        if self._slots_version != version:
            self._slots = {
                expert: slot
                for expert, slot in enumerate(self.service.host.mapping(row))
                if slot >= 0
            }
            self._slots_version = version
        return self._slots

    def __contains__(self, expert_id: int) -> bool:
        return self.service.host.contains(self._row, int(expert_id))

    def touch(self, expert_id: int) -> None:
        self.service.host.touch(self._row, int(expert_id))

    def assign(
        self, expert_id: int, protected=frozenset()
    ) -> tuple[int, Optional[int]]:
        return self.service.host.assign(
            self._row, int(expert_id), [int(e) for e in protected]
        )

    def release(self, slot: int) -> None:
        self.service.host.release(self._row, int(slot))

    def mapping(self, num_experts: int) -> list[int]:
        return self.service.host.mapping(self._row)

    # PinnedRowFills (SGLANG_DSV41_ENABLE_PREFILL_FILLS): the service's reader fills
    # this layer's slots while the tier's host use holds the thread paused.
    def fill_begin(self, experts, protected, fallback: bool) -> tuple[list[int], int]:
        return self.service.host.fill_begin(self._row, experts, protected, fallback)

    def fill_wait(self, rows: int) -> None:
        self.service.host.fill_wait(rows, self.service.fill_timeout_s)

    def fill_landed(self) -> int:
        return self.service.host.fill_landed()

    def fill_end(self) -> bool:
        return self.service.host.fill_end()

    def before_host_use(self, cache) -> None:
        """Pause the service thread (the service counts nesting).

        At this table's outermost level only, also push the layer's hot set to C++ and
        refresh the device slot map if the thread changed the C++ map.
        """
        self.service.before_host_use()
        try:
            if self._depth == 0:
                self._push_hot()
                version = self.service.host.version()
                if version != self._seen_version:
                    cache._refresh_mapping()
                    self._seen_version = version
        except BaseException:
            self.service.after_host_use()
            raise
        self._depth += 1

    def after_host_use(self, cache) -> None:
        self._depth -= 1
        self.service.after_host_use()

    def _push_hot(self) -> None:
        """Make the C++ victim choice protect what ``is_pinned`` protects right now.

        Until the updater is attached (start-up promotions) that is the hot cache's
        slots, which it reserves before promoting them in chunks, each in its own host
        use. Without this push, a later chunk's admission could evict an earlier
        chunk's expert from RAM, leaving a VRAM-hot expert with no pinned copy. After
        the updater is attached it is the snapshot taken at the pause.
        """
        streamer = self.streamer_of()
        hot = None if streamer is None else getattr(streamer, "hot_cache", None)
        if hot is not None:
            self.service.host.set_hot(
                self._row,
                self.service.hot_experts(self.layer_id)
                if self.service.gpu_hot_enabled
                else hot.slot_to_expert,
            )


def padded_plan_width(capacity: int, lanes: int) -> int:
    """The length of a row backend's ``planned`` tensor: its plan width, and at least the build's lanes."""
    return max(capacity, lanes)


class Exl3RamMissRowBackend(PinnedTierRowBackend):
    """``PinnedTierRowBackend`` whose gather is the lease chain.

    One gather posts a linear chain of kernels in one stream (see
    ``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "The chain"): the post kernel, which
    types the lanes from the device's slot map; C1, which copies the compacted
    ``HIT_SM`` lanes; S, which copies the misses piece by piece; the copy wait (CW); a
    stream wait; and the copy check (CC).

    ``routes`` holds the layer's routed experts (the protect set); ``_apply_graph``
    copies them in before each gather. ``planned`` pads the plan's expert ids to at
    least the wire's lanes (-1 past the plan): the post kernel reads
    ``min(count, lanes)`` lanes and ``count`` lives on the device, so no host check
    can bound it. Every lane of a gather is delivered or the process stops, so ``keep``
    stays 1 and the delivered count is the plan's.

    With CPU experts, ``_apply_graph`` sets ``cpu_input`` (``x`` and the routes'
    weights) around each gather and the post stages it for the CPU expert thread.
    """

    name = "exl3_ram_miss"

    def __init__(
        self,
        segments,
        host_row_map: torch.Tensor,
        device_side: ExpertStreamDevice,
        row: int,
        capacity: int,
        stream_maps: Mapping[int, torch.Tensor],
        hot_slots: Optional[torch.Tensor] = None,
        hot_capacity: int = 0,
        route_log=None,
        copy_engine: bool = False,
        copy_sm_table: Optional[torch.Tensor] = None,
        cpu_experts: bool = False,
        streamer_of: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__(segments, host_row_map, capacity)
        self.device_side = device_side
        self.row = row
        self.routes = torch.full(
            (capacity,), -1, dtype=torch.int64, device=host_row_map.device
        )
        self.planned = torch.full(
            (padded_plan_width(capacity, device_side.wire.lanes),),
            -1,
            dtype=torch.int64,
            device=host_row_map.device,
        )
        self.hot_slots = hot_slots
        self.hot_capacity = hot_capacity
        # Per tag, the miss stream's map of that tag's copy table (stream_segment_map).
        if set(stream_maps) != set(self.segments):
            raise ValueError(
                "the stream kernel needs a stream segment map for every copy table"
            )
        self.stream_maps = dict(stream_maps)
        # A GraphRouteLog only when the stage trace is on; None is the untraced chain.
        self.route_log = route_log
        # The service may copy a captured post's hits with its copy engine, which the
        # copy wait waits for.
        self.copy_engine = copy_engine
        # SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES: rows the copy wait copies.
        self.copy_sm_table = copy_sm_table
        # SGLANG_DSV41_CPU_EXPERTS: the service may compute resident lanes on the CPU.
        if cpu_experts and not copy_engine:
            raise ValueError("CPU experts are completed by the copy engine's copy wait")
        if cpu_experts and streamer_of is None:
            raise ValueError(
                "CPU experts check the layer's miss order at capture: pass streamer_of"
            )
        self.cpu_experts = cpu_experts
        self.cpu_input = None
        self.streamer_of = streamer_of
        self._delivered: Optional[torch.Tensor] = None

    @property
    def delivered_count(self) -> torch.Tensor:
        if self._delivered is None:
            raise RuntimeError(
                "EXL3 DIRECT reads the delivered count of a gather this backend posted"
            )
        return self._delivered

    def _stage_planned(self, plan) -> None:
        """Copy the plan's lane experts into the captured ``planned`` buffer.

        The plan is bounded first; the post kernel reads each lane's expert out of
        ``planned``, which is otherwise still at its -1 fill.
        """
        lanes = plan.expert_ids.numel()
        if lanes > self.planned.numel():
            raise ValueError(
                f"a plan of {lanes} lanes does not fit the backend's {self.planned.numel()}"
            )
        # A device copy: captured, so every replay refreshes it.
        self.planned[:lanes].copy_(plan.expert_ids)

    def post(self, tag, plan) -> None:
        """Enqueue this gather's lease chain on the current stream (see class doc)."""
        if self.route_log is not None:
            # ``routes`` (by _apply_graph) and ``plan.count`` (by the planner) were
            # both refreshed earlier in this gather on this stream.
            self.route_log.record(self.row, self.routes, plan.count)
        self._stage_planned(plan)
        # Only a captured graph lets the service copy: an eager forward may load a
        # kernel module while the copy wait holds the stream, and a load blocks the copy
        # thread's driver calls (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine").
        captured = self.copy_engine and torch.cuda.is_current_stream_capturing()
        if captured and self.cpu_experts and self.cpu_input is None:
            raise RuntimeError(
                f"CPU experts: row {self.row} gathered without its input (Exl3MoEMethod._apply_graph)"
            )
        if captured and self.cpu_experts:
            streamer = self.streamer_of()
            if streamer is None:
                raise RuntimeError(f"CPU experts: row {self.row}'s streamer is gone")
            if getattr(streamer, "_plan_miss_keys", None) is None:
                # attach installs the keys and a later enable_graph_gather resets them.
                # An unsorted plan would send the CPU the last-routed misses, not the
                # lowest-scored ones.
                raise RuntimeError(
                    f"CPU experts: row {self.row}'s route plan has no miss order (Exl3RamMissService.attach)"
                )
        side = self.device_side
        side.post(
            self.row,
            self.planned,
            plan.count,
            self.routes,
            plan.slots,
            self.hot_slots,
            self.hot_capacity,
            captured=captured,
            cpu_input=self.cpu_input if captured and self.cpu_experts else None,
        )
        side.copy_engine_captured |= captured
        copy_expert_row_segments_gpu(
            self.segments[tag], side.host_rows_1, side.dst_slots_1, side.go_1
        )
        side.stream(
            self.row,
            self.planned,
            plan.count,
            plan.slots,
            self.segments[tag],
            self.stream_maps[tag],
        )
        side.copy_wait(plan.count, plan.slots, self.copy_sm_table)
        self._delivered = plan.count


# Decode forwards, counted after the first copy-engine capture, that run unarmed. The
# kernels a decode step launches for the first time (lazily loaded under
# CUDA_MODULE_LOADING=LAZY, outside any hook) then load while no copy wait can spin
# (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine"). Counting batches instead
# armed inside the server's 8-token warm-up request, whose first decode steps then
# deadlocked (at thresholds of 4 batches and of 1). 16 covers the warm-up; the value
# beyond that is arbitrary.
COPY_ENGINE_ARM_DECODES = 16
# The prefill share: one eager gather chunk, the most rows gather_rows protects at once
# (the replay's best bound, analysis/dsv41-drive/prefill-evict/ram_replay.py).
PREFILL_SHARE_ROWS = EXL3_MAX_GATHER_ROWS
# CPU experts log their cumulative counters every this many batches, about twice per
# 128-token turn at batch 1.
CPU_STATS_LOG_BATCHES = 64

# The copy engine needs every kernel of every library loaded before it arms. Under LAZY
# (torch's default) a kernel loads at its first launch, and a first launch after arming
# can stop the copy thread's copies until the copy-wait timeout fail-stops the server. A
# seeded stress run of two min_p requests (a 525-token prompt, then a 3,314-token one)
# did so on the second's 14th decode step every time, and never under EAGER. EAGER loads
# a library's kernels when it is loaded, so what remains is a library loaded after
# arming, which the module-load guard drains for.
COPY_ENGINE_MODULE_LOADING = "EAGER"


def check_copy_engine_module_loading(environ: Mapping[str, str]) -> None:
    """Refuse the copy engine unless ``CUDA_MODULE_LOADING`` is EAGER.

    Torch sets LAZY when the variable is unset, so an unset variable is refused too.
    """
    value = environ.get("CUDA_MODULE_LOADING", "")
    if value != COPY_ENGINE_MODULE_LOADING:
        raise RuntimeError(
            "exl3 RAM miss: SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE needs CUDA_MODULE_LOADING="
            f"{COPY_ENGINE_MODULE_LOADING} in the server's environment (it is {value or 'unset'}): a kernel loaded "
            "lazily after the copy engine arms can hold its copies back until the RAM-miss deadline stops the server. "
            "EAGER costs ~1 GiB of device memory; raise --mem-fraction-static by ~0.03."
        )


def watchdog_wait_s(timeout_ms: int) -> float:
    """The C++ watchdog's abort limit for a wait timeout of ``timeout_ms``, in seconds.

    The limit is ``max(30 s, 3 x timeout)``. It must outlast the miss stream's deadline
    (a slow demand traps the device at the timeout) and the eager pause bound of
    ``2 x timeout + 1 s`` (``before_host_use``). Three times the timeout exceeds both
    for any timeout over 1 s; 30 s covers the rest.
    """
    return max(30.0, 3.0 * timeout_ms / 1000)


def parse_fault(spec: str) -> Optional[tuple[int, float]]:
    """Parse ``SGLANG_TEST_DSV41_RAM_MISS_FAULT``, ``"<demands>:<seconds>"``.

    The fault delays each demand read by ``seconds`` once ``demands`` demands have read
    rows. An empty spec means no fault (``None``).
    """
    if not spec:
        return None
    demands, seconds = spec.split(":")
    return int(demands), float(seconds)


class Exl3RamMissService:
    """The process-wide RAM-miss service: native thread, device kernels and hooks.

    A singleton (``get``). Pinned tiers register their ``NativePinnedSlotTable`` before
    the service starts; it starts lazily on first use (``ensure_started``), once every
    layer's slabs exist, and refuses a tier built afterwards. ``attach`` then runs once
    per streamed layer to install its row backend.

    Eager use of a pinned tier brackets with ``before_host_use`` / ``after_host_use``,
    which drain queued device work and pause the thread (nesting counted); the eager
    paths' map changes reach the device before the thread runs again. The scheduler's
    fail-stop hook (``fail_stop_check``) does the per-batch upkeep.

    The copy engine is enabled at start but armed only by ``fail_stop_check``, once
    ``COPY_ENGINE_ARM_DECODES`` decode forwards have run after a copy-engine graph was
    captured, so the first decode steps load their kernels unarmed. Once armed, an
    eager forward and a module load first drain the device.

    ``shutdown`` frees the tiers only when it can establish that no GPU reader and no
    service thread still runs; otherwise it quarantines every buffer until process
    exit (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Shutdown").
    """

    _instance: Optional[Exl3RamMissService] = None

    @classmethod
    def get(cls) -> Exl3RamMissService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.tables: dict[int, NativePinnedSlotTable] = {}
        self.host: Optional[ExpertStreamHost] = None
        self.device_side: Optional[ExpertStreamDevice] = None
        self.page = None
        self.slot_map = None
        self._rows: dict[int, int] = {}
        self._manager = None
        # The only caller of host.pause()/resume(), which are not reentrant.
        self._pause_depth = 0
        self._trace_rows: Optional[list[int]] = None
        self._trace_graph: Optional[list[int]] = None
        # One reusable pinned snapshot. A busy slot coalesces later cumulative register
        # values; neither the scheduler nor the CUDA graph waits for it.
        self._trace_snapshot: Optional[torch.Tensor] = None
        self._trace_event: Optional[torch.cuda.Event] = None
        self._trace_pending_rows: Optional[list[int]] = None
        # The host records a stage line per request (trace runs only).
        self._stages_traced = False
        self._stages_dropped = 0
        # Trace runs only.
        self.route_log: Optional[GraphRouteLog] = None
        # Routed rows of one batch-1 decode step over all in-graph layers (attach sums).
        self.routed_rows_per_step = 0
        self._shut_down = False
        self._quarantined = False
        self._completed = False
        # Enabled at start; armed later (see the class doc).
        self.copy_engine = False
        self.sm_small_copies = False
        # SGLANG_DSV41_CPU_EXPERTS: the CPU expert service; None when off.
        self.cpu_experts = None
        self._cpu_retune_batches = 0
        self._cpu_batches = 0
        self._cpu_log_batches = 0
        # SGLANG_DSV41_ENABLE_LEASE_PDL: the lease chain's kernels launch with PDL.
        self.lease_pdl = False
        # The widest gather any layer planned (plan_gather_width); None until one does.
        self._gather_planned: Optional[int] = None
        # The build's wire, fixed at start from the planned width.
        self._wire: Optional[WireLayout] = None
        self._copy_armed = False
        self._copy_decodes = 0
        # Module loads that drained the device first.
        self.copy_engine_module_loads = 0
        # A prefill fill's wait gives up after the watchdog's limit, so a hung fill ends
        # in the watchdog's abort first.
        self.fill_timeout_s = watchdog_wait_s(2000) + 5.0
        self.hot_page = None
        self.gpu_hot_enabled = False
        self._gpu_hot_updater = None
        self._hot_snapshot = None
        self._hot_lists: dict[int, list[int]] = {}

    def register(self, layer_id: int, table: NativePinnedSlotTable) -> None:
        """Register a layer's slot table; only valid before the service starts."""
        if self.host is not None:
            raise RuntimeError(
                "exl3 RAM miss: a pinned tier was built after the service started"
            )
        self.tables[layer_id] = table

    def plan_gather_width(self, rows: int) -> None:
        """Plan a layer whose graph gather misses ``rows`` ids; the widest layer sets the build's lane count.

        Only valid before the service starts. Raises ValueError for a width outside 1..32.
        """
        if self.host is not None:
            raise RuntimeError(
                "exl3 RAM miss: the graph gather width was planned after the service started"
            )
        planned = max(1, int(rows))
        wire_layout(planned)
        self._gather_planned = (
            planned
            if self._gather_planned is None
            else max(self._gather_planned, planned)
        )

    @property
    def wire(self) -> WireLayout:
        """The started build's wire; raises before ``ensure_started`` fixes the lane count."""
        if self._wire is None:
            raise RuntimeError(
                "exl3 RAM miss: the build's wire is read before the service started"
            )
        return self._wire

    @property
    def lanes(self) -> int:
        """The started build's lane count; raises before ``ensure_started``."""
        return self.wire.lanes

    def resolved_lanes(self) -> int:
        """The lane count the service builds for: the widest planned gather, rounded up to 8."""
        return wire_layout(self._gather_planned or 1).lanes

    def staging_width(self) -> int:
        """The staging slots every row asks for: the planned gather width, else the build's lanes."""
        return self._gather_planned or self.resolved_lanes()

    def staging_for(self, capacity: int) -> int:
        """The staging slots a row of ``capacity`` slots keeps: the planned width, and never its last slot."""
        return min(self.staging_width(), capacity - 1)

    def planned_padding(self, capacity: int) -> int:
        """The length of a row backend's ``planned`` tensor: its plan width, and at least the build's lanes."""
        return padded_plan_width(capacity, self.resolved_lanes())

    def row_of(self, layer_id: int) -> int:
        """The service row (position in the tables) of a streamed layer."""
        return self._rows[layer_id]

    def _refuse_if_shut_down(self) -> None:
        # shutdown() closes the C++ host but keeps self.host, whose calls would then
        # fail with an opaque "unknown handle".
        if self._shut_down:
            raise RuntimeError(
                "exl3 RAM miss: the option C service was shut down; its pinned tiers are closed"
            )

    def ensure_started(self) -> None:
        """Start the service on first use; a no-op once started.

        Validates the launch configuration, builds the tables over every layer's slabs
        and row images, creates the native host, enables the copy engine, CPU experts
        and trace as configured, starts the thread and installs the exit quarantine
        hook. Raises if the service was shut down or the configuration is refused.
        """
        self._refuse_if_shut_down()
        if self.host is not None:
            return
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        if (
            envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.get()
            and not get_exl3_stream_trace().enabled
        ):
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_DSV41_ROUTER_CAPTURE_PATH needs SGLANG_DSV41_EXPERT_TRACE_PATH: the "
                "router capture is written by the stage trace's graph route log"
            )
        cfg = Dsv41Config.from_envs()
        streamers = {
            layer_id: table.streamer_of()
            for layer_id, table in sorted(self.tables.items())
        }
        missing = [
            layer_id
            for layer_id, s in streamers.items()
            if s is None or s.pinned_host_cache is None
        ]
        if missing:
            raise RuntimeError(
                f"exl3 RAM miss: layers {missing} have no pinned tier yet"
            )
        fmt = next(iter(streamers.values())).format
        # The mirror roots and direct-read setting the eager row source uses, validated
        # by the same code.
        mirrors = fmt.mirror_table_args()
        direct = fmt._resolve_direct()
        tables = exl3_ram_miss_tables(
            fmt.layout,
            fmt.segment_map(),
            {
                layer_id: s.pinned_host_cache.tensors
                for layer_id, s in streamers.items()
            },
            **mirrors,
            row_images=open_service_row_images(
                fmt.layout, fmt.segment_map(), mirrors, direct, streamers
            ),
        )
        lanes = self.resolved_lanes()
        layout_names, _ = host_layout(lanes=lanes)
        if layout_names != EXL3_STREAMED_NAMES:
            raise RuntimeError(
                f"exl3 RAM miss: the host module's layout {layout_names} is not EXL3_STREAMED_NAMES {EXL3_STREAMED_NAMES}"
            )
        numa = ThreadingConfig.from_env(
            cpu_experts=envs.SGLANG_DSV41_CPU_EXPERTS.get(),
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
        for line in numa.log_lines():
            logger.info("exl3 RAM miss %s", line)
        node_ranges = None
        if numa.nodes > 1:
            # Rows in layer order, as exl3_ram_miss_tables numbers them; a slot across a seam is in no range.
            node_ranges = group_ranges(
                [
                    slot_nodes(s.pinned_host_cache.tensors, s.pinned_host_cache.capacity)
                    for _, s in sorted(streamers.items())
                ],
                [plan.node for plan in numa.plans],
            )
        pin = torch.cuda.is_available()
        self._wire = wire_layout(lanes, numa.nodes)
        page = new_page(pin=pin, wire=self._wire)
        hot_page = new_hot_page(tables.starts.shape[1], pin=pin)
        slot_map = torch.full(tuple(tables.starts.shape), -1, dtype=torch.int32)
        slot_map = slot_map.pin_memory() if pin else slot_map
        # The native build is chosen once, here: the instrumented one whenever this
        # service will enable the stage trace or inject the test fault, however they
        # were switched on (the cached trace object or the config, not only their env
        # vars). That way it never loads the production build and then calls an entry
        # point production refuses.
        fault = parse_fault(cfg.ram_miss_fault)
        instrumented = get_exl3_stream_trace().enabled or fault is not None
        host = ExpertStreamHost(
            tables,
            page=page,
            slot_map=slot_map,
            hot_page=hot_page,
            variant="instr" if instrumented else None,
            lanes=self.lanes,
            node_ranges=node_ranges,
            sq_thread_cpus=[-1 if p.sq is None else p.sq for p in numa.plans],
        )
        try:
            # Before any slot is filled: the hot cache fills the tiers first, after
            # plan_gather_width.
            host.reserve_staging(self.staging_width())
            copy_engine = cfg.enable_ram_miss_copy_engine
            check_sm_small_copies(cfg)
            if copy_engine:
                check_copy_engine_module_loading(os.environ)
                # Before the thread starts, and unarmed. The watchdog bounds each copy
                # wait by the RAM-miss timeout.
                host.enable_copy_engine(
                    torch.cuda.current_device(),
                    wait_timeout_ms=cfg.ram_miss_timeout_ms,
                    cpus=numa.copy_cpus,
                )
            cpu_experts = None
            if envs.SGLANG_DSV41_CPU_EXPERTS.get():
                cpu_experts = self._start_cpu_experts(cfg, host, fmt, streamers, pin, numa)
            if get_exl3_stream_trace().enabled:
                # Before the thread starts: without a trace file it takes no timestamps.
                host.enable_trace()
                self._stages_traced = True
            share_recorder = None
            if cfg.enable_prefill_share:
                from sglang.srt.eplb.expert_distribution import (
                    _ExpertDistributionRecorderNoop,
                    get_global_expert_distribution_recorder,
                )

                share_recorder = get_global_expert_distribution_recorder()
                if isinstance(share_recorder, _ExpertDistributionRecorderNoop):
                    # The no-op recorder drops pre-forward observers, so the share would
                    # never be set.
                    raise RuntimeError(
                        "exl3 RAM miss: SGLANG_DSV41_ENABLE_PREFILL_SHARE needs the expert distribution recorder "
                        "(--expert-distribution-recorder-mode), which calls the pre-forward observer that sets it"
                    )
            cores = [-1 if plan.ram is None else plan.ram for plan in numa.plans]
            busy_poll = all(plan.busy_poll for plan in numa.plans)
            if numa.nodes == 1 and cores[0] == -1 and not busy_poll:
                # The server's affinity: the thread spins there with PAUSE.
                host.start_thread(fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms))
            else:
                host.start_thread(
                    cpu_core=cores if numa.nodes > 1 else cores[0],
                    busy_poll=busy_poll,
                    fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms),
                )
            if fault is not None:
                demands, seconds = fault
                host.inject(delay_s=seconds, delay_after_demands=demands)
                logger.warning(
                    "exl3 RAM miss TEST FAULT: demand reads sleep %.1f s after %d demands",
                    seconds,
                    demands,
                )
        except BaseException:
            # The thread writes into the tiers' slabs through raw addresses: join it
            # before anything can release them.
            host.stop()
            raise
        self.page, self.slot_map, self.host = page, slot_map, host
        self.numa = numa
        self.hot_page = hot_page
        self.copy_engine = copy_engine
        self.cpu_experts = cpu_experts
        self._cpu_retune_batches = envs.SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES.get()
        self.sm_small_copies = cfg.enable_ram_miss_sm_small_copies
        self.lease_pdl = cfg.enable_lease_pdl
        self.fill_timeout_s = watchdog_wait_s(cfg.ram_miss_timeout_ms) + 5.0
        # Order matters. atexit runs last-registered first, and weakref.finalize
        # installs its single exit hook when the first finalizer (any tier's slab
        # unregister) is created. Registering here, after every tier exists (register()
        # refuses a later one), makes this hook run before that one, so the slabs are
        # quarantined and their finalizers detached before they could unregister.
        # Registered earlier, the quarantine would be silently undone.
        atexit.register(_quarantine_service_at_exit, weakref.ref(self))
        self._rows = {layer_id: row for row, layer_id in enumerate(tables.layer_ids)}
        if share_recorder is not None:
            share_recorder.register_pre_forward_observer(self._set_prefill_share)
        logger.info(
            "exl3 RAM miss thread started: %d layers, %d files, slot bytes %d, wait timeout %d ms, "
            "copy engine %s, CPU experts %s, build %s",
            len(tables.layer_ids),
            len(tables.paths),
            tables.slot_bytes,
            cfg.ram_miss_timeout_ms,
            "on" if copy_engine else "off",
            "on" if cpu_experts is not None else "off",
            self.host.variant,
        )

    @staticmethod
    def _start_cpu_experts(cfg, host, fmt, streamers, pin: bool, numa: ThreadingConfig):
        """Build the CPU expert service for ``SGLANG_DSV41_CPU_EXPERTS``.

        Runs after the copy engine is enabled and before the service thread starts. CPU
        lanes complete in the copy wait and leave the fused MoE through layer fusion's
        route tables, so the copy engine and layer fusion are both required; so is the
        fused plan, and the prefetch pull join must be off.
        """
        from sglang.srt.layers.moe.cpu_experts.service import (
            CpuExpertGroups,
            configured_split,
            cpu_trait_for,
        )

        if not cfg.enable_ram_miss_copy_engine:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_DSV41_CPU_EXPERTS needs SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"
            )
        if not cfg.enable_layer_fusion:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_DSV41_CPU_EXPERTS needs SGLANG_DSV41_ENABLE_LAYER_FUSION"
            )
        if envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off":
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_DSV41_CPU_EXPERTS cannot run with the prefetch pull join"
            )
        if not envs.SGLANG_MOE_EXPERT_FUSED_PLAN.get():
            # Only the fused plan sorts the miss lanes, which makes the tail lanes the
            # lowest-scored.
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_DSV41_CPU_EXPERTS needs SGLANG_MOE_EXPERT_FUSED_PLAN"
            )
        # Rows in layer order, as exl3_ram_miss_tables numbers them.
        caches = {
            row: s.pinned_host_cache.tensors
            for row, (_, s) in enumerate(sorted(streamers.items()))
        }
        trait = cpu_trait_for(fmt.key)
        hidden = {trait.hidden_size(slabs) for slabs in caches.values()}
        if len(hidden) != 1:
            raise RuntimeError(
                f"exl3 RAM miss: CPU experts need one hidden size, the layers have {hidden}"
            )
        return CpuExpertGroups(
            host,
            trait,
            caches,
            hidden=hidden.pop(),
            plans=numa.plans,
            split=configured_split(host.wire.lanes),
            pin=pin,
        )

    def before_host_use(self) -> None:
        """Begin eager pinned-tier use: drain queued device work, then pause the thread.

        Nesting is counted; only the outermost call pauses. The pause is bounded by
        ``2 x timeout + 1 s``, which the watchdog's limit exceeds (``watchdog_wait_s``).
        """
        self.ensure_started()
        if self._pause_depth == 0:
            if self.gpu_hot_enabled:
                self._hot_snapshot.copy_(
                    self._gpu_hot_updater.slot_to_expert, non_blocking=True
                )
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                with (
                    torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_stream_sync")
                    if _SYNC_WAIT_NVTX
                    else nullcontext()
                ):
                    torch.cuda.current_stream().synchronize()
            with (
                torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_thread_pause")
                if _SYNC_WAIT_NVTX
                else nullcontext()
            ):
                self.host.pause(
                    2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0
                )
            if self.gpu_hot_enabled:
                self._refresh_hot_lists()
        self._pause_depth += 1

    def after_host_use(self) -> None:
        """End eager pinned-tier use; the outermost call applies map changes."""
        self._pause_depth -= 1
        if self._pause_depth == 0:
            # The eager paths' map changes reach the device before the service runs
            # again, on the stream every later post follows
            # (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Deltas and the bulk delta").
            bulk = self.host.take_bulk_delta()
            # With no device map yet the list is dropped: a new device side starts from
            # a snapshot (attach).
            if self.device_side is not None and bulk.numel():
                self.device_side.map_bulk_apply(bulk)
            self.host.resume()

    def attach(self, manager, streamer) -> None:
        """The format's ``attach_hot_cache_manager``: hooks once, a backend per layer.

        The first call also creates the device side and seeds its slot map from the
        host's. Requires DIRECT residency on the manager. A layer without a graph pinned
        tier is left alone.
        """
        self.ensure_started()
        if self._manager is None:
            updater = getattr(manager, "gpu_residency", None)
            if updater is None or not updater.insert_direct:
                raise RuntimeError(
                    "exl3 RAM miss: the service needs DIRECT residency (SGLANG_MOE_GPU_RESIDENCY_UPDATE=1, "
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2): every record carries its hot set, and the CPU experts' "
                    "miss lanes are ordered by its ranking"
                )
            self._manager = manager
            manager.register_fail_stop_check(self.fail_stop_check)
            self._enable_gpu_hot(updater)
            if self.cpu_experts is not None:
                updater.enable_miss_order()
        if not getattr(streamer, "_graph_pinned_tier", False):
            return
        width = streamer.graph_miss_width
        if width > self.lanes:
            # The post kernel requests min(count, lanes) lanes and traps on a wider
            # plan. A verify's misses take lanes, not its routes.
            raise ValueError(
                f"exl3 RAM miss: layer {streamer.layer_id} gathers up to {streamer.graph_gather_rows} rows "
                f"({width} miss lanes) per call but the service requests at most {self.lanes} lanes"
            )
        if self.cpu_experts is not None and width < streamer.graph_gather_rows:
            raise ValueError(
                f"exl3 RAM miss: CPU experts serve one token; layer {streamer.layer_id}'s gather serves {width} "
                f"misses of {streamer.graph_gather_rows} routes (a verify)"
            )
        cache = streamer.hot_cache
        if self.cpu_experts is not None:
            # The fused plan sorts this layer's miss lanes by residency key, highest
            # first, and the CPU lanes are the tail, so the lanes worth inserting are
            # the ones copied.
            direct = manager.gpu_residency
            streamer._plan_miss_keys = direct.miss_keys[streamer.residency_row]
        if self.device_side is None:
            self.device_side = ExpertStreamDevice(
                self.page,
                # The host allocates the lease block; the device reads the same one.
                self.host.lease_block,
                device=cache.device,
                layers=len(self._rows),
                experts=self.host.experts,
                timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
                piece_runs=self.host.piece_runs(),
                row_capacities=[int(c) for c in self.host.tables.capacity],
                hot_page=self.hot_page,
                lease_pdl=self.lease_pdl,
                hit_copy=envs.SGLANG_DSV41_RAM_HIT_COPY.get(),
                cpu_misses=envs.SGLANG_DSV41_CPU_EXPERTS_MISSES.get(),
                lanes=self.lanes,
                nodes=self.wire.nodes,
            )
            if self.cpu_experts is not None:
                self.device_side.enable_cpu_experts(
                    self.cpu_experts.x_rows, self.cpu_experts.out_rows
                )
                self.cpu_experts.attach_device(self.device_side)
            # The device's map starts empty: every row the tier maps now reaches it as
            # one bulk of entries, taken paused. The bulk list before this point was
            # dropped (after_host_use), so the snapshot is the source.
            self.before_host_use()
            try:
                self.host.take_bulk_delta()
                snapshot = [
                    (row, expert, slot)
                    for row in range(len(self._rows))
                    for expert, slot in enumerate(self.host.mapping(row))
                    if slot >= 0
                ]
                if snapshot:
                    self.device_side.map_bulk_apply(
                        torch.tensor(snapshot, dtype=torch.int32)
                    )
            finally:
                self.after_host_use()
            if self._stages_traced:
                self._start_route_log(cache.device)
            if self.copy_engine:
                from sglang.srt.eplb.expert_distribution import (
                    get_global_expert_distribution_recorder,
                )

                get_global_expert_distribution_recorder().register_pre_forward_observer(
                    self._copy_engine_barrier
                )
                self._install_copy_engine_module_load_guard()
        self.routed_rows_per_step += streamer.graph_gather_rows
        row = self.row_of(streamer.layer_id)
        # The host staged staging_for(capacity) slots for the row at start. A post requests
        # at most the miss width (a verify's clamp keeps its count there), not the routes.
        want = max(1, streamer.graph_miss_width)
        staged = self.staging_for(int(self.host.tables.capacity[row]))
        if staged < want:
            logger.warning(
                "exl3 RAM miss: row %d stages %d slots, not %d: its tier has %d",
                row,
                staged,
                want,
                int(self.host.tables.capacity[row]),
            )
        if self.route_log is not None:
            self.route_log.bind(row, streamer.layer_id, cache.capacity)
        previous = streamer.row_backend
        sm_mask, copy_sm = 0, None
        if self.copy_engine:
            if list(previous.segments) != [streamer.row_tag]:
                raise ValueError(
                    f"exl3 RAM miss copy engine: layer {streamer.layer_id} has copy tables {list(previous.segments)}"
                )
            segments = previous.segments[streamer.row_tag]
            if self.sm_small_copies:
                # The pinned tier's copy table has one row per host source, in the
                # sources' order.
                names = list(streamer._graph_sources)
                if len(names) != segments.table.shape[0]:
                    raise ValueError(
                        f"exl3 RAM miss: layer {streamer.layer_id}'s copy table does not match its sources {names}"
                    )
                sm_mask = sm_copy_mask(names, self.lanes)
                copy_sm = sm_copy_table(segments, sm_mask) if sm_mask else None
        streamer.row_backend = Exl3RamMissRowBackend(
            previous.segments,
            previous.host_row_map,
            self.device_side,
            row,
            streamer.graph_gather_rows,
            {
                tag: stream_segment_map(segments, self.host.tables, row)
                for tag, segments in previous.segments.items()
            },
            hot_slots=manager.gpu_residency.slot_to_expert[
                manager.gpu_residency.layer_ids.index(streamer.layer_id)
            ],
            hot_capacity=cache.capacity,
            route_log=self.route_log,
            copy_engine=self.copy_engine,
            copy_sm_table=copy_sm,
            cpu_experts=self.cpu_experts is not None,
            streamer_of=self.tables[streamer.layer_id].streamer_of,
        )
        if self.copy_engine:
            dst_rows = min(
                int(destination.shape[0]) for _, destination in segments.pairs
            )
            self.host.set_copy_table(row, segments.table, dst_rows, sm_mask=sm_mask)
            self.device_side.set_row_copy(row, dst_rows)

    def _start_route_log(self, device) -> None:
        """Trace runs only: log every graph forward's routes in a ``GraphRouteLog``.

        Entries are stamped with the scheduler's forward identity and, when the updater
        owns them, the GPU residency bank's hot sets.
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        router_prefix = envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.get()
        # The router rings hold ~400 KB per forward, so they get a shallower ring: 32
        # deep is ~13 MB of VRAM.
        log = GraphRouteLog(
            len(self._rows), self.lanes, device, depth=32 if router_prefix else 64
        )
        if router_prefix:
            log.enable_router(router_prefix)
        if self._gpu_hot_updater is not None:
            log.bind_hot(
                self._gpu_hot_updater.slot_to_expert, self._gpu_hot_updater.layer_ids
            )
        trace = get_exl3_stream_trace()
        trace.graph_seq_source = log.read_seq
        trace.forward_meta_source = log.current_meta
        get_global_expert_distribution_recorder().register_pre_forward_observer(
            log.on_pre_forward
        )
        self.route_log = log

    def _refresh_hot_lists(self) -> None:
        """Rebuild each layer's hot-expert list from the host copy of the GPU map."""
        updater = self._gpu_hot_updater
        for row, layer_id in enumerate(updater.layer_ids):
            capacity = updater.caches[row].capacity
            self._hot_lists[layer_id] = [
                int(e) for e in self._hot_snapshot[row, :capacity].tolist() if e >= 0
            ]

    def _enable_gpu_hot(self, updater) -> None:
        """Adopt the updater's GPU residency as the hot set and push it to every row."""
        self._gpu_hot_updater = updater
        self._hot_snapshot = torch.empty_like(
            updater.slot_to_expert,
            device="cpu",
            pin_memory=updater.device.type == "cuda",
        )
        self._hot_snapshot.copy_(updater.slot_to_expert, non_blocking=True)
        if updater.device.type == "cuda":
            torch.cuda.current_stream(updater.device).synchronize()
        self._refresh_hot_lists()
        self.before_host_use()
        try:
            for layer_id, experts in self._hot_lists.items():
                self.host.set_hot(self.row_of(layer_id), experts)
        finally:
            self.after_host_use()
        self.gpu_hot_enabled = True

    def hot_experts(self, layer_id: int) -> list[int]:
        """The layer's GPU-resident experts as of the last pause."""
        return self._hot_lists.get(layer_id, [])

    def fail_stop_check(self) -> None:
        """The service's per-batch upkeep, run from the scheduler's fail-stop hook.

        Arms the copy engine, retunes and logs CPU experts, and drains the trace. A
        failed request never reaches here: the host aborts and the device traps
        (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Fail-stop").
        """
        if self.host is None or self._shut_down:
            return
        self._arm_copy_engine()
        if self.cpu_experts is not None and self._cpu_retune_batches > 0:
            self._cpu_batches += 1
            if self._cpu_batches >= self._cpu_retune_batches:
                self._cpu_batches = 0
                self.cpu_experts.retune()
        if self.cpu_experts is not None:
            self._cpu_log_batches += 1
            if self._cpu_log_batches >= CPU_STATS_LOG_BATCHES:
                self._cpu_log_batches = 0
                self.cpu_experts.log_stats()
        self._trace_step()
        self._trace_stages()

    def _set_prefill_share(self, forward_pass_id: int, forward_batch) -> None:
        """Pre-forward observer for ``SGLANG_DSV41_ENABLE_PREFILL_SHARE``.

        A prefill's pinned-tier admissions own at most ``PREFILL_SHARE_ROWS`` rows per
        layer (then they evict their own); any other forward's own none. The value
        holds until the next forward's observer runs.
        """
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        self.host.set_prefill_share(PREFILL_SHARE_ROWS if prefill else 0)

    def _copy_engine_barrier(self, forward_pass_id: int, forward_batch) -> None:
        """Pre-forward observer: drain the device before an eager forward once armed.

        An eager forward may load a kernel module, and a load blocks the copy thread's
        ``cuMemcpyAsync`` until the deadline while a decode graph in flight is held in
        its copy wait, so that step must end first. Captured decode forwards are
        counted toward arming.
        """
        if not forward_batch.forward_mode.is_decode():
            if self._copy_armed:
                torch.cuda.synchronize()
        elif self.device_side is not None and self.device_side.copy_engine_captured:
            self._copy_decodes += 1

    def _arm_copy_engine(self) -> None:
        """Arm the copy engine once enough decode forwards have run since capture."""
        if not self.copy_engine or self._copy_armed or self.device_side is None:
            return
        if self._copy_decodes >= COPY_ENGINE_ARM_DECODES:
            if self.cpu_experts is not None:
                self._calibrate_cpu_split()
            self.host.arm_copy_engine()
            self._copy_armed = True
            logger.info(
                "exl3 RAM miss copy engine armed after %d decode forwards since capture",
                self._copy_decodes,
            )

    def _calibrate_cpu_split(self) -> None:
        """Calibrate the CPU split before arming, while no CPU or copy lane is typed.

        The device drains first. Under the overlap scheduler this runs while the next
        decode still replays on the forward stream, which ``before_host_use``'s
        current-stream sync does not cover; a paused RAM thread under it would leave a
        miss lane unserved until its device deadline traps.
        """
        torch.cuda.synchronize()
        self.before_host_use()
        try:
            self.cpu_experts.calibrate(torch.cuda.current_device())
        finally:
            self.after_host_use()

    def _copy_engine_module_load_guard(self, load):
        """Wrap a module loader (Triton's ``load_binary``, tvm-ffi's ``load_module``).

        Once armed, the device drains before a module loads. ``cuModuleLoadData`` takes
        the driver lock that the copy thread's ``cuMemcpyAsync`` needs, then waits for
        the device, whose stream may be held in a copy wait for exactly that copy. That
        deadlock ends in a fail-stop at the copy-wait timeout (observed: the scheduler
        in ``loadBinary`` -> ``cuModuleLoadData`` and the copy thread in
        ``cuMemcpyAsync`` on the lock, for 20 s). Draining first is safe because no load
        holds the lock yet.
        """

        def guarded(*args, **kwargs):
            if self._copy_armed and not torch.cuda.is_current_stream_capturing():
                self.copy_engine_module_loads += 1
                torch.cuda.synchronize()
            return load(*args, **kwargs)

        return guarded

    def _install_copy_engine_module_load_guard(self) -> None:
        """Guard tvm-ffi's ``load_module`` and Triton's ``load_binary``.

        tvm-ffi loads every JIT library (sglang's ``load_jit`` and flashinfer's), and
        under EAGER its ``dlopen`` loads the library's kernels there and then: the same
        hazard as a Triton load. Both callers look ``load_module`` up on the module at
        call time, so wrapping the attribute covers them.
        """
        try:
            import tvm_ffi
        except ImportError:
            tvm_ffi = None
        if tvm_ffi is not None:
            tvm_ffi.load_module = self._copy_engine_module_load_guard(
                tvm_ffi.load_module
            )
        try:
            from triton.runtime import driver
        except ImportError:
            return
        utils = driver.active.utils
        utils.load_binary = self._copy_engine_module_load_guard(utils.load_binary)

    def _trace_stages(self) -> None:
        """Move the stage records produced since the last check into the trace."""
        if not self._stages_traced:
            return
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        get_exl3_stream_trace().record_ram_miss_requests(
            self.host.drain_trace(), sorted(self._rows, key=self._rows.__getitem__)
        )
        dropped = self.host.trace_dropped()
        if dropped > self._stages_dropped:
            logger.warning(
                "exl3 RAM miss: %d stage records dropped (ring full)",
                dropped - self._stages_dropped,
            )
            self._stages_dropped = dropped

    def _graph_rows(
        self, rows: list[int], *, final: bool = False
    ) -> Optional[tuple[list[int], list[int]]]:
        """Poll a pinned graph-counter readback and queue the next one without waiting.

        The manager's registers survive the forward observer's zeroing of each
        streamer's graph counters. Cumulative snapshots may coalesce while the sole
        pinned slot is in flight. ``final`` runs only after shutdown's GPU barrier; it
        drains the slot and takes the last register value.
        """
        registers = getattr(self._manager, "_registers", None) or {}
        decode = registers.get("decode")
        if decode is None:
            return None
        source = decode["graph_rows"]
        if not source.is_cuda:
            return [int(value) for value in source.sum(dim=0).tolist()], rows

        if not final and torch.cuda.is_current_stream_capturing():
            return None
        ready = None
        if self._trace_pending_rows is not None:
            if final:
                self._trace_event.synchronize()
            elif not self._trace_event.query():
                return None
            ready = (
                [int(value) for value in self._trace_snapshot.tolist()],
                self._trace_pending_rows,
            )
            self._trace_pending_rows = None
        if not final:
            if self._trace_snapshot is None:
                self._trace_snapshot = torch.empty(
                    2, dtype=torch.int64, pin_memory=True
                )
                self._trace_event = torch.cuda.Event(enable_timing=False)
            self._trace_snapshot.copy_(source.sum(dim=0), non_blocking=True)
            self._trace_event.record(torch.cuda.current_stream(source.device))
            self._trace_pending_rows = rows
        return ready

    def _record_graph_snapshot(self, trace, graph: list[int], rows: list[int]) -> None:
        """Record the step delta against the previous snapshot, then keep this one."""
        # A register reset (discard_graph_capture_routes) moves the totals back:
        # re-baseline.
        if self._trace_graph is not None and graph[0] > self._trace_graph[0]:
            routed = graph[0] - self._trace_graph[0]
            # A lagged read can fold multiple bs-1 decode graph steps into one line.
            steps = (
                max(1, round(routed / self.routed_rows_per_step))
                if self.routed_rows_per_step
                else 1
            )
            trace.record_graph_step(
                layer_rows_delta=[a - b for a, b in zip(rows, self._trace_rows)],
                routed_rows=routed,
                routed_misses=graph[1] - self._trace_graph[1],
                thread=self.host.counters(),
                steps=steps,
            )
        self._trace_rows, self._trace_graph = rows, graph

    def _trace_step(self, *, final: bool = False) -> None:
        """Record one graph decode step's gathers and RAM misses (trace runs only)."""
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        trace = get_exl3_stream_trace()
        if not trace.enabled:
            return
        if self.route_log is not None:
            self.route_log.poll(trace, final=final)
        rows = self.host.layer_rows()
        sample = self._graph_rows(rows, final=final)
        if sample is not None:
            self._record_graph_snapshot(trace, *sample)
        if final and self._trace_snapshot is not None:
            # The orderly shutdown barrier already finished all graph work. This last
            # copy captures registers updated after the prior snapshot.
            registers = self._manager._registers["decode"]
            final_graph = [
                int(value) for value in registers["graph_rows"].sum(dim=0).tolist()
            ]
            self._record_graph_snapshot(trace, final_graph, rows)

    def _cuda_active(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_initialized()

    def _synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def _barrier_devices(self) -> list[torch.device]:
        """The CUDA devices whose kernels may read the slabs, each with an index.

        Chosen on the calling thread: a helper thread starts on device 0, so a
        device-less synchronize there waits for the wrong device on any rank that serves
        another GPU (and would resolve an index-less ``cuda`` to device 0 the same way).
        """
        candidates = []
        if self.device_side is not None:
            candidates.append(self.device_side.state.device)
        for layer_id in sorted(self.tables):
            tier = getattr(
                self.tables[layer_id].streamer_of(), "pinned_host_cache", None
            )
            if tier is not None:
                candidates.append(tier.device)
        devices: list[torch.device] = []
        for device in candidates:
            if device.type != "cuda":
                continue
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device not in devices:
                devices.append(device)
        return devices or [torch.device("cuda", torch.cuda.current_device())]

    def _completion_deadline_s(self) -> float:
        # The miss stream is bounded by its deadline (a trap), the copy wait by the
        # watchdog's copy-wait timeout (an abort).
        return envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 5.0

    def _establish_gpu_completion(self) -> Optional[str]:
        """None when every GPU reader is known to have finished, else the reason not.

        The device-wide synchronize runs on a helper thread with a deadline: it is not
        interruptible, so a hung kernel would otherwise hang shutdown with no way to say
        "uncertain".
        """
        if not self._cuda_active():
            return None
        outcome: dict = {}
        devices = self._barrier_devices()

        def run() -> None:
            try:
                for device in devices:
                    self._synchronize(device)
                outcome["done"] = True
            except BaseException as error:  # noqa: BLE001
                # Any CUDA error means completion is not established.
                outcome["error"] = error

        deadline = self._completion_deadline_s()
        helper = threading.Thread(target=run, daemon=True, name="exl3-shutdown-sync")
        helper.start()
        helper.join(deadline)
        if helper.is_alive():
            return f"the device synchronize did not return within {deadline:.1f} s"
        if "error" in outcome:
            return f"the device synchronize failed: {outcome['error']!r}"
        return None

    def shutdown(self, *, at_exit: bool = False) -> None:
        """Shut the service down: free the tiers if safe, else quarantine them.

        Idempotent; the tiers must not be used afterwards. The sequence is a GPU barrier
        proving no reader runs while the service still serves, then close admission,
        stop the thread and free. Anything that leaves it unknown whether the service
        thread or a GPU reader still runs (the barrier, or stopping the thread) makes
        the shutdown quarantine instead (analysis/dsv41-drive/LEASE_PROTOCOL.md,
        "Shutdown").

        The barrier runs before admission closes: an in-flight chain waits on the
        service, so closing first would leave it spinning until the miss stream's
        deadline traps. ``_shut_down`` is set first, so no new forward starts.

        ``at_exit`` is the exit hook's path: no device barrier is attempted in an exit
        handler, so the tiers are quarantined unconditionally. The scheduler's graceful
        shutdown takes the orderly path through ``shutdown_exl3_ram_miss_service``.
        """
        if self._completed or (self._shut_down and not at_exit):
            return
        # At exit a shutdown that started and did not complete is finished as a
        # quarantine, never left half done.
        uncertain: Optional[str] = (
            "an earlier shutdown did not complete" if self._shut_down else None
        )
        self._shut_down = True
        stop_error: Optional[BaseException] = None
        # A KeyboardInterrupt or SystemExit: re-raised once the slabs are safe.
        interrupt: Optional[BaseException] = None
        try:
            if self.host is not None:
                if uncertain is None:
                    uncertain = (
                        "process exit: no device barrier is attempted"
                        if at_exit
                        else self._establish_gpu_completion()
                    )
                self.host.close_admission()
                if uncertain is None and self._stages_traced:
                    # host.stop() closes its native handle, so freeze the reader between
                    # requests before taking its final demand counters. The GPU barrier
                    # above has retired every demand and lease.
                    self.host.pause(
                        2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0
                    )
                    self._trace_step(final=True)
        except BaseException as error:  # noqa: BLE001
            # A failure even to close admission leaves the state uncertain.
            uncertain = f"closing admission or the barrier failed: {error!r}"
            if not isinstance(error, Exception):
                interrupt = error
        finally:
            try:
                # The service thread writes into the slabs through raw addresses, so it
                # stops before anything is freed. A thread hung in a read ends this in
                # the service watchdog's abort.
                if self.host is not None:
                    logger.info(
                        "exl3 RAM miss: stopping the service thread; a read that hangs ends in the watchdog's abort "
                        "after max(30 s, 3 x the wait timeout)"
                    )
                    self.host.stop()
            except BaseException as error:  # noqa: BLE001
                # A thread that may still run must not have its slabs freed.
                stop_error = error
                uncertain = (
                    uncertain or f"stopping the service thread failed: {error!r}"
                )
                if not isinstance(error, Exception):
                    interrupt = interrupt or error
            finally:
                if uncertain is None:
                    for layer_id in sorted(self.tables):
                        streamer = self.tables[layer_id].streamer_of()
                        tier = getattr(streamer, "pinned_host_cache", None)
                        if tier is not None:
                            tier.close()
                else:
                    self._quarantine(uncertain)
                self._completed = True
        if interrupt is not None:
            # KeyboardInterrupt and SystemExit go on, after the quarantine.
            raise interrupt
        if stop_error is not None:
            logger.error(
                "exl3 RAM miss: stopping the service thread failed: %r", stop_error
            )

    def _quarantine(self, reason: str) -> None:
        """Keep everything a GPU kernel may still read or write alive until exit.

        Frees nothing: the tiers' slabs, the request page, the slot map, the lease block
        and every device buffer of the chain are handed to ``quarantine_host_slabs``.
        """
        logger.error(
            "exl3 RAM miss: shutdown could not establish that no GPU reader is running (%s); the pinned slabs, the "
            "request page, the slot map, the lease block and the device buffers are quarantined until process exit",
            reason,
        )
        owned: list[torch.Tensor] = []
        if self.host is not None:
            owned += [self.host.page, self.host.slot_map, self.host.lease_block]
            if self.host.hot_page is not None:
                owned.append(self.host.hot_page)
        if self.device_side is not None:
            side = self.device_side
            owned += [
                side.state,
                side.go_1,
                side.host_rows_1,
                side.dst_slots_1,
                side.lane_kind,
                side.lane_slot,
                side.lane_node,
                side.ce_mask,
                side.cpu_lanes,
                side.piece_runs,
                *side.map_bank.values(),
            ]
        if self._trace_snapshot is not None:
            # A copy can still be writing this pinned block when the device barrier
            # failed.
            owned.append(self._trace_snapshot)
        if self._hot_snapshot is not None:
            owned.append(self._hot_snapshot)
        if self.route_log is not None:
            owned += self.route_log.tensors()
        if self._gpu_hot_updater is not None:
            owned.append(self._gpu_hot_updater.slot_to_expert)
        for layer_id in sorted(self.tables):
            streamer = self.tables[layer_id].streamer_of()
            tier = getattr(streamer, "pinned_host_cache", None)
            if tier is not None:
                tier.quarantine()
            backend = getattr(streamer, "row_backend", None)
            for name in (
                "host_rows",
                "ram_miss",
                "keep",
                "routes",
                "planned",
                "hot_slots",
            ):
                tensor = getattr(backend, name, None)
                if isinstance(tensor, torch.Tensor):
                    owned.append(tensor)
        quarantine_host_slabs(owned)
        self._quarantined = True


def shutdown_exl3_ram_miss_service() -> None:
    """The scheduler's graceful-shutdown entry: shut the service down if one exists.

    A no-op when no service was ever created, so a run without EXL3 does not construct
    the singleton at shutdown.
    """
    service = Exl3RamMissService._instance
    if service is not None:
        service.shutdown()
