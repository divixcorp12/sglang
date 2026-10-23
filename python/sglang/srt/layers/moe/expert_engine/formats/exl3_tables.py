"""The EXL3 flat tables the native RAM-miss reader runs from.

``Exl3RamMissTables`` and ``exl3_ram_miss_tables`` flatten a checkpoint's row images
and each layer's pinned slabs into the int64 tensors the native thread reads;
``open_service_row_images`` opens the images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_read_split import SplitPolicy, StaticSplitPolicy
from sglang.srt.layers.moe.exl3_row_image import RowImageSet
from sglang.srt.layers.moe.expert_engine.formats.exl3 import (
    EXL3_STREAMED_NAMES,
    RowSegment,
)


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
            "exl3 RAM miss: the RAM-miss service reads row images and needs SGLANG_EXPERT_STREAM_MIRROR's dirs (roots "
            "holding images built by scripts/dsv41/build_row_images.py)"
        )
    if not direct:
        raise RuntimeError(
            "exl3 RAM miss: the RAM-miss service reads row images and needs "
            "SGLANG_EXPERT_STREAM_FILE_READER=uring_direct"
        )
    from sglang.srt.layers.moe.exl3_row_image import open_row_images

    return open_row_images(
        mirrors["roots"], layout, segments, mirrors["source_root"], list(streamers)
    )
