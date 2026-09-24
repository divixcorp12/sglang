"""Row images: EXL3 expert rows re-laid so an O_DIRECT readv lands them straight in the pinned slabs.

A checkpoint row packs its tensors with no padding at odd file offsets, so its bytes can only reach the per-name
slabs through a bounce buffer and a copy. A row image is the same row with every byte already where the slabs hold
it: the six streamed names' slab rows back to back, in ``EXL3_STREAMED_NAMES`` order. Every slab row is a multiple
of 512 bytes, so every name starts on a 512-byte boundary of the image, and a readv whose file offset and segment
lengths are 512-aligned scatters an image range into the slab rows (probe: analysis/dsv41-drive/direct-read-probe).

On disk, under ``<mirror root>/exl3_row_images/``:

* ``layer-LLL.rows`` per layer: expert ``e``'s image at byte ``e * row_stride``, zero-padded to ``row_stride``
  (``image_bytes`` rounded up to a page). A reader reads exactly ``image_bytes``; the padding is never read.
* ``manifest.json``, written last (by rename): the format, the image layout, the source it was built from, and a
  digest per row. A root without a complete manifest that matches the live layout is refused, never read.

The byte mapping is ``image[image_offset(s) : + s.nbytes] = row[s.src_offset : + s.nbytes]`` for each RowSegment
``s`` of the format's segment map, the same copies the packing path makes into the slabs.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, Mapping, Sequence

import msgspec

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout

ROW_IMAGE_SUBDIR = "exl3_row_images"
FORMAT = "exl3-row-images"
VERSION = 1
MANIFEST = "manifest.json"
# O_DIRECT on the mirror drives (XFS and ext4, 512 B logical blocks): file offsets and segment lengths.
IO_ALIGN = 512
PAGE = 4096


def layer_file_name(layer_id: int) -> str:
    return f"layer-{layer_id:03d}.rows"


def row_image_dir(root: str) -> str:
    return os.path.join(root, ROW_IMAGE_SUBDIR)


class RowImageLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Where each streamed name's slab row sits in a row image; indexed in ``EXL3_STREAMED_NAMES`` order."""

    row_bytes: tuple[int, ...]
    name_offsets: tuple[int, ...]
    image_bytes: int
    row_stride: int
    segments: tuple[RowSegment, ...]

    def image_offset(self, segment: RowSegment) -> int:
        return self.name_offsets[EXL3_STREAMED_NAMES.index(segment.name)] + segment.dst_offset

    def to_json(self) -> dict:
        return {
            "names": list(EXL3_STREAMED_NAMES),
            "row_bytes": list(self.row_bytes),
            "name_offsets": list(self.name_offsets),
            "image_bytes": self.image_bytes,
            "row_stride": self.row_stride,
            "segments": [[s.name, s.part, s.dst_offset, s.src_offset, s.nbytes] for s in self.segments],
        }


def row_image_layout(segments: Sequence[RowSegment]) -> RowImageLayout:
    """The image layout of a format's segment map; refuses a map a readv could not land directly."""
    row_bytes = []
    for name in EXL3_STREAMED_NAMES:
        spans = sorted((s.dst_offset, s.dst_offset + s.nbytes) for s in segments if s.name == name)
        if not spans:
            raise ValueError(f"row images: no segment fills streamed name {name}")
        cursor = 0
        for lo, hi in spans:
            if lo != cursor:
                raise ValueError(f"row images: {name}'s segments leave a gap or overlap at byte {cursor}")
            cursor = hi
        if cursor % IO_ALIGN:
            raise ValueError(f"row images: {name}'s slab row is {cursor} B, not a multiple of {IO_ALIGN}")
        row_bytes.append(cursor)
    unknown = sorted({s.name for s in segments} - set(EXL3_STREAMED_NAMES))
    if unknown:
        raise ValueError(f"row images: segments name {unknown}, which are not streamed")
    offsets, at = [], 0
    for size in row_bytes:
        offsets.append(at)
        at += size
    return RowImageLayout(
        row_bytes=tuple(row_bytes),
        name_offsets=tuple(offsets),
        image_bytes=at,
        row_stride=-(-at // PAGE) * PAGE,
        segments=tuple(segments),
    )


def image_of_row(layout: RowImageLayout, row: bytes | bytearray | memoryview) -> bytearray:
    """The image (``image_bytes``, no padding) of one on-disk row (the record's ``nbytes`` from its file offset)."""
    raw = memoryview(row)
    image = bytearray(layout.image_bytes)
    for s in layout.segments:
        if s.src_offset + s.nbytes > len(raw):
            raise ValueError(f"row images: segment {s} runs past the {len(raw)} B row")
        at = layout.image_offset(s)
        image[at : at + s.nbytes] = raw[s.src_offset : s.src_offset + s.nbytes]
    return image


def row_digest(image: bytes | bytearray | memoryview) -> str:
    return hashlib.blake2b(image, digest_size=8).hexdigest()


def source_fingerprint(layout: Exl3ExpertLayout, source_root: str) -> dict:
    """What an image set was built from: every shard the layout reads (relative path, size) and a digest of the
    row records and tensor spans. Two layouts with equal fingerprints read the same bytes into the same places."""
    shards = sorted({r.path for r in layout.records.values()})
    records = hashlib.sha256()
    for key in sorted(layout.records):
        r = layout.records[key]
        records.update(f"{r.layer},{r.expert},{os.path.relpath(r.path, source_root)},{r.file_offset},{r.nbytes};".encode())
    for t in layout.tensors:
        records.update(f"{t.name},{t.rel_offset},{t.nbytes},{t.dtype},{tuple(t.shape)};".encode())
    return {
        "shards": [[os.path.relpath(p, source_root), os.path.getsize(p)] for p in shards],
        "num_layers": layout.num_layers,
        "num_experts": layout.num_experts,
        "row_bytes": layout.row_bytes,
        "records_sha256": records.hexdigest(),
    }


def manifest_json(
    image_layout: RowImageLayout,
    fingerprint: dict,
    digests: Mapping[int, Sequence[str]],
) -> dict:
    """The manifest of a complete image set; ``digests[layer_id][expert]`` is ``row_digest`` of that image."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "complete": True,
        "layout": image_layout.to_json(),
        "source": fingerprint,
        "layers": {str(layer): {"file": layer_file_name(layer), "digests": list(d)} for layer, d in sorted(digests.items())},
    }


def write_manifest(root: str, manifest: dict) -> None:
    """Write ``manifest`` into ``root``'s image dir atomically (tmp, fsync, rename, fsync dir). Call it only after
    every layer file is written and fsynced: the manifest is what makes the set readable."""
    directory = row_image_dir(root)
    tmp = os.path.join(directory, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(directory, MANIFEST))
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_manifest(root: str) -> dict:
    path = os.path.join(row_image_dir(root), MANIFEST)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"row images: {path} is missing; build the images with scripts/dsv41/build_row_images.py") from None


class RowImageSet(msgspec.Struct, frozen=True, kw_only=True):
    """Validated row images on one or more roots. ``paths[layer_id][p]`` is root ``p``'s file for that layer."""

    roots: tuple[str, ...]
    layout: RowImageLayout
    num_experts: int
    paths: Mapping[int, tuple[str, ...]]


def open_row_images(
    roots: Sequence[str],
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    source_root: str,
    layer_ids: Iterable[int],
) -> RowImageSet:
    """The images of ``layer_ids`` on every root, refused unless each root's manifest is complete, describes this
    exact layout and source, carries the same row digests as the others, and its files have their full size."""
    if not roots:
        raise ValueError("row images need at least one root")
    image_layout = row_image_layout(segments)
    want_layout = image_layout.to_json()
    want_source = source_fingerprint(layout, source_root)
    layer_ids = sorted(layer_ids)
    first_digests = None
    paths: dict[int, list[str]] = {layer: [] for layer in layer_ids}
    for root in roots:
        m = read_manifest(root)
        where = row_image_dir(root)
        if m.get("format") != FORMAT or m.get("version") != VERSION:
            raise ValueError(f"row images at {where}: format {m.get('format')!r} v{m.get('version')!r}, expected {FORMAT!r} v{VERSION}")
        if m.get("complete") is not True:
            raise ValueError(f"row images at {where}: the manifest is not marked complete")
        if m.get("layout") != want_layout:
            raise ValueError(f"row images at {where}: built for a different image layout than this checkpoint's")
        if m.get("source") != want_source:
            raise ValueError(f"row images at {where}: built from a different source than {source_root}")
        layers = m.get("layers", {})
        missing = [layer for layer in layer_ids if str(layer) not in layers]
        if missing:
            raise ValueError(f"row images at {where}: no images for layers {missing}")
        digests = {layer: layers[str(layer)]["digests"] for layer in layer_ids}
        for layer in layer_ids:
            if len(digests[layer]) != layout.num_experts:
                raise ValueError(f"row images at {where}: layer {layer} has {len(digests[layer])} digests for {layout.num_experts} experts")
            path = os.path.join(where, layers[str(layer)]["file"])
            size = os.path.getsize(path) if os.path.exists(path) else -1
            if size != layout.num_experts * image_layout.row_stride:
                raise ValueError(f"row images: {path} is {size} B, expected {layout.num_experts * image_layout.row_stride}")
            paths[layer].append(path)
        if first_digests is None:
            first_digests = digests
        elif digests != first_digests:
            raise ValueError(f"row images at {where}: row digests differ from {row_image_dir(roots[0])}'s")
    return RowImageSet(
        roots=tuple(roots),
        layout=image_layout,
        num_experts=layout.num_experts,
        paths={layer: tuple(p) for layer, p in paths.items()},
    )
