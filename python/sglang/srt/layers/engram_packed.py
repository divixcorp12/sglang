"""The page-packed Engram table format: a row's fp8 weights and e8m0 scales stored together.

A split shard (the checkpoint's own layout) keeps a row's 256 B of weights and its 8 B of scales ~98 GB apart, so a
native-store miss reads two 4 KiB pages, three for the 1 row in 16 whose weights cross a page. A packed table stores
the 264 B row whole, 15 rows to a 4 KiB page with the page's last 136 B zero, behind a safetensors header padded so
the data starts on a page. Row r is page r // 15, byte (r % 15) * 264 of it: no row crosses a page, and a miss reads
exactly one page with O_DIRECT.

The file is one U8 tensor ``layers.{L}.engram.embed.packed`` of shape [pages, 4096]; ``__metadata__`` names the format,
its version and geometry, the source shard and the sha256 of the data region (scripts/dsv41/engram_relayout.py).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

PAGE_BYTES = 4096
FORMAT = "engram-packed"
VERSION = "1"
# The header (8-byte length + JSON) is padded to this many bytes, so the data region starts on a page.
HEADER_BYTES = PAGE_BYTES
# The checksum placeholder the converter writes first and replaces in place, at the same length, when done.
UNSET_SHA256 = "0" * 64


def packed_key(layer_id: int) -> str:
    return f"layers.{layer_id}.engram.embed.packed"


@dataclass(frozen=True)
class PackedGeometry:
    rows: int
    dim: int
    scale_dim: int

    @property
    def row_bytes(self) -> int:
        return self.dim + self.scale_dim

    @property
    def rows_per_page(self) -> int:
        return PAGE_BYTES // self.row_bytes

    @property
    def pages(self) -> int:
        return -(-self.rows // self.rows_per_page)

    @property
    def data_bytes(self) -> int:
        return self.pages * PAGE_BYTES

    def row_offset(self, row: int) -> int:
        """Row ``row``'s byte offset in the data region."""
        return (row // self.rows_per_page) * PAGE_BYTES + (row % self.rows_per_page) * self.row_bytes


def packed_header(layer_id: int, geometry: PackedGeometry, metadata: dict[str, str]) -> bytes:
    """The file's first HEADER_BYTES: length, JSON header and space padding (safetensors allows it)."""
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "rows": str(geometry.rows),
        "dim": str(geometry.dim),
        "scale_dim": str(geometry.scale_dim),
        "rows_per_page": str(geometry.rows_per_page),
        "page_bytes": str(PAGE_BYTES),
        **metadata,
    }
    header = {
        "__metadata__": meta,
        packed_key(layer_id): {
            "dtype": "U8",
            "shape": [geometry.pages, PAGE_BYTES],
            "data_offsets": [0, geometry.data_bytes],
        },
    }
    blob = json.dumps(header).encode()
    if 8 + len(blob) > HEADER_BYTES:
        raise ValueError(f"packed Engram header is {len(blob)} bytes, more than {HEADER_BYTES - 8}")
    blob += b" " * (HEADER_BYTES - 8 - len(blob))
    return struct.pack("<Q", len(blob)) + blob


def read_header(path: str) -> tuple[int, dict]:
    """(data start, header) with ``__metadata__`` kept (read_safetensors_header drops it)."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return 8 + length, json.loads(f.read(length))


def is_packed(header: dict, weight_key: str) -> bool:
    return weight_key not in header and packed_key_of(weight_key) in header


def packed_key_of(weight_key: str) -> str:
    """``layers.L.engram.embed.weight`` -> ``layers.L.engram.embed.packed``."""
    return weight_key.rsplit(".", 1)[0] + ".packed"


def read_packed_geometry(
    path: str, weight_key: str, num_embeddings: int, dim: int, block: int
) -> tuple[int, PackedGeometry]:
    """(data offset, geometry) of a packed table, validated against the layer it is opened for; raise on any
    mismatch, and on a file whose conversion did not finish (no checksum)."""
    data_start, header = read_header(path)
    meta = header.get("__metadata__") or {}
    if (meta.get("format"), meta.get("version")) != (FORMAT, VERSION):
        raise ValueError(
            f"{path}: packed Engram format {meta.get('format')!r} version {meta.get('version')!r}, "
            f"expected {FORMAT!r} version {VERSION!r}"
        )
    geometry = PackedGeometry(num_embeddings, dim, dim // block)
    want = {
        "rows": geometry.rows, "dim": geometry.dim, "scale_dim": geometry.scale_dim,
        "rows_per_page": geometry.rows_per_page, "page_bytes": PAGE_BYTES,
    }
    got = {k: int(meta.get(k, -1)) for k in want}
    if got != want:
        raise ValueError(f"{path}: packed Engram geometry {got} != {want}")
    entry = header[packed_key_of(weight_key)]
    if entry["dtype"] != "U8" or entry["shape"] != [geometry.pages, PAGE_BYTES]:
        raise ValueError(f"{path}: packed tensor {entry['dtype']} {entry['shape']} != U8 [{geometry.pages}, 4096]")
    offset = data_start + entry["data_offsets"][0]
    if offset % PAGE_BYTES:
        raise ValueError(f"{path}: packed data starts at {offset}, not on a page")
    if os.path.getsize(path) < offset + geometry.data_bytes:
        raise ValueError(f"{path}: {os.path.getsize(path)} bytes, short of {offset + geometry.data_bytes}")
    if meta.get("data_sha256", UNSET_SHA256) == UNSET_SHA256:
        raise ValueError(f"{path}: packed Engram table has no data checksum; its conversion did not finish")
    return offset, geometry


def pack_rows(weight: np.ndarray, scale: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """[n, dim] and [n, scale_dim] uint8 rows as [ceil(n / rows_per_page), 4096] packed pages, written into ``out``
    (at least that many pages; the converter's aligned buffer) or a new array. Padding and empty slots are zero."""
    n, dim = weight.shape
    geometry = PackedGeometry(n, dim, scale.shape[1])
    rpp = geometry.rows_per_page
    pages = np.empty((geometry.pages, PAGE_BYTES), np.uint8) if out is None else out[: geometry.pages]
    slots = packed_rows_view(pages, dim, scale.shape[1])
    # The [pages, rpp, row_bytes] view does not flatten to rows without a copy, so full pages and the last partial
    # page are written separately, each through the view.
    full = n // rpp
    slots[:full, :, :dim] = weight[: full * rpp].reshape(full, rpp, dim)
    slots[:full, :, dim:] = scale[: full * rpp].reshape(full, rpp, -1)
    if full < geometry.pages:
        tail = n - full * rpp
        slots[full, :tail, :dim] = weight[full * rpp :]
        slots[full, :tail, dim:] = scale[full * rpp :]
        slots[full, tail:] = 0
    pages[:, rpp * geometry.row_bytes :] = 0
    return pages


def packed_rows_view(pages: np.ndarray, dim: int, scale_dim: int) -> np.ndarray:
    """[pages, rows_per_page, row_bytes] view of packed pages (a memmap stays a memmap)."""
    rb = dim + scale_dim
    rpp = PAGE_BYTES // rb
    return pages[:, : rpp * rb].reshape(pages.shape[0], rpp, rb)
