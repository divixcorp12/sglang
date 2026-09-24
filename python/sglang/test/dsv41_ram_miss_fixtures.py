"""A fake EXL3 checkpoint plus per-layer pinned-slab stand-ins for the option C CPU tests."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.test.dsv41_fake_exl3 import write_fake_exl3


@dataclass
class RamMissSetup:
    layout: object
    fmt: Exl3ExpertFormat
    specs: dict
    slabs: dict
    tables: object
    roots: tuple = ()  # byte-identical copies of the checkpoint, one per mirror weight

    def reference(self, layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
        """Exl3ShardRowSource's split of ``experts`` of ``layer`` (the byte oracle)."""
        out = {
            name: torch.empty((len(experts),) + self.specs[name].row_shape, dtype=self.specs[name].dtype)
            for name in EXL3_STREAMED_NAMES
        }
        Exl3ShardRowSource.for_layer(self.layout, layer, self.fmt.segment_map(), direct=False).read(
            torch.tensor(experts), out
        )
        return out


# Fake-expert dimensions whose six streamed names' slab rows are all multiples of 512 bytes, as row images need
# (dsv41's are; write_fake_exl3's defaults give 256-byte w2_suh/w2_svh rows): 1024, 1024, 49152, 512, 512, 24576,
# an image of 76800 bytes (18.75 pages, so the last part ends inside a page) and a row stride of 77824.
ROW_IMAGE_DIM = 256


def write_row_images(layout, segments, source_root: str, roots: Sequence[str], layer_ids: Optional[Iterable[int]] = None):
    """A reference build of row images on every root, from exl3_row_image's primitives only (the converter,
    scripts/dsv41/build_row_images.py, is what production uses; tests must not depend on it): each row's image by
    ``image_of_row`` of its checkpoint bytes, zero-padded to ``row_stride``, then the manifest. Returns the layout."""
    image_layout = ri.row_image_layout(segments)
    layer_ids = list(range(layout.num_layers)) if layer_ids is None else list(layer_ids)
    digests, files = {}, {}
    for layer in layer_ids:
        out = bytearray()
        digests[layer] = []
        for expert in range(layout.num_experts):
            record = layout.records[(layer, expert)]
            with open(record.path, "rb") as f:
                f.seek(record.file_offset)
                raw = f.read(record.nbytes)
            image = ri.image_of_row(image_layout, raw)
            digests[layer].append(ri.row_digest(image))
            out += image + bytes(image_layout.row_stride - image_layout.image_bytes)
        files[layer] = bytes(out)
    fingerprint = ri.source_fingerprint(layout, source_root)
    for root in roots:
        os.makedirs(ri.row_image_dir(root), exist_ok=True)
        for layer, data in files.items():
            with open(os.path.join(ri.row_image_dir(root), ri.layer_file_name(layer)), "wb") as f:
                f.write(data)
        ri.write_manifest(root, ri.manifest_json(image_layout, fingerprint, digests))
    return image_layout


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ram_miss_setup(
    tmp_path,
    *,
    capacity: int = 3,
    layers: int = 2,
    experts: int = 6,
    mirror_weights=None,
    hidden=None,
    inter=None,
    row_images: bool = False,
) -> RamMissSetup:
    """``mirror_weights``: one weight per mirror root; copies are made beside ``tmp_path`` and the
    tables split every row across them (``parts == len(mirror_weights)``). ``hidden``/``inter`` size the fake
    experts (write_fake_exl3's defaults when None): larger ones give rows many pages long.

    ``row_images``: the tables read row images (the reader's direct mode) instead of the checkpoint. The fake experts
    default to ``ROW_IMAGE_DIM`` (the images need 512-byte slab rows), each mirror root (one root, weight 1, when
    ``mirror_weights`` is None) holds images built by ``write_row_images`` and no shard copies, and ``reference`` is
    still the checkpoint read by Exl3ShardRowSource, so a test compares the direct mode against the bounce path's
    oracle."""
    if row_images:
        hidden = ROW_IMAGE_DIM if hidden is None else hidden
        inter = ROW_IMAGE_DIM if inter is None else inter
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy

    dims = {k: v for k, v in (("hidden", hidden), ("inter", inter)) if v is not None}
    write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=experts, **dims)
    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    slabs = {
        layer: {
            name: allocate_host_slab(capacity, specs[name].row_shape, specs[name].dtype, register=False)
            for name in EXL3_STREAMED_NAMES
        }
        for layer in range(layers)
    }
    roots = ()
    mirrors = {}
    if row_images:
        from sglang.srt.layers.moe.exl3_row_image import open_row_images

        weights = (1.0,) if mirror_weights is None else tuple(mirror_weights)
        roots = tuple(str(tmp_path.parent / f"{tmp_path.name}_images{i}") for i in range(len(weights)))
        write_row_images(layout, fmt.segment_map(), str(tmp_path), roots)
        images = open_row_images(roots, layout, fmt.segment_map(), str(tmp_path), range(layers))
        mirrors = dict(roots=roots, policy=StaticSplitPolicy(weights), source_root=str(tmp_path), row_images=images)
    elif mirror_weights is not None:
        roots = tuple(str(tmp_path.parent / f"{tmp_path.name}_mirror{i}") for i in range(len(mirror_weights)))
        for root in roots:
            shutil.copytree(tmp_path, root)
        mirrors = dict(roots=roots, policy=StaticSplitPolicy(mirror_weights), source_root=str(tmp_path))
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs, **mirrors)
    return RamMissSetup(layout, fmt, specs, slabs, tables, roots)
