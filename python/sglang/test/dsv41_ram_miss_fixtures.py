"""A fake EXL3 checkpoint plus per-layer pinned-slab stand-ins for the option C CPU tests."""

from __future__ import annotations

import contextlib
import os
import pathlib
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
# (dsv41's are; write_fake_exl3's defaults give 256-byte w2_suh/w2_svh rows): trellis rows of 49152 (w13) and 24576
# (w2) bytes and scale rows of 512 and 1024, an image of 76800 bytes (18.75 pages, so the last part ends inside a
# page) and a row stride of 77824.
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


def _takes_o_direct(path: str) -> bool:
    try:
        os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
        return True
    except OSError:
        return False


def require_o_direct(path: str) -> None:
    """The reader reads row images with O_DIRECT only (plan 2026-09-29-hotpath-zero-overhead): a test filesystem that
    refuses it (tmpfs before Linux 6.6, some overlays) must fail loudly here, not run a buffered read that production
    can no longer take."""
    if not _takes_o_direct(path):
        raise RuntimeError(f"{path}: the RAM-miss tests need a filesystem that takes O_DIRECT (tmpfs needs Linux 6.6+)")


def service_row_images(source) -> contextlib.ExitStack:
    """What an option C service needs to read row images, for a test that builds its tiers over the fake checkpoint
    at ``source`` (written by write_fake_exl3 with ``ROW_IMAGE_DIM`` multiples as ``hidden``/``inter``): a mirror root
    beside it holding a copy of its shards and a reference build of its row images, and the env the service reads
    them with. Returns the entered env overrides; close the stack when the test is done.

    The formats must be built inside the stack with ``source_root=source`` and no ``direct`` (so
    SGLANG_MOE_EXPERT_FILE_READER=uring_direct decides it, as in production). SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES
    and SGLANG_DSV41_ENABLE_RAM_MISS_LEASES (which the images need) are set until those knobs are removed (plan
    2026-09-29-hotpath-zero-overhead Task 7). Modelled on test_exl3_prefill_fills_service._build."""
    from sglang.srt.environ import envs

    source = pathlib.Path(source)
    root = source.parent / f"{source.name}_images_mirror"
    shutil.copytree(source, root)
    layout = build_exl3_expert_layout(str(source))
    segments = Exl3ExpertFormat(layout, 0, direct=False, source_root=str(source)).segment_map()
    write_row_images(layout, segments, str(source), [str(root)])
    require_o_direct(os.path.join(ri.row_image_dir(str(root)), ri.layer_file_name(0)))
    stack = contextlib.ExitStack()
    for env, value in (
        (envs.SGLANG_MOE_EXPERT_MIRROR_DIRS, str(root)),
        (envs.SGLANG_MOE_EXPERT_FILE_READER, "uring_direct"),
        (envs.SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES, True),
        (envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES, True),
    ):
        stack.enter_context(env.override(value))
    return stack


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
    row_images: bool = True,
) -> RamMissSetup:
    """``mirror_weights``: one weight per mirror root; copies are made beside ``tmp_path`` and the
    tables split every row across them (``parts == len(mirror_weights)``). ``hidden``/``inter`` size the fake
    experts (write_fake_exl3's defaults when None): larger ones give rows many pages long.

    ``row_images`` (the default, plan 2026-09-29-hotpath-zero-overhead): the tables read row images (the reader's direct
    mode) instead of the checkpoint. The fake experts default to ``ROW_IMAGE_DIM`` (the images need 512-byte slab
    rows), each mirror root (one root, weight 1, when ``mirror_weights`` is None) holds images built by
    ``write_row_images`` and no shard copies, and ``reference`` is still the checkpoint read by Exl3ShardRowSource, so
    a test compares the direct mode against the bounce path's oracle. The image files must take O_DIRECT
    (``require_o_direct``): production reads them with nothing else. ``row_images=False`` builds shard tables for the
    bounce-only tests still pinned to the packed reader."""
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
    if row_images:
        require_o_direct(tables.paths[0])
    return RamMissSetup(layout, fmt, specs, slabs, tables, roots)
