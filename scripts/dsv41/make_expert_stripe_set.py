"""Offline repack: split every EXL3 expert row into K page-aligned stripe fragments.

Reads the source shards through the same `Exl3RowReader` the runtime uses (one
O_DIRECT read per row, `direct=True`) and writes each row's fragments to K
per-drive directories, one `layer-<L>.bin` per layer, streamed one row at a
time so a whole layer is never held in RAM. Fragment placement follows
`StripeGeometry` exactly (`fragment_bytes` for payload sizes, `strides` for the
page-aligned on-disk slot size, `slot_offset` for where a slot starts) so this
tool never recomputes alignment math of its own.

Interruption and re-run safety:
- Each `layer-<L>.bin` is built under a sibling temporary name
  (`layer-<L>.bin.partial`) and only `os.replace()`d to its final name once
  fully written and flushed. `os.replace` is atomic within one filesystem, so
  an interrupted run never leaves a `layer-<L>.bin` that looks complete but
  isn't; it can leave a stray `.partial` file, which the next run's directory
  check below also refuses to run into.
- `manifest.json` is written last, after every layer file has been renamed
  into place: it is this stripe set's completion marker. **A stripe
  directory without `manifest.json` is incomplete and must not be read by a
  consumer** (Task 3's row source, `verify_stripe_set`, or anything else).
- `write_stripe_set` refuses to write into an `out_dir` that already
  contains `layer-*.bin`, `layer-*.bin.partial`, or `manifest.json`, unless
  `force=True`, in which case it deletes exactly those stale files first so
  two repack generations are never mixed in the same directory.

The `--layers` caveat a Task 3 consumer needs: **the manifest's
`num_layers`/`num_experts` describe the SOURCE checkpoint, not what this run
actually wrote.** A partial run (`--layers` given, or one interrupted and
resumed with a narrower `--layers`) can produce a manifest whose
`num_layers` exceeds the number of `layer-<L>.bin` files actually present.
A consumer must probe for a given layer's file, not assume its existence
from `num_layers`.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_stripe_layout import (
    PAGE_BYTES,
    StripeGeometry,
    StripeInfo,
    StripeManifest,
)

_STALE_PATTERNS = ("layer-*.bin", "layer-*.bin.partial", "manifest.json")


def _default_sample_pairs(layer_ids: Sequence[int], num_experts: int) -> set[tuple[int, int]]:
    """A handful of (layer, expert) pairs spread across the layers actually written."""
    if not layer_ids:
        return set()
    mid = layer_ids[len(layer_ids) // 2]
    return {
        (layer_ids[0], 0),
        (mid, num_experts // 2),
        (layer_ids[-1], num_experts - 1),
    }


def _stale_paths(out_dir: str) -> list[str]:
    paths = []
    for pattern in _STALE_PATTERNS:
        paths.extend(glob.glob(os.path.join(out_dir, pattern)))
    return sorted(paths)


def _prepare_out_dir(out_dir: str, force: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stale = _stale_paths(out_dir)
    if not stale:
        return
    if not force:
        raise FileExistsError(
            f"{out_dir} already has stripe-set contents: {stale}; "
            "pass force=True (CLI: --force) to overwrite, or pick an empty directory"
        )
    for path in stale:
        os.remove(path)


def write_stripe_set(
    source_dir: str,
    out_dirs: Sequence[str],
    weights: Sequence[float],
    layers: Optional[Sequence[int]] = None,
    *,
    force: bool = False,
) -> StripeManifest:
    """Repack `source_dir`'s expert rows into a K-stripe set under `out_dirs`.

    Streams one row at a time: reads it once with `Exl3RowReader` (O_DIRECT,
    matching production), slices it per `StripeGeometry`, and appends each
    slice straight to its stripe's layer file. Writing proceeds expert 0..N-1
    in order for a fixed layer, so a plain sequential append naturally lands
    every slot at `slot_offset(stripe, expert)` with no seeking needed.

    Every layer file is written to a `.partial` sibling and atomically
    `os.replace()`d into place only once complete, so an interrupted run
    leaves no truncated file under the final `layer-<L>.bin` name. Each
    `out_dir` must be empty of stripe-set contents (`layer-*.bin`,
    `layer-*.bin.partial`, `manifest.json`) unless `force=True`, in which case
    those stale files are removed before writing starts, never mixed with the
    new generation. `manifest.json` is written last, after every layer file
    has been renamed into place, because a stripe directory without it is
    incomplete by definition — see the module docstring.

    Caution for callers that pass `layers`: the returned manifest's
    `num_layers`/`num_experts` always describe `source_dir`'s full layout,
    not which `layer-<L>.bin` files this call actually wrote. Check which
    files exist rather than trusting `num_layers` as an inventory.
    """
    layout = build_exl3_expert_layout(source_dir)
    geometry = StripeGeometry(row_bytes=layout.row_bytes, weights=tuple(weights))
    num_stripes = len(geometry.fragment_bytes)
    if len(out_dirs) != num_stripes:
        raise ValueError(f"{len(out_dirs)} out_dirs given for {num_stripes} weights")

    layer_ids = list(range(layout.num_layers)) if layers is None else list(layers)
    for out_dir in out_dirs:
        _prepare_out_dir(out_dir, force)

    reader = Exl3RowReader(layout, direct=True)
    # One page-aligned host buffer, reused for every row: `reader.read` writes
    # into it and returns where the row starts (the shard offset is not itself
    # page-aligned, see exl3_row_reader.py).
    storage = torch.zeros(reader.buffer_bytes + PAGE_BYTES, dtype=torch.uint8)
    pad = (-storage.data_ptr()) % PAGE_BYTES
    row_buf = storage[pad : pad + reader.buffer_bytes]

    sample_pairs = _default_sample_pairs(layer_ids, layout.num_experts)
    row_sha256_sample: dict[str, str] = {}
    bytes_written = [0] * num_stripes

    start_time = time.monotonic()
    for layer in layer_ids:
        final_paths = [os.path.join(out_dir, f"layer-{layer}.bin") for out_dir in out_dirs]
        partial_paths = [path + ".partial" for path in final_paths]
        files = [open(path, "wb") for path in partial_paths]
        try:
            for expert in range(layout.num_experts):
                (row_start,) = reader.read([(layer, expert)], [row_buf.data_ptr()])
                row = bytes(row_buf[row_start : row_start + layout.row_bytes].numpy())
                if (layer, expert) in sample_pairs:
                    row_sha256_sample[f"{layer}:{expert}"] = hashlib.sha256(row).hexdigest()
                for stripe, f in enumerate(files):
                    payload = row[geometry.starts[stripe] : geometry.starts[stripe] + geometry.fragment_bytes[stripe]]
                    f.write(payload)
                    tail = geometry.strides[stripe] - len(payload)
                    if tail:
                        f.write(b"\x00" * tail)
                    bytes_written[stripe] += len(payload) + tail
        finally:
            for f in files:
                f.close()
        # Rename only after every stripe's file for this layer is fully
        # written and closed (flushed to the fs), so a crash between layers
        # never leaves a `layer-<L>.bin` that looks complete but isn't. A
        # crash mid-layer leaves `.partial` files instead, which the next
        # run's `_prepare_out_dir` refuses to run into (unless `--force`).
        for partial_path, final_path in zip(partial_paths, final_paths):
            os.replace(partial_path, final_path)
    elapsed = time.monotonic() - start_time

    stripes = tuple(
        StripeInfo(
            index=stripe,
            weight=float(geometry.weights[stripe]),
            fragment_bytes=geometry.fragment_bytes[stripe],
            dir_hint=os.path.abspath(out_dirs[stripe]),
        )
        for stripe in range(num_stripes)
    )

    manifests = []
    for stripe, out_dir in enumerate(out_dirs):
        manifest = StripeManifest(
            version=1,
            source=os.path.abspath(source_dir),
            num_layers=layout.num_layers,
            num_experts=layout.num_experts,
            row_bytes=layout.row_bytes,
            tensor_order=tuple(t.name for t in layout.tensors),
            stripes=stripes,
            row_sha256_sample=row_sha256_sample,
            index=stripe,
        )
        # manifest.json is this stripe set's completion marker (see module
        # docstring): written last, and only after every layer file above has
        # already been renamed into its final name.
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            f.write(manifest.to_json())
        manifests.append(manifest)

    print(f"wrote stripe set for {len(layer_ids)} layer(s) in {elapsed:.2f}s")
    for stripe, out_dir in enumerate(out_dirs):
        print(f"  stripe {stripe} ({out_dir}): {bytes_written[stripe]} bytes")

    return manifests[0]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Source EXL3 checkpoint directory")
    parser.add_argument(
        "--out",
        dest="out_dirs",
        action="append",
        required=True,
        help="Output stripe directory; repeat once per stripe, in stripe order",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Colon-separated per-stripe weights, one per --out, in the same order",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help=(
            "Comma-separated layer indices to write (default: every layer). "
            "The written manifest's num_layers/num_experts always describe the "
            "SOURCE checkpoint, not which layer-<L>.bin files this run wrote; "
            "a consumer must probe for a layer's file rather than trust "
            "num_layers as an inventory of what is on disk."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete any existing layer-*.bin, layer-*.bin.partial, or "
            "manifest.json in each --out directory before writing, instead of "
            "refusing to run into a populated directory. Without this flag, "
            "a populated --out directory is left untouched and the run fails."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    weights = [float(w) for w in args.weights.split(":")]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    write_stripe_set(args.source, args.out_dirs, weights, layers=layers, force=args.force)


if __name__ == "__main__":
    main()
