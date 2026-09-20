"""Round-trip a synthetic EXL3 checkpoint through the offline stripe repack.

No real checkpoint and no GPU: the source is a tiny fake EXL3 export built by
`sglang.test.dsv41_fake_exl3.write_fake_exl3` (the same fixture
`test_exl3_row_reader.py` uses), whose default dimensions already produce a
`row_bytes` that is not a multiple of 4096 (2700 bytes for HIDDEN=INTER=128),
so the odd-row-size case the plan calls out is exercised by default rather
than needing a special-cased fixture.
"""

from __future__ import annotations

import os

import pytest

from sglang.srt.layers.moe.exl3_stripe_layout import PAGE_BYTES, StripeManifest
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

from scripts.dsv41.make_expert_stripe_set import write_stripe_set

NUM_LAYERS = 2
NUM_EXPERTS = 4


def _read_fragment(out_dir: str, layer: int, expert: int, offset: int, length: int) -> bytes:
    with open(os.path.join(out_dir, f"layer-{layer}.bin"), "rb") as f:
        f.seek(offset)
        return f.read(length)


def test_row_bytes_is_not_page_aligned(tmp_path):
    # Sanity check on the fixture itself: if this ever stops holding, the
    # roundtrip test below silently loses its odd-row-size coverage.
    os.makedirs(tmp_path / "source")
    write_fake_exl3(str(tmp_path / "source"), num_layers=1, num_experts=1)
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout

    layout = build_exl3_expert_layout(str(tmp_path / "source"))
    assert layout.row_bytes % PAGE_BYTES != 0


def test_stripe_set_roundtrip_byte_identical(tmp_path):
    source_dir = str(tmp_path / "source")
    os.makedirs(source_dir)
    rows = write_fake_exl3(source_dir, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS)

    out_dirs = [str(tmp_path / "stripe-0"), str(tmp_path / "stripe-1")]
    manifest = write_stripe_set(source_dir, out_dirs, weights=[3.0, 1.0])

    assert isinstance(manifest, StripeManifest)
    assert manifest.row_bytes % PAGE_BYTES != 0, "fixture must exercise a non-4096-multiple row"
    assert len(manifest.stripes) == 2

    starts = [manifest.stripes[0].fragment_bytes, manifest.stripes[1].fragment_bytes]
    assert starts[0] + starts[1] == manifest.row_bytes

    # Every slot offset on every stripe must be 4096-aligned, including odd
    # expert indices on the stripe whose fragment does not evenly divide 4096
    # (the case the geometry fix's `strides` field exists for).
    from sglang.srt.layers.moe.exl3_stripe_layout import StripeGeometry

    geometry = StripeGeometry(row_bytes=manifest.row_bytes, weights=(3.0, 1.0))
    for stripe in range(2):
        for expert in range(NUM_EXPERTS):
            offset = geometry.slot_offset(stripe, expert)
            assert offset % PAGE_BYTES == 0, (stripe, expert, offset)

    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            fragments = []
            for stripe, out_dir in enumerate(out_dirs):
                offset = geometry.slot_offset(stripe, expert)
                length = geometry.fragment_bytes[stripe]
                fragments.append(_read_fragment(out_dir, layer, expert, offset, length))
            reassembled = b"".join(fragments)
            assert reassembled == rows[(layer, expert)], (layer, expert)

    # manifest.json is written identically (up to `index`) into every out_dir.
    manifests = [
        StripeManifest.from_json(open(os.path.join(out_dir, "manifest.json")).read())
        for out_dir in out_dirs
    ]
    assert [m.index for m in manifests] == [0, 1]
    for field in ("version", "source", "num_layers", "num_experts", "row_bytes", "tensor_order", "stripes", "row_sha256_sample"):
        assert getattr(manifests[0], field) == getattr(manifests[1], field), field

    assert len(manifests[0].row_sha256_sample) > 0


def test_layers_flag_limits_which_layers_are_written(tmp_path):
    source_dir = str(tmp_path / "source")
    os.makedirs(source_dir)
    write_fake_exl3(source_dir, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS)

    out_dirs = [str(tmp_path / "stripe-0"), str(tmp_path / "stripe-1")]
    write_stripe_set(source_dir, out_dirs, weights=[1.0, 1.0], layers=[1])

    for out_dir in out_dirs:
        assert not os.path.exists(os.path.join(out_dir, "layer-0.bin"))
        assert os.path.exists(os.path.join(out_dir, "layer-1.bin"))


def test_single_stripe_is_the_whole_row(tmp_path):
    source_dir = str(tmp_path / "source")
    os.makedirs(source_dir)
    rows = write_fake_exl3(source_dir, num_layers=1, num_experts=NUM_EXPERTS)

    out_dirs = [str(tmp_path / "stripe-0")]
    manifest = write_stripe_set(source_dir, out_dirs, weights=[1.0])

    assert len(manifest.stripes) == 1
    assert manifest.stripes[0].fragment_bytes == manifest.row_bytes

    from sglang.srt.layers.moe.exl3_stripe_layout import StripeGeometry

    geometry = StripeGeometry(row_bytes=manifest.row_bytes, weights=(1.0,))
    for expert in range(NUM_EXPERTS):
        offset = geometry.slot_offset(0, expert)
        assert offset % PAGE_BYTES == 0
        got = _read_fragment(out_dirs[0], 0, expert, offset, manifest.row_bytes)
        assert got == rows[(0, expert)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
