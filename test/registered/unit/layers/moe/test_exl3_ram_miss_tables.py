"""The extent table ``exl3_ram_miss_tables`` builds for the native (in-graph) RAM-miss reader (CPU).

Row images are the only tables (plan 2026-09-29-hotpath-zero-overhead D4): file ``row * parts + p`` is root ``p``'s
image file of streamed row ``row``, and expert ``e``'s image is split across the roots as the policy splits its
page-rounded ``row_stride``, each part clipped at ``image_bytes``.
"""

import os

import pytest

from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYERS, EXPERTS = 2, 7


class _Checkpoint:
    def __init__(self, tmp_path, weights=None):
        self.s = ram_miss_setup(tmp_path, capacity=2, layers=LAYERS, experts=EXPERTS, mirror_weights=weights)
        self.source = str(tmp_path)
        self.weights = (1.0,) if weights is None else tuple(weights)
        self.images = ri.open_row_images(
            self.s.roots, self.s.layout, self.s.fmt.segment_map(), self.source, range(LAYERS)
        )
        self.image = self.images.layout

    @property
    def tables(self):
        return self.s.tables

    def build(self, **kwargs):
        return exl3_ram_miss_tables(self.s.layout, self.s.fmt.segment_map(), self.s.slabs, **kwargs)

    def clipped(self, policy):
        split = policy.plan(self.image.row_stride)
        return [
            (min(start, self.image.image_bytes), min(start + nbytes, self.image.image_bytes))
            for start, nbytes in zip(split.starts, split.part_bytes)
        ]


def test_one_root_reads_each_expert_image_whole(tmp_path):
    """Converted from the shard tables' test_no_roots_reproduces_the_single_file_reads: one part per row, reading the
    expert's whole image from its own offset in the layer's file, and nothing past it."""
    ck = _Checkpoint(tmp_path)
    t, image = ck.tables, ck.image
    assert t.row_images and t.parts == 1
    assert t.extents.shape == (LAYERS, EXPERTS, 1, 4) and t.starts.shape == (LAYERS, EXPERTS)
    for row in range(LAYERS):
        for expert in range(EXPERTS):
            assert t.extents[row, expert, 0].tolist() == [row, expert * image.row_stride, image.image_bytes, 0]
    assert not t.starts.any()
    assert t.paths == [ck.images.paths[layer][0] for layer in range(LAYERS)]
    assert t.file_sizes.tolist() == [os.path.getsize(p) for p in t.paths] == [EXPERTS * image.row_stride] * LAYERS
    assert t.slot_bytes == image.image_bytes


def test_two_equal_roots_split_every_row_in_two(tmp_path):
    ck = _Checkpoint(tmp_path, (1.0, 1.0))
    t, image = ck.tables, ck.image
    assert t.parts == 2 and t.extents.shape == (LAYERS, EXPERTS, 2, 4)
    (lo0, hi0), (lo1, hi1) = ck.clipped(StaticSplitPolicy((1.0, 1.0)))
    assert lo0 == 0 and hi0 == lo1 and hi1 == image.image_bytes  # the parts tile the image, and no more
    for row in range(LAYERS):
        for expert in range(EXPERTS):
            base = expert * image.row_stride
            assert t.extents[row, expert].tolist() == [
                [row * 2, base + lo0, hi0 - lo0, lo0],
                [row * 2 + 1, base + lo1, hi1 - lo1, lo1],
            ]
    # Row-major, root-minor: each layer's image file under root 0, then root 1.
    assert t.paths == [ck.images.paths[layer][part] for layer in range(LAYERS) for part in range(2)]
    assert all(path.startswith(ck.s.roots[i % 2]) for i, path in enumerate(t.paths))
    assert len(t.paths) == t.file_sizes.numel()


def test_a_zero_weight_root_keeps_its_part_with_no_bytes(tmp_path):
    ck = _Checkpoint(tmp_path, (1.0, 0.0))
    t, image = ck.tables, ck.image
    assert t.parts == 2
    for row in range(LAYERS):
        for expert in range(EXPERTS):
            (f0, o0, n0, d0), (f1, o1, n1, d1) = t.extents[row, expert].tolist()
            assert (n0, d0, o0) == (image.image_bytes, 0, expert * image.row_stride)
            assert (n1, d1) == (0, image.image_bytes)  # present, empty, positioned at the end of the image
            assert f1 == f0 + 1


def test_the_native_split_is_the_eager_split(tmp_path):
    weights = (3.0, 1.0, 2.0)
    ck = _Checkpoint(tmp_path, weights)
    t = ck.tables
    clipped = ck.clipped(StaticSplitPolicy(weights))
    for row in range(LAYERS):
        for expert in range(EXPERTS):
            assert t.extents[row, expert, :, 2].tolist() == [hi - lo for lo, hi in clipped]
            assert t.extents[row, expert, :, 3].tolist() == [lo for lo, _ in clipped]


def test_the_policy_is_planned_once(tmp_path):
    """Every expert's image has the same stride, so the split is planned once for all of them."""
    ck = _Checkpoint(tmp_path, (1.0, 1.0))
    calls = []

    class Counting(StaticSplitPolicy):
        def plan(self, length):
            calls.append(length)
            return super().plan(length)

    ck.build(roots=ck.s.roots, policy=Counting((1.0, 1.0)), row_images=ck.images)
    assert calls == [ck.image.row_stride]


def test_tables_without_row_images_are_refused_naming_the_converter(tmp_path):
    """Shard tables would need the deleted bounce-and-pack path: refused, naming how to build images."""
    ck = _Checkpoint(tmp_path)
    with pytest.raises(ValueError, match="build_row_images"):
        ck.build()
    with pytest.raises(ValueError, match="row images only"):
        ck.build(roots=ck.s.roots, policy=StaticSplitPolicy((1.0,)), source_root=ck.source)


def test_inconsistent_mirror_arguments_are_refused(tmp_path):
    ck = _Checkpoint(tmp_path, (1.0, 1.0))
    with pytest.raises(ValueError, match="row images are on"):
        ck.build(roots=ck.s.roots[::-1], policy=StaticSplitPolicy((1.0, 1.0)), row_images=ck.images)
    with pytest.raises(ValueError, match="need the split policy"):
        ck.build(roots=ck.s.roots, row_images=ck.images)
    with pytest.raises(ValueError, match="plans 1 parts for 2 row-image roots"):
        ck.build(roots=ck.s.roots, policy=StaticSplitPolicy((1.0,)), row_images=ck.images)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
