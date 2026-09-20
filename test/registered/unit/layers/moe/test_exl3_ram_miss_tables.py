"""The extent table ``exl3_ram_miss_tables`` builds for the native (in-graph) RAM-miss reader (CPU)."""

import os
import shutil

import pytest
import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYERS, EXPERTS = 2, 7  # write_fake_exl3 puts 3 experts per shard: 3 shards per layer


class _Checkpoint:
    def __init__(self, tmp_path, num_roots=2):
        self.source = tmp_path / "ckpt"
        self.source.mkdir()
        write_fake_exl3(str(self.source), num_layers=LAYERS, num_experts=EXPERTS)
        self.layout = build_exl3_expert_layout(str(self.source))
        fmt = Exl3ExpertFormat(self.layout, 0, direct=False)
        self.segments = fmt.segment_map()
        specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
        self.slabs = {
            layer: {
                name: allocate_host_slab(2, specs[name].row_shape, specs[name].dtype, register=False)
                for name in EXL3_STREAMED_NAMES
            }
            for layer in range(LAYERS)
        }
        self.roots = []
        for i in range(num_roots):
            root = tmp_path / f"drive_{i}" / "copy"
            shutil.copytree(self.source, root)
            self.roots.append(str(root))

    def tables(self, weights=None):
        if weights is None:
            return exl3_ram_miss_tables(self.layout, self.segments, self.slabs)
        return exl3_ram_miss_tables(
            self.layout,
            self.segments,
            self.slabs,
            roots=self.roots[: len(weights)],
            policy=StaticSplitPolicy(weights),
            source_root=str(self.source),
        )


def _records(ck):
    for row, layer in enumerate(sorted(ck.slabs)):
        for expert in range(EXPERTS):
            yield row, expert, ck.layout.records[(layer, expert)]


def test_no_roots_reproduces_the_single_file_reads(tmp_path):
    ck = _Checkpoint(tmp_path)
    t = ck.tables()
    assert t.parts == 1
    assert t.extents.shape == (LAYERS, EXPERTS, 1, 4)
    assert t.starts.shape == (LAYERS, EXPERTS)
    # The file index is the shard's first-seen index, the destination offset 0: today's `reads`.
    first_seen = {}
    for row, expert, record in _records(ck):
        index = first_seen.setdefault(record.path, len(first_seen))
        offset, length, start = record.aligned_read(PAGE_BYTES)
        assert t.extents[row, expert, 0].tolist() == [index, offset, length, 0]
        assert t.starts[row, expert].item() == start
    assert t.paths == list(first_seen)
    assert t.file_sizes.tolist() == [os.path.getsize(p) for p in t.paths]
    assert t.slot_bytes == max(r.aligned_read(PAGE_BYTES)[1] for _, _, r in _records(ck))


def test_two_equal_roots_split_every_row_in_two(tmp_path):
    ck = _Checkpoint(tmp_path)
    t = ck.tables((1.0, 1.0))
    assert t.parts == 2
    assert t.extents.shape == (LAYERS, EXPERTS, 2, 4)
    shard_of = {}
    for row, expert, record in _records(ck):
        shard = shard_of.setdefault(record.path, len(shard_of))
        offset, length, start = record.aligned_read(PAGE_BYTES)
        (f0, o0, n0, d0), (f1, o1, n1, d1) = t.extents[row, expert].tolist()
        assert n0 + n1 == length and n0 % PAGE_BYTES == 0 and n1 % PAGE_BYTES == 0
        assert (d0, d1) == (0, n0)
        assert (o0, o1) == (offset, offset + n0)
        assert (f0, f1) == (shard * 2, shard * 2 + 1)
        assert t.starts[row, expert].item() == start
    # Shard-major, root-minor: each source shard under root 0, then root 1.
    sources = list(shard_of)
    root0, root1 = ck.roots
    assert t.paths == [
        os.path.join(root, os.path.relpath(p, str(ck.source))) for p in sources for root in (root0, root1)
    ]
    # Every part of a shard is bounded by the source shard's size.
    assert t.file_sizes.tolist() == [os.path.getsize(p) for p in sources for _ in range(2)]
    assert len(t.paths) == t.file_sizes.numel()


def test_a_zero_weight_root_keeps_its_part_with_no_bytes(tmp_path):
    ck = _Checkpoint(tmp_path)
    t = ck.tables((1.0, 0.0))
    assert t.parts == 2
    for row, expert, record in _records(ck):
        offset, length, _ = record.aligned_read(PAGE_BYTES)
        (f0, o0, n0, d0), (f1, o1, n1, d1) = t.extents[row, expert].tolist()
        assert (n0, d0, o0) == (length, 0, offset)
        assert (n1, d1) == (0, length)  # present, empty, positioned at the end of the row
        assert f1 == f0 + 1


def test_the_native_split_is_the_eager_split(tmp_path):
    ck = _Checkpoint(tmp_path, num_roots=3)
    weights = (3.0, 1.0, 2.0)
    t = ck.tables(weights)
    policy = StaticSplitPolicy(weights)
    for row, expert, record in _records(ck):
        split = policy.plan(record.aligned_read(PAGE_BYTES)[1])
        assert t.extents[row, expert, :, 2].tolist() == list(split.part_bytes)
        assert t.extents[row, expert, :, 3].tolist() == list(split.starts)


def test_the_policy_is_planned_once_per_distinct_length(tmp_path):
    ck = _Checkpoint(tmp_path)
    calls = []

    class Counting(StaticSplitPolicy):
        def plan(self, length):
            calls.append(length)
            return super().plan(length)

    exl3_ram_miss_tables(
        ck.layout, ck.segments, ck.slabs, roots=ck.roots, policy=Counting((1.0, 1.0)), source_root=str(ck.source)
    )
    lengths = {r.aligned_read(PAGE_BYTES)[1] for _, _, r in _records(ck)}
    assert len(calls) <= len(lengths) + 1  # one probe of the part count, then one plan per length
    assert lengths <= set(calls)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(roots=("a", "b")), "policy"),
        (dict(roots=("a", "b"), policy=StaticSplitPolicy((1.0, 1.0))), "source root"),
        (dict(policy=StaticSplitPolicy((1.0, 1.0))), "needs mirror roots"),
        (dict(roots=("a", "b"), policy=StaticSplitPolicy((1.0,)), source_root="s"), "plans 1 parts for 2 roots"),
    ],
)
def test_inconsistent_mirror_arguments_are_refused(tmp_path, kwargs, message):
    ck = _Checkpoint(tmp_path)
    with pytest.raises(ValueError, match=message):
        exl3_ram_miss_tables(ck.layout, ck.segments, ck.slabs, **kwargs)


def test_a_source_root_that_does_not_contain_the_layout_is_refused(tmp_path):
    ck = _Checkpoint(tmp_path)
    with pytest.raises(ValueError, match="not under source root"):
        exl3_ram_miss_tables(
            ck.layout,
            ck.segments,
            ck.slabs,
            roots=ck.roots,
            policy=StaticSplitPolicy((1.0, 1.0)),
            source_root=str(tmp_path / "elsewhere"),
        )
