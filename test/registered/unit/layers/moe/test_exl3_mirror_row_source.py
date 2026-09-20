"""EXL3 expert rows read from K byte-identical mirror copies of a checkpoint."""

import os
import shutil

import pytest
import torch

from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_mirror_row_source import Exl3MirrorRowSource
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_shard_row_source import (
    Exl3ShardRowSource,
    shared_row_reader,
)
from sglang.srt.layers.moe.expert_row_source import ExpertRowSource, HostSlotLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYER = 1


class _Mirrored:
    """A fake checkpoint under ``ckpt`` plus identical copies in two roots.

    The roots have different absolute prefixes from each other and from the
    source, so a record's path can only reach them through its path relative
    to the source root.
    """

    def __init__(self, tmp_path, num_roots=2):
        self.source = tmp_path / "ckpt"
        self.source.mkdir()
        # 3 experts per shard: a layer's rows span shards, and some rows end a shard.
        write_fake_exl3(str(self.source), num_layers=2, num_experts=5)
        self.layout = build_exl3_expert_layout(str(self.source))
        self.fmt = Exl3ExpertFormat(self.layout, LAYER, direct=False)
        self.roots = []
        for i in range(num_roots):
            root = tmp_path / f"drive_{i}" / "copy"
            shutil.copytree(self.source, root)
            self.roots.append(str(root))

    def single(self, bounce_rows=2):
        reader = Exl3RowReader(self.layout, direct=False)
        return Exl3ShardRowSource(
            reader, LAYER, self.fmt.segment_map(), bounce_rows=bounce_rows
        )

    def mirror(self, weights=(1.0, 1.0), roots=None, **reader_kwargs):
        reader_kwargs.setdefault("source_root", str(self.source))
        reader = Exl3RowReader(self.layout, direct=False, **reader_kwargs)
        return Exl3MirrorRowSource(
            reader,
            LAYER,
            self.fmt.segment_map(),
            roots=self.roots if roots is None else roots,
            policy=StaticSplitPolicy(weights),
            bounce_rows=2,
        )

    def zeros(self, rows):
        return {
            spec.name: torch.zeros((rows,) + spec.row_shape, dtype=spec.dtype)
            for spec in self.fmt.tensor_specs(None)
        }


def _bytes_of(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).numpy())


def _opened_roots(source):
    return {root for root, _path in source.reader._files if root is not None}


def test_reads_the_same_bytes_as_the_single_root_source(tmp_path):
    m = _Mirrored(tmp_path)
    single, mirror = m.single(), m.mirror()
    # More rows than the bounce ring holds, so the read chunks; every expert.
    experts, slots = [4, 0, 3, 2, 1], [4, 0, 2, 1, 3]
    want, got = m.zeros(5), m.zeros(5)
    single.read(torch.tensor(experts), want, torch.tensor(slots))
    mirror.read(torch.tensor(experts), got, torch.tensor(slots))
    for name in want:
        assert _bytes_of(got[name]) == _bytes_of(want[name]), name
        assert any(got[name].view(torch.uint8).flatten().tolist()), name  # not all zero


def test_the_split_really_uses_both_roots(tmp_path):
    m = _Mirrored(tmp_path)
    mirror = m.mirror(weights=(1.0, 1.0))
    # Each row's aligned read spans several pages, so a 1:1 split gives both
    # roots a non-empty part; a one-page row could not.
    assert mirror.reader.layout.records[(LAYER, 0)].aligned_read(4096)[1] >= 2 * 4096
    mirror.read(torch.tensor([0, 1]), m.zeros(2))
    assert _opened_roots(mirror) == set(m.roots)


def test_a_zero_weight_drops_that_root_and_the_bytes_stay_identical(tmp_path):
    m = _Mirrored(tmp_path)
    want, got = m.zeros(5), m.zeros(5)
    m.single().read(torch.arange(5), want)
    mirror = m.mirror(weights=(1.0, 0.0))
    mirror.read(torch.arange(5), got)
    assert _opened_roots(mirror) == {m.roots[0]}
    for name in want:
        assert _bytes_of(got[name]) == _bytes_of(want[name]), name


def test_an_uneven_split_over_three_roots_is_identical_too(tmp_path):
    m = _Mirrored(tmp_path, num_roots=3)
    want, got = m.zeros(5), m.zeros(5)
    m.single().read(torch.arange(5), want)
    mirror = m.mirror(weights=(3.0, 1.0, 2.0))
    mirror.read(torch.arange(5), got)
    assert _opened_roots(mirror) == set(m.roots)
    for name in want:
        assert _bytes_of(got[name]) == _bytes_of(want[name]), name


def test_file_bytes_per_expert_matches_the_single_root_source(tmp_path):
    m = _Mirrored(tmp_path)
    assert m.mirror().file_bytes_per_expert == m.single().file_bytes_per_expert
    stats_single = m.single().read(torch.tensor([3, 4]), m.zeros(2))
    stats_mirror = m.mirror().read(torch.tensor([3, 4]), m.zeros(2))
    assert stats_mirror.file_bytes == stats_single.file_bytes
    assert stats_mirror.split_bytes == stats_single.split_bytes
    assert stats_mirror.rows == 2


def test_has_the_public_surface_of_the_shard_source(tmp_path):
    m = _Mirrored(tmp_path)
    single, mirror = m.single(bounce_rows=2), m.mirror()
    assert isinstance(mirror, ExpertRowSource)
    assert mirror.host_layouts == frozenset({HostSlotLayout.PER_NAME})
    assert mirror.requires_page_aligned_destinations is False
    assert mirror.covers("w13_trellis") and not mirror.covers("w13_mul1")
    assert mirror.register_destinations([torch.zeros(4)]) == 0
    assert mirror.close() is None
    assert mirror.preferred_batch_rows == single.preferred_batch_rows
    assert mirror.slot_bytes == single.slot_bytes
    assert mirror.num_experts == 5
    assert mirror.bounce.shape == single.bounce.shape
    assert mirror.bounce.data_ptr() % 4096 == 0
    ticket = mirror.submit(torch.tensor([3]), m.zeros(1))
    assert ticket.done() and ticket.wait().rows == 1


def test_rejects_the_same_bad_requests_as_the_shard_source(tmp_path):
    m = _Mirrored(tmp_path)
    mirror = m.mirror()
    with pytest.raises(ValueError, match="does not cover"):
        mirror.read(
            torch.tensor([0]), {"w13_mul1": torch.zeros(1, 2, dtype=torch.int32)}
        )
    with pytest.raises(ValueError, match="outside"):
        mirror.read(torch.tensor([5]), m.zeros(1))
    with pytest.raises(ValueError, match="match in length"):
        mirror.read(torch.tensor([0, 1]), m.zeros(2), torch.tensor([0]))
    assert mirror.read(torch.tensor([], dtype=torch.long), m.zeros(1)).rows == 0


def test_a_root_missing_a_shard_names_the_root_and_the_path(tmp_path):
    m = _Mirrored(tmp_path)
    record = m.layout.records[(LAYER, 0)]
    relative = os.path.relpath(record.path, str(m.source))
    os.remove(os.path.join(m.roots[1], relative))
    with pytest.raises(Exception) as caught:
        m.mirror()
    message = str(caught.value)
    assert m.roots[1] in message
    assert relative in message
    assert m.roots[0] not in message


def test_a_truncated_mirror_still_fails_loudly(tmp_path):
    # The reader's size check is deliberate; the source must not defeat it.
    m = _Mirrored(tmp_path)
    record = m.layout.records[(LAYER, 0)]
    relative = os.path.relpath(record.path, str(m.source))
    with open(os.path.join(m.roots[1], relative), "r+b") as f:
        f.truncate(os.path.getsize(record.path) - 4096)
    mirror = m.mirror()
    with pytest.raises(RuntimeError, match="incomplete or stale"):
        mirror.read(torch.tensor([0]), m.zeros(1))


def test_a_reader_without_a_source_root_is_refused_at_construction(tmp_path):
    m = _Mirrored(tmp_path)
    reader = Exl3RowReader(m.layout, direct=False)
    with pytest.raises(ValueError, match="source_root"):
        Exl3MirrorRowSource(
            reader,
            LAYER,
            m.fmt.segment_map(),
            roots=m.roots,
            policy=StaticSplitPolicy((1.0, 1.0)),
        )


def test_the_policy_must_plan_one_part_per_root(tmp_path):
    m = _Mirrored(tmp_path)
    with pytest.raises(ValueError, match="parts"):
        m.mirror(weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="root"):
        m.mirror(roots=[], weights=(1.0,))


def test_for_layer_shares_the_reader_and_the_bounce_ring(tmp_path):
    m = _Mirrored(tmp_path)
    policy = StaticSplitPolicy((1.0, 1.0))
    kwargs = dict(direct=False, roots=m.roots, policy=policy, source_root=str(m.source))
    a = Exl3MirrorRowSource.for_layer(m.layout, 0, m.fmt.segment_map(), **kwargs)
    b = Exl3MirrorRowSource.for_layer(m.layout, 1, m.fmt.segment_map(), **kwargs)
    assert a.reader is b.reader
    assert a.reader is shared_row_reader(m.layout, False, source_root=str(m.source))
    assert a.bounce.data_ptr() == b.bounce.data_ptr()
    assert a.reader.source_root == str(m.source)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
