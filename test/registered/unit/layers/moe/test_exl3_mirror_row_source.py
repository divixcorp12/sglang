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
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES
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

    def __init__(self, tmp_path, num_roots=2, **fake_kwargs):
        self.source = tmp_path / "ckpt"
        self.source.mkdir()
        # 3 experts per shard: a layer's rows span shards, and some rows end a shard.
        write_fake_exl3(str(self.source), num_layers=2, num_experts=5, **fake_kwargs)
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


def _poison_root(root):
    """Flip every byte of every shard in ``root``'s copy; sizes are unchanged."""
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.endswith(".safetensors"):
            with open(path, "r+b") as f:
                data = f.read()
                f.seek(0)
                f.write(bytes(b ^ 0xFF for b in data))


def _pages(m, expert):
    return m.layout.records[(LAYER, expert)].aligned_read(PAGE_BYTES)[1] // PAGE_BYTES


def _expected_rows(m, weights, poisoned):
    """``{name: [each expert's row bytes]}`` worked out from the source files alone.

    Byte ``q`` of a row's page-aligned read is served by the root whose part of
    the split holds ``q``; a poisoned root serves every byte it is given
    flipped. The split comes from ``StaticSplitPolicy`` and the layout, and the
    bytes from the source shard, so nothing here goes through the source under
    test. ``poisoned`` may be None for the clean rows.
    """
    policy = StaticSplitPolicy(weights)
    segments = m.fmt.segment_map()
    row_bytes = {}
    for segment in segments:
        end = segment.dst_offset + segment.nbytes
        row_bytes[segment.name] = max(row_bytes.get(segment.name, 0), end)
    rows = {name: [] for name in row_bytes}
    for expert in range(5):
        record = m.layout.records[(LAYER, expert)]
        offset, length, start = record.aligned_read(PAGE_BYTES)
        with open(record.path, "rb") as f:
            f.seek(offset)
            superset = bytearray(f.read(length))
        superset.extend(bytes(length - len(superset)))  # past end of file: never used
        split = policy.plan(length)
        for root, (part_start, part_bytes) in enumerate(
            zip(split.starts, split.part_bytes)
        ):
            if root == poisoned:
                part = superset[part_start : part_start + part_bytes]
                superset[part_start : part_start + part_bytes] = bytes(
                    b ^ 0xFF for b in part
                )
        buffers = {name: bytearray(size) for name, size in row_bytes.items()}
        for segment in segments:
            at = start + segment.src_offset
            buffers[segment.name][
                segment.dst_offset : segment.dst_offset + segment.nbytes
            ] = superset[at : at + segment.nbytes]
        for name, buffer in buffers.items():
            rows[name].append(bytes(buffer))
    return rows


def _read_all(mirror, m):
    got = m.zeros(5)
    mirror.read(torch.arange(5), got)
    return {name: [_bytes_of(got[name][e]) for e in range(5)] for name in got}


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


@pytest.mark.parametrize(
    "weights, poisoned",
    [
        ((1.0, 1.0), 0),
        ((1.0, 1.0), 1),
        ((3.0, 1.0, 2.0), 0),
        ((3.0, 1.0, 2.0), 1),
        ((3.0, 1.0, 2.0), 2),
    ],
)
def test_each_root_serves_exactly_the_byte_ranges_the_split_gives_it(
    tmp_path, weights, poisoned
):
    # The copies are identical, so equal bytes cannot show which root served
    # them. Poison one copy: exactly the ranges the split assigns to that root
    # must come back flipped, and every other range must not.
    m = _Mirrored(tmp_path, num_roots=len(weights))
    assert {_pages(m, e) for e in range(5)} == {6}  # every row splits across roots
    _poison_root(m.roots[poisoned])
    got = _read_all(m.mirror(weights=weights), m)
    poisoned_rows = _expected_rows(m, weights, poisoned)
    clean_rows = _expected_rows(m, weights, None)
    assert got == poisoned_rows
    # Guard against a degenerate case: the poison reached the destinations, and
    # so did healthy bytes.
    flipped = sum(
        a != b
        for name in got
        for row_a, row_b in zip(got[name], clean_rows[name])
        for a, b in zip(row_a, row_b)
    )
    total = sum(len(row) for rows in got.values() for row in rows)
    assert 0 < flipped < total


def test_a_zero_weight_root_is_never_read(tmp_path):
    m = _Mirrored(tmp_path)
    want = m.zeros(5)
    m.single().read(torch.arange(5), want)
    _poison_root(m.roots[1])
    got = m.zeros(5)
    m.mirror(weights=(1.0, 0.0)).read(torch.arange(5), got)
    for name in want:
        assert _bytes_of(got[name]) == _bytes_of(want[name]), name


def test_rows_too_small_to_split_are_served_whole_by_the_heaviest_root(tmp_path):
    # 32-wide experts are one or two pages per row, so the split cannot give
    # every root a part: 1 page plans (1, 0, 0) and 2 pages plan (2, 0, 0)
    # under weights (3, 1, 2), leaving roots 1 and 2 idle.
    m = _Mirrored(tmp_path, num_roots=3, hidden=32, inter=32)
    pages = [_pages(m, e) for e in range(5)]
    assert min(pages) == 1 and max(pages) == 2
    weights = (3.0, 1.0, 2.0)
    policy = StaticSplitPolicy(weights)
    assert policy.plan(PAGE_BYTES).part_bytes == (PAGE_BYTES, 0, 0)
    assert policy.plan(2 * PAGE_BYTES).part_bytes == (2 * PAGE_BYTES, 0, 0)
    want = m.zeros(5)
    m.single().read(torch.arange(5), want)
    want_rows = {n: [_bytes_of(want[n][e]) for e in range(5)] for n in want}
    _poison_root(m.roots[1])
    _poison_root(m.roots[2])
    assert _read_all(m.mirror(weights=weights), m) == want_rows
    # Poisoning the heaviest root instead flips every row.
    _poison_root(m.roots[0])  # now all three are poisoned
    every_row_flipped = _read_all(m.mirror(weights=weights), m)
    assert every_row_flipped != want_rows
    assert all(
        a != b for n in want_rows for a, b in zip(every_row_flipped[n], want_rows[n])
    )


def test_a_split_that_leaves_a_one_page_row_whole_still_uses_the_second_root(tmp_path):
    # Weights (1, 1): a 1-page row plans (1, 0), all on root 0; a 2-page row
    # plans (1, 1), so only its second page can be poisoned.
    m = _Mirrored(tmp_path, num_roots=2, hidden=32, inter=32)
    pages = [_pages(m, e) for e in range(5)]
    assert min(pages) == 1 and max(pages) == 2
    _poison_root(m.roots[1])
    weights = (1.0, 1.0)
    assert _read_all(m.mirror(weights=weights), m) == _expected_rows(m, weights, 1)


def test_file_bytes_per_expert_is_what_the_row_reads_move(tmp_path):
    m = _Mirrored(tmp_path)
    mirror = m.mirror()

    def moved(expert):
        record = m.layout.records[(LAYER, expert)]
        offset, length, _ = record.aligned_read(PAGE_BYTES)
        return min(length, os.path.getsize(record.path) - offset)

    per_expert = [moved(e) for e in range(5)]
    assert mirror.file_bytes_per_expert == sum(per_expert) // 5
    # Whole pages around a row, and never less than the row itself.
    assert all(
        m.layout.row_bytes <= n <= m.layout.row_bytes + 2 * PAGE_BYTES
        for n in per_expert
    )
    assert mirror.file_bytes_per_expert == m.single().file_bytes_per_expert
    stats = mirror.read(torch.tensor([3, 4]), m.zeros(2))
    assert stats.file_bytes == per_expert[3] + per_expert[4]
    assert stats.rows == 2


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


def _shard_of(m, expert=0):
    record = m.layout.records[(LAYER, expert)]
    return record.path, os.path.relpath(record.path, str(m.source))


def test_a_root_missing_a_shard_names_the_root_and_the_path(tmp_path):
    m = _Mirrored(tmp_path)
    path, relative = _shard_of(m)
    os.remove(os.path.join(m.roots[1], relative))
    with pytest.raises(FileNotFoundError) as caught:
        m.mirror()
    message = str(caught.value)
    assert m.roots[1] in message
    assert os.path.join(m.roots[1], relative) in message
    assert m.roots[0] not in message


def test_a_truncated_mirror_is_refused_at_construction_with_both_sizes(tmp_path):
    m = _Mirrored(tmp_path)
    path, relative = _shard_of(m)
    source_bytes = os.path.getsize(path)
    with open(os.path.join(m.roots[1], relative), "r+b") as f:
        f.truncate(source_bytes - 4096)
    with pytest.raises(RuntimeError) as caught:
        m.mirror()
    message = str(caught.value)
    assert m.roots[1] in message
    assert os.path.join(m.roots[1], relative) in message
    assert str(source_bytes) in message and str(source_bytes - 4096) in message
    assert m.roots[0] not in message


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unreadable_root_is_not_reported_as_missing(tmp_path):
    m = _Mirrored(tmp_path)
    _path, relative = _shard_of(m)
    directory = os.path.dirname(os.path.join(m.roots[1], relative))
    os.chmod(directory, 0)
    try:
        with pytest.raises(PermissionError) as caught:
            m.mirror()
    finally:
        os.chmod(directory, 0o755)
    message = str(caught.value)
    assert m.roots[1] in message and "does not exist" not in message


def test_a_mirror_truncated_after_construction_still_fails_at_the_read(tmp_path):
    # The reader's own first-open size check stays in force behind the
    # construction-time one.
    m = _Mirrored(tmp_path)
    mirror = m.mirror()
    path, relative = _shard_of(m)
    with open(os.path.join(m.roots[1], relative), "r+b") as f:
        f.truncate(os.path.getsize(path) - 4096)
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


def test_for_mirrored_layer_shares_the_reader_and_the_bounce_ring(tmp_path):
    m = _Mirrored(tmp_path)
    policy = StaticSplitPolicy((1.0, 1.0))
    kwargs = dict(direct=False, roots=m.roots, policy=policy, source_root=str(m.source))
    make = Exl3MirrorRowSource.for_mirrored_layer
    a = make(m.layout, 0, m.fmt.segment_map(), **kwargs)
    b = make(m.layout, 1, m.fmt.segment_map(), bounce_rows=3, **kwargs)
    assert a.reader is b.reader
    assert a.reader is shared_row_reader(m.layout, False, source_root=str(m.source))
    assert a.reader.source_root == str(m.source)
    assert b.preferred_batch_rows == 3 and a.preferred_batch_rows != 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
