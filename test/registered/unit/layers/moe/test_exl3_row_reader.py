"""One aligned read per EXL3 expert, straight from the shards."""

import ctypes
import os
import shutil

import pytest
import torch

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4096


def _buffers(count, nbytes):
    storage = torch.zeros(count * nbytes + PAGE, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE
    buf = storage[start : start + count * nbytes].view(count, nbytes)
    return storage, buf


def test_reads_rows_at_their_offsets(tmp_path):
    rows = write_fake_exl3(str(tmp_path), num_layers=2, num_experts=4)
    layout = build_exl3_expert_layout(str(tmp_path))
    reader = Exl3RowReader(layout, direct=False)
    assert reader.buffer_bytes % PAGE == 0
    assert reader.buffer_bytes >= layout.row_bytes

    keys = [(1, 3), (0, 0), (1, 1)]
    _keep, buf = _buffers(len(keys), reader.buffer_bytes)
    starts = reader.read(keys, [buf[i].data_ptr() for i in range(len(keys))])

    for i, (key, start) in enumerate(zip(keys, starts)):
        got = bytes(buf[i, start : start + layout.row_bytes].numpy())
        assert got == rows[key], key


def test_last_row_of_a_shard_reads_up_to_end_of_file(tmp_path):
    rows = write_fake_exl3(
        str(tmp_path), num_layers=1, num_experts=3, experts_per_shard=3
    )
    layout = build_exl3_expert_layout(str(tmp_path))
    reader = Exl3RowReader(layout, direct=False)
    _keep, buf = _buffers(1, reader.buffer_bytes)
    (start,) = reader.read([(0, 2)], [buf[0].data_ptr()])
    assert bytes(buf[0, start : start + layout.row_bytes].numpy()) == rows[(0, 2)]


def test_rejects_unaligned_destination(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=1)
    reader = Exl3RowReader(build_exl3_expert_layout(str(tmp_path)), direct=False)
    _keep, buf = _buffers(1, reader.buffer_bytes)
    with pytest.raises(ValueError, match="page-aligned"):
        reader.read([(0, 0)], [buf[0].data_ptr() + 16])


def test_empty_read_is_a_no_op(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=1)
    reader = Exl3RowReader(build_exl3_expert_layout(str(tmp_path)), direct=False)
    assert reader.read([], []) == []


class _RecordingReader:
    """A stand-in ``UringFileReader`` over real (tiny) files in ``tmp_path``.

    Like the native reader: ``open`` dedupes by ``realpath`` and raises
    ``RuntimeError`` naming the path when the file cannot be opened; ``read``
    copies the requested bytes to the destination addresses and returns the
    bytes actually available (short at end of file). Every call is recorded.
    """

    def __init__(self, short_by=0, fail_in=None):
        self.short_by = short_by
        self.fail_in = fail_in
        self.opened = []  # every open() call's path, in order
        self.calls = []  # every read() call's extents
        self._ids = {}
        self._paths = []

    def open(self, path, *, direct):
        self.opened.append(str(path))
        real = os.path.realpath(path)
        if real not in self._ids:
            if not os.path.isfile(real):
                raise RuntimeError(f"open_file failed for {real}: No such file")
            self._ids[real] = len(self._paths)
            self._paths.append(real)
        return self._ids[real]

    def file_size(self, file_id):
        return os.path.getsize(self._paths[file_id])

    def read(self, file_ids, offsets, destinations, lengths):
        entries = list(
            zip(
                file_ids.tolist(),
                offsets.tolist(),
                destinations.tolist(),
                lengths.tolist(),
            )
        )
        self.calls.append(entries)
        total = 0
        for file_id, offset, destination, length in entries:
            if self.fail_in and self.fail_in in self._paths[file_id]:
                raise OSError(f"read failed on {self._paths[file_id]}")
            with open(self._paths[file_id], "rb") as f:
                f.seek(offset)
                data = f.read(length)
            ctypes.memmove(destination, data, len(data))
            total += len(data)
        return total - self.short_by


class _Mirrored:
    """A fake checkpoint under ``ckpt`` plus identical copies under mirror_0..K-1."""

    def __init__(self, tmp_path, num_roots=2, num_experts=3, experts_per_shard=3):
        self.source = tmp_path / "ckpt"
        self.source.mkdir()
        self.rows = write_fake_exl3(
            str(self.source),
            num_layers=1,
            num_experts=num_experts,
            experts_per_shard=experts_per_shard,
        )
        self.layout = build_exl3_expert_layout(str(self.source))
        self.roots = []
        for i in range(num_roots):
            root = tmp_path / f"mirror_{i}"
            shutil.copytree(self.source, root)
            self.roots.append(str(root))
        self.shard = os.path.basename(self.layout.records[(0, 0)].path)

    def read(self, keys, weights, reader=None, **reader_kwargs):
        reader = reader if reader is not None else _RecordingReader()
        reader_kwargs.setdefault("source_root", str(self.source))
        row_reader = Exl3RowReader(self.layout, reader, direct=False, **reader_kwargs)
        _keep, buf = _buffers(len(keys), row_reader.buffer_bytes)
        starts = row_reader.read_split(
            keys,
            [buf[i].data_ptr() for i in range(len(keys))],
            roots=self.roots[: len(weights)],
            policy=StaticSplitPolicy(weights),
        )
        return reader, buf, starts

    def row(self, buf, starts, i):
        return bytes(buf[i, starts[i] : starts[i] + self.layout.row_bytes].numpy())


def test_read_split_is_one_submit_with_one_read_per_nonempty_part(tmp_path):
    m = _Mirrored(tmp_path)
    keys = [(0, 0), (0, 1), (0, 2)]
    reader, buf, starts = m.read(keys, (1.0, 1.0))

    assert len(reader.calls) == 1
    assert len(reader.calls[0]) == 6
    file_of = {
        os.path.basename(os.path.dirname(p)): i for i, p in enumerate(reader._paths)
    }
    for i, key in enumerate(keys):
        offset, length, _start = m.layout.records[key].aligned_read(PAGE)
        split = StaticSplitPolicy((1.0, 1.0)).plan(length)
        for r in range(2):
            assert reader.calls[0][2 * i + r] == (
                file_of[f"mirror_{r}"],
                offset + split.starts[r],
                buf[i].data_ptr() + split.starts[r],
                split.part_bytes[r],
            )
    for i, key in enumerate(keys):
        assert m.row(buf, starts, i) == m.rows[key]


def test_read_split_opens_each_roots_copy_once(tmp_path):
    m = _Mirrored(tmp_path)
    reader = _RecordingReader()
    row_reader = Exl3RowReader(
        m.layout, reader, direct=False, source_root=str(m.source)
    )
    _keep, buf = _buffers(2, row_reader.buffer_bytes)
    for key in [(0, 0), (0, 1)]:
        row_reader.read_split(
            [key],
            [buf[0].data_ptr()],
            roots=m.roots,
            policy=StaticSplitPolicy((1.0, 1.0)),
        )
    for root in m.roots:
        assert reader.opened.count(f"{root}/{m.shard}") == 1


def test_a_zero_part_issues_no_read_and_never_opens_that_root(tmp_path):
    m = _Mirrored(tmp_path)
    keys = [(0, 0), (0, 1)]
    reader, buf, starts = m.read(keys, (1.0, 0.0))
    assert len(reader.calls) == 1
    assert len(reader.calls[0]) == 2  # one per row, none for the dropped root
    assert not any("mirror_1" in path for path in reader.opened)
    for i, key in enumerate(keys):
        assert m.row(buf, starts, i) == m.rows[key]


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (3.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),  # the straddling part is the middle one
        (2.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
    ],
)
def test_a_split_row_that_overruns_end_of_file_is_not_a_short_read(tmp_path, weights):
    m = _Mirrored(tmp_path, num_roots=len(weights))
    key = (0, 2)  # last row of its shard: its aligned superset runs past EOF
    offset, length, _start = m.layout.records[key].aligned_read(PAGE)
    file_bytes = os.path.getsize(m.layout.records[key].path)
    assert offset + length > file_bytes, "fixture must overrun end of file"

    reader, buf, starts = m.read([key], weights)
    (call,) = reader.calls
    assert len(call) == sum(1 for w in weights if w > 0)
    # Exactly one part reaches past EOF (the last non-empty one) and the reader
    # serves it short; the batch must still count as fully read.
    over = [o + n - file_bytes for _f, o, _d, n in call if o + n > file_bytes]
    assert len(over) == 1 and 0 < over[0] < PAGE
    assert m.row(buf, starts, 0) == m.rows[key]


def test_a_batch_across_shards_opens_every_shard_once_per_root(tmp_path):
    m = _Mirrored(tmp_path, num_experts=4, experts_per_shard=2)
    keys = [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert len({m.layout.records[key].path for key in keys}) == 2
    reader, buf, starts = m.read(keys, (1.0, 1.0))
    assert len(reader.calls) == 1 and len(reader.calls[0]) == 8
    mirror_opens = [p for p in reader.opened if "/mirror_" in p]
    assert len(mirror_opens) == 4 and len(set(mirror_opens)) == 4
    for i, key in enumerate(keys):
        assert m.row(buf, starts, i) == m.rows[key]


def test_roots_that_are_the_same_file_share_one_file_id(tmp_path):
    m = _Mirrored(tmp_path)
    alias = tmp_path / "alias"
    os.symlink(m.roots[0], alias)
    m.roots = [m.roots[0], str(alias)]
    reader, buf, starts = m.read([(0, 1)], (1.0, 1.0))
    assert len({entry[0] for entry in reader.calls[0]}) == 1
    assert m.row(buf, starts, 0) == m.rows[(0, 1)]


def test_a_truncated_mirror_fails_loudly_instead_of_returning_a_short_row(tmp_path):
    m = _Mirrored(tmp_path)
    key = (0, 0)
    offset, length, _start = m.layout.records[key].aligned_read(PAGE)
    truncated = os.path.join(m.roots[1], m.shard)
    # Cut the second copy inside the range that root serves for this row.
    with open(truncated, "r+b") as f:
        f.truncate(offset + length // 2 + 5000)
    source_bytes = os.path.getsize(m.layout.records[key].path)
    with pytest.raises(RuntimeError) as raised:
        m.read([key], (1.0, 1.0))
    message = str(raised.value)
    assert truncated in message and m.layout.records[key].path in message
    assert str(source_bytes) in message and str(offset + length // 2 + 5000) in message


def test_a_mirror_that_is_larger_than_the_source_is_also_rejected(tmp_path):
    m = _Mirrored(tmp_path)
    with open(os.path.join(m.roots[0], m.shard), "ab") as f:
        f.write(b"\0" * 10)
    with pytest.raises(RuntimeError, match="size"):
        m.read([(0, 0)], (1.0, 1.0))


def test_a_short_read_fails_the_batch_like_the_single_root_path(tmp_path):
    m = _Mirrored(tmp_path)
    with pytest.raises(RuntimeError, match="ended early"):
        m.read([(0, 0), (0, 1)], (1.0, 1.0), reader=_RecordingReader(short_by=PAGE))


def test_a_per_root_read_error_propagates(tmp_path):
    m = _Mirrored(tmp_path)
    reader = _RecordingReader(fail_in="mirror_1")  # the second root's copy
    with pytest.raises(OSError, match="read failed"):
        m.read([(0, 0)], (1.0, 1.0), reader=reader)


def test_a_root_missing_the_file_names_the_root_and_path(tmp_path):
    m = _Mirrored(tmp_path)
    missing = os.path.join(m.roots[1], m.shard)
    os.remove(missing)
    # The native reader raises RuntimeError with the path in the message.
    with pytest.raises(RuntimeError, match=missing):
        m.read([(0, 0)], (1.0, 1.0))


def test_roots_may_have_different_absolute_prefixes(tmp_path):
    m = _Mirrored(tmp_path)
    other = tmp_path / "elsewhere" / "deeper"
    other.parent.mkdir()
    shutil.move(m.roots[1], other)
    m.roots[1] = str(other)
    reader, buf, starts = m.read([(0, 1)], (1.0, 1.0))
    assert m.row(buf, starts, 0) == m.rows[(0, 1)]


def test_a_checkpoint_shard_in_a_subdirectory_resolves_under_each_root(tmp_path):
    # Every shard sits in one subfolder; only an explicit source_root can tell
    # the checkpoint root from that subfolder.
    source = tmp_path / "ckpt"
    (source / "sub").mkdir(parents=True)
    rows = write_fake_exl3(str(source / "sub"), num_layers=1, num_experts=3)
    layout = build_exl3_expert_layout(str(source / "sub"))
    roots = []
    for name in ("mirror_0", "mirror_1"):
        shutil.copytree(source, tmp_path / name)
        roots.append(str(tmp_path / name))
    # A same-named decoy one level up would be picked by a wrong root.
    for name in os.listdir(source / "sub"):
        if name.endswith(".safetensors"):
            for root in roots:
                shutil.copy(source / "sub" / name, os.path.join(root, name))
    reader = _RecordingReader()
    row_reader = Exl3RowReader(layout, reader, direct=False, source_root=str(source))
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    (start,) = row_reader.read_split(
        [(0, 1)],
        [buf[0].data_ptr()],
        roots=roots,
        policy=StaticSplitPolicy((1.0, 1.0)),
    )
    assert bytes(buf[0, start : start + layout.row_bytes].numpy()) == rows[(0, 1)]
    for root in roots:
        assert f"{root}/sub/model-00001.safetensors" in reader.opened
        assert f"{root}/model-00001.safetensors" not in reader.opened


def test_read_split_requires_an_explicit_source_root(tmp_path):
    m = _Mirrored(tmp_path)
    reader = _RecordingReader()
    row_reader = Exl3RowReader(m.layout, reader, direct=False)
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    with pytest.raises(ValueError, match="source_root"):
        row_reader.read_split(
            [(0, 0)],
            [buf[0].data_ptr()],
            roots=m.roots,
            policy=StaticSplitPolicy((1.0, 1.0)),
        )
    assert reader.calls == []


def test_a_record_outside_source_root_is_rejected(tmp_path):
    m = _Mirrored(tmp_path)
    with pytest.raises(ValueError, match="not under source root"):
        m.read([(0, 0)], (1.0, 1.0), source_root=str(tmp_path / "unrelated"))


def test_read_split_rejects_unaligned_destination(tmp_path):
    m = _Mirrored(tmp_path)
    row_reader = Exl3RowReader(
        m.layout, _RecordingReader(), direct=False, source_root=str(m.source)
    )
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    with pytest.raises(ValueError, match="page-aligned"):
        row_reader.read_split(
            [(0, 0)],
            [buf[0].data_ptr() + 16],
            roots=m.roots,
            policy=StaticSplitPolicy((1.0, 1.0)),
        )


def test_read_split_rejects_a_policy_for_a_different_root_count(tmp_path):
    m = _Mirrored(tmp_path)
    row_reader = Exl3RowReader(
        m.layout, _RecordingReader(), direct=False, source_root=str(m.source)
    )
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    with pytest.raises(ValueError, match="roots"):
        row_reader.read_split(
            [(0, 0)],
            [buf[0].data_ptr()],
            roots=m.roots,
            policy=StaticSplitPolicy((1.0, 1.0, 1.0)),
        )


def test_read_split_empty_is_a_no_op(tmp_path):
    m = _Mirrored(tmp_path)
    reader = _RecordingReader()
    row_reader = Exl3RowReader(m.layout, reader, direct=False)
    policy = StaticSplitPolicy((1.0,))
    assert row_reader.read_split([], [], roots=m.roots[:1], policy=policy) == []
    assert reader.calls == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
