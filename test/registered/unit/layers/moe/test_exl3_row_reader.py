"""One aligned read per EXL3 expert, straight from the shards."""

import ctypes
import os

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
    """A stand-in ``UringFileReader`` that serves virtual files from memory.

    ``files`` maps a path to its bytes. ``read`` records every call, copies the
    requested bytes to the destination addresses and returns the bytes actually
    available (short at end of file), as the real reader does.
    """

    def __init__(self, files, short_by=0, fail_file=None):
        self.files = files
        self.short_by = short_by
        self.fail_file = fail_file
        self.opened = []
        self.calls = []
        self._paths = []

    def open(self, path, *, direct):
        if path not in self.files:
            raise FileNotFoundError(path)
        self.opened.append(path)
        self._paths.append(path)
        return len(self._paths) - 1

    def file_size(self, file_id):
        return len(self.files[self._paths[file_id]])

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
            if file_id == self.fail_file:
                raise OSError(f"read failed on file {file_id}")
            data = self.files[self._paths[file_id]][offset : offset + length]
            ctypes.memmove(destination, data, len(data))
            total += len(data)
        return total - self.short_by


ROOTS = ("/mirror_a", "/mirror_b")
SHARD = "model-00001.safetensors"


def _mirrored(tmp_path, roots=ROOTS):
    """A fake checkpoint plus one virtual identical copy of it under each root."""
    rows = write_fake_exl3(
        str(tmp_path), num_layers=1, num_experts=3, experts_per_shard=3
    )
    layout = build_exl3_expert_layout(str(tmp_path))
    files = {}
    for name in os.listdir(tmp_path):
        if name.endswith(".safetensors"):
            data = (tmp_path / name).read_bytes()
            for root in roots:
                files[f"{root}/{name}"] = data
    return rows, layout, files


def _split_read(layout, reader, keys, weights, roots=ROOTS):
    row_reader = Exl3RowReader(layout, reader, direct=False)
    _keep, buf = _buffers(len(keys), row_reader.buffer_bytes)
    starts = row_reader.read_split(
        keys,
        [buf[i].data_ptr() for i in range(len(keys))],
        roots=list(roots),
        policy=StaticSplitPolicy(weights),
    )
    return row_reader, buf, starts


def test_read_split_is_one_submit_with_one_read_per_nonempty_part(tmp_path):
    rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files)
    keys = [(0, 0), (0, 1), (0, 2)]
    _row_reader, buf, starts = _split_read(layout, reader, keys, (1.0, 1.0))

    assert len(reader.calls) == 1
    assert len(reader.calls[0]) == 6
    file_of = {path: i for i, path in enumerate(reader._paths)}
    for i, key in enumerate(keys):
        offset, length, _start = layout.records[key].aligned_read(PAGE)
        split = StaticSplitPolicy((1.0, 1.0)).plan(length)
        for r, root in enumerate(ROOTS):
            assert reader.calls[0][2 * i + r] == (
                file_of[f"{root}/{SHARD}"],
                offset + split.starts[r],
                buf[i].data_ptr() + split.starts[r],
                split.part_bytes[r],
            )
    for i, (key, start) in enumerate(zip(keys, starts)):
        assert bytes(buf[i, start : start + layout.row_bytes].numpy()) == rows[key]


def test_read_split_opens_each_roots_copy_once(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files)
    row_reader, _buf, _starts = _split_read(
        layout, reader, [(0, 0), (0, 1), (0, 2)], (1.0, 1.0)
    )
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    row_reader.read_split(
        [(0, 1)],
        [buf[0].data_ptr()],
        roots=list(ROOTS),
        policy=StaticSplitPolicy((1.0, 1.0)),
    )
    assert sorted(reader.opened) == [f"{root}/{SHARD}" for root in ROOTS]


def test_a_zero_part_issues_no_read(tmp_path):
    rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files)
    keys = [(0, 0), (0, 1)]
    _row_reader, buf, starts = _split_read(layout, reader, keys, (1.0, 0.0))
    assert len(reader.calls) == 1
    assert len(reader.calls[0]) == 2  # one per row, none for the dropped root
    assert reader.opened == [f"/mirror_a/{SHARD}"]
    for i, (key, start) in enumerate(zip(keys, starts)):
        assert bytes(buf[i, start : start + layout.row_bytes].numpy()) == rows[key]


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, 1.0)])
def test_a_split_row_that_overruns_end_of_file_is_not_a_short_read(tmp_path, weights):
    rows, layout, files = _mirrored(tmp_path)
    key = (0, 2)  # last row of its shard: its aligned superset runs past EOF
    offset, length, _start = layout.records[key].aligned_read(PAGE)
    file_bytes = len(files[f"/mirror_a/{SHARD}"])
    assert offset + length > file_bytes, "fixture must overrun end of file"

    reader = _RecordingReader(files)
    _row_reader, buf, starts = _split_read(layout, reader, [key], weights)
    (call,) = reader.calls
    # Exactly one part reaches past EOF (the last non-empty one) and the reader
    # serves it short; the batch must still count as fully read.
    over = [o + n - file_bytes for _f, o, _d, n in call if o + n > file_bytes]
    assert len(over) == 1 and 0 < over[0] < PAGE
    assert bytes(buf[0, starts[0] : starts[0] + layout.row_bytes].numpy()) == rows[key]


def test_a_short_read_fails_the_batch_like_the_single_root_path(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files, short_by=PAGE)
    with pytest.raises(RuntimeError, match="ended early"):
        _split_read(layout, reader, [(0, 0), (0, 1)], (1.0, 1.0))


def test_a_per_root_read_error_propagates(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files, fail_file=1)  # the second root's copy
    with pytest.raises(OSError, match="read failed"):
        _split_read(layout, reader, [(0, 0)], (1.0, 1.0))


def test_a_root_missing_the_file_names_the_root_and_path(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    del files[f"/mirror_b/{SHARD}"]
    with pytest.raises(FileNotFoundError, match=f"/mirror_b/{SHARD}"):
        _split_read(layout, _RecordingReader(files), [(0, 0)], (1.0, 1.0))


def test_roots_may_have_different_absolute_prefixes(tmp_path):
    rows, layout, files = _mirrored(tmp_path, roots=("/mnt/a/ckpt", "/other/b"))
    reader = _RecordingReader(files)
    _row_reader, buf, starts = _split_read(
        layout, reader, [(0, 1)], (1.0, 1.0), roots=("/mnt/a/ckpt", "/other/b")
    )
    assert sorted(reader.opened) == [f"/mnt/a/ckpt/{SHARD}", f"/other/b/{SHARD}"]
    assert (
        bytes(buf[0, starts[0] : starts[0] + layout.row_bytes].numpy()) == rows[(0, 1)]
    )


def test_read_split_rejects_unaligned_destination(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    row_reader = Exl3RowReader(layout, _RecordingReader(files), direct=False)
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    with pytest.raises(ValueError, match="page-aligned"):
        row_reader.read_split(
            [(0, 0)],
            [buf[0].data_ptr() + 16],
            roots=list(ROOTS),
            policy=StaticSplitPolicy((1.0, 1.0)),
        )


def test_read_split_rejects_a_policy_for_a_different_root_count(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    row_reader = Exl3RowReader(layout, _RecordingReader(files), direct=False)
    _keep, buf = _buffers(1, row_reader.buffer_bytes)
    with pytest.raises(ValueError, match="roots"):
        row_reader.read_split(
            [(0, 0)],
            [buf[0].data_ptr()],
            roots=list(ROOTS),
            policy=StaticSplitPolicy((1.0, 1.0, 1.0)),
        )


def test_read_split_empty_is_a_no_op(tmp_path):
    _rows, layout, files = _mirrored(tmp_path)
    reader = _RecordingReader(files)
    row_reader = Exl3RowReader(layout, reader, direct=False)
    policy = StaticSplitPolicy((1.0,))
    assert row_reader.read_split([], [], roots=["/mirror_a"], policy=policy) == []
    assert reader.calls == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
