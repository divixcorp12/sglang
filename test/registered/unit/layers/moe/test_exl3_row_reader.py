"""One aligned read per EXL3 expert, straight from the shards."""

import pytest
import torch

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
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
    rows = write_fake_exl3(str(tmp_path), num_layers=1, num_experts=3, experts_per_shard=3)
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
