"""The C++ row reader reads and splits EXL3 rows byte for byte like Exl3ShardRowSource (CPU)."""

import os

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import read_rows_once
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def test_tables_describe_every_row(tmp_path):
    s = ram_miss_setup(tmp_path)
    assert s.tables.reads.shape == (2, 6, 4)
    assert s.tables.capacity.tolist() == [3, 3]
    assert s.tables.slabs[1, 0].item() == s.slabs[1]["w13_trellis"].data_ptr()
    assert s.tables.slot_bytes % 4096 == 0
    assert s.tables.segments.shape == (len(s.fmt.segment_map()), 4)
    assert s.tables.row_bytes.tolist() == [
        s.slabs[0][n].numel() * s.slabs[0][n].element_size() // 3 for n in EXL3_STREAMED_NAMES
    ]


@pytest.mark.parametrize("layer, experts, slots", [(0, [0], [2]), (1, [5, 2, 3], [0, 1, 2])])
def test_rows_are_split_like_the_python_row_source(tmp_path, layer, experts, slots):
    s = ram_miss_setup(tmp_path)
    assert read_rows_once(s.tables, layer, experts, slots, direct=False) == 1
    reference = s.reference(layer, experts)
    for i, slot in enumerate(slots):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[layer][name][slot], reference[name][i]), (name, experts[i])


def test_the_last_row_of_a_shard_is_clamped_at_end_of_file(tmp_path):
    s = ram_miss_setup(tmp_path)
    # write_fake_exl3 puts 3 experts per shard: expert 5 of layer 1 is the last row of the last shard.
    assert read_rows_once(s.tables, 1, [5], [0], direct=False) == 1
    assert same_bytes(s.slabs[1]["w2_svh"][0], s.reference(1, [5])["w2_svh"][0])


def test_a_short_file_fails_the_read(tmp_path):
    s = ram_miss_setup(tmp_path)
    path = s.tables.paths[int(s.tables.reads[0, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.reads[0, 0, 1]) + 100)  # cut inside expert 0's superset
    assert read_rows_once(s.tables, 0, [0], [0], direct=False) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
