"""The C++ row reader reads and splits EXL3 rows byte for byte like Exl3ShardRowSource (CPU)."""

import errno
import os

import pytest
import torch

from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import read_rows_once
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def test_tables_describe_every_row(tmp_path):
    s = ram_miss_setup(tmp_path)
    assert s.tables.extents.shape == (2, 6, 1, 4)
    assert s.tables.starts.shape == (2, 6)
    assert s.tables.parts == 1
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
    path = s.tables.paths[int(s.tables.extents[0, 0, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.extents[0, 0, 0, 1]) + 100)  # cut inside expert 0's superset
    assert read_rows_once(s.tables, 0, [0], [0], direct=False) == 0


def _assert_rows(s, layer, experts, slots):
    reference = s.reference(layer, experts)
    for i, slot in enumerate(slots):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[layer][name][slot], reference[name][i]), (name, experts[i], slot)


def test_rows_span_several_batches(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12)
    experts = list(range(11))[::-1]  # 11 rows: a full batch of 8, then 3
    slots = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
    assert read_rows_once(s.tables, 1, experts, slots, direct=False) == 1
    _assert_rows(s, 1, experts, slots)


def test_one_row_per_batch(tmp_path):
    s = ram_miss_setup(tmp_path)
    assert read_rows_once(s.tables, 0, [4, 1, 3], [2, 0, 1], direct=False, step=1) == 1
    _assert_rows(s, 0, [4, 1, 3], [2, 0, 1])


def test_direct_reads_split_like_the_python_row_source(tmp_path):
    s = ram_miss_setup(tmp_path)
    try:
        os.close(os.open(s.tables.paths[0], os.O_RDONLY | os.O_DIRECT))
    except OSError as error:
        pytest.skip(f"{tmp_path} does not support O_DIRECT: {error}")
    assert read_rows_once(s.tables, 1, [5, 0, 2], [1, 2, 0], direct=True) == 1
    _assert_rows(s, 1, [5, 0, 2], [1, 2, 0])


@pytest.mark.parametrize(
    "row, experts, slots",
    [(0, [0, 1], [0]), (2, [0], [0]), (-1, [0], [0]), (0, [6], [0]), (0, [-1], [0]), (0, [0], [3]), (0, [0], [-1])],
)
def test_bad_arguments_raise(tmp_path, row, experts, slots):
    s = ram_miss_setup(tmp_path)
    with pytest.raises(ValueError):
        read_rows_once(s.tables, row, experts, slots, direct=False)


# (fault keyword arguments, result of the faulted read). Every case then reads other experts on the
# same reader: that read must succeed and land byte-exact, so no stale SQE or CQE survived the fault.
FAULTS = [
    (dict(submit_error=errno.EINTR, submit_call=1, submit_first=False), 1),
    (dict(submit_error=errno.EINTR, submit_call=1, submit_first=True), 1),
    (dict(submit_error=errno.EAGAIN, submit_call=1, submit_first=True), 1),
    (dict(submit_error=errno.EIO, submit_call=1, submit_first=False), 0),  # prepared SQEs never submitted
    (dict(submit_error=errno.EIO, submit_call=1, submit_first=True), 0),  # reads in flight at the error
    (dict(cqe_error=errno.EAGAIN, cqe_call=2), 1),  # resubmitted
    (dict(cqe_error=errno.EINTR, cqe_call=1), 1),
    (dict(cqe_error=errno.EIO, cqe_call=1), 0),  # the other reads are drained
]


@pytest.mark.parametrize("fault, first_result", FAULTS)
def test_a_fault_leaves_the_ring_clean(tmp_path, fault, first_result):
    s = ram_miss_setup(tmp_path, capacity=6)
    first, then = [0, 1, 2], [5, 4, 3]
    results = ops.read_rows_with_fault(
        s.tables, 1, first, [0, 1, 2], then, [3, 4, 5], direct=False, **fault
    )
    assert results == (first_result, 1)
    if first_result == 1:
        _assert_rows(s, 1, first, [0, 1, 2])
    _assert_rows(s, 1, then, [3, 4, 5])


# ---- Mirrored rows: one extent per root, each with its own completion ----

EIO = errno.EIO
PAGE = 4096
WEIGHTS = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, 1.0, 2.0)]


def _sentinel(s, layer, slot):
    for name in EXL3_STREAMED_NAMES:
        s.slabs[layer][name][slot].view(torch.uint8).fill_(0xAB)


def _untouched(s, layer, slot):
    return all(
        bool((s.slabs[layer][name][slot].view(torch.uint8) == 0xAB).all()) for name in EXL3_STREAMED_NAMES
    )


@pytest.mark.parametrize("weights", WEIGHTS)
def test_mirrored_rows_are_split_like_the_python_row_source(tmp_path, weights):
    s = ram_miss_setup(tmp_path, mirror_weights=weights)
    assert s.tables.parts == len(weights)
    assert read_rows_once(s.tables, 1, [5, 2, 3], [0, 1, 2], direct=False) == 1
    _assert_rows(s, 1, [5, 2, 3], [0, 1, 2])


def test_a_full_batch_of_three_part_rows_fits_the_ring(tmp_path):
    # 8 rows x 3 parts = 24 extents in one batch, more than the 16 entries a one-part ring holds.
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 1.0, 1.0))
    experts, slots = list(range(8)), list(range(8))[::-1]
    assert read_rows_once(s.tables, 1, experts, slots, direct=False) == 1
    _assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize("weights, extents_per_row", [((1.0, 1.0), 2), ((1.0, 0.0), 1), ((0.0, 1.0), 1)])
def test_a_zero_length_part_issues_no_read(tmp_path, weights, extents_per_row):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=weights)
    cqes = []
    first, then = [0, 1, 2], [5, 4]
    assert ops.read_rows_with_fault(
        s.tables, 1, first, [0, 1, 2], then, [3, 4], direct=False, cqes=cqes
    ) == (1, 1)
    # One completion per non-empty extent, so an empty part neither reads nor is waited for.
    assert cqes == [3 * extents_per_row, 3 * extents_per_row + 2 * extents_per_row]
    _assert_rows(s, 1, first, [0, 1, 2])
    _assert_rows(s, 1, then, [3, 4])


@pytest.mark.parametrize("part", [0, 1])
def test_a_failed_part_fails_the_row_and_publishes_nothing(tmp_path, part):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    for slot in (0, 1, 2):
        _sentinel(s, 1, slot)
    then = [5, 4, 3]
    results = ops.read_rows_with_fault(
        s.tables, 1, [0, 1, 2], [0, 1, 2], then, [3, 4, 5], direct=False, part=part, part_error=EIO
    )
    assert results == (0, 1)  # the batch fails, the reader stays usable
    assert all(_untouched(s, 1, slot) for slot in (0, 1, 2))  # not one row of it was published
    _assert_rows(s, 1, then, [3, 4, 5])


@pytest.mark.parametrize("part", [0, 1])
@pytest.mark.parametrize("direct", [False, True])
def test_a_short_read_resubmits_only_its_own_extent(tmp_path, part, direct):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    if direct:
        try:
            os.close(os.open(s.tables.paths[0], os.O_RDONLY | os.O_DIRECT))
        except OSError as error:
            pytest.skip(f"{tmp_path} does not support O_DIRECT: {error}")
    cqes = []
    assert ops.read_rows_with_fault(
        s.tables, 1, [0, 1, 2], [0, 1, 2], [5], [3], direct=direct, part=part, part_short=PAGE, cqes=cqes
    ) == (1, 1)
    # 6 extents, one of them completing twice: the resubmit read the rest of that extent, no other.
    assert cqes[0] == 7
    _assert_rows(s, 1, [0, 1, 2], [0, 1, 2])
    _assert_rows(s, 1, [5], [3])


def test_a_short_mirror_copy_fails_the_read(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0))
    # The copy is bounded by the SOURCE's size, so truncating it is seen, not clamped away.
    shard, offset = int(s.tables.extents[0, 0, 1, 0]), int(s.tables.extents[0, 0, 1, 1])
    with open(s.tables.paths[shard], "r+b") as f:
        f.truncate(offset + 100)
    assert read_rows_once(s.tables, 0, [0], [0], direct=False) == 0


def test_tables_keep_their_slabs_alive(tmp_path):
    s = ram_miss_setup(tmp_path)
    kept = {id(t) for t in s.tables.keepalive}
    assert all(id(s.slabs[layer][name]) in kept for layer in s.slabs for name in EXL3_STREAMED_NAMES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
