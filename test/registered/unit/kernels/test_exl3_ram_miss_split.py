"""The C++ row reader reads and splits EXL3 rows byte for byte like Exl3ShardRowSource (CPU)."""

import errno
import os
import shutil
import tempfile

import pytest
import torch

from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import STAGE_ORDER, read_rows_once, read_rows_traced
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


def test_open_refuses_a_source_file_that_is_not_the_size_the_tables_were_built_from(tmp_path):
    s = ram_miss_setup(tmp_path)
    path = s.tables.paths[int(s.tables.extents[0, 0, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.extents[0, 0, 0, 1]) + 100)  # cut inside expert 0's superset
    with pytest.raises(RuntimeError, match="has size 100 bytes but its source"):
        read_rows_once(s.tables, 0, [0], [0], direct=False)


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


@pytest.mark.parametrize("delta", [-100, +100, -PAGE])
def test_open_refuses_a_mirror_copy_of_the_wrong_size_naming_both_files(tmp_path, delta):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0))
    # File index = shard * parts + part: file 1 is root 1's copy of the first shard.
    mirror, source = s.tables.paths[1], s.tables.source_paths[1]
    assert mirror.startswith(s.roots[1]) and source == s.tables.source_paths[0]
    source_bytes = os.path.getsize(source)
    with open(mirror, "r+b") as f:
        f.truncate(source_bytes + delta)
    for attempt in (
        lambda: read_rows_once(s.tables, 0, [0], [0], direct=False),
        lambda: ops.Exl3RamMissHost(
            s.tables, page=ops.new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False
        ),
    ):
        with pytest.raises(RuntimeError) as caught:
            attempt()
        message = str(caught.value)
        assert mirror in message and source in message
        assert f"size {source_bytes + delta} bytes" in message and f"size {source_bytes} bytes" in message


def test_open_accepts_mirror_copies_of_the_source_size(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0, 1.0))
    assert s.tables.source_paths == [p for p in dict.fromkeys(s.tables.source_paths) for _ in range(3)]
    assert read_rows_once(s.tables, 0, [0, 5], [0, 1], direct=False) == 1


# ---- End of file: the native reader and Exl3RowReader must agree (handoff 3A) ----

EOF_PAGE = 4096
EOF_FILE_BYTES = 5 * EOF_PAGE + 1000  # not a multiple of any logical block size
EOF_NAME = "shard.bin"


def _eof_checkpoint(tmp_path, weights, *, claimed_rows):
    """One shard whose size is not block-aligned, one copy of it per root, and a layout of
    ``claimed_rows`` (file offset, nbytes) records, expert e = position, all in layer 0.

    Returns (layout, segments, slabs, roots, policy_weights, data). A row is read whole: its
    single segment copies the entire aligned superset (offset and length are page multiples, so
    the row starts the buffer), which lets the test see the bytes past EOF as well.
    """
    from sglang.srt.layers.moe.exl3_expert_format import RowSegment
    from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout, Exl3ExpertRecord

    source = tmp_path / "ckpt"
    source.mkdir()
    data = bytes((i * 7 + 13) % 251 for i in range(EOF_FILE_BYTES))
    (source / EOF_NAME).write_bytes(data)
    roots = []
    for i in range(len(weights or ())):
        root = tmp_path / f"drive_{i}"
        root.mkdir()
        (root / EOF_NAME).write_bytes(data)
        roots.append(str(root))
    records = {
        (0, expert): Exl3ExpertRecord(0, expert, str(source / EOF_NAME), offset, nbytes)
        for expert, (offset, nbytes) in enumerate(claimed_rows)
    }
    layout = Exl3ExpertLayout(
        tensors=(), row_bytes=0, records=records, num_layers=1, num_experts=len(claimed_rows)
    )
    superset = records[(0, 0)].aligned_read(EOF_PAGE)[1]
    segments = [RowSegment("w13_trellis", 0, 0, 0, superset)]
    slabs = {
        0: {
            name: torch.zeros((len(claimed_rows), superset if name == "w13_trellis" else 0), dtype=torch.uint8)
            for name in EXL3_STREAMED_NAMES
        }
    }
    return layout, segments, slabs, roots, data, str(source)


# Row 0 is whole and inside the file; row 1 is the shard's last row: it ends exactly at end of
# file, so its 4-page aligned superset [2, 6) overruns by 3096 B (page 5 holds 1000 valid bytes).
EOF_ROWS = [(0, 4 * EOF_PAGE), (2 * EOF_PAGE, EOF_FILE_BYTES - 2 * EOF_PAGE)]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("weights", [None, (1.0,), (1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (0.0, 1.0)])
def test_a_row_crossing_end_of_file_reads_the_same_natively_and_eagerly(tmp_path, weights, direct):
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    layout, segments, slabs, roots, data, source = _eof_checkpoint(tmp_path, weights, claimed_rows=EOF_ROWS)
    if direct:
        try:
            os.close(os.open(str(tmp_path / "ckpt" / EOF_NAME), os.O_RDONLY | os.O_DIRECT))
        except OSError as error:
            pytest.skip(f"{tmp_path} does not support O_DIRECT: {error}")
    mirrors = {} if weights is None else dict(roots=roots, policy=StaticSplitPolicy(weights), source_root=source)
    tables = exl3_ram_miss_tables(layout, segments, slabs, **mirrors)

    # Native: row 0 then row 1 through one reader, both in bounce slot 0. Row 1's bytes past end
    # of file are never written, so they are still row 0's.
    results = ops.read_rows_with_fault(tables, 0, [0], [0], [1], [1], direct=direct)
    assert results == (1, 1)
    native = slabs[0]["w13_trellis"]

    # Eager: the same row into a page-aligned buffer that already holds row 0's bytes.
    superset = 4 * EOF_PAGE
    buffer = allocate_host_slab(1, (superset,), torch.uint8, register=False)
    buffer[0].copy_(native[0])
    reader = Exl3RowReader(layout, direct=direct, source_root=source)
    address = buffer[0].data_ptr()
    if weights is None:
        reader.read([(0, 1)], [address])
    else:
        reader.read_split([(0, 1)], [address], roots=roots, policy=StaticSplitPolicy(weights))

    assert bytes(native[0].numpy()) == data[0:superset]
    valid = EOF_FILE_BYTES - 2 * EOF_PAGE
    row_1 = bytes(native[1].numpy())
    assert row_1[:valid] == data[2 * EOF_PAGE :]  # the row itself
    assert row_1[valid:] == data[valid:superset]  # past end of file: row 0's bytes, untouched
    assert row_1 == bytes(buffer[0].numpy())  # and the eager reader left the same bytes


def test_a_part_entirely_past_end_of_file_fails_natively_as_it_does_eagerly(tmp_path):
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    # A record that claims bytes beyond the file (a layout no export produces): with two roots
    # its second page lies entirely past end of file.
    rows = [(0, 2 * EOF_PAGE), (5 * EOF_PAGE, 2 * EOF_PAGE)]
    weights = (1.0, 1.0)
    layout, segments, slabs, roots, _data, source = _eof_checkpoint(tmp_path, weights, claimed_rows=rows)
    policy = StaticSplitPolicy(weights)
    tables = exl3_ram_miss_tables(layout, segments, slabs, roots=roots, policy=policy, source_root=source)

    buffer = allocate_host_slab(1, (2 * EOF_PAGE,), torch.uint8, register=False)
    eager = Exl3RowReader(layout, direct=False, source_root=source)
    with pytest.raises(RuntimeError, match="ended early"):
        eager.read_split([(0, 1)], [buffer[0].data_ptr()], roots=roots, policy=policy)
    assert read_rows_once(tables, 0, [1], [0], direct=False) == 0
    assert read_rows_once(tables, 0, [0], [0], direct=False) == 1  # the sound row still reads


def test_tables_keep_their_slabs_alive(tmp_path):
    s = ram_miss_setup(tmp_path)
    kept = {id(t) for t in s.tables.keepalive}
    assert all(id(s.slabs[layer][name]) in kept for layer in s.slabs for name in EXL3_STREAMED_NAMES)


def _bytes_read(s, layer, experts):
    """What the reader must read for ``experts``: each row's aligned length, cut at its file's end."""
    total = 0
    for expert in experts:
        for part in range(s.tables.parts):
            file, offset, length, _ = (int(v) for v in s.tables.extents[layer, expert, part])
            if length:  # an empty part reads nothing, wherever its offset points
                total += min(length, int(s.tables.file_sizes[file]) - offset)
    return total


def _assert_stages_ordered(record, reached=STAGE_ORDER[2:7]):
    stamps = [record[name] for name in reached]
    assert all(stamp > 0 for stamp in stamps), record
    assert stamps == sorted(stamps), record


def test_a_read_records_its_stages_and_bytes(tmp_path):
    s = ram_miss_setup(tmp_path)
    experts, slots = [5, 0, 2], [1, 2, 0]
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False)
    assert result == 1 and record["ok"] == 1
    _assert_stages_ordered(record)
    assert record["batches"] == 1 and record["extents"] == 3
    expected = _bytes_read(s, 1, experts)
    assert record["bytes"] == expected
    assert sum(drive["bytes"] for drive in record["drives"]) == expected
    assert sum(drive["extents"] for drive in record["drives"]) == 3
    # One filesystem holds the whole fake checkpoint: one drive, named by its st_dev.
    assert [drive["dev"] for drive in record["drives"]] == [os.stat(s.tables.paths[0]).st_dev]
    assert record["submit_to_first_cqe_ns"] >= 0 and record["first_to_last_cqe_ns"] >= 0 and record["pack_ns"] > 0
    assert record["pack_end"] - record["pack_start"] == record["pack_ns"]  # one batch


def test_a_read_over_several_batches_sums_them(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12)
    experts = list(range(11))[::-1]
    result, record = read_rows_traced(s.tables, 1, experts, list(range(11)), direct=False)
    assert result == 1 and record["batches"] == 2 and record["extents"] == 11
    _assert_stages_ordered(record)  # the first batch's stamps, pack_end the last batch's
    assert record["bytes"] == _bytes_read(s, 1, experts)
    assert sum(drive["bytes"] for drive in record["drives"]) == record["bytes"]
    # pack_ns adds both batches' packing, which the first-to-last span of the stamps includes.
    assert 0 < record["pack_ns"] <= record["pack_end"] - record["pack_start"]


def test_a_traced_read_moves_the_same_bytes_as_an_untraced_one(tmp_path):
    s = ram_miss_setup(tmp_path)
    assert read_rows_traced(s.tables, 1, [5, 0, 2], [1, 2, 0], direct=False)[0] == 1
    _assert_rows(s, 1, [5, 0, 2], [1, 2, 0])


def test_mirrored_reads_are_accounted_per_drive(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    # Move root 1's copy onto another filesystem (tmpfs /dev/shm), so the two parts are two drives.
    shm = tempfile.mkdtemp(dir="/dev/shm")
    try:
        for index, path in enumerate(s.tables.paths):
            if index % s.tables.parts == 1:
                moved = os.path.join(shm, f"{index}_{os.path.basename(path)}")
                shutil.copy(path, moved)
                s.tables.paths[index] = moved
        if len({os.stat(path).st_dev for path in s.tables.paths}) < 2:
            pytest.skip("/dev/shm is on the same filesystem as the checkpoint")
        experts = [0, 1, 2]
        result, record = read_rows_traced(s.tables, 1, experts, [0, 1, 2], direct=False)
        assert result == 1
        assert record["extents"] == 6 and len(record["drives"]) == 2
        assert {d["dev"] for d in record["drives"]} == {os.stat(p).st_dev for p in s.tables.paths}
        assert [d["extents"] for d in record["drives"]] == [3, 3]
        assert sum(d["bytes"] for d in record["drives"]) == record["bytes"] == _bytes_read(s, 1, experts)
        assert all(d["bytes"] > 0 for d in record["drives"])
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def test_an_empty_part_is_not_an_extent_and_reads_no_drive(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 0.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2], [0, 1, 2], direct=False)
    assert result == 1 and record["extents"] == 3
    assert sum(d["bytes"] for d in record["drives"]) == record["bytes"] == _bytes_read(s, 1, [0, 1, 2])


# A retried or resubmitted read must count each byte once; a hard error leaves a partial, consistent record.
@pytest.mark.parametrize(
    "fault, result",
    [
        (dict(cqe_error=errno.EAGAIN, cqe_call=2), 1),
        (dict(cqe_error=errno.EINTR, cqe_call=1), 1),
        (dict(submit_error=errno.EINTR, submit_call=1, submit_first=True), 1),
        (dict(cqe_error=errno.EIO, cqe_call=1), 0),
        (dict(submit_error=errno.EIO, submit_call=1, submit_first=True), 0),
    ],
)
def test_a_fault_leaves_a_consistent_record(tmp_path, fault, result):
    s = ram_miss_setup(tmp_path)
    experts = [0, 1, 2]
    got, record = read_rows_traced(s.tables, 1, experts, [0, 1, 2], direct=False, **fault)
    assert got == result and record["ok"] == result
    assert record["bytes"] == sum(drive["bytes"] for drive in record["drives"])
    if result == 1:
        assert record["bytes"] == _bytes_read(s, 1, experts)
        _assert_stages_ordered(record)
    else:
        assert record["bytes"] <= _bytes_read(s, 1, experts)
        assert record["pack_start"] == 0 and record["pack_end"] == 0  # nothing was packed


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
