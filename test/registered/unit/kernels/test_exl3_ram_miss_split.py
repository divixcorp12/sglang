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


@pytest.mark.parametrize("credit", [1, 3, 5])
def test_a_batch_needing_more_reads_than_its_credit_completes_through_refill(tmp_path, credit):
    # 8 rows x 2 parts = 16 extents, prepared a few at a time. Production constants keep a
    # batch inside the ring (kBounceRows rows against a kQueueDepth * parts ring), so credit
    # never binds there; capping it is the only way to reach the refill path. A credit of 1
    # serialises the batch completely.
    # Slots 8 and 9 hold the clean follow-up read: it must not overwrite what is asserted.
    s = ram_miss_setup(tmp_path, capacity=10, experts=8, mirror_weights=(1.0, 1.0))
    cqes = []
    first, then = list(range(8)), [5, 4]
    assert ops.read_rows_with_fault(
        s.tables, 1, first, list(range(8)), then, [8, 9], direct=False,
        max_outstanding=credit, cqes=cqes,
    ) == (1, 1)
    # Every extent still reads exactly once, however few were in flight at a time.
    assert cqes[0] == 16
    _assert_rows(s, 1, first, list(range(8)))
    _assert_rows(s, 1, then, [8, 9])


@pytest.mark.parametrize("credit", [0, 3])
def test_completions_processed_back_to_front_land_the_same_bytes(tmp_path, credit):
    # Nothing in the reader may depend on the order completions arrive: each extent is keyed
    # by its own user_data and carries its own done/expected/retries. The kernel gives no
    # ordering guarantee across drives, so reversing each reaped batch must change nothing.
    # This cannot force a reap to return more than one completion, so it is a necessary
    # check rather than a proof that reordering was exercised on every run.
    s = ram_miss_setup(tmp_path, capacity=10, experts=8, mirror_weights=(1.0, 1.0))
    cqes = []
    first, then = list(range(8)), [5, 4]
    assert ops.read_rows_with_fault(
        s.tables, 1, first, list(range(8)), then, [8, 9], direct=False,
        reverse_cqes=True, max_outstanding=credit, cqes=cqes,
    ) == (1, 1)
    assert cqes[0] == 16
    _assert_rows(s, 1, first, list(range(8)))
    _assert_rows(s, 1, then, [8, 9])


@pytest.mark.parametrize("part", [0, 1])
def test_a_short_read_resubmits_its_own_extent_under_reversed_completions(tmp_path, part):
    # A retry re-enters through the credit queue, so it takes credit like any other read.
    # Reversing delivery must not misattribute the short completion to another extent.
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    cqes = []
    assert ops.read_rows_with_fault(
        s.tables, 1, [0, 1, 2], [0, 1, 2], [5], [3], direct=False,
        part=part, part_short=PAGE, reverse_cqes=True, max_outstanding=2, cqes=cqes,
    ) == (1, 1)
    assert cqes[0] == 7  # 6 extents, one completing twice
    _assert_rows(s, 1, [0, 1, 2], [0, 1, 2])
    _assert_rows(s, 1, [5], [3])


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


# ---- End of file (handoff 3A) ----
#
# What a reader owes a row is that every byte the expert needs is read correctly. The bytes a
# page-aligned superset overruns past end of file are undefined and never asserted on.

EOF_PAGE = 4096
EOF_FILE_BYTES = 5 * EOF_PAGE + 1000  # not a multiple of any logical block size
EOF_NAME = "shard.bin"


def _eof_checkpoint(tmp_path, weights, *, rows, need_bytes=None):
    """One shard whose size is not block-aligned, one copy of it per root, and a layout with one
    (file offset, nbytes) record per entry of ``rows`` (expert = position, all layer 0).

    Every row starts on a page, so it starts its buffer. The single segment copies the first
    ``need_bytes`` of the row (default: the whole aligned superset of row 0) into the slab.
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
        for expert, (offset, nbytes) in enumerate(rows)
    }
    layout = Exl3ExpertLayout(tensors=(), row_bytes=0, records=records, num_layers=1, num_experts=len(rows))
    need = need_bytes or records[(0, 0)].aligned_read(EOF_PAGE)[1]
    segments = [RowSegment("w13_trellis", 0, 0, 0, need)]
    slabs = {
        0: {
            name: torch.zeros((len(rows), need if name == "w13_trellis" else 0), dtype=torch.uint8)
            for name in EXL3_STREAMED_NAMES
        }
    }
    return layout, segments, slabs, roots, data, str(source)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("weights", [None, (1.0,), (1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (0.0, 1.0)])
def test_the_last_row_of_a_shard_reads_correctly_natively_and_eagerly(tmp_path, weights, direct):
    """The production case: a valid row whose minimal aligned superset overruns EOF."""
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    # The row ends exactly at end of file: its 4-page superset [2, 6) overruns by 3096 B, and page
    # 5 holds only 1000 valid bytes. Its needed bytes (start 0, all of them) end at EOF.
    row = (2 * EOF_PAGE, EOF_FILE_BYTES - 2 * EOF_PAGE)
    layout, segments, slabs, roots, data, source = _eof_checkpoint(
        tmp_path, weights, rows=[row], need_bytes=row[1]
    )
    if direct:
        try:
            os.close(os.open(str(tmp_path / "ckpt" / EOF_NAME), os.O_RDONLY | os.O_DIRECT))
        except OSError as error:
            pytest.skip(f"{tmp_path} does not support O_DIRECT: {error}")
    mirrors = {} if weights is None else dict(roots=roots, policy=StaticSplitPolicy(weights), source_root=source)
    tables = exl3_ram_miss_tables(layout, segments, slabs, **mirrors)
    assert tables.extents[0, 0, :, 2].sum().item() == 4 * EOF_PAGE  # the overrunning superset is what is read

    assert read_rows_once(tables, 0, [0], [0], direct=direct) == 1
    needed = data[2 * EOF_PAGE :]
    assert bytes(slabs[0]["w13_trellis"][0].numpy()) == needed

    buffer = allocate_host_slab(1, (4 * EOF_PAGE,), torch.uint8, register=False)
    reader = Exl3RowReader(layout, direct=direct, source_root=source)
    if weights is None:
        reader.read([(0, 0)], [buffer[0].data_ptr()])
    else:
        reader.read_split([(0, 0)], [buffer[0].data_ptr()], roots=roots, policy=StaticSplitPolicy(weights))
    assert bytes(buffer[0].numpy())[: len(needed)] == needed


def test_a_padding_extent_wholly_past_end_of_file_is_clamped_to_nothing(tmp_path):
    """A synthetic edge case of the clamp arithmetic (no export produces it): an extent wholly past
    EOF has negative room, expects 0 bytes, and does not fail a row whose needed bytes are all
    present. This is not the production case above."""
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy

    weights = (1.0, 1.0)
    row = (3 * EOF_PAGE, 2 * EOF_PAGE)  # wholly inside the file; one page per root
    layout, segments, slabs, roots, data, source = _eof_checkpoint(
        tmp_path, weights, rows=[row], need_bytes=EOF_PAGE  # the expert needs only the first page
    )
    tables = exl3_ram_miss_tables(
        layout, segments, slabs, roots=roots, policy=StaticSplitPolicy(weights), source_root=source
    )
    file_1 = int(tables.extents[0, 0, 1, 0])
    tables.extents[0, 0, 1] = torch.tensor([file_1, 7 * EOF_PAGE, EOF_PAGE, EOF_PAGE])  # wholly past EOF
    assert read_rows_once(tables, 0, [0], [0], direct=False) == 1
    assert bytes(slabs[0]["w13_trellis"][0].numpy()) == data[3 * EOF_PAGE : 4 * EOF_PAGE]


@pytest.mark.parametrize("weights", [None, (1.0, 1.0)])
def test_a_row_that_needs_bytes_the_file_lacks_fails_in_both_readers(tmp_path, weights):
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    # Row 1's record claims [5 pages, 7 pages) of a file that ends 1000 B into page 5: the row
    # itself is not inside the file (not merely its aligned tail).
    layout, segments, slabs, roots, _data, source = _eof_checkpoint(
        tmp_path, weights, rows=[(0, 2 * EOF_PAGE), (5 * EOF_PAGE, 2 * EOF_PAGE)]
    )
    mirrors = {} if weights is None else dict(roots=roots, policy=StaticSplitPolicy(weights), source_root=source)
    tables = exl3_ram_miss_tables(layout, segments, slabs, **mirrors)
    buffer = allocate_host_slab(1, (2 * EOF_PAGE,), torch.uint8, register=False)
    eager = Exl3RowReader(layout, direct=False, source_root=source)
    with pytest.raises(RuntimeError, match="needs bytes"):
        if weights is None:
            eager.read([(0, 1)], [buffer[0].data_ptr()])
        else:
            eager.read_split([(0, 1)], [buffer[0].data_ptr()], roots=roots, policy=StaticSplitPolicy(weights))
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
