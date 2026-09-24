"""The C++ row reader reads and splits EXL3 rows byte for byte like Exl3ShardRowSource (CPU)."""

import errno
import os
import shutil
import tempfile

import pytest
import torch

from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import read_rows_once, read_rows_traced
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# True while test_exl3_ram_miss_pack_workers reruns these tests with SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM's
# reader, which issues a nonzero part as its sub-reads: a count of reads is then a count of sub-reads.
PIECE_STREAM = False


def _reads(s, layer, experts, parts=None):
    """The reads a read of ``experts`` issues (of ``parts`` only, when given): one per nonzero part, or with piece
    streaming one per sub-read."""
    total = 0
    for expert in experts:
        for part in range(s.tables.parts) if parts is None else parts:
            if PIECE_STREAM:
                total += sum(sub["part"] == part for sub in ops.piece_geometry(s.tables, layer, expert)[0])
            else:
                total += int(s.tables.extents[layer, expert, part, 2] > 0)
    return total


def _n(flag_off, s, layer, experts, *, parts=None, extra=0):
    """A read count the test states as ``flag_off`` for the one-read-per-part reader (checked against the tables);
    under piece streaming the same reads counted as sub-reads. ``extra``: resubmissions, in both."""
    count = _reads(s, layer, experts, parts) + extra
    if not PIECE_STREAM:
        assert count == flag_off, (count, flag_off)
    return count


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
    assert cqes[0] == _n(16, s, 1, first)
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
    assert cqes[0] == _n(16, s, 1, first)
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
    first_reads = _n(3 * extents_per_row, s, 1, first)
    assert cqes == [first_reads, first_reads + _n(2 * extents_per_row, s, 1, then)]
    _assert_rows(s, 1, first, [0, 1, 2])
    _assert_rows(s, 1, then, [3, 4])


@pytest.mark.parametrize("part", [0, 1])
def test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one(tmp_path, part):
    # Rows pack as they complete, so the rows that landed before the failure may already be in their
    # (unpublished) slots: the tier releases those slots and publishes nothing. What must hold at the
    # reader is that the failing row's slot is untouched and every other slot is either untouched or a
    # WHOLE, byte-exact row: a row is never packed before all of its own extents landed.
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    for slot in (0, 1, 2):
        _sentinel(s, 1, slot)
    then = [5, 4, 3]
    results = ops.read_rows_with_fault(
        s.tables, 1, [0, 1, 2], [0, 1, 2], then, [3, 4, 5], direct=False, part=part, part_error=EIO, ordinal=1
    )
    assert results == (0, 1)  # the read fails, the reader stays usable
    assert _untouched(s, 1, 1)  # the failed row was not packed
    reference = s.reference(1, [0, 1, 2])
    for slot in (0, 2):
        whole = all(same_bytes(s.slabs[1][name][slot], reference[name][slot]) for name in EXL3_STREAMED_NAMES)
        assert whole or _untouched(s, 1, slot), slot
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


def test_a_table_whose_parts_disagree_on_the_rows_base_is_refused(tmp_path):
    """This byte pattern used to be read successfully, with the extent wholly past EOF clamped to an
    expectation of 0. It is now refused when the table is built, which is the stronger property.

    Why it can no longer be handled instead: the reader decides a row's EOF question from ONE of its
    parts, so every part of a row must agree on the aligned base (offset - dest) and the file size.
    The hand-written extent below disagrees, and nothing an exporter produces does.

    Why the clamp it used to cover is gone rather than untested: a part spans whole pages and a row's
    minimal aligned superset overruns EOF by less than one page, so every reading extent starts
    strictly inside its file and an expectation of 0 is unreachable. admit_batch now refuses such an
    extent outright rather than clamping it, because an extent that expects nothing retires having
    read nothing while its row still packs and publishes.

    Keep this test pointed at the table build: if the row-consistency rule is ever relaxed, this fails
    loudly instead of silently going green on a table the reader can no longer decide.
    """
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
    with pytest.raises(RuntimeError, match="aligned base or their file size"):
        read_rows_once(tables, 0, [0], [0], direct=False)


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


def _assert_stages_ordered(record):
    """The stamps of a read that packed. Reads and packing overlap, so last_cqe and pack_start are NOT
    ordered (a row packs as soon as its own extents landed): what holds is that packing cannot start
    before the first completion, and that the last completion is followed by the last row's packing."""
    names = ("submit", "first_cqe", "last_cqe", "pack_start", "pack_end")
    assert all(record[name] > 0 for name in names), record
    assert record["submit"] <= record["first_cqe"] <= record["last_cqe"], record
    assert record["first_cqe"] <= record["pack_start"] <= record["pack_end"], record
    assert record["last_cqe"] <= record["pack_end"], record


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
    # pack_ns adds the rows' own spans; the first start to the last end also holds the waits between rows.
    assert 0 < record["pack_ns"] <= record["pack_end"] - record["pack_start"]


def test_a_read_over_several_batches_sums_them(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12)
    experts = list(range(11))[::-1]
    result, record = read_rows_traced(s.tables, 1, experts, list(range(11)), direct=False)
    assert result == 1 and record["batches"] == 2 and record["extents"] == 11
    _assert_stages_ordered(record)
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
        assert record["extents"] == _n(6, s, 1, experts) and len(record["drives"]) == 2
        assert {d["dev"] for d in record["drives"]} == {os.stat(p).st_dev for p in s.tables.paths}
        assert [d["extents"] for d in record["drives"]] == [_n(3, s, 1, experts, parts=[p]) for p in (0, 1)]
        assert sum(d["bytes"] for d in record["drives"]) == record["bytes"] == _bytes_read(s, 1, experts)
        assert all(d["bytes"] > 0 for d in record["drives"])
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def test_an_empty_part_is_not_an_extent_and_reads_no_drive(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 0.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2], [0, 1, 2], direct=False)
    assert result == 1 and record["extents"] == _n(3, s, 1, [0, 1, 2])
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


# ---- Byte split, per-row packing and per-extent CQE stamps (see StageRecord) ----


def _lengths(s, layer, experts):
    """The first attempts' total: every extent's length, as submitted (not clamped at end of file)."""
    return sum(int(s.tables.extents[layer, e, p, 2]) for e in experts for p in range(s.tables.parts))


def _segment_bytes(s):
    return int(s.tables.segments[:, 3].sum())


def test_the_byte_split_of_a_clean_read(tmp_path):
    s = ram_miss_setup(tmp_path)
    experts = [5, 0, 2]  # expert 5 is the last row of its shard: its aligned tail passes end of file
    result, record = read_rows_traced(s.tables, 1, experts, [1, 2, 0], direct=False)
    assert result == 1
    assert record["submitted_bytes"] == _lengths(s, 1, experts)
    assert record["bytes"] == _bytes_read(s, 1, experts)
    assert record["useful_bytes"] == len(experts) * _segment_bytes(s)
    assert record["retried_bytes"] == 0 and record["cancelled_bytes"] == 0
    assert 0 < record["useful_bytes"] <= record["bytes"] <= record["submitted_bytes"]


@pytest.mark.parametrize("part", [0, 1])
def test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful(tmp_path, part):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(
        s.tables, 1, [0], [0], direct=False, part=part, part_short=PAGE, reverse_cqes=True
    )
    length = int(s.tables.extents[1, 0, part, 2])
    assert result == 1
    assert record["retried_bytes"] == length - PAGE  # the resubmit asked for the rest of that extent only
    assert record["submitted_bytes"] == _lengths(s, 1, [0]) + record["retried_bytes"]
    assert record["bytes"] == _bytes_read(s, 1, [0])  # every byte completed once
    assert record["useful_bytes"] == _segment_bytes(s) and record["cancelled_bytes"] == 0
    assert record["extents"] == 2  # a resubmission is not a new extent
    _assert_rows(s, 1, [0], [0])


def test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried(tmp_path):
    s = ram_miss_setup(tmp_path)
    result, record = read_rows_traced(s.tables, 1, [0], [0], direct=False, cqe_error=errno.EINTR, cqe_call=1)
    length = int(s.tables.extents[1, 0, 0, 2])
    assert result == 1 and record["retried_bytes"] == length
    assert record["submitted_bytes"] == 2 * length and record["bytes"] == _bytes_read(s, 1, [0])


# A failed read: the bytes its extents were still owed are cancelled, so completed + cancelled is the
# whole expected total, and nothing was packed.
@pytest.mark.parametrize(
    "fault",
    [
        dict(cqe_error=EIO, cqe_call=1),
        dict(submit_error=EIO, submit_call=1, submit_first=False),  # prepared, never submitted
        dict(submit_error=EIO, submit_call=1, submit_first=True),  # in flight at the error
    ],
)
def test_a_failed_read_cancels_the_bytes_it_never_received(tmp_path, fault):
    s = ram_miss_setup(tmp_path)
    experts = [0, 1, 2]
    result, record = read_rows_traced(s.tables, 1, experts, [0, 1, 2], direct=False, **fault)
    assert result == 0 and record["ok"] == 0
    assert record["bytes"] + record["cancelled_bytes"] == _bytes_read(s, 1, experts)
    assert record["cancelled_bytes"] > 0 and record["useful_bytes"] == 0
    assert record["bytes"] <= record["submitted_bytes"]
    # No row packed: each row asked for is listed with 0/0, and an extent that never completed has no stamp.
    assert record["rows_asked"] == 3
    assert [(row["row"], row["start"], row["end"]) for row in record["row_pack"]] == [(k, 0, 0) for k in range(3)]
    assert all(row["admit"] > 0 for row in record["row_pack"])  # admitted, read, never packed
    assert record["pack_start"] == 0 and record["pack_end"] == 0
    assert any(extent["cqe"] == 0 for extent in record["extent_cqe"])


def test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12)
    experts = list(range(9))  # batch 1: 8 rows, batch 2: 1 row
    # Both batches are admitted at once (two banks). A credit of one serialises the reads in queue order,
    # so completion 9 is the second batch's only read and every row of the first batch has packed by then:
    # without the cap the order in which rows landed and packed would decide what the failure found.
    result, record = read_rows_traced(
        s.tables, 1, experts, list(range(9)), direct=False, cqe_error=EIO, cqe_call=_n(9, s, 1, experts[:8], extra=1),
        max_outstanding=1,
    )
    assert result == 0 and record["batches"] == 2
    assert record["useful_bytes"] == 8 * _segment_bytes(s)  # the first batch packed and counted
    assert record["cancelled_bytes"] == _bytes_read(s, 1, [8])
    assert record["bytes"] + record["cancelled_bytes"] == _bytes_read(s, 1, experts)
    packed = [row for row in record["row_pack"] if row["end"]]
    assert [row["row"] for row in packed] == list(range(8))  # the failed batch's row never packed
    failed_row = record["row_pack"][8]
    assert (failed_row["row"], failed_row["start"], failed_row["end"]) == (8, 0, 0) and failed_row["admit"] > 0


def _assert_row_causality(record, reads_per_row):
    """Per row only. Another row's extents may complete after this row packs (Task 4 overlaps them),
    so no ordering across rows is asserted, and the stamps as a whole are not required to be sorted.
    ``reads_per_row[k]``: the extents row k issues; a row whose extents ran past the record's extent slots
    (only possible with piece streaming's sub-reads) has no complete set to check."""
    by_row = {}
    for extent in record["extent_cqe"]:
        by_row.setdefault(extent["row"], []).append(extent)
    for row in record["row_pack"]:
        if sum(reads_per_row[: row["row"] + 1]) > ops.STAGE_TRACE_EXTENTS:
            assert PIECE_STREAM
            continue
        extents = by_row[row["row"]]
        assert len(extents) == reads_per_row[row["row"]]
        assert all(record["submit"] > 0 and extent["cqe"] > 0 for extent in extents), record
        assert row["start"] > 0 and row["end"] >= row["start"], row
        if PIECE_STREAM:
            # Each piece packs once its own sub-reads landed: the row's first piece after its first sub-read, and
            # the piece that needs its last sub-read after that one.
            assert min(extent["cqe"] for extent in extents) <= row["start"], (row, extents)
            assert max(extent["cqe"] for extent in extents) <= row["end"], (row, extents)
            continue
        # A row's packing starts after all of ITS extents completed.
        assert max(extent["cqe"] for extent in extents) <= row["start"], (row, extents)
    for extent in record["extent_cqe"]:
        assert record["submit"] <= record["first_cqe"] <= extent["cqe"], extent
        # first_cqe/last_cqe cover the whole read.
        assert extent["cqe"] <= record["last_cqe"], extent


@pytest.mark.parametrize(
    "weights, fault",
    [
        (None, {}),
        ((1.0, 1.0), {}),
        ((1.0, 1.0), dict(reverse_cqes=True, max_outstanding=2)),
        ((1.0, 1.0), dict(part=1, part_short=PAGE, reverse_cqes=True, max_outstanding=3)),
    ],
)
def test_each_rows_stamps_are_causally_ordered_over_several_batches(tmp_path, weights, fault):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights)
    experts = list(range(11))[::-1]  # a batch of 8 rows, then 3
    result, record = read_rows_traced(s.tables, 1, experts, list(range(11)), direct=False, **fault)
    assert result == 1 and record["batches"] == 2
    reads = _n(11 * s.tables.parts, s, 1, experts)
    assert len(record["row_pack"]) == 11 and len(record["extent_cqe"]) == min(reads, ops.STAGE_TRACE_EXTENTS)
    _assert_row_causality(record, [_n(s.tables.parts, s, 1, [e]) for e in experts])
    # The aggregate is the rows': it starts with the first row to pack, ends with the last, adds their spans.
    rows = record["row_pack"]
    assert record["pack_start"] == min(r["start"] for r in rows) and record["pack_end"] == max(r["end"] for r in rows)
    assert record["pack_ns"] == sum(r["end"] - r["start"] for r in rows)
    _assert_rows(s, 1, experts, list(range(11)))


def test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=18, experts=18, mirror_weights=(1.0, 1.0))
    experts = list(range(18))
    result, record = read_rows_traced(s.tables, 1, experts, list(range(18)), direct=False)
    assert result == 1
    assert len(record["row_pack"]) == ops.STAGE_TRACE_ROWS and record["rows_untraced"] == 2
    assert record["extents"] == _n(36, s, 1, experts) and len(record["extent_cqe"]) == ops.STAGE_TRACE_EXTENTS
    assert record["extents_untraced"] == _n(36, s, 1, experts) - ops.STAGE_TRACE_EXTENTS
    assert record["useful_bytes"] == 18 * _segment_bytes(s)  # the totals still cover every row
    assert record["submitted_bytes"] == _lengths(s, 1, experts)
    _assert_rows(s, 1, experts, list(range(18)))


def _wipe(s):
    for layer in s.slabs.values():
        for slab in layer.values():
            slab.view(torch.uint8).fill_(0xAB)


def _slabs(s, slots):
    return [s.slabs[1][name][slots].clone() for name in EXL3_STREAMED_NAMES]


@pytest.mark.parametrize(
    "weights, fault",
    [
        (None, {}),
        ((1.0, 1.0), {}),
        ((1.0, 1.0), dict(part=0, part_short=PAGE, reverse_cqes=True, max_outstanding=2)),
        ((1.0, 1.0), dict(cqe_error=errno.EINTR, cqe_call=3)),
    ],
)
def test_a_traced_read_is_byte_identical_to_an_untraced_one(tmp_path, weights, fault):
    """The trace only reads the clock and adds to its record, so an untraced read (the record pointer
    null: no clock read, no vector, no stamp) must leave exactly the bytes a traced one does. Both
    paths run the same faults, so retries and reordering are covered too."""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights)
    experts = list(range(11))[::-1]
    slots = list(range(11))
    _wipe(s)
    if fault:
        # The untraced fault harness; its clean follow-up read lands in slot 11, outside the compared slots.
        assert ops.read_rows_with_fault(
            s.tables, 1, experts, slots, [5], [11], direct=False, **fault
        ) == (1, 1)
    else:
        assert read_rows_once(s.tables, 1, experts, slots, direct=False) == 1
    untraced = _slabs(s, slots)
    _wipe(s)
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, **fault)
    assert result == 1 and record["ok"] == 1
    assert all(same_bytes(a, b) for a, b in zip(untraced, _slabs(s, slots)))
    _assert_rows(s, 1, experts, slots)


# ---- Task 4: two bounce banks, row-level packing, and what may and may not overlap ----
#
# Cached buffered reads complete inside submit, so a test cannot make an extent slow. `hold_ordinal`
# simulates a slow drive: that row's completions are withheld from the reader until every other row is
# done (the kernel is already finished with them, so no memory is at stake). Everything else is real.


def _row_packs(record):
    return {row["row"]: row for row in record["row_pack"]}


def _extent_cqes(record, row):
    return [extent["cqe"] for extent in record["extent_cqe"] if extent["row"] == row]


def _exact_or_untouched(s, layer, experts, slots):
    """Every slot is either still the sentinel or a WHOLE byte-exact row: never a half-packed one. With piece
    streaming a failed read may leave a row's published pieces behind (the device may already hold them), so the
    unit is the piece: each is either all sentinel or all the row's bytes, never a half-copied one."""
    reference = s.reference(layer, experts)
    for i, slot in enumerate(slots):
        whole = all(same_bytes(s.slabs[layer][name][slot], reference[name][i]) for name in EXL3_STREAMED_NAMES)
        if PIECE_STREAM and not whole:
            _assert_pieces_exact_or_untouched(s, layer, experts[i], slot, {n: reference[n][i] for n in reference})
            continue
        assert whole or _untouched(s, layer, slot), (experts[i], slot)


def _assert_pieces_exact_or_untouched(s, layer, expert, slot, reference):
    _, pieces = ops.piece_geometry(s.tables, layer, expert)
    segments = s.tables.segments.tolist()
    for j, piece in enumerate(pieces):
        states = set()
        for (name_index, _dst, _src, _n), (lo, hi) in zip(segments, piece["runs"]):
            if lo == hi:
                continue
            name = EXL3_STREAMED_NAMES[name_index]
            got = s.slabs[layer][name][slot].contiguous().view(torch.uint8)[lo:hi]
            want = reference[name].contiguous().view(torch.uint8)[lo:hi]
            states.add("exact" if torch.equal(got, want) else "untouched" if bool((got == 0xAB).all()) else "torn")
        assert len(states) <= 1 and "torn" not in states, (expert, slot, j, states)


@pytest.mark.parametrize("weights", [None, (1.0, 1.0)])
def test_a_row_packs_while_another_rows_read_is_still_outstanding(tmp_path, weights):
    """The timeline gate. Every row's reads are submitted together (one submit, before any row packs);
    row 5's completion is then held back, as a slow drive would. Rows 0-4 must pack while it is
    outstanding: each one's row_pack_start precedes row 5's last extent_cqe, and follows the first submit."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=weights)
    experts, slots = [3, 0, 5, 1, 4, 2], list(range(6))
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, hold_ordinal=5)
    assert result == 1 and record["rows_reading_max"] == 6
    packs = _row_packs(record)
    held = _extent_cqes(record, 5)
    # All of row 5's extents, as far as the record's extent slots reach (sub-reads can run past them).
    traced = min(_n(s.tables.parts, s, 1, [experts[5]]), ops.STAGE_TRACE_EXTENTS - _reads(s, 1, experts[:5]))
    assert len(held) == traced and all(held)
    for k in range(5):
        assert record["submit"] < packs[k]["start"] < max(held), (k, packs[k], held)
    # Row 5 itself packs only after ALL of its own extents completed, and after the others.
    assert max(held) <= packs[5]["start"]
    assert all(packs[k]["end"] <= packs[5]["start"] for k in range(5))
    _assert_stages_ordered(record)
    # Packing started before the last completion: the overlap, in the aggregate stamps.
    assert record["pack_start"] < record["last_cqe"]
    _assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize(
    "fault",
    [
        {},
        dict(reverse_cqes=True),
        dict(pack_delay_ns=2_000_000),
        dict(poison=True),
        dict(reverse_cqes=True, pack_delay_ns=1_000_000, poison=True, max_outstanding=3),
    ],
)
def test_a_bank_is_not_reused_until_every_row_in_it_has_packed(tmp_path, fault):
    """12 rows in batches of 4: batch 2 needs batch 0's bank. Row 0 (in that bank) is held back, so the
    bank cannot retire until it has been released and packed. Batch 2's reads must therefore start after
    row 0 packed: every one of its extents completes after row 0's packing ended. Delayed packing,
    reversed completions and poisoned bounce slots and descriptors must not change one byte."""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(11, -1, -1)), list(range(12))
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, step=4, hold_ordinal=0, **fault
    )
    assert result == 1 and record["batches"] == 3
    assert record["bank_stalls"] >= 1  # batch 2 did wait for its bank
    assert record["rows_reading_max"] == 8  # two banks of 4: never more rows admitted than they hold
    packs = _row_packs(record)
    for row in (8, 9, 10, 11):
        assert all(cqe > packs[0]["end"] for cqe in _extent_cqes(record, row)), (row, packs[0])
    _assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize("credit", [0, 2, 3, 5])
def test_ring_credit_is_independent_of_the_banks(tmp_path, credit):
    """16 rows fill both banks (16 slots) whatever the ring's credit: rows admitted are bounded by the
    banks, SQEs prepared by the credit, and neither depends on the other."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(16)), list(range(16))
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, max_outstanding=credit)
    assert result == 1 and record["batches"] == 2
    assert record["rows_reading_max"] == 16
    ring = 16 * s.tables.parts
    assert record["pending_max"] <= (credit or ring)
    if credit:
        assert record["pending_max"] == credit  # 32 extents against a credit of at most 5: it bound
    _assert_rows(s, 1, experts, slots)


def test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack(tmp_path):
    """The plan's Task 4 item asks for evidence that refill() runs before bounded packing work, not just
    before packing starts once. Loop order (exl3_ram_miss_host.cpp:642-655): collect_packed, admit,
    refill, reap, pack_one -- every turn re-submits whatever credit allows BEFORE that turn spends the
    CPU on one row's (blocking, inline) pack.

    Three single-extent rows, credit for two: row 0 and row 1 submit and retire together, then row 0
    packs (150 ms, inline, blocks the owner). Row 2 was never submitted (credit exhausted by rows 0-1),
    so it can only submit once a slot frees -- which happens when row 0's turn ends and control returns
    to the loop, BEFORE row 1 (the next row in packing order) is packed. So row 2's submit must land
    before row 1's pack_start. A reader that called refill() after pack_one() instead would submit
    nothing on the very first turn (pending and ready both start at 0), fail admission's post-loop
    clean-state check, and return 0.

    Note what that mutant does and does not pin. It kills this test through ``result == 1``, because
    moving refill() after pack_one() stops the loop submitting anything at all -- so the submit-ordering
    assertion below never gets to discriminate. No cheaper mutant is known that leaves the reader
    working and only reorders refill against packing. Treat the ordering assertion as documentation of
    the intended invariant with a coarse guard behind it, not as an independently falsified claim."""
    s = ram_miss_setup(tmp_path, capacity=3)
    experts, slots = [3, 0, 5], [0, 1, 2]
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, max_outstanding=2, pack_delay_ns=150_000_000,
    )
    assert result == 1
    packs = _row_packs(record)
    submits = {e["row"]: e["submit"] for e in record["extent_cqe"]}
    assert submits[2] < packs[1]["start"], (submits, packs)
    _assert_stages_ordered(record)
    _assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize(
    "fault",
    [
        {},
        dict(reverse_cqes=True, max_outstanding=3),
        dict(pack_delay_ns=500_000, poison=True),
        dict(generation_start=2**32 - 3, poison=True),
    ],
)
def test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte(tmp_path, fault):
    """20 rows through 2-row batches recycle every descriptor and slot repeatedly. A retired descriptor is
    scribbled and a slot is filled with a pattern when a row takes it and another when it packs, so a row
    packed before its reads landed, or from a recycled descriptor, shows in the bytes."""
    s = ram_miss_setup(tmp_path, capacity=20, experts=20, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(19, -1, -1)), list(range(20))
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, step=2, **fault)
    assert result == 1 and record["batches"] == 10
    _assert_rows(s, 1, experts, slots)


def test_the_generation_counter_wraps_and_the_reader_goes_on(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=13, experts=11, mirror_weights=(1.0, 1.0))
    stats = {}
    experts = list(range(11))
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, list(range(11)), [5, 4], [11, 12], direct=False, step=2,
        generation_start=2**32 - 4, poison=True, stats=stats,
    ) == (1, 1)
    assert stats["generation_wraps"] >= 1 and stats["stale_cqes"] == 0  # 22 extents from 3 below the wrap
    _assert_rows(s, 1, experts, list(range(11)))
    _assert_rows(s, 1, [5, 4], [11, 12])


@pytest.mark.parametrize("reverse", [False, True])
def test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor(tmp_path, reverse):
    """The first extent to retire has its completion delivered AGAIN once its descriptor holds another
    extent. Without the generation in user_data that completion would finish the new extent, marking its
    row ready to pack before its bytes landed and publishing the bounce's stale contents. It must fail the
    read instead, and every slot must be untouched or a whole byte-exact row."""
    s = ram_miss_setup(tmp_path, capacity=18, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(16)), list(range(16))
    for slot in slots:
        _sentinel(s, 1, slot)
    stats = {}
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [16, 17], direct=False, step=4,
        stale_cqe_call=1, poison=True, reverse_cqes=reverse, stats=stats,
    ) == (0, 1)
    assert stats["stale_cqes"] == 1
    _exact_or_untouched(s, 1, experts, slots)
    _assert_rows(s, 1, [5, 4], [16, 17])  # the reader is clean afterwards


@pytest.mark.parametrize("call", [1, 2, 3])
@pytest.mark.parametrize("weights", [None, (1.0, 1.0)])
def test_a_submit_that_consumes_nothing_is_resubmitted(tmp_path, call, weights):
    """A partial submit leaves prepared SQEs the kernel never saw, still counted against the credit.
    Nothing may wait on them: the next submit sends them."""
    s = ram_miss_setup(tmp_path, capacity=18, experts=16, mirror_weights=weights)
    experts, slots = list(range(16)), list(range(16))
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [16, 17], direct=False, step=4, max_outstanding=5,
        submit_short_call=call,
    ) == (1, 1)
    _assert_rows(s, 1, experts, slots)
    _assert_rows(s, 1, [5, 4], [16, 17])


# A fault with both banks in flight and rows already packed: the read fails, nothing is published, the
# other bank's reads are reaped, and the reader is clean for the next read.
BOTH_BANKS_FAULTS = [
    dict(cqe_error=EIO, cqe_call=1),
    dict(cqe_error=EIO, cqe_call=13),
    dict(cqe_error=EIO, cqe_call=32),
    dict(part=1, part_error=EIO, ordinal=11),
    dict(part=0, part_error=EIO, ordinal=3, hold_ordinal=12),
    # A credit of 4 makes the read need many submits (cached reads otherwise finish inside the first).
    dict(submit_error=EIO, submit_call=3, submit_first=True, max_outstanding=4),
    dict(submit_error=EIO, submit_call=3, submit_first=False, max_outstanding=4),
    dict(cqe_error=EIO, cqe_call=9, max_outstanding=3, reverse_cqes=True),
]


@pytest.mark.parametrize("fault", BOTH_BANKS_FAULTS)
def test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring(tmp_path, fault):
    s = ram_miss_setup(tmp_path, capacity=18, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(16)), list(range(16))
    for slot in slots:
        _sentinel(s, 1, slot)
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [16, 17], direct=False, step=8, poison=True, **fault
    )[0] == 0
    _exact_or_untouched(s, 1, experts, slots)
    _assert_rows(s, 1, [5, 4], [16, 17])


@pytest.mark.parametrize("ordinal", [2, 9, 15])
@pytest.mark.parametrize("direct", [False, True])
def test_a_short_read_in_either_bank_resubmits_only_its_own_extent(tmp_path, ordinal, direct):
    s = ram_miss_setup(tmp_path, capacity=18, experts=16, mirror_weights=(1.0, 1.0))
    if direct:
        try:
            os.close(os.open(s.tables.paths[0], os.O_RDONLY | os.O_DIRECT))
        except OSError as error:
            pytest.skip(f"{tmp_path} does not support O_DIRECT: {error}")
    experts, slots = list(range(16)), list(range(16))
    cqes = []
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [16, 17], direct=direct, step=8,
        part=1, part_short=PAGE, ordinal=ordinal, reverse_cqes=True, max_outstanding=4, cqes=cqes,
    ) == (1, 1)
    assert cqes[0] == 33  # 32 extents, the shortened one completing twice
    _assert_rows(s, 1, experts, slots)


# Cancellation: `abandon_after` stops admitting new batches. What was admitted is still reaped and
# packed (the kernel keeps its destination until it is done), what was not is never read.
@pytest.mark.parametrize("fault", [{}, dict(hold_ordinal=1), dict(reverse_cqes=True, pack_delay_ns=500_000, poison=True)])
def test_cancellation_reaps_what_was_submitted_and_reads_nothing_more(tmp_path, fault):
    s = ram_miss_setup(tmp_path, capacity=8, experts=6, mirror_weights=(1.0, 1.0))
    experts, slots = [4, 1, 5, 0, 3, 2], [0, 1, 2, 3, 4, 5]
    for slot in slots:
        _sentinel(s, 1, slot)
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, step=1, abandon_after=3, **fault)
    assert result == -1 and record["status"] == "cancelled" and record["batches"] == 3
    packed = [row["row"] for row in record["row_pack"] if row["end"]]
    assert packed == [0, 1, 2]  # every row admitted before the stop was completed, the rest never started
    assert record["extents"] == _n(6, s, 1, experts[:3]) and record["submitted_bytes"] == _lengths(s, 1, experts[:3])
    assert record["cancelled_bytes"] == 0  # nothing was abandoned mid-flight: it was all reaped
    _assert_rows(s, 1, experts[:3], slots[:3])
    assert all(_untouched(s, 1, slot) for slot in slots[3:])
    # The reader is clean for the next read.
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [6, 7], direct=False, step=1, abandon_after=2, **fault
    ) == (-1, 1)
    _assert_rows(s, 1, [5, 4], [6, 7])


def test_one_row_batches_are_bounded_by_the_banks(tmp_path):
    """Batches of one row give each bank one row: at most two rows can be outstanding, however many rows
    the request has (the third batch waits for a bank). The tier's advisory rule, one row outstanding, is
    max_reading_rows and is tested at the tier (test_exl3_ram_miss_thread)."""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8)
    experts, slots = list(range(8)), list(range(8))
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, step=1)
    assert result == 1 and record["batches"] == 8
    assert record["rows_reading_max"] <= 2 and record["bank_stalls"] >= 1
    _assert_rows(s, 1, experts, slots)


def test_a_row_with_nothing_to_read_fails_instead_of_packing_stale_bytes(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    s.tables.extents[1, 3, :, 2] = 0  # both roots serve none of expert 3
    for slot in (0, 1, 2):
        _sentinel(s, 1, slot)
    assert read_rows_once(s.tables, 1, [0, 3, 2], [0, 1, 2], direct=False) == 0
    assert _untouched(s, 1, 1)
    _exact_or_untouched(s, 1, [0, 3, 2], [0, 1, 2])



@pytest.mark.parametrize("weights", [None, (1.0, 1.0)])
def test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing(tmp_path, weights):
    """The last defence against publishing bytes no drive delivered. The table is built and the file
    is big enough, so the table build and admit_batch's EOF guard both accept the row; only the
    coverage check at pack time sees that the extents, read whole and clean, still fall a page short of
    the last byte the segments copy. It must fail the request and leave the slot untouched."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=weights)
    need_end = int((s.tables.segments[:, 2] + s.tables.segments[:, 3]).max())
    experts, slots = [0, 1, 2], [0, 1, 2]
    reading = [p for p in range(s.tables.parts) if int(s.tables.extents[1, 1, p, 2]) > 0]
    delivered = sum(int(s.tables.extents[1, 1, p, 2]) for p in reading)
    needed = int(s.tables.starts[1, 1]) + need_end
    assert needed <= delivered < needed + PAGE  # the table is minimal: one page fewer is short
    s.tables.extents[1, 1, reading[-1], 2] -= PAGE
    for slot in slots:
        _sentinel(s, 1, slot)
    assert ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [5, 4], [3, 4], direct=False, poison=True
    ) == (0, 1)
    assert _untouched(s, 1, 1)
    _exact_or_untouched(s, 1, experts, slots)
    _assert_rows(s, 1, [5, 4], [3, 4])  # the reader is clean afterwards


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
