"""The C++ row reader (CPU): row images read with one O_DIRECT readv per read straight into the pinned slab rows
(plan 2026-09-24-dsv41-row-images Part B). Since plan 2026-09-29-hotpath-zero-overhead D4 this is the only reader:
the packed path is gone, shard tables and buffered reads are refused, and every suite of the split, thread, piece-stream
and two-phase modules runs on row images itself. The tests written out here are the ones only this reader can break:
every slab byte against the checkpoint oracle, a resubmission that resumes inside the iovec list, the refusals at
open, and the service's wiring of the images.
"""

import errno
import resource
import types

import pytest
import torch

import test_exl3_ram_miss_split as split
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_pieces, read_rows_sqes, read_rows_traced
from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.exl3_ram_miss import check_piece_stream, exl3_ram_miss_tables, open_service_row_images
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=240, suite="base-a-test-cpu")


_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft != resource.RLIM_INFINITY and _soft < 16384:
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384 if _hard == resource.RLIM_INFINITY else min(16384, _hard), _hard))

EIO = errno.EIO
PAGE = 4096


def test_the_fixture_builds_row_image_tables_by_default_and_requires_o_direct(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    s = ram_miss_setup(tmp_path / "a")
    assert s.tables.row_images and all(path.endswith(".rows") for path in s.tables.paths)
    from sglang.test import dsv41_ram_miss_fixtures as fx

    monkeypatch.setattr(fx, "_takes_o_direct", lambda path: False)
    with pytest.raises(RuntimeError, match="O_DIRECT"):
        fx.ram_miss_setup(tmp_path / "b")
    with pytest.raises(ValueError, match="row images only"):
        fx.ram_miss_setup(tmp_path / "c", row_images=False)  # shard tables: the packed path that read them is gone


# ---- Every slab byte is the checkpoint's ----


def _sentinel_all(s):
    for layer in s.slabs:
        for name in EXL3_STREAMED_NAMES:
            s.slabs[layer][name].view(torch.uint8).fill_(0x5A)


@pytest.mark.parametrize(
    "weights",
    [None, (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (3.0, 1.0, 2.0)],
    ids=["one", "halves", "first", "second", "3to1", "thirds", "zero_middle", "uneven_thirds"],
)
@pytest.mark.parametrize("pieces", [False, True], ids=["rows", "pieces"])
def test_every_slab_byte_is_the_checkpoints_after_a_run_of_reads(tmp_path, weights, pieces):
    """Derived property: an image read is a re-layout of the checkpoint row, so after any run of requests every slot
    a request named holds its last expert's rows exactly as the checkpoint oracle (``s.reference``) splits them, and
    every slot none named still holds the sentinel, byte for byte. The requests are the split suite's shapes: one row,
    rows over several batches (11 rows, 8 + 3), one row per batch, and every mirror split; the image's last part ends
    inside a page, so the padding is never read. (It replaced the comparison with the deleted bounce path, whose own
    oracle was the same checkpoint read.)"""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights)
    assert s.tables.slot_bytes % PAGE != 0  # the image ends inside its last page
    requests = [(0, [4], [2], 8), (1, list(range(11))[::-1], [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4], 8), (0, [4, 1, 3], [2, 0, 1], 1)]
    _sentinel_all(s)
    holds = {}
    for row, experts, slots, step in requests:
        extra = dict(piece_stream=True) if pieces else {}
        assert read_rows_traced(s.tables, row, experts, slots, step=step, **extra)[0] == 1
        holds.update({(row, slot): expert for expert, slot in zip(experts, slots)})
    for layer in s.slabs:
        for slot in range(12):
            expert = holds.get((layer, slot))
            want = s.reference(layer, [expert]) if expert is not None else None
            for name in EXL3_STREAMED_NAMES:
                have = s.slabs[layer][name][slot]
                if want is None:
                    assert bool((have.contiguous().view(torch.uint8) == 0x5A).all()), (layer, slot, name)
                else:
                    assert same_bytes(have, want[name][0]), (layer, slot, name, expert)


# ---- A resubmission resumes inside the iovec list ----


# The fake image's slab rows (EXL3_STREAMED_NAMES order): 49152, 1024, 1024, 24576, 512, 512 bytes.
IMAGE_ROW_STARTS = [0, 49152, 50176, 51200, 75776, 76288]


@pytest.mark.parametrize(
    "short, where",
    [(512, "inside the first slab row"), (49152, "exactly at a slab row boundary"),
     (49152 + 512, "one block past a boundary"), (51200 - 512, "one block before a boundary")],
)
def test_a_short_read_resumes_at_the_byte_it_stopped_even_mid_row(tmp_path, short, where):
    """Derived property: after a short O_DIRECT read of ``done`` bytes, the resubmission must scatter the rest of the
    read from image offset ``done`` on: its first iovec starts ``done - s.src`` into the slab row of the segment
    holding that offset, and the later iovecs are whole rows. An off-by-one-segment (or restarting the iovecs from the
    part's start at the new file offset) lands every later byte in the wrong row, which the byte check shows."""
    s = ram_miss_setup(tmp_path, capacity=6)
    assert s.tables.segments[:, 2].tolist() == IMAGE_ROW_STARTS  # the boundaries the cases are placed around
    experts, slots = [3, 0, 5], [0, 1, 2]
    result, sqes, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, part=0, part_short=short, ordinal=1
    )
    assert result == 1, where
    image = int(s.tables.slot_bytes)
    resubmitted = [q for q in sqes if q[2] == image - short]
    assert len(resubmitted) == 1 and resubmitted[0][1] == int(s.tables.extents[1, 0, 0, 1]) + short, (where, sqes)
    assert record["retried_bytes"] == image - short
    split._assert_rows(s, 1, experts, slots)


def test_a_short_sub_read_resumes_mid_sub_read_across_slab_rows(tmp_path):
    """The same with piece streaming, where a sub-read spans several slab rows: sub-read 0 of the only part is the
    image's first 20480 bytes (a quarter, page-rounded), inside the first name's row; sub-read 2 spans three rows. A
    short read of 1536 bytes leaves a resubmission that starts 1536 bytes into the first name's row and runs on
    across the next ones; every piece is still published once, and every byte is exact."""
    s = ram_miss_setup(tmp_path, capacity=6)
    experts, slots = [2, 4], [1, 0]
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=7, piece_stream=True, part=0, sub=0, part_short=3 * 512,
        ordinal=0,
    )
    assert result == 1 and info["refused"] == 0
    assert record["retried_bytes"] > 0
    assert all(int(w) & 0xFF == 0xFF for w in masks.view(-1).tolist())
    split._assert_rows(s, 1, experts, slots)


# ---- An I/O error ----


@pytest.mark.parametrize("pieces", [False, True], ids=["rows", "pieces"])
def test_an_io_error_fails_the_read_leaves_the_ring_clean_and_the_next_read_lands_exact(tmp_path, pieces):
    """Bug-shaped guard for the direct mode's failure rule: the kernel writes the slab rows itself, so after an
    error nothing may still be in flight when read() returns (the caller releases the slots then). The next read on
    the same reader lands other experts byte-exact in the same slots, which it could not if a straggling write or a
    stale descriptor from the failed read were still live. (What the failed read left in them is not asserted: in the
    direct mode an unpublished slot may hold part of a failed row, and the caller releases it.)"""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 1.0))
    experts, slots = [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]
    for slot in slots:
        split._sentinel(s, 1, slot)
    extra = dict(piece_stream=True) if pieces else {}
    # The clean read puts OTHER experts into the failed read's slots (its failed row's included), so a write of the
    # failed read landing late would show as the wrong expert's bytes.
    results = ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [6, 7, 0], [2, 0, 1], part=1, part_error=EIO, ordinal=2,
        max_outstanding=2, **extra,
    )
    assert results == (0, 1)
    split._assert_rows(s, 1, [6, 7, 0], [2, 0, 1])


FAILURES = [
    dict(cqe_error=EIO, cqe_call=1),
    dict(cqe_error=EIO, cqe_call=13),
    dict(part=1, part_error=EIO, ordinal=11),
    dict(part=0, part_error=EIO, ordinal=3, hold_ordinal=12),
    dict(submit_error=EIO, submit_call=3, submit_first=True, max_outstanding=4),
    dict(submit_error=EIO, submit_call=3, submit_first=False, max_outstanding=4),
    dict(cqe_error=EIO, cqe_call=9, max_outstanding=3, reverse_cqes=True),
    dict(stale_cqe_call=1, step=4),  # four batches: a descriptor is recycled within the read
    dict(part=1, sub=2, part_error=EIO, ordinal=1),  # the last of part 1's three sub-reads
]


@pytest.mark.parametrize("fault", FAILURES)
def test_a_failed_read_publishes_only_whole_exact_pieces(tmp_path, fault):
    """The direct mode's failure rule (the replacement of the bounce's untouched-slot rule): the drive writes the
    slot itself, so a failed read may leave any unpublished bytes behind, but every piece whose bit a device could
    see must be whole and exact, while the read runs and after it returns (a C++ thread checks the bytes behind
    every bit it sees, as the device would copy them). The faults are the split suite's both-banks errors, a stale
    completion and a failed sub-read, over 16 rows in two banks with the destination poisoned at admission."""
    s = ram_miss_setup(tmp_path, capacity=32, experts=16, mirror_weights=(1.0, 1.0))
    fault = dict(fault)
    step = fault.pop("step", 8)
    experts, slots, ref_slots = list(range(16)), list(range(16)), list(range(16, 32))
    assert read_rows_traced(s.tables, 1, experts, ref_slots)[0] == 1
    split._assert_rows(s, 1, experts, ref_slots)
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=9, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, step=step, poison=True, **fault,
    )
    assert result == 0
    published = sum(bin(int(word) & 0xFF).count("1") for word in masks.view(-1).tolist())
    assert info["differed"] == 0 and info["checked"] == published == record["pieces_published"]


# ---- Refusals ----


def test_the_reader_refuses_shard_tables(tmp_path):
    """The packed path is gone (plan 2026-09-29-hotpath-zero-overhead D4): a table that is not a row-image table is
    refused at open, naming the converter, never read through a bounce buffer."""
    s = ram_miss_setup(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="row image"):
        exl3_ram_miss_tables(s.layout, s.fmt.segment_map(), s.slabs)  # no row_images: shard tables


def test_the_reader_refuses_buffered_reads_of_row_images(tmp_path):
    """Row images are read with O_DIRECT only: a buffered read would copy through the page cache (spec section 3)."""
    s = ram_miss_setup(tmp_path)
    fault = ops._fault_tensor()
    record = torch.zeros(ops._stage_words("exl3"), dtype=torch.int64)
    with pytest.raises(Exception, match="O_DIRECT"):
        ops._host_module("exl3").expert_stream_read_rows_traced(
            *ops._table_args(s.tables, False), 0, torch.tensor([1]), torch.tensor([0]), 8, fault, record, -1)


def _tables_with(s, **fields):
    tables = types.SimpleNamespace(**vars(s.tables))
    for name, value in fields.items():
        setattr(tables, name, value)
    return tables


def _shifted(tensor, index, delta):
    out = tensor.clone()
    out[index] += delta
    return out


def _moved(extents, *, offset=0, cut=0, gap=0):
    """Row (0, 1)'s two parts with both file offsets moved by ``offset``, the cut between them moved by ``cut``, or
    part 1 moved ``gap`` bytes later in the image and the file (shortened to keep it inside the row)."""
    out = extents.clone()
    out[0, 1, :, 1] += offset
    out[0, 1, 0, 2] += cut
    out[0, 1, 1, 1] += cut + gap
    out[0, 1, 1, 2] -= cut + gap
    out[0, 1, 1, 3] += cut + gap
    return out


@pytest.mark.parametrize(
    "edit, message",
    [
        # O_DIRECT would refuse these reads (EINVAL) at run time: refused at open instead.
        (lambda t: dict(extents=_moved(t.extents, offset=256)), "offset and length 512 B aligned"),
        (lambda t: dict(extents=_moved(t.extents, cut=-256)), "offset and length 512 B aligned"),
        (lambda t: dict(slabs=_shifted(t.slabs, (1, 3), 128)), "slab row 512 B aligned"),
        # A table the direct mode could not land exactly: refused when the tables are read.
        # Past the image into its padding: the image-sized slot refuses it before the image check can.
        (lambda t: dict(extents=_shifted(t.extents, (1, 2, 1, 2), 512)), "outside its bounce slot"),
        (lambda t: dict(extents=_shifted(t.extents, (1, 2, 1, 2), -512)), "exactly its image"),
        (lambda t: dict(starts=_shifted(t.starts, (0, 0), 512)), "start every row at 0"),
        (lambda t: dict(segments=_shifted(t.segments, (2, 2), 512)), "tile the image"),
        (lambda t: dict(extents=_moved(t.extents, gap=512)), "tile it in part order"),
    ],
    ids=["offset", "length", "slab", "padding", "short", "start", "segment_gap", "part_gap"],
)
def test_open_refuses_a_table_the_direct_mode_cannot_read(tmp_path, edit, message):
    """Completeness of the refusals: each edit makes one read land wrong or be refused by the kernel, and each must
    fail loudly before any read, naming why. The offset, length and gap edits keep the row's parts on one base
    (tables_from's own check), so each is refused for the reason it names alone."""
    s = ram_miss_setup(tmp_path, capacity=3, mirror_weights=(1.0, 1.0))
    tables = _tables_with(s, **edit(s.tables))
    with pytest.raises(RuntimeError, match=message):
        read_rows_traced(tables, 0, [1], [0])


def test_an_image_whose_names_are_not_512_byte_rows_is_refused(tmp_path):
    """The fake checkpoint's default dimensions give 256-byte w2 rows: no image layout exists (the contract refuses
    it), so the tables are never built."""
    with pytest.raises(ValueError, match="not a multiple of 512"):
        ram_miss_setup(tmp_path, hidden=128, inter=128)


def test_the_tables_refuse_images_opened_for_other_roots(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=3, mirror_weights=(1.0, 1.0))
    images = ri.open_row_images(s.roots, s.layout, s.fmt.segment_map(), str(tmp_path), [0, 1])
    with pytest.raises(ValueError, match="row images are on"):
        exl3_ram_miss_tables(
            s.layout, s.fmt.segment_map(), s.slabs, roots=s.roots[::-1], policy=StaticSplitPolicy((1.0, 1.0)),
            source_root=str(tmp_path), row_images=images,
        )


# ---- The trace ----


def test_pack_stamps_are_publish_times_and_no_pool_is_started(tmp_path):
    """Critical-path bookkeeping (the measurement reads it): in the direct mode a row's first and last publish
    bracket its pieces' publishes and come after the landing that vetted them, the record says no packing workers
    ran, and the byte split has no superset or padding in it: over two batches and two
    drives every byte submitted, completed and useful is an image byte, once."""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(11))[::-1], list(range(11))
    result, record = read_rows_traced(s.tables, 1, experts, slots, piece_stream=True)
    assert result == 1
    assert record["pack_workers"] == 0 and record["piece_stream"] == 1 and record["batches"] == 2
    image = len(experts) * int(s.tables.slot_bytes)
    assert record["useful_bytes"] == record["bytes"] == record["submitted_bytes"] == image
    assert sum(drive["bytes"] for drive in record["drives"]) == image
    checked = 0
    for row in record["row_pack"]:
        cqes = [e["cqe"] for e in record["extent_cqe"] if e["row"] == row["row"]]
        if len(cqes) < len(_sub_reads(s, 1, experts[row["row"]])):
            continue  # past STAGE_TRACE_EXTENTS: the row's extents are not all stamped
        checked += 1
        assert 0 < min(cqes) <= row["start"] <= row["end"] and max(cqes) <= row["end"], (row, cqes)
    assert checked >= 3
    assert record["pieces_published"] == len(experts) * ops.STAGE_PIECES
    split._assert_rows(s, 1, experts, slots)


def _sub_reads(s, row, expert):
    return ops.piece_geometry(s.tables, row, expert)[0]


# ---- The service ----


def _cfg(**overrides):
    return Dsv41Config(**{**{f: getattr(Dsv41Config.from_envs(), f) for f in Dsv41Config.__struct_fields__}, **overrides})


def test_the_service_opens_images_only_with_the_flag_mirrors_o_direct_and_leases(tmp_path):
    """Negative-branch contract of the wiring: off means the shard tables (None), and on is refused without mirror
    dirs (nowhere to read), without O_DIRECT (the readv into the slabs is the point) or without leases (only the lease
    check keeps a direct read from overwriting a slot a GPU copy may still read), naming the missing setting."""
    s = ram_miss_setup(tmp_path, capacity=3)
    mirrors = dict(roots=s.roots, policy=StaticSplitPolicy((1.0,)), source_root=str(tmp_path))
    segments, layers = s.fmt.segment_map(), {0: None, 1: None}
    assert open_service_row_images(_cfg(enable_ram_miss_row_images=False), s.layout, segments, mirrors, True, layers) is None
    with pytest.raises(RuntimeError, match="SGLANG_DSV41_ENABLE_RAM_MISS_LEASES"):
        open_service_row_images(
            _cfg(enable_ram_miss_row_images=True, enable_ram_miss_leases=False), s.layout, segments, mirrors, True, layers
        )
    on = _cfg(enable_ram_miss_row_images=True, enable_ram_miss_leases=True)
    with pytest.raises(RuntimeError, match="SGLANG_MOE_EXPERT_MIRROR_DIRS"):
        open_service_row_images(on, s.layout, segments, {}, True, layers)
    with pytest.raises(RuntimeError, match="uring_direct"):
        open_service_row_images(on, s.layout, segments, mirrors, False, layers)
    images = open_service_row_images(on, s.layout, segments, mirrors, True, layers)
    assert sorted(images.paths) == [0, 1] and images.roots == s.roots


@pytest.mark.parametrize(
    "workers, images, refused",
    [(0, False, True), (0, True, False), (2, False, False)],
    ids=["no_publisher", "images_publish_inline", "workers"],
)
def test_the_service_refuses_piece_streaming_without_a_publisher_unless_it_reads_row_images(workers, images, refused):
    """Negative-branch contract of the service gate: with two-phase and leases on, piece streaming still needs a
    publisher, packing workers or the direct mode's own; without either it is refused, not run on the inline packer."""
    cfg = _cfg(
        enable_ram_miss_piece_stream=True, enable_ram_miss_two_phase=True, enable_ram_miss_leases=True,
        ram_miss_pack_workers=workers,
    )
    if refused:
        with pytest.raises(RuntimeError, match="PACK_WORKERS > 0"):
            check_piece_stream(cfg, row_images=images)
    else:
        check_piece_stream(cfg, row_images=images)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
