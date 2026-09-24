"""The C++ row reader's direct mode (CPU): row images read with one O_DIRECT readv per read straight into the pinned
slab rows (SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES, plan 2026-09-24-dsv41-row-images Part B).

Every test of the split, thread and piece-stream suites that builds its tables with ``ram_miss_setup`` runs again
here with the tables reading row images through O_DIRECT (the ``row_images`` fixture below sets the mode), so a
promise those suites make about the reader and the tier is checked for the direct mode without a second copy of the
test. Their byte oracle is still the checkpoint read by Exl3ShardRowSource, the bounce path's. The tests written out
below are the ones only the direct mode can break: its equality with the bounce path over the same requests, a
resubmission that resumes inside the iovec list, the refusals at open, and the service's wiring of the flag.
"""

import errno
import inspect
import os
import resource
import types

import pytest
import torch

import test_exl3_ram_miss_piece_stream as piece_stream
import test_exl3_ram_miss_split as split
import test_exl3_ram_miss_thread as thread
import test_exl3_ram_miss_two_phase as two_phase
import test_exl3_ram_miss_two_phase_victim as two_phase_victim
from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, read_rows_pieces, read_rows_sqes, read_rows_traced
from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables, open_service_row_images
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=240, suite="base-a-test-cpu")

from test_exl3_ram_miss_thread import hang_guard  # noqa: E402,F401  (autouse fixture of the reused thread tests)
from test_exl3_ram_miss_two_phase import running  # noqa: E402,F401  (fixture of the reused two-phase tests)

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft != resource.RLIM_INFINITY and _soft < 16384:
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384 if _hard == resource.RLIM_INFINITY else min(16384, _hard), _hard))

EIO = errno.EIO
PAGE = 4096
SETUP_USERS = (split, thread, piece_stream, two_phase, two_phase_victim)


def _o_direct_or_skip(path):
    try:
        os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
    except OSError as error:
        pytest.skip(f"{path} does not support O_DIRECT: {error}")


def _images_setup(tmp_path, **kwargs):
    s = ram_miss_setup(tmp_path, row_images=True, **kwargs)
    _o_direct_or_skip(s.tables.paths[0])
    return s


@pytest.fixture(params=[False, True], ids=["img", "img_pieces"])
def row_images(request, monkeypatch):
    """Every ``ram_miss_setup`` a reused test calls builds row-image tables, every read of them is O_DIRECT (the
    production mode; a test's ``direct=False`` names the bounce path's buffered default), and in the piece-streaming
    mode every reader streams pieces (which the direct mode does without packing workers)."""
    yield from _row_images(request.param, monkeypatch)


@pytest.fixture
def row_images_without_pieces(monkeypatch):
    yield from _row_images(False, monkeypatch)


@pytest.fixture
def row_images_with_pieces(monkeypatch):
    yield from _row_images(True, monkeypatch)


def _row_images(pieces, monkeypatch):
    for module in SETUP_USERS:
        monkeypatch.setattr(module, "ram_miss_setup", _images_setup)
    monkeypatch.setattr(split, "PIECE_STREAM", pieces)
    fault_tensor, table_args, host_init = ops._fault_tensor, ops._table_args, Exl3RamMissHost.__init__

    def faults_with_pieces(**faults):
        if pieces:
            faults.setdefault("piece_stream", True)
        return fault_tensor(**faults)

    def direct_table_args(tables, direct):
        return table_args(tables, direct or tables.row_images)

    def direct_host(self, tables, *args, direct, **kwargs):
        host_init(self, tables, *args, direct=direct or tables.row_images, **kwargs)

    monkeypatch.setattr(ops, "_fault_tensor", faults_with_pieces)
    monkeypatch.setattr(ops, "_table_args", direct_table_args)
    monkeypatch.setattr(Exl3RamMissHost, "__init__", direct_host)
    yield pieces


def _clone(test, fixture):
    clone = types.FunctionType(test.__code__, test.__globals__, test.__name__, test.__defaults__, test.__closure__)
    clone.__kwdefaults__ = test.__kwdefaults__
    clone.__dict__.update({k: list(v) if k == "pytestmark" else v for k, v in test.__dict__.items()})
    return pytest.mark.usefixtures(fixture)(clone)


# Reused tests whose assertions are about the bounce path itself, each with the reason the direct mode differs.
NOT_REUSED = {
    # Shard geometry: the tables' files are image files, their extents tile a page-rounded image, not a superset.
    "test_tables_describe_every_row": "asserts the shard tables' shape and page-rounded slot",
    "test_the_last_row_of_a_shard_is_clamped_at_end_of_file": "images have no end-of-shard clamp",
    "test_open_refuses_a_source_file_that_is_not_the_size_the_tables_were_built_from": "truncates a shard",
    "test_open_refuses_a_mirror_copy_of_the_wrong_size_naming_both_files": "truncates a mirrored shard",
    "test_open_accepts_mirror_copies_of_the_source_size": "reads the mirrored shards' sizes",
    "test_direct_reads_split_like_the_python_row_source": "every read here is direct already",
    "test_bad_arguments_raise": "validates arguments in Python, identical for both modes",
    "test_tables_keep_their_slabs_alive": "the keepalive tuple is built by the same code in both modes",
    # Tables the test builds or edits itself as shard tables (the images' refusals are test_open_refuses_* below).
    "test_the_last_row_of_a_shard_reads_correctly_natively_and_eagerly": "builds its own shard checkpoint",
    "test_a_table_whose_parts_disagree_on_the_rows_base_is_refused": "builds its own shard tables",
    "test_a_row_that_needs_bytes_the_file_lacks_fails_in_both_readers": "builds its own shard tables",
    "test_a_row_with_nothing_to_read_fails_instead_of_packing_stale_bytes": (
        "zeroes a row's parts: image tables refuse that at construction (test_open_refuses_a_table_*)"
    ),
    "test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing": (
        "shortens a part: image tables refuse that at construction (test_open_refuses_a_table_*)"
    ),
    # Packing: the direct mode has no bounce, no copy and no packing pool.
    "test_a_read_records_its_stages_and_bytes": "useful < bytes holds for supersets, an image read is all useful",
    "test_a_read_over_several_batches_sums_them": "as above",
    "test_the_byte_split_of_a_clean_read": "as above",
    "test_a_row_packs_while_another_rows_read_is_still_outstanding": "asserts a pack span of the pack_delay fault",
    "test_a_bank_is_not_reused_until_every_row_in_it_has_packed": "a bank holds no memory: nothing to wait for",
    "test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack": "no blocking pack to overtake",
    # The bounce's failure rule: a slot is untouched or a whole row, since nothing reaches it before its row landed.
    # The drive writes a direct-mode slot as each read lands, so after a failure an unpublished slot may hold part
    # of a row (and the poison fault's fill); the caller releases it. What must hold instead, that every published
    # piece is whole and exact, is test_a_failed_read_publishes_only_whole_exact_pieces below.
    "test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one": "the bounce's untouched-slot rule",
    "test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor": "as above",
    "test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring": "as above",
    "test_u2_a_failed_sub_read_leaves_its_pieces_unvetted_and_its_row_unpacked": "as above",
    # Piece streaming with packing workers, or the bounce's geometry.
    "test_u1_geometry_of_every_row_of_a_random_layout": "synthetic shard tables",
    "test_u1_geometry_of_every_row_of_a_real_layout": "shard geometry",
    "test_u1_the_geometry_is_what_the_reader_reads": "compares SQEs to bounce offsets",
    "test_u10_flag_off_issues_todays_sqes_and_credit_and_packs_the_same_bytes": "compares with the shard SQEs",
    "test_the_reader_refuses_piece_streaming_without_packing_workers": "the direct mode needs none",
    "test_the_tier_refuses_the_flag_without_packing_workers": "the direct mode needs none",
    "test_the_reader_refuses_a_slab_row_base_that_is_not_128_byte_aligned": "the direct mode refuses 512 first",
    "test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name": "needs three mirror parts of shards",
}

# As in test_exl3_ram_miss_pack_workers: a part's single read faulted or counted, which with piece streaming is a
# sub-read of another size; the piece-stream suite's sub-read counterparts run here in the pieces mode.
ONE_READ_PER_PART = {
    "test_a_short_read_resubmits_its_own_extent_under_reversed_completions",
    "test_a_short_read_resubmits_only_its_own_extent",
    "test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful",
    "test_a_short_read_in_either_bank_resubmits_only_its_own_extent",
    "test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried",
}


def _reuse(module, fixture_of, prefix=""):
    for name, test in vars(module).items():
        if not (name.startswith("test_") and inspect.isfunction(test)) or name in NOT_REUSED:
            continue
        if test.__module__ != module.__name__:
            continue  # a test the module itself reuses from another one is reused from its own module
        if "ram_miss_setup(" not in inspect.getsource(test) and "_host(" not in inspect.getsource(test) and (
            "_tier(" not in inspect.getsource(test)
        ):
            continue
        fixture = fixture_of(name)
        if fixture is not None:
            globals()[prefix + name] = _clone(test, fixture)


_reuse(split, lambda name: "row_images_without_pieces" if name in ONE_READ_PER_PART else "row_images")
# The thread suite's tiers run without lease mode, which a piece-streaming tier refuses per request.
_reuse(thread, lambda name: "row_images_without_pieces")
# The piece-stream suite sets the flag itself; its flag-off tests (U10, flag_off_*) run in the mode without it.
_reuse(piece_stream, lambda name: "row_images_without_pieces" if "flag_off" in name else "row_images_with_pieces")


# ---- The direct mode lands the same bytes as the bounce path ----


def _all_slabs(s):
    return {(layer, name): s.slabs[layer][name].clone() for layer in s.slabs for name in EXL3_STREAMED_NAMES}


def _sentinel_all(s):
    for layer in s.slabs:
        for name in EXL3_STREAMED_NAMES:
            s.slabs[layer][name].view(torch.uint8).fill_(0x5A)


@pytest.mark.parametrize(
    "weights", [None, (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, 1.0)], ids=["one", "halves", "first", "second", "3to1"]
)
@pytest.mark.parametrize("pieces", [False, True], ids=["rows", "pieces"])
def test_the_direct_mode_leaves_every_slab_byte_as_the_bounce_path_does(tmp_path, weights, pieces):
    """Derived property: an image read is a re-layout of the checkpoint row, so for any request the slabs after a
    direct read equal the slabs after a bounce read, byte for byte, including every slot the request does not name.
    The requests are the split suite's shapes: one row, rows over several batches (11 rows, 8 + 3), one row per
    batch, and every mirror split; the image's last part ends inside a page, so the padding is never read."""
    for name in ("bounce", "direct"):
        (tmp_path / name).mkdir()
    bounce = ram_miss_setup(tmp_path / "bounce", capacity=12, experts=12, mirror_weights=weights, hidden=256, inter=256)
    direct = _images_setup(tmp_path / "direct", capacity=12, experts=12, mirror_weights=weights)
    assert direct.tables.slot_bytes % PAGE != 0  # the image ends inside its last page
    requests = [(0, [4], [2], 8), (1, list(range(11))[::-1], [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4], 8), (0, [4, 1, 3], [2, 0, 1], 1)]
    for s in (bounce, direct):
        _sentinel_all(s)
    for row, experts, slots, step in requests:
        extra = dict(piece_stream=True, pack_workers=2) if pieces else {}
        got = [read_rows_traced(s.tables, row, experts, slots, direct=s is direct, step=step, **extra)[0] for s in (bounce, direct)]
        assert got == [1, 1]
    want, have = _all_slabs(bounce), _all_slabs(direct)
    for key in want:
        assert same_bytes(want[key], have[key]), key


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
    s = _images_setup(tmp_path, capacity=6)
    assert s.tables.segments[:, 2].tolist() == IMAGE_ROW_STARTS  # the boundaries the cases are placed around
    experts, slots = [3, 0, 5], [0, 1, 2]
    result, sqes, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, direct=True, part=0, part_short=short, ordinal=1
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
    s = _images_setup(tmp_path, capacity=6)
    experts, slots = [2, 4], [1, 0]
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=True, generation=7, piece_stream=True, part=0, sub=0, part_short=3 * 512,
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
    the same reader lands byte-exact in the same slots, which it could not if a straggling DMA or a stale
    descriptor from the failed read were still live; the failed rows' untouched slots keep their sentinel or hold
    whole landed rows only."""
    s = _images_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 1.0))
    experts, slots = [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]
    for slot in slots:
        split._sentinel(s, 1, slot)
    extra = dict(piece_stream=True) if pieces else {}
    results = ops.read_rows_with_fault(
        s.tables, 1, experts, slots, [6, 7, 0], [6, 7, 0], direct=True, part=1, part_error=EIO, ordinal=2,
        max_outstanding=2, **extra,
    )
    assert results == (0, 1)
    split._assert_rows(s, 1, [6, 7, 0], [6, 7, 0])


FAILURES = [
    dict(cqe_error=EIO, cqe_call=1),
    dict(cqe_error=EIO, cqe_call=13),
    dict(part=1, part_error=EIO, ordinal=11),
    dict(part=0, part_error=EIO, ordinal=3, hold_ordinal=12),
    dict(submit_error=EIO, submit_call=3, submit_first=True, max_outstanding=4),
    dict(submit_error=EIO, submit_call=3, submit_first=False, max_outstanding=4),
    dict(cqe_error=EIO, cqe_call=9, max_outstanding=3, reverse_cqes=True),
    dict(stale_cqe_call=1),
    dict(part=1, sub=3, part_error=EIO, ordinal=1),
]


@pytest.mark.parametrize("fault", FAILURES)
def test_a_failed_read_publishes_only_whole_exact_pieces(tmp_path, fault):
    """The direct mode's failure rule (the replacement of the bounce's untouched-slot rule): the drive writes the
    slot itself, so a failed read may leave any unpublished bytes behind, but every piece whose bit a device could
    see must be whole and exact, while the read runs and after it returns (a C++ thread checks the bytes behind
    every bit it sees, as the device would copy them). The faults are the split suite's both-banks errors, a stale
    completion and a failed sub-read, over 16 rows in two banks with the destination poisoned at admission."""
    s = _images_setup(tmp_path, capacity=32, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots, ref_slots = list(range(16)), list(range(16)), list(range(16, 32))
    assert read_rows_traced(s.tables, 1, experts, ref_slots, direct=True)[0] == 1
    split._assert_rows(s, 1, experts, ref_slots)
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=True, generation=9, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, step=8, poison=True, **fault,
    )
    assert result == 0
    published = sum(bin(int(word) & 0xFF).count("1") for word in masks.view(-1).tolist())
    assert info["differed"] == 0 and info["checked"] == published == record["pieces_published"]


# ---- Refusals ----


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
    s = _images_setup(tmp_path, capacity=3, mirror_weights=(1.0, 1.0))
    tables = _tables_with(s, **edit(s.tables))
    with pytest.raises(RuntimeError, match=message):
        read_rows_traced(tables, 0, [1], [0], direct=True)


def test_an_image_whose_names_are_not_512_byte_rows_is_refused(tmp_path):
    """The fake checkpoint's default dimensions give 256-byte w2 rows: no image layout exists (the contract refuses
    it), so the tables are never built."""
    with pytest.raises(ValueError, match="not a multiple of 512"):
        ram_miss_setup(tmp_path, row_images=True, hidden=128, inter=128)


def test_the_tables_refuse_images_opened_for_other_roots(tmp_path):
    s = _images_setup(tmp_path, capacity=3, mirror_weights=(1.0, 1.0))
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
    ran (asking for three starts none), and useful_bytes counts every image byte that landed in a slab."""
    s = _images_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    experts, slots = [3, 0, 5, 1], [0, 1, 2, 3]
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=True, piece_stream=True, pack_workers=3)
    assert result == 1
    assert record["pack_workers"] == 0 and record["piece_stream"] == 1
    assert record["useful_bytes"] == len(experts) * int(s.tables.slot_bytes) == record["bytes"]
    for row in record["row_pack"]:
        cqes = [e["cqe"] for e in record["extent_cqe"] if e["row"] == row["row"]]
        assert 0 < min(cqes) <= row["start"] <= row["end"] and max(cqes) <= row["end"], (row, cqes)
    assert record["pieces_published"] == len(experts) * ops.STAGE_PIECES
    split._assert_rows(s, 1, experts, slots)


# ---- The service ----


def _cfg(**overrides):
    return Dsv41Config(**{**{f: getattr(Dsv41Config.from_envs(), f) for f in Dsv41Config.__struct_fields__}, **overrides})


def test_the_service_opens_images_only_with_the_flag_mirrors_and_o_direct(tmp_path):
    """Negative-branch contract of the wiring: off means the shard tables (None), and on is refused without mirror
    dirs (nowhere to read) or without O_DIRECT (the readv into the slabs is the point), naming the missing setting."""
    s = _images_setup(tmp_path, capacity=3)
    mirrors = dict(roots=s.roots, policy=StaticSplitPolicy((1.0,)), source_root=str(tmp_path))
    segments, layers = s.fmt.segment_map(), {0: None, 1: None}
    assert open_service_row_images(_cfg(enable_ram_miss_row_images=False), s.layout, segments, mirrors, True, layers) is None
    on = _cfg(enable_ram_miss_row_images=True)
    with pytest.raises(RuntimeError, match="SGLANG_MOE_EXPERT_MIRROR_DIRS"):
        open_service_row_images(on, s.layout, segments, {}, True, layers)
    with pytest.raises(RuntimeError, match="uring_direct"):
        open_service_row_images(on, s.layout, segments, mirrors, False, layers)
    images = open_service_row_images(on, s.layout, segments, mirrors, True, layers)
    assert sorted(images.paths) == [0, 1] and images.roots == s.roots


def test_the_flag_defaults_to_off():
    assert envs.SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES.get() is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
