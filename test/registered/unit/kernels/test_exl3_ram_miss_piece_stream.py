"""Piece streaming in the C++ row reader and tier (CPU): sub-reads, piece geometry, per-piece vetting, packing
and publishing.

SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM reads each part of a row as up to 4 page-aligned sub-reads and cuts the
row's needed bytes into 8 pieces. Each piece is vetted once the sub-reads it depends on have landed, packed by its
own job, and published by the reader's owner into the readiness words (lease area P) of the lanes that name its row,
with a generation-checked compare-and-swap. The tier initialises those words at reservation. U1 (geometry), U2
(order under reordered completions), U3 (a bit implies its bytes), U6 (a double publish), U8 (the publish primitive)
and U10 (the flag off leaves the reader as it was) are the plan's names for these tests.
"""

import random
import threading
import time
import types
from types import SimpleNamespace

import pytest
import torch

import test_exl3_ram_miss_split as split
import test_exl3_ram_miss_two_phase as two_phase
from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import (
    Exl3RamMissHost,
    new_page,
    page_word,
    piece_geometry,
    piece_word,
    publish_piece,
    read_rows_pieces,
    read_rows_sqes,
    read_rows_traced,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

from test_exl3_ram_miss_two_phase import hang_guard, running  # noqa: E402,F401  (the reused two-phase tests' fixtures)

PAGE = 4096
SUB_READS = 4
PIECES = ops.STAGE_PIECES
ALIGN = 128


def _round_up(value, to):
    return -(-value // to) * to


def _expected_sub_reads(part):
    """Plan section 4.1, written out independently of the C++: len_k = round_up(ceil(len / 4), page)."""
    file, offset, length, dest = part
    if length <= 0:
        return []
    len_k = _round_up(-(-length // SUB_READS), PAGE)
    return [(file, offset + at, min(len_k, length - at), dest + at) for at in range(0, length, len_k)]


def _row_sub_reads(tables, row, expert):
    out = []
    for p in range(tables.extents.shape[2]):
        out += [(p, k, sub) for k, sub in enumerate(_expected_sub_reads(tables.extents[row, expert, p].tolist()))]
    return out


def _assert_geometry(tables, row, expert):
    """U1 for one row: sub-reads tile each part, pieces partition the needed bytes on 128 B cuts that are the
    sub-read boundaries, and each dependency mask is exactly the sub-reads the piece's bytes come from."""
    got = piece_geometry(tables, row, expert)
    assert got is not None, (row, expert)
    sub_reads, pieces = got
    expected = _row_sub_reads(tables, row, expert)
    assert [(s["part"], s["k"], (s["file"], s["offset"], s["length"], s["dest"])) for s in sub_reads] == expected
    for p in range(tables.extents.shape[2]):
        file, offset, length, dest = tables.extents[row, expert, p].tolist()
        mine = [s for s in sub_reads if s["part"] == p]
        assert len(mine) <= SUB_READS
        for s in mine:
            assert s["offset"] % PAGE == 0 and s["dest"] % PAGE == 0 and s["length"] % PAGE == 0 and s["length"] > 0
        # They tile the part: contiguous, in order, covering it exactly.
        assert [s["dest"] for s in mine] == [dest + sum(m["length"] for m in mine[:i]) for i in range(len(mine))]
        assert sum(s["length"] for s in mine) == max(length, 0)
    start = int(tables.starts[row, expert])
    segments = tables.segments.tolist()
    for i, (_name, dst, src, nbytes) in enumerate(segments):
        runs = [piece["runs"][i] for piece in pieces]
        # A partition of the segment, in piece order.
        assert runs[0][0] == dst and runs[-1][1] == dst + nbytes
        for (lo, hi), (next_lo, _) in zip(runs, runs[1:]):
            assert lo <= hi == next_lo
        for j, (lo, hi) in enumerate(runs):
            if dst < lo < dst + nbytes:
                assert lo % ALIGN == 0, (i, j, lo)
                # The cut is sub-read j's boundary, rounded down in the segment's destination coordinates.
                boundary = sub_reads[j]["dest"] - start - src + dst
                assert lo == max(dst, boundary // ALIGN * ALIGN), (i, j)
    for j, piece in enumerate(pieces):
        deps = 0
        for i, (_name, dst, src, nbytes) in enumerate(segments):
            lo, hi = piece["runs"][i]
            if lo == hi:
                continue
            a, b = start + src + lo - dst, start + src + hi - dst  # the run's bytes in the bounce slot
            for k, s in enumerate(sub_reads):
                if a < s["dest"] + s["length"] and s["dest"] < b:
                    deps |= 1 << k
        assert piece["deps"] == deps, (row, expert, j)
        if j >= len(sub_reads):
            assert deps == 0 and all(lo == hi for lo, hi in piece["runs"])
        elif j > 0:
            # A cut moves back by less than 128 B, never a whole page: a piece reads its own sub-read and at most
            # the tail of the one before.
            assert deps & ~((1 << j) | (1 << (j - 1))) == 0, (j, bin(deps))
    return sub_reads, pieces


# ---- U1: geometry ----


def _synthetic_tables(seed, *, experts=8, parts=2):
    """Random rows: random starts, random segment sizes (so cuts fall anywhere), random mirror weights including a
    zero-length part and a one-page last part, and a last row clamped at end of file."""
    rng = random.Random(seed)
    names = 6
    segments, src, dst = [], 0, [0] * names
    for _ in range(9):
        name = rng.randrange(names)
        nbytes = rng.randrange(2, 70_000, 2)
        segments.append((name, dst[name], src, nbytes))
        dst[name] += nbytes
        src += nbytes + rng.choice((0, 4, 6))
    need_end = max(s + n for _, _, s, n in segments)
    layers = 2
    rows = layers * experts
    extents = torch.zeros((layers, experts, parts, 4), dtype=torch.int64)
    starts = torch.zeros((layers, experts), dtype=torch.int64)
    slot_bytes, base = 0, 0
    for r in range(rows):
        start = rng.randrange(PAGE)
        pages = _round_up(start + need_end, PAGE) // PAGE
        if parts == 1:
            split_pages = [pages]
        else:
            # Row 0 is split evenly, so the layout always has a row with every sub-read.
            first = pages // 2 if r == 0 else rng.choice([0, pages, 1, pages - 1, rng.randrange(pages + 1)])
            split_pages = [first, pages - first]
        dest = 0
        for p, n in enumerate(split_pages):
            extents[r // experts, r % experts, p] = torch.tensor([p, base + dest, n * PAGE, dest])
            dest += n * PAGE
        starts[r // experts, r % experts] = start
        slot_bytes = max(slot_bytes, pages * PAGE)
        last_end = base + start + need_end
        base += pages * PAGE
    return SimpleNamespace(
        extents=extents,
        starts=starts,
        file_sizes=torch.tensor([last_end] * parts, dtype=torch.int64),  # the last row's tail is past EOF
        segments=torch.tensor(segments, dtype=torch.int64),
        slabs=torch.zeros((layers, names), dtype=torch.int64),
        row_bytes=torch.tensor([max(d, 1) for d in dst], dtype=torch.int64),
        paths=[f"/nonexistent/{p}" for p in range(parts)],
        source_paths=["/nonexistent/source"] * parts,
        slot_bytes=slot_bytes,
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("parts", [1, 2])
def test_u1_geometry_of_every_row_of_a_random_layout(seed, parts):
    tables = _synthetic_tables(seed, parts=parts)
    counts = set()
    for row in range(tables.extents.shape[0]):
        for expert in range(tables.extents.shape[1]):
            sub_reads, _ = _assert_geometry(tables, row, expert)
            counts.add(len(sub_reads))
    assert max(counts) == SUB_READS * parts  # the layout has rows long enough for every sub-read


@pytest.mark.parametrize(
    "weights, dims",
    [(None, {}), ((1.0, 1.0), {}), ((0.0, 1.0), {}), ((1.0, 1.0), dict(hidden=256, inter=512)),
     ((3.0, 1.0), dict(hidden=256, inter=512)), (None, dict(hidden=256, inter=512))],
    ids=["one_part", "halves", "zero_first_part", "large_halves", "small_last_part", "large_one_part"],
)
def test_u1_geometry_of_every_row_of_a_real_layout(tmp_path, weights, dims):
    s = ram_miss_setup(tmp_path, capacity=2, mirror_weights=weights, **dims)
    for row in range(s.tables.extents.shape[0]):
        for expert in range(s.tables.extents.shape[1]):
            _assert_geometry(s.tables, row, expert)


def test_u1_the_geometry_is_what_the_reader_reads(tmp_path):
    """The exported geometry is the reader's own: a piece-streaming read issues exactly those sub-reads."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [3, 0, 5], [0, 1, 2]
    result, log, info, _ = read_rows_sqes(s.tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2)
    assert result == 1
    expected = []
    for i, expert in enumerate(experts):
        for sub in piece_geometry(s.tables, 1, expert)[0]:
            expected.append((sub["file"], sub["offset"], sub["length"], i * s.tables.slot_bytes + sub["dest"]))
    assert log == expected
    split._assert_rows(s, 1, experts, slots)


# ---- U2: pieces are vetted in dependency order, whatever order completions arrive in ----


def _landings(record):
    return sorted(seq for row in record["pieces"] for seq in row["sub_seq"] if seq)


@pytest.mark.parametrize("workers, chunks", [(1, 1), (3, 3)])
def test_u2_pieces_are_vetted_in_dependency_order_under_reversed_cqes_and_a_held_sub_read(tmp_path, workers, chunks):
    """Completions are processed back to front, and sub-read 1 of part 0 of row 0 lands last of all. Every piece
    must be vetted right after its last dependency lands (no landing in between), never before, and the pieces that
    depend on the held sub-read after every other piece. Rows are still packed whole, byte-exact."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=workers, pack_split=chunks,
        reverse_cqes=True, hold_ordinal=0, part=0, sub=1, poison=True,
    )
    assert result == 1 and record["piece_stream"] == 1
    split._assert_rows(s, 1, experts, slots)
    assert record["pieces_vetted"] == PIECES * len(experts)
    landings = _landings(record)
    order = []
    for row, entry in zip(record["pieces"], (piece_geometry(s.tables, 1, e) for e in experts)):
        sub_reads, pieces = entry
        assert len(sub_reads) == 2 * SUB_READS  # the rows are long enough for 8 sub-reads
        assert all(row["sub_seq"][k] > 0 for k in range(len(sub_reads)))
        for j, piece in enumerate(pieces):
            deps = [k for k in range(len(sub_reads)) if piece["deps"] >> k & 1]
            assert deps, j  # every piece of these rows has bytes
            last = max(row["sub_seq"][k] for k in deps)
            seq = row["seq"][j]
            assert seq > last, (row["row"], j)
            assert not [x for x in landings if last < x < seq], (row["row"], j)  # vetted as its last dependency landed
            order.append((seq, row["row"], j, deps))
    held = next(k for k, sub in enumerate(piece_geometry(s.tables, 1, experts[0])[0]) if (sub["part"], sub["k"]) == (0, 1))
    row0 = record["pieces"][0]
    assert row0["sub_seq"][held] == landings[-1]  # the held sub-read landed last of the whole read
    dependent = [j for seq, r, j, deps in order if r == 0 and held in deps]
    assert dependent and all(
        row0["seq"][j] > seq for j in dependent for seq, r, i, deps in order if held not in deps or r != 0
    )
    # So row 0's pieces were not vetted in index order: a later piece became ready before an earlier one.
    by_seq = [j for _, r, j, _ in sorted(order) if r == 0]
    assert by_seq != sorted(by_seq)
    # And the clock agrees: the held sub-read's dependents were vetted at its (later) reap.
    assert all(row0["cqe"][j] == max(row0["cqe"]) for j in dependent)
    assert all(row0["cqe"][j] < max(row0["cqe"]) for j in range(PIECES) if j not in dependent)


def test_u2_a_failed_sub_read_leaves_its_pieces_unvetted_and_its_row_unpacked(tmp_path):
    """Sub-read 3 of part 1 of row 1 fails with EIO. The read fails, the row is never packed, and no piece that
    depends on the sub-read is vetted."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2, part=1, sub=3, ordinal=1,
        part_error=5,
    )
    assert result == 0
    assert split._untouched(s, 1, 1)
    sub_reads, pieces = piece_geometry(s.tables, 1, experts[1])
    failed = next(k for k, sub in enumerate(sub_reads) if (sub["part"], sub["k"]) == (1, 3))
    row1 = record["pieces"][1]
    assert row1["sub_seq"][failed] == 0
    assert all(row1["seq"][j] == 0 for j, piece in enumerate(pieces) if piece["deps"] >> failed & 1)


# ---- Publishing: U2's order, U3 (a bit implies its bytes), U6 (twice), U8 (the primitive) ----

FULL = 0xFF
GEN = (7 << 32) | 12345  # a request generation: an epoch over a sequence number
MASK64 = (1 << 64) - 1


def _words(masks):
    return [int(word) & MASK64 for word in masks[:, 0]]


@pytest.mark.parametrize("workers, chunks", [(1, 1), (3, 3)])
def test_u2_pieces_publish_in_dependency_order_under_reversed_cqes_and_a_held_sub_read(tmp_path, workers, chunks):
    """The same read as U2's, now publishing. Every piece is published after it is vetted, the pieces that depend on
    the held sub-read after every other piece of the read, and each row's readiness word ends full under its
    generation. The reversal is shown to have taken effect, not assumed: in a row that was not held, a later sub-read
    landed before an earlier one."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=workers,
        pack_split=chunks, reverse_cqes=True, hold_ordinal=0, part=0, sub=1, poison=True,
    )
    assert result == 1 and info["refused"] == 0 and record["piece_publish_refused"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    assert record["pieces_published"] == PIECES * len(experts)
    split._assert_rows(s, 1, experts, slots)
    assert any(
        row["sub_seq"][later] < row["sub_seq"][earlier]
        for row in record["pieces"][1:]
        for earlier in range(PIECES)
        for later in range(earlier + 1, PIECES)
    ), "no row that was not held landed its sub-reads out of order: the reversal did nothing"
    for row in record["pieces"]:
        assert all(row["publish"][j] > row["seq"][j] > 0 for j in range(PIECES)), row
    sub_reads, pieces = piece_geometry(s.tables, 1, experts[0])
    held = next(k for k, sub in enumerate(sub_reads) if (sub["part"], sub["k"]) == (0, 1))
    dependent = [j for j, piece in enumerate(pieces) if piece["deps"] >> held & 1]
    row0 = record["pieces"][0]
    others = [row["publish"][j] for row in record["pieces"] for j in range(PIECES) if row["row"] or j not in dependent]
    assert dependent and min(row0["publish"][j] for j in dependent) > max(others)
    by_publish = [j for _, j in sorted((row0["publish"][j], j) for j in range(PIECES))]
    assert by_publish != list(range(PIECES)) and record["pieces_out_of_order"] >= 1


@pytest.mark.parametrize("workers, chunks", [(2, 1), (3, 3)])
def test_u3_every_bit_a_reader_can_see_names_bytes_already_stored(tmp_path, workers, chunks):
    """A thread polls each row's readiness word while the read runs (as the device will) and, for every bit it sees,
    compares that piece's destination bytes with a reference copy. The slabs start as a sentinel, the bounce is
    poisoned and every piece's copy is slow, so a bit published before its job stored the bytes (at dispatch, say)
    is seen with the sentinel behind it."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    assert read_rows_traced(s.tables, 1, experts, ref_slots, direct=False)[0] == 1
    split._assert_rows(s, 1, experts, ref_slots)
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, pack_workers=workers, pack_split=chunks, pack_delay_ns=10_000_000, poison=True,
    )
    assert result == 1 and info["refused"] == 0
    assert info["checked"] == PIECES * len(experts) and info["differed"] == 0
    assert info["early"] > 0  # the checker saw bits while the read was still running
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    split._assert_rows(s, 1, experts, slots)


def test_u6_a_piece_published_twice_is_refused_and_fails_the_read(tmp_path):
    """A re-dispatched piece would be published a second time. The readiness word refuses it, the refusal is
    counted, and the read fails rather than reporting a row whose piece was handled twice."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=2, publish_twice=3
    )
    assert result == 0
    assert info["refused"] == 1 and record["piece_publish_refused"] == 1
    # The refused attempt changed nothing: every word still carries the generation, and only bits once each.
    assert all(word >> 8 == GEN for word in _words(masks))
    assert record["pieces_published"] == sum(bin(word & FULL).count("1") for word in _words(masks))


def test_u8_the_publish_primitive_refuses_another_generation_and_a_set_bit():
    """The owner's compare-and-swap, directly: the single-threaded reader never meets a stale publisher, so the
    two refusals are tested here. A plain fetch_or passes neither."""
    other = GEN + 1
    for word in (piece_word(other), piece_word(other, 0x03)):
        assert publish_piece(word, GEN, 0x04) == (False, word)
    for word in (piece_word(GEN, 0x04), piece_word(GEN, FULL)):
        assert publish_piece(word, GEN, 0x04) == (False, word)
    assert publish_piece(piece_word(GEN), GEN, 0x04) == (True, piece_word(GEN, 0x04))
    assert publish_piece(piece_word(GEN, 0x81), GEN, 0x04) == (True, piece_word(GEN, 0x85))
    # The generation is 56 bits: the top one sets the word's sign bit, and nothing above it is compared.
    top = (1 << 56) - 1
    assert publish_piece(piece_word(top), top, 0x80) == (True, piece_word(top, 0x80))
    assert publish_piece(piece_word(top), top | 1 << 60, 0x80) == (True, piece_word(top, 0x80))


def test_a_sub_read_that_ends_short_leaves_its_pieces_unvetted_unpublished_and_fails_the_read(tmp_path):
    """Sub-read 2 of part 0 of row 1 ends one page in, as a file ending there would: it retires with fewer bytes than
    the pieces that depend on it need. Those pieces are never vetted or published and the read fails; without the
    per-piece coverage check the bounce's poison behind them would be packed and published."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    sub_reads, pieces = piece_geometry(s.tables, 1, experts[1])
    short = next(k for k, sub in enumerate(sub_reads) if (sub["part"], sub["k"]) == (0, 2))
    assert sub_reads[short]["length"] > PAGE
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=2, part=0, sub=2,
        ordinal=1, part_short=PAGE, short_is_eof=True, poison=True,
    )
    assert result == 0 and info["refused"] == 0
    row1, bits = record["pieces"][1], _words(masks)[1] & FULL
    assert row1["sub_seq"][short] > 0  # it landed, short
    dependent = [j for j, piece in enumerate(pieces) if piece["deps"] >> short & 1]
    assert dependent and all(row1["seq"][j] == 0 and not bits >> j & 1 for j in dependent)


# ---- U10: with the flag off the reader is today's ----


def _baseline_sqes(tables, row, experts):
    """Today's SQE stream for a read whose credit never binds: one SQE per nonzero part, row by row in request order,
    part by part, into bounce slot bank * 8 + row-in-batch."""
    out = []
    for i, expert in enumerate(experts):
        slot = (i // ops.BOUNCE_ROWS) % 2 * ops.BOUNCE_ROWS + i % ops.BOUNCE_ROWS
        for file, offset, length, dest in tables.extents[row, expert].tolist():
            if length > 0:
                out.append((file, offset, length, slot * tables.slot_bytes + dest))
    return out


def _merged(log):
    """Contiguous SQEs (same file, adjacent in the file and in the bounce) joined into one range."""
    out = []
    for file, offset, length, bounce in sorted(log, key=lambda e: e[3]):
        if out and out[-1][0] == file and out[-1][1] + out[-1][2] == offset and out[-1][3] + out[-1][2] == bounce:
            out[-1] = (file, out[-1][1], out[-1][2] + length, out[-1][3])
        else:
            out.append((file, offset, length, bounce))
    return out


@pytest.mark.parametrize("weights", [None, (1.0, 1.0), (0.0, 1.0)], ids=["one_part", "halves", "zero_first_part"])
@pytest.mark.parametrize("workers", [0, 2])
def test_u10_flag_off_issues_todays_sqes_and_credit_and_packs_the_same_bytes(tmp_path, weights, workers):
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights, hidden=256, inter=512)
    parts = s.tables.extents.shape[2]
    experts = list(range(11))[::-1]  # 11 rows: two batches
    slots = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
    result, log, info, record = read_rows_sqes(s.tables, 1, experts, slots, direct=False, pack_workers=workers)
    assert result == 1 and record["piece_stream"] == 0 and record["pieces"] == []
    baseline = _baseline_sqes(s.tables, 1, experts)
    assert log == baseline
    assert info == dict(sqes=len(baseline), descriptors=16 * parts, credit=16 * parts, cqes=len(baseline))
    assert record["extents"] == len(baseline) and record["submitted_bytes"] == sum(e[2] for e in baseline)
    assert all(e["sub"] == 0 for e in record["extent_cqe"]) and record["pieces_vetted"] == 0
    assert record["pieces_published"] == record["pieces_out_of_order"] == record["piece_publish_refused"] == 0
    # Every row packed whole, as one copy: its span is one job's, and useful bytes count each row once.
    assert all(0 < row["start"] <= row["end"] for row in record["row_pack"])
    assert record["useful_bytes"] == len(experts) * int(s.tables.segments[:, 3].sum())
    split._assert_rows(s, 1, experts, slots)
    if workers == 0:
        return  # piece streaming needs workers
    # Flag on, the same read: the same bytes read into the same places, cut finer; the same credit.
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, on_log, on_info, on_record = read_rows_sqes(
        s.tables, 1, experts, slots, direct=False, pack_workers=workers, piece_stream=True
    )
    assert result == 1 and on_record["piece_stream"] == 1
    assert _merged(on_log) == _merged(baseline) and len(on_log) > len(baseline)
    assert on_info["descriptors"] == 16 * parts * SUB_READS and on_info["credit"] == info["credit"]
    assert on_record["pending_max"] <= on_info["credit"]
    assert on_record["pieces_published"] == PIECES * len(experts) and on_record["piece_publish_refused"] == 0
    assert on_record["useful_bytes"] == record["useful_bytes"]
    split._assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize("credit", [1, 3, 5])
def test_u10_a_sub_read_takes_one_credit_like_a_part(tmp_path, credit):
    """Credit counts SQEs, flag on or off: the capped reader keeps exactly `credit` reads outstanding at its peak."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [3, 0, 5, 1], [0, 1, 2, 3]
    for piece_stream in (False, True):
        result, log, info, record = read_rows_sqes(
            s.tables, 1, experts, slots, direct=False, pack_workers=2, piece_stream=piece_stream, max_outstanding=credit
        )
        assert result == 1 and record["pending_max"] == credit, piece_stream
        assert info["sqes"] == info["cqes"] == record["extents"]
        split._assert_rows(s, 1, experts, slots)


# ---- Fault hooks name a part and a sub-read (the split suite's part_short / EINTR tests, per sub-read) ----


def _sub_read(s, row, expert, part, k):
    return next(sub for sub in piece_geometry(s.tables, row, expert)[0] if (sub["part"], sub["k"]) == (part, k))


@pytest.mark.parametrize("ordinal, part, k", [(2, 1, 0), (9, 0, 2), (15, 1, 3)])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_short_sub_read_resubmits_only_its_own_sub_read(tmp_path, ordinal, part, k, reverse):
    """Sub-read k of part `part` of one row, in either bank, returns one page. Only the rest of that sub-read is
    read again (one extra completion, its length minus a page of retried bytes), under a tight credit and reversed
    completions, and every row still lands byte-exact."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = list(range(16)), list(range(16))
    sub = _sub_read(s, 1, experts[ordinal], part, k)
    assert sub["length"] > PAGE  # the fault can fire
    result, log, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2, step=8, part=part, sub=k,
        ordinal=ordinal, part_short=PAGE, reverse_cqes=reverse, max_outstanding=4,
    )
    assert result == 1
    reads = sum(len(piece_geometry(s.tables, 1, e)[0]) for e in experts)
    assert info["cqes"] == info["sqes"] == reads + 1 and record["extents"] == reads
    assert record["retried_bytes"] == sub["length"] - PAGE
    bounce = (ordinal // 8 * 8 + ordinal % 8) * s.tables.slot_bytes + sub["dest"]
    assert log.count((sub["file"], sub["offset"] + PAGE, sub["length"] - PAGE, bounce + PAGE)) == 1  # the resubmit
    split._assert_rows(s, 1, experts, slots)


def test_an_interrupted_sub_read_is_resubmitted_whole_and_counted_as_retried(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    sub = _sub_read(s, 1, 0, 1, 2)
    result, record = read_rows_traced(
        s.tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=1, part=1, sub=2, part_error=4  # EINTR
    )
    total = sum(e["length"] for e in piece_geometry(s.tables, 1, 0)[0])
    assert result == 1 and record["retried_bytes"] == sub["length"]
    assert record["submitted_bytes"] == total + sub["length"]
    split._assert_rows(s, 1, [0], [0])


# ---- The flag's refusals, and how it reaches the reader ----


def test_the_reader_refuses_piece_streaming_without_packing_workers(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0))
    with pytest.raises(RuntimeError, match="needs packing workers"):
        read_rows_traced(s.tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=0)


def test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0, 1.0))
    with pytest.raises(RuntimeError, match="mirror parts"):
        read_rows_traced(s.tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=1)


@pytest.mark.parametrize("field", ["slabs", "row_bytes"])
def test_the_reader_refuses_a_slab_row_base_that_is_not_128_byte_aligned(tmp_path, field):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0, 1.0))
    assert read_rows_traced(s.tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=1)[0] == 1
    tables = SimpleNamespace(**vars(s.tables))
    setattr(tables, field, getattr(s.tables, field).clone())
    getattr(tables, field).view(-1)[0] += 64
    with pytest.raises(RuntimeError, match="128 B aligned"):
        read_rows_traced(tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=1)


def _host(tmp_path, workers, *, lease_mode=True, two_phase=True, piece_stream=True):
    """A tier as the service builds it for piece streaming: lease mode, two-phase, packing workers, the flag."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False, pack_workers=workers
    )
    if lease_mode:
        host.enable_lease_mode()
    if two_phase:
        host.enable_two_phase()
    if piece_stream:
        host.enable_piece_stream()
    return s, page, host, LeaseSim(host, page, s.slabs)


def _piece_word_of(sim, req, lane):
    return sim.read_u64(sim.layout.piece_offset + (req.idx * lease.LANES + lane) * lease.PIECE_MASK_LINE_BYTES)


def _area_p(host):
    start = host.lease_layout.piece_offset
    return host.lease_block[start : start + lease.AREA_PIECE_MASK_BYTES]


def _assert_mapped(s, host, experts):
    mapping = host.mapping(1)
    reference = s.reference(1, list(experts))
    for i, expert in enumerate(experts):
        for name in reference:
            assert torch.equal(s.slabs[1][name][mapping[expert]].view(torch.uint8), reference[name][i].view(torch.uint8))


def _serve(host, sim, lanes, timeout_s=5.0):
    req = sim.post(1, lanes)
    assert host.pump() == 1
    return req, sim.wait(req, timeout_s=timeout_s)


def test_the_tier_publishes_every_piece_of_a_read_row_into_each_lane_that_names_it(tmp_path):
    """Expert 3 is resident (a hit); 4 and 5 are read, and 4 is named by two lanes. Every lane that names a row read
    ends with all eight bits under the request's generation; the hit lane's word is never written."""
    s, page, host, sim = _host(tmp_path, 2)
    try:
        _serve(host, sim, [3])
        host.enable_trace()
        req, waited = _serve(host, sim, [3, 4, 5, 4])
        assert waited.status == 1 and waited.go == 4
        assert [_piece_word_of(sim, req, lane) for lane in range(4)] == [0] + [piece_word(req.gen, FULL)] * 3
        (record,) = host.drain_trace()
        assert record["piece_stream"] == 1 and record["pieces_vetted"] == record["pieces_published"] == 2 * PIECES
        assert record["extents"] == 2 * 2 * SUB_READS and record["piece_publish_refused"] == 0
        assert host.counters()["piece_publish_refused"] == 0
        _assert_mapped(s, host, [4, 5])
        host.start_thread()
        with pytest.raises(RuntimeError, match="before the service thread starts"):
            host.enable_piece_stream()
    finally:
        host.stop()


def _serve_threaded(sim, lanes):
    req = sim.post(1, lanes)
    return req, sim.wait(req, timeout_s=5.0)


def test_the_miss_lanes_words_carry_the_generation_from_reservation_while_the_read_runs(tmp_path):
    """The words are initialised in the reservation hold, before the hit grant, not after the read: while the read
    is held up, the hit lane's row result is already READY and each miss lane's word is the request's generation with
    no bit. Without that initialisation every publish would be refused (another generation) and the request fail."""
    s, page, host, sim = _host(tmp_path, 2)
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    try:
        _, first = _serve_threaded(sim, [3])
        assert first.status == 1
        host.inject(delay_s=1.0)
        req = sim.post(1, [3, 4])
        seen = {}

        def observe():
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and sim.row_result(req, 0)["tag"] != lease.READY:
                time.sleep(0.001)
            seen["hit"] = sim.row_result(req, 0)["tag"]
            seen["miss"] = _piece_word_of(sim, req, 1)
            seen["done"] = page_word(page, "demand_done")

        watcher = threading.Thread(target=observe)
        watcher.start()
        waited = sim.wait(req, timeout_s=10.0)
        watcher.join()
        assert seen["hit"] == lease.READY and seen["done"] != req.seq  # observed inside the read's delay
        assert seen["miss"] == piece_word(req.gen)
        assert waited.status == 1 and _piece_word_of(sim, req, 1) == piece_word(req.gen, FULL)
    finally:
        host.stop()


def test_a_double_publish_inside_the_tier_fails_the_request_and_is_counted(tmp_path):
    """U6 through the service: the refused publish fails the read, so the request answers failed, and the tier's
    counter says why."""
    s, page, host, sim = _host(tmp_path, 2)
    try:
        host.inject_fault(publish_twice=2)
        req, waited = _serve(host, sim, [4, 5])
        assert waited.status == 2
        counters = host.counters()
        assert counters["piece_publish_refused"] == 1 and counters["read_errors"] == 1 and counters["rows_read"] == 0
    finally:
        host.stop()


@pytest.mark.parametrize("lease_mode, two_phase", [(False, False), (True, False)], ids=["no_leases", "no_two_phase"])
def test_the_tier_refuses_piece_streaming_without_two_phase_and_leases(tmp_path, lease_mode, two_phase):
    """The service refuses the flag unless two-phase and lease mode are on; the tier refuses too, per request and
    before any slot is taken, so a misconfigured tier reads nothing and publishes nothing."""
    s, page, host, sim = _host(tmp_path, 2, lease_mode=lease_mode, two_phase=two_phase)
    try:
        if lease_mode:
            req, waited = _serve(host, sim, [4])
            status = waited.status
        else:
            seq = ops.sim_post(page, 1, need=[4], protect=[4])
            assert host.pump() == 1
            status = ops.sim_wait(page, seq, timeout_s=1.0)
        assert status == 2
        counters = host.counters()
        assert counters["piece_stream_refused"] == 1 and counters["rows_read"] == 0 and counters["read_errors"] == 0
        assert host.mapping(1)[4] == -1 and all(state == 0 for state, *_ in host.slot_info(1))
        assert not _area_p(host).any()
    finally:
        host.stop()


@pytest.mark.parametrize("workers", [0, 2])
def test_flag_off_a_lease_two_phase_tier_never_writes_the_readiness_words(tmp_path, workers):
    """U10 for the tier: with the flag off, hits and misses are served as before and area P stays all zero."""
    s, page, host, sim = _host(tmp_path, workers, piece_stream=False)
    try:
        for lanes in ([3], [3, 4, 5], [4, 0]):
            _, waited = _serve(host, sim, lanes)
            assert waited.status == 1
        counters = host.counters()
        assert counters["piece_stream_refused"] == counters["piece_publish_refused"] == 0
        assert not _area_p(host).any()
        _assert_mapped(s, host, [3, 4, 5, 0])
    finally:
        host.stop()


def test_the_tier_refuses_the_flag_without_packing_workers(tmp_path):
    _, _, host, _ = _host(tmp_path, 0, piece_stream=False)
    try:
        with pytest.raises(RuntimeError, match="needs packing workers"):
            host.enable_piece_stream()
    finally:
        host.stop()


# ---- The two-phase suite, served with the flag on ----
# Its tests drive a lease-mode, two-phase tier through the Python device stand-in; here every host they build packs
# on workers and streams pieces, so each grant and release rule they check is checked with publishing in the read.


@pytest.fixture
def piece_streaming_hosts(monkeypatch):
    init = Exl3RamMissHost.__init__

    def with_pieces(self, *args, pack_workers=None, **kwargs):
        init(self, *args, pack_workers=2 if pack_workers is None else pack_workers, **kwargs)
        self.enable_piece_stream()

    monkeypatch.setattr(Exl3RamMissHost, "__init__", with_pieces)


def _reuse_two_phase():
    for name, test in vars(two_phase).items():
        if name.startswith("test_") and callable(test):
            clone = types.FunctionType(test.__code__, test.__globals__, name, test.__defaults__, test.__closure__)
            clone.__dict__.update(test.__dict__)
            globals()[f"{name}_with_piece_streaming"] = pytest.mark.usefixtures("piece_streaming_hosts")(clone)


_reuse_two_phase()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
