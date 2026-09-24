"""Piece streaming in the C++ row reader and tier (CPU): sub-reads, piece geometry, per-piece vetting, packing
and publishing.

SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM reads each part of a row as up to 4 page-aligned sub-reads and cuts the
row's needed bytes into 8 pieces. Each piece is vetted once the sub-reads it depends on have landed, packed by its
own job, and published by the reader's owner into the readiness words (lease area P) of the lanes that name its row,
with a generation-checked compare-and-swap. The tier initialises those words at reservation. U1 (geometry), U2
(order under reordered completions), U3 (a bit implies its bytes), U6 (a double publish), U8 (the publish primitive)
and U10 (the flag off leaves the reader as it was) are the plan's names for these tests. The tier grants every lane
at reservation, a miss lane under tag LOADING (U4), and quarantines a leased slot whose read failed instead of
releasing it (U5, U7, U9, and the two-phase suite's T1-T4 adapted to both).
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
import test_exl3_ram_miss_two_phase_victim as two_phase_victim
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
        0 < row["sub_seq"][later] < row["sub_seq"][earlier]
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


# ---- The loading grant (U4), quarantine (U5, U9) and a void while reading (U7) ----

FREE, LOADING_SLOT, READY_SLOT, QUARANTINE = 0, 1, 2, 3
GEN_BITS = (1 << 56) - 1


def _slot(host, row, slot):
    state, expert, leases, _ = host.slot_info(row)[slot]
    return state, expert, leases


def _accept(sim, req):
    """The stream kernel's verdict on a request whose status is already known served, without sim_wait (which
    returns "fatal already raised" once any request of the page has failed): every lane's row result names this
    request and its expert, READY, or LOADING with every piece published."""
    ctx = []
    for lane, expert in enumerate(req.lanes):
        result = sim.row_result(req, lane)
        assert result["gen"] == req.gen and result["expert"] == expert, (lane, result)
        assert result["tag"] == lease.READY or (
            result["tag"] == lease.LOADING and sim.piece_word(req, lane) == piece_word(req.gen, FULL)
        ), (lane, result, hex(sim.piece_word(req, lane)))
        ctx.append((result["host_slot"], result["slot_generation"]))
    return types.SimpleNamespace(status=1, go=len(req.lanes), ctx=ctx)


def _pump_until_done(host, page, req):
    assert host.pump() == 1
    assert page_word(page, "demand_done") == req.seq


def test_u4_miss_lanes_are_granted_loading_and_hit_lanes_ready_before_the_read(tmp_path):
    """U4. While the read is held up, the hit lane's row result is READY and each miss lane's is LOADING with its
    final payload (this request's generation, its expert, the slot reserved for it and that slot's generation); the
    miss slots are leased while still kLoading, and the ring entry owes no further grant. A tight observer reads each
    miss lane as the stream kernel will -- the ready word, then its PieceMask word -- and must never see a LOADING
    word of this generation over a PieceMask word of another: the mask is initialised before the ready word. After
    the read the miss lanes are still LOADING: there is no second grant.

    Mutants: grant the miss lanes after the read (today's S3) -- red on the LOADING tags; grant a miss lane READY --
    red on the tag; drop the PieceMask init (task 3's no_mask_init) -- red on the ordering check and on the result."""
    s, page, host, sim = _host(tmp_path, 2)
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    try:
        _, first = _serve_threaded(sim, [3])
        assert first.status == 1
        host.inject(delay_s=1.0)
        req = sim.post(1, [3, 4, 5, 4])
        seen, torn, looks = {}, [], [0]

        def observe():
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and sim.row_result(req, 0)["tag"] != lease.READY:
                pass
            seen["results"] = [sim.row_result(req, lane) for lane in range(4)]
            seen["slots"] = host.slot_info(1)
            seen["entry"] = host.lease_entry(req.idx)
            seen["done"] = page_word(page, "demand_done")
            while page_word(page, "demand_done") != req.seq and time.perf_counter() < deadline + 10.0:
                for lane in (1, 2, 3):
                    result = sim.row_result(req, lane)
                    word = sim.piece_word(req, lane)  # read after the ready word, as the device does
                    if result["tag"] == lease.LOADING and result["gen"] == req.gen:
                        looks[0] += 1
                        if word >> 8 != req.gen & GEN_BITS:
                            torn.append((lane, hex(word)))

        watcher = threading.Thread(target=observe)
        watcher.start()
        waited = sim.wait(req, timeout_s=10.0)
        watcher.join()

        assert seen["done"] != req.seq, "demand_done had reached the request: the read was not still running"
        results, slots = seen["results"], seen["slots"]
        assert results[0]["tag"] == lease.READY and results[0]["gen"] == req.gen and results[0]["expert"] == 3
        for lane, expert in ((1, 4), (2, 5), (3, 4)):
            result = results[lane]
            assert result["tag"] == lease.LOADING and result["gen"] == req.gen and result["expert"] == expert, result
            state, owner, leases, generation = slots[result["host_slot"]]
            assert (state, owner, generation) == (LOADING_SLOT, expert, result["slot_generation"]), (lane, result)
            assert leases == (2 if expert == 4 else 1), "the miss slot was not leased when its row result was published"
        entry = seen["entry"]
        assert entry["active"] and not entry["grants_pending"], "the loading grant left a second grant owed"
        assert entry["lane_state"][:4] == [1] * 4 and entry["gen"] == req.gen
        assert looks[0] > 0 and not torn, f"a LOADING ready word was visible over an uninitialised PieceMask: {torn}"

        assert waited.status == 1 and waited.go == 4
        assert [sim.row_result(req, lane)["tag"] for lane in range(4)] == [lease.READY] + [lease.LOADING] * 3
        counters = host.counters()
        assert counters["hit_leases_granted"] == 1 and counters["leases_granted"] == 1 + 4
        assert counters["slots_quarantined"] == 0
    finally:
        host.stop()


def test_u5_a_failed_read_after_a_partial_publish_quarantines_its_leased_slots(tmp_path):
    """U5. Sub-read 3 of part 1 of row 1 (expert 1) fails. Both miss slots are leased under LOADING, so neither is
    released: each is kQuarantine with its lease and no expert, and the map never names it. The quarantined slots
    are neither free, evictable nor leased victims; a later request never lands on one, even when it must evict to
    find a slot. The terminal that voids the failed request's lanes frees them.

    Mutant M4: release a still-loading slot on a failed read -- red on state == kQuarantine (release_locked does not
    touch leases, so no underflow fires)."""
    s, page, host, sim = _host(tmp_path, 2)
    try:
        host.inject_fault(part=1, sub=3, ordinal=1, part_error=5)  # EIO
        failed = sim.post(1, [4, 1])
        _pump_until_done(host, page, failed)
        counters = host.counters()
        assert counters["read_errors"] == 1 and counters["rows_read"] == 0 and counters["slots_quarantined"] == 2
        quarantined = []
        for lane, expert in enumerate(failed.lanes):
            result = sim.row_result(failed, lane)
            assert result["tag"] == lease.LOADING and result["gen"] == failed.gen
            slot = result["host_slot"]
            assert _slot(host, 1, slot) == (QUARANTINE, -1, 1), (lane, host.slot_info(1))
            assert host.slot_to_expert(1)[slot] == -1 and host.mapping(1)[expert] == -1
            assert not host.contains(1, expert)
            quarantined.append(slot)
        word = sim.piece_word(failed, 1)
        assert word >> 8 == failed.gen & GEN_BITS and word & FULL != FULL, hex(word)  # the failed row: not every piece
        assert host.victim_census(1, []) == (2, 0, 0), "a quarantined slot was counted as a victim or as leased"

        host.inject_fault()
        again = sim.post(1, [1, 5])  # the failed expert again, and another
        _pump_until_done(host, page, again)
        mapping = host.mapping(1)
        assert mapping[1] not in quarantined and mapping[5] not in quarantined and -1 not in (mapping[1], mapping[5])
        _assert_mapped(s, host, [1, 5])
        sim.ack(again, _accept(sim, again))
        sim.deliver()
        host.pump()  # idle: retires the acknowledgements
        assert all(leases == 0 for _, e, leases, _ in host.slot_info(1) if e in (1, 5))

        pressed = sim.post(1, [0])  # no free slot left: it must evict, and may evict only 1 or 5
        _pump_until_done(host, page, pressed)
        assert host.mapping(1)[0] not in quarantined and host.counters()["evictions"] == 1
        assert [_slot(host, 1, slot) for slot in quarantined] == [(QUARANTINE, -1, 1)] * 2

        sim.terminal(failed, mask=0b11)  # the device voids the failed request's lanes
        sim.deliver()
        host.pump()
        assert [_slot(host, 1, slot) for slot in quarantined] == [(FREE, -1, 0)] * 2
        assert host.counters()["leases_voided"] == 2 and host.counters()["lease_double_signal"] == 0
        assert sim.wait(failed, publish_terminal=False).status == 2
    finally:
        host.stop()


def test_u9_voiding_a_quarantined_slot_leaves_its_expert_where_it_was_read_again(tmp_path):
    """U9. Expert 1 fails into slot s (quarantined); the next request reads it into slot t. Voiding s must free s
    and leave expert 1 mapped at t: in the tier, in slot_to_expert and in the published slot map.

    Mutant M4b: clear only expert_slot on entry, keeping slot_to_expert[s] -- the release then unmaps expert 1 from t.
    """
    s, page, host, sim = _host(tmp_path, 2)
    try:
        host.inject_fault(part=0, sub=0, ordinal=0, part_error=5)
        failed = sim.post(1, [1])
        _pump_until_done(host, page, failed)
        quarantined = sim.row_result(failed, 0)["host_slot"]
        state, _, leases = _slot(host, 1, quarantined)  # its expert field is U5's check; this test is the consequence
        assert (state, leases) == (QUARANTINE, 1)

        host.inject_fault()
        again = sim.post(1, [1])
        _pump_until_done(host, page, again)
        t = host.mapping(1)[1]
        assert t >= 0 and t != quarantined
        sim.ack(again, _accept(sim, again))

        sim.terminal(failed, mask=1)
        sim.deliver()
        host.pump()
        assert _slot(host, 1, quarantined) == (FREE, -1, 0)
        assert host.mapping(1)[1] == t and host.slot_to_expert(1)[t] == 1 and host.contains(1, 1)
        assert _slot(host, 1, t) == (READY_SLOT, 1, 0)
        assert int(host.slot_map[1, 1]) == t, "the published slot map lost the expert's new slot"
        _assert_mapped(s, host, [1])
    finally:
        host.stop()


@pytest.mark.parametrize("fails", [False, True], ids=["read_succeeds", "read_fails"])
def test_u7_a_lane_voided_while_its_row_is_read_leaves_the_slot_loading_until_the_read_ends(tmp_path, fails):
    """U7. The device gives up on a request (its terminal voids the miss lane) while the row is still being read.
    The lease is retired from inside read(), but the slot stays kLoading -- the workers are still writing it -- and
    only the post-read step decides it: READY and mapped when the read succeeded, FREE (not quarantined: nothing
    leases it) when it failed."""
    s, page, host, sim = _host(tmp_path, 2)
    host.inject_fault(pack_delay_ns=40_000_000, publish_twice=3 if fails else 0)
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    try:
        req = sim.post(1, [4])
        assert two_phase._until(lambda: sim.row_result(req, 0)["tag"] == lease.LOADING)
        slot = sim.row_result(req, 0)["host_slot"]
        sim.terminal(req, mask=1)
        sim.deliver()
        seen = {}

        def voided():
            state, _, leases = _slot(host, 1, slot)
            seen.update(state=state, leases=leases, done=page_word(page, "demand_done"))
            return leases == 0

        assert two_phase._until(voided, timeout_s=10.0)
        assert seen["done"] != req.seq, "the read had ended before the void was retired"
        assert seen["state"] == LOADING_SLOT, "a voided slot left kLoading while its row was still being read"
        assert two_phase._until(lambda: page_word(page, "demand_done") == req.seq, timeout_s=20.0)
        counters = host.counters()
        assert counters["leases_voided"] == 1 and counters["slots_quarantined"] == 0
        if fails:
            assert counters["piece_publish_refused"] == 1
            assert _slot(host, 1, slot) == (FREE, -1, 0) and host.mapping(1)[4] == -1
        else:
            assert _slot(host, 1, slot) == (READY_SLOT, 4, 0) and host.mapping(1)[4] == slot
            _assert_mapped(s, host, [4])
    finally:
        host.stop()


# ---- The two-phase suite's T1-T4, adapted to the loading grant (plan section 6) ----


def test_t4_a_failed_request_voids_its_leases_and_quarantines_its_miss_slot(piece_streaming_hosts, running):
    """T4 and T4b under piece streaming. The miss lane is now leased at reservation, so the failure path holds a
    miss lease too: its slot goes to kQuarantine (never released), and the hit slot keeps its row and lease. Neither
    slot is handed to the next request. The device's terminal retires both leases; the quarantined slot is then free.
    """
    s, page, host, sim = running
    hit_slot = two_phase._make_resident(host, sim, 0, 3)
    host.inject(fail_reads=True)
    req = sim.post(0, [3, 4])
    assert two_phase._until(lambda: page_word(page, "demand_done") == req.seq, timeout_s=10.0)
    miss = sim.row_result(req, 1)
    assert miss["tag"] == lease.LOADING and miss["gen"] == req.gen
    miss_slot = miss["host_slot"]
    assert two_phase._until(lambda: _slot(host, 0, miss_slot)[0] == QUARANTINE)
    assert _slot(host, 0, hit_slot) == (READY_SLOT, 3, 1), "the failed request released or corrupted the hit slot"
    assert _slot(host, 0, miss_slot) == (QUARANTINE, -1, 1), "the leased miss slot was released, not quarantined"
    assert host.mapping(0)[4] == -1

    host.inject(fail_reads=False)
    sim.post(0, [5])
    assert two_phase._until(lambda: any(st == READY_SLOT and e == 5 for st, e, _, _ in host.slot_info(0)), timeout_s=10.0)
    assert two_phase._slot_of(host, 0, 5) not in (hit_slot, miss_slot), "a leased slot was handed to the next request"

    sim.terminal(req, mask=0b11)
    sim.deliver()
    assert two_phase._until(lambda: _slot(host, 0, miss_slot) == (FREE, -1, 0))
    assert _slot(host, 0, hit_slot) == (READY_SLOT, 3, 0)
    counters = host.counters()
    assert counters["leases_voided"] >= 2 and counters["lease_double_signal"] == 0
    assert counters["slots_quarantined"] == 1


def test_t4c_the_loading_grant_leaves_no_grant_pending(piece_streaming_hosts, running):
    """T4c is not applicable under piece streaming: every lane is granted in the reservation hold, so no grant is
    owed across the read. The entry opens with grants_pending false and every lane granted, and retirement driven
    from the test thread during the read disturbs nothing."""
    s, page, host, sim = running
    two_phase._make_resident(host, sim, 0, 3)
    host.inject(delay_s=two_phase.READ_DELAY_S)
    req = sim.post(0, [3, 4])
    assert two_phase._until(lambda: sim.row_result(req, 1)["tag"] == lease.LOADING, timeout_s=two_phase.READ_DELAY_S)
    entry = host.lease_entry(req.idx)
    for _ in range(20):
        sim.deliver()
        time.sleep(0.005)
    assert page_word(page, "demand_done") != req.seq
    assert entry["active"] and not entry["grants_pending"] and entry["lane_state"][:2] == [1, 1]
    waited = sim.wait(req, timeout_s=two_phase.READ_DELAY_S + 10.0)
    assert waited.status == 1 and waited.go == 2
    assert host.counters()["leases_granted"] == 3 and host.counters()["lease_double_signal"] == 0


@pytest.mark.parametrize("workers", [0, 2])
def test_flag_off_the_tier_grants_miss_lanes_after_the_read_and_releases_a_failed_read(tmp_path, workers):
    """Flag-off identity for the grant, retire, post-read and release: the miss lane is ungranted (tag 0, lease 0,
    grant pending) while the read runs, granted READY after it, and a failed read releases its miss slot to FREE
    with nothing quarantined and no miss lease."""
    s, page, host, sim = _host(tmp_path, workers, piece_stream=False)
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    try:
        _, first = _serve_threaded(sim, [3])
        assert first.status == 1
        host.inject(delay_s=0.5)
        req = sim.post(1, [3, 4])
        assert two_phase._until(lambda: sim.row_result(req, 0)["tag"] == lease.READY, timeout_s=0.5)
        entry, miss = host.lease_entry(req.idx), sim.row_result(req, 1)
        loading = [slot for slot, (st, e, _, _) in enumerate(host.slot_info(1)) if st == LOADING_SLOT and e == 4]
        assert entry["grants_pending"] and entry["lane_state"][:2] == [1, 0] and miss["tag"] == 0
        assert len(loading) == 1 and _slot(host, 1, loading[0]) == (LOADING_SLOT, 4, 0)
        waited = sim.wait(req, timeout_s=10.0)
        assert waited.status == 1 and sim.row_result(req, 1)["tag"] == lease.READY
        assert not host.lease_entry(req.idx)["grants_pending"]

        host.inject(delay_s=0.0, fail_reads=True)
        failed = sim.post(1, [3, 5])
        assert two_phase._until(lambda: page_word(page, "demand_done") == failed.seq)
        assert sim.row_result(failed, 1)["tag"] == 0 and host.mapping(1)[5] == -1
        assert all(state != QUARANTINE and (e != 5 or state == FREE) for state, e, _, _ in host.slot_info(1))
        counters = host.counters()
        assert counters["slots_quarantined"] == 0 and counters["hit_leases_granted"] == 2
        assert not _area_p(host).any()
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


# Replaced above by their adaptations to the loading grant: T1 asserts the miss lane is unpublished (tag 0) during
# the read, where it is now LOADING (U4 is its adaptation); T4c tests a pending second grant that no longer exists.
# T1b, T4b and T3 hold unchanged; T4 does too, and its quarantine adaptation is test_t4_... above.
_NOT_REUSED = {
    "test_a_resident_lane_is_published_before_read_returns",
    "test_an_ungranted_lane_keeps_the_ring_entry_open",
}


def _reuse_two_phase():
    for module in (two_phase, two_phase_victim):
        for name, test in vars(module).items():
            if name.startswith("test_") and callable(test) and name not in _NOT_REUSED:
                clone = types.FunctionType(test.__code__, test.__globals__, name, test.__defaults__, test.__closure__)
                clone.__dict__.update(test.__dict__)
                globals()[f"{name}_with_piece_streaming"] = pytest.mark.usefixtures("piece_streaming_hosts")(clone)


_reuse_two_phase()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
