"""Piece streaming in the C++ row reader and tier (CPU): sub-reads, piece geometry, per-piece vetting, packing
and publishing.

Piece streaming reads each part of a row as up to 4 page-aligned sub-reads and cuts the
row's needed bytes into 8 pieces. Each piece is vetted once the sub-reads it depends on have landed, packed by its
own job, and published by the reader's owner into the readiness words (lease area P) of the lanes that name its row,
with a generation-checked compare-and-swap. The tier initialises those words at reservation. U1 (geometry), U2
(order under reordered completions), U3 (a bit implies its bytes), U6 (a double publish), U8 (the publish primitive)
and U10 (a sub-read's credit) are the plan's names for these tests. The service always streams pieces: it grants every
lane at reservation, a miss lane under tag LOADING, and a read that fails aborts the process.
"""

import faulthandler
import random
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import test_exl3_ram_miss_split as split
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.kernels.ops.moe.expert_stream_transport import (
    ExpertStreamHost,
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
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import attached_host, assert_aborted, ram_miss_setup, run_host_script

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


PAGE = 4096
SUB_READS = 4  # the most sub-reads a part is cut into (the C++ kSubReads)
PIECES = ops.STAGE_PIECES
ALIGN = 128


def _round_up(value, to):
    return -(-value // to) * to


def _per_part(reading):
    """Sub-reads per reading part of a row that reads ``reading`` nonzero parts (the C++ sub_reads_per_part): the
    pieces shared out, at most SUB_READS. 1 or 2 -> 4 (the cut before N parts), 3 or 4 -> 2, 5 to 8 -> 1."""
    return SUB_READS if reading <= 0 else min(SUB_READS, PIECES // reading)


def _expected_sub_reads(part, per_part=SUB_READS):
    """Plan section 4.1, written out independently of the C++: len_k = round_up(ceil(len / per_part), page)."""
    file, offset, length, dest = part
    if length <= 0:
        return []
    len_k = _round_up(-(-length // per_part), PAGE)
    return [(file, offset + at, min(len_k, length - at), dest + at) for at in range(0, length, len_k)]


def _row_parts(tables, row, expert):
    return [tables.extents[row, expert, p].tolist() for p in range(tables.extents.shape[2])]


def _row_sub_reads(tables, row, expert):
    parts = _row_parts(tables, row, expert)
    per_part = _per_part(sum(length > 0 for _, _, length, _ in parts))
    out = []
    for p, part in enumerate(parts):
        out += [(p, k, sub) for k, sub in enumerate(_expected_sub_reads(part, per_part))]
    return out


def _assert_geometry(tables, row, expert):
    """U1 for one row: sub-reads tile each part, pieces partition the needed bytes on 128 B cuts that are the
    sub-read boundaries, and each dependency mask is exactly the sub-reads the piece's bytes come from."""
    got = piece_geometry(tables, row, expert)
    assert got is not None, (row, expert)
    sub_reads, pieces = got
    expected = _row_sub_reads(tables, row, expert)
    assert [(s["part"], s["k"], (s["file"], s["offset"], s["length"], s["dest"])) for s in sub_reads] == expected
    parts = _row_parts(tables, row, expert)
    per_part = _per_part(sum(length > 0 for _, _, length, _ in parts))
    assert len(sub_reads) <= PIECES
    for p, (file, offset, length, dest) in enumerate(parts):
        mine = [s for s in sub_reads if s["part"] == p]
        assert len(mine) <= per_part
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
        elif parts == 2:
            # Row 0 is split evenly, so the layout always has a row with every sub-read.
            first = pages // 2 if r == 0 else rng.choice([0, pages, 1, pages - 1, rng.randrange(pages + 1)])
            split_pages = [first, pages - first]
        elif r == 0:
            split_pages = [pages // parts] * (parts - 1) + [pages - pages // parts * (parts - 1)]
        else:
            # Any cut, and every third row gives one part's pages to its neighbour (a 0 share, or a rounding).
            cuts = sorted(rng.randrange(pages + 1) for _ in range(parts - 1))
            split_pages = [b - a for a, b in zip([0] + cuts, cuts + [pages])]
            if r % 3 == 1:
                z = rng.randrange(parts)
                split_pages[(z + 1) % parts] += split_pages[z]
                split_pages[z] = 0
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
@pytest.mark.parametrize("parts", [1, 2, 3, 4, 8])
def test_u1_geometry_of_every_row_of_a_random_layout(seed, parts):
    tables = _synthetic_tables(seed, parts=parts)
    counts = set()
    for row in range(tables.extents.shape[0]):
        for expert in range(tables.extents.shape[1]):
            sub_reads, _ = _assert_geometry(tables, row, expert)
            counts.add(len(sub_reads))
    # Row 0 reads all `parts` parts, split evenly and long enough for every sub-read (per_part(parts) * parts of
    # them: 4, 8, 6, 8, 8). But every third row (r % 3 == 1) zeroes one part's pages, dropping `reading` below
    # `parts` for that row, and per_part(reading) can rise faster than reading falls (per_part(2) == 4 vs
    # per_part(3) == 2, so 2 reading parts give 8 sub-reads -- more than 3 parts' 6): the true ceiling over a
    # layout with parts >= 2 is max(per_part(reading) * reading for reading in 1..parts), not row 0's own count.
    assert max(counts) == max(_per_part(reading) * reading for reading in range(1, parts + 1))


def _one_row_tables(part_pages):
    """One layer, one expert, one segment spanning the row: part p reads part_pages[p] pages of file p, in order.

    slabs/row_bytes are indexed by name (the layout's EXL3_STREAMED_NAMES, unrelated to the row's mirror parts) and
    the exl3 host module's tables_from checks their width against the layout's fixed name count, so they are padded
    to it here even though only name 0 (the one segment) is used."""
    parts = len(part_pages)
    names = len(EXL3_STREAMED_NAMES)
    total = sum(part_pages) * PAGE
    extents = torch.zeros((1, 1, parts, 4), dtype=torch.int64)
    dest = 0
    for p, pages in enumerate(part_pages):
        extents[0, 0, p] = torch.tensor([p, dest, pages * PAGE, dest])
        dest += pages * PAGE
    row_bytes = torch.zeros((names,), dtype=torch.int64)
    row_bytes[0] = total
    return SimpleNamespace(
        extents=extents,
        starts=torch.zeros((1, 1), dtype=torch.int64),
        file_sizes=torch.tensor([total] * parts, dtype=torch.int64),
        segments=torch.tensor([(0, 0, 0, total)], dtype=torch.int64),
        slabs=torch.zeros((1, names), dtype=torch.int64),
        row_bytes=row_bytes,
        paths=[f"/nonexistent/{p}" for p in range(parts)],
        source_paths=["/nonexistent/source"] * parts,
        slot_bytes=total,
    )


def _assert_empty_past(pieces, n):
    """Pieces 0..n-1 have bytes and dependencies; pieces n..7 have neither, so the reader publishes them at admission."""
    assert all(piece["deps"] != 0 for piece in pieces[:n])
    assert all(piece["deps"] == 0 and all(lo == hi for lo, hi in piece["runs"]) for piece in pieces[n:])


@pytest.mark.parametrize("parts, per_part", [(1, 4), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1)])
def test_a_row_reading_every_part_cuts_each_into_its_share_of_the_pieces(parts, per_part):
    sub_reads, pieces = piece_geometry(_one_row_tables((8,) * parts), 0, 0)
    assert [sum(s["part"] == p for s in sub_reads) for p in range(parts)] == [per_part] * parts
    assert [(s["part"], s["k"]) for s in sub_reads] == [(p, k) for p in range(parts) for k in range(per_part)]
    _assert_empty_past(pieces, parts * per_part)


@pytest.mark.parametrize(
    "part_pages, per_part",
    [((8, 0, 8), 4), ((0, 8, 8), 4), ((8, 8, 0), 4), ((8, 0, 0), 4), ((8, 8, 8), 2), ((8, 1, 8), 2), ((1, 1, 1), 2),
     ((8, 8, 8, 0), 2), ((8, 0, 8, 0), 4)],
)
def test_a_zero_part_gives_its_pieces_to_the_parts_that_read(part_pages, per_part):
    """A zero-length part (a 0 mirror weight, or a rounding) reads nothing and does not count: the reading parts share
    all eight pieces. A part shorter than its share in pages is cut into fewer sub-reads, never an empty one."""
    tables = _one_row_tables(part_pages)
    sub_reads, pieces = piece_geometry(tables, 0, 0)
    counts = [sum(s["part"] == p for s in sub_reads) for p in range(len(part_pages))]
    assert counts == [min(per_part, pages) for pages in part_pages]
    _assert_empty_past(pieces, len(sub_reads))
    _assert_geometry(tables, 0, 0)


def test_nine_parts_cannot_be_cut():
    assert piece_geometry(_one_row_tables((1,) * 9), 0, 0) is None


# ---- U2: pieces are vetted in dependency order, whatever order completions arrive in ----


def _landings(record):
    return sorted(seq for row in record["pieces"] for seq in row["sub_seq"] if seq)


def test_u2_pieces_are_vetted_in_dependency_order_under_reversed_cqes_and_a_held_sub_read(tmp_path):
    """Completions are processed back to front, and sub-read 1 of part 0 of row 0 lands last of all. Every piece
    must be vetted right after its last dependency lands (no landing in between), never before, and the pieces that
    depend on the held sub-read after every other piece. Rows are still packed whole, byte-exact."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, piece_stream=True, reverse_cqes=True, hold_ordinal=0, part=0, sub=1, poison=True,
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


# ---- Publishing: U2's order, U3 (a bit implies its bytes), U6 (twice), U8 (the primitive) ----

FULL = 0xFF
GEN = (7 << 32) | 12345  # a request generation: an epoch over a sequence number
MASK64 = (1 << 64) - 1


def _words(masks):
    return [int(word) & MASK64 for word in masks[:, 0]]


def test_u2_pieces_publish_in_dependency_order_under_reversed_cqes_and_a_held_sub_read(tmp_path):
    """The same read as U2's, now publishing. Every piece is published after it is vetted, the pieces that depend on
    the held sub-read after every other piece of the read, and each row's readiness word ends full under its
    generation. The reversal is shown to have taken effect, not assumed: in a row that was not held, a later sub-read
    landed before an earlier one."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, reverse_cqes=True, hold_ordinal=0, part=0, sub=1, poison=True,
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


def test_u3_every_bit_a_reader_can_see_names_bytes_already_stored(tmp_path):
    """A thread polls each row's readiness word while the read runs (as the device will) and, for every bit it sees,
    compares that piece's destination bytes with a reference copy. The slabs start as a sentinel, the bounce is
    poisoned and every piece's copy is slow, so a bit published before its job stored the bytes (at dispatch, say)
    is seen with the sentinel behind it."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    assert read_rows_traced(s.tables, 1, experts, ref_slots)[0] == 1
    split._assert_rows(s, 1, experts, ref_slots)
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, pack_delay_ns=10_000_000, poison=True,
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
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, publish_twice=3
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
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, part=0, sub=2,
        ordinal=1, part_short=PAGE, short_is_eof=True, poison=True,
    )
    assert result == 0 and info["refused"] == 0
    row1, bits = record["pieces"][1], _words(masks)[1] & FULL
    assert row1["sub_seq"][short] > 0  # it landed, short
    dependent = [j for j, piece in enumerate(pieces) if piece["deps"] >> short & 1]
    assert dependent and all(row1["seq"][j] == 0 and not bits >> j & 1 for j in dependent)


# ---- Three mirror parts (up to eight): the same properties, one more part cutting the pieces finer ----

THREE = [(1.0, 1.0, 1.0), (3.0, 1.0, 2.0), (1.0, 0.0, 1.0)]
THREE_IDS = ["thirds", "uneven", "zero_middle"]


@pytest.mark.parametrize("weights", THREE, ids=THREE_IDS)
def test_three_parts_publish_each_piece_after_its_sub_reads_and_the_empty_ones_at_admission(tmp_path, weights):
    """Three mirror parts, completions reversed and part 2's second sub-read of row 0 held back. Every piece with bytes
    is vetted after its last dependency lands and published after it is vetted. Every piece past the row's sub-reads
    (6 and 7 when all three parts read) has no dependencies and is vetted at admission, before any of the row's
    sub-reads landed, so the device's all-eight-bits exit is reached with no device change. Rows land exact."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    assert _sub_read(s, 1, experts[0], 2, 1)["length"] > 0  # the held sub-read exists for every weighting here
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, reverse_cqes=True, hold_ordinal=0, part=2, sub=1, poison=True,
    )
    assert result == 1 and info["refused"] == 0 and record["piece_publish_refused"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    assert record["pieces_published"] == PIECES * len(experts)
    split._assert_rows(s, 1, experts, slots)
    for row, expert in zip(record["pieces"], experts):
        sub_reads, pieces = piece_geometry(s.tables, 1, expert)
        n = len(sub_reads)
        assert n == len(_row_sub_reads(s.tables, 1, expert))
        assert n == (8 if 0.0 in weights else 6), (expert, n)  # 1:0:1 cuts like two parts
        first_landing = min(row["sub_seq"][k] for k in range(n))
        for j, piece in enumerate(pieces):
            assert row["publish"][j] > row["seq"][j] > 0, (row["row"], j)
            deps = [k for k in range(n) if piece["deps"] >> k & 1]
            if j >= n:
                assert not deps and row["seq"][j] < first_landing, (row["row"], j)
            else:
                assert deps and row["seq"][j] > max(row["sub_seq"][k] for k in deps), (row["row"], j)


@pytest.mark.parametrize("weights", THREE, ids=THREE_IDS)
def test_three_parts_every_bit_a_reader_can_see_names_bytes_already_stored(tmp_path, weights):
    """U3 over three parts: a thread polls the words while the read runs and checks each bit it sees against a
    reference copy, the empty pieces' bits included (they name no bytes, so they can never differ)."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=weights, hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    assert read_rows_traced(s.tables, 1, experts, ref_slots)[0] == 1
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, pack_delay_ns=10_000_000, poison=True,
    )
    assert result == 1 and info["refused"] == 0
    assert info["checked"] == PIECES * len(experts) and info["differed"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    split._assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize(
    "fault", [dict(part_error=5), dict(part_short=PAGE, short_is_eof=True)], ids=["eio", "short_at_eof"]
)
def test_three_parts_a_failed_sub_read_publishes_none_of_its_pieces(tmp_path, fault):
    """Part 2's second sub-read of row 1 fails (EIO) or ends a page in, as at end of file. The read fails, no piece that
    depends on it is vetted or published, and nothing is published under another generation. The row's empty pieces
    may already be published: the device still sees a word short of all eight bits and a request not served."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    sub_reads, pieces = piece_geometry(s.tables, 1, experts[1])
    bad = next(k for k, sub in enumerate(sub_reads) if (sub["part"], sub["k"]) == (2, 1))
    assert sub_reads[bad]["length"] > PAGE  # the short fault can fire
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, part=2, sub=1, ordinal=1, poison=True, **fault,
    )
    assert result == 0 and info["refused"] == 0
    row1, bits = record["pieces"][1], _words(masks)[1] & FULL
    dependent = [j for j, piece in enumerate(pieces) if piece["deps"] >> bad & 1]
    assert dependent and all(row1["seq"][j] == 0 and not bits >> j & 1 for j in dependent)
    assert bits != FULL
    assert all(word >> 8 == GEN for word in _words(masks))


@pytest.mark.parametrize("credit", [1, 5, 48])
def test_three_parts_a_prefill_sized_read_over_both_banks_lands_every_row(tmp_path, credit):
    """16 rows, both banks full (the most a request carries), of three parts: 6 sub-reads a row, 96 SQEs, against the
    3-part ring's 48 credits or fewer. Every sub-read is issued once, every piece published once, every row exact."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots = list(range(16))[::-1], list(range(16))
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, log, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, piece_stream=True, step=8, max_outstanding=credit
    )
    assert result == 1
    reads = sum(len(piece_geometry(s.tables, 1, e)[0]) for e in experts)
    assert reads == 16 * 6
    assert info["sqes"] == info["cqes"] == record["extents"] == reads
    assert info["descriptors"] == 16 * 3 * SUB_READS and info["credit"] == 16 * 3
    assert record["pending_max"] <= credit
    if credit < 16 * 3:
        assert record["pending_max"] == credit
    assert record["pieces_published"] == PIECES * len(experts) and record["piece_publish_refused"] == 0
    split._assert_rows(s, 1, experts, slots)


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


@pytest.mark.parametrize("credit", [1, 3, 5])
def test_u10_a_sub_read_takes_one_credit_like_a_part(tmp_path, credit):
    """Credit counts SQEs, flag on or off: the capped reader keeps exactly `credit` reads outstanding at its peak."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    experts, slots = [3, 0, 5, 1], [0, 1, 2, 3]
    for piece_stream in (False, True):
        result, log, info, record = read_rows_sqes(
            s.tables, 1, experts, slots, piece_stream=piece_stream, max_outstanding=credit
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
        s.tables, 1, experts, slots, piece_stream=True, step=8, part=part, sub=k,
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
        s.tables, 1, [0], [0], piece_stream=True, part=1, sub=2, part_error=4  # EINTR
    )
    total = sum(e["length"] for e in piece_geometry(s.tables, 1, 0)[0])
    assert result == 1 and record["retried_bytes"] == sub["length"]
    assert record["submitted_bytes"] == total + sub["length"]
    split._assert_rows(s, 1, [0], [0])


# ---- The flag's refusals, and how it reaches the reader ----


def test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0,) * 9)
    with pytest.raises(RuntimeError, match="at most 8 mirror parts .* not 9"):
        read_rows_traced(s.tables, 1, [0], [0], piece_stream=True)


@pytest.mark.parametrize("weights", [(1.0,) * 3, (1.0,) * 4, (1.0,) * 8], ids=["three", "four", "eight"])
def test_the_reader_streams_pieces_over_three_to_eight_mirror_parts(tmp_path, weights):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, generation=GEN, piece_stream=True, poison=True
    )
    assert result == 1 and info["refused"] == 0 and record["piece_publish_refused"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    assert record["pieces_published"] == PIECES * len(experts)
    split._assert_rows(s, 1, experts, slots)


def _host(tmp_path, *, weights=(1.0, 1.0)):
    """A tier over row images and ``weights`` mirror parts; the service always streams pieces."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=weights, hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page, k=2)
    return s, page, host, ChainSim(host, page, s.slabs)


def _assert_mapped(s, host, experts):
    mapping = host.mapping(1)
    reference = s.reference(1, list(experts))
    for i, expert in enumerate(experts):
        for name in reference:
            assert torch.equal(s.slabs[1][name][mapping[expert]].view(torch.uint8), reference[name][i].view(torch.uint8))


def _serve(host, sim, lanes, timeout_s=5.0):
    """Post, pump (serves), check every miss's pieces landed."""
    req = sim.post(1, lanes)
    assert host.pump() == 1
    assert sim.wait_served(req, timeout_s=timeout_s)
    return req


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, 1.0, 1.0)], ids=["two_parts", "three_parts"])
def test_the_tier_publishes_every_piece_of_a_read_row_into_each_lane_that_names_it(tmp_path, weights):
    """Expert 3 is resident (a hit); 4 and 5 are read. Every lane that names a row read ends with all eight bits under
    the request's generation (with three parts, six with bytes and two empty); the hit lane's word is never written."""
    s, page, host, sim = _host(tmp_path, weights=weights)
    try:
        assert s.tables.parts == len(weights)
        _serve(host, sim, [3])
        host.enable_trace()
        req = _serve(host, sim, [3, 4, 5])
        assert [sim.piece_word(req, lane) for lane in range(3)] == [0] + [piece_word(req.gen, FULL)] * 2
        (record,) = [r for r in host.drain_trace() if r["seq"] == req.seq]
        assert record["piece_stream"] == 1 and record["pieces_vetted"] == record["pieces_published"] == 2 * PIECES
        extents = 2 * 2 * SUB_READS if len(weights) == 2 else 2 * 3 * 2  # rows * parts * sub-reads per part
        assert record["extents"] == extents and record["piece_publish_refused"] == 0
        assert host.counters()["piece_publish_refused"] == 0
        _assert_mapped(s, host, [4, 5])
    finally:
        host.stop()


def test_the_miss_lanes_words_carry_the_generation_before_the_read_lands(tmp_path):
    """The words are initialised before the read, not after it: while the read is held up, each miss lane's word is the
    request's generation with no bit, and the row's delta is already published. Without that initialisation every
    publish would be refused (another generation) and the process would abort."""
    s, page, host, sim = _host(tmp_path)
    host.start_thread(fatal_wait_s=60.0)
    try:
        first = sim.post(1, [3])
        assert sim.wait_served(first, timeout_s=5.0) and sim.wait_handled(first)
        host.inject(delay_s=1.0)
        req = sim.post(1, [3, 4])
        seen = {}

        def observe():
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and sim.piece_word(req, 1) != piece_word(req.gen):
                time.sleep(0.001)
            seen["miss"] = sim.piece_word(req, 1)
            seen["served"] = sim.served(req)
            seen["delta"] = sim.delta(1)[0]

        watcher = threading.Thread(target=observe)
        watcher.start()
        served = sim.wait_served(req, timeout_s=10.0)
        watcher.join()
        assert seen["miss"] == piece_word(req.gen) and not seen["served"]  # observed inside the read's delay
        assert seen["delta"] == req.chain
        assert served and sim.piece_word(req, 1) == piece_word(req.gen, FULL)
    finally:
        host.stop()


def test_a_double_publish_inside_the_tier_aborts_the_process(tmp_path):
    """U6 through the service: the refused publish fails the read, which fails stop before the pieces say landed."""
    result = run_host_script(
        tmp_path,
        """
        host.inject_fault(publish_twice=2)
        sim.post(1, [4, 5])
        host.pump()
        print("reached")
        """,
        staging=2,
    )
    assert_aborted(result, "the read failed")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
