"""The per-drive in-flight cap (design 2026-10-09-dsv41-drive-aware-reads, change (3), counts only): with
SGLANG_MOE_EXPERT_MIRROR_DYNAMIC each piece-stream sub-read picks its mirror root when it is first prepared. A root is
open while its sub-reads in flight (demand and speculative, every reader of the tier's DriveLoad) are below its cap;
among the open roots the one with the fewest bytes in flight wins, ties going to the sub-read's own root, and with every
root at its cap the sub-read keeps its own root (never blocks). Only the file changes: offset, length and destination
are the static geometry's, because every root holds the same image at the same offsets.

The tests preload the shared DriveLoad (``preload``: reads and bytes per root, taken back when reader A returns) and,
where the choice must be deterministic, issue one SQE at a time (``max_outstanding=1``), so every choice sees only the
preload.

Mutants: invert the cap comparison, or ignore the cap -- red (test_a_capped_root_receives_no_sub_read_while_another_is_
open, test_every_root_at_its_cap_keeps_the_static_roots_without_blocking)."""

import dataclasses
import errno
import json

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_drive_load, read_rows_sqes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import load, prefetch_rig

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROOTS = 3
ROW = 1
EXPERTS, SLOTS = [0, 3, 5, 6], [0, 1, 2, 3]
INSTANT = ("demand_reads", "spec_reads", "demand_inflight_bytes", "spec_inflight_bytes")


@pytest.fixture
def setup(tmp_path):
    # Multi-page rows over three roots: piece streaming cuts each part into two sub-reads.
    return ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0,) * ROOTS, hidden=256, inter=512)


def expected_bytes(tables, row, experts):
    """Per root, the bytes the static split reads for `experts` of `row` (each part clamped at end of file)."""
    out = [0] * int(tables.extents.shape[2])
    for e in experts:
        for part in range(len(out)):
            file, offset, length, _dest = (int(v) for v in tables.extents[row, e, part])
            if length > 0:
                out[file % len(out)] += min(length, int(tables.file_sizes[file]) - offset)
    return out


def static_sub_reads(tables, row, experts):
    """Per root, the sub-reads the static split issues (one SQE each: no cuts, no faults)."""
    result, sqes, _, _ = read_rows_sqes(tables, row, experts, list(range(len(experts))), piece_stream=True)
    assert result == 1
    parts = int(tables.extents.shape[2])
    out = [0] * parts
    for file, *_ in sqes:
        out[file % parts] += 1
    return out


def idle(snap):
    return all(root[k] == 0 for root in snap["roots"] for k in INSTANT)


def field(snap, name):
    return [root[name] for root in snap["roots"]]


def rows_exact(s, row, experts, slots):
    oracle = s.reference(s.tables.layer_ids[row], experts)
    return all(same_bytes(s.slabs[row][name][slot], oracle[name][i]) for name in oracle for i, slot in enumerate(slots))


def preload_reads(*reads):
    return [(r, 0) for r in reads]


def test_an_idle_load_keeps_every_sub_read_on_its_own_root(setup):
    """One SQE at a time sees an empty load, so every choice is a tie and goes to the sub-read's own root: the SQEs
    are the static ones."""
    _, static, _, _ = read_rows_sqes(setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, max_outstanding=1)
    result, dynamic, _, _ = read_rows_sqes(
        setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, max_outstanding=1, mirror_caps=(4, 4, 4))
    assert result == 1 and dynamic == static


def test_a_capped_root_receives_no_sub_read_while_another_is_open(setup):
    want = expected_bytes(setup.tables, ROW, EXPERTS)
    moved = static_sub_reads(setup.tables, ROW, EXPERTS)[1]
    got = read_rows_drive_load(
        setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, poison=True, max_outstanding=1,
        mirror_caps=(8, 2, 8), preload=preload_reads(0, 2, 0))
    assert got["a"] == 1 and rows_exact(setup, ROW, EXPERTS, SLOTS)
    after = got["after_a"]
    assert idle(after), after
    # Root 1 holds its cap: its sub-reads go to the open root with the fewest bytes, root 0 (lowest of a tie).
    assert field(after, "demand_bytes") == [want[0] + want[1], 0, want[2]], after
    assert field(after, "moved_from") == [0, moved, 0] and field(after, "moved_to") == [moved, 0, 0], after


def test_every_root_at_its_cap_keeps_the_static_roots_without_blocking(setup):
    got = read_rows_drive_load(
        setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, poison=True,
        mirror_caps=(1, 1, 1), preload=preload_reads(1, 1, 1))
    assert got["a"] == 1 and rows_exact(setup, ROW, EXPERTS, SLOTS)
    after = got["after_a"]
    assert idle(after)
    assert field(after, "demand_bytes") == expected_bytes(setup.tables, ROW, EXPERTS)
    assert field(after, "moved_from") == [0] * ROOTS and field(after, "moved_to") == [0] * ROOTS


def test_among_open_roots_the_fewest_bytes_in_flight_wins(setup):
    want = expected_bytes(setup.tables, ROW, EXPERTS)
    got = read_rows_drive_load(
        setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, max_outstanding=1,
        mirror_caps=(64, 64, 64), preload=[(0, 1 << 40), (0, 0), (0, 0)])
    assert got["a"] == 1 and rows_exact(setup, ROW, EXPERTS, SLOTS)
    after = got["after_a"]
    assert idle(after)
    assert field(after, "demand_bytes") == [0, want[0] + want[1], want[2]], after


def test_concurrent_sub_reads_under_low_caps_read_the_eager_bytes(setup):
    """Every SQE of the batch in flight at once: the choice follows the reader's own load as it grows."""
    experts, slots = list(range(8)), list(range(8))
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, piece_stream=True, poison=True, mirror_caps=(2, 2, 2))
    assert got["a"] == 1 and rows_exact(setup, ROW, experts, slots)
    after = got["after_a"]
    assert idle(after)
    assert sum(field(after, "demand_bytes")) == sum(expected_bytes(setup.tables, ROW, experts))
    assert sum(field(after, "moved_from")) == sum(field(after, "moved_to")) > 0, after


def test_every_sub_read_redirected_reads_the_eager_bytes(tmp_path):
    """Two roots, the second with weight 0: every sub-read is root 0's, and with root 0 at its cap every one of them
    reads root 1's copy of its row."""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 0.0), hidden=256, inter=512)
    experts, slots = [1, 2, 4, 7], [0, 1, 2, 3]
    subs = static_sub_reads(s.tables, ROW, experts)
    assert subs[1] == 0 and subs[0] > 0
    for row in (0, 1):
        got = read_rows_drive_load(
            s.tables, row, experts, slots, piece_stream=True, poison=True, mirror_caps=(1, 8),
            preload=preload_reads(1, 0))
        assert got["a"] == 1 and rows_exact(s, row, experts, slots), row
        after = got["after_a"]
        assert idle(after)
        assert field(after, "demand_bytes") == [0, sum(expected_bytes(s.tables, row, experts))]
        assert field(after, "moved_from") == [subs[0], 0] and field(after, "moved_to") == [0, subs[0]]


def test_every_sub_read_keeps_its_geometry_and_row_only_the_root_changes(setup):
    _, static, _, _ = read_rows_sqes(setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True)
    result, dynamic, _, _ = read_rows_sqes(setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, mirror_caps=(2, 2, 2))
    assert result == 1
    key = lambda sqe: (sqe[3], sqe[1], sqe[2])  # noqa: E731 -- destination, offset, length
    assert [key(q) for q in sorted(static, key=key)] == [key(q) for q in sorted(dynamic, key=key)]
    by_dest = {key(q): q[0] for q in static}
    assert all(q[0] - q[0] % ROOTS == by_dest[key(q)] - by_dest[key(q)] % ROOTS for q in dynamic)
    assert any(q[0] != by_dest[key(q)] for q in dynamic)


FAULTS = {
    "cqe_error": dict(cqe_error=errno.EIO, cqe_call=2),
    "part_error": dict(part=1, part_error=errno.EIO),
    "part_short_eof": dict(part=1, part_short=4096, short_is_eof=True),
    "submit_error_in_flight": dict(submit_error=errno.EIO, submit_call=2, submit_first=True, step=1),
    "held_then_error": dict(hold_ordinal=0, cqe_error=errno.EIO, cqe_call=3),
    "root_error": dict(root=0, part_error=errno.EIO),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_every_failed_dynamic_read_leaves_the_counts_and_redirects_consistent(setup, fault):
    experts, slots = [0, 1, 2, 3], [0, 1, 2, 3]
    got = read_rows_drive_load(
        setup.tables, ROW, experts, slots, [4, 7], [4, 5], piece_stream=True, mirror_caps=(2, 1, 2),
        preload=preload_reads(0, 1, 0), **FAULTS[fault])
    assert got["a"] == 0 and not got["a_raised"], got
    assert idle(got["after_a"]), got["after_a"]
    assert got["b"] == 1 and idle(got["after_b"]) and rows_exact(setup, ROW, [4, 7], [4, 5])
    for snap in (got["after_a"], got["after_b"]):
        assert sum(field(snap, "moved_from")) == sum(field(snap, "moved_to")), snap


def test_a_root_fault_follows_the_drive_not_the_part(tmp_path):
    """fault.root matches the root a sub-read reads (file % parts); fault.part still matches its part slot. With root
    0 capped every sub-read reads root 1: a root-1 fault fires, a part-1 fault (no sub-read has part 1) cannot."""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 0.0), hidden=256, inter=512)
    experts, slots = [1, 2], [0, 1]
    common = dict(piece_stream=True, mirror_caps=(1, 8), preload=preload_reads(1, 0))
    on_root = read_rows_drive_load(s.tables, ROW, experts, slots, root=1, part_error=errno.EIO, **common)
    assert on_root["a"] == 0 and idle(on_root["after_a"])
    on_part = read_rows_drive_load(s.tables, ROW, experts, slots, part=1, part_error=errno.EIO, **common)
    assert on_part["a"] == 1 and rows_exact(s, ROW, experts, slots)
    static = read_rows_drive_load(s.tables, ROW, experts, slots, root=1, part_error=errno.EIO, piece_stream=True)
    assert static["a"] == 1  # without the cap nothing reads root 1


def test_an_incomplete_mirror_set_is_refused(setup):
    paths = list(setup.tables.source_paths)
    paths[ROOTS + 1] = paths[ROOTS + 1] + ".other"  # row 1's root-1 file names another source
    broken = dataclasses.replace(setup.tables, source_paths=paths)
    with pytest.raises(RuntimeError, match="not a full mirror set"):
        read_rows_drive_load(broken, ROW, [0], [0], piece_stream=True, mirror_caps=(2, 2, 2))
    got = read_rows_drive_load(broken, ROW, [0], [0], piece_stream=True)  # the static split never reads across rows
    assert got["a"] == 1


@pytest.mark.parametrize(
    "caps, piece_stream, match",
    [
        ((2, 2), True, "one in-flight cap per mirror root"),
        ((2, 0, 2), True, "positive"),
        ((2, 2, 2), False, "piece streaming"),
    ],
)
def test_bad_caps_are_refused(setup, caps, piece_stream, match):
    with pytest.raises(RuntimeError, match=match):
        read_rows_drive_load(setup.tables, ROW, [0], [0], piece_stream=piece_stream, mirror_caps=caps)


def test_the_production_build_reads_dynamically(setup):
    want = expected_bytes(setup.tables, ROW, EXPERTS)
    got = read_rows_drive_load(
        setup.tables, ROW, EXPERTS, SLOTS, piece_stream=True, mirror_caps=(64, 2, 64), preload=preload_reads(0, 2, 0),
        variant="prod")
    assert got["a"] == 1 and rows_exact(setup, ROW, EXPERTS, SLOTS)
    after = got["after_a"]
    assert idle(after) and field(after, "demand_bytes")[1] == 0 and sum(field(after, "demand_bytes")) == sum(want)
    assert field(after, "moved_from")[1] == static_sub_reads(setup.tables, ROW, EXPERTS)[1]


def _slot_of(host, row, expert):
    return host.slot_to_expert(row).index(expert)


def test_the_tier_chooses_demand_and_speculative_roots_and_logs_its_redirects(tmp_path, capfd):
    rig = prefetch_rig(tmp_path, mirror_weights=(1.0,) * ROOTS)
    try:
        s = rig.setup
        with pytest.raises(RuntimeError, match="one in-flight cap per mirror root"):
            rig.host.set_mirror_caps((1, 1))
        rig.host.set_mirror_caps((1, 1, 1))
        load(rig, 0, [0, 1])
        oracle = s.reference(s.tables.layer_ids[0], [0, 1])
        for i, expert in enumerate((0, 1)):
            slot = _slot_of(rig.host, 0, expert)
            assert all(same_bytes(s.slabs[0][name][slot], oracle[name][i]) for name in oracle), expert
        after = rig.host.drive_load()
        assert idle(after)
        assert sum(field(after, "moved_from")) == sum(field(after, "moved_to")) > 0, after
        assert sum(field(after, "demand_bytes")) == sum(expected_bytes(s.tables, 0, [0, 1]))
        slot = rig.host.spec_place(1, 3)
        spec_oracle = s.reference(s.tables.layer_ids[1], [3])
        assert all(same_bytes(s.slabs[1][name][slot], spec_oracle[name][0]) for name in spec_oracle)
        pooled = rig.host.drive_load()
        assert idle(pooled) and sum(field(pooled, "spec_bytes")) == sum(expected_bytes(s.tables, 1, [3]))
        assert sum(field(pooled, "moved_from")) > sum(field(after, "moved_from")), pooled
        want = rig.host.drive_load()
    finally:
        rig.host.stop()
    lines = [line for line in capfd.readouterr().err.splitlines() if line.startswith("exl3 RAM miss drive load ")]
    logged = json.loads(lines[-1][len("exl3 RAM miss drive load "):])
    assert field(logged, "moved_from") == field(want, "moved_from")
    assert field(logged, "moved_to") == field(want, "moved_to")


def test_turning_the_caps_off_restores_the_static_roots(tmp_path):
    rig = prefetch_rig(tmp_path, mirror_weights=(1.0,) * ROOTS)
    try:
        rig.host.set_mirror_caps((1, 1, 1))
        rig.host.set_mirror_caps(())
        load(rig, 0, [0, 1])
        after = rig.host.drive_load()
        assert field(after, "moved_from") == [0] * ROOTS
        assert field(after, "demand_bytes") == expected_bytes(rig.setup.tables, 0, [0, 1])
    finally:
        rig.host.stop()
