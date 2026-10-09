"""The tier-wide per-drive in-flight accounting, observe-only (design 2026-10-09-dsv41-drive-aware-reads, step 1;
host/drive_load.h): every reader of the tier counts its sub-reads and bytes in flight per root (file % parts), split
into demand and speculative, with the time each kind, and both at once, kept a root busy. Every read() return leaves a
reader's share at 0, failed and throwing reads included, so the shared counts are the sum of the readers' shares.

Mutant: skip the subtract in ReaderCore::drain -- red (the failed reads leave their sub-reads counted)."""

import errno
import json

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_drive_load
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

ROOTS = 3
ROW = 1
INSTANT = ("demand_reads", "spec_reads", "demand_inflight_bytes", "spec_inflight_bytes")


@pytest.fixture
def setup(tmp_path):
    # Multi-page rows over three roots, so piece streaming cuts each part into several sub-reads.
    return ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0,) * ROOTS, hidden=256, inter=512)


def expected_bytes(tables, row, experts):
    """Per root, the bytes the drives return for `experts` of `row`: each part's extent clamped at end of file."""
    out = [0] * int(tables.extents.shape[2])
    for e in experts:
        for part in range(len(out)):
            file, offset, length, _dest = (int(v) for v in tables.extents[row, e, part])
            if length > 0:
                out[file % len(out)] += min(length, int(tables.file_sizes[file]) - offset)
    return out


def idle(snap):
    """Nothing in flight on any root."""
    return all(root[k] == 0 for root in snap["roots"] for k in INSTANT)


def landed(snap, kind):
    return [root[f"{kind}_bytes"] for root in snap["roots"]]


def rows_exact(s, row, experts, slots):
    oracle = s.reference(s.tables.layer_ids[row], experts)
    return all(same_bytes(s.slabs[row][name][slot], oracle[name][i])
               for name in oracle for i, slot in enumerate(slots))


@pytest.mark.parametrize("piece_stream", [False, True])
def test_a_demand_read_counts_its_bytes_per_root_and_returns_idle(setup, piece_stream):
    experts, slots = [0, 3, 5], [0, 1, 2]
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, piece_stream=piece_stream)
    assert got["a"] == 1 and rows_exact(setup, ROW, experts, slots)
    after = got["after_a"]
    assert len(after["roots"]) == ROOTS and idle(after), after
    want = expected_bytes(setup.tables, ROW, experts)
    assert all(want) and landed(after, "demand") == want, (landed(after, "demand"), want)
    assert landed(after, "spec") == [0] * ROOTS
    for root in after["roots"]:
        assert root["demand_busy_ns"] > 0 and root["spec_busy_ns"] == 0 and root["overlap_ns"] == 0, root
        assert root["demand_busy_ns"] <= after["elapsed_ns"], root
    assert after["clock_reads"] > 0


def test_a_speculative_read_is_counted_as_speculative(setup):
    experts, slots = [2, 6], [3, 4]
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, a_spec=True, piece_stream=True)
    assert got["a"] == 1 and idle(got["after_a"])
    assert landed(got["after_a"], "spec") == expected_bytes(setup.tables, ROW, experts)
    assert landed(got["after_a"], "demand") == [0] * ROOTS
    for root in got["after_a"]["roots"]:
        assert root["spec_busy_ns"] > 0 and root["demand_busy_ns"] == 0 and root["overlap_ns"] == 0, root


FAULTS = {
    "cqe_error": dict(cqe_error=errno.EIO, cqe_call=2),
    "part_error": dict(part=1, part_error=errno.EIO),
    "part_error_pieces": dict(part=2, part_error=errno.EIO, piece_stream=True),
    "part_short_eof": dict(part=1, part_short=4096, short_is_eof=True, piece_stream=True),
    # Reads still in flight when the read fails: the drain settles them, and no completion reaches process().
    "submit_error_in_flight": dict(submit_error=errno.EIO, submit_call=2, submit_first=True, step=1),
    "held_then_error": dict(hold_ordinal=0, cqe_error=errno.EIO, cqe_call=3, piece_stream=True),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_every_failed_read_leaves_the_shared_counts_at_zero(setup, fault):
    experts, slots = [0, 1, 2, 3], [0, 1, 2, 3]
    then_experts, then_slots = [4, 7], [4, 5]
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, then_experts, then_slots, **FAULTS[fault])
    assert got["a"] == 0 and not got["a_raised"], got
    assert idle(got["after_a"]), got["after_a"]
    # The next reader over the same load counts from a clean slate and returns idle too.
    assert got["b"] == 1 and idle(got["after_b"]), got["after_b"]
    b_bytes = [y - x for x, y in zip(landed(got["after_a"], "demand"), landed(got["after_b"], "demand"))]
    assert b_bytes == expected_bytes(setup.tables, ROW, then_experts)


def test_a_short_completion_resubmitted_counts_every_byte_once(setup):
    experts, slots = [0, 1, 2], [0, 1, 2]
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, part=1, part_short=4096, piece_stream=True)
    assert got["a"] == 1 and rows_exact(setup, ROW, experts, slots)
    assert idle(got["after_a"]) and landed(got["after_a"], "demand") == expected_bytes(setup.tables, ROW, experts)


def test_a_read_whose_guard_throws_leaves_the_shared_counts_at_zero(setup):
    """The failure path's ring reset throws ("io_uring ring reset failed") with reads prepared: read() rethrows, and
    the reader's share was taken back before the throw."""
    got = read_rows_drive_load(
        setup.tables, ROW, [0, 1, 2, 3], [0, 1, 2, 3], [4], [4],
        submit_error=errno.EIO, submit_call=1, ring_reset_fail=True)
    assert got["a_raised"] and got["a"] == -2, got
    assert idle(got["after_a"]), got["after_a"]
    assert got["b"] == 1 and idle(got["after_b"])


@pytest.mark.parametrize("b_spec", [False, True])
def test_two_readers_over_one_load_sum_and_a_speculative_read_overlaps_a_demand(setup, b_spec):
    """Reader A holds row 0's completions (a slow drive), so its demand sub-reads stay in flight on every root while
    reader B reads to the end inside A's read. The shared counts are then exactly A's share (B returned at 0), and B's
    whole busy time on each root is overlap when B is speculative."""
    experts, slots = [0, 1, 2, 3], [0, 1, 2, 3]
    then_experts, then_slots = [4, 5], [4, 5]
    got = read_rows_drive_load(
        setup.tables, ROW, experts, slots, then_experts, then_slots,
        b_spec=b_spec, nested=True, hold_ordinal=0, piece_stream=True)
    assert got["a"] == 1 and got["b"] == 1, got
    assert rows_exact(setup, ROW, experts + then_experts, slots + then_slots)
    nested = got["nested"]
    assert got["a_in_flight"] > 0
    assert sum(root["demand_reads"] for root in nested["roots"]) == got["a_in_flight"], nested
    assert all(root["spec_reads"] == 0 and root["spec_inflight_bytes"] == 0 for root in nested["roots"])
    b_kind = "spec" if b_spec else "demand"
    assert landed(nested, b_kind) == expected_bytes(setup.tables, ROW, then_experts)
    for root in nested["roots"]:
        assert root["demand_reads"] > 0, root  # row 0 has a sub-read on every root
        if b_spec:
            assert root["spec_busy_ns"] > 0 and root["overlap_ns"] == root["spec_busy_ns"], root
        else:
            assert root["overlap_ns"] == 0, root
    after = got["after_b"]
    assert idle(after)
    want = [x + y for x, y in zip(expected_bytes(setup.tables, ROW, experts),
                                  expected_bytes(setup.tables, ROW, then_experts) if not b_spec else [0] * ROOTS)]
    assert landed(after, "demand") == want
    for root in after["roots"]:
        assert root["overlap_ns"] <= min(root["demand_busy_ns"], root["spec_busy_ns"]), root


def test_the_production_build_counts_too_and_refuses_a_fault(setup):
    experts, slots = [1, 4], [0, 1]
    got = read_rows_drive_load(setup.tables, ROW, experts, slots, piece_stream=True, variant="prod")
    assert got["a"] == 1 and idle(got["after_a"])
    assert landed(got["after_a"], "demand") == expected_bytes(setup.tables, ROW, experts)
    with pytest.raises(RuntimeError, match="instrumented"):
        read_rows_drive_load(setup.tables, ROW, experts, slots, cqe_error=errno.EIO, cqe_call=1, variant="prod")


def _row_bytes(tables, row, experts):
    return sum(expected_bytes(tables, row, experts))


@pytest.mark.parametrize("nodes", [1, 2])
def test_every_group_of_the_tier_counts_into_its_one_load(tmp_path, nodes):
    """Demand misses of both NUMA groups land in the tier's load; spec_place's pool read counts as speculative."""
    rig = prefetch_rig(tmp_path, nodes=nodes, share=1)
    try:
        tables = rig.setup.tables
        before = rig.host.drive_load()
        assert idle(before) and sum(landed(before, "demand")) == 0
        experts = [0, 1]  # with two nodes, one homed on each group (home = expert % nodes)
        load(rig, 0, experts)
        after = rig.host.drive_load()
        assert idle(after)
        assert sum(landed(after, "demand")) == _row_bytes(tables, 0, experts), after
        assert sum(landed(after, "spec")) == 0
        rig.host.spec_place(1, 3)
        pooled = rig.host.drive_load()
        assert idle(pooled)
        assert sum(landed(pooled, "spec")) == _row_bytes(tables, 1, [3])
        assert landed(pooled, "demand") == landed(after, "demand")
        assert all(root["spec_busy_ns"] > 0 for root in pooled["roots"])
    finally:
        rig.host.stop()


def test_stop_logs_the_drive_load_line(tmp_path, capfd):
    rig = prefetch_rig(tmp_path)
    load(rig, 0, [2])
    want = rig.host.drive_load()
    rig.host.stop()
    lines = [line for line in capfd.readouterr().err.splitlines() if line.startswith("exl3 RAM miss drive load ")]
    assert len(lines) == 1, lines
    logged = json.loads(lines[0][len("exl3 RAM miss drive load "):])
    assert [root["demand_bytes"] for root in logged["roots"]] == [root["demand_bytes"] for root in want["roots"]]
    assert idle(logged) and logged["elapsed_ns"] >= want["elapsed_ns"]
