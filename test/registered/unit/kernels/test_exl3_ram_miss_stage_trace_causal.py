"""Per-row causal stamps, drop locations and the disabled-trace bypass of the RAM-miss stage trace (CPU).

Ordering only: nothing here asserts a wall-time bound.
"""

import collections
import errno
import faulthandler
from pathlib import Path

import pytest
import torch

import sglang.kernels.ops.moe.exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, read_rows_traced, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 4096
CPP = Path(ops.__file__).resolve().parents[2] / "jit" / "csrc" / "moe" / "exl3_ram_miss_host.cpp"


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _host(tmp_path, *, trace_capacity=None, capacity=6):
    """A tier and its host; ``trace_capacity`` None leaves the trace off."""
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, capacity), -1, dtype=torch.int32), direct=False)
    if trace_capacity is not None:
        host.enable_trace(capacity=trace_capacity)
    return s, page, host


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    assert sim_wait(page, seq, timeout_s=1.0) == 1
    return seq


def _extents_by_row(record):
    by_row = {}
    for extent in record["extent_cqe"]:
        by_row.setdefault(extent["row"], []).append(extent)
    return by_row


def _assert_row_chain(record, rows=None):
    """One row's own stamps, never one sorted list across rows: admit <= each of its extents' submit <=
    that extent's cqe <= the row's pack_start <= pack_end. Every stamp of a packed row must exist."""
    by_row = _extents_by_row(record)
    for row in record["row_pack"] if rows is None else rows:
        assert row["admit"] > 0 and row["start"] > 0 and row["end"] >= row["start"], row
        for extent in by_row[row["row"]]:
            assert row["admit"] <= extent["submit"] <= extent["cqe"] <= row["start"], (row, extent)


@pytest.mark.parametrize("weights", [None, (1.0, 1.0)])
@pytest.mark.parametrize(
    "fault",
    [{}, dict(reverse_cqes=True), dict(max_outstanding=3), dict(pack_delay_ns=500_000)],
)
def test_a_rows_chain_holds_while_a_global_stage_order_does_not(tmp_path, weights, fault):
    """Row 5's completions are withheld (a slow drive), so rows 0-4 pack while it is still outstanding.
    Each row's own chain holds. The order 'every completion before any packing' does not, which is why
    the chain is checked per row."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=weights)
    result, record = read_rows_traced(
        s.tables, 1, [3, 0, 5, 1, 4, 2], list(range(6)), direct=False, hold_ordinal=5, **fault
    )
    assert result == 1 and record["status"] == "served"
    _assert_row_chain(record)
    last_cqe = max(extent["cqe"] for extent in record["extent_cqe"])
    packing_before_last_cqe = [row["row"] for row in record["row_pack"] if row["start"] < last_cqe]
    assert packing_before_last_cqe == [0, 1, 2, 3, 4]


def test_the_first_prepared_extent_precedes_the_first_submit(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2, 3], list(range(4)), direct=False)
    assert result == 1
    assert 0 < min(extent["submit"] for extent in record["extent_cqe"]) <= record["submit"] <= record["first_cqe"]
    assert all(extent["attempts"] == 0 for extent in record["extent_cqe"])
    _assert_row_chain(record)


def test_a_credit_bound_read_prepares_its_extents_in_later_turns(tmp_path):
    """One SQE of credit: extents are prepared turn by turn, so their submit stamps are issue-ordered
    and not all one instant, and each is still before its own completion."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2, 3], list(range(4)), direct=False, max_outstanding=1)
    assert result == 1
    submits = [extent["submit"] for extent in record["extent_cqe"]]
    assert len(set(submits)) > 1 and submits == sorted(submits)
    _assert_row_chain(record)


@pytest.mark.parametrize(
    "fault",
    [dict(cqe_error=errno.EINTR, cqe_call=1), dict(part=0, part_short=PAGE)],
    ids=["interrupted", "short_read"],
)
def test_a_resubmitted_extent_counts_an_attempt_and_keeps_its_first_submit(tmp_path, fault):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2], list(range(3)), direct=False, **fault)
    assert result == 1
    assert record["retried_bytes"] > 0, "the fault did not fire"
    assert sum(extent["attempts"] for extent in record["extent_cqe"]) >= 1
    _assert_row_chain(record)


def test_a_failure_after_some_rows_packed_marks_exactly_the_stages_not_reached(tmp_path):
    """One SQE of credit, five rows, row 3's extent fails. Rows 0-2 were read and packed; row 3 was
    admitted and prepared but never completed; row 4 was admitted but never prepared."""
    s = ram_miss_setup(tmp_path, capacity=6)
    result, record = read_rows_traced(
        s.tables, 1, [0, 1, 2, 3, 4], list(range(5)), direct=False,
        max_outstanding=1, part=0, part_error=errno.EIO, ordinal=3,
    )
    assert result == 0 and record["status"] == "failed" and record["ok"] == 0
    packs = {row["row"]: row for row in record["row_pack"]}
    extents = {extent["row"]: extent for extent in record["extent_cqe"]}
    assert sorted(packs) == [0, 1, 2, 3, 4]
    _assert_row_chain(record, rows=[packs[k] for k in (0, 1, 2)])
    assert (packs[3]["start"], packs[3]["end"], extents[3]["cqe"]) == (0, 0, 0)
    assert packs[3]["admit"] > 0 and extents[3]["submit"] > 0
    assert (packs[4]["start"], packs[4]["end"], extents[4]["cqe"], extents[4]["submit"]) == (0, 0, 0, 0)
    assert packs[4]["admit"] > 0
    assert record["useful_bytes"] > 0 and record["cancelled_bytes"] > 0


def test_a_served_request_is_reserved_before_any_row_is_admitted(tmp_path):
    s, page, host = _host(tmp_path, trace_capacity=8)
    try:
        _serve(page, host, 1, need=[2, 5, 4], protect=[2, 5, 4])
        (record,) = host.drain_trace()
    finally:
        host.stop()
    assert record["observed"] <= record["reserved"] <= min(row["admit"] for row in record["row_pack"])
    assert record["dropped_before"] == 0
    _assert_row_chain(record)


def test_a_full_ring_says_where_the_records_were_lost(tmp_path):
    s, page, host = _host(tmp_path, trace_capacity=2)
    try:
        seqs = [_serve(page, host, 0, need=[], protect=[1]) for _ in range(5)]
        first = host.drain_trace()
        assert [r["seq"] for r in first] == seqs[:2] and [r["dropped_before"] for r in first] == [0, 0]
        assert host.trace_dropped() == 3
        after_seq = _serve(page, host, 0, need=[], protect=[1])  # the ring was drained: this one gets in
        (after,) = host.drain_trace()
        assert after["seq"] == after_seq and after["dropped_before"] == 3
        _serve(page, host, 0, need=[], protect=[1])
        (later,) = host.drain_trace()
        assert later["dropped_before"] == 0  # reported once, not cumulatively
    finally:
        host.stop()


def test_each_loss_episode_is_reported_at_its_own_position(tmp_path):
    s, page, host = _host(tmp_path, trace_capacity=1)
    try:
        seen = []
        for lost in (2, 3):
            for _ in range(1 + lost):  # the first fills the single slot, the rest are dropped
                _serve(page, host, 0, need=[], protect=[1])
            seen.extend(host.drain_trace())
        _serve(page, host, 0, need=[], protect=[1])
        seen.extend(host.drain_trace())
        assert [r["dropped_before"] for r in seen] == [0, 2, 3]
        assert sum(r["dropped_before"] for r in seen) == host.trace_dropped() == 5
    finally:
        host.stop()


def _mixed_traffic(page, host):
    _serve(page, host, 1, need=[2, 5], protect=[2, 5])  # reads
    _serve(page, host, 1, need=[], protect=[2])  # resident: no read
    _serve(page, host, 1, need=[3], protect=[2, 3, 5])  # reads one row


def test_a_disabled_trace_reads_no_clock_and_builds_no_record(tmp_path):
    s, page, host = _host(tmp_path)
    try:
        before = host.trace_clock_reads()
        _mixed_traffic(page, host)
        assert host.trace_clock_reads() == before
        assert host.drain_trace() == [] and host.trace_dropped() == 0
    finally:
        host.stop()


def test_an_enabled_trace_reads_the_clock_at_every_stamp(tmp_path):
    """A request with nothing to read has exactly four stamps: observed, reserved, mapped, done. That the
    disabled run above reads zero is only meaningful because this counter does move when tracing is on."""
    s, page, host = _host(tmp_path, trace_capacity=64)
    try:
        _serve(page, host, 1, need=[2], protect=[2])
        before = host.trace_clock_reads()
        for _ in range(3):
            _serve(page, host, 1, need=[], protect=[2])
        assert host.trace_clock_reads() - before == 3 * 4
        before = host.trace_clock_reads()
        _serve(page, host, 1, need=[5], protect=[2, 5])
        assert host.trace_clock_reads() - before > 4  # a read stamps admission, submit, cqe and packing too
    finally:
        host.stop()


# Every clock read in the file that is not a trace stamp. A new one fails this test until it is either
# routed through stamp() (so the disabled trace skips it) or added here as a deliberate non-trace read.
NON_TRACE_CLOCK_READS = {
    "inline int64_t now_ns() {": 1,  # the clock itself
    "return now_ns();": 1,  # stamp() itself
    "busy_since_.store(now_ns());": 2,  # the watchdog's stuck-request marker, needed with the trace off
    "const int64_t deadline = now_ns() + timeout_ns;": 2,
    "if (now_ns() > deadline) {": 2,
    "const int64_t deadline = now_ns() + duration_ns;": 1,
    "while (now_ns() < deadline) {": 1,
    "if (now_ns() - last_active < spin_ns_) {": 1,
    # RamThread's watchdog poll (unchanged) and RowReader::read()'s progress-callback gate (phase 1,
    # ram-miss-progress-phase1): only reached when a `progress` callback was passed, so the disabled
    # (production-off) path still reads no clock when nothing is registered to run periodically.
    "const int64_t now = now_ns();": 2,
    # The hold_until_probe_ms test fault (piece streaming, G2): read only when that fault is set; the second
    # line's `c.hold_until == 0` short-circuits before the clock read otherwise.
    "c.hold_until = now_ns() + fault_.hold_until_probe_ms * 1000000;": 1,
    "if (c.hold_until == 0 || c.failed || now_ns() >= c.hold_until) return false;": 1,
    # The copy engine (LEASE_PROTOCOL.md 7.6), reached only with it enabled: its thread's spin and drain clock (the
    # second "last_active" pair), the idle waits of pause() and of the test hook, and four reads per copy job for the
    # always-on copy_issue_ns and copy_latency_ns counters, which are how stages.jsonl times the copies.
    "int64_t last_active = now_ns();": 2,
    "last_active = now_ns();": 2,
    "} else if (!in_flight.empty() || now_ns() - last_active < spin_ns_) {": 1,
    "if (stopping && (in_flight.empty() || now_ns() > drain_deadline)) break;": 1,
    "drain_deadline_ = now_ns() + drain_ns;": 1,
    "if (now_ns() > deadline_ns) return false;": 1,
    "tier_->wait_copy_idle(now_ns() + timeout_ns);": 1,
    "return exl3_ram_miss::find(handle)->wait_copy_idle(exl3_ram_miss::now_ns() + timeout_ns) ? 1 : 0;": 1,
    "job.submit_ns = now_ns();": 1,
    "const int64_t start = now_ns();": 1,
    "counters_[kCopyIssueNs].fetch_add(now_ns() - start);": 1,
    "const int64_t latency = now_ns() - job.submit_ns;": 1,
}


def test_no_clock_read_bypasses_the_trace_gate():
    lines = [line.strip() for line in CPP.read_text().splitlines()]
    reads = collections.Counter(line for line in lines if "now_ns()" in line and not line.startswith("//"))
    assert reads == NON_TRACE_CLOCK_READS, {
        line: count for line, count in (reads - collections.Counter(NON_TRACE_CLOCK_READS)).items()
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
