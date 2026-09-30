"""Per-row causal stamps, drop locations and the disabled-trace bypass of the RAM-miss stage trace (CPU).

Ordering only: nothing here asserts a wall-time bound.
"""

import collections
import errno
import faulthandler
import re

import pytest
import torch

import sglang.kernels.ops.moe.expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, read_rows_traced
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
from sglang.test.expert_stream_sources import host_sources, joined_text

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 4096


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _host(tmp_path, *, trace_capacity=None, capacity=6):
    """A tier and its host; ``trace_capacity`` None leaves the trace off."""
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, capacity), -1, dtype=torch.int32))
    if trace_capacity is not None:
        host.enable_trace(capacity=trace_capacity)
    return s, page, host


def _serve(page, host, row, lanes, protect=None):
    """One demand for ``lanes``, served, then Done (the next pump retires its leases); returns its seq."""
    sim = LeaseSim(host, page, None)
    req = sim.post(row, lanes, protect=protect)
    assert host.pump() == 1
    assert sim.wait(req, timeout_s=1.0).served
    sim.done(req)
    return req.seq


def _extents_by_row(record):
    by_row = {}
    for extent in record["extent_cqe"]:
        by_row.setdefault(extent["row"], []).append(extent)
    return by_row


def _assert_row_chain(record, rows=None, *, pieces=False):
    """One row's own stamps, never one sorted list across rows: admit <= each of its extents' submit <=
    that extent's cqe <= the row's pack_start <= pack_end. Every stamp of a packed row must exist. With ``pieces``
    (the service streams them) a row starts packing once its first sub-read landed, so only the first cqe precedes
    pack_start, and every cqe precedes pack_end."""
    by_row = _extents_by_row(record)
    for row in record["row_pack"] if rows is None else rows:
        assert row["admit"] > 0 and row["start"] > 0 and row["end"] >= row["start"], row
        extents = by_row[row["row"]]
        for extent in extents:
            assert row["admit"] <= extent["submit"] <= extent["cqe"] <= (row["end"] if pieces else row["start"]), (
                row, extent)
        assert min(extent["cqe"] for extent in extents) <= row["start"], row


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
        s.tables, 1, [3, 0, 5, 1, 4, 2], list(range(6)), hold_ordinal=5, **fault
    )
    assert result == 1 and record["status"] == "served"
    _assert_row_chain(record)
    last_cqe = max(extent["cqe"] for extent in record["extent_cqe"])
    packing_before_last_cqe = [row["row"] for row in record["row_pack"] if row["start"] < last_cqe]
    assert packing_before_last_cqe == [0, 1, 2, 3, 4]


def test_the_first_prepared_extent_precedes_the_first_submit(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2, 3], list(range(4)))
    assert result == 1
    assert 0 < min(extent["submit"] for extent in record["extent_cqe"]) <= record["submit"] <= record["first_cqe"]
    assert all(extent["attempts"] == 0 for extent in record["extent_cqe"])
    _assert_row_chain(record)


def test_a_credit_bound_read_prepares_its_extents_in_later_turns(tmp_path):
    """One SQE of credit: extents are prepared turn by turn, so their submit stamps are issue-ordered
    and not all one instant, and each is still before its own completion."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2, 3], list(range(4)), max_outstanding=1)
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
    result, record = read_rows_traced(s.tables, 1, [0, 1, 2], list(range(3)), **fault)
    assert result == 1
    assert record["retried_bytes"] > 0, "the fault did not fire"
    assert sum(extent["attempts"] for extent in record["extent_cqe"]) >= 1
    _assert_row_chain(record)


def test_a_failure_after_some_rows_packed_marks_exactly_the_stages_not_reached(tmp_path):
    """One SQE of credit, five rows, row 3's extent fails. Rows 0-2 were read and packed; row 3 was
    admitted and prepared but never completed; row 4 was admitted but never prepared."""
    s = ram_miss_setup(tmp_path, capacity=6)
    result, record = read_rows_traced(
        s.tables, 1, [0, 1, 2, 3, 4], list(range(5)), max_outstanding=1, part=0, part_error=errno.EIO, ordinal=3,
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
        _serve(page, host, 1, [2, 5, 4])
        (record,) = host.drain_trace()
    finally:
        host.stop()
    assert record["observed"] <= record["reserved"] <= min(row["admit"] for row in record["row_pack"])
    assert record["dropped_before"] == 0
    _assert_row_chain(record, pieces=True)


def test_a_full_ring_says_where_the_records_were_lost(tmp_path):
    s, page, host = _host(tmp_path, trace_capacity=2)
    try:
        seqs = [_serve(page, host, 0, [1]) for _ in range(5)]
        first = host.drain_trace()
        assert [r["seq"] for r in first] == seqs[:2] and [r["dropped_before"] for r in first] == [0, 0]
        assert host.trace_dropped() == 3
        after_seq = _serve(page, host, 0, [1])  # the ring was drained: this one gets in
        (after,) = host.drain_trace()
        assert after["seq"] == after_seq and after["dropped_before"] == 3
        _serve(page, host, 0, [1])
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
                _serve(page, host, 0, [1])
            seen.extend(host.drain_trace())
        _serve(page, host, 0, [1])
        seen.extend(host.drain_trace())
        assert [r["dropped_before"] for r in seen] == [0, 2, 3]
        assert sum(r["dropped_before"] for r in seen) == host.trace_dropped() == 5
    finally:
        host.stop()


def _mixed_traffic(page, host):
    _serve(page, host, 1, [2, 5])  # reads
    _serve(page, host, 1, [2])  # resident: no read
    _serve(page, host, 1, [3], protect=[2, 3, 5])  # reads one row


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
        _serve(page, host, 1, [2])
        before = host.trace_clock_reads()
        for _ in range(3):
            _serve(page, host, 1, [2])
        assert host.trace_clock_reads() - before == 3 * 4
        before = host.trace_clock_reads()
        _serve(page, host, 1, [5], protect=[2, 5])
        assert host.trace_clock_reads() - before > 4  # a read stamps admission, submit, cqe and packing too
    finally:
        host.stop()


# Every clock read in the file that is not a trace stamp: now_ns(), and any direct libc or std::chrono clock read
# (CLOCK_READ below). A new one fails this test until it is either routed through stamp() (so the disabled trace
# skips it) or added here as a deliberate non-trace read.
NON_TRACE_CLOCK_READS = {
    "inline int64_t now_ns() {": 1,  # the clock itself
    "clock_gettime(CLOCK_MONOTONIC, &ts);": 1,  # now_ns()'s body
    "return now_ns();": 1,  # stamp() itself
    # Deadlines of callers that wait, never the service or copy thread serving a request: RamThread::pause() (and its
    # copy-idle wait), and fill_wait (the second "deadline" line and the "return -1" line).
    "const int64_t deadline = now_ns() + timeout_ns;": 2,
    "if (now_ns() > deadline) {": 1,
    "tier_->wait_copy_idle_owned(now_ns() + timeout_ns);": 1,
    "if (now_ns() > deadline) return -1;": 1,
    # The spin budget (spec M8): idle_budget() times kProbe pauses once, on the thread that calls start() (RamThread and
    # CopyEngine), so neither the service nor the copy thread reads the clock to pace itself.
    "const int64_t t0 = now_ns();": 1,
    "const int64_t per_pause = std::max<int64_t>(1, (now_ns() - t0) / kProbe);": 1,
    # The watchdog's poll (D6): it times how long one busy episode persists, on its own thread.
    "const int64_t now = now_ns();": 1,
    # The copy engine (LEASE_PROTOCOL.md 7.6): its idle waits (CopyEngine::wait_idle, and the FFI's copy_engine_idle
    # through it) and stop()'s drain deadline, which the copy thread reads only once a stop was asked for.
    "if (now_ns() > deadline_ns) return false;": 1,
    "return find(handle)->wait_copy_idle(expert_stream::now_ns() + timeout_ns) ? 1 : 0;": 1,
    "drain_deadline_.store(now_ns() + drain_ns, std::memory_order_relaxed);": 1,
    "if (stopping && ((in_flight.empty() && acking.empty()) || now_ns() > drain_deadline())) break;": 1,
    # Metrics, compiled only into InstrBuild (each inside `if constexpr (Build::kMetrics)`): copy_issue_ns, and the copy
    # latency's submit and completion reads.
    "if constexpr (Build::kMetrics) start = now_ns();  // copy_issue_ns, a metric": 1,
    "count<kCopyIssueNs>(now_ns() - start);": 1,
    "if constexpr (Build::kMetrics) job.submit_ns = now_ns();  // copy_latency_ns, a metric": 1,
    "const int64_t latency = now_ns() - job.submit_ns;": 1,
    # CpuExpertEngine (cpu_experts.h): one pair per CPU forward, a job of 0.5 ms or more on the CPU expert thread, off
    # the service thread; ProdBuild keeps it because the split's re-tune (cpu_stats' forward ns) reads it in serving.
    "const int64_t start = now_ns();": 1,
    "compute_ns_.fetch_add(now_ns() - start, std::memory_order_relaxed);": 1,
    # UringReader, std::chrono directly. register_resources(): the buffer registration's duration (register_ms), at
    # open and at a ring reset, never per read.
    "const auto t0 = std::chrono::steady_clock::now();": 1,
    "register_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();": 1,
    # flush_as_nops(): the NOP drain's soft-error retry window, reached only when a read failed with SQEs unconsumed.
    "const auto give_up = std::chrono::steady_clock::now() + kNopRetryWindow;": 1,
    "if (!soft_error(rc) || std::chrono::steady_clock::now() >= give_up) {": 1,
}
CLOCK_READ = re.compile(r"\bnow_ns\(\)|\b(?:steady|system|high_resolution)_clock::now\b|\bclock_gettime\s*\(")


def test_no_ungated_clock_read_remains_on_the_request_path():
    """Spec M3/M4 (plan 2026-09-29-hotpath-zero-overhead Task 9): the watchdog's clock-stamped marker and the drain
    loop's clock-gated progress are gone from the sources; the shim's clock count is the runtime proof."""
    text = joined_text(host_sources())
    assert "busy_since_" not in text and "kProgressIntervalNs" not in text


def test_no_clock_read_bypasses_the_trace_gate():
    lines = [line.strip() for line in joined_text(host_sources()).splitlines()]
    reads = collections.Counter(line for line in lines if CLOCK_READ.search(line) and not line.startswith("//"))
    assert reads == NON_TRACE_CLOCK_READS, {
        line: count for line, count in (reads - collections.Counter(NON_TRACE_CLOCK_READS)).items()
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
