"""The RAM-miss service's per-request stage records: one per request, ordered, byte-consistent (CPU)."""

import faulthandler
import os
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import (
    STAGE_ORDER,
    ExpertStreamHost,
    new_page,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim, post_record
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    yield s, page, host
    host.stop()


def _post(page, host, row, lanes, protect=None):
    return LeaseSim(host, page, None).post(row, lanes, protect=protect)


def _serve(page, host, row, lanes, protect=None):
    """One demand for ``lanes``, served, then Done (the next pump retires its leases); returns its seq."""
    sim = LeaseSim(host, page, None)
    req = sim.post(row, lanes, protect=protect)
    assert host.pump() == 1
    assert sim.wait(req, timeout_s=1.0).served
    sim.done(req)
    return req.seq


def _assert_ordered(record):
    """The documented contract (ops/moe/expert_stream_transport.py): non-zero stamps never decrease along
    STAGE_ORDER EXCEPT last_cqe against pack_start, because a row packs as soon as ITS OWN extents
    land. What holds instead is first_cqe <= pack_start and last_cqe <= pack_end.

    Asserting the whole chain sorted, as this did, passes only while every extent of a request
    retires in one reap - which is what cached buffered reads do, since they complete inside
    io_uring_submit. It would therefore stay green in CI and break on real drive latency.
    """
    for earlier, later in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        if earlier == "last_cqe" and later == "pack_start":
            continue  # the overlap itself
        if record[earlier] and record[later]:
            assert record[earlier] <= record[later], (earlier, later, record)
    if record["first_cqe"] and record["pack_start"]:
        assert record["first_cqe"] <= record["pack_start"], record
    if record["last_cqe"] and record["pack_end"]:
        assert record["last_cqe"] <= record["pack_end"], record


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_no_records_without_enable(tier):
    s, page, host = tier
    _serve(page, host, 0, [1, 2])
    assert host.drain_trace() == [] and host.trace_dropped() == 0


def test_one_record_per_request_in_order(tier):
    s, page, host = tier
    host.enable_trace()
    seqs = [
        _serve(page, host, 1, [2, 5]),
        _serve(page, host, 1, [2]),  # resident already: served with no read
        _serve(page, host, 1, [3], protect=[2, 3, 5]),  # 2 and 5 are resident: one row read
    ]
    records = host.drain_trace()
    assert [r["seq"] for r in records] == seqs
    assert host.drain_trace() == []  # drained once
    assert [r["kind"] for r in records] == ["demand"] * 3
    assert [r["rows"] for r in records] == [2, 0, 1]
    assert all(r["ok"] == 1 and r["row"] in (0, 1) for r in records)
    previous_done = 0
    for record in records:
        _assert_ordered(record)
        assert record["observed"] > 0 and record["reserved"] >= record["observed"]
        assert record["mapped"] >= record["reserved"] and record["done"] >= record["mapped"]
        assert record["prev_done"] == previous_done and record["observed"] >= previous_done
        previous_done = record["done"]
    read, empty, one = records
    # The service streams pieces: every row is the same number of sub-reads (one part, up to 4 page-aligned cuts).
    assert one["extents"] >= 1
    assert read["batches"] == 1 and read["extents"] == 2 * one["extents"] and read["submit"] > 0
    assert empty["batches"] == 0 and empty["extents"] == 0 and empty["bytes"] == 0 and empty["submit"] == 0


def test_per_drive_bytes_sum_to_the_request_total(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 1, [0, 1, 2, 3, 4, 5])
    (record,) = host.drain_trace()
    per_row = {row: sum(e["row"] == row for e in record["extent_cqe"]) for row in range(6)}
    assert len(set(per_row.values())) == 1 and record["extents"] == 6 * per_row[0] and record["bytes"] > 0
    assert sum(d["bytes"] for d in record["drives"]) == record["bytes"]
    assert sum(d["extents"] for d in record["drives"]) == record["extents"]
    assert [d["dev"] for d in record["drives"]] == [os.stat(s.tables.paths[0]).st_dev]


def test_an_unarmed_record_is_a_touch(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 0, [1])
    seq = post_record(page, 0, [1], armed=False)
    assert host.pump() == 1
    read, touch = host.drain_trace()
    assert (touch["kind"], touch["seq"], touch["ok"], touch["extents"]) == ("touch", seq, 1, 0)


def test_a_full_ring_drops_and_counts(tier):
    s, page, host = tier
    host.enable_trace(capacity=2)
    for _ in range(5):
        _serve(page, host, 0, [1])
    assert len(host.drain_trace()) == 2 and host.trace_dropped() == 3
    _serve(page, host, 0, [1])  # draining frees the ring
    assert len(host.drain_trace()) == 1


def test_a_backlog_is_seen_when_records_queue_behind_a_slow_one(tier):
    s, page, host = tier
    host.enable_trace()
    seqs = [_post(page, host, 0, [1]) for _ in range(3)]  # three posted before any is served
    for _ in seqs:
        assert host.pump() == 1
    assert [r["backlog"] for r in host.drain_trace()] == [2, 1, 0]


READ_STAGES = ["submit", "first_cqe", "last_cqe", "pack_start", "pack_end"]


def _assert_terminal(record, status, missing):
    """Every record carries a terminal status and names the stages it never reached."""
    assert record["status"] == status, record
    assert record["missing_stages"] == missing, record
    assert all(record[name] == 0 for name in missing)
    assert all(record[name] > 0 for name in STAGE_ORDER if name not in missing)


def test_a_served_request_has_per_row_stamps_in_causal_order(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 1, [2, 5, 4])
    (record,) = host.drain_trace()
    _assert_terminal(record, "served", [])
    assert record["rows"] == record["rows_asked"] == 3 and record["extents"] == len(record["extent_cqe"]) >= 3
    assert len(record["row_pack"]) == 3
    for row in record["row_pack"]:
        # Only this row's own completions are compared with its packing: with pieces streamed, the row starts packing
        # once its first sub-read landed and ends after its last.
        cqes = [extent["cqe"] for extent in record["extent_cqe"] if extent["row"] == row["row"]]
        assert record["submit"] <= min(cqes) <= row["start"] <= row["end"] <= record["mapped"], row
        assert max(cqes) <= row["end"], row
    # pack_one picks the lowest-ordinal row among those currently READY, so packing follows
    # completion order, not request order: the aggregate brackets the rows, it does not track
    # ordinal 0 and ordinal -1.
    assert record["pack_start"] == min(r["start"] for r in record["row_pack"])
    assert record["pack_end"] == max(r["end"] for r in record["row_pack"])
    assert 0 < record["useful_bytes"] <= record["bytes"] <= record["submitted_bytes"]
    assert record["retried_bytes"] == 0 and record["cancelled_bytes"] == 0


def test_a_zero_miss_request_marks_the_read_stages_missing(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 1, [2])
    _serve(page, host, 1, [2])  # resident: nothing to read
    _, empty = host.drain_trace()
    _assert_terminal(empty, "no_read", READ_STAGES)
    assert empty["rows"] == empty["rows_asked"] == 0 and empty["batches"] == 0
    assert empty["row_pack"] == [] and empty["extent_cqe"] == []
    assert [empty[k] for k in ("useful_bytes", "submitted_bytes", "bytes", "retried_bytes", "cancelled_bytes")] == [0] * 5


def test_a_touch_is_terminal_and_reads_nothing(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 0, [1])
    post_record(page, 0, [1], armed=False)
    assert host.pump() == 1
    _, touch = host.drain_trace()
    _assert_terminal(touch, "touch", ["reserved", *READ_STAGES, "mapped"])


def test_disabled_tracing_serves_the_same_bytes_and_state_as_enabled(tmp_path):
    def run(name, trace):
        root = tmp_path / name
        root.mkdir()
        s = ram_miss_setup(root, capacity=6)
        page = new_page(pin=False)
        slot_map = torch.full((2, 6), -1, dtype=torch.int32)
        host = ExpertStreamHost(s.tables, page=page, slot_map=slot_map)
        if trace:
            host.enable_trace()
        try:
            for row, need, protect in [(1, [2, 5, 4], [2, 5, 4]), (1, [0], [0, 2]), (0, [1, 3], [1, 3])]:
                _serve(page, host, row, need, protect=protect)
            return s, slot_map, host.counters(), len(host.drain_trace())
        finally:
            host.stop()

    off, on = run("off", False), run("on", True)
    assert off[3] == 0 and on[3] == 3  # the switch is what differs
    assert torch.equal(off[1], on[1]) and off[2] == on[2]
    slot_map = off[1]
    assert int((slot_map >= 0).sum()) == 6  # experts 0, 2, 4, 5 of layer 1 and 1, 3 of layer 0
    for layer in (0, 1):
        for slot in slot_map[layer][slot_map[layer] >= 0].tolist():  # the other slots were never written
            for name in off[0].slabs[layer]:
                assert same_bytes(off[0].slabs[layer][name][slot], on[0].slabs[layer][name][slot]), (layer, slot, name)


def test_the_service_thread_records_every_demand(tier):
    s, page, host = tier
    host.enable_trace()
    host.start_thread(fatal_wait_s=5.0)
    sim = LeaseSim(host, page, None)
    reqs = [sim.post(1, [e]) for e in (0, 1, 2)]
    assert sim.wait(reqs[-1], 10).served
    seqs = [req.seq for req in reqs]
    records = []
    assert _until(lambda: records.extend(host.drain_trace()) or len(records) >= 3)
    assert [r["seq"] for r in records] == seqs
    for record in records:
        _assert_ordered(record)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
