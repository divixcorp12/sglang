"""The RAM-miss service's per-request stage records: one per request, ordered, byte-consistent (CPU)."""

import faulthandler
import os
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import (
    STAGE_ORDER,
    Exl3RamMissHost,
    new_page,
    page_word,
    sim_post,
    sim_wait,
)
from sglang.test.ci.ci_register import register_cpu_ci
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
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    yield s, page, host
    host.stop()


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    assert sim_wait(page, seq, timeout_s=1.0) == 1
    return seq


def _assert_ordered(record):
    """The documented contract (ops/moe/exl3_ram_miss.py): non-zero stamps never decrease along
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
    _serve(page, host, 0, need=[1, 2], protect=[1, 2])
    assert host.drain_trace() == [] and host.trace_dropped() == 0


def test_one_record_per_request_in_order(tier):
    s, page, host = tier
    host.enable_trace()
    seqs = [
        _serve(page, host, 1, need=[2, 5], protect=[2, 5]),
        _serve(page, host, 1, need=[], protect=[2]),  # resident already: served with no read
        _serve(page, host, 1, need=[3], protect=[2, 3, 5]),  # 2 and 5 are resident: one row read
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
    assert read["batches"] == 1 and read["extents"] == 2 and read["submit"] > 0 and read["pack_ns"] > 0
    assert empty["batches"] == 0 and empty["extents"] == 0 and empty["bytes"] == 0 and empty["submit"] == 0
    assert one["extents"] == 1


def test_per_drive_bytes_sum_to_the_request_total(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 1, need=[0, 1, 2, 3, 4, 5], protect=[0, 1, 2, 3, 4, 5])
    (record,) = host.drain_trace()
    assert record["extents"] == 6 and record["bytes"] > 0
    assert sum(d["bytes"] for d in record["drives"]) == record["bytes"]
    assert sum(d["extents"] for d in record["drives"]) == record["extents"]
    assert [d["dev"] for d in record["drives"]] == [os.stat(s.tables.paths[0]).st_dev]


def test_an_unarmed_record_is_a_touch_and_a_failed_read_is_recorded(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 0, need=[1], protect=[1])
    seq = sim_post(page, 0, need=[1], protect=[1], armed=False)
    assert host.pump() == 1
    host.inject(fail_reads=True)
    failed = sim_post(page, 0, need=[4], protect=[4])
    assert host.pump() == 1 and sim_wait(page, failed, 1.0) == 2
    read, touch, error = host.drain_trace()
    assert (touch["kind"], touch["seq"], touch["ok"], touch["extents"]) == ("touch", seq, 1, 0)
    assert (error["kind"], error["ok"], error["rows"]) == ("demand", 0, 0)
    assert error["done"] >= error["reserved"] > 0


def test_a_full_ring_drops_and_counts(tier):
    s, page, host = tier
    host.enable_trace(capacity=2)
    for _ in range(5):
        _serve(page, host, 0, need=[], protect=[1])
    assert len(host.drain_trace()) == 2 and host.trace_dropped() == 3
    _serve(page, host, 0, need=[], protect=[1])  # draining frees the ring
    assert len(host.drain_trace()) == 1


def test_a_backlog_is_seen_when_records_queue_behind_a_slow_one(tier):
    s, page, host = tier
    host.enable_trace()
    seqs = [sim_post(page, 0, need=[], protect=[1]) for _ in range(3)]  # three posted before any is served
    for _ in seqs:
        assert host.pump() == 1
    assert [r["backlog"] for r in host.drain_trace()] == [2, 1, 0]


def test_a_served_advisory_is_recorded_and_a_skipped_one_is_not(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_trace()
    host.start_thread(fatal_wait_s=5.0)
    try:
        sim_post(page, 1, need=[4, 5], protect=[4, 5], advisory=True, after=page_word(page, "demand_head") + 10)
        assert _until(lambda: host.counters()["advisory_rows"] == 2)
        seq = sim_post(page, 0, need=[], protect=[0])
        assert sim_wait(page, seq, 10) == 1
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=seq - 1)  # its demand is already served
        assert _until(lambda: host.counters()["advisories_skipped"] >= 1)
        records = []
        assert _until(lambda: records.extend(host.drain_trace()) or len(records) >= 2)
        assert [r["kind"] for r in records] == ["advisory", "demand"]
        advisory = records[0]
        assert advisory["rows"] == 2 and advisory["batches"] == 2  # an advisory reads one row per batch
        assert advisory["bytes"] == sum(d["bytes"] for d in advisory["drives"]) > 0
        assert [r["kind"] for r in host.drain_trace()] == []
    finally:
        host.stop()


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
    _serve(page, host, 1, need=[2, 5, 4], protect=[2, 5, 4])
    (record,) = host.drain_trace()
    _assert_terminal(record, "served", [])
    assert record["rows"] == record["rows_asked"] == 3 and record["extents"] == 3
    assert len(record["row_pack"]) == 3 and len(record["extent_cqe"]) == 3
    cqe_by_row = {extent["row"]: extent["cqe"] for extent in record["extent_cqe"]}
    for row in record["row_pack"]:
        # Only this row's own completion must precede its packing; other rows' are not compared.
        assert record["submit"] <= cqe_by_row[row["row"]] <= row["start"] <= row["end"] <= record["mapped"], row
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
    _serve(page, host, 1, need=[2], protect=[2])
    _serve(page, host, 1, need=[], protect=[2])  # resident: nothing to read
    _, empty = host.drain_trace()
    _assert_terminal(empty, "no_read", READ_STAGES)
    assert empty["rows"] == empty["rows_asked"] == 0 and empty["batches"] == 0
    assert empty["row_pack"] == [] and empty["extent_cqe"] == []
    assert [empty[k] for k in ("useful_bytes", "submitted_bytes", "bytes", "retried_bytes", "cancelled_bytes")] == [0] * 5


def test_a_touch_is_terminal_and_reads_nothing(tier):
    s, page, host = tier
    host.enable_trace()
    _serve(page, host, 0, need=[1], protect=[1])
    sim_post(page, 0, need=[1], protect=[1], armed=False)
    assert host.pump() == 1
    _, touch = host.drain_trace()
    _assert_terminal(touch, "touch", ["reserved", *READ_STAGES, "mapped"])


def test_an_error_request_marks_the_unreached_stages_and_is_terminal(tier):
    s, page, host = tier
    host.enable_trace()
    host.inject(fail_reads=True)
    failed = sim_post(page, 0, need=[4], protect=[4])
    assert host.pump() == 1 and sim_wait(page, failed, 1.0) == 2
    (record,) = host.drain_trace()
    _assert_terminal(record, "failed", READ_STAGES)
    assert record["ok"] == 0 and record["rows"] == 0 and record["done"] >= record["mapped"] > 0
    assert record["useful_bytes"] == 0 and record["extent_cqe"] == []


def test_an_invalid_request_is_failed_not_a_silent_no_read(tier):
    s, page, host = tier
    host.enable_trace()
    bad = sim_post(page, 0, need=[6], protect=[6])  # expert 6 does not exist (6 per layer)
    assert host.pump() == 1 and sim_wait(page, bad, 1.0) == 2
    (record,) = host.drain_trace()
    _assert_terminal(record, "failed", READ_STAGES)


def test_a_cancelled_advisory_is_terminal_and_names_its_missing_stages(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_trace()
    host.inject(delay_s=0.6, delay_after_demands=10**6)  # only the advisory sleeps, before its first batch
    host.start_thread(fatal_wait_s=5.0)
    try:
        sim_post(page, 1, need=[4, 5], protect=[4, 5], advisory=True, after=page_word(page, "demand_head") + 10)
        time.sleep(0.2)  # the advisory is asleep in its read
        seq = sim_post(page, 0, need=[], protect=[0])  # a demand posted behind it: the advisory gives up
        assert sim_wait(page, seq, 10) == 1
        records = []
        assert _until(lambda: records.extend(host.drain_trace()) or len(records) >= 2)
        advisory, demand = records
        assert (advisory["kind"], demand["kind"]) == ("advisory", "demand")
        _assert_terminal(advisory, "cancelled", READ_STAGES)
        assert advisory["ok"] == 0 and advisory["rows"] == 0 and advisory["rows_asked"] == 2
        assert advisory["row_pack"] == [
            {"row": 0, "admit": 0, "start": 0, "end": 0},
            {"row": 1, "admit": 0, "start": 0, "end": 0},
        ]
        assert advisory["extent_cqe"] == [] and advisory["cancelled_bytes"] == 0  # nothing was in flight
        _assert_terminal(demand, "served", [])  # its protected expert 0 was not resident: one row read
        assert host.counters()["advisory_rows"] == 0  # the abandoned advisory's rows were released
    finally:
        host.stop()


def test_disabled_tracing_serves_the_same_bytes_and_state_as_enabled(tmp_path):
    def run(name, trace):
        root = tmp_path / name
        root.mkdir()
        s = ram_miss_setup(root, capacity=6)
        page = new_page(pin=False)
        slot_map = torch.full((2, 6), -1, dtype=torch.int32)
        host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
        if trace:
            host.enable_trace()
        try:
            for row, need, protect in [(1, [2, 5, 4], [2, 5, 4]), (1, [0], [0, 2]), (0, [1, 3], [1, 3])]:
                _serve(page, host, row, need=need, protect=protect)
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
    seqs = [sim_post(page, 1, need=[e], protect=[e]) for e in (0, 1, 2)]
    assert sim_wait(page, seqs[-1], 10) == 1
    records = []
    assert _until(lambda: records.extend(host.drain_trace()) or len(records) >= 3)
    assert [r["seq"] for r in records] == seqs
    for record in records:
        _assert_ordered(record)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
