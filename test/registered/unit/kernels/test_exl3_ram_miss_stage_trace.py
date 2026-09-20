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
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

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
    stamps = [record[name] for name in STAGE_ORDER if record[name]]
    assert stamps == sorted(stamps), record


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
