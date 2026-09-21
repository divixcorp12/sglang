"""The option C service thread, its pause handshake and its watchdog (CPU, simulated device)."""

import faulthandler
import gc
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # The service thread and its handshakes run in C++: a broken handshake or join hangs,
    # so dump every stack and exit instead (pytest-timeout could not interrupt it).
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _host(tmp_path, capacity=3, fatal_wait_s=5.0):
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
    host.start_thread(fatal_wait_s=fatal_wait_s)
    return s, page, slot_map, host


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_the_thread_serves_demands_without_a_pump(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        assert sim_wait(page, sim_post(page, 0, need=[1, 2], protect=[1, 2]), 10) == 1
        assert host.contains(0, 1) and host.counters()["running"] == 1
        with pytest.raises(RuntimeError, match="pump"):
            host.pump()
    finally:
        host.stop()


def test_a_slow_read_times_out_the_wait_and_raises_fatal(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.inject(delay_s=1.0)
        seq = sim_post(page, 0, need=[1], protect=[1])
        started = time.perf_counter()
        assert sim_wait(page, seq, timeout_s=0.05) == 0
        assert time.perf_counter() - started < 0.5
        assert host.fatal_seq() == seq
        assert sim_wait(page, sim_post(page, 0, need=[], protect=[]), 1.0) == 3  # sticky
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_no_advisory_starts_while_paused(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.pause(timeout_s=1.0)
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 5)
        time.sleep(0.2)
        assert host.counters()["advisories"] == 0 and not host.contains(1, 3)
        # Python owns the slots now: an eager assignment cannot race an advisory.
        host.assign(1, 4, protected=[4])
        host.resume()
        # The advisory posted during the pause is skipped (it predates the eager use).
        assert _until(lambda: host.counters()["advisories_skipped"] == 1)
        assert not host.contains(1, 3) and host.contains(1, 4)
    finally:
        host.stop()


def test_a_pause_waits_for_an_advisory_in_flight_and_cuts_it_short(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.inject(delay_s=0.3)  # the advisory sleeps once, before its first row
        sim_post(page, 1, need=[1, 2, 3], protect=[1, 2, 3], advisory=True, after=page_word(page, "demand_head") + 5)
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        host.pause(timeout_s=2.0)
        waited = time.perf_counter() - started
        assert waited < 1.0
        assert not any(host.contains(1, e) for e in (1, 2, 3))  # the abandoned advisory released its rows
        host.resume()
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_concurrent_eager_use_and_advisories_never_share_a_slot(tmp_path):
    s, page, slot_map, host = _host(tmp_path, capacity=4)
    stop = threading.Event()
    errors = []

    def eager():
        for expert in range(200):
            try:
                host.pause(timeout_s=2.0)
                try:
                    e = expert % 6
                    if not host.contains(0, e):
                        host.assign(0, e, protected=[e])
                    slot = host.mapping(0)[e]
                    time.sleep(0.002)  # an eager fill; no advisory may take the slot meanwhile
                    if host.slot_to_expert(0)[slot] != e:
                        errors.append(("slot taken while paused", e, slot))
                    mapping = host.mapping(0)
                    slots = [m for m in mapping if m >= 0]
                    if len(slots) != len(set(slots)):
                        errors.append(mapping)
                finally:
                    host.resume()
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(repr(error))
        stop.set()

    worker = threading.Thread(target=eager)
    worker.start()
    try:
        expert = 0
        while not stop.is_set():
            sim_post(page, 0, need=[expert % 6], protect=[expert % 6], advisory=True, after=page_word(page, "demand_head") + 5)
            expert += 1
            time.sleep(0.001)
        worker.join(timeout=30)
        assert not errors, errors[:3]
        host.pause(timeout_s=2.0)
        try:
            mapping = host.mapping(0)
            slots = [m for m in mapping if m >= 0]
            assert len(slots) == len(set(slots))
            assert slot_map[0].tolist() == mapping  # the device-visible map equals the READY slots
        finally:
            host.resume()
    finally:
        host.stop()


_SCRIPT_HEAD = """
import pathlib, sys, time
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
import torch
s = ram_miss_setup(pathlib.Path(sys.argv[1]))
page = new_page(pin=False)
host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
"""


def _run_script(tmp_path, body, timeout_s=60):
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT_HEAD + textwrap.dedent(body), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def test_the_watchdog_aborts_a_process_that_does_not_stop_after_fatal(tmp_path):
    # A failed read raises fatal with nothing in service: only the fatal-held rule can fire.
    result = _run_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=0.3)
        host.inject(fail_reads=True)
        assert sim_wait(page, sim_post(page, 0, need=[1], protect=[1]), timeout_s=1.0) == 2
        time.sleep(3.0)
        print("still alive")
        """,
    )
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "still alive" not in result.stdout
    assert "without the process stopping" in result.stderr


def test_the_watchdog_aborts_a_hung_read(tmp_path):
    result = _run_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=0.3)
        host.inject(delay_s=30.0)
        sim_wait(page, sim_post(page, 0, need=[1], protect=[1]), timeout_s=0.05)
        time.sleep(3.0)
        print("still alive")
        """,
    )
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "still alive" not in result.stdout
    assert "stayed in service" in result.stderr


def test_a_stop_during_a_hung_read_still_ends_in_the_watchdog_abort(tmp_path):
    # stop() waits for the service thread; the watchdog must outlive that wait.
    result = _run_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=0.5)
        host.inject(delay_s=30.0)
        sim_post(page, 0, need=[1], protect=[1])
        time.sleep(0.1)
        host.stop()
        print("stopped")
        """,
        timeout_s=25,
    )
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "stopped" not in result.stdout
    assert "stayed in service" in result.stderr


def test_a_stop_during_a_hung_advisory_still_ends_in_the_watchdog_abort(tmp_path):
    # Minor 6: an advisory's give-up check runs between rows, not inside a blocking read,
    # so an advisory hung in io_uring blocks stop()'s join like a hung demand does.
    result = _run_script(
        tmp_path,
        """
        from sglang.kernels.ops.moe.exl3_ram_miss import page_word
        host.start_thread(fatal_wait_s=0.5)
        host.inject(delay_s=8.0)  # advisories sleep before their first read
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 10)
        while host.counters()["advisories"] == 0:
            time.sleep(0.01)
        host.stop()
        print("stopped")
        """,
        timeout_s=25,
    )
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "stopped" not in result.stdout
    assert "stayed in service" in result.stderr


def test_a_process_that_stops_after_fatal_is_not_aborted(tmp_path):
    result = _run_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=1.0)
        host.inject(fail_reads=True)
        assert sim_wait(page, sim_post(page, 0, need=[1], protect=[1]), timeout_s=1.0) == 2
        host.stop()
        time.sleep(2.0)
        print("still alive")
        """,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert "still alive" in result.stdout


def test_the_thread_runs_on_the_core_it_is_pinned_to(tmp_path):
    s = ram_miss_setup(tmp_path)
    host = Exl3RamMissHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False
    )
    try:
        core = min(os.sched_getaffinity(0))
        assert core < 64
        host.start_thread(cpu_core=core)
        assert host.counters()["spin_cpu"] == core
    finally:
        host.stop()


@pytest.mark.parametrize("core, error, match", [(71, ValueError, "64-71"), (1000, RuntimeError, "pin")])
def test_a_reserved_or_unusable_core_is_refused(tmp_path, core, error, match):
    s = ram_miss_setup(tmp_path)
    host = Exl3RamMissHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False
    )
    try:
        with pytest.raises(error, match=match):
            host.start_thread(cpu_core=core)
        assert not host.threaded
    finally:
        host.stop()


def test_a_pause_that_times_out_raises_and_leaves_the_thread_running(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.inject(delay_s=1.0)
        seq = sim_post(page, 0, need=[1], protect=[1])
        assert _until(lambda: page_word(page, "busy_seq") != 0)
        with pytest.raises(RuntimeError, match="did not pause"):
            host.pause(timeout_s=0.1)
        assert sim_wait(page, seq, 3.0) == 1
        host.inject(delay_s=0.0)
        assert sim_wait(page, sim_post(page, 0, need=[2], protect=[2]), 3.0) == 1
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_stop_while_paused_returns_promptly(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    host.pause(timeout_s=1.0)
    started = time.perf_counter()
    host.stop()
    assert time.perf_counter() - started < 1.0 and not host._close.alive


def test_collecting_a_threaded_host_stops_its_thread(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    beat = page_word(page, "heartbeat")
    assert _until(lambda: page_word(page, "heartbeat") != beat)
    del host
    gc.collect()
    beat = page_word(page, "heartbeat")
    time.sleep(0.5)
    assert page_word(page, "heartbeat") == beat


# ---- Task 4: the pipelined reader as the service uses it ----


def _tier(tmp_path, capacity=6):
    """A host with no service thread: the tests pump it, so nothing races."""
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    return s, page, host


def _post_advisory(page, row, ids):
    return sim_post(page, row, need=ids, protect=ids, advisory=True, after=page_word(page, "demand_head") + 10)


def _assert_resident_rows_exact(s, host, row):
    """Every READY row holds exactly its expert's bytes, and nothing is left LOADING."""
    mapping = host.mapping(row)
    resident = [e for e, slot in enumerate(mapping) if slot >= 0]
    reference = s.reference(row, resident)
    for i, expert in enumerate(resident):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[row][name][mapping[expert]], reference[name][i]), (row, expert)
    slots = [slot for slot in mapping if slot >= 0]
    assert len(slots) == len(set(slots))
    owned = [e for e in host.slot_to_expert(row) if e >= 0]
    assert sorted(owned) == sorted(resident)  # a slot with an owner is mapped: none was left LOADING


def test_a_cancelled_advisory_keeps_the_rows_that_completed_and_releases_the_rest(tmp_path):
    s, page, host = _tier(tmp_path)
    try:
        host.enable_trace()
        host.inject(abandon_after_batches=2)  # the advisory stops admitting after its second row
        _post_advisory(page, 1, [1, 2, 3, 4])
        assert host.pump() == 2
        assert [host.contains(1, e) for e in (1, 2, 3, 4)] == [True, True, False, False]
        counters = host.counters()
        assert counters["advisory_rows"] == 2 and counters["rows_read"] == 2
        (record,) = host.drain_trace()
        assert record["status"] == "cancelled" and record["ok"] == 0
        assert record["rows"] == 2 and record["rows_asked"] == 4 and record["batches"] == 2
        packed = [row["row"] for row in record["row_pack"] if row["end"]]
        assert packed == [0, 1]
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules(tmp_path):
    s, page, host = _tier(tmp_path, capacity=3)
    try:
        host.inject(abandon_after_batches=2)
        _post_advisory(page, 1, [1, 2, 3])
        assert host.pump() == 2
        assert host.contains(1, 1) and host.contains(1, 2) and not host.contains(1, 3)
        host.inject(abandon_after_batches=0)
        host.set_hot(1, [1])  # an inclusive-hot row is never a victim, kept or not
        seq = sim_post(page, 1, need=[4, 5], protect=[4, 5])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        # One free slot, then the least recently used row that is neither hot nor protected: expert 2.
        assert [host.contains(1, e) for e in (1, 2, 4, 5)] == [True, False, True, True]
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_an_advisory_has_one_row_outstanding_and_a_demand_may_have_several(tmp_path):
    s, page, host = _tier(tmp_path)
    try:
        host.enable_trace()
        _post_advisory(page, 1, [0, 1, 2, 3])
        assert host.pump() == 2
        seq = sim_post(page, 1, need=[4, 5], protect=[4, 5])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        advisory, demand = host.drain_trace()
        assert (advisory["kind"], advisory["status"], advisory["rows"]) == ("advisory", "served", 4)
        assert advisory["rows_reading_max"] == 1 and advisory["batches"] == 4  # one row at a time
        assert (demand["kind"], demand["status"], demand["rows"]) == ("demand", "served", 2)
        assert demand["rows_reading_max"] == 2 and demand["batches"] == 1  # both rows in one batch
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_a_full_cache_serves_a_demand_that_exactly_fills_it_and_fails_one_that_cannot_fit(tmp_path):
    s, page, host = _tier(tmp_path, capacity=3)
    try:
        host.enable_trace()
        seq = sim_post(page, 1, need=[0, 1, 2], protect=[0, 1, 2])  # fills the cache exactly
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        # Four rows into three slots, all protected: no victim, so no read is even started.
        seq = sim_post(page, 0, need=[0, 1, 2, 3], protect=[0, 1, 2, 3])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 2
        served, failed = host.drain_trace()
        assert (served["status"], served["rows"], served["rows_reading_max"]) == ("served", 3, 3)
        assert failed["status"] == "failed" and failed["batches"] == 0 and failed["submitted_bytes"] == 0
        assert host.counters()["no_victim"] == 1
        assert not any(host.contains(0, e) for e in range(4))  # the failed request published nothing
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


# ---- What protects a resident row from the request that is served for it ----
#
# A record carries `need` (planned experts missing from RAM) and `protect` (the routed experts); the
# service never sees which experts the device planned. serve() reserves slots with take_slot_locked, and
# `wanted` (protect and need) is the only thing that keeps a resident row from being that request's own
# victim. So a planned RAM hit that is absent from protect is a legal victim: a device that read
# slot_map early and copied it would copy a slot the same request then overwrites, and nothing re-reads
# the map afterwards. The kBusySeq-gated hit phase (PER_ROW_TRANSFER, V1b) depends on this; these tests
# pin the mechanism it depends on.


def _fill(page, host, experts):
    for expert in experts:  # each is a demand of its own, so the first one served is the least recently used
        seq = sim_post(page, 1, need=[expert], protect=[expert])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1


def _resident(host, row=1):
    return [e for e in range(6) if host.contains(row, e)]


def test_a_resident_expert_outside_protect_is_evicted_by_the_request_that_needs_another(tmp_path):
    s, page, host = _tier(tmp_path, capacity=3)
    try:
        _fill(page, host, [0, 1, 2])  # expert 0 is the least recently used
        assert _resident(host) == [0, 1, 2] and host.counters()["evictions"] == 0
        seq = sim_post(page, 1, need=[4], protect=[4])  # expert 0 is neither needed nor protected
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        assert _resident(host) == [1, 2, 4]
        assert host.counters()["evictions"] == 1
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_a_resident_expert_in_protect_is_not_the_victim(tmp_path):
    s, page, host = _tier(tmp_path, capacity=3)
    try:
        _fill(page, host, [0, 1, 2])
        seq = sim_post(page, 1, need=[4], protect=[4, 0])  # the same request, now protecting expert 0
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        assert _resident(host) == [0, 2, 4]  # expert 1 went instead
    finally:
        host.stop()


def test_when_every_resident_expert_is_protected_the_request_fails_and_evicts_nothing(tmp_path):
    """The case that separates protection from mere recency: a request that touches its protected
    residents makes them the most recently used, so it is only when NO other victim exists that the
    protection is the only thing standing between a resident row and eviction."""
    s, page, host = _tier(tmp_path, capacity=3)
    try:
        _fill(page, host, [0, 1, 2])
        seq = sim_post(page, 1, need=[4], protect=[0, 1, 2])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 2
        assert _resident(host) == [0, 1, 2]
        assert host.counters()["no_victim"] == 1 and host.counters()["evictions"] == 0
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed(tmp_path):
    s, page, host = _tier(tmp_path)
    try:
        host.enable_trace()
        host.inject(fail_reads=True)
        seq = sim_post(page, 1, need=[0, 1, 2], protect=[0, 1, 2])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 2
        assert not any(host.contains(1, e) for e in (0, 1, 2)) and host.mapping(1) == [-1] * 6
        assert host.slot_to_expert(1) == [-1] * 6
    finally:
        host.inject(fail_reads=False)
        host.stop()


def test_a_stop_during_an_advisory_returns_promptly_and_leaves_the_tier_consistent(tmp_path):
    s, page, slot_map, host = _host(tmp_path, capacity=6)
    try:
        host.inject(delay_s=0.4)  # the advisory sleeps before its first row
        _post_advisory(page, 1, [1, 2, 3])
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        host._module.exl3_ram_miss_stop_thread(host.handle)  # the thread only: the tier stays inspectable
        host.threaded = False
        assert time.perf_counter() - started < 2.0
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_multi_row_advisories_cut_short_by_pauses_leave_only_whole_rows(tmp_path):
    """Advisories of several rows are cut short at arbitrary points by the eager path's pauses; each one
    keeps the rows it completed. At the end every resident row must hold exactly its expert's bytes, the
    device-visible map must equal the READY slots, and no slot may be left LOADING."""
    s, page, slot_map, host = _host(tmp_path, capacity=4)
    try:
        for i in range(120):
            _post_advisory(page, 0, [(i + k) % 6 for k in range(4)])
            if i % 3 == 0:
                host.pause(timeout_s=2.0)
                try:
                    expert = i % 6
                    if not host.contains(0, expert):
                        host.assign(0, expert, protected=[expert])
                finally:
                    host.resume()
            time.sleep(0.001)
        host.pause(timeout_s=2.0)
        try:
            assert slot_map[0].tolist() == host.mapping(0)
            _assert_resident_rows_exact(s, host, 0)
        finally:
            host.resume()
    finally:
        host.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
