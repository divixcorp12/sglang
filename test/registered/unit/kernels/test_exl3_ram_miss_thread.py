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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
