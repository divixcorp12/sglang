"""The option C service thread, its pause handshake and its watchdog (CPU, simulated device)."""

import faulthandler
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
        host.inject(delay_s=0.3)  # every advisory read sleeps first
        sim_post(page, 1, need=[1, 2, 3], protect=[1, 2, 3], advisory=True, after=page_word(page, "demand_head") + 5)
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        host.pause(timeout_s=2.0)
        waited = time.perf_counter() - started
        assert waited < 1.0  # at most the one row in flight, not three
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


_ABORT_SCRIPT = textwrap.dedent(
    """
    import pathlib, time
    from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
    from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
    import torch
    s = ram_miss_setup(pathlib.Path({tmp!r}))
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.start_thread(fatal_wait_s=0.3)
    host.inject(delay_s=30.0)
    sim_wait(page, sim_post(page, 0, need=[1], protect=[1]), timeout_s=0.05)
    time.sleep(3.0)
    print("still alive")
    """
)


def test_the_watchdog_aborts_a_process_that_does_not_stop_after_fatal(tmp_path):
    script = _ABORT_SCRIPT.format(tmp=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "still alive" not in result.stdout
    assert "exl3 RAM miss" in result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
