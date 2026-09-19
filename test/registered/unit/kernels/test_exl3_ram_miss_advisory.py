"""Advisory (next-layer prefetch) records on the RAM-miss thread (CPU, simulated device)."""

import faulthandler
import sys
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # The service thread and its handshakes run in C++: a broken handshake or join hangs,
    # so dump every stack and exit instead (pytest-timeout could not interrupt it).
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _host(tmp_path):
    s = ram_miss_setup(tmp_path)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.start_thread(fatal_wait_s=5.0)
    return page, host


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_an_advisory_loads_rows_before_their_demand(tmp_path):
    page, host = _host(tmp_path)
    try:
        sim_post(page, 1, need=[4, 5], protect=[4, 5], advisory=True, after=page_word(page, "demand_head") + 10)
        # contains() is true once the thread has claimed a slot, before the read: the rows are
        # loaded when the per-layer advisory count says so. The thread bumps its global
        # counters after that, so wait for the bump too: the baseline below must include it.
        assert _until(lambda: host.layer_advisory_rows() == [0, 2] and host.counters()["advisory_rows"] == 2)
        assert host.contains(1, 4) and host.contains(1, 5) and host.layer_rows() == [0, 0]
        # The demand then finds them in RAM: a touch-only request, no read.
        before = host.counters()["rows_read"]
        assert sim_wait(page, sim_post(page, 1, need=[], protect=[4, 5]), 10) == 1
        assert host.counters()["rows_read"] == before
    finally:
        host.stop()


def test_an_advisory_whose_layer_already_posted_its_demand_is_skipped(tmp_path):
    page, host = _host(tmp_path)
    try:
        seq = sim_post(page, 0, need=[], protect=[0])
        assert sim_wait(page, seq, 10) == 1
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=seq - 1)  # its demand seq is reached
        assert _until(lambda: host.counters()["advisories_skipped"] >= 1)
        assert not host.contains(1, 3)
    finally:
        host.stop()


def test_a_demand_preempts_an_advisory_in_flight(tmp_path):
    page, host = _host(tmp_path)
    try:
        host.inject(delay_s=0.3)
        sim_post(page, 1, need=[1, 2, 3], protect=[1, 2, 3], advisory=True, after=page_word(page, "demand_head") + 10)
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        assert sim_wait(page, sim_post(page, 0, need=[], protect=[]), 10) == 1
        assert time.perf_counter() - started < 0.8  # the advisory gave up after at most its first read delay
        assert not any(host.contains(1, e) for e in (1, 2, 3))
    finally:
        host.inject(delay_s=0.0)
        host.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
