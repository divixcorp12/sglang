"""Advisory (next-layer prefetch) records on the RAM-miss thread (CPU, simulated device)."""

import sys
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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
        assert _until(lambda: host.contains(1, 4) and host.contains(1, 5))
        assert host.layer_advisory_rows() == [0, 2] and host.layer_rows() == [0, 0]
        # The thread publishes the rows before it bumps its counters: wait for the bump so
        # the baseline below includes the advisory's rows.
        assert _until(lambda: host.counters()["advisory_rows"] == 2)
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
