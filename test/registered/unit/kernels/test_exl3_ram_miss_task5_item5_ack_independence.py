"""Task 5 item 5 (CPU): a worker with a lease outstanding keeps reading and retiring.

The plan's clause is a negative: the worker must never synchronously wait for a Done word inside its I/O progress
loop. ``test_exl3_ram_miss_lease_thread.py`` pins it for the deferral, a pause and a stop. This file pins it for the
two places a wait would hide from those: the READ itself (the reader's per-batch callback) and the top of
``pump_demand``.

A violation is detected by TIME, not by hang: with a lease deliberately left unretired, a request that needs a real
read must complete within a bound far below the wait the mutants add. The unretired lease is asserted to still be
outstanding when the request completes, so the request cannot have been helped by a retirement.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PROMPT = 1.0  # seconds: far below the 2 s a waiting worker would add per call


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _served(sim, row, lanes):
    req = sim.post(row, lanes)
    waited = sim.wait(req, timeout_s=5.0)
    assert waited.served and waited.go == len(lanes)
    return req, waited


def test_a_demand_that_needs_a_read_is_served_promptly_while_another_requests_done_is_withheld(running):
    """Mutations: the read's per-batch callback, or the top of pump_demand, waits (bounded, 2 s) for outstanding
    leases to retire. The withheld Done never comes, so the wait would run its bound."""
    s, page, host, sim = running
    first, _ = _served(sim, 0, [3])
    assert host.counters()["leases_granted"] == 1 and host.counters()["leases_acked"] == 0
    rows_before = host.counters()["rows_read"]
    start = time.perf_counter()
    second = sim.post(1, [4])  # a miss: it has a real read to do
    waited = sim.wait(second, timeout_s=10.0)
    elapsed = time.perf_counter() - start
    assert waited.served and waited.go == 1
    assert host.counters()["rows_read"] == rows_before + 1, "a row was really read"
    assert host.counters()["leases_acked"] == 0, "and the first request's Done was still withheld"
    assert elapsed < PROMPT, f"the read waited on a Done word ({elapsed:.2f} s)"
    sim.done(first)


@pytest.fixture
def world(tmp_path):
    """The service driven by hand (no thread): every retirement pass is a ``pump``."""
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def test_a_done_word_of_another_generation_with_the_same_low_bits_releases_no_lease(world):
    """Done is what the host releases a request's leases on. A Done word of another lap has the slot's low 32 bits but
    not its generation: acting on it would release a lease the GPU may still be reading through. Mutation: the service
    compares Done's low 32 bits, or accepts any Done at or past the generation."""
    s, page, host, sim = world
    req = sim.post(0, [3])
    assert host.pump() == 1
    assert sim.wait(req).go == 1
    for stale in (req.gen ^ (1 << 32), req.gen - 16):
        assert stale != req.gen
        sim.done(req, generation=stale)
        host.pump()
        counters = host.counters()
        assert counters["leases_acked"] == 0, counters
        assert host.slot_info(0)[[i for i, info in enumerate(host.slot_info(0)) if info[1] == 3][0]][2] == 1, "leased"
    sim.done(req)
    host.pump()
    assert host.counters()["leases_acked"] == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
