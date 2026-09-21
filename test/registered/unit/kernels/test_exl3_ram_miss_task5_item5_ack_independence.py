"""Task 5 item 5 (CPU): a worker with an acknowledgement outstanding keeps reading, cancelling and retiring.

The plan's clause is a negative: the worker must never synchronously wait for an acknowledgement inside its I/O
progress loop. ``test_exl3_ram_miss_lease_thread.py`` pins it for the deferral, a pause and a stop. This file
pins it for the two places a wait would hide from those: the READ itself (the reader's per-batch callback, which
worker-mode packing evaluates on many turns) and the top of ``pump_demand``.

A violation is detected by TIME, not by hang: with a lease deliberately left unacknowledged, a request that needs a
real read must complete within a bound far below the wait the mutants add. The unacknowledged lease is asserted to
still be outstanding when the request completes, so the request cannot have been helped by a retirement.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PROMPT = 1.0  # seconds: far below the 2 s a waiting worker would add per call


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(params=[0, 2], ids=["inline_pack", "two_pack_workers"])
def running(request, tmp_path):
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables,
        page=page,
        slot_map=torch.full((2, 6), -1, dtype=torch.int32),
        direct=False,
        pack_workers=request.param,
    )
    host.enable_lease_mode()
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _served(sim, row, lanes):
    req = sim.post(row, lanes)
    waited = sim.wait(req, timeout_s=5.0)
    assert waited.status == 1 and waited.go == len(lanes)
    return req, waited


def test_a_demand_that_needs_a_read_is_served_promptly_while_another_requests_acknowledgement_is_withheld(running):
    """Mutations: the read's per-batch callback, or the top of pump_demand, waits (bounded, 2 s) for outstanding
    leases to retire. The withheld acknowledgement never comes, so the wait would run its bound."""
    s, page, host, sim = running
    first, first_waited = _served(sim, 0, [3])
    assert host.counters()["leases_granted"] == 1 and host.counters()["leases_acked"] == 0
    rows_before = host.counters()["rows_read"]
    start = time.perf_counter()
    second = sim.post(1, [4])  # a miss: it has a real read to do
    waited = sim.wait(second, timeout_s=10.0)
    elapsed = time.perf_counter() - start
    assert waited.status == 1 and waited.go == 1
    assert host.counters()["rows_read"] == rows_before + 1, "a row was really read"
    assert host.counters()["leases_acked"] == 0, "and the first request's acknowledgement was still withheld"
    assert elapsed < PROMPT, f"the read waited on an acknowledgement ({elapsed:.2f} s)"
    sim.ack(first, first_waited)
    sim.deliver()


def test_a_demand_cancels_an_advisory_and_is_served_promptly_while_an_acknowledgement_is_withheld(running):
    """Cancellation while an acknowledgement is outstanding. An advisory for row 1 is asleep in an injected delay
    when a demand for row 1 is posted; when it wakes, its callback sees the pending demand and gives up before
    reading a row. Mutation: that callback waits for the withheld acknowledgement first."""
    s, page, host, sim = running
    first, first_waited = _served(sim, 0, [3])
    delay = 0.3
    host.inject(delay_s=delay)
    advisory = sim_post(page, 1, need=[1, 2], protect=[1, 2], advisory=True, after=page_word(page, "demand_head"))
    time.sleep(0.05)  # the advisory is asleep in its delay
    start = time.perf_counter()
    demand = sim.post(1, [5])
    waited = sim.wait(demand, timeout_s=10.0)
    elapsed = time.perf_counter() - start
    host.inject(delay_s=0.0)
    assert waited.status == 1 and waited.go == 1
    assert page_word(page, "advise_done") == advisory, "the advisory was consumed"
    counters = host.counters()
    assert counters["advisory_rows"] == 0, "it gave up before reading a row: the demand was pending"
    assert counters["leases_acked"] == 0, "the acknowledgement was withheld throughout"
    assert elapsed < 2 * delay + PROMPT, f"the cancellation or the demand waited on an acknowledgement ({elapsed:.2f} s)"
    sim.ack(first, first_waited)
    sim.deliver()


def test_a_terminal_retires_its_lanes_while_the_worker_serves_and_another_acknowledgement_stays_withheld(running):
    """The cancellation handshake, in thread mode and with no pump. Request B is given up on by the device (a terminal
    naming its lane) while A's acknowledgement is withheld: B's lease is voided by the running worker, A's is
    untouched, and a third request C is served on the way."""
    s, page, host, sim = running
    a, a_waited = _served(sim, 0, [3])
    b, b_waited = _served(sim, 1, [4])
    sim.terminal(b, mask=0b1)
    sim.deliver()
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and host.counters()["leases_voided"] < 1:
        time.sleep(0.002)
    counters = host.counters()
    assert counters["leases_voided"] == 1 and counters["leases_acked"] == 0, counters
    assert host.slot_info(1)[[i for i, info in enumerate(host.slot_info(1)) if info[1] == 4][0]][2] == 0
    assert host.slot_info(0)[[i for i, info in enumerate(host.slot_info(0)) if info[1] == 3][0]][2] == 1
    start = time.perf_counter()
    c = sim.post(1, [5])
    assert sim.wait(c, timeout_s=10.0).status == 1
    assert time.perf_counter() - start < PROMPT
    sim.ack(a, a_waited)
    sim.deliver()
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and host.counters()["leases_acked"] < 1:
        time.sleep(0.002)
    assert host.counters()["leases_acked"] == 1


@pytest.fixture
def world(tmp_path):
    """The service driven by hand (no thread): every retirement pass is a ``pump``."""
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def test_a_terminal_of_another_generation_with_the_same_low_bits_voids_no_lease(world):
    """The terminal is what the device publishes when it gives up, and what the service voids a lane on. A terminal left
    in a request slot by an EARLIER lap has the slot's low 32 bits but not its generation: acting on it would release
    a lease the GPU may still be reading through. Mutation: the service checks a terminal's tag and not its generation."""
    s, page, host, sim = world
    req = sim.post(0, [3])
    assert host.pump() == 1
    waited = sim.wait(req)
    assert waited.go == 1
    stale = req.gen ^ (1 << 32)
    assert stale & 0xFFFFFFFF == req.gen & 0xFFFFFFFF and stale != req.gen
    offset = sim.terminal_offset(req)
    sim._i32(offset, 2)[:] = torch.tensor([0b1, 1], dtype=torch.int32)
    sim.write_u64(offset + lease.TERMINAL_FIELDS["gen"], lease.tagged(lease.TERMINAL_TAG, stale))
    host.pump()
    counters = host.counters()
    assert counters["leases_voided"] == 0 and counters["leases_acked"] == 0, counters
    assert host.slot_info(0)[[i for i, info in enumerate(host.slot_info(0)) if info[1] == 3][0]][2] == 1, "still leased"
    sim.ack(req, waited)
    sim.deliver()
    host.pump()
    assert host.counters()["leases_acked"] == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
