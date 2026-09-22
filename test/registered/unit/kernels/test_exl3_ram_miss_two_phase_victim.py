"""Task 6 V1 (two-phase): a hit lane's slot must never become its own request's eviction victim, and a failed
request must void its hit leases without releasing the slot (T3, T4 of
docs/superpowers/plans/task6-v1-checklist.md section 5).

Host-only, same LeaseSim/Exl3RamMissHost harness as test_exl3_ram_miss_two_phase.py (T1/T1b/T4b/T4c). Each test
names the mutation of the service it must fail under; a mutant that leaves a test green is reported rather than
escalated.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

READY = 2


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
    host.enable_two_phase()
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def _leases(host, row):
    return [info[2] for info in host.slot_info(row)]


def _slot_of(host, row, expert):
    return next(slot for slot, (state, e, _, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


def _make_resident(host, sim, row, expert):
    """Serve and fully retire a request for ``expert`` so a later request finds it a RAM hit."""
    req = sim.post(row, [expert])
    waited = sim.wait(req, timeout_s=5.0)
    assert waited.status == 1
    sim.ack(req, waited)
    sim.deliver()
    assert _until(lambda: all(count == 0 for count in _leases(host, row)))
    return _slot_of(host, row, expert)


def test_a_hit_lanes_slot_is_never_its_own_requests_victim(running):
    """T3. Tier at capacity exactly k=3, every slot resident. The request's protect ids deliberately leave out
    the hit lane's own expert (0) -- write_lane_request still names it, since it is the lane being leased -- and
    need names a fourth expert (5), forcing take_slot_locked to pick a victim among the three resident slots. The
    only thing that can keep expert 0's slot from being that victim is the lane_experts -> wanted loop in serve():
    protect carries none of the protection here.

    Two mutants, both required (checklist T3):
      (a) force listed(protect, expert) false in take_slot_locked -- per the checklist this may legitimately stay
          green through recency (the reservation loop bumps the hit slot's stamp via its lookup of `wanted` before
          eviction runs, so LRU alone would spare it), and finding that out is the result, not a defect in the test.
      (b) delete the lane_experts -> wanted loop (the loop appending request.lane_experts into wanted) -- the
          mutant matched to the claim itself, must kill.
    """
    s, page, host, sim = running
    hit_expert = 0
    resident = [_make_resident(host, sim, 0, e) for e in (0, 1, 2)]
    hit_slot = resident[0]
    assert _slot_of(host, 0, hit_expert) == hit_slot

    req = sim.post(0, [hit_expert], protect=[], need=[5])
    waited = sim.wait(req, timeout_s=10.0)
    assert waited.status == 1, "the request that leases the hit lane failed"

    state, expert, leases, _ = host.slot_info(0)[hit_slot]
    assert (state, expert) == (READY, hit_expert), "the hit lane's own slot was evicted by its own request"
    assert leases == 1, "the hit lane was not leased by the early grant"

    sim.ack(req, waited)
    sim.terminal(req, mask=1 << 0)
    sim.deliver()
    assert _until(lambda: _leases(host, 0)[hit_slot] == 0)
    assert host.counters()["lease_double_signal"] == 0


def test_a_failed_request_voids_hit_leases_and_releases_no_slot(running):
    """T4. A mixed request whose hit lane was granted at S2, then fails in read(): right after the request
    answers the hit slot is still READY and leased; the device's terminal names the hit lane; after it lands,
    leases_voided increments and the slot was never handed to another request.

    Mutant: add a release_locked call for the hit slot to the !ok path (the cleanup at serve()'s reservation
    bail). Must go red on `leases`; a second assertion -- issuing a request for a different expert -- catches the
    consequence a build that corrupts the lease count without reusing the slot would still pass.
    """
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(fail_reads=True)  # the request fails after the hit lane has been granted (miss lane: expert 4)

    req = sim.post(0, [3, 4])
    waited = sim.wait(req, timeout_s=10.0)
    assert waited.status != 1, "the injected read failure did not fail the request"

    state, expert, leases, _ = host.slot_info(0)[hit_slot]
    assert (state, expert, leases) == (READY, 3, 1), "the failed request released or corrupted the hit lane's slot"

    host.inject(fail_reads=False)
    other = sim.post(0, [5])
    sim.wait(other, timeout_s=10.0)
    assert _slot_of(host, 0, 5) != hit_slot, "a leased hit slot was handed to the next request"

    sim.terminal(req, mask=1 << 0)
    sim.deliver()
    assert _until(lambda: _leases(host, 0)[hit_slot] == 0)
    assert host.counters()["leases_voided"] >= 1
    assert host.counters()["lease_double_signal"] == 0
