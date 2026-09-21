"""The service thread with leases outstanding: the pause split, and a worker that never waits for an acknowledgement
(CPU); LEASE_PROTOCOL.md step 3c, table F10 and items 9(a), 9(b) and 10 of section 18.2.

Written from the requirement text before the service code. A pause counts GRAPH-LANE leases only (a promotion's host
lease must not block an eager pause, PROMOTION_ASYNC R2); ``inject_lease`` stands in for a host lease, since the
service's own leases are all graph lanes. Each test names the mutation of the service it must fail under.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PROMPT = 1.0  # seconds: "promptly" for a pause or a stop that must not wait for anything


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
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


def _served(sim, row, lanes):
    req = sim.post(row, lanes)
    waited = sim.wait(req, timeout_s=5.0)
    assert waited.status == 1 and waited.go == len(lanes)
    return req, waited


def test_a_pause_is_refused_promptly_while_a_graph_lane_lease_is_outstanding_and_granted_once_it_retires(running):
    """Mutation: the pause ignores graph-lane leases (an eager caller would then own slots a gather may still read)."""
    s, page, host, sim = running
    req, waited = _served(sim, 0, [3])
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="lease"):
        host.pause(2.0)
    assert time.perf_counter() - start < PROMPT, "a refusal is an answer, not a wait"
    req2, waited2 = _served(sim, 1, [4])  # the service went on serving: the refused pause resumed it
    sim.ack(req, waited)
    sim.ack(req2, waited2)
    sim.deliver()
    assert _until(lambda: host.counters()["leases_acked"] == 2)
    host.pause(2.0)
    host.resume()


def test_a_host_lease_does_not_block_an_eager_pause(running):
    """Mutation: the pause counts every lease, so one in-flight promotion refuses every eager host use (R2)."""
    s, page, host, sim = running
    host.assign(0, 3, protected=[3])
    host.inject_lease(0, 0, +1)
    start = time.perf_counter()
    host.pause(2.0)
    assert time.perf_counter() - start < PROMPT
    host.resume()
    assert host.slot_info(0)[0][2] == 1, "the lease is still held, and only protecting the slot is left to it"


def test_a_deferred_demand_does_not_stop_the_worker_from_pausing_stopping_or_taking_advisories(running):
    """Mutation: the worker waits for the acknowledgement inside serve (or the deferral): the pause times out, an
    advisory is never consumed and stop hangs. The lease is a host lease here, so the pause is granted, not refused."""
    s, page, host, sim = running
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    for slot in (0, 1):
        host.inject_lease(0, slot, +1)
    seq = sim.post(0, [1, 2]).seq
    assert _until(lambda: host.counters()["deferred"] == 1), "precondition: a demand is deferred"
    assert page_word(page, "demand_done") != seq
    advisory = sim_post(page, 1, need=[5], protect=[5], advisory=True, after=page_word(page, "demand_head") + 10)
    assert _until(lambda: page_word(page, "advise_done") == advisory), "an advisory is still consumed while a demand waits"
    start = time.perf_counter()
    host.pause(2.0)
    host.resume()
    assert time.perf_counter() - start < PROMPT
    start = time.perf_counter()
    host.stop()
    assert time.perf_counter() - start < PROMPT, "stop does not wait for a lease to retire"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
