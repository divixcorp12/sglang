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


def test_a_deferred_demand_does_not_stop_the_worker_from_pausing_or_stopping(running):
    """Mutation: the worker waits for the acknowledgement inside serve (or the deferral): the pause times out and stop
    hangs. The lease is a host lease here, so the pause is granted, not refused. That an advisory is still processed
    is asserted by the two R2 tests below, on counters that a stale skip cannot move."""
    s, page, host, sim = running
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    for slot in (0, 1):
        host.inject_lease(0, slot, +1)
    seq = sim.post(0, [1, 2]).seq
    assert _until(lambda: host.counters()["deferred"] == 1), "precondition: a demand is deferred"
    assert page_word(page, "demand_done") != seq
    start = time.perf_counter()
    host.pause(2.0)
    host.resume()
    assert time.perf_counter() - start < PROMPT
    start = time.perf_counter()
    host.stop()
    assert time.perf_counter() - start < PROMPT, "stop does not wait for a lease to retire"


def _deferred_demand(running):
    """Row 0 full of host leases, and a demand for two experts it cannot hold: deferred, and observed to be."""
    s, page, host, sim = running
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    for slot in (0, 1):
        host.inject_lease(0, slot, +1)
    seq = sim.post(0, [1, 2]).seq
    assert _until(lambda: host.counters()["deferred"] == 1), "precondition: a demand is deferred"
    assert page_word(page, "demand_done") != seq
    return seq


def _advisory_processed(host, page, before, row, expert):
    """Post an advisory after the demand (so it is not stale by the `after` rule) and wait until it was SERVED:
    the ``advisories`` counter moved, which a stale skip never does (it moves ``advisories_skipped``)."""
    advisory = sim_post(page, row, need=[expert], protect=[expert], advisory=True, after=page_word(page, "demand_head"))
    assert _until(lambda: page_word(page, "advise_done") == advisory), "the advisory was consumed"
    now = host.counters()
    assert now["advisories"] == before["advisories"] + 1, "it entered serve; it was not dropped"
    assert now["advisories_skipped"] == before["advisories_skipped"], "and it was not skipped as stale"
    return now


def test_an_advisory_for_another_row_arriving_during_a_deferral_gives_up_before_reading(running):
    """Mutations: the advisory is dropped as stale (advisories_skipped moves, advisories does not); it reads while a
    demand waits (advisory_rows moves: demand_pending() in the give-up test is made false); the deferral of the
    demand is counted again by it (deferred moves)."""
    s, page, host, sim = running
    seq = _deferred_demand(running)
    row0 = host.slot_info(0)
    before = host.counters()
    assert page_word(page, "demand_head") == seq and page_word(page, "demand_done") != seq, "a demand is pending"
    now = _advisory_processed(host, page, before, row=1, expert=5)
    assert now["advisory_rows"] == before["advisory_rows"] == 0, "it gave up before reading a row"
    assert now["deferred"] == 1, "the advisory did not count as a second deferral"
    assert now["rows_read"] == before["rows_read"]
    assert host.slot_info(0) == row0, "nothing of the deferred demand's row moved"
    assert page_word(page, "demand_done") != seq, "and the demand is still deferred, not served or failed"


def test_an_advisory_for_the_deferred_demands_own_row_takes_no_leased_slot(running):
    """Mutations: the advisory evicts a leased slot (its map, generation or lease count changes); the advisory counts
    as a deferral of its own; the advisory is dropped as stale."""
    s, page, host, sim = running
    seq = _deferred_demand(running)
    row0, gens = host.slot_info(0), host.mapped_slot_generations(0)
    before = host.counters()
    now = _advisory_processed(host, page, before, row=0, expert=5)
    assert now["advisory_rows"] == 0
    assert now["evictions"] == before["evictions"], "no slot was evicted for it"
    assert now["deferred"] == 1, "an advisory that cannot be served gives up; only a demand is deferred"
    assert host.slot_info(0) == row0 and host.mapped_slot_generations(0) == gens
    assert page_word(page, "demand_done") != seq


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
