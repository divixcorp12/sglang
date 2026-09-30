"""A demand that only leases stand in the way of is deferred, not aborted and not served (CPU); LEASE_PROTOCOL.md,
"Reuse".

Written from the requirement text before the service code. The real service is driven through the request page and
the lease block by the Python stand-in for the device (sglang/test/dsv41_lease_sim.py); it shows the service's
behaviour, never the kernels'. Each test names the mutation of the service it must fail under.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import DEMAND_RECORDS, ExpertStreamHost, new_page, page_word
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
def world(tmp_path, request):
    capacity = getattr(request, "param", 2)
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    host.enable_trace()
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _leases(host, row):
    return [info[2] for info in host.slot_info(row)]


def _done(page):
    return page_word(page, "demand_done")


def _serve(host, sim, row, lanes):
    req = sim.post(row, lanes)
    assert host.pump() == 1
    assert sim.wait(req).served
    return req


def _full_of_leases(host, sim):
    """Experts 3 and 4 resident, each leased by a request of its own whose Done is withheld: the tier has no victim
    without a retirement, and each retirement frees exactly one slot (Done retires a whole request)."""
    held3, held4 = _serve(host, sim, 0, [3]), _serve(host, sim, 0, [4])
    assert sorted(_leases(host, 0)) == [1, 1]
    return held3, held4


def test_a_demand_whose_only_victims_are_leased_is_neither_served_nor_failed_and_evicts_nothing(world):
    """Mutation: the demand is served by evicting a leased slot, aborts, or evicts the unleased row before it gives
    up. The request also names a resident lane (3) and needs a victim for 5: the only candidate is 4, leased."""
    s, page, host, sim = world
    held3, held4 = _full_of_leases(host, sim)
    sim.post(0, [3, 5])
    assert host.victim_census(0, wanted=[3, 5]) == (0, 0, 1), "precondition: no victim, one lease away"
    before = (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0))
    for _ in range(5):
        assert host.pump() == 0
    assert _done(page) == held4.seq, "the deferred demand is not answered"
    assert (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0)) == before
    assert host.counters()["deferred"] == 1, "one deferral, not one per poll"
    assert host.busy_episode() == 0, (
        "a deferral is not a request in service: the watchdog's stuck rule would count the wait as a hung read"
    )


def test_a_deferred_demand_is_served_after_a_lease_retires_and_leases_its_hit_lane_too(world):
    """Mutation: the retirement does not wake the deferred demand, or it is served without leasing every lane."""
    s, page, host, sim = world
    held3, held4 = _full_of_leases(host, sim)
    req = sim.post(0, [3, 5])
    assert host.pump() == 0
    sim.done(held4)  # expert 4's lease
    assert host.pump() == 1
    waited = sim.wait(req)
    assert waited.served and waited.go == 2
    assert host.contains(0, 3) and host.contains(0, 5) and not host.contains(0, 4)
    slot3 = next(slot for slot, (state, e, _) in enumerate(host.slot_info(0)) if state == READY and e == 3)
    assert _leases(host, 0)[slot3] == 2, "expert 3 is leased by the first request and by this one"


def test_a_lane_expert_the_record_did_not_protect_still_counts_when_deciding_to_defer(world):
    """Mutation: the deferral decision ignores the lane experts. Expert 3 (a lane, resident, unleased) is the only
    slot the record's own protect set would let it take; with the lanes counted it is protected and the request must
    wait for expert 4's lease instead of aborting on no victim."""
    s, page, host, sim = world
    host.assign(0, 3, protected=[3])
    first = _serve(host, sim, 0, [4])
    assert sorted(_leases(host, 0)) == [0, 1]
    req = sim.post(0, [3, 5], protect=[5])
    assert host.pump() == 0, "deferred, not aborted"
    assert host.counters()["deferred"] == 1
    sim.done(first)
    assert host.pump() == 1 and sim.wait(req).go == 2
    assert host.contains(0, 3) and host.contains(0, 5)


def test_a_deferral_that_a_retirement_does_not_end_is_still_one_deferral(world):
    """Mutation: every retry that is refused again counts as a new deferral. The request needs two victims and the
    first retirement frees only one, so it is retried, refused again, and served after the second."""
    s, page, host, sim = world
    held3, held4 = _full_of_leases(host, sim)
    req = sim.post(0, [5, 1])
    assert host.pump() == 0 and host.counters()["deferred"] == 1
    sim.done(held3)
    assert host.pump() == 0, "one victim is not enough: refused again, without evicting"
    assert host.counters()["deferred"] == 1 and host.counters()["evictions"] == 0
    sim.done(held4)
    assert host.pump() == 1 and sim.wait(req).go == 2


def test_a_request_slot_is_not_reused_until_its_leases_retire(world):
    """Mutation: request slot idx is served while the request that last used it still holds leases (its lease row
    would be overwritten and the old leases never released). Sixteen requests with no Done fill the ring."""
    s, page, host, sim = world
    held = [_serve(host, sim, 0, [3]) for _ in range(DEMAND_RECORDS)]
    assert host.counters()["leases_granted"] == DEMAND_RECORDS
    seventeenth = sim.post(0, [3])
    assert seventeenth.idx == held[0].idx
    assert host.pump() == 0 and _done(page) == held[-1].seq
    assert host.counters()["deferred_reuse"] == 1 and host.counters()["deferred"] == 0
    sim.done(held[0])
    assert host.pump() == 1
    assert sim.wait(seventeenth).go == 1
    assert host.counters()["leases_acked"] == 1 and host.counters()["leases_granted"] == DEMAND_RECORDS + 1


def test_polling_a_deferred_demand_neither_reads_the_clock_nor_pushes_a_stage_record(world):
    """Mutation: the deferral re-enters begin_stage on every poll (a stage record, or at least a clock read, per poll:
    the 8192-slot ring would fill within milliseconds), or it pushes a record when it defers."""
    s, page, host, sim = world
    held3, held4 = _full_of_leases(host, sim)
    host.drain_trace()
    req = sim.post(0, [3, 5])
    t_first = time.monotonic_ns()
    assert host.pump() == 0  # the first observation: begin_stage runs once here
    clock_reads = host.trace_clock_reads()
    for _ in range(300):
        assert host.pump() == 0
    assert host.trace_clock_reads() == clock_reads, "a poll of an unchanged deferral read the clock"
    assert [r for r in host.drain_trace() if r["seq"] == req.seq] == [], "a deferral pushed a stage record"
    time.sleep(0.05)
    t_release = time.monotonic_ns()
    sim.done(held4)
    assert host.pump() == 1
    records = [r for r in host.drain_trace() if r["seq"] == req.seq]
    assert len(records) == 1, "exactly one stage record for the request"
    assert t_first <= records[0]["observed"] < t_release, "observed is the FIRST observation, so the deferral shows in the trace"


def test_a_demand_blocked_only_by_an_injected_lease_defers(world):
    """The census defers whatever holds the lease, not only a request's own: the demand waits, evicts nothing, and
    is served once the lease drops."""
    s, page, host, sim = world
    first = _serve(host, sim, 0, [3, 4])
    sim.done(first)
    assert host.pump() == 0 and sorted(_leases(host, 0)) == [0, 0]
    slot4 = next(slot for slot, (state, e, _) in enumerate(host.slot_info(0)) if e == 4)
    host.inject_lease(0, slot4, +1)
    host.inject_lease(0, 1 - slot4, +1)
    before = (host.counters()["evictions"], host.slot_info(0))
    req = sim.post(0, [1])
    assert host.pump() == 0 and _done(page) == first.seq
    assert (host.counters()["evictions"], host.slot_info(0)) == before and host.counters()["deferred"] == 1
    host.inject_lease(0, slot4, -1)
    assert host.pump() == 1 and sim.wait(req).served

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
