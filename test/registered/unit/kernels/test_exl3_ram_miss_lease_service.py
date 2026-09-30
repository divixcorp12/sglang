"""The real service driven through the request page and lease block by a Python stand-in for the device (CPU);
LEASE_PROTOCOL.md, "Generations" and "Reuse": grant, publish and retire.

The stand-in (sglang/test/dsv41_lease_sim.py) is written from the same specification as the service: it shows the
SERVICE's behaviour, never the kernels'. Each test names the mutation of the service it must fail under.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import DEMAND_RECORDS, ExpertStreamHost, new_page, page_word
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

READY = 2


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def world(tmp_path, request):
    capacity = getattr(request, "param", 3)
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _leases(host, row):
    return [info[2] for info in host.slot_info(row)]


def _slot_of(host, row, expert):
    return next(slot for slot, (state, e, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


def _serve(host, sim, row, lanes, **kw):
    req = sim.post(row, lanes, **kw)
    assert host.pump() == 1
    waited = sim.wait(req)
    assert waited.served
    return req, waited


def _release(host, sim, req):
    sim.done(req)
    host.pump()  # an idle pump still runs the retire step


def test_each_lane_gets_a_row_result_naming_its_slot_and_a_lease_on_that_slot(world):
    """Mutation: a lane is published with the wrong slot, tag or generation, or is not leased. Both lanes were read,
    so both are LOADING with every piece published."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 1, [2, 5])
    assert waited.go == 2
    for lane, expert in enumerate((2, 5)):
        slot = _slot_of(host, 1, expert)
        assert sim.row_result(req, lane) == {"tag": lease.LOADING, "gen": req.gen, "host_slot": slot}
        assert _leases(host, 1)[slot] == 1
    delivered = sim.copy(req, waited)
    reference = s.reference(1, [2, 5])
    for lane in range(2):
        assert all(same_bytes(delivered[lane][n], reference[n][lane]) for n in EXL3_STREAMED_NAMES)
    assert host.counters()["leases_granted"] == 2


def test_a_lane_whose_expert_is_already_resident_is_leased_too(world):
    """Mutation: only rows the request read are leased; a RAM hit is left unprotected."""
    s, page, host, sim = world
    req, _ = _serve(host, sim, 0, [3])
    _release(host, sim, req)
    assert _leases(host, 0) == [0, 0, 0]
    reads = host.counters()["rows_read"]
    req, waited = _serve(host, sim, 0, [3])  # a hit: nothing to read
    assert host.counters()["rows_read"] == reads
    assert waited.go == 1 and _leases(host, 0)[_slot_of(host, 0, 3)] == 1
    assert sim.row_result(req, 0) == {"tag": lease.READY, "gen": req.gen, "host_slot": _slot_of(host, 0, 3)}


def test_two_lanes_naming_one_expert_take_two_leases_and_one_done_releases_both(world):
    """Mutation: leases are deduplicated per expert (the Done would then underflow the slot's count), or Done
    releases only one lane's lease."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [4, 4])
    slot = _slot_of(host, 0, 4)
    assert waited.go == 2 and _leases(host, 0)[slot] == 2, "two lanes: two leases, before the Done"
    assert sim.row_result(req, 0)["host_slot"] == sim.row_result(req, 1)["host_slot"] == slot
    _release(host, sim, req)
    assert _leases(host, 0)[slot] == 0


def test_a_done_of_another_generation_with_the_same_low_bits_does_not_retire_a_lease(world):
    """Mutation: the service compares only the low 32 bits of the generation (a request slot idle for 2**32 requests
    would otherwise be retired by a word from its previous epoch)."""
    s, page, host, sim = world
    req, _ = _serve(host, sim, 0, [3])
    slot = _slot_of(host, 0, 3)
    stale_gen = req.gen ^ (1 << 32)
    assert stale_gen & 0xFFFFFFFF == req.gen & 0xFFFFFFFF and stale_gen != req.gen
    sim.done(req, generation=stale_gen)
    host.pump()
    assert _leases(host, 0)[slot] == 1, "the stale word must not retire the lease"
    _release(host, sim, req)
    assert _leases(host, 0)[slot] == 0


def test_an_armed_request_whose_lane_request_was_overwritten_is_an_overrun_and_leases_nothing(world):
    """The seqlock re-check: a later request already wrote over this slot's lane request."""
    s, page, host, sim = world
    req = sim.post(0, [1])
    base = lease.LANE_REQUEST + req.idx * lease.LANE_REQUEST_BYTES + lease.LANE_REQUEST_FIELDS["gen"]
    sim.write_u64(base, req.gen + DEMAND_RECORDS)
    overruns = host.counters()["overruns"]
    assert host.pump() == 1
    assert host.counters()["overruns"] == overruns + 1
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_granted"] == 0


def test_a_request_slot_is_reused_after_its_leases_retire_without_leaking_a_lease(world):
    """Mutation: a retired entry is never freed, so the ring's second lap finds its request slot still occupied."""
    s, page, host, sim = world
    for i in range(3 * DEMAND_RECORDS):
        req, waited = _serve(host, sim, i % 2, [i % 6])
        assert waited.go == 1, f"request {i}"
        _release(host, sim, req)
    assert _leases(host, 0) == [0, 0, 0] and _leases(host, 1) == [0, 0, 0]
    assert host.counters()["leases_granted"] == 3 * DEMAND_RECORDS == host.counters()["leases_acked"]


def test_closing_admission_stops_new_service_and_keeps_retiring(world):
    """Shutdown, step one. Mutation: a request posted after the close is served, or retirement stops with admission
    (a Done of work already in flight would then never land)."""
    s, page, host, sim = world
    first, _ = _serve(host, sim, 0, [3])
    host.close_admission()
    late = sim.post(0, [4])
    assert host.pump() == 0 and page_word(page, "demand_done") == first.seq, "nothing new is served"
    _release(host, sim, first)
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_acked"] == 1, "retirement goes on"
    assert late.seq != first.seq


def _slab_is(s, row, slot, value):
    return all(bool((s.slabs[row][name][slot].view(torch.uint8) == value).all()) for name in EXL3_STREAMED_NAMES)


@pytest.mark.parametrize("world", [2], indirect=True)
def test_a_served_requests_leased_slot_keeps_its_bytes_under_newer_demands(world):
    """The Done is delayed while two newer demands reserve and READ on the same row, and a sentinel written into the
    leased slot after publication survives, with its lease count. The leased slot is the least recently used one, so
    it is exactly the victim an unguarded eviction would choose. Mutations: the eviction ignores the lease (the
    sentinel is overwritten); the eviction reloads the leased slot with the new row (same)."""
    s, page, host, sim = world
    req, _ = _serve(host, sim, 1, [2])  # expert 2 in the older slot, leased, its Done withheld
    leased = _slot_of(host, 1, 2)
    host.assign(1, 5, protected=[5])  # a newer, unleased neighbour: the only legal victim
    other = _slot_of(host, 1, 5)
    assert other != leased
    for name in EXL3_STREAMED_NAMES:
        s.slabs[1][name][leased].view(torch.uint8).fill_(0xAB)  # after publication: only a rewrite can change it
    held = host.slot_info(1)[leased]
    assert held[2] == 1
    before = host.counters()

    for expert in (4, 1):
        newer, _ = _serve(host, sim, 1, [expert])
        _release(host, sim, newer)  # the newer requests complete normally; only the first Done stays late
    now = host.counters()

    # path witness: both newer requests reserved AND read, each on the only slot they were allowed to take
    assert now["evictions"] == before["evictions"] + 2 and now["rows_read"] == before["rows_read"] + 2
    assert now["deferred"] == before["deferred"]
    assert host.mapping(1)[1] == other, "the last expert landed in the unleased slot"
    # property: the leased slot is exactly as it was published
    assert _slab_is(s, 1, leased, 0xAB), "the sentinel in the leased slot was overwritten"
    assert host.slot_info(1)[leased] == held

    # control: the lease is the only thing that protected it
    _release(host, sim, req)
    assert _leases(host, 1)[leased] == 0
    third, _ = _serve(host, sim, 1, [3])
    assert host.mapping(1)[3] == leased and not _slab_is(s, 1, leased, 0xAB), "once retired, the slot is reusable"


def test_a_lease_done_mid_read_retires_before_that_read_returns(tmp_path):
    """retire_leases() runs inside a demand read too (RowReader::read()'s drain loop calls progress), not only at
    the top of pump(). Property: a lease whose Done lands partway through a LATER read is retired before that read
    returns, not merely once it does.

    Mutation: remove read()'s in-loop progress() call, leaving retire_leases() reachable only from the top of
    pump(). The Done (delivered once the second request's read has verifiably started, and while it
    is still packing, slowed by pack_delay_ns) would then not retire until the read returns and pump() loops
    back to serve nothing -- this test's poll, which ends at req2's return, would see demand_done name req2
    while leases_acked is still 0, so it fails there rather than on a wrong value.

    The Done is delivered only once req2 is in service: pump_demand() itself starts with an unconditional
    retire_leases() call, so delivering it any earlier races that call -- it could retire the lease before req2's
    read even starts, which would pass under the mutant too and prove nothing.

    Progress runs once per drain-loop turn, and one turn packs every row whose
    reads have landed (publish_landed), each behind pack_delay_ns. When all three rows landed in one reap, the only
    progress after the ack was the one just before read() returned, microseconds ahead of demand_done: a fixed
    100 ms poll missed it, and no poll could have told it from the mutant's retirement just after. So the last row is
    withheld (hold_ordinal) until the others have packed: the turn that releases it runs progress first, one whole
    pack delay before req2 returns. The poll is bounded by req2's return, not by a wall-clock window; its deadline
    is only a hang guard, derived from the injected delay.
    """
    s = ram_miss_setup(tmp_path, capacity=4)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    host.start_thread(fatal_wait_s=60.0, spin_us=200)
    sim = LeaseSim(host, page, s.slabs)
    try:
        req1 = sim.post(0, [1])
        waited1 = sim.wait(req1, timeout_s=5.0)
        assert waited1.served and waited1.go == 1
        assert host.counters()["leases_granted"] == 1, "req1's lease was not granted; nothing to retire mid-flight"

        # Packing is inline on the owner thread, one row at a time, so three missing rows at 50 ms each span
        # ~150 ms. The last row (ordinal 2) is withheld until rows 0 and 1 have packed, so a later turn, whose
        # progress call comes first, still has a whole pack delay to run after it: the ack is retired there.
        pack_delay_ns, rows = 50_000_000, 3
        host.inject_fault(pack_delay_ns=pack_delay_ns, hold_ordinal=rows - 1)
        req2 = sim.post(0, [2, 3, 4])

        started = time.perf_counter() + 2.0
        while host.busy_episode() == 0:
            assert time.perf_counter() < started, "req2 was never picked up by the service thread"
            time.sleep(0.0005)
        sim.done(req1)

        # Until the lease retires or req2 returns, whichever comes first. The retirement count is read before
        # demand_done, so a retirement seen with req2 still unanswered happened before req2 returned.
        hang_guard = time.perf_counter() + 20 * rows * pack_delay_ns / 1e9 + 2.0
        while True:
            if host.counters()["leases_acked"] == 1:
                returned = page_word(page, "demand_done") == req2.seq
                break
            if page_word(page, "demand_done") == req2.seq:
                returned = True
                break
            assert time.perf_counter() < hang_guard, "req2 neither returned nor retired the lease"
            time.sleep(0.002)
        assert not returned, "the lease was not retired before req2's read returned"

        waited2 = sim.wait(req2, timeout_s=10.0)
        assert waited2.served and waited2.go == 3
        sim.done(req2)
    finally:
        host.stop()



if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
