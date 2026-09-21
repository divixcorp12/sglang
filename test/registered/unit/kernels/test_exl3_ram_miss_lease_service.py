"""The real service in lease mode, driven through the request page and lease block by a Python stand-in for the
device (CPU); LEASE_PROTOCOL.md step 3a: grant, publish and retire.

The stand-in (sglang/test/dsv41_lease_sim.py) is written from the same specification as the service: it shows the
SERVICE's behaviour, never the kernels'. These tests were written from the requirement text of section 18.2 before
the service code they exercise, and each names the mutation of the service it must fail under; the mutations were
applied and observed failing (see the commit message for the ledger).
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import DEMAND_RECORDS, Exl3RamMissHost, new_page, page_word
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

READY = 2
DEMAND_TAG = 1


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
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _leases(host, row):
    return [info[2] for info in host.slot_info(row)]


def _slot_of(host, row, expert):
    return next(slot for slot, (state, e, _, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


def _serve(host, sim, row, lanes, **kw):
    req = sim.post(row, lanes, **kw)
    assert host.pump() == 1
    return req, sim.wait(req)


def _release(host, sim, req, waited):
    sim.ack(req, waited)
    sim.deliver()
    host.pump()  # an idle pump still runs the retire step


def test_each_lane_gets_a_row_result_naming_its_expert_and_slot_and_a_lease_on_that_slot(world):
    """Mutation: a lane is published with the wrong slot, expert or generation, or is not leased."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 1, [2, 5])
    assert waited.status == 1 and waited.go == 2
    generations = host.mapped_slot_generations(1)
    for lane, expert in enumerate((2, 5)):
        result = sim.row_result(req, lane)
        slot = _slot_of(host, 1, expert)
        assert (result["tag"], result["gen"], result["expert"], result["host_slot"]) == (lease.READY, req.gen, expert, slot)
        assert result["slot_generation"] == generations[slot] == 1
        assert _leases(host, 1)[slot] == 1
    delivered = sim.copy(req, waited)
    reference = s.reference(1, [2, 5])
    for lane in range(2):
        assert all(same_bytes(delivered[lane][n], reference[n][lane]) for n in EXL3_STREAMED_NAMES)
    assert host.counters()["leases_granted"] == 2


def test_a_lane_whose_expert_is_already_resident_is_leased_too(world):
    """Mutation: only rows the request read are leased; a RAM hit is left unprotected."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [3])
    _release(host, sim, req, waited)
    assert _leases(host, 0) == [0, 0, 0]
    reads = host.counters()["rows_read"]
    req, waited = _serve(host, sim, 0, [3])  # a hit: nothing to read
    assert host.counters()["rows_read"] == reads
    assert waited.go == 1 and _leases(host, 0)[_slot_of(host, 0, 3)] == 1
    assert sim.row_result(req, 0)["host_slot"] == _slot_of(host, 0, 3)


def test_two_lanes_naming_one_expert_take_two_leases_and_one_acknowledgement_releases_one(world):
    """Mutation: leases are deduplicated per expert, or one acknowledgement releases every lease on the slot."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [4, 4])
    slot = _slot_of(host, 0, 4)
    assert waited.go == 2 and _leases(host, 0)[slot] == 2, "two lanes: two leases, before any acknowledgement"
    assert sim.row_result(req, 0)["host_slot"] == sim.row_result(req, 1)["host_slot"] == slot
    sim.ack(req, waited, lanes=[0])
    sim.deliver()
    host.pump()
    assert _leases(host, 0)[slot] == 1
    sim.ack(req, waited, lanes=[1])
    sim.deliver()
    host.pump()
    assert _leases(host, 0)[slot] == 0


@pytest.mark.parametrize("world", [2], indirect=True)
def test_a_lane_expert_the_record_did_not_protect_is_still_never_a_victim_of_its_own_request(world):
    """Mutation: the victim choice protects only the record's protect and need sets. The post kernel protects its
    routes, and whether the planned lanes are always among them is open (LEASE_PROTOCOL.md OPEN 12), so the service
    must not depend on it: here the older resident row is a lane expert the record left out."""
    s, page, host, sim = world
    host.assign(0, 3, protected=[3])  # the LRU row, and a lane of the request below
    host.assign(0, 4, protected=[4])
    req = sim.post(0, [3, 5], need=[5], protect=[5])
    assert host.pump() == 1
    waited = sim.wait(req)
    assert waited.status == 1 and waited.go == 2, "the request must succeed by evicting 4, not its own lane's expert 3"
    assert host.contains(0, 3) and host.contains(0, 5) and not host.contains(0, 4)


def test_a_request_the_service_fails_leases_and_publishes_nothing(world):
    """Mutation: leases are granted at reservation and not voided when the read then fails."""
    s, page, host, sim = world
    host.inject(fail_reads=True)
    req = sim.post(0, [1, 2])
    assert host.pump() == 1
    waited = sim.wait(req, publish_terminal=False)
    assert waited.status == 2 and waited.go == 0
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_granted"] == 0
    assert sim.row_result(req, 0)["tag"] == 0, "no row result is published for a failed request"


@pytest.mark.parametrize("world", [2], indirect=True)
def test_a_request_that_fails_on_another_row_publishes_and_leases_nothing_even_for_its_resident_lane(world):
    """Mutation: leases are granted whether or not the request succeeded. Here the lane's own expert is resident (a
    hit), so only the request's failure, not the lane, stands in the way: the record also needs expert 5 and every
    other row is protected, so there is no victim."""
    s, page, host, sim = world
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    req = sim.post(0, [3], need=[5], protect=[3, 4, 5])
    assert host.pump() == 1
    waited = sim.wait(req, publish_terminal=False)
    assert waited.status == 2 and waited.go == 0
    assert _leases(host, 0) == [0, 0] and host.counters()["leases_granted"] == 0
    assert sim.row_result(req, 0)["tag"] == 0


def test_with_lease_mode_off_the_lane_request_is_ignored_and_nothing_is_leased(tmp_path):
    """Off is today's behaviour: a service that was not told to lease never reads the lane request."""
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    try:
        sim = LeaseSim(host, page, s.slabs)
        req = sim.post(0, [1])
        assert host.pump() == 1
        assert sim.wait(req, publish_terminal=False).status == 1
        assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_granted"] == 0
    finally:
        host.stop()


def test_an_acknowledgement_of_another_generation_with_the_same_low_bits_does_not_retire_a_lease(world):
    """Mutation: the service compares only the low 32 bits of the generation (a lane idle for 2**32 requests would
    otherwise be retired by a word from its previous epoch)."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [3])
    slot = _slot_of(host, 0, 3)
    stale_gen = req.gen ^ (1 << 32)
    assert stale_gen & 0xFFFFFFFF == req.gen & 0xFFFFFFFF and stale_gen != req.gen
    sim.write_u64(sim.ack_offset(req, 0), lease.tagged(lease.CONSUMED, stale_gen))
    host.pump()
    assert _leases(host, 0)[slot] == 1, "the stale word must not retire the lease"
    sim.ack(req, waited)
    sim.deliver()
    host.pump()
    assert _leases(host, 0)[slot] == 0


def test_a_terminal_mask_retires_exactly_the_lanes_it_names(world):
    """Mutation: a terminal retires every lane of the request, or ignores its mask."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [1, 2])
    slot0, slot1 = _slot_of(host, 0, 1), _slot_of(host, 0, 2)
    sim.terminal(req, mask=0b01)
    sim.deliver()
    host.pump()
    assert _leases(host, 0)[slot0] == 0 and _leases(host, 0)[slot1] == 1
    assert host.counters()["leases_voided"] == 1
    sim.ack(req, waited, lanes=[1])
    sim.deliver()
    host.pump()
    assert _leases(host, 0)[slot1] == 0


def test_a_lane_signalled_by_both_an_acknowledgement_and_a_terminal_is_counted_and_released_once(world):
    """Mutation: both signals decrement (the lease goes below zero), or the second is silently dropped."""
    s, page, host, sim = world
    req, waited = _serve(host, sim, 0, [1, 2])
    slot0, slot1 = _slot_of(host, 0, 1), _slot_of(host, 0, 2)
    sim.ack(req, waited, lanes=[0])
    sim.terminal(req, mask=0b01)
    sim.deliver()  # both are visible before the next retirement pass
    host.pump()
    assert _leases(host, 0)[slot0] == 0 and _leases(host, 0)[slot1] == 1
    assert host.counters()["lease_double_signal"] == 1
    assert host.counters()["leases_acked"] + host.counters()["leases_voided"] == 1


def test_a_request_the_device_already_gave_up_on_is_dropped_without_a_lease(world):
    """Mutation: the service serves and leases a request whose terminal it has already seen."""
    s, page, host, sim = world
    req = sim.post(0, [1])
    sim.terminal(req, mask=0b1)
    sim.deliver()
    assert host.pump() == 1
    assert host.counters()["late_after_terminal"] == 1
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_granted"] == 0
    assert page_word(page, "demand_done") == req.seq, "the sequence still advances, or every later request is stuck"


def test_an_armed_request_whose_lane_request_was_overwritten_is_an_overrun_and_leases_nothing(world):
    """The seqlock re-check: a later request already wrote over this slot's lane request."""
    s, page, host, sim = world
    req = sim.post(0, [1])
    sim.write_u64(sim._d(lease.LANE_REQUEST + req.idx * lease.LANE_REQUEST_BYTES), lease.tagged(1, req.gen + DEMAND_RECORDS))
    overruns = host.counters()["overruns"]
    assert host.pump() == 1
    assert host.counters()["overruns"] == overruns + 1
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_granted"] == 0


def test_a_request_slot_is_reused_after_its_leases_retire_without_leaking_a_lease(world):
    """Mutation: a retired entry is never freed, so the ring's second lap finds its request slot still occupied."""
    s, page, host, sim = world
    for i in range(3 * DEMAND_RECORDS):
        req, waited = _serve(host, sim, i % 2, [i % 6])
        assert waited.status == 1 and waited.go == 1, f"request {i}"
        _release(host, sim, req, waited)
    assert _leases(host, 0) == [0, 0, 0] and _leases(host, 1) == [0, 0, 0]
    assert host.counters()["leases_granted"] == 3 * DEMAND_RECORDS == host.counters()["leases_acked"]


def test_the_leases_exist_before_the_device_can_see_demand_done(tmp_path):
    """Mutation: the leases and row results are published after demand_done. The service is stalled between serving
    and storing demand_done; a device that saw done at that instant would otherwise find no lease."""
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
    try:
        sim = LeaseSim(host, page, s.slabs)
        host.inject_done_stall(0.6)
        host.start_thread(fatal_wait_s=30.0)
        req = sim.post(0, [1, 2])
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and host.counters()["leases_granted"] < 2:
            time.sleep(0.002)
        assert host.counters()["leases_granted"] == 2, "the stall window was never reached"
        assert page_word(page, "demand_done") != req.seq, "the window is the one before demand_done"
        assert all(sim.row_result(req, lane)["tag"] == lease.READY for lane in range(2))
        assert sim.wait(req, timeout_s=10.0).go == 2
    finally:
        host.stop()


def test_closing_admission_sets_the_header_word_stops_new_service_and_keeps_retiring(world):
    """Shutdown, step one. Mutation: the header word is not set, a request posted after the close is served, or
    retirement stops with admission (an acknowledgement of work already in flight would then never land)."""
    s, page, host, sim = world
    first, first_wait = _serve(host, sim, 0, [3])
    assert host.lease_header()["shutdown"] == 0
    host.close_admission()
    assert host.lease_header()["shutdown"] == 1
    late = sim.post(0, [4])
    assert host.pump() == 0 and page_word(page, "demand_done") == first.seq, "nothing new is served"
    sim.ack(first, first_wait)
    sim.deliver()
    host.pump()
    assert _leases(host, 0) == [0, 0, 0] and host.counters()["leases_acked"] == 1, "retirement goes on"
    assert late.seq != first.seq


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
