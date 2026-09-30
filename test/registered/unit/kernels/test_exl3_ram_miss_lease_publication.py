"""The service publishes every lane of a request in its reservation hold, before it reads the missing rows (CPU).

A resident lane is READY and a missing one LOADING while the read is still running; S waits for the LOADING lanes'
pieces and for demand_done. The device is the Python stand-in (sglang/test/dsv41_lease_sim.py), so every test here
is host-only: what is under test is when and what serve() publishes, not what any kernel does with it.

Each test names the mutation of the service it must fail under.
"""

import faulthandler
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, page_word
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

READY = 2
READ_DELAY_S = 2.0  # long enough that the test thread observes the read in progress without racing it


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
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
    return next(slot for slot, (state, e, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


def _make_resident(host, sim, row, expert):
    """Serve and fully retire a request for ``expert`` so a later request finds it a RAM hit."""
    req = sim.post(row, [expert])
    assert sim.wait(req, timeout_s=5.0).served
    sim.done(req)
    assert _until(lambda: all(count == 0 for count in _leases(host, row)))
    return _slot_of(host, row, expert)


def test_every_lane_is_published_before_read_returns(running):
    """The hit lane's RowResult carries READY and this request's generation, and the miss lane's LOADING, while the
    service is still inside read(); demand_done has not reached the request, and the hit slot is leased.

    Mutant: publish the lanes after the read (grant after reader_.read). It must go red ON THE TAGS and not on a
    timeout: a test that only asserts both are eventually set passes a post-read grant and proves nothing."""
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(delay_s=READ_DELAY_S)

    req = sim.post(0, [3, 4])  # lane 0 is the resident hit, lane 1 (expert 4) has never been read
    seen = {}

    def observe():
        # Inside the injected read delay: both lanes must already be published.
        assert _until(lambda: sim.row_result(req, 0)["gen"] == req.gen, timeout_s=READ_DELAY_S)
        seen["hit"] = sim.row_result(req, 0)
        seen["miss"] = sim.row_result(req, 1)
        seen["done"] = page_word(page, "demand_done")
        seen["hit_leases"] = _leases(host, 0)[hit_slot]

    watcher = threading.Thread(target=observe)
    watcher.start()
    waited = sim.wait(req, timeout_s=READ_DELAY_S + 10.0)
    watcher.join()

    assert seen["hit"] == {"tag": lease.READY, "gen": req.gen, "host_slot": hit_slot}
    assert seen["miss"]["tag"] == lease.LOADING and seen["miss"]["gen"] == req.gen
    assert seen["done"] != req.seq, "demand_done had already reached the request: the read was not still running"
    assert seen["hit_leases"] == 1, "the hit slot was not leased when its RowResult was published"
    assert waited.served and waited.go == 2
    sim.done(req)


def test_the_payload_lands_before_the_ready_word(running):
    """A lane whose ready word names this request has its host_slot already written: W1 and S read host_slot after
    one acquire of ready, with no seqlock re-read.

    Mutant: store the ready word before host_slot, or drop the sfence between them. Repeated, because a single read
    can miss a reorder."""
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(delay_s=READ_DELAY_S)
    req = sim.post(0, [3, 4])
    torn = []

    def observe():
        deadline = time.perf_counter() + READ_DELAY_S
        while time.perf_counter() < deadline:
            result = sim.row_result(req, 0)
            if result["gen"] != req.gen:
                continue
            if (result["tag"], result["host_slot"]) != (lease.READY, hit_slot):
                torn.append(result)
                return

    watcher = threading.Thread(target=observe)
    watcher.start()
    assert sim.wait(req, timeout_s=READ_DELAY_S + 10.0).served
    watcher.join()
    sim.done(req)
    assert not torn, f"a published RowResult carried a payload that was not this request's: {torn}"


def test_a_hit_lanes_slot_is_never_its_own_requests_victim(running):
    """Tier at capacity exactly 3, every slot resident. The request's protect ids leave out the hit lane's own expert
    (0) and name a fourth expert (5), forcing take_slot_locked to pick a victim among the three resident slots. The
    only thing that keeps expert 0's slot from being that victim is serve()'s lane_experts -> wanted loop.

    Mutant: delete the lane_experts -> wanted loop. Expert 0 is then the least recently used row, so it is taken."""
    s, page, host, sim = running
    resident = [_make_resident(host, sim, 0, e) for e in (0, 1, 2)]
    hit_slot = resident[0]

    req = sim.post(0, [0], protect=[5])
    waited = sim.wait(req, timeout_s=10.0)
    assert waited.served

    state, expert, leases = host.slot_info(0)[hit_slot]
    assert (state, expert) == (READY, 0), "the hit lane's own slot was evicted by its own request"
    assert leases == 1, "the hit lane was not leased"
    assert waited.ctx == [(lease.READY, hit_slot)]
    sim.done(req)
    assert _until(lambda: _leases(host, 0)[hit_slot] == 0), "Done did not retire the lease"


def test_the_leases_exist_before_the_device_can_see_demand_done(running):
    """Both lanes miss, and the read is held open. Inside it, each lane's RowResult names its slot, that slot is
    leased, and only then is demand_done read, still short of the request: a device that saw done had leases to find.

    Mutant: count a lane's lease after the read (or after demand_done), not in the reservation hold."""
    s, page, host, sim = running
    host.inject(delay_s=READ_DELAY_S)
    req = sim.post(0, [1, 2])
    assert _until(lambda: all(sim.row_result(req, lane)["gen"] == req.gen for lane in (0, 1)), timeout_s=READ_DELAY_S)
    slots = [sim.row_result(req, lane)["host_slot"] for lane in (0, 1)]
    leases = _leases(host, 0)
    done = page_word(page, "demand_done")
    assert done != req.seq, "demand_done had already reached the request: the read was not still running"
    assert [leases[slot] for slot in slots] == [1, 1], (slots, leases)
    assert host.counters()["leases_granted"] == 2
    assert sim.wait(req, timeout_s=READ_DELAY_S + 10.0).served
    sim.done(req)
    assert _until(lambda: not any(_leases(host, 0)))


def test_a_lane_expert_the_record_did_not_protect_is_still_never_a_victim_of_its_own_request(running):
    """The tier is full (3, 4, 0, with 3 the least recently used). The request's lanes are 3 (a hit) and 5 (a miss),
    and its record protects only 5, so 5's victim must come from 4 and 0: the post protects its routes, and the
    service must not depend on every lane being among them.

    Mutant: the victim choice protects only the record's protect ids -- expert 3, the LRU row, is taken."""
    s, page, host, sim = running
    for expert in (3, 4, 0):
        _make_resident(host, sim, 0, expert)
    req = sim.post(0, [3, 5], protect=[5])
    waited = sim.wait(req, timeout_s=10.0)
    assert waited.served and waited.go == 2
    resident = {e for state, e, _ in host.slot_info(0) if state == READY}
    assert resident == {3, 5, 0}, resident
    sim.done(req)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
