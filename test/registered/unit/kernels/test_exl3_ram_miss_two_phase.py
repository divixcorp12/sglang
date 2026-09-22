"""Task 6 V1 (two-phase): the service publishes a resident lane's row result BEFORE it reads the missing rows (CPU).

The device is the Python stand-in (sglang/test/dsv41_lease_sim.py), so every test here is host-only: what is under
test is when serve() grants, not what any kernel does with the grant.

Each test names the mutation of the service it must fail under. A test whose mutant leaves it green is vacuous and
is worth nothing here, so the mutant is part of the test rather than part of a commit message.
"""

import faulthandler
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

READY = 2
READ_DELAY_S = 2.0  # long enough that the test thread observes the gap between the two grants without racing it


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


def test_a_resident_lane_is_published_before_read_returns(running):
    """T1. The hit lane's row result carries READY and this request's generation while the service is still
    inside read(), the miss lane's ready word is still zero, and demand_done has not reached the request.

    Mutant: move the S2 grant back to the post-read grant site (Task 5's behaviour). It must go red ON THE READY
    WORD and not on a timeout -- a test that only asserts both words are eventually set passes unmodified Task 5
    code and proves nothing about when the grant happened.
    """
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(delay_s=READ_DELAY_S)

    req = sim.post(0, [3, 4])  # lane 0 is the resident hit, lane 4 has never been read
    seen = {}

    def observe():
        # Inside the injected read delay: the hit lane must already be published, the miss lane must not be.
        assert _until(lambda: sim.row_result(req, 0)["tag"] == lease.READY, timeout_s=READ_DELAY_S)
        seen["hit"] = sim.row_result(req, 0)
        seen["miss"] = sim.row_result(req, 1)
        seen["done"] = page_word(page, "demand_done")
        seen["hit_leases"] = _leases(host, 0)[hit_slot]

    watcher = threading.Thread(target=observe)
    watcher.start()
    waited = sim.wait(req, timeout_s=READ_DELAY_S + 10.0)
    watcher.join()

    assert seen["hit"]["tag"] == lease.READY, "the resident lane was not published before read() returned"
    assert seen["hit"]["gen"] == req.gen and seen["hit"]["expert"] == 3 and seen["hit"]["host_slot"] == hit_slot
    assert seen["miss"]["tag"] == 0, "the missing lane was published before its row had been read"
    assert seen["done"] != req.seq, "demand_done had already reached the request: the read was not still running"
    # T2's observation, taken at the same moment. It has no falsifying mutant (see the checklist's O3): with one
    # service thread, taking the lease in a SECOND mutex_ hold right after the reservation leaves this green.
    assert seen["hit_leases"] == 1, "the hit slot was not leased when its row result was published"

    assert waited.status == 1 and waited.go == 2
    counters = host.counters()
    assert counters["hit_leases_granted"] == 1, "the hit lane was not granted by the early phase"
    assert counters["leases_granted"] == 3, "both phases together grant every lane exactly once"


def test_the_payload_lands_before_the_ready_word(running):
    """T1b. The seqlock re-read holds: a lane whose ready word is set has its payload already written.

    Mutant: drop S1's per-group fence, or store the ready word before the payload. If this stays green the
    ordering claim is untested and the fence is decoration.

    With one writer this is checked the way the device checks it: read the ready word, the payload, then the ready
    word again, and require the payload to name this request. Repeated, because a single read can miss a reorder.
    """
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(delay_s=READ_DELAY_S)
    req = sim.post(0, [3, 4])
    torn = []

    def observe():
        deadline = time.perf_counter() + READ_DELAY_S
        while time.perf_counter() < deadline:
            result = sim.row_result(req, 0)
            if result["tag"] != lease.READY:
                continue
            if (result["gen"], result["expert"], result["host_slot"]) != (req.gen, 3, hit_slot):
                torn.append(result)
                return

    watcher = threading.Thread(target=observe)
    watcher.start()
    sim.wait(req, timeout_s=READ_DELAY_S + 10.0)
    watcher.join()
    assert not torn, f"a ready row result carried a payload that was not this request's: {torn}"


def test_a_hit_lanes_slot_is_never_in_the_requests_release_set(running):
    """T4b. The whole proof that V1 escapes the release_locked landmine: a hit lane's slot is never among the
    slots serve() releases, because those come only from the take loop over `missing`, and a hit lane's expert is
    by definition not in `missing`. release_locked checks no lease and take_slot_locked's free-slot loop consults
    none, so a hit slot reaching either is silent corruption -- wrong bytes, no diagnostic.

    Mutant: push the hit lane's slot into `slots` in the reservation loop (the plausible refactor, since `slots`
    reads like "the slots this request touched"). The membership assertion must go red AND the slot-reuse
    assertion below must go red; if only the first fires, the consequence is untested.
    """
    s, page, host, sim = running
    hit_slot = _make_resident(host, sim, 0, 3)
    host.inject(fail_reads=True)  # the request fails after the hit lane has been granted

    req = sim.post(0, [3, 4])
    waited = sim.wait(req, timeout_s=10.0)
    assert waited.status != 1, "the injected read failure did not fail the request"

    # The hit slot kept its row and its lease: the failure released nothing.
    state, expert, leases, _ = host.slot_info(0)[hit_slot]
    assert (state, expert, leases) == (READY, 3, 1), "the failed request released or re-used the hit lane's slot"

    # The consequence, which the membership fact alone does not cover: the slot is not handed to the next request.
    host.inject(fail_reads=False)
    sim.post(0, [5])
    # Deliberately not sim.wait: the failed request above latched the page's fatal word, and
    # exl3_ram_miss_sim_wait returns 3 on a latched fatal as its first statement, before it ever consults
    # demand_done. Waiting on it therefore synchronizes with nothing, leaving the assertion below racing the
    # service thread -- and _slot_of raises StopIteration, not -1, when expert 5 is not mapped yet, so the
    # race surfaces as an intermittent error rather than a clean failure. Poll the actual condition instead.
    assert _until(lambda: any(st == READY and e == 5 for st, e, _, _ in host.slot_info(0))), "expert 5 never landed"
    assert _slot_of(host, 0, 5) != hit_slot, "a leased hit slot was handed to the next request"

    # And the lease is retired by the device, never by the host.
    sim.terminal(req, mask=1 << 0)
    sim.deliver()
    assert _until(lambda: _leases(host, 0)[hit_slot] == 0)
    assert host.counters()["leases_voided"] >= 1
    assert host.counters()["lease_double_signal"] == 0


def test_an_ungranted_lane_keeps_the_ring_entry_open(running):
    """T4c. Between the two grants a miss lane is granted-pending, and the ring entry must stay open: otherwise
    retire_leases clears `active` while a grant is still to come, and the entry.active guard stops protecting the
    ring slot.

    Mutant: revert S4 -- restore the open loop to counting only state == 1. It must go red on the reuse of the
    ring index, which is what entry.active protects.
    """
    s, page, host, sim = running
    _make_resident(host, sim, 0, 3)
    host.inject(delay_s=READ_DELAY_S)
    req = sim.post(0, [3, 4])

    reused = {}

    def observe():
        # Drive retirement from the test thread while the service sits inside read() with the miss lane ungranted.
        deadline = time.perf_counter() + READ_DELAY_S * 0.6
        while time.perf_counter() < deadline:
            sim.deliver()
            time.sleep(0.005)
        reused["deferred"] = host.counters()["deferred_reuse"]

    watcher = threading.Thread(target=observe)
    watcher.start()
    waited = sim.wait(req, timeout_s=READ_DELAY_S + 10.0)
    watcher.join()

    assert waited.status == 1 and waited.go == 2, "retiring mid-request disturbed the request itself"
    # The entry survived the retire sweep with its second grant still pending: every lane was granted in the end.
    assert host.counters()["leases_granted"] == 3
    assert host.counters()["lease_double_signal"] == 0, "the ring entry was reused while a grant was pending"
