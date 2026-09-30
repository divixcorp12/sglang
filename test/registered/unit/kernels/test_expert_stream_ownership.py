"""The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Tasks 13-15): what an unpaused Python call does while
the service thread runs, that queued commands are applied in order before the next request, that a copy
completion comes back to the owner, which releases its lease (D7), that a prefill fill's epilogue runs on the owner
after the join, and that the tier declares no mutex but the Python callers' own."""

import faulthandler
import re
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # A broken ownership handoff hangs in C++ (a snapshot nobody answers): dump every stack and exit instead.
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, layers=2, experts=8)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32))
    for expert in range(4):
        host.assign(0, expert)  # pump mode: the caller owns the tier
    host.start_thread(fatal_wait_s=60.0, spin_us=2000)
    yield s, page, host
    host.stop()


def test_unpaused_eager_calls_refuse_or_snapshot(running):
    s, page, host = running
    for call in (
        lambda: host.assign(0, 5),
        lambda: host.touch(0, 1),
        lambda: host.contains(0, 1),
        lambda: host.release(0, 0),
        lambda: host.fill_begin(0, [5]),
    ):
        with pytest.raises(RuntimeError, match="paused"):
            call()
    info = host.slot_info(0)
    mapping = host.mapping(0)
    assert sorted(e for _, e, _ in info if e >= 0) == [0, 1, 2, 3]
    assert all(mapping[e] >= 0 and info[mapping[e]][1] == e for e in range(4))
    assert host.lease_entry(0)["active"] in (0, False)
    assert sorted(host.lru_order(0)) == [0, 1, 2, 3] and sorted(host.slot_to_expert(0)) == [0, 1, 2, 3]
    assert host.victim_census(0, []) == (0, 4, 0)
    host.pause(5.0)
    try:
        slot, _evicted = host.assign(0, 5)  # the owner may assign (it evicts an LRU row: capacity is full)
        assert slot >= 0 and host.mapping(0)[5] == slot
        assert host.contains(0, 5)
    finally:
        host.resume()


def test_a_set_hot_burst_past_the_ring_is_applied_in_order(running):
    s, page, host = running
    host.inject(delay_s=0.3)  # instr: every demand read sleeps 300 ms first
    applied = host.counters()["commands_applied"]  # instr: every command the owner applied, queued or direct
    sim = LeaseSim(host, page, s.slabs)
    req = sim.post(1, [6])  # a long read on row 1 keeps the service busy
    time.sleep(0.05)
    start = time.perf_counter()
    for i in range(200):  # the command ring holds 64: the producer must wait, never drop
        # Every command carries its own payload (the bits of i over the row's 8 experts), so none can stand in for
        # another: a dropped or reordered one changes the count below or the last writer.
        host.set_hot(0, [e for e in range(8) if i >> e & 1])
    host.set_hot(0, [0, 1, 2])
    # Taken before the wait (Task 16's M3 passed the old check, taken after the 300 ms read): queued, the 65th set_hot
    # waits for the service to drain the full ring, which it does only once the read ends, ~0.25 s from `start`.
    # Applied directly, the whole burst takes milliseconds.
    burst_s = time.perf_counter() - start
    assert sim.wait(req, 5.0).served
    sim.done(req)
    free, evictable, leased = host.victim_census(0, [])  # a snapshot: queued behind the burst, answered after it
    assert (free, evictable, leased) == (0, 1, 0)  # only expert 3 is neither hot nor wanted: the LAST set_hot won
    assert host.counters()["commands_applied"] - applied == 201 + 1, "a command was dropped (the +1 is the census)"
    assert burst_s > 0.1, f"the burst returned in {burst_s:.3f} s, while the read ran: nothing was queued"


def _copy_request(s, page, host, sim):
    """A resident expert 0 of row 0, then a request whose lane 0 is COPYING; returns (req, slot)."""
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    first = sim.post(0, [0])
    while host.pump():
        pass
    sim.done(first)
    while host.pump():
        pass
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    req = sim.post(0, [0], dst=[0], captured=True)
    while host.pump():
        pass
    return req, host.mapping(0)[0]


def test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread(tmp_path):
    """D7: the copy thread publishes CopyDone and hands the job back; the owner releases the lease when it next runs."""
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req, slot = _copy_request(s, page, host, sim)
        host.copy_engine_release(-1)
        deadline = time.time() + 5
        while sim.copy_done(req) != req.gen:
            assert time.time() < deadline
            time.sleep(0.001)
        assert host.slot_info(0)[slot][2] == 1, "the copy thread released the lease itself"
        host.pump()
        assert host.slot_info(0)[slot][2] == 0, "the owner's poll did not release it"
    finally:
        host.stop()


def test_a_pause_retires_a_copy_that_completed_while_parked(tmp_path):
    """Review Focus 3: pause() waits for the copy engine to go idle, then the pausing caller (the owner) drains the
    completion ring and retires the COPYING lease itself, so the pause is not refused. A pause with nothing
    outstanding is not refused either."""
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req, slot = _copy_request(s, page, host, sim)
        host.start_thread(fatal_wait_s=60.0, spin_us=2000)
        threading.Timer(0.2, lambda: host.copy_engine_release(-1)).start()
        host.pause(5.0)  # waits for the copy engine to go idle, then the pausing caller drains and retires
        try:
            assert host.slot_info(0)[slot][2] == 0
            assert sim.copy_done(req) == req.gen
        finally:
            host.resume()
        host.pause(5.0)  # nothing outstanding: the second pause is not refused
        host.resume()
    finally:
        host.stop()


def _code(path):
    """The source with its // comments cut, so prose that names a lock proves nothing either way."""
    return "\n".join(line.split("//", 1)[0] for line in path.read_text().splitlines())


def _struct(code, name, opener=None):
    """The text of `struct name {` (or `opener`) up to the first `};` or closing `}` at the opener's indent."""
    opener = opener or f"struct {name} {{"
    start = code.index(opener)
    indent = code.rfind("\n", 0, start) + 1
    close = "\n" + " " * (start - indent) + "}"
    return code[start : code.index(close, start)]


def test_the_tier_declares_only_the_callers_mutex():
    """Task 15's source backstop to the shim's runtime zero (plan F11: names, not a count). The tier declares
    caller_mutex_ (Python callers against each other only) and the two InstrBuild-only guards, TraceState::mutex and
    TierFaults::fault_mutex, which ProdBuild's static_asserts leave without storage; it locks nothing else, and no other
    lock stands in for the deleted mutex_: no rwlock, recursive, timed or pthread lock, no atomic_flag or exchange spin
    lock, no raw futex (the only futex is the copy engine's documented idle protocol, in spsc_ring.h), and the one
    compare-exchange is cas_gate's open of the copy wait's gate (LEASE_PROTOCOL.md, "Copy engine")."""
    code = _code(MOE / "expert_stream" / "host" / "ram_tier.h")
    assert set(re.findall(r"std::mutex\s+(\w+)\s*[;{]", code)) == {"caller_mutex_", "fault_mutex", "mutex"}
    assert "std::mutex fault_mutex;" in _struct(code, "TierFaults"), "fault_mutex left TierFaults"
    assert "std::mutex mutex;" in _struct(code, "TraceState"), "the trace guard left TraceState"
    locked = set(re.findall(r"(?:lock_guard|unique_lock|scoped_lock)<[^>]*>\s*\w+\(([^)]*)\)", code))
    assert locked == {"caller_mutex_", "trace_.mutex", "faults_.fault_mutex"}, locked
    assert "std::mutex mutex_" not in code and "guard(mutex_)" not in code and "self->mutex_" not in code
    for other in ("shared_mutex", "recursive_mutex", "timed_mutex", "pthread_", "atomic_flag", "test_and_set",
                  ".exchange(", "futex", ".wait(", "notify_one", "notify_all", "condition_variable"):
        assert other not in code, other
    assert code.count("compare_exchange") == 1 and "__atomic_compare_exchange_n(" in code
    assert "__atomic_compare_exchange_n(" in _struct(code, "cas_gate", opener="void cas_gate(")


# One row's pack takes this long under the pack_delay fault, so a fill of a few rows is still running when checked.
SLOW_PACK_NS = 300_000_000


def _fill_host(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4)
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full(tuple(s.tables.starts.shape), -1, dtype=torch.int32)
    )
    return s, host


def _fill_thread_done(host, claimed):
    """Returns once the fill thread made its last store (fill_state_ left running): a wait for one row more than it
    claimed fails exactly then."""
    with pytest.raises(RuntimeError, match="failed before"):
        host.fill_wait(claimed + 1, 10.0)


def test_a_fill_holds_its_slots_until_the_owner_joins_it(tmp_path):
    """Ownership rule 3 and spec 6.3 item 3: the fill thread writes no tier state, so its slots stay filling after its
    read ended, until the owner's fill_end runs the epilogue. Meanwhile the owner's own admission (Python's chunked
    admission, take_admit_slot_locked) may run on the same row with no lock: it cannot take a filling slot, and nothing
    else writes the tier to race it."""
    s, host = _fill_host(tmp_path)
    try:
        slots, _ = host.fill_begin(0, [0, 1, 2, 3])
        assert len(slots) == 4
        _fill_thread_done(host, 4)
        assert host.fill_landed() == 4
        with pytest.raises(RuntimeError, match="fill"):
            host.release(0, slots[3])  # the fill thread has ended, but the epilogue is the owner's
        with pytest.raises(RuntimeError, match="protected or leased"):
            host.assign(0, 5, protected_fallback=True)  # every slot is still filling: no victim
        assert host.fill_end()
        host.release(0, slots[3])  # the epilogue ran on this thread, at fill_end
        assert host.assign(0, 5)[0] >= 0
    finally:
        host.fill_end()
        host.stop()


def test_a_failed_fill_is_released_and_counted_by_the_owner_at_fill_end(tmp_path):
    """The epilogue of a failed fill -- releasing its rows that did not land, counting the read error -- runs on the
    owner at fill_end, not on the fill thread when its read fails."""
    s, host = _fill_host(tmp_path)
    try:
        path = s.tables.paths[int(s.tables.extents[1, 2, 0, 0])]
        with open(path, "r+b") as f:
            f.truncate(int(s.tables.extents[1, 2, 0, 1]) + 100)
        host.fill_begin(1, [2])
        with pytest.raises(RuntimeError, match="failed"):
            host.fill_wait(1, 10.0)  # fill_state_ is failed: the fill thread's last store
        assert host.contains(1, 2), "the fill thread released a row itself"
        assert host.counters()["read_errors"] == 0, "the fill thread counted the read error itself"
        assert not host.fill_end()
        assert not host.contains(1, 2)
        assert host.counters()["read_errors"] == 1
    finally:
        host.stop()


def test_a_stop_mid_pause_joins_the_running_fill_before_its_final_settle(tmp_path):
    """Controller ruling (Task 15): stop_thread can arrive while the pausing caller's fill still runs. Its final settle,
    on the stopping caller once the service joined, joins the fill and runs the epilogue there before it drains and
    settles, with no tier lock: when stop_thread returns the fill's slots are no longer filling."""
    s, host = _fill_host(tmp_path)
    try:
        host.start_thread(fatal_wait_s=30.0)
        host.pause(5.0)
        host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
        slots, _ = host.fill_begin(0, [0, 1])
        assert host.fill_landed() < 2  # still running
        host._module.expert_stream_stop_thread(host.handle)  # the service stops mid-pause; the tier stays open
        host.threaded = False
        assert host.fill_landed() == 2
        host.release(0, slots[1])  # the final settle ran the epilogue: the slot is no longer filling
        assert host.fill_end()  # nothing left to join
        host.inject_fault()
    finally:
        host.stop()


def test_start_thread_refuses_a_tier_whose_fill_is_still_running(tmp_path):
    """B3 review Important 1: a pump-mode caller that began a fill and then starts the service thread would leave the
    service and the fill thread sharing the reader (spec 6.2/D8), and fill_end() would then run the epilogue on the
    Python thread while the service owns the tier. start_thread refuses instead, until the fill's owner ended it; after
    fill_end it starts, and a later fill_end has nothing owed and touches nothing."""
    s, host = _fill_host(tmp_path)
    try:
        host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
        slots, _ = host.fill_begin(0, [0, 1])  # pump mode: the caller owns the tier
        assert len(slots) == 2 and host.fill_landed() < 2  # still running
        with pytest.raises(RuntimeError, match="fill_end"):
            host.start_thread(fatal_wait_s=30.0)
        assert not host.threaded
        assert host.fill_landed() < 2, "start_thread waited for the fill instead of refusing"
        assert host.fill_end()  # the owner (still the pump caller) joins it and runs the epilogue
        host.release(0, slots[1])  # the epilogue ran here: the slot is no longer filling
        host.inject_fault()
        host.start_thread(fatal_wait_s=30.0)  # nothing owed now
        assert host.fill_end()  # unpaused, nothing owed: no epilogue, so nothing to refuse
        with pytest.raises(RuntimeError, match="paused"):
            host.release(0, slots[0])  # and the service owns the tier again
    finally:
        host.stop()
