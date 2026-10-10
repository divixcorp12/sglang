"""The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Tasks 13-15): what an unpaused Python call does while
the service thread runs, that the copy thread publishes a completed copy's CopyDone itself, that a prefill fill's epilogue runs on the owner after the join, and that
the tier declares no mutex but the Python callers' own."""

import faulthandler
import re
import threading

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test import hotpath_script as hp
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # A broken ownership handoff hangs in C++: dump every stack and exit instead.
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=5, layers=2, experts=8)  # one staging slot: four mappable rows
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page)
    for expert in range(4):
        host.assign(0, expert)  # pump mode: the caller owns the tier
    sim = ChainSim(host, page, s.slabs)
    sim.sync_bulk()  # the device learns the eager assignments before any post
    host.start_thread(fatal_wait_s=60.0)
    yield s, page, host, sim
    host.stop()


def test_unpaused_calls_that_need_the_tier_refuse_and_the_published_map_reads(running):
    s, page, host, sim = running
    for call in (
        lambda: host.assign(0, 5),
        lambda: host.touch(0, 1),
        lambda: host.contains(0, 1),
        lambda: host.release(0, 0),
        lambda: host.fill_begin(0, [5]),
        lambda: host.set_hot(0, [0]),
        lambda: host.slot_info(0),
        lambda: host.slot_to_expert(0),
        lambda: host.lru_order(0),
        lambda: host.victim_census(0, []),
    ):
        with pytest.raises(RuntimeError, match="needs the service thread paused"):
            call()
    mapping = host.mapping(0)
    assert sorted(e for e, slot in enumerate(mapping) if slot >= 0) == [0, 1, 2, 3]
    host.pause(5.0)
    try:
        info = host.slot_info(0)
        assert sorted(e for _, e, _ in info if e >= 0) == [0, 1, 2, 3]
        assert all(info[mapping[e]][1] == e for e in range(4))
        assert sorted(host.lru_order(0)) == [0, 1, 2, 3]
        assert sorted(e for e in host.slot_to_expert(0) if e >= 0) == [0, 1, 2, 3]  # the staging slot holds none
        assert host.victim_census(0, []) == (0, 4)
        host.set_hot(0, [0, 1, 2])
        assert host.victim_census(0, []) == (0, 1)  # only expert 3 is not hot
        slot, _evicted = host.assign(0, 5)  # the owner may assign (it evicts an LRU row: capacity is full)
        assert slot >= 0 and host.mapping(0)[5] == slot
        assert host.contains(0, 5)
    finally:
        host.resume()


def _copy_request(s, page, host, sim):
    """A resident expert 0 of row 0, then a request whose lane 0 the copy engine copies; returns the request."""
    first = sim.post(0, [0])
    while host.pump():
        pass
    assert sim.served(first)
    req = sim.post(0, [0], dst=[0], captured=True)
    while host.pump():
        pass
    assert req.kinds == [LaneKind.HIT_COPY]
    return req


def test_the_copy_thread_publishes_copy_done_once_its_mark_completes(tmp_path):
    """The copy thread itself publishes CopyDone (and opens the gate) when the job's copies complete; nothing waits for
    the owner's next poll."""
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req = _copy_request(s, page, host, sim)
        assert sim.copy_done(req) != req.gen  # the mark is held
        host.copy_engine_release(-1)
        assert sim.copy_wait(req, 5.0)
    finally:
        host.stop()


def test_a_pause_waits_for_an_outstanding_copy(tmp_path):
    """Review Focus 3: pause() waits for the copy engine to go idle, so a copy that completes while the caller waits is
    not a refusal. A pause with nothing outstanding is not refused either."""
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req = _copy_request(s, page, host, sim)
        host.start_thread(fatal_wait_s=60.0)
        threading.Timer(0.2, lambda: host.copy_engine_release(-1)).start()
        host.pause(5.0)  # waits for the copy engine to go idle
        try:
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
    lock, no raw futex (the only futex is the idle threads' Doorbell, in spsc_ring.h), and no compare-exchange: the one
    the tier relies on is the lease channel's cas_gate, its open of the copy wait's gate (host/lease_channel.h,
    LEASE_PROTOCOL.md "The lease channel").

    The RAM prefetch (ram_prefetch.h) adds two, off the request path's own state: the pool's per-group mutex
    (SpecPool::mutex, taken as pool_->mutex(g)) over its entries' transitions, and SpecGroup::turn, the shared
    reader's turn between a demand read and a speculative one, which an idle-drive group
    (SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE, its own reader) never takes."""
    code = _code(MOE / "expert_stream" / "host" / "ram_tier.h")
    assert set(re.findall(r"std::mutex\s+(\w+)\s*[;{]", code)) == {"caller_mutex_", "fault_mutex", "mutex", "turn"}
    assert "std::mutex fault_mutex;" in _struct(code, "TierFaults"), "fault_mutex left TierFaults"
    assert "std::mutex mutex;" in _struct(code, "TraceState"), "the trace guard left TraceState"
    assert "std::mutex turn;" in _struct(code, "SpecGroup"), "the shared reader's turn left SpecGroup"
    locked = set(re.findall(r"(?:lock_guard|unique_lock|scoped_lock)<[^>]*>\s*\w+\(([^)]*)\)", code))
    # The regex stops at the first ")": the pool's mutex reads as "pool_->mutex(g".
    assert locked == {
        "caller_mutex_", "trace_.mutex", "faults_.fault_mutex", "pool_->mutex(g", "spec.turn, std::defer_lock"
    }, locked
    assert "std::mutex mutex_" not in code and "guard(mutex_)" not in code and "self->mutex_" not in code
    # pthread_ names and pins threads (pthread_setname_np, pthread_setaffinity_np); none of its locks may appear.
    for other in ("shared_mutex", "recursive_mutex", "timed_mutex", "pthread_mutex", "pthread_rwlock", "pthread_spin",
                  "pthread_cond", "atomic_flag", "test_and_set",
                  ".exchange(", "futex", ".wait(", "notify_one", "notify_all", "condition_variable"):
        assert other not in code, other
    assert "compare_exchange" not in code
    channel = _code(MOE / "expert_stream" / "host" / "lease_channel.h")
    assert channel.count("compare_exchange") == 1
    assert "__atomic_compare_exchange_n(" in _struct(channel, "cas_gate", opener="void cas_gate(")


# One row's pack takes this long under the pack_delay fault, so a fill of a few rows is still running when checked.
SLOW_PACK_NS = 300_000_000


def _fill_host(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4)
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False, wire=wire_layout(8)), slot_map=torch.full(tuple(s.tables.starts.shape), -1, dtype=torch.int32)
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
    on the stopping caller once the service joined, joins the fill and runs the epilogue there before it settles,
    with no tier lock: when stop_thread returns the fill's slots are no longer filling."""
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
