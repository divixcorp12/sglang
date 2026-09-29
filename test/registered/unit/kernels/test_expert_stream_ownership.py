"""The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Tasks 13-14): what an unpaused Python call does while
the service thread runs, that queued commands are applied in order before the next request, and that a copy
completion comes back to the owner, which releases its lease (D7)."""

import faulthandler
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, sim_post, sim_wait
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

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
    assert sorted(e for _, e, _, _ in info if e >= 0) == [0, 1, 2, 3]
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
    host.inject(delay_s=0.3)  # instr: every advisory and demand read sleeps 300 ms first
    applied = host.counters()["commands_applied"]  # instr: every command the owner applied, queued or direct
    seq = sim_post(page, 1, need=[6], protect=[6])  # a long read on row 1 keeps the service busy
    time.sleep(0.05)
    start = time.perf_counter()
    for i in range(200):  # the command ring holds 64: the producer must wait, never drop
        # Every command carries its own payload (the bits of i over the row's 8 experts), so none can stand in for
        # another: a dropped or reordered one changes the count below or the last writer.
        host.set_hot(0, [e for e in range(8) if i >> e & 1])
    host.set_hot(0, [0, 1, 2])
    assert sim_wait(page, seq, 5.0) == 1
    free, evictable, leased = host.victim_census(0, [])  # a snapshot: queued behind the burst, answered after it
    assert (free, evictable, leased) == (0, 1, 0)  # only expert 3 is neither hot nor wanted: the LAST set_hot won
    assert host.counters()["commands_applied"] - applied == 201 + 1, "a command was dropped (the +1 is the census)"
    assert time.perf_counter() - start > 0.1, "the burst was applied while the read ran: nothing was queued"


def _copy_request(s, page, host, sim):
    """A resident expert 0 of row 0, then a request whose lane 0 is COPYING; returns (req, slot)."""
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    first = sim.post(0, [0])
    while host.pump():
        pass
    waited, lanes = hp.accept(sim, first)
    sim.ack(first, waited, lanes=lanes)
    sim.deliver()
    while host.pump():
        pass
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    req = sim.post(0, [0], dst=[0], copy_engine=True)
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
        while sim.copy_done(req)[:2] != (hp.lease.COPIED, req.gen):
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
            assert sim.copy_done(req)[:2] == (hp.lease.COPIED, req.gen)
        finally:
            host.resume()
        host.pause(5.0)  # nothing outstanding: the second pause is not refused
        host.resume()
    finally:
        host.stop()
