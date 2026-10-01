"""The service thread, its pause handshake, its watchdog, and what a demand may evict (CPU, simulated device)."""

import faulthandler
import gc
import os
import sys
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script, same_bytes

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # The service thread and its handshakes run in C++: a broken handshake or join hangs,
    # so dump every stack and exit instead (pytest-timeout could not interrupt it).
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _attached(s, page, slot_map, k):
    host = ExpertStreamHost(s.tables, page=page, slot_map=slot_map)
    host.reserve_staging(k)
    return host


def _host(tmp_path, capacity=3, fatal_wait_s=5.0, k=1):
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = _attached(s, page, slot_map, k)
    host.start_thread(fatal_wait_s=fatal_wait_s)
    return s, page, slot_map, host, ChainSim(host, page, s.slabs)


def _holds(host, row, expert):
    """True once ``expert`` holds a slot of ``row`` (LOADING or READY), through a snapshot: ``contains`` refuses
    while the service thread runs unpaused."""
    return expert in host.slot_to_expert(row)


def _ready_map(host, row, experts=6):
    """Each expert's READY slot, else -1, from the tier's own slot states: ``mapping()`` reads the published slot map
    itself, so a check of that map must derive the READY slots independently."""
    ready = [-1] * experts
    for slot, (state, expert, _) in enumerate(host.slot_info(row)):
        if state == 2 and expert >= 0:
            ready[expert] = slot
    return ready


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _serve(sim, row, lanes, timeout_s=10.0, **post):
    """One whole device chain: post, then wait as S does (every miss's pieces) and for the service to finish it."""
    req = sim.post(row, lanes, **post)
    assert sim.wait_served(req, timeout_s=timeout_s), f"request {req.seq} was not served"
    assert sim.wait_handled(req, timeout_s=timeout_s)
    return req


def test_the_thread_serves_demands_without_a_pump(tmp_path):
    s, page, slot_map, host, sim = _host(tmp_path, capacity=4, k=2)
    try:
        _serve(sim, 0, [1, 2])
        assert _holds(host, 0, 1) and host.counters()["running"] == 1
        with pytest.raises(RuntimeError, match="pump"):
            host.pump()
    finally:
        host.stop()


def test_concurrent_eager_use_and_demands_never_share_a_slot(tmp_path):
    """Between pause() and resume() Python owns the tier: a demand the service would serve meanwhile must not take
    the slot an eager assignment just made, and the device-visible map stays the READY slots throughout. The eager
    side hands its bulk delta to the device before resuming, as Exl3RamMissService.after_host_use does."""
    s, page, slot_map, host, sim = _host(tmp_path, capacity=4)
    stop = threading.Event()
    device = threading.Lock()  # the device's map: one chain at a time, the bulk apply between chains
    errors = []

    def eager():
        for expert in range(200):
            try:
                with device:
                    host.pause(timeout_s=2.0)
                    try:
                        e = expert % 6
                        if not host.contains(0, e):
                            host.assign(0, e, protected=[e])
                        slot = host.mapping(0)[e]
                        time.sleep(0.002)  # an eager fill; no demand may take the slot meanwhile
                        if slot >= 0 and host.slot_to_expert(0)[slot] != e:
                            errors.append(("slot taken while paused", e, slot))
                        mapping = host.mapping(0)
                        slots = [m for m in mapping if m >= 0]
                        if len(slots) != len(set(slots)):
                            errors.append(mapping)
                        sim.apply_bulk_like_device(host.take_bulk_delta().tolist())
                    finally:
                        host.resume()
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(repr(error))
        stop.set()

    worker = threading.Thread(target=eager)
    worker.start()
    try:
        expert = 0
        while not stop.is_set():
            with device:
                _serve(sim, 0, [expert % 6])
            expert += 1
        worker.join(timeout=30)
        assert not errors, errors[:3]
        assert expert > 0
        host.pause(timeout_s=2.0)
        try:
            mapping = host.mapping(0)
            slots = [m for m in mapping if m >= 0]
            assert len(slots) == len(set(slots))
            assert slot_map[0].tolist() == mapping == _ready_map(host, 0)  # the device-visible map: the READY slots
            sim.apply_bulk_like_device(host.take_bulk_delta().tolist())
            assert sim.replica.ram_slot[0] == mapping  # the device's map, deltas and bulk applied, is the host's
        finally:
            host.resume()
    finally:
        host.stop()


def test_a_failed_read_aborts_the_process_before_its_pieces_say_landed(tmp_path):
    """Fail-stop: a read failure is not reported to the device; the process dies before a piece word could say
    landed for a row that never did."""
    result = run_host_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=5.0)
        host.inject(fail_reads=True)
        req = sim.post(0, [1])
        sim.wait_served(req, timeout_s=5.0)
        print("reached", sim.served(req))
        """,
    )
    assert_aborted(result, "a test fault failed the read")


def test_the_watchdog_aborts_a_hung_read(tmp_path):
    result = run_host_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=0.3)
        host.inject(delay_s=30.0)
        sim.post(0, [1])
        time.sleep(3.0)
        print("reached")
        """,
    )
    assert_aborted(result, "stayed in service")


def test_a_stop_during_a_hung_read_still_ends_in_the_watchdog_abort(tmp_path):
    # stop() waits for the service thread; the watchdog must outlive that wait.
    result = run_host_script(
        tmp_path,
        """
        host.start_thread(fatal_wait_s=0.5)
        host.inject(delay_s=30.0)
        sim.post(0, [1])
        time.sleep(0.1)
        host.stop()
        print("reached")
        """,
        timeout_s=25,
    )
    assert_aborted(result, "stayed in service")


def test_the_thread_runs_on_the_core_it_is_pinned_to(tmp_path):
    s = ram_miss_setup(tmp_path)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    try:
        core = min(os.sched_getaffinity(0))
        assert core < 64
        host.start_thread(cpu_core=core)
        assert host.counters()["spin_cpu"] == core
    finally:
        host.stop()


@pytest.mark.parametrize("core, error, match", [(71, ValueError, "64-71"), (1000, RuntimeError, "pin")])
def test_a_reserved_or_unusable_core_is_refused(tmp_path, core, error, match):
    s = ram_miss_setup(tmp_path)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    try:
        with pytest.raises(error, match=match):
            host.start_thread(cpu_core=core)
        assert not host.threaded
    finally:
        host.stop()


def test_a_pause_that_times_out_raises_and_leaves_the_thread_running(tmp_path):
    s, page, slot_map, host, sim = _host(tmp_path)
    try:
        host.inject(delay_s=1.0)
        req = sim.post(0, [1])
        assert _until(lambda: host.busy_episode() != 0)
        with pytest.raises(RuntimeError, match="did not pause"):
            host.pause(timeout_s=0.1)
        assert sim.wait_served(req, 3.0) and sim.wait_handled(req)
        host.inject(delay_s=0.0)
        _serve(sim, 0, [2], timeout_s=3.0)
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_stop_while_paused_returns_promptly(tmp_path):
    s, page, slot_map, host, sim = _host(tmp_path)
    host.pause(timeout_s=1.0)
    started = time.perf_counter()
    host.stop()
    assert time.perf_counter() - started < 1.0 and not host._close.alive


def test_collecting_a_threaded_host_stops_its_thread(tmp_path):
    s, page, slot_map, host, sim = _host(tmp_path)
    _serve(sim, 0, [1])
    del host
    gc.collect()
    req = sim.post(0, [2])  # a miss: only a live service could publish its pieces
    time.sleep(0.5)
    assert not sim.served(req), "a collected host's thread still served a demand"


# ---- Pump-driven: what a demand reads, and what it may evict ----


def _tier(tmp_path, capacity=6, k=1):
    """A host with no service thread: the tests pump it, so nothing races."""
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = _attached(s, page, torch.full((2, 6), -1, dtype=torch.int32), k)
    return s, page, host, ChainSim(host, page, s.slabs)


def _pump_serve(host, sim, row, lanes, **post):
    """Post, pump once (serving it), and check every miss's pieces landed."""
    req = sim.post(row, lanes, **post)
    assert host.pump() == 1 and sim.wait_served(req, 1.0)
    return req


def _assert_resident_rows_exact(s, host, row):
    """Every READY row holds exactly its expert's bytes, and nothing is left LOADING (staging slots have no owner)."""
    mapping = host.mapping(row)
    resident = [e for e, slot in enumerate(mapping) if slot >= 0]
    reference = s.reference(row, resident)
    for i, expert in enumerate(resident):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[row][name][mapping[expert]], reference[name][i]), (row, expert)
    slots = [slot for slot in mapping if slot >= 0]
    assert len(slots) == len(set(slots))
    owned = [e for e in host.slot_to_expert(row) if e >= 0]
    assert sorted(owned) == sorted(resident)  # a slot with an owner is mapped: none was left LOADING


def test_a_demand_reads_all_its_rows_in_one_batch(tmp_path):
    s, page, host, sim = _tier(tmp_path, k=2)
    try:
        host.enable_trace()
        _pump_serve(host, sim, 1, [4, 5])
        (demand,) = host.drain_trace()
        assert (demand["kind"], demand["status"], demand["rows"]) == ("demand", "served", 2)
        assert demand["rows_reading_max"] == 2 and demand["batches"] == 1  # both rows in one batch
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_a_demand_that_takes_every_staging_slot_is_served(tmp_path):
    s, page, host, sim = _tier(tmp_path, capacity=6, k=3)
    try:
        host.enable_trace()
        _pump_serve(host, sim, 1, [0, 1, 2])
        (record,) = host.drain_trace()
        assert (record["status"], record["rows"], record["rows_reading_max"]) == ("served", 3, 3)
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_two_misses_naming_one_staging_slot_abort_the_process(tmp_path):
    """The device assigns each miss its own staging slot; a record that reuses one would read two rows into it."""
    result = run_host_script(
        tmp_path,
        """
        stage = sim.staging(0)[0]
        sim.post(0, [0, 1], kinds=[4, 4], slots=[stage, stage])
        host.pump()
        print("reached")
        """,
        capacity=4,
        staging=2,
    )
    assert_aborted(result, "shares its staging slot")


# A record carries `protect` (the routed experts) and its lanes. serve_record chooses each miss's RAM victim with
# take_victim_locked, and `wanted` (protect and the lanes) is the only thing that keeps a resident row from being that
# request's own victim: a resident expert in neither is a legal victim.


def _fill(host, sim, experts):
    for expert in experts:  # each is a demand of its own, so the first one served is the least recently used
        _pump_serve(host, sim, 1, [expert])


def _resident(host, row=1):
    return [e for e in range(6) if host.contains(row, e)]


def test_a_resident_expert_outside_protect_is_evicted_by_the_request_that_needs_another(tmp_path):
    s, page, host, sim = _tier(tmp_path, capacity=4)
    try:
        _fill(host, sim, [0, 1, 2])  # expert 0 is the least recently used
        assert _resident(host) == [0, 1, 2] and host.counters()["evictions"] == 0
        _pump_serve(host, sim, 1, [4])  # expert 0 is neither a lane nor protected
        assert _resident(host) == [1, 2, 4]
        assert host.counters()["evictions"] == 1
        _assert_resident_rows_exact(s, host, 1)
    finally:
        host.stop()


def test_a_resident_expert_in_protect_is_not_the_victim(tmp_path):
    s, page, host, sim = _tier(tmp_path, capacity=4)
    try:
        _fill(host, sim, [0, 1, 2])
        _pump_serve(host, sim, 1, [4], protect=[4, 0])  # the same request, now protecting expert 0
        assert _resident(host) == [0, 2, 4]  # expert 1 went instead
    finally:
        host.stop()


def test_when_every_resident_expert_is_protected_the_miss_is_served_and_not_cached(tmp_path):
    """The case that separates protection from mere recency: only when NO other victim exists is the protection the
    only thing standing between a resident row and eviction. The miss is still read and served; it is not cached."""
    s, page, host, sim = _tier(tmp_path, capacity=4)
    try:
        _fill(host, sim, [0, 1, 2])
        _pump_serve(host, sim, 1, [4], protect=[0, 1, 2, 4])
        assert _resident(host) == [0, 1, 2]
        assert host.counters()["ram_insert_skipped"] == 1 and host.counters()["evictions"] == 0
    finally:
        host.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
