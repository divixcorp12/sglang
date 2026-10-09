"""The speculative threads (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Demand priority", "Threads and
cores", "Failure handling"): one per group, started and stopped with the service threads, taking turns with the
demand read on the group's reader, quiesced by pause (and so by a prefill fill), timed by the watchdog (CPU)."""

import faulthandler
import json
import os
import time

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, spawn_child
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, prefetch_rig, trigger

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # Joins and handshakes run in C++: dump every stack and exit instead of hanging the suite.
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _until(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _spec_tids():
    tids = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                if f.read().strip().startswith("exl3-spec"):
                    tids.append(int(tid))
        except (FileNotFoundError, ProcessLookupError):
            pass  # the thread exited mid-scan
    return tids


def _started(tmp_path, *, delay_s=0.0, cores=None, per_token=1, per_layer=1, rows=2):
    rig = prefetch_rig(tmp_path, rows=rows)
    enable(rig, LOGITS, cores=cores, per_token=per_token, per_layer=per_layer)
    if delay_s:
        rig.host.inject_spec(delay_s=delay_s)
    rig.host.start_thread(fatal_wait_s=5.0)
    return rig


def _pooled(host, row):
    return sorted(e["expert"] for e in host.spec_pool(row) if e["state"] == "landed")


def test_the_speculative_thread_runs_on_its_cores_and_needs_no_pump(tmp_path):
    core = sorted(os.sched_getaffinity(0))[-1]
    rig = _started(tmp_path, cores=[[core]])
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 1)
        tids = _spec_tids()
        assert len(tids) == 1 and os.sched_getaffinity(tids[0]) == {core}
    finally:
        rig.host.stop()
    assert _spec_tids() == []


def test_without_the_prefetch_no_speculative_thread_starts(tmp_path):
    rig = prefetch_rig(tmp_path)  # a pool, but never enabled
    try:
        rig.host.start_thread(fatal_wait_s=5.0)
        assert _spec_tids() == []
    finally:
        rig.host.stop()


def test_a_demand_read_waits_for_at_most_the_speculative_row_in_flight(tmp_path):
    """Mutant: the speculative read outside the turn lock -- red (the demand is served while the read still sleeps)."""
    rig = _started(tmp_path, delay_s=0.5)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        start = time.monotonic()
        req = rig.sim.post(1, [3])  # a GPU miss of row 1, not the pick
        assert rig.sim.wait_served(req, timeout_s=5.0)
        waited = time.monotonic() - start
        c = rig.host.counters()
        assert c["spec_landed"] == 1, "the demand read ran beside the speculative read"
        assert c["spec_delayed"] == 1 and waited < 2.0
    finally:
        rig.host.stop()


def test_a_demand_waiting_at_the_turn_goes_before_the_jobs_next_read(tmp_path):
    """Picks 2 then 4; a GPU miss of row 2 (neither the job's source nor its target, so the job stays live) arrives
    while 2 reads. Mutant: no demand_waiting gate -- red (the thread retakes the turn and reads 4 first)."""
    rig = _started(tmp_path, delay_s=0.4, per_token=2, per_layer=2, rows=3)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        req = rig.sim.post(2, [3])
        assert rig.sim.wait_served(req, timeout_s=5.0)
        c = rig.host.counters()
        assert (c["spec_landed"], c["spec_delayed"]) == (1, 1), "the demand waited for more than one speculative read"
        assert _until(lambda: rig.host.counters()["spec_landed"] == 2)
    finally:
        rig.host.stop()


def test_a_forced_miss_whose_pool_read_fails_reads_the_row_itself(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        rig.host.inject_spec(delay_s=0.4, fail=True)
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rows = rig.host.counters()["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_failed"], c["spec_used"], c["rows_read"]) == (1, 1, 0, rows + 1)
        assert rig.host.mapping(1)[2] >= 0 and all(e["state"] == "empty" for e in rig.host.spec_pool(1))
    finally:
        rig.host.stop()


def test_a_forced_miss_on_a_row_still_reading_waits_for_it_and_swaps_it_in(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rows = rig.host.counters()["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_used"], c["rows_read"]) == (1, 1, rows)
        assert rig.host.mapping(1)[2] >= 0
    finally:
        rig.host.stop()


def test_a_gpu_miss_on_a_row_still_reading_reads_into_staging_and_the_pool_row_lands(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        req = rig.sim.post(1, [2])
        assert rig.sim.wait_served(req, timeout_s=5.0) and rig.sim.wait_handled(req, timeout_s=5.0)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 1)
        pooled = next(e for e in rig.host.spec_pool(1) if e["expert"] == 2)
        assert pooled["state"] == "landed" and rig.host.mapping(1)[2] not in (-1, pooled["slot"])
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_used"]) == (0, 0)
    finally:
        rig.host.stop()


def test_a_job_whose_target_row_is_in_service_reads_no_further_pick(tmp_path):
    """Picks 2 then 4; while 2 reads, a GPU miss of 4 in the target row waits at the turn. The record in service makes
    the job stale, so 4 is read once, by the demand. Mutant: note the row's seq after handle_record -- red."""
    rig = _started(tmp_path, delay_s=0.4, per_token=2, per_layer=2)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        req = rig.sim.post(1, [4])
        assert rig.sim.wait_served(req, timeout_s=5.0) and rig.sim.wait_handled(req, timeout_s=5.0)
        assert _until(lambda: rig.host.counters()["spec_landed"] + rig.host.counters()["spec_dropped"] >= 2)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_landed"], c["spec_dropped"], c["spec_delayed"]) == (1, 1, 1, 1)
        assert _pooled(rig.host, 1) == [2] and rig.host.mapping(1)[4] >= 0
    finally:
        rig.host.stop()


def test_pause_waits_for_the_speculative_read_and_a_prefill_fill_reads_alone(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rig.host.pause(10.0)
        try:
            assert rig.host.counters()["spec_landed"] == 1, "pause returned with a speculative read in flight"
            slots, _ = rig.host.fill_begin(1, [3])
            assert len(slots) == 1 and rig.host.fill_end()
            rig.sim.sync_bulk()
        finally:
            rig.host.resume()
        rig.host.inject_spec()
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 2), "the thread did not resume"
    finally:
        rig.host.stop()


@pytest.mark.parametrize("meanwhile", ["mapped", "pooled"])
def test_pause_holds_a_job_between_its_reads_and_the_next_pick_is_rechecked(tmp_path, meanwhile):
    """Picks 2 then 4: pause returns once 2 landed, before 4 is read; the owner then maps or pools 4, and after the
    resume the job drops it instead of reading it again."""
    rig = _started(tmp_path, delay_s=0.4, per_token=2, per_layer=2)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rig.host.pause(10.0)
        try:
            c = rig.host.counters()
            assert (c["spec_issued"], c["spec_landed"], c["spec_dropped"]) == (1, 1, 0), "pause waited for the job"
            if meanwhile == "mapped":
                slots, _ = rig.host.fill_begin(1, [4])
                assert len(slots) == 1 and rig.host.fill_end()
                rig.sim.sync_bulk()
            else:
                rig.host.spec_place(1, 4)
            rig.host.inject_spec()
        finally:
            rig.host.resume()
        assert _until(lambda: rig.host.counters()["spec_dropped"] == 1)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_landed"]) == (1, 1)
        assert _pooled(rig.host, 1) == ([2] if meanwhile == "mapped" else [2, 4])
    finally:
        rig.host.stop()


def test_stop_with_a_read_in_flight_joins_and_reports_the_pool(tmp_path, capfd):
    rig = _started(tmp_path, delay_s=0.4)
    trigger(rig)
    assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
    rig.host.stop()
    lines = [l for l in capfd.readouterr().err.splitlines() if l.startswith("exl3 RAM miss thread counters ")]
    counters = json.loads(lines[-1].removeprefix("exl3 RAM miss thread counters "))
    assert (counters["spec_issued"], counters["spec_landed"]) == (1, 1)
    assert _spec_tids() == []


_HUNG = """
import pathlib, sys, time
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, prefetch_rig, trigger
rig = prefetch_rig(pathlib.Path(sys.argv[1]))
enable(rig, LOGITS)
rig.host.inject_spec(delay_s=30.0)
rig.host.start_thread(fatal_wait_s=0.3)
trigger(rig)
time.sleep(3.0)
print("reached")
"""


def test_the_watchdog_aborts_a_hung_speculative_read(tmp_path):
    result = spawn_child(_HUNG, tmp_path, timeout_s=90, variant="instr")
    assert_aborted(result, "a speculative read stayed in service")
