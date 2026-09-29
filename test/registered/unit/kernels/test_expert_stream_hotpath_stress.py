"""Every party that touches the RAM-miss service concurrently -- the device, the copy thread's completions, unpaused
Python calls, and an eager caller's pause/resume -- against the running service thread, with the invariants that the
tier mutex protected at ba01695c35 checked at every pause and at the end (plan 2026-09-29-hotpath-zero-overhead
Task 4). Green at ba01695c35; the lock-free single-owner tier (Tasks 13-15) must keep it green, and Task 16 runs
``run_stress`` under ThreadSanitizer."""

import random
import threading
import time

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import DEMAND_RECORDS, page_word, sim_wait
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import same_bytes

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

READY, FREE = 2, 0  # slot_info states
# The device routes only to these experts; the noise party marks only the others VRAM-hot. apply_gpu_hot rewrites a
# row's hot set from the (empty) sidecar record at every armed demand, so a set_hot landing between that and the take
# loop would otherwise shrink the victims the demand's census counted and fail it (no_victim), a legal outcome that
# is not what this test is about. Hot experts that are never resident keep set_hot's concurrent writes and remove that.
DEVICE_EXPERTS = tuple(range(hp.EXPERTS - 2))
HOT_ONLY = tuple(range(hp.EXPERTS - 2, hp.EXPERTS))
# The grants a served request's lane may carry: a resident hit, a miss streamed in pieces, a hit the copy engine copies.
LANE_TAGS = {lease.READY: "lanes_ready", lease.LOADING: "lanes_loading", lease.COPYING: "lanes_copying"}


def _check_tier(s, host, rows):
    """The tier invariants for an owner that holds the service parked: no lease, no LOADING or QUARANTINE slot,
    mapping and slot_info agree both ways, the lease block's SlotGen words are the tier's generations, and every READY
    slot holds its expert's checkpoint bytes. Returns the rows' (slot_info, mapping)."""
    out = {}
    for row in rows:
        info, mapping, gens = host.slot_info(row), host.mapping(row), host.mapped_slot_generations(row)
        for slot, (state, expert, leases, gen) in enumerate(info):
            assert leases == 0, f"row {row} slot {slot} still leased: {info}"
            assert state in (FREE, READY), f"row {row} slot {slot} in state {state} with the service parked: {info}"
            assert gens[slot] == gen & 0xFFFFFFFF, f"row {row} slot {slot}: SlotGen {gens[slot]} != generation {gen}"
            if state == READY:
                assert mapping[expert] == slot, f"row {row}: READY slot {slot} of expert {expert}, mapping {mapping}"
                oracle = s.reference(s.tables.layer_ids[row], [expert])
                assert all(same_bytes(s.slabs[row][n][slot], oracle[n][0]) for n in EXL3_STREAMED_NAMES), (
                    f"row {row} slot {slot}: READY bytes differ from the checkpoint's expert {expert}")
        for expert, slot in enumerate(mapping):
            assert slot < 0 or (info[slot][0] == READY and info[slot][1] == expert), (
                f"row {row}: expert {expert} maps to slot {slot} = {info[slot] if slot >= 0 else None}")
        out[row] = {"info": info, "mapping": mapping}
    assert all(not host.lease_entry(i)["active"] for i in range(DEMAND_RECORDS)), "a lease entry is still active"
    return out


def _fill_while_owned(s, host, rng):
    """The paused owner starts a prefill fill of up to two missing experts of a row, admits a third into that row while
    the fill thread reads (the owner's census reads the filling flags the fill must leave alone), then ends the fill
    or, half the time, leaves it running for resume() to join and finish before the service runs again. True when it
    left the fill running."""
    row = rng.randrange(hp.LAYERS)
    mapping = host.mapping(row)
    missing = [e for e in DEVICE_EXPERTS if mapping[e] < 0]
    rng.shuffle(missing)
    slots, _evictions = host.fill_begin(row, missing[:2], fallback=True)
    if len(missing) > 2:
        slot, _evicted = host.assign(row, missing[2])
        oracle = s.reference(s.tables.layer_ids[row], [missing[2]])
        for n in EXL3_STREAMED_NAMES:
            s.slabs[row][n][slot].copy_(oracle[n][0])
    host.slot_info(row)
    if slots and rng.random() < 0.5:
        host.fill_wait(len(slots), 10.0)
        assert host.fill_end(), "a prefill fill failed"
        return False
    return bool(slots)


def run_stress(tmp_path, *, variant=None, seconds=8.0, seed=1, fills=False):
    """Run the four parties for ``seconds`` against a threaded host, then park the service and check the tier.
    Raises AssertionError on a party's error or a broken tier invariant; returns the parties' counts, ``fatal``, the
    service's counters and the final rows, for the caller's checks that every party ran.

    ``fills`` (Task 16's ThreadSanitizer child): every pause also starts a prefill fill, admits a row into the filled
    row while the fill thread reads (spec 6.3 item 3), and either ends the fill or leaves it for ``resume()`` to join,
    so the fill thread and the owner's epilogue meet every other party.

    The pauser runs on the calling thread, the one that built the host: every eager call that may touch an io_uring
    reader (``pause``, ``assign``, ``resume``, and ``RamMissSetup.reference``, whose shared Exl3ShardRowSource reader
    belongs to the thread that first opened it) must come from that owner, as master's eager contract requires. The
    device, the copy releaser and the noise run on worker threads."""
    s, page, host, sim, dst = hp.build_host(tmp_path, variant=variant)
    host.start_thread(fatal_wait_s=60.0, spin_us=2000)
    stop, quiet, idle = threading.Event(), threading.Event(), threading.Event()
    device_gone = threading.Event()  # set once the device thread has left: the releaser outlives its last wait
    stats = {
        "armed": 0, "copy_posts": 0, "terminals": 0, "timeouts": 0, "failed": 0,
        "lanes_ready": 0, "lanes_loading": 0, "lanes_copying": 0,
        "releases": 0, "marked": 0, "noise": 0, "pauses": 0, "refused": 0, "assigns": 0, "fills": 0, "errors": [],
    }

    def guard(name, fn):
        def run():
            try:
                fn()
            except BaseException as error:  # noqa: BLE001 - reported through stats["errors"], and every party stops
                stats["errors"].append(f"{name}: {error!r}")
                stop.set()
        return run

    def device():
        rng = random.Random(seed)
        try:
            while not stop.is_set():
                if quiet.is_set():
                    idle.set()
                    time.sleep(0.001)
                    continue
                row = rng.randrange(hp.LAYERS)
                lanes = rng.sample(DEVICE_EXPERTS, rng.randint(1, 3))
                hp.write_hot_record(page, host, hp.next_seq(page), [])
                use_copy = rng.random() < 0.5
                req = sim.post(row, lanes, dst=list(range(len(lanes))) if use_copy else None, copy_engine=use_copy)
                stats["armed"] += 1
                stats["copy_posts"] += int(use_copy)
                status = sim_wait(page, req.seq, 10.0)
                if status != 1:
                    stats["timeouts" if status == 0 else "failed"] += 1
                    raise AssertionError(f"request {req.seq} (row {row}, lanes {lanes}) ended with sim_wait {status}")
                for lane in range(len(req.lanes)):
                    result = sim.row_result(req, lane)
                    if result["gen"] != req.gen or result["tag"] not in LANE_TAGS:
                        raise AssertionError(f"request {req.seq} lane {lane} was served without a grant: {result}")
                    stats[LANE_TAGS[result["tag"]]] += 1
                waited, ack_lanes = hp.accept(sim, req)
                if rng.random() < 0.1:
                    sim.terminal(req, (1 << len(req.lanes)) - 1)
                    stats["terminals"] += 1
                else:
                    sim.ack(req, waited, lanes=ack_lanes)
                sim.deliver()
        finally:
            idle.set()  # a device that left (stop, or an error) is quiet: a pauser waiting on it must not time out
            device_gone.set()

    def releaser():
        # HostCopyBackend::release(n) adds n marks of credit and release(-1) is sticky ("every mark, from now on"), so
        # each tick releases one mark, and only while a mark is outstanding (marked > released): copies complete one
        # by one, at the releaser's pace, instead of all of them from the first tick on. -1 is kept for the final drain.
        # It runs until the device has left, not until stop: the device's request in flight at stop may be deferred on
        # a COPYING lease that only a released mark retires, and stopping first deadlocks it until sim_wait times out.
        rng = random.Random(seed + 3)
        while not device_gone.is_set():
            if host.copy_engine_marked() > stats["releases"]:
                host.copy_engine_release(1)
                stats["releases"] += 1
            time.sleep(rng.uniform(0.0002, 0.001))

    def noise():
        rng = random.Random(seed + 1)
        while not stop.is_set():
            host.set_hot(rng.randrange(hp.LAYERS), rng.sample(HOT_ONLY, rng.randint(0, len(HOT_ONLY))))
            host.counters()
            host.mapping(rng.randrange(hp.LAYERS))
            host.mapped_slot_generations(rng.randrange(hp.LAYERS))
            stats["noise"] += 1
            time.sleep(0.0002)

    def quiesce_device():
        """Ask the device to stop posting and wait until it has; False when every party is stopping instead."""
        idle.clear()
        quiet.set()
        deadline = time.monotonic() + 15.0
        while not idle.wait(0.05):
            if stop.is_set():
                return False
            if time.monotonic() > deadline:
                raise AssertionError("the device did not go quiet within 15 s")
        return True

    def pauser():
        rng = random.Random(seed + 2)
        deadline = time.monotonic() + seconds
        while not stop.wait(0.2) and time.monotonic() < deadline:
            try:
                if not quiesce_device():
                    return
                try:
                    host.pause(5.0)
                except RuntimeError as error:
                    if "graph-lane lease" not in str(error):
                        raise  # a pause that timed out is a hang, not a legal refusal
                    stats["refused"] += 1
                    continue
                left_running = False
                try:
                    stats["pauses"] += 1
                    _check_tier(s, host, range(hp.LAYERS))
                    host.lru_order(0)
                    host.counters()
                    # The eager caller's one write: an expert that is not resident, assigned and filled by the owner.
                    row = rng.randrange(hp.LAYERS)
                    mapping = host.mapping(row)
                    resident = [e for e in DEVICE_EXPERTS if mapping[e] >= 0]
                    missing = [e for e in DEVICE_EXPERTS if mapping[e] < 0]
                    expert = rng.choice(missing)
                    slot, _evicted = host.assign(row, expert, protected=resident[:1])
                    oracle = s.reference(s.tables.layer_ids[row], [expert])
                    for n in EXL3_STREAMED_NAMES:
                        s.slabs[row][n][slot].copy_(oracle[n][0])
                    stats["assigns"] += 1
                    _check_tier(s, host, [row])
                    if fills:
                        left_running = _fill_while_owned(s, host, rng)
                        stats["fills"] += 1
                finally:
                    if left_running:
                        # The device may post while the fill still runs: resume() must join the fill and run its
                        # epilogue before it hands the tier back, or the service serves that demand beside them.
                        quiet.clear()
                        time.sleep(0.005)
                    host.resume()
            finally:
                quiet.clear()

    threads = [threading.Thread(target=guard(name, fn), name=f"stress-{name}", daemon=True)
               for name, fn in (("device", device), ("releaser", releaser), ("noise", noise))]
    try:
        for t in threads:
            t.start()
        guard("pauser", pauser)()  # on this thread, the host's owner
        stop.set()
        for t in threads:
            t.join(30.0)
        alive = [t.name for t in threads if t.is_alive()]
        assert not alive, f"parties still running after stop: {alive}; errors {stats['errors']}"
        assert not stats["errors"], f"a party failed: {stats['errors']}"
        # Final quiesce: every ack and terminal was delivered, and the copy thread drains once its marks complete.
        stats["marked"] = host.copy_engine_marked()
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0), "the copy engine did not go idle"
        host.pause(5.0)
        try:
            rows = _check_tier(s, host, range(hp.LAYERS))
            report = {"stats": stats, "fatal": page_word(page, "fatal"), "counters": host.counters(), "rows": rows}
        finally:
            host.resume()
    finally:
        stop.set()
        host.stop()
    return report


def test_every_party_against_the_service_keeps_the_tier_invariants(tmp_path):
    report = run_stress(tmp_path, seconds=8.0, seed=1)
    stats, counters = report["stats"], report["counters"]
    print("STRESS stats", {k: v for k, v in stats.items() if k != "errors"}, "counters", counters)
    assert stats["errors"] == [], stats["errors"]
    assert stats["timeouts"] == 0 and stats["failed"] == 0
    assert report["fatal"] == 0 and counters["read_errors"] == 0, (report["fatal"], counters)
    # Every armed post was served exactly once: read rows (served) or found every row resident (touch_only).
    assert counters["served"] + counters["touch_only"] == stats["armed"], (counters, stats)
    # Every party ran, and every path it exists to exercise was taken.
    assert stats["armed"] > 200, stats
    assert counters["served"] > 0 and counters["touch_only"] > 0, counters
    assert stats["lanes_loading"] > 0 and stats["lanes_ready"] > 0, stats
    assert stats["copy_posts"] > 0 and stats["lanes_copying"] > 0, stats
    assert stats["terminals"] > 0, stats
    # Copies completed one mark at a time (C: the releaser never grants credit ahead of an outstanding mark).
    assert 100 < stats["releases"] <= stats["marked"], stats
    assert stats["noise"] > 100, stats
    assert stats["pauses"] >= 8 and stats["assigns"] == stats["pauses"], stats
