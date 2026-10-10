"""Speculative reads only to an idle drive (SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE; design
2026-10-09-dsv41-drive-aware-reads, change (2), section 5): each group's speculative thread reads its pool rows through a
reader of its own (IdleRoots), one piece at a time, from a mirror root no demand sub-read is on in any group; it waits
for one without a ring wait, abandons on a deadline, a settled stale target, a hold or a stop, and a forced miss that
waits on it boosts it to a demand's speed. The demand reads keep their table roots and take no turn.

The reader tests drive one speculative read (read_rows_idle) against a fake demand load; the tier tests run the
speculative thread over three mirror roots, the demand load faked with inject_demand_load.

Mutants (divix01, reverted): gate_root keeps the row's root without rechecking it -- red
(test_each_piece_reads_a_root_no_demand_reads_and_moves_when_demand_arrives); boost ignored -- red
(test_a_boosted_read_reads_its_table_roots_despite_demand, test_a_forced_miss_boosts_the_read_it_waits_on); one
piece limit dropped -- red (the two one-root tests); ReaderCore::drain keeps the read's drive-load share -- red (the
in-flight submit faults)."""

import errno
import faulthandler
import json
import time

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_idle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes, warm_host_modules
from sglang.test.dsv41_ram_prefetch_fixtures import LANES, LOGITS, enable, forced, load, prefetch_rig, trigger

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROOTS = 3
ROW = 1
EXPERT = 3
INSTANT = ("demand_reads", "spec_reads", "demand_inflight_bytes", "spec_inflight_bytes")


@pytest.fixture(autouse=True)
def hang_guard():
    # Joins and deferral loops run in C++: dump every stack and exit instead of hanging the suite. A cold JIT build
    # takes 50-100 s, so the rig's build loads first, outside the guard.
    warm_host_modules("instr", lanes=LANES, nodes=1)
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def setup(tmp_path):
    # Multi-page rows over three roots, so piece streaming cuts each part into several sub-reads (pieces).
    return ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0,) * ROOTS, hidden=256, inter=512)


def idle(snap):
    """Nothing in flight on any root."""
    return all(root[k] == 0 for root in snap["roots"] for k in INSTANT)


def row_exact(s, slot=0):
    oracle = s.reference(s.tables.layer_ids[ROW], [EXPERT])
    return all(same_bytes(s.slabs[ROW][name][slot], oracle[name][0]) for name in oracle)


def test_each_piece_reads_a_root_no_demand_reads_and_moves_when_demand_arrives(setup):
    """Root 0 carries demand throughout; demand arrives on the first piece's root while it is in flight. The rest of
    the row moves to the one root left, one piece at a time, and lands byte for byte. Mutant: no recheck -- red."""
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0], preload=(1, 0, 0), follow=True)
    assert got["result"] == 1 and not got["abandoned"] and not got["boosted"]
    roots = [root for root, _, _ in got["sqes"]]
    assert len(roots) == got["pieces"] >= 3, got
    first = roots[0]
    assert first != 0
    rest = {1, 2} - {first}
    assert set(roots[1:]) == rest, roots
    assert got["moves"] == 1 and got["widest"] == 1
    assert row_exact(setup)
    assert got["share_after"] == 0 and idle(got["load"])


def test_with_no_demand_the_row_reads_one_root_one_piece_at_a_time(setup):
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0])
    roots = {root for root, _, _ in got["sqes"]}
    assert got["result"] == 1 and len(roots) == 1 and got["widest"] == 1
    assert (got["moves"], got["deferrals"]) == (0, 0)
    # Every root's file serves the whole row: the bytes are the row's, whichever root read them.
    assert sum(length for _, _, length in got["sqes"]) >= sum(
        int(n) for n in setup.tables.extents[ROW, EXPERT, :, 2].tolist()
    )
    assert row_exact(setup) and idle(got["load"])


def test_a_piece_with_no_idle_root_waits_then_the_read_is_abandoned(setup):
    start = time.monotonic()
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0], preload=(1, 1, 1), deadline_s=0.05)
    assert time.monotonic() - start >= 0.05
    assert got["result"] == 0 and got["abandoned"] and got["deferrals"] == 1
    assert got["sqes"] == [] and got["share_after"] == 0 and idle(got["load"])


def test_give_up_abandons_mid_row_and_drains_the_piece_in_flight(setup):
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0], give_up_after=1)
    assert got["result"] == 0 and got["abandoned"] and got["pieces"] == 1
    assert got["share_after"] == 0 and idle(got["load"])


def test_a_boosted_read_reads_its_table_roots_despite_demand(setup):
    """Mutant: boost ignored -- red (the read waits for an idle root until its deadline and is abandoned)."""
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0], preload=(1, 1, 1), boost=True, deadline_s=0.05)
    assert got["result"] == 1 and got["boosted"] and not got["abandoned"] and got["deferrals"] == 0
    assert {root for root, _, _ in got["sqes"]} == set(range(ROOTS)) and got["widest"] > 1
    assert row_exact(setup) and idle(got["load"])


@pytest.mark.parametrize(
    "fault",
    [
        dict(part=0, part_error=errno.EIO),
        dict(cqe_error=errno.EIO, cqe_call=2),
        dict(part=-1, root=1, part_short=4096, short_is_eof=True),
        # Pieces still in flight when the read fails (one; every table root's, boosted): the drain settles them.
        dict(submit_error=errno.EIO, submit_call=1, submit_first=True),
        dict(boost=True, submit_error=errno.EIO, submit_call=1, submit_first=True),
    ],
    ids=["part_error", "cqe_error", "part_short_eof", "submit_error_in_flight", "boosted_submit_error_in_flight"],
)
def test_a_faulted_idle_drive_read_fails_without_abandoning_and_leaves_the_load_idle(setup, fault):
    got = read_rows_idle(setup.tables, ROW, [EXPERT], [0], preload=(1, 0, 1) if "root" in fault else (), **fault)
    assert got["result"] == 0 and not got["abandoned"], got
    assert got["share_after"] == 0 and idle(got["load"])


# ---- The tier: the speculative thread on its own reader ----


def _rig(tmp_path, *, deadline_s, delay_s=0.0, per_token=1, per_layer=1, rows=2):
    rig = prefetch_rig(tmp_path, mirror_weights=(1.0,) * ROOTS, rows=rows)
    enable(rig, LOGITS, per_token=per_token, per_layer=per_layer, idle_deadline_s=deadline_s)
    if delay_s:
        rig.host.inject_spec(delay_s=delay_s)
    rig.host.start_thread(fatal_wait_s=5.0)
    return rig


def _until(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _pool(host, row):
    return {e["expert"]: e["state"] for e in host.spec_pool(row) if e["expert"] >= 0}


def _waiting(rig):
    """The job's read of expert 2 into row 1 is issued and still reading; the counters only count its deferrals once
    it returns, so give it a moment to be in its wait."""
    assert _until(lambda: rig.host.counters()["spec_issued"] == 1 and _pool(rig.host, 1) == {2: "reading"})
    time.sleep(0.05)
    assert _pool(rig.host, 1) == {2: "reading"}


def test_a_demand_is_served_while_a_slowed_idle_drive_read_holds_a_drive(tmp_path):
    """The inverse of the shared reader's turn: the demand reads beside the speculative piece in flight (at most one
    piece shares its drive) instead of waiting for the whole row. The demand is of row 2, neither the job's source nor
    its target, so the job stays live."""
    rig = _rig(tmp_path, deadline_s=1.0, delay_s=0.5, rows=3)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        assert _until(lambda: sum(r["spec_reads"] for r in rig.host.drive_load()["roots"]) == 1)
        start = time.monotonic()
        req = rig.sim.post(2, [3])
        assert rig.sim.wait_served(req, timeout_s=5.0)
        waited = time.monotonic() - start
        c = rig.host.counters()
        assert c["spec_landed"] == 0 and waited < 0.4, "the demand waited for the speculative read"
        assert c["spec_delayed"] == 0
        assert _until(lambda: rig.host.counters()["spec_landed"] == 1)
    finally:
        rig.host.stop()


def test_an_idle_drive_read_with_every_drive_busy_is_abandoned_and_its_entry_left_empty(tmp_path, capfd):
    rig = _rig(tmp_path, deadline_s=0.05)
    try:
        rig.host.inject_demand_load([1] * ROOTS)
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_abandoned"] == 1)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_landed"], c["spec_failed"]) == (1, 0, 0) and c["spec_deferred"] >= 1
        assert _pool(rig.host, 1) == {}
        rig.host.inject_demand_load([-1] * ROOTS)
        assert idle(rig.host.drive_load())
    finally:
        rig.host.stop()
    lines = [l for l in capfd.readouterr().err.splitlines() if l.startswith("exl3 RAM miss thread counters ")]
    counters = json.loads(lines[-1].removeprefix("exl3 RAM miss thread counters "))
    assert counters["spec_abandoned"] == 1 and counters["spec_deferred"] >= 1


def test_a_forced_miss_boosts_the_read_it_waits_on(tmp_path):
    """Every drive carries demand, so the read waits; the forced miss on its expert boosts it, and it lands at once
    and is swapped in. Mutant: boost ignored -- red (the miss waits out the 2 s deadline and reads the row itself)."""
    rig = _rig(tmp_path, deadline_s=2.0)
    try:
        rig.host.inject_demand_load([1] * ROOTS)
        trigger(rig)
        _waiting(rig)
        rows = rig.host.counters()["rows_read"]
        start = time.monotonic()
        forced(rig, 1, [2])
        elapsed = time.monotonic() - start
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_boosted"], c["spec_used"], c["spec_abandoned"]) == (1, 1, 1, 0)
        assert c["rows_read"] == rows and elapsed < 1.0
        assert rig.host.mapping(1)[2] >= 0
        rig.host.inject_demand_load([-1] * ROOTS)
    finally:
        rig.host.stop()


def test_a_stale_read_is_abandoned_once_its_record_was_served_without_it(tmp_path):
    """A record of the target row is served and does not want the pick: the waiting read has no use left and gives up
    well before its deadline."""
    rig = _rig(tmp_path, deadline_s=5.0)
    try:
        rig.host.inject_demand_load([1] * ROOTS)
        trigger(rig)
        _waiting(rig)
        start = time.monotonic()
        req = rig.sim.post(1, [3])
        assert rig.sim.wait_served(req, timeout_s=5.0) and rig.sim.wait_handled(req, timeout_s=5.0)
        assert _until(lambda: rig.host.counters()["spec_abandoned"] == 1, timeout_s=2.0)
        assert time.monotonic() - start < 2.0 and rig.host.counters()["spec_landed"] == 0
        rig.host.inject_demand_load([-1] * ROOTS)
    finally:
        rig.host.stop()


def test_pause_abandons_a_waiting_read_and_the_fill_reads_alone(tmp_path):
    rig = _rig(tmp_path, deadline_s=5.0)
    try:
        rig.host.inject_demand_load([1] * ROOTS)
        trigger(rig)
        _waiting(rig)
        start = time.monotonic()
        rig.host.pause(10.0)
        try:
            assert time.monotonic() - start < 2.0 and rig.host.counters()["spec_abandoned"] == 1
            slots, _ = rig.host.fill_begin(1, [3])
            assert len(slots) == 1 and rig.host.fill_end()
            rig.sim.sync_bulk()
        finally:
            rig.host.resume()
        rig.host.inject_demand_load([-1] * ROOTS)
    finally:
        rig.host.stop()


def test_a_fault_reaches_the_speculative_reader_and_the_load_returns_idle(tmp_path):
    rig = _rig(tmp_path, deadline_s=1.0)
    try:
        load(rig, 0, [5])  # trigger's resident, read now so the fault meets the speculative read
        rig.host.inject_fault(part=0, part_error=errno.EIO)
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_failed"] == 1)
        c = rig.host.counters()
        assert (c["spec_landed"], c["spec_abandoned"], c["read_errors"]) == (0, 0, 0)
        assert idle(rig.host.drive_load())
    finally:
        rig.host.stop()
