"""The copy engine's host half (CPU; LEASE_PROTOCOL.md, "Copy engine"): COPYING grants, CopyDone, the gate, and
lease release.

The service publishes a resident lane of a captured request as COPYING and hands the copy to its copy thread. The CPU
test backend lands a job's bytes only when the test releases its mark, so each test can hold a copy in flight and look
at what the service has published meanwhile: no CopyDone, a held lease, and no victim.
"""

import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, page_word
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
DST_ROWS = 6


def _host(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    return s, page, host, LeaseSim(host, page, s.slabs)


def _copy_table(s, dst):
    return torch.tensor(
        [[slab.data_ptr(), dst[name].data_ptr(), slab[0].numel() * slab.element_size()] for name, slab in s.slabs[ROW].items()],
        dtype=torch.int64,
    )


def _copy_engine(s, host, *, arm=True, wait_timeout_ms=2000):
    """The CPU backend, a copy table from the row's slabs to host "destination" tensors, and (by default) armed."""
    host.enable_copy_engine(-1, spin_us=200, wait_timeout_ms=wait_timeout_ms)
    dst = {name: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for name, slab in s.slabs[ROW].items()}
    host.set_copy_table(ROW, _copy_table(s, dst), DST_ROWS)
    if arm:
        host.arm_copy_engine()
    return dst


def _load(sim, host, experts):
    """Make ``experts`` resident in row ROW through a plain (uncaptured) request, and retire its leases."""
    req = sim.post(ROW, experts)
    assert host.pump() == 1 and sim.wait(req).served
    sim.done(req)
    host.pump()  # retires first
    return req


def _slot_of(host, expert):
    return host.mapping(ROW)[expert]


def _rows_equal(dst, slabs, dst_row, host_slot):
    return all(torch.equal(dst[n][dst_row].view(torch.uint8), slabs[n][host_slot].view(torch.uint8)) for n in dst)


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def test_a_hit_lane_is_copying_its_lease_holds_and_copydone_waits_for_the_observed_completion(tmp_path):
    """Mutants: complete a job without querying its mark, or publish CopyDone at the grant -- red on the CopyDone and
    byte checks made while the mark is held; release the lease on the request's Done -- red on the held lease."""
    s, page, host, sim = _host(tmp_path)
    try:
        dst = _copy_engine(s, host)
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        req = sim.post(ROW, [3], dst=[2], captured=True)
        assert host.pump() == 1 and page_word(page, "demand_done") == req.seq
        assert sim.row_result(req, 0) == {"tag": lease.COPYING, "gen": req.gen, "host_slot": slot}
        entry = host.lease_entry(req.idx)
        assert entry["lane_state"][0] == 1 and entry["lane_copy_engine"][0] == 1

        time.sleep(0.05)  # the copy thread has had every chance to publish early
        assert sim.copy_done(req) != req.gen, "CopyDone published before the copy completed"
        assert not any(dst[n][2].view(torch.uint8).any() for n in dst), "bytes landed before the release"
        assert host.slot_info(ROW)[slot][2] == 1, "the lease was released before the copy completed"
        # CW's Done ends the request's kernels, but a COPYING lane's lease belongs to the copy.
        sim.done(req)
        host.pump()
        assert host.lease_entry(req.idx)["lane_state"][0] == 1 and host.slot_info(ROW)[slot][2] == 1

        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(req) == req.gen
        assert _rows_equal(dst, s.slabs[ROW], 2, slot)
        host.pump()  # the owner drains the completion
        assert host.lease_entry(req.idx)["lane_state"][0] == 2 and host.slot_info(ROW)[slot][2] == 0
        counters = host.counters()
        assert counters["leases_copied"] == 1 and counters["copy_jobs"] == 1 and counters["copy_lanes"] == 1
        assert not host.lease_entry(req.idx)["active"], "the entry stayed open after its last lease was released"
    finally:
        host.stop()


def test_a_copy_in_flight_keeps_its_slot_from_being_a_victim_until_it_completes(tmp_path):
    """Victim reuse: the only slot a demand could evict is under a copy-engine copy. The demand defers, and is served
    by evicting that slot only once the copy completed, after its bytes reached the destination."""
    s, page, host, sim = _host(tmp_path)
    try:
        dst = _copy_engine(s, host)
        _load(sim, host, [0, 1, 2, 3])
        slot = _slot_of(host, 3)
        expected = {n: s.slabs[ROW][n][slot].clone() for n in dst}
        req = sim.post(ROW, [3], dst=[4], captured=True)
        assert host.pump() == 1
        sim.done(req)
        assert host.victim_census(ROW, [0, 1, 2]) == (0, 0, 1)
        demand = sim.post(ROW, [4], protect=[0, 1, 2, 4])
        assert host.pump() == 0 and page_word(page, "demand_done") == req.seq, "a leased slot was evicted"
        assert host.counters()["deferred"] >= 1
        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert all(torch.equal(dst[n][4].view(torch.uint8), expected[n].view(torch.uint8)) for n in dst)
        assert host.pump() == 1 and page_word(page, "demand_done") == demand.seq
        assert _slot_of(host, 4) == slot and _slot_of(host, 3) < 0
    finally:
        host.stop()


def test_only_hit_lanes_are_copying(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        dst = _copy_engine(s, host)
        _load(sim, host, [2, 3])
        req = sim.post(ROW, [3, 5, 2], dst=[0, 1, 5], captured=True)
        assert host.pump() == 1
        tags = [sim.row_result(req, lane)["tag"] for lane in range(3)]
        assert tags == [lease.COPYING, lease.LOADING, lease.COPYING]
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(req) == req.gen
        assert _rows_equal(dst, s.slabs[ROW], 0, _slot_of(host, 3)) and _rows_equal(dst, s.slabs[ROW], 5, _slot_of(host, 2))
        host.pump()
        entry = host.lease_entry(req.idx)
        assert entry["lane_copy_engine"][:3] == [1, 0, 1] and entry["lane_state"][:3] == [2, 1, 2]
    finally:
        host.stop()


@pytest.mark.parametrize("case", ["uncaptured", "unarmed", "no_slot", "slot_past_table"])
def test_a_lane_the_copy_engine_cannot_take_is_published_ready(tmp_path, case):
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host, arm=case != "unarmed")
        _load(sim, host, [3])
        dst = {"no_slot": [-1], "slot_past_table": [DST_ROWS]}.get(case, [1])
        req = sim.post(ROW, [3], dst=dst, captured=case != "uncaptured")
        assert host.pump() == 1
        assert sim.row_result(req, 0)["tag"] == lease.READY
        counters = host.counters()
        assert counters["copy_jobs"] == 0 and host.copy_engine_marked() == 0
        assert counters["copy_fallbacks"] == (1 if case in ("no_slot", "slot_past_table") else 0)
    finally:
        host.stop()


_COPY_ENGINE_SCRIPT = """
ROW = 1
dst = {n: torch.zeros((6,) + tuple(slab.shape[1:]), dtype=slab.dtype) for n, slab in s.slabs[ROW].items()}
table = torch.tensor(
    [[slab.data_ptr(), dst[n].data_ptr(), slab[0].numel() * slab.element_size()] for n, slab in s.slabs[ROW].items()],
    dtype=torch.int64,
)
host.enable_copy_engine(-1, spin_us=200, wait_timeout_ms=50)
host.set_copy_table(ROW, table, 6)
host.arm_copy_engine()
req = sim.post(ROW, [3])
assert host.pump() == 1
sim.done(req)
host.pump()
"""


@pytest.mark.parametrize("fault", ["issue", "query"])
def test_a_copy_whose_completion_cannot_be_established_aborts_the_process(tmp_path, fault):
    """A lease released without an observed completion could hand a slot under an in-flight copy to the next read."""
    result = run_host_script(
        tmp_path,
        _COPY_ENGINE_SCRIPT
        + f"""
host.copy_engine_fail(issue={fault == "issue"}, query={fault == "query"})
req = sim.post(ROW, [3], dst=[1], captured=True)
host.pump()
host.copy_engine_release(-1)
host.copy_engine_idle(5.0)
time.sleep(0.5)
print("reached")
""",
    )
    assert_aborted(result, "RAM miss copy engine: copy of request")


def test_a_copy_wait_held_past_its_timeout_aborts_the_process(tmp_path):
    """A copy held forever (its mark never released) against a 50 ms copy-wait timeout: the watchdog aborts rather
    than leave the decode stream waiting on the gate."""
    result = run_host_script(
        tmp_path,
        _COPY_ENGINE_SCRIPT
        + """
req = sim.post(ROW, [3], dst=[2], captured=True)
assert host.pump() == 1
host.start_thread()
sim.close_copy_gate(req)
time.sleep(3.0)
print("reached")
""",
    )
    assert_aborted(result, "a copy wait held the decode stream")


def test_arming_the_copy_engine_before_it_is_enabled_is_refused(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        with pytest.raises(Exception, match="not enabled"):
            host.arm_copy_engine()
    finally:
        host.stop()


# ---- SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES: CW reads the small tensors, the lease waits for its Done ----


def _sm_copy_engine(s, host):
    """As _copy_engine, with the row's small (non-trellis) entries left to CW's SM reads."""
    from sglang.srt.layers.moe.exl3_ram_miss import sm_copy_mask

    host.enable_copy_engine(-1, spin_us=200)
    names = list(s.slabs[ROW])
    dst = {name: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for name, slab in s.slabs[ROW].items()}
    mask = sm_copy_mask(names)
    host.set_copy_table(ROW, _copy_table(s, dst), DST_ROWS, sm_mask=mask)
    host.arm_copy_engine()
    sm_names = [n for i, n in enumerate(names) if mask >> i & 1]
    return dst, sm_names, [n for n in names if n not in sm_names]


def test_the_sm_mask_leaves_exactly_the_trellis_tensors_on_the_copy_engine():
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
    from sglang.srt.layers.moe.exl3_ram_miss import sm_copy_mask

    assert list(EXL3_STREAMED_NAMES) == ["w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"]
    assert sm_copy_mask(EXL3_STREAMED_NAMES) == 0b110110


def test_the_python_names_are_the_host_modules_layout():
    """EXL3_STREAMED_NAMES orders the slab table and the copy table; the C++ trait orders the SM mask. A reorder on
    one side would SM-copy a trellis and DMA a scale vector."""
    from sglang.kernels.ops.moe.expert_stream_transport import host_layout
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES

    names, small_mask = host_layout()
    assert names == EXL3_STREAMED_NAMES
    assert small_mask == 0b110110


def test_a_copy_table_sm_mask_naming_a_trellis_is_refused(tmp_path):
    """set_copy_table refuses an SM mask outside the layout's small tensors: SM-reading a 13 MB trellis in the copy
    wait would stall the chain instead of using the DMA engine."""
    s, _page, host, _sim = _host(tmp_path)
    try:
        host.enable_copy_engine(-1, spin_us=200)
        table = torch.zeros((6, 3), dtype=torch.int64)
        with pytest.raises(RuntimeError, match="exl3 RAM miss: .*small"):
            host.set_copy_table(ROW, table, DST_ROWS, sm_mask=0b000001)
    finally:
        host.stop()


def test_sm_entries_skip_the_dma_and_the_lease_holds_until_cw_publishes_done(tmp_path):
    """The DMA copies only the trellis tensors; CopyDone is published on its completion, but the lease is released only
    once CW's Done shows its SM reads of the slot finished. Mutant: release the lease on the DMA's completion alone --
    red on the held lease and the sentinel below."""
    s, page, host, sim = _host(tmp_path)
    try:
        dst, sm_names, dma_names = _sm_copy_engine(s, host)
        assert len(sm_names) == 4 and len(dma_names) == 2
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        original = {n: s.slabs[ROW][n][slot].clone() for n in dst}
        req = sim.post(ROW, [3], dst=[2], captured=True)
        assert host.pump() == 1 and sim.row_result(req, 0)["tag"] == lease.COPYING

        host.copy_engine_release(-1)
        assert _until(lambda: sim.copy_done(req) == req.gen), "CopyDone never published"
        assert all(torch.equal(dst[n][2].view(torch.uint8), original[n].view(torch.uint8)) for n in dma_names)
        assert not any(dst[n][2].view(torch.uint8).any() for n in sm_names), "the copy engine copied an SM entry"

        time.sleep(0.05)  # the copy thread has had every chance to release early
        host.pump()
        assert host.slot_info(ROW)[slot][2] == 1, "the lease was released before CW's Done"
        assert host.lease_entry(req.idx)["lane_state"][0] == 1 and host.lease_entry(req.idx)["active"]
        assert not host.copy_engine_idle(0.01), "a job awaiting its Done counts as outstanding"

        sim.sm_fetch(req, dst, sm_names)
        sim.done(req)
        assert host.copy_engine_idle(5.0)
        host.pump()
        assert host.slot_info(ROW)[slot][2] == 0 and host.lease_entry(req.idx)["lane_state"][0] == 2
        assert not host.lease_entry(req.idx)["active"]
        # Only now may the service rewrite the slot: the destination must keep the bytes read under the lease.
        for n in dst:
            s.slabs[ROW][n][slot].view(torch.uint8).fill_(0xAB)
        assert all(torch.equal(dst[n][2].view(torch.uint8), original[n].view(torch.uint8)) for n in dst)
        assert host.counters()["leases_copied"] == 1
    finally:
        host.stop()


def test_a_done_of_an_earlier_generation_releases_nothing_and_a_later_one_releases(tmp_path):
    """Done is per ring index: a stale one (an earlier request in the same index) must not hand the job back, and a
    later one (CW of a later request ran, so this request's CW finished) must."""
    s, page, host, sim = _host(tmp_path)
    try:
        _sm_copy_engine(s, host)
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        req = sim.post(ROW, [3], dst=[1], captured=True)
        assert host.pump() == 1
        host.copy_engine_release(-1)
        assert _until(lambda: sim.copy_done(req) == req.gen)
        assert req.gen > 1  # _load posted the earlier request
        sim.done(req, generation=req.gen - 1)
        time.sleep(0.05)
        host.pump()
        assert host.slot_info(ROW)[slot][2] == 1, "a stale Done released the lease"
        sim.done(req, generation=req.gen + 16)
        assert host.copy_engine_idle(5.0)
        host.pump()
        assert host.slot_info(ROW)[slot][2] == 0
    finally:
        host.stop()


def test_a_copy_in_flight_under_sm_reads_keeps_its_slot_from_being_a_victim_until_done(tmp_path):
    """Victim reuse at the SM step: the DMA has completed, CW has not read yet. A demand that could only evict that
    slot defers until Done, and the destination holds the pre-eviction bytes."""
    s, page, host, sim = _host(tmp_path)
    try:
        dst, sm_names, _ = _sm_copy_engine(s, host)
        _load(sim, host, [0, 1, 2, 3])
        slot = _slot_of(host, 3)
        expected = {n: s.slabs[ROW][n][slot].clone() for n in dst}
        req = sim.post(ROW, [3], dst=[4], captured=True)
        assert host.pump() == 1
        host.copy_engine_release(-1)
        assert _until(lambda: sim.copy_done(req) == req.gen)
        demand = sim.post(ROW, [4], protect=[0, 1, 2, 4])
        assert host.pump() == 0 and page_word(page, "demand_done") == req.seq, "a slot under SM reads was evicted"
        sim.sm_fetch(req, dst, sm_names)
        sim.done(req)
        assert host.copy_engine_idle(5.0)
        assert all(torch.equal(dst[n][4].view(torch.uint8), expected[n].view(torch.uint8)) for n in dst)
        assert host.pump() == 1 and page_word(page, "demand_done") == demand.seq
        assert _slot_of(host, 4) == slot
    finally:
        host.stop()


def test_sm_small_copies_are_refused_without_the_copy_engine():
    import msgspec

    from sglang.srt.dsv41_config import Dsv41Config
    from sglang.srt.layers.moe.exl3_ram_miss import check_sm_small_copies

    base = Dsv41Config.from_envs()
    check_sm_small_copies(base)  # off: nothing to check
    with pytest.raises(RuntimeError, match="SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"):
        check_sm_small_copies(msgspec.structs.replace(base, enable_ram_miss_sm_small_copies=True))
    check_sm_small_copies(
        msgspec.structs.replace(base, enable_ram_miss_sm_small_copies=True, enable_ram_miss_copy_engine=True)
    )


def test_the_sm_table_refuses_a_small_tensor_off_16_byte_alignment():
    """CW reads 16-byte units only when both ends allow it and otherwise falls back to 1-byte loads: refuse that at
    bind time rather than run the slow path silently."""
    from types import SimpleNamespace

    from sglang.srt.layers.moe.exl3_ram_miss import sm_copy_table

    table = torch.tensor(
        [[1 << 20, 2 << 20, 1 << 16], [(1 << 20) + 8, 2 << 20, 64], [1 << 20, (2 << 20) + 4, 64], [1 << 20, 2 << 20, 40]],
        dtype=torch.int64,
    )
    segments = SimpleNamespace(table=table)
    assert sm_copy_table(segments, 0b0001).tolist() == [[1 << 20, 2 << 20, 1 << 16]]
    for bad in (0b0010, 0b0100, 0b1000):
        with pytest.raises(ValueError, match="16-byte"):
            sm_copy_table(segments, bad)


# ---- The copy wait's gate (LEASE_PROTOCOL.md, "Copy engine"): CW closes it, the host opens it ----


def test_the_gate_starts_open_so_a_copy_wait_that_closes_nothing_passes(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host)
        assert sim.copy_gate() == lease.gate_word(0, "open")
    finally:
        host.stop()


def test_the_copy_thread_opens_a_closed_gate_once_copydone_is_published(tmp_path):
    """CW closed the gate before the copy completed (and so found no CopyDone): the copy thread opens it right after
    CopyDone. Mutant: open on the grant, or before CopyDone -- red on the closed-gate check made while the mark is
    held."""
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host)
        _load(sim, host, [3])
        req = sim.post(ROW, [3], dst=[2], captured=True)
        assert host.pump() == 1 and sim.row_result(req, 0)["tag"] == lease.COPYING
        sim.close_copy_gate(req)
        host.pump()
        time.sleep(0.05)  # every releaser has had its chance to open early
        assert sim.copy_gate() == lease.gate_word(req.seq, "closed"), "the gate opened before CopyDone"
        host.copy_engine_release(1)
        assert _until(lambda: sim.copy_gate() == lease.gate_word(req.seq, "open")), sim.copy_gate()
        assert sim.copy_done(req) == req.gen
    finally:
        host.stop()


def test_a_copy_completion_for_g_leaves_the_next_requests_closed_gate_alone(tmp_path):
    """G's copy completes after CW of G + 1 already closed the gate for G + 1 (CW opened G's gate itself): the host
    changes the gate only by a CAS from G's exact closed word, so G + 1's closed gate stays. Mutant: a plain store of
    open(G) -- red on the gate word."""
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host)
        _load(sim, host, [3])
        req = sim.post(ROW, [3], dst=[2], captured=True)
        assert host.pump() == 1
        next_closed = lease.gate_word(req.seq + 1, "closed")
        sim._set_gate(next_closed)
        host.copy_engine_release(1)
        assert _until(lambda: sim.copy_done(req) == req.gen)
        host.pump()
        assert sim.copy_gate() == next_closed
    finally:
        host.stop()


def test_stopping_the_service_thread_opens_a_closed_gate(tmp_path):
    """Teardown (ExpertStreamHost.stop, the atexit hook): with no copy thread to follow, a copy wait still closed would
    hold its stream forever, so stop_thread opens it. Mutant: no open in stop_thread -- red on the gate word."""
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host)
        _load(sim, host, [3])
        req = sim.post(ROW, [3], dst=[2], captured=True)
        assert host.pump() == 1
        host.start_thread()
        sim.close_copy_gate(req)  # the mark is held: no CopyDone, nothing else would open it
        host._module.expert_stream_stop_thread(host.handle)
        host.threaded = False
        assert sim.copy_gate() == lease.gate_word(req.seq, "open")
    finally:
        host.copy_engine_release(-1)
        host.stop()


def test_the_host_changes_the_gate_only_by_a_cas_from_the_closed_word():
    """Source pin for the release above: no plain store to the gate is left in the host after its initialisation."""
    from pathlib import Path

    import sglang.kernels.ops.moe.expert_stream_transport as transport

    tier = (Path(transport.__file__).resolve().parents[2] / "jit/csrc/moe/expert_stream/host/ram_tier.h").read_text()
    assert "store_release(lease_ + kLeaseCopyGate" not in tier
    assert tier.count("std::memcpy(lease_ + kLeaseCopyGate") == 1  # init_lease_block, before any thread
    assert "reinterpret_cast<uint32_t*>(lease_ + kLeaseCopyGate), &expected" in tier
