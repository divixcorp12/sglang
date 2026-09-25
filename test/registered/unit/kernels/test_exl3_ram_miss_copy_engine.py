"""The copy engine's host half (CPU; LEASE_PROTOCOL.md 7.6): COPYING grants, completion ordering and lease release.

The service publishes a resident lane of a request whose post allows it as COPYING and hands the copy to its copy
thread. The CPU test backend lands a job's bytes only when the test releases its mark, so each test can hold a copy in
flight and look at what the service has published meanwhile: no CopyDone, a held lease, and no victim.
"""

import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
DST_ROWS = 6


def _host(tmp_path, *, piece_stream=True, arm=True):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False, pack_workers=2
    )
    host.enable_lease_mode()
    host.enable_two_phase()
    if piece_stream:
        host.enable_piece_stream()
    return s, page, host, LeaseSim(host, page, s.slabs)


def _copy_engine(s, host, *, arm=True):
    """The CPU backend, a copy table from the row's slabs to host "destination" tensors, and (by default) armed."""
    host.enable_copy_engine(-1, spin_us=200)
    dst = {name: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for name, slab in s.slabs[ROW].items()}
    table = torch.tensor(
        [[slab.data_ptr(), dst[name].data_ptr(), slab[0].numel() * slab.element_size()] for name, slab in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    if arm:
        host.arm_copy_engine()
    return dst


def _accept_and_ack(sim, req):
    """The device's S/A path for a served request whose lanes are all READY or LOADING: acknowledge every lane."""
    host = sim.host
    ctx = []
    for lane in range(len(req.lanes)):
        result = sim.row_result(req, lane)
        assert result["gen"] == req.gen and result["tag"] in (lease.READY, lease.LOADING), (lane, result)
        ctx.append((result["host_slot"], result["slot_generation"]))
    waited = type("W", (), {"go": len(ctx), "ctx": ctx})()
    sim.ack(req, waited)
    sim.deliver()
    host.pump()  # retires first


def _load(sim, host, experts):
    """Make ``experts`` resident in row ROW through a plain (no copy engine) request, and retire its leases."""
    req = sim.post(ROW, experts)
    assert host.pump() == 1
    _accept_and_ack(sim, req)
    return req


def _slot_of(host, expert):
    return host.mapping(ROW)[expert]


def _rows_equal(dst, slabs, dst_row, host_slot):
    return all(torch.equal(dst[n][dst_row].view(torch.uint8), slabs[n][host_slot].view(torch.uint8)) for n in dst)


def test_a_hit_lane_is_copying_its_lease_holds_and_copydone_waits_for_the_observed_completion(tmp_path):
    """Mutants: complete a job without querying its mark, or publish CopyDone at the grant -- red on the CopyDone
    and byte checks made while the mark is held; release the lease on a terminal bit -- red on the held lease."""
    s, page, host, sim = _host(tmp_path)
    try:
        dst = _copy_engine(s, host)
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        req = sim.post(ROW, [3], dst=[2], copy_engine=True)
        assert host.pump() == 1 and page_word(page, "demand_done") == req.seq
        result = sim.row_result(req, 0)
        assert result["tag"] == lease.COPYING and result["gen"] == req.gen and result["host_slot"] == slot
        entry = host.lease_entry(req.idx)
        assert entry["lane_state"][0] == 1 and entry["lane_copy_engine"][0] == 1

        time.sleep(0.05)  # the copy thread has had every chance to publish early
        assert sim.copy_done(req)[:2] != (lease.COPIED, req.gen), "CopyDone published before the copy completed"
        assert not any(dst[n][2].view(torch.uint8).any() for n in dst), "bytes landed before the release"
        assert host.slot_info(ROW)[slot][2] == 1, "the lease was released before the copy completed"
        # The device gives up on the request; a COPYING lane's bit in the terminal must not release its lease.
        sim.terminal(req, 1)
        sim.deliver()
        host.pump()
        assert host.lease_entry(req.idx)["lane_state"][0] == 1 and host.slot_info(ROW)[slot][2] == 1

        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(req) == (lease.COPIED, req.gen, 1)
        assert _rows_equal(dst, s.slabs[ROW], 2, slot)
        assert host.lease_entry(req.idx)["lane_state"][0] == 2 and host.slot_info(ROW)[slot][2] == 0
        counters = host.counters()
        assert counters["leases_copied"] == 1 and counters["copy_jobs"] == 1 and counters["copy_lanes"] == 1
        assert counters["lease_double_signal"] == 0 and counters["copy_errors"] == 0
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
        req = sim.post(ROW, [3], dst=[4], copy_engine=True)
        assert host.pump() == 1
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


def test_only_hit_lanes_are_copying_and_the_mask_names_exactly_them(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        dst = _copy_engine(s, host)
        _load(sim, host, [2, 3])
        req = sim.post(ROW, [3, 5, 2], dst=[0, 1, 5], copy_engine=True)
        assert host.pump() == 1
        tags = [sim.row_result(req, lane)["tag"] for lane in range(3)]
        assert tags == [lease.COPYING, lease.LOADING, lease.COPYING]
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(req) == (lease.COPIED, req.gen, 0b101)
        assert _rows_equal(dst, s.slabs[ROW], 0, _slot_of(host, 3)) and _rows_equal(dst, s.slabs[ROW], 5, _slot_of(host, 2))
        entry = host.lease_entry(req.idx)
        assert entry["lane_copy_engine"][:3] == [1, 0, 1] and entry["lane_state"][:3] == [2, 1, 2]
    finally:
        host.stop()


@pytest.mark.parametrize("case", ["flag_off", "unarmed", "no_slot", "slot_past_table"])
def test_a_lane_the_copy_engine_cannot_take_is_published_ready(tmp_path, case):
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host, arm=case != "unarmed")
        _load(sim, host, [3])
        dst = {"no_slot": [-1], "slot_past_table": [DST_ROWS]}.get(case, [1])
        req = sim.post(ROW, [3], dst=dst, copy_engine=case != "flag_off")
        assert host.pump() == 1
        assert sim.row_result(req, 0)["tag"] == lease.READY
        counters = host.counters()
        assert counters["copy_jobs"] == 0 and host.copy_engine_marked() == 0
        assert counters["copy_fallbacks"] == (1 if case in ("no_slot", "slot_past_table") else 0)
    finally:
        host.stop()


@pytest.mark.parametrize("fault", ["issue", "query"])
def test_a_copy_whose_completion_cannot_be_established_fails_stop_and_keeps_its_lease(tmp_path, fault):
    s, page, host, sim = _host(tmp_path)
    try:
        _copy_engine(s, host)
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        host.copy_engine_fail(issue=fault == "issue", query=fault == "query")
        req = sim.post(ROW, [3], dst=[1], copy_engine=True)
        assert host.pump() == 1
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        assert page_word(page, "fatal") != 0
        assert sim.copy_done(req)[:2] != (lease.COPIED, req.gen)
        assert host.slot_info(ROW)[slot][2] == 1 and host.lease_entry(req.idx)["lane_state"][0] == 1
        assert host.counters()["copy_errors"] >= 1
    finally:
        host.copy_engine_fail()
        host.stop()


def test_the_copy_engine_is_refused_without_piece_streaming_and_before_it_is_enabled(tmp_path):
    s, page, host, sim = _host(tmp_path, piece_stream=False)
    try:
        with pytest.raises(Exception, match="piece streaming"):
            host.enable_copy_engine(-1)
        with pytest.raises(Exception, match="not enabled"):
            host.arm_copy_engine()
    finally:
        host.stop()


def test_the_lane_request_carries_the_destination_slots_and_the_flag(tmp_path):
    """The layout both kernels and the service read: dst_slot[] and flags after expert[] (ABI 3)."""
    s, page, host, sim = _host(tmp_path)
    try:
        req = sim.post(ROW, [3, 4], dst=[7, 9], copy_engine=True)
        base = sim.layout.d_offset + lease.LANE_REQUEST + req.idx * lease.LANE_REQUEST_BYTES
        f = lease.LANE_REQUEST_FIELDS
        assert sim._i32(base + f["dst_slot"], lease.LANES).tolist() == [7, 9] + [-1] * (lease.LANES - 2)
        assert int(sim._i32(base + f["flags"])[0]) == lease.LANE_REQUEST_FLAG_COPY_ENGINE
        assert host.lease_header()["abi_version"] == lease.ABI_VERSION == 3
        assert host.lease_header()["copy_offset"] == host.lease_layout.copy_offset
    finally:
        host.stop()
