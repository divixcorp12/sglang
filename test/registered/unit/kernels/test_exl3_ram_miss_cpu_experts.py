"""CPU experts' host half (CPU; plan 2026-09-29-dsv41-cpu-experts, "Step B"): the grant's CPU lanes and their
completion.

The service's grant tags the last ``split[n]`` of a copy-engine request's n resident lanes CPU (RowResult tag 4) when
the post carried CPU input and the row is registered; the device plan sorts miss lanes highest-scored first. The copy
thread hands those lanes to the CPU expert thread and copies only the rest; CopyDone carries the whole mask once both
are done. The forward here is a ctypes fake that records its calls
and writes a known partial sum.

The fake runs on the CPU expert thread and needs the GIL, which the host's blocking calls hold, so every wait for the
CPU lanes polls from Python (``_wait``) before any blocking call such as ``copy_engine_idle``.
"""

import ctypes
import os
import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
ROWS = 2
DST_ROWS = 6
HIDDEN = 8
HANDLE = 7
NO_SPLIT = [0] * (lease.LANES + 1)

_FORWARD = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
)


class FakeForward:
    """out[j] = sum_i weights[i] * (slots[i] + 1) + j; returns ``result``."""

    def __init__(self, result: int = 0):
        self.result = result
        self.calls = []
        self.c = _FORWARD(self._run)  # kept alive for as long as the host may call it

    def _run(self, layer, x, slots, weights, k, out, threads):
        s = [slots[i] for i in range(k)]
        w = [weights[i] for i in range(k)]
        self.calls.append((layer, s, w, threads))
        if self.result == 0:
            total = sum(wi * (si + 1) for si, wi in zip(s, w))
            for j in range(HIDDEN):
                out[j] = total + j
        return self.result

    @property
    def address(self) -> int:
        return ctypes.cast(self.c, ctypes.c_void_p).value


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _cores() -> list[int]:
    return sorted(os.sched_getaffinity(0))[:2]


def _host(tmp_path, *, split, forward, copy_engine=True):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((ROWS, 6), -1, dtype=torch.int32))
    dst = None
    if copy_engine:
        host.enable_copy_engine(-1, spin_us=200)
        dst = {name: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for name, slab in s.slabs[ROW].items()}
        table = torch.tensor(
            [[slab.data_ptr(), dst[name].data_ptr(), slab[0].numel() * slab.element_size()] for name, slab in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, DST_ROWS)
        host.arm_copy_engine()
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, HIDDEN), dtype=torch.float32)
    if copy_engine:
        host.enable_cpu_experts(forward.address, split, _cores(), x_rows, out_rows, threads=2, spin_us=200)
    return s, page, host, LeaseSim(host, page, s.slabs), dst, out_rows


def _load(sim, host, experts):
    """Make ``experts`` resident in row ROW through a plain request, and retire its leases."""
    req = sim.post(ROW, experts)
    assert host.pump() == 1 and sim.wait(req).served
    sim.done(req)
    host.pump()


def _slot_of(host, expert):
    return host.mapping(ROW)[expert]


def _tags(sim, req):
    return [sim.row_result(req, lane)["tag"] for lane in range(len(req.lanes))]


def _split(**entries):
    split = list(NO_SPLIT)
    for n, k in entries.items():
        split[int(n[1:])] = k
    return split


def test_the_grant_sends_split_n_resident_lanes_to_the_cpu_and_copydone_waits_for_both(tmp_path):
    """Two resident lanes and a miss; split[2] = 1: the tail lane 2 goes to the CPU, lane 0 is copied. Mutants: publish
    CopyDone after the DMA alone -- red on the CopyDone check made before the forward ran; copy a CPU lane too -- red on
    dst row 5; drop the lane's weight -- red on the forward's weights; take the head lane -- red on the tags."""
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n2=1), forward=forward)
    try:
        _load(sim, host, [2, 3])
        host.set_cpu_layer(ROW, HANDLE)
        slot3, slot2 = _slot_of(host, 3), _slot_of(host, 2)
        req = sim.post(ROW, [3, 5, 2], dst=[0, 1, 5], captured=True, cpu_weights=[0.5, 0.25, 0.125])
        assert host.pump() == 1
        assert _tags(sim, req) == [lease.COPYING, lease.LOADING, lease.CPU]
        entry = host.lease_entry(req.idx)
        assert entry["lane_copy_engine"][:3] == [1, 0, 1], "a CPU lane's lease is the copy engine's to release"

        assert _wait(lambda: len(forward.calls) == 1)
        assert forward.calls[0] == (HANDLE, [slot2], [0.125], 2)
        assert torch.equal(out_rows[ROW], torch.arange(HIDDEN, dtype=torch.float32) + 0.125 * (slot2 + 1))
        assert not out_rows[1 - ROW].any()
        time.sleep(0.05)
        assert sim.copy_done(req) != req.gen, "CopyDone before the lane-0 copy completed"

        host.copy_engine_release(-1)
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(req) == req.gen
        assert not any(dst[n][5].view(torch.uint8).any() for n in dst), "the CPU lane's slot was copied"
        assert all(torch.equal(dst[n][0].view(torch.uint8), s.slabs[ROW][n][slot3].view(torch.uint8)) for n in dst)
        host.pump()  # the owner drains the completion and releases the job's leases
        entry = host.lease_entry(req.idx)
        assert entry["lane_state"][0] == 2 and entry["lane_state"][2] == 2
        assert host.slot_info(ROW)[slot2][2] == 0, "the CPU lane's lease outlived its completion"
        counters = host.counters()
        assert counters["cpu_jobs"] == 1 and counters["cpu_lanes"] == 1
        assert host.cpu_stats()["jobs"] == 1 and host.cpu_stats()["lanes"] == 1
    finally:
        host.stop()


def test_a_job_of_cpu_lanes_only_completes_without_any_copy(tmp_path):
    """No DMA lane: the job has no copy to query and completes on the CPU alone (no mark is ever released)."""
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n1=1, n2=2), forward=forward)
    try:
        _load(sim, host, [1, 3])
        host.set_cpu_layer(ROW, HANDLE)
        req = sim.post(ROW, [3, 1], dst=[2, 4], captured=True, cpu_weights=[1.0, 2.0])
        assert host.pump() == 1
        assert _tags(sim, req) == [lease.CPU, lease.CPU]
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert sim.copy_done(req) == req.gen
        assert forward.calls == [(HANDLE, [_slot_of(host, 3), _slot_of(host, 1)], [1.0, 2.0], 2)]
        assert host.copy_engine_marked() == 0
        assert host.copy_engine_idle(5.0)
        assert all(not dst[n].view(torch.uint8).any() for n in dst)
    finally:
        host.stop()


@pytest.mark.parametrize("case", ["unregistered", "split_zero", "uncaptured"])
def test_a_request_the_cpu_may_not_take_is_copied_as_before(tmp_path, case):
    forward = FakeForward()
    split = NO_SPLIT if case == "split_zero" else _split(n1=1)
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=split, forward=forward)
    try:
        _load(sim, host, [3])
        if case != "unregistered":
            host.set_cpu_layer(ROW, HANDLE)
        req = sim.post(ROW, [3], dst=[1], captured=case != "uncaptured", cpu_weights=[1.0])
        assert host.pump() == 1
        expected = lease.READY if case == "uncaptured" else lease.COPYING
        assert _tags(sim, req) == [expected]
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        assert forward.calls == [] and host.counters()["cpu_lanes"] == 0
    finally:
        host.stop()


def test_the_cpu_takes_the_last_job_lanes_past_a_loading_lane(tmp_path):
    """Three resident lanes around a miss, split[3] = 2: the last two job lanes (1 and 3) go to the CPU, the LOADING
    lane 2 is not a job lane and does not count, and lane 0, the plan's highest-scored miss, is copied."""
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n3=2), forward=forward)
    try:
        _load(sim, host, [1, 2, 3])
        host.set_cpu_layer(ROW, HANDLE)
        req = sim.post(ROW, [3, 1, 5, 2], dst=[0, 1, 2, 3], captured=True, cpu_weights=[1.0, 2.0, 3.0, 4.0])
        assert host.pump() == 1
        assert _tags(sim, req) == [lease.COPYING, lease.CPU, lease.LOADING, lease.CPU]
        assert _wait(lambda: len(forward.calls) == 1)
        assert sorted(forward.calls[0][1]) == sorted([_slot_of(host, 1), _slot_of(host, 2)])
        host.copy_engine_release(-1)
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
        assert host.counters()["cpu_lanes"] == 2
    finally:
        host.stop()


def test_the_split_is_retuned_at_run_time_and_checked(tmp_path):
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT, forward=forward)
    try:
        _load(sim, host, [3])
        host.set_cpu_layer(ROW, HANDLE)
        with pytest.raises(Exception, match="registered once"):
            host.set_cpu_layer(ROW, HANDLE)
        with pytest.raises(Exception, match="0 <= split"):
            host.set_cpu_split(_split(n1=2))
        host.set_cpu_split(_split(n1=1))
        req = sim.post(ROW, [3], dst=[1], captured=True, cpu_weights=[1.0])
        assert host.pump() == 1
        assert _tags(sim, req) == [lease.CPU]
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
    finally:
        host.stop()


def test_a_failed_forward_aborts_the_process(tmp_path):
    """A failed CPU forward ends the process in the CPU expert thread, before the lane could be marked done."""
    here = os.path.dirname(os.path.abspath(__file__))
    result = run_host_script(
        tmp_path,
        f"""
        sys.path.insert(0, {here!r})
        from test_exl3_ram_miss_cpu_experts import FakeForward, ROW, ROWS, HIDDEN, HANDLE, DST_ROWS, _cores, _split
        forward = FakeForward(result=-5)
        host.enable_copy_engine(-1, spin_us=200)
        dst = {{n: torch.zeros((DST_ROWS,) + tuple(v.shape[1:]), dtype=v.dtype) for n, v in s.slabs[ROW].items()}}
        table = torch.tensor(
            [[v.data_ptr(), dst[n].data_ptr(), v[0].numel() * v.element_size()] for n, v in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, DST_ROWS)
        host.arm_copy_engine()
        x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
        out_rows = torch.zeros((ROWS, HIDDEN), dtype=torch.float32)
        host.enable_cpu_experts(forward.address, _split(n1=1), _cores(), x_rows, out_rows, threads=2, spin_us=200)
        req = sim.post(ROW, [3])
        assert host.pump() == 1
        sim.done(req)
        host.pump()
        host.set_cpu_layer(ROW, HANDLE)
        sim.post(ROW, [3], dst=[1], captured=True, cpu_weights=[1.0])
        host.pump()
        time.sleep(3.0)
        print("reached")
        """,
        capacity=4,
    )
    assert_aborted(result, f"CPU expert forward of row {ROW} failed (-5)")


def test_cpu_experts_need_the_copy_engine(tmp_path):
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT, forward=forward, copy_engine=False)
    try:
        with pytest.raises(Exception, match="need the copy engine"):
            host.enable_cpu_experts(
                forward.address, NO_SPLIT, _cores(), torch.zeros((ROWS, 16), dtype=torch.uint8),
                torch.zeros((ROWS, HIDDEN)), threads=1,
            )
    finally:
        host.stop()
