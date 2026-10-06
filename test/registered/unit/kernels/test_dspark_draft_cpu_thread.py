"""The draft CPU thread (host/draft_cpu_thread.h): the host half of the DSpark draft channel, on the instr build's fake
CPU expert kernel.

The device half is played by test exports: draft_test_post writes what draft_post_kernel's thread 0 writes (the record
under its seqlock, then the head), and _finish closes the gate with the device's Dekker half (draft_test_finish_close),
spins until the gate opens and checks done[G], as the finish kernel, the stream wait and the commit do.
"""

import dataclasses
import os
import time
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer, spawn_child

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

H, STAGES, VARIANT, LANES = 64, 3, "instr", 8
CAPACITY = 1 << 20  # stage s's layer has capacity CAPACITY + s, which the fake records per call


def _module():
    return ops._host_module("exl3", VARIANT)


def _u32(t: torch.Tensor, off: int) -> int:
    return int(t[off : off + 4].view(torch.int32)[0]) & 0xFFFFFFFF


def _done(areas, seq: int) -> int:
    off = areas.wire.done + 8 * ((seq - 1) % areas.wire.records)
    return int(areas.channel[off : off + 8].view(torch.int64)[0]) & 0xFFFFFFFFFFFFFFFF


def _calls() -> list[dict]:
    m = _module()
    width = 6 + 2 * LANES
    count = int(m.expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
    out = torch.zeros((count, width), dtype=torch.float64)
    m.expert_stream_test_kernel_calls(out)
    return [
        {"core": int(r[0]), "k": int(r[4]), "slots": [int(s) for s in r[5 : 5 + int(r[4])]],
         "capacity": int(r[5 + 2 * LANES])}
        for r in out.tolist()
    ]


def _post(areas, stage, rows, k, seq, epoch=0):
    _module().expert_stream_draft_test_post(areas.channel.data_ptr(), stage, rows, k, seq, epoch)


def _finish(areas, seq, epoch=0, timeout_s=2.0) -> float:
    """The device's finish: close the gate (Dekker), wait for it to open, commit on done[G]. Returns the wait in s."""
    started = time.perf_counter()
    _module().expert_stream_draft_test_finish_close(areas.channel.data_ptr(), seq, epoch)
    opened = lease.gate_word(seq, "open")
    while _u32(areas.channel, areas.wire.gate) != opened:
        if time.perf_counter() - started > timeout_s:
            raise AssertionError(f"the gate never opened for record {seq}")
    waited = time.perf_counter() - started
    assert _done(areas, seq) == (epoch << 32) | seq
    return waited


def _stage(areas, stage, rows, k, seed=0):
    g = torch.Generator().manual_seed(seed + 10 * stage)
    areas.x[stage, :rows] = torch.randn(rows, H, generator=g).half()
    for t in range(rows):
        areas.slots[stage, t, :k] = torch.tensor([t, -1, 2] + list(range(3, k)))[:k]
        areas.weights[stage, t, :k] = torch.rand(k, generator=g)


def _want(areas, stage, t, k):
    s = areas.slots[stage, t, :k].double()
    w = areas.weights[stage, t, :k].double()
    return (torch.arange(H, dtype=torch.float64) + float((w * (s + 1)).sum())).float()


def _cores():
    return sorted(os.sched_getaffinity(0))[:2]


def _host(request, *, spin_us=-1, keep_warm_us=0, fatal_wait_s=30.0, ns_per_expert=0):
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas, DraftCpuHost

    areas = DraftCpuAreas(STAGES, H, pin=False)
    kernel = int(_module().expert_stream_test_kernel_address(ns_per_expert, 0, 0))
    host = DraftCpuHost(areas, cores=_cores(), threads=2, spin_us=spin_us, keep_warm_us=keep_warm_us,
                        fatal_wait_s=fatal_wait_s, variant=VARIANT)
    for stage in range(STAGES):
        host.set_layer(stage, kernel, dataclasses.replace(fake_cpu_layer(H), capacity=CAPACITY + stage))
    host.start()
    request.addfinalizer(host.stop)  # the fake's counters are per process: no thread may outlive its test
    return areas, host


def _draft_cpu_s() -> float:
    """The draft CPU thread's user + system time, from /proc."""
    for task in Path("/proc/self/task").iterdir():
        if (task / "comm").read_text().strip() == "dspark-cpu":
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    raise AssertionError("no draft CPU thread")


def _until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def test_one_stage_serves_m_rows(request):
    areas, host = _host(request)
    rows, k = 5, 3
    _stage(areas, 1, rows, k)
    _post(areas, 1, rows, k, seq=1)
    _finish(areas, 1)
    for t in range(rows):
        assert torch.equal(areas.out[1, t], _want(areas, 1, t, k)), t
    calls = _calls()
    assert len(calls) == rows
    assert all(c["core"] == _cores()[0] and c["capacity"] == CAPACITY + 1 for c in calls)
    assert [c["slots"] for c in calls] == [[t, -1, 2] for t in range(rows)]
    stats = host.stats()
    assert (stats["jobs"], stats["rows"]) == (1, rows)


def test_stats_count_the_jobs_whose_routes_share_a_slot(request):
    """The EXL3 kernel groups a call's live routes by slot (up to CHUNK_M tokens per chunk), so a job whose routes name
    one slot twice runs a chunk of several tokens: collided_jobs counts those jobs, shared_routes the routes past each slot's first, and
    collided_forward_ns their forward time. -1 is not a route."""
    areas, host = _host(request, ns_per_expert=1000)
    k = 3
    jobs = [
        [[0, 1, 2], [3, 4, 5]],  # every slot once
        [[0, 1, 2], [2, -1, 5]],  # slot 2 twice
        [[7, 7, -1]],  # one token routed twice to slot 7
    ]
    for seq, slots in enumerate(jobs, start=1):
        rows = len(slots)
        _stage(areas, 0, rows, k, seed=seq)
        areas.slots[0, :rows, :k] = torch.tensor(slots, dtype=areas.slots.dtype)
        _post(areas, 0, rows, k, seq=seq)
        _finish(areas, seq)
    stats = host.stats()
    assert (stats["jobs"], stats["collided_jobs"], stats["shared_routes"]) == (3, 2, 2)
    assert 0 < stats["collided_forward_ns"] < stats["forward_ns"]


def test_three_stages_in_turn_each_run_their_own_layer(request):
    areas, host = _host(request)
    for seq, stage in enumerate(range(STAGES), start=1):
        _stage(areas, stage, 2, 4, seed=seq)
        _post(areas, stage, 2, 4, seq=seq, epoch=7)
        _finish(areas, seq, epoch=7)
        for t in range(2):
            assert torch.equal(areas.out[stage, t], _want(areas, stage, t, 4)), (stage, t)
    assert [c["capacity"] for c in _calls()] == [CAPACITY + s for s in range(STAGES) for _ in range(2)]
    assert host.stats()["jobs"] == STAGES


def test_the_head_store_ends_the_hold(request):
    areas, host = _host(request, spin_us=-1, keep_warm_us=500_000)
    _stage(areas, 0, 1, 2)
    _post(areas, 0, 1, 2, seq=1)
    _finish(areas, 1)
    m = _module()
    _until(lambda: int(m.expert_stream_test_keep_warm_calls()) >= 1)
    before = int(m.expert_stream_test_keep_warm_calls())
    _post(areas, 0, 1, 2, seq=2)
    assert _finish(areas, 2) < 0.005
    _until(lambda: int(m.expert_stream_test_keep_warm_calls()) == before + 1)
    time.sleep(0.05)
    assert int(m.expert_stream_test_keep_warm_calls()) == before + 1


def test_an_idle_thread_sleeps_and_still_serves(request):
    areas, host = _host(request, spin_us=20_000, keep_warm_us=0)
    _stage(areas, 0, 1, 2)
    _post(areas, 0, 1, 2, seq=1)
    _finish(areas, 1)
    time.sleep(0.1)  # past the hold and the spin budget
    before = _draft_cpu_s()
    time.sleep(0.3)
    assert _draft_cpu_s() - before < 0.05
    _post(areas, 0, 1, 2, seq=2)
    assert _finish(areas, 2) < 0.02


def test_stop_with_a_closed_gate_returns_and_leaves_it_open(request):
    areas, host = _host(request)
    m = _module()
    core = _cores()[0]
    m.expert_stream_test_kernel_hold(core, 1)
    try:
        _stage(areas, 0, 1, 2)
        _post(areas, 0, 1, 2, seq=1)
        m.expert_stream_draft_test_finish_close(areas.channel.data_ptr(), 1, 0)
    finally:
        m.expert_stream_test_kernel_hold(core, 0)
    started = time.monotonic()
    host.stop()
    assert time.monotonic() - started < 5.0
    assert _u32(areas.channel, areas.wire.gate) == lease.gate_word(1, "open")


_CHILD = """
import dataclasses, os, sys, time
import torch
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas, DraftCpuHost
from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer
case = sys.argv[1]
m = ops._host_module("exl3", "instr")
areas = DraftCpuAreas(3, 64, pin=False)
kernel = int(m.expert_stream_test_kernel_address(0, 1 if case == "fail" else 0, 0))
cores = sorted(os.sched_getaffinity(0))[:2]
host = DraftCpuHost(areas, cores=cores, threads=2, spin_us=-1, keep_warm_us=0, fatal_wait_s=0.5, variant="instr")
for stage in range(3):
    host.set_layer(stage, kernel, fake_cpu_layer(64))
host.start()
if case == "hold":
    m.expert_stream_test_kernel_hold(cores[0], 1)
if case == "tear":
    m.expert_stream_draft_test_tear(areas.channel.data_ptr(), 1)
else:
    m.expert_stream_draft_test_post(areas.channel.data_ptr(), 2, 3, 2, 1, 0)
time.sleep(2)
print("late", flush=True)  # the watchdog's bar: fail-stopped within 2 s of the post
time.sleep(3)
print("survived", flush=True)
"""


@pytest.mark.parametrize(
    "case, says",
    [("fail", ["DSpark draft CPU experts", "record 1", "failed"]), ("tear", ["DSpark draft CPU experts", "torn"]),
     ("hold", ["DSpark draft CPU experts", "record 1", "0.5"])],
)
def test_a_failure_fail_stops(case, says):
    result = spawn_child(_CHILD, case, timeout_s=60, variant=VARIANT)
    assert "late" not in result.stdout, (result.stdout, result.stderr[-2000:])
    assert result.returncode != 0
    for text in says:
        assert text in result.stderr, result.stderr[-2000:]
    if case == "hold":
        assert "incomplete" in next(line for line in result.stderr.splitlines() if "FATAL" in line)
