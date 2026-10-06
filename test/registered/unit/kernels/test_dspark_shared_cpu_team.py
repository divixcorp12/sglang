```python
"""One CPU expert team per node (plan 2026-10-06 Task 11): node 0's engine serves the target's lease records and the
DSpark draft channel on one thread, one job at a time, on the instr build's fake kernel (CPU)."""

import dataclasses
import os
import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import draft_cpu_host, fake_cpu_layer, spawn_child

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

H, STAGES, ROW, CAPACITY = 64, 3, 1, 1 << 20


def _m():
    return ops._host_module("exl3", "instr")


def _cores():
    return sorted(os.sched_getaffinity(0))[:2]


def _until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)


def _shared(tmp_path, request, fatal_wait_s=30.0):
    areas = DraftCpuAreas(STAGES, H, pin=False)
    kernel = int(_m().expert_stream_test_kernel_address(0, 0, 0))
    draft = draft_cpu_host("shared", areas, kernel, cores=_cores(), threads=2, spin_us=-1, keep_warm_us=0,
                           fatal_wait_s=fatal_wait_s, tmp_path=tmp_path)
    for stage in range(STAGES):
        draft.set_layer(stage, kernel, dataclasses.replace(fake_cpu_layer(H), capacity=CAPACITY + stage))
    draft.start()
    request.addfinalizer(draft.expert_host.stop)
    request.addfinalizer(draft.stop)
    return areas, draft


def _post_draft(areas, seq, stage=0, rows=2, k=2):
    areas.slots[stage, :rows, :k] = torch.tensor([[0, 1]] * rows, dtype=torch.int32)
    areas.weights[stage, :rows, :k] = 0.5
    _m().expert_stream_draft_test_post(areas.channel.data_ptr(), stage, rows, k, seq, 0)


def _draft_done(areas, seq):
    off = areas.wire.done + 8 * ((seq - 1) % areas.wire.records)
    return (int(areas.channel[off : off + 8].view(torch.int64)[0]) & 0xFFFFFFFF) == seq


def _post_target(draft):
    """A captured CPU-hit lane of row ROW on the same team: expert 2 made resident, then posted as a CPU lane."""
    sim, host = draft.sim, draft.expert_host
    req = sim.post(ROW, [2])
    assert host.pump() == 1 and sim.wait_served(req)
    req = sim.post(ROW, [2], captured=True, cpu_on=True, dst=[0], weights=[1.0])
    assert host.pump() == 1
    return req


def _kinds():
    """The fake's calls in order: 'draft' (a stage's layer, capacity >= CAPACITY) or 'target'."""
    width = 6 + 2 * 8
    count = int(_m().expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
    out = torch.zeros((count, width), dtype=torch.float64)
    _m().expert_stream_test_kernel_calls(out)
    return ["draft" if int(r[5 + 2 * 8]) >= CAPACITY else "target" for r in out.tolist()]


@pytest.mark.parametrize("first", ["draft", "target"])
def test_one_team_runs_a_job_of_each_kind_one_after_the_other(tmp_path, request, first):
    """(a) The first job is held in its forward; the second kind arrives meanwhile and runs only once the first
    completes, on the same team (core 0 is both jobs' worker 0). Mutation: serve the draft from a second thread -- the
    second job's calls interleave with or precede the held one's."""
    areas, draft = _shared(tmp_path, request)
    core = _cores()[0]
    _m().expert_stream_test_kernel_hold(core, 1)
    if first == "draft":
        _post_draft(areas, 1)
        time.sleep(0.05)
        _post_target(draft)
    else:
        _post_target(draft)
        time.sleep(0.05)
        _post_draft(areas, 1)
    time.sleep(0.1)
    assert _kinds() == [], "nothing completes while the first job is held"
    _m().expert_stream_test_kernel_hold(core, 0)
    _until(lambda: _draft_done(areas, 1) and draft.expert_host.cpu_stats()["jobs"] == 1)
    second = "target" if first == "draft" else "draft"
    calls = _kinds()
    assert calls == [first] * calls.count(first) + [second] * calls.count(second), calls
    assert calls.count("draft") == 2 and calls.count("target") == 1  # the draft record's 2 rows, the target's 1


_STUCK_CHILD = """
import os, sys, time, tempfile, dataclasses, pathlib
import torch
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas
from sglang.test.dsv41_ram_miss_fixtures import draft_cpu_host, fake_cpu_layer
kind, when = sys.argv[1], sys.argv[2]   # kind: draft | target; when: run (left stuck) | stop (stop() during it)
m = ops._host_module("exl3", "instr")
slow = when == "slow"
kernel = int(m.expert_stream_test_kernel_address(200_000_000 if slow else 0, 0, 0))
areas = DraftCpuAreas(3, 64, pin=False)
cores = sorted(os.sched_getaffinity(0))[:2]
draft = draft_cpu_host("shared", areas, kernel, cores=cores, threads=2, spin_us=-1, keep_warm_us=0,
                       fatal_wait_s=5.0 if slow else 0.5, tmp_path=pathlib.Path(tempfile.mkdtemp(dir=os.getcwd())))
for stage in range(3):
    draft.set_layer(stage, kernel, dataclasses.replace(fake_cpu_layer(64), capacity=(1 << 20) + stage))
draft.start()
host, sim = draft.expert_host, draft.sim
req = sim.post(1, [2]); assert host.pump() == 1 and sim.wait_served(req)
if not slow:
    m.expert_stream_test_kernel_hold(cores[0], 1)
if kind == "draft":
    areas.slots[0, :2, :2] = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    m.expert_stream_draft_test_post(areas.channel.data_ptr(), 0, 2, 2, 1, 0)
else:
    sim.post(1, [2], captured=True, cpu_on=True, dst=[0], weights=[1.0]); assert host.pump() == 1
time.sleep(0.1)
if when in ("stop", "slow"):
    host.stop()   # the engine's stop() with a job of `kind` in its forward
    print("stopped", flush=True)
    if slow:
        if kind == "draft":
            done = int(areas.channel[areas.wire.done : areas.wire.done + 8].view(torch.int64)[0]) & 0xFFFFFFFF
            print("draft done" if done == 1 else "draft not done", flush=True)
        else:
            width = 6 + 2 * 8
            count = int(m.expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
            calls = torch.zeros((count, width), dtype=torch.float64)
            m.expert_stream_test_kernel_calls(calls)
            target = [r for r in calls.tolist() if int(r[5 + 2 * 8]) < (1 << 20)]
            print("target done" if target else "target not done", flush=True)
time.sleep(2)
print("late", flush=True)
"""


@pytest.mark.parametrize("kind", ["draft", "target"])
@pytest.mark.parametrize("when", ["run", "stop"])
def test_a_stuck_job_of_either_kind_fail_stops(kind, when):
    """(b) A held forward of either kind fail-stops within the fatal wait (0.5 s), with the job's identity. (c, hung) A
    stop() during it still fail-stops: the watchdog outlives the run thread's join."""
    result = spawn_child(_STUCK_CHILD, kind, when, timeout_s=60, variant="instr")
    assert "late" not in result.stdout, (result.stdout, result.stderr[-2000:])
    assert result.returncode != 0
    says = ["DSpark draft CPU experts", "record 1", "incomplete"] if kind == "draft" else ["CPU job of row 1", "incomplete"]
    for text in says:
        assert text in result.stderr, result.stderr[-2000:]


@pytest.mark.parametrize("kind", ["draft", "target"])
def test_stop_during_a_job_of_either_kind_lets_it_complete(kind):
    """(c) stop() while a 400 ms forward of either kind runs: stop returns after it completes, the job is done (the
    draft record's done word, or the target's CPU job count), nothing fail-stops."""
    result = spawn_child(_STUCK_CHILD, kind, "slow", timeout_s=60, variant="instr")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "stopped" in result.stdout and f"{kind} done" in result.stdout, result.stdout
    assert "FATAL" not in result.stderr, result.stderr[-2000:]


def test_a_launch_without_a_draft_source_holds_on_the_doorbell_alone(tmp_path, request):
    """The non-DSpark engine is today's: no watchdog thread, and its holds call the one-word keep_warm (the fake counts
    which). Mutation: always hold on both words -- the two-word count moves."""
    from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

    s = ram_miss_setup(tmp_path, capacity=7, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, ops.new_page(pin=False, wire=ops.expert_lease_block.wire_layout(8)), k=3)
    request.addfinalizer(host.stop)
    host.enable_copy_engine(-1)
    kernel = int(_m().expert_stream_test_kernel_address(0, 0, 0))
    threads_before = len(os.listdir("/proc/self/task"))
    host.enable_cpu_experts(kernel, [0] * (host.wire.lanes + 1), _cores(), torch.zeros((2, 16), dtype=torch.uint8),
                            torch.zeros((2, 2, 8), dtype=torch.float32), threads=2, spin_us=-1, keep_warm_us=0)
    _until(lambda: int(_m().expert_stream_test_keep_warm_calls()) >= 1)
    assert int(_m().expert_stream_test_keep_warm_either_calls()) == 0
    assert len(os.listdir("/proc/self/task")) - threads_before == 1, "the engine thread only, no watchdog"
