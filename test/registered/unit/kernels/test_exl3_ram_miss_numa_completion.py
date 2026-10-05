"""The completion side with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3 "Combiner" and
"Errors"; Testing 3): each group's CPU lanes run on its own cores into its own output parts, the copy thread stores
CopyDone only when every group with a host lane is done, and a stalled group is named by the copy-wait abort.

The kernel is the instr build's fake CpuExpertKernel (ExpertStreamHost.test_kernel_address): one per tier, so a call's
group is told by the first core it was given, and test_kernel_hold stalls one group's forward."""

import os
import textwrap
import time

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import (
    assert_aborted,
    fake_cpu_layer,
    ram_miss_setup,
    spawn_child,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROW, ROWS, HIDDEN, DST_ROWS, EXPERTS = 1, 2, 8, 6, 8
HALVES = [[(0, 10)] * 2, [(10, 20)] * 2]
ONE_OF_TWO = [0, 0, 1] + [0] * 6  # of n = 2 eligible lanes, 1 on the CPU
NONE = [0] * 9


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _cores():
    return sorted(os.sched_getaffinity(0))


def _host(tmp_path, *, split, arm=True, ns_per_expert=0):
    s = ram_miss_setup(tmp_path, capacity=20, experts=EXPERTS, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr",
        node_ranges=HALVES,
    )
    host.reserve_staging(2)
    host.enable_copy_engine(-1)
    host.copy_engine_release(-1)  # the CPU test backend completes only released marks
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    if arm:
        host.arm_copy_engine()
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 4, HIDDEN), dtype=torch.float32)
    cores = _cores()
    kernel = host.test_kernel_address(ns_per_expert)  # one kernel serves both groups
    for group in range(2):
        host.enable_cpu_experts(
            kernel, split[group], cores[2 * group : 2 * group + 2], x_rows, out_rows, threads=2,
            group=group,
        )
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    sim = ChainSim(host, page, s.slabs)
    sim.dst = dst  # the copy thread writes into these: they must outlive the test body
    req = sim.post(ROW, [0, 1, 2, 3])  # make all four resident: two misses per node
    assert host.pump() == 1 and sim.wait_served(req)
    return s, host, sim, out_rows


def _cpu_post(sim):
    return sim.post(ROW, [0, 1, 2, 3], captured=True, cpu_on=True, dst=[0, 1, 2, 3])


def test_each_groups_cpu_lanes_go_to_its_own_cores_and_part(tmp_path):
    cores = _cores()
    s, host, sim, out_rows = _host(tmp_path, split=[ONE_OF_TWO, ONE_OF_TWO])
    try:
        req = _cpu_post(sim)
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 2)
        assert sim.copy_wait(req)
        slot2, slot3 = host.mapping(ROW)[2], host.mapping(ROW)[3]
        assert 0 <= slot2 < 10 <= slot3 < 20
        calls = host.test_kernel_calls()
        assert sorted((c["core"], c["slots"]) for c in calls) == [(cores[0], [slot2]), (cores[2], [slot3])]
        assert out_rows[ROW, 0].tolist() == [j + slot2 + 1 for j in range(HIDDEN)], "group 0's hits: part 0"
        assert out_rows[ROW, 2].tolist() == [j + slot3 + 1 for j in range(HIDDEN)], "group 1's hits: part 2"
        assert not out_rows[ROW, 1].any() and not out_rows[ROW, 3].any()
        assert host.cpu_stats(group=0)["jobs"] == 1 and host.cpu_stats(group=1)["jobs"] == 1
    finally:
        host.stop()


def test_copydone_waits_for_every_groups_cpu_engine(tmp_path):
    cores = _cores()
    s, host, sim, out_rows = _host(tmp_path, split=[ONE_OF_TWO, ONE_OF_TWO])
    try:
        host.test_kernel_hold(cores[2])
        req = _cpu_post(sim)
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        time.sleep(0.05)
        assert sim.copy_done(req) != req.gen, "CopyDone before group 1's CPU job finished"
        host.test_kernel_hold(cores[2], False)
        assert sim.copy_wait(req)
    finally:
        host.test_kernel_hold(cores[2], False)
        host.stop()


def test_a_zero_split_keeps_a_node_off_the_cpu_and_copydone_off_its_engine(tmp_path):
    cores = _cores()
    s, host, sim, out_rows = _host(tmp_path, split=[ONE_OF_TWO, NONE])
    try:
        host.test_kernel_hold(cores[2])
        req = _cpu_post(sim)
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.HIT_COPY]
        assert host.pump() == 1
        assert sim.copy_wait(req), "group 1 has no CPU lane: CopyDone must not wait on its engine"
        assert [c["core"] for c in host.test_kernel_calls()] == [cores[0]], "group 1 ran no CPU job"
    finally:
        host.test_kernel_hold(cores[2], False)
        host.stop()


def test_a_groups_only_host_lane_being_a_cpu_miss_is_waited_for_and_completes_once(tmp_path):
    """Group 1's only host lane is a CPU miss (the late_cpu / late_seq path): its part carries no copy and no hit job,
    yet CopyDone waits for group 1's engine, and is stored once. Mutants: drop group 1's bit from CopyJob::groups --
    red on the early CopyDone; skip the late-CPU wait in the copy thread -- red likewise."""
    cores = _cores()
    s, host, sim, out_rows = _host(tmp_path, split=[NONE, [0, 1] + [0] * 7])
    try:
        host.test_kernel_hold(cores[2])
        req = sim.post(ROW, [0, 5], captured=True, cpu_on=True, cpu_misses=True, dst=[0, 1], weights=[1.0, 0.5])
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.MISS_CPU]
        assert host.pump() == 1
        assert _wait(lambda: host.counters()["copy_jobs"] == 2)  # one part per group
        time.sleep(0.05)
        assert sim.copy_done(req) != req.gen, "CopyDone before group 1's CPU miss finished"
        host.test_kernel_hold(cores[2], False)
        assert sim.copy_wait(req)
        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        assert [c["core"] for c in host.test_kernel_calls()] == [cores[2]]
        time.sleep(0.05)
        assert sim.copy_done(req) == req.gen
        assert host.counters()["cpu_jobs"] == 1
    finally:
        host.test_kernel_hold(cores[2], False)
        host.stop()


def test_each_groups_split_is_its_own_node_table(tmp_path):
    s, host, sim, out_rows = _host(tmp_path, split=[ONE_OF_TWO, NONE])
    try:
        host.set_cpu_split([0, 1] + [0] * 7, group=1)
        assert sim.split() == ONE_OF_TWO + [0, 1] + [0] * 7
    finally:
        host.stop()


def test_calibration_runs_on_its_groups_engine(tmp_path):
    # the fake kernel's spin keeps a blocking call (calibration) from deadlocking on a Python callback
    s, host, sim, out_rows = _host(tmp_path, split=[NONE, NONE], arm=False, ns_per_expert=1000)
    try:
        scratch = torch.empty(8 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
        grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, group=1)
        assert grid.numel() > 0
        assert host.cpu_stats(group=1)["jobs"] > 0 and host.cpu_stats(group=0)["jobs"] == 0
    finally:
        host.stop()


_SCRIPT = """
import os, pathlib, sys, time, torch
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer, ram_miss_setup
ROW = 1
s = ram_miss_setup(pathlib.Path(sys.argv[1]), capacity=20, experts=8, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
page = new_page(pin=False, wire=wire_layout(8, 2))
host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32), variant="instr",
                        node_ranges=[[(0, 10)] * 2, [(10, 20)] * 2])
host.reserve_staging(2)
host.enable_copy_engine(-1, wait_timeout_ms=200)
host.copy_engine_release(-1)
dst = {n: torch.zeros((6,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
host.set_copy_table(ROW, torch.tensor([[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()]
                                       for n, t in s.slabs[ROW].items()], dtype=torch.int64), 6)
host.arm_copy_engine()
x_rows, out_rows = torch.zeros((2, 16), dtype=torch.uint8), torch.zeros((2, 4, 8), dtype=torch.float32)
cores = sorted(os.sched_getaffinity(0))
split = [0, 0, 1] + [0] * 6
kernel = host.test_kernel_address(0)
host.enable_cpu_experts(kernel, split, cores[0:2], x_rows, out_rows, threads=2, group=0)
host.enable_cpu_experts(kernel, split, cores[2:4], x_rows, out_rows, threads=2, group=1)
host.set_cpu_layer(ROW, fake_cpu_layer(8))
sim = ChainSim(host, page, s.slabs)
req = sim.post(ROW, [0, 1, 2, 3])
assert host.pump() == 1 and sim.wait_served(req)
"""


CHILD_VARIANT, CHILD_NODES = "instr", 2  # what _SCRIPT constructs; test_the_child_script_builds_what_the_parent_warms pins it


def test_the_child_script_builds_what_the_parent_warms():
    assert f'variant="{CHILD_VARIANT}"' in _SCRIPT and f"wire_layout(8, {CHILD_NODES})" in _SCRIPT


def _run(tmp_path, body):
    return spawn_child(
        textwrap.dedent(_SCRIPT) + textwrap.dedent(body), tmp_path, timeout_s=120, variant=CHILD_VARIANT, nodes=CHILD_NODES
    )


def test_a_group_whose_cpu_job_stalls_is_named_by_the_copy_wait_abort(tmp_path):
    """Review Focus 2: group 0's CPU lane completes, group 1's forward is held and never returns. CopyDone is never stored, and
    the 200 ms copy-wait abort names group 1, not group 0."""
    result = _run(tmp_path, """
        host.test_kernel_hold(cores[2])
        req = sim.post(ROW, [0, 1, 2, 3], captured=True, cpu_on=True, dst=[0, 1, 2, 3])
        assert host.pump() == 1
        host.start_thread(fatal_wait_s=60.0)
        sim.close_copy_gate(req)
        time.sleep(3.0)
        print("reached")
    """)
    assert_aborted(result, "a copy wait held the decode stream")
    assert "group 1: its CPU job" in result.stderr, result.stderr[-2000:]


def test_a_group_that_never_sends_its_part_is_named_by_the_copy_wait_abort(tmp_path):
    """Review Focus 2: both nodes have a copy-engine lane; group 1's thread stalls before it reads the record, so its
    part never reaches the copy thread. Group 0's part alone must not store CopyDone."""
    result = _run(tmp_path, """
        host.start_thread(fatal_wait_s=60.0)
        host.inject_group_stall(1, 10.0)
        req = sim.post(ROW, [0, 1], captured=True, dst=[0, 1])
        sim.close_copy_gate(req)
        time.sleep(3.0)
        print("reached")
    """)
    assert_aborted(result, "a copy wait held the decode stream")
    assert "group 1: sent no part" in result.stderr, result.stderr[-2000:]
