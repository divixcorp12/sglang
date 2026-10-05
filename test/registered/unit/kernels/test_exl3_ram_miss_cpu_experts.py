"""CPU experts' host half (CPU; plan 2026-09-29-dsv41-cpu-experts, "Step B"; slot-map plan, R1-1).

Covers the CPU lanes the device typed and their completion.
The device types the last ``split[n]`` of a captured post's n eligible lanes CPU: RAM hits, and NVMe misses with
CPU-computed misses on (ram_slot_map.type_lanes; ChainSim plays it here).
The service hands the CPU hits to the CPU expert thread as part 0 before the copy job, and the copy thread copies only
the copy-engine hits; the service sends each CPU miss into part 1 once its row landed, adding into the part after the
record's first miss job.
CopyDone covers every host lane of the record once all of them are done.
The kernel here is the instr build's fake CpuExpertKernel (ExpertStreamHost.test_kernel_address), which records its
calls and writes a known partial sum.

The fake runs natively on the CPU expert thread, so no wait for the CPU lanes depends on the GIL;
``_wait`` still polls from Python before any blocking call such as ``copy_engine_idle``.
"""

import dataclasses
import gc
import os
import time
import weakref

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import (
    assert_aborted,
    attached_host,
    fake_cpu_layer,
    ram_miss_setup,
    run_host_script,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
ROWS = 2
DST_ROWS = 6
HIDDEN = 8
NO_SPLIT = [0] * (lease.wire_layout(8).lanes + 1)


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _cores() -> list[int]:
    return sorted(os.sched_getaffinity(0))[:2]


def _host(tmp_path, *, split, copy_engine=True, parts=2, fail=0):
    # Seven slots, three staging: four mappable rows.
    s = ram_miss_setup(tmp_path, capacity=7, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page, k=3)
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
    out_rows = torch.zeros((ROWS, 2, HIDDEN) if parts == 2 else (ROWS, HIDDEN), dtype=torch.float32)
    if copy_engine:
        host.enable_cpu_experts(host.test_kernel_address(fail=fail), split, _cores(), x_rows, out_rows, threads=2, spin_us=200)
    return s, page, host, ChainSim(host, page, s.slabs), dst, out_rows


def _calls(host):
    """The fake kernel's calls as (slots, weights, threads)."""
    return [(c["slots"], c["weights"], c["threads"]) for c in host.test_kernel_calls()]


def _load(sim, host, experts):
    """Make ``experts`` resident in row ROW through a plain request (each a miss)."""
    req = sim.post(ROW, experts)
    assert host.pump() == 1 and sim.wait_served(req)


def _slot_of(host, expert):
    return host.mapping(ROW)[expert]


def _post(sim, experts, **kw):
    """A captured post with CPU experts on, as the device types it."""
    return sim.post(ROW, experts, captured=True, cpu_on=True, **kw)


def _split(**entries):
    split = list(NO_SPLIT)
    for n, k in entries.items():
        split[int(n[1:])] = k
    return split


def _sum(slots, weights):
    return sum(w * (s + 1) for s, w in zip(slots, weights))


def test_split_n_hit_lanes_go_to_the_cpu_and_copydone_waits_for_both(tmp_path):
    """Two resident lanes and a miss; split[2] = 1: the tail hit (lane 2) goes to the CPU, lane 0 is copied. Mutants:
    publish CopyDone after the DMA alone -- red on the CopyDone check made before the forward ran; copy a CPU lane too
    -- red on dst row 5; drop the lane's weight -- red on the forward's weights."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n2=1))
    try:
        _load(sim, host, [2, 3])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        slot3, slot2 = _slot_of(host, 3), _slot_of(host, 2)
        req = _post(sim, [3, 5, 2], dst=[0, 1, 5], weights=[0.5, 0.25, 0.125])
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU, LaneKind.HIT_CPU]
        assert host.pump() == 1

        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        assert _calls(host)[0] == ([slot2], [0.125], 2)
        assert torch.equal(out_rows[ROW, 0], torch.arange(HIDDEN, dtype=torch.float32) + 0.125 * (slot2 + 1))
        assert not out_rows[ROW, 1].any() and not out_rows[1 - ROW].any()
        time.sleep(0.05)
        assert sim.copy_done(req) != req.gen, "CopyDone before the lane-0 copy completed"

        host.copy_engine_release(-1)
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
        assert not any(dst[n][5].view(torch.uint8).any() for n in dst), "the CPU lane's slot was copied"
        assert all(torch.equal(dst[n][0].view(torch.uint8), s.slabs[ROW][n][slot3].view(torch.uint8)) for n in dst)
        counters = host.counters()
        assert counters["cpu_jobs"] == 1 and counters["cpu_lanes"] == 1
        assert host.cpu_stats()["jobs"] == 1 and host.cpu_stats()["lanes"] == 1
    finally:
        host.stop()


def test_the_groups_cores_reach_every_cpu_forward(tmp_path):
    """The cores enable_cpu_experts takes are the ones every forward of the CPU expert thread carries, so the kernel
    runs its team there. Mutant: leave ForwardCall.cores empty in CpuExpertEngine::run -- red (core -1)."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n1=1))
    try:
        _load(sim, host, [2])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = _post(sim, [2])
        assert req.kinds == [LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        assert sim.copy_wait(req)
        assert [c["core"] for c in host.test_kernel_calls()] == [_cores()[0]]
    finally:
        host.stop()


def test_a_rows_layer_is_made_once_by_the_enabled_kernel(tmp_path):
    """Review Focus 4: a second layer for a row, a layer before any group is enabled, a row past the tier and a
    malformed slab table are refused, naming why."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT, copy_engine=False)
    try:
        with pytest.raises(Exception, match="not enabled"):
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    finally:
        host.stop()
    (tmp_path / "on").mkdir()
    s, page, host, sim, dst, out_rows = _host(tmp_path / "on", split=NO_SPLIT)
    try:
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match="registered once"):
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match=f"CPU expert layer for row {ROWS + 5} of {ROWS}"):
            host.set_cpu_layer(ROWS + 5, fake_cpu_layer(HIDDEN))
        bad = dataclasses.replace(fake_cpu_layer(HIDDEN), slabs=((1, 2),) * 9)
        with pytest.raises(Exception, match="has at most 8 slabs"):
            host.set_cpu_layer(0, bad)
    finally:
        host.stop()


def test_a_test_kernel_layer_keeps_its_slabs_until_it_is_dropped(tmp_path):
    """kernel_layer's layer reads spec.keep's tensors: a caller that passes the spec as a temporary must not free them
    under the layer."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT)
    try:
        slab = torch.zeros(64)
        alive = weakref.ref(slab)
        layer = es.kernel_layer(
            host.test_kernel_address(), dataclasses.replace(fake_cpu_layer(HIDDEN), keep=(slab,)), variant="instr"
        )
        del slab
        gc.collect()
        assert alive() is not None
        es.kernel_drop(layer, variant="instr")
        gc.collect()
        assert alive() is None
    finally:
        host.stop()


def test_a_job_of_cpu_lanes_only_completes_without_any_copy(tmp_path):
    """No DMA lane: the job has no copy to query and completes on the CPU alone (no mark is ever released)."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n1=1, n2=2))
    try:
        _load(sim, host, [1, 3])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = _post(sim, [3, 1], dst=[2, 4], weights=[1.0, 2.0])
        assert req.kinds == [LaneKind.HIT_CPU, LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert _calls(host) == [([_slot_of(host, 3), _slot_of(host, 1)], [1.0, 2.0], 2)]
        assert [c["affinity"] for c in host.test_kernel_calls()] == [_cores()[0]]
        assert host.copy_engine_marked() == 0
        assert host.copy_engine_idle(5.0)
        assert all(not dst[n].view(torch.uint8).any() for n in dst)
    finally:
        host.stop()


@pytest.mark.parametrize("case", ["unregistered", "split_zero", "uncaptured"])
def test_a_request_the_cpu_may_not_take_is_copied_as_before(tmp_path, case):
    """The device types no CPU lane for an unregistered row (cpu_ok), a split of 0, or an eager post."""
    split = NO_SPLIT if case == "split_zero" else _split(n1=1)
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=split)
    try:
        _load(sim, host, [3])
        if case != "unregistered":
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = sim.post(ROW, [3], dst=[1], captured=case != "uncaptured", cpu_on=True, cpu_ok=case != "unregistered",
                       weights=[1.0])
        assert req.kinds == [LaneKind.HIT_SM if case == "uncaptured" else LaneKind.HIT_COPY]
        assert host.pump() == 1
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        assert host.test_kernel_calls() == [] and host.counters()["cpu_lanes"] == 0
    finally:
        host.stop()


def test_the_cpu_takes_the_last_hit_lanes_past_a_miss_lane(tmp_path):
    """Three resident lanes around a miss, split[3] = 2: the last two hit lanes (1 and 3) go to the CPU, the miss lane
    2 is not eligible and does not count, and lane 0, the plan's highest-scored lane, is copied."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n3=2))
    try:
        _load(sim, host, [1, 2, 3])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = _post(sim, [3, 1, 5, 2], dst=[0, 1, 2, 3], weights=[1.0, 2.0, 3.0, 4.0])
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        assert sorted(_calls(host)[0][0]) == sorted([_slot_of(host, 1), _slot_of(host, 2)])
        host.copy_engine_release(-1)
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
        assert host.counters()["cpu_lanes"] == 2
    finally:
        host.stop()


def test_a_cpu_miss_runs_as_part_one_after_its_row_lands_and_copydone_waits_for_it(tmp_path):
    """CPU-computed misses: split[2] = 2 over a hit and a miss puts both on the CPU. The hit is part 0 at record time,
    the miss part 1 once its row landed in its staging slot, and CopyDone waits for both. Mutants: the late job writes
    part 0 -- red on out_rows[ROW, 0]; the copy job completes without its late job -- red on the forward count when
    CopyDone shows."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n2=2))
    try:
        _load(sim, host, [3])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = _post(sim, [3, 5], dst=[0, 1], weights=[0.5, 0.25], cpu_misses=True)
        assert req.kinds == [LaneKind.HIT_CPU, LaneKind.MISS_CPU]
        staging = req.slots[1]
        assert host.pump() == 1
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert len(host.test_kernel_calls()) == 2, "CopyDone before the CPU miss's job ran"
        calls = _calls(host)
        assert calls[0] == ([_slot_of(host, 3)], [0.5], 2)
        assert calls[1] == ([staging], [0.25], 2)
        base = torch.arange(HIDDEN, dtype=torch.float32)
        assert torch.equal(out_rows[ROW, 0], base + _sum([_slot_of(host, 3)], [0.5]))
        assert torch.equal(out_rows[ROW, 1], base + _sum([staging], [0.25]))
        assert host.mapping(ROW)[5] == staging, "a CPU miss is still cached in RAM"
        landed = sim.read_slot(ROW, staging)
        reference = s.reference(ROW, [5])
        assert all(torch.equal(landed[n].view(torch.uint8), reference[n][0].view(torch.uint8)) for n in landed)
        assert host.copy_engine_marked() == 0 and host.copy_engine_idle(5.0)
    finally:
        host.stop()



# One row's pack takes this long under the pack_delay fault, so a record's two miss rows land well apart.
SLOW_PACK_NS = 200_000_000


def _two_cpu_misses(tmp_path, **fault):
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n2=2))
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    if fault:
        host.inject_fault(**fault)
    req = _post(sim, [5, 4], dst=[0, 1], weights=[0.5, 0.25], cpu_misses=True)
    assert req.kinds == [LaneKind.MISS_CPU, LaneKind.MISS_CPU]
    return host, sim, req, out_rows


def test_each_cpu_miss_runs_as_its_row_lands_and_later_jobs_add_into_part_one(tmp_path):
    """Two CPU misses whose rows land apart go to the CPU as two part-1 jobs, the second adding into the part, and
    CopyDone covers both. Mutants: batch the misses until all land -- red on the one-slot calls; the second job
    overwrites -- red on out_rows[ROW, 1]; the first job adds -- red on the accumulate flags."""
    host, sim, req, out_rows = _two_cpu_misses(tmp_path, pack_delay_ns=SLOW_PACK_NS)
    try:
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 2)
        assert [c["accumulate"] for c in host.test_kernel_calls()] == [False, True]
        first, second = _calls(host)
        assert len(first[0]) == 1 and len(second[0]) == 1, "the misses were batched, not sent as each landed"
        assert sorted(first[0] + second[0]) == sorted(req.slots)
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        weights = dict(zip(req.slots, [0.5, 0.25]))
        assert sorted(first[1] + second[1]) == [0.25, 0.5] and all(weights[c[0][0]] == c[1][0] for c in _calls(host))
        expected = torch.arange(HIDDEN, dtype=torch.float32) + _sum(req.slots, [0.5, 0.25])
        assert torch.equal(out_rows[ROW, 1], expected)
        assert not out_rows[ROW, 0].any(), "a CPU miss wrote the hits' part"
        counters = host.counters()
        assert counters["cpu_jobs"] == 2 and counters["cpu_lanes"] == 2
        assert host.copy_engine_idle(5.0)
    finally:
        host.inject_fault()
        host.stop()


def test_cpu_misses_landing_together_or_apart_sum_once_each_into_part_one(tmp_path):
    """Whatever batching the reader's landing order gives, every CPU miss is computed exactly once, only the record's
    first part-1 job overwrites, and CopyDone follows the last job."""
    host, sim, req, out_rows = _two_cpu_misses(tmp_path)
    try:
        assert host.pump() == 1
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        calls = host.test_kernel_calls()
        assert sorted(slot for call in calls for slot in call["slots"]) == sorted(req.slots)
        assert [c["accumulate"] for c in calls] == [False] + [True] * (len(calls) - 1)
        expected = torch.arange(HIDDEN, dtype=torch.float32) + _sum(req.slots, [0.5, 0.25])
        assert torch.equal(out_rows[ROW, 1], expected)
        assert host.copy_engine_idle(5.0)
    finally:
        host.stop()


def test_the_split_is_retuned_at_run_time_and_checked(tmp_path):
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT)
    try:
        _load(sim, host, [3])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match="registered once"):
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match="0 <= split"):
            host.set_cpu_split(_split(n1=2))
        host.set_cpu_split(_split(n1=1))
        assert sim.split() == _split(n1=1), "the device reads the split from the completion block"
        req = _post(sim, [3], dst=[1], weights=[1.0])
        assert req.kinds == [LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        assert host.copy_engine_idle(5.0)
    finally:
        host.stop()


_SCRIPT_HOST = """
        sys.path.insert(0, {here!r})
        from test_exl3_ram_miss_cpu_experts import ROW, ROWS, HIDDEN, DST_ROWS, _cores, _split
        from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer
        from sglang.srt.layers.moe.ram_slot_map import LaneKind
        host.enable_copy_engine(-1, spin_us=200)
        dst = {{n: torch.zeros((DST_ROWS,) + tuple(v.shape[1:]), dtype=v.dtype) for n, v in s.slabs[ROW].items()}}
        table = torch.tensor(
            [[v.data_ptr(), dst[n].data_ptr(), v[0].numel() * v.element_size()] for n, v in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, DST_ROWS)
        host.arm_copy_engine()
        x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
        out_rows = torch.zeros(({out_shape}), dtype=torch.float32)
        host.enable_cpu_experts(host.test_kernel_address(fail={result}), _split(n1=1, n2=2), _cores(), x_rows, out_rows, threads=2, spin_us=200)
        req = sim.post(ROW, [3])
        assert host.pump() == 1
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
"""


def _script(tmp_path, body, *, result=0, parts=2):
    here = os.path.dirname(os.path.abspath(__file__))
    shape = "ROWS, 2, HIDDEN" if parts == 2 else "ROWS, HIDDEN"
    return run_host_script(
        tmp_path, _SCRIPT_HOST.format(here=here, result=result, out_shape=shape) + body, capacity=4, staging=2
    )


def test_a_failed_forward_aborts_the_process(tmp_path):
    """A failed CPU forward ends the process in the CPU expert thread, before the lane could be marked done."""
    result = _script(tmp_path, """
        sim.post(ROW, [3], dst=[1], captured=True, cpu_on=True, weights=[1.0])
        host.pump()
        time.sleep(3.0)
        print("reached")
    """, result=-5)
    assert_aborted(result, f"CPU expert forward of row {ROW} failed: fake CPU expert forward failed (-5)")


def test_a_cpu_miss_read_failure_aborts_before_copy_done(tmp_path):
    """Review Focus 4: the NVMe read of a CPU miss fails. The process aborts with FATAL before CopyDone, and the CPU
    never runs on the partial bytes."""
    result = _script(tmp_path, """
        host.inject(fail_reads=True)
        req = sim.post(ROW, [5], dst=[1], captured=True, cpu_on=True, cpu_misses=True, weights=[1.0])
        assert req.kinds == [LaneKind.MISS_CPU]
        host.pump()
        print("reached", sim.copy_done(req) == req.gen, len(host.test_kernel_calls()))
    """)
    assert_aborted(result, "a test fault failed the read")


def test_a_cpu_miss_needs_the_second_output_part(tmp_path):
    """One output part: a CPU miss would overwrite the CPU hits' sum, so the host refuses the lane."""
    result = _script(tmp_path, """
        sim.post(ROW, [5], dst=[1], captured=True, cpu_on=True, cpu_misses=True, weights=[1.0])
        host.pump()
        print("reached")
    """, parts=1)
    assert_aborted(result, "miss part")


def test_cpu_experts_need_the_copy_engine(tmp_path):
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT, copy_engine=False)
    try:
        with pytest.raises(Exception, match="need the copy engine"):
            host.enable_cpu_experts(
                host.test_kernel_address(), NO_SPLIT, _cores(), torch.zeros((ROWS, 16), dtype=torch.uint8),
                torch.zeros((ROWS, HIDDEN)), threads=1,
            )
    finally:
        host.stop()
