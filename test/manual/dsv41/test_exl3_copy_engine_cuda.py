"""The copy engine (LEASE_PROTOCOL.md 7.6) against the real C++ service and the real kernel chain. GPU only.

The service copies each request's resident lanes with cuMemcpyAsync on its copy thread and publishes CopyDone once
cuEventQuery observed the copies complete; the chain post -> W1 -> C1 -> A1 -> S -> A2 -> CW -> F waits for it in CW.
Every byte check reads a snapshot the test enqueues on the decode stream right after F, before anything synchronizes
the device: a CopyDone published before the copies completed, or a CW that does not wait, shows as stale bytes there,
most visibly under the ballast, which delays every copy job's completion by one large extra copy.

Run on divix01 holding cc-gpu.lock, with PYTHONPATH pointing at the tree under test and SGLANG_EXL3_SRC set.
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import exl3_lease_block as lease  # noqa: E402
from test_exl3_piece_stream_cuda import (  # noqa: E402
    CAPACITY,
    EXPERTS,
    TOP_K,
    StreamService,
    _chain,
    _kernel_nodes,
    cuda_drv,
)

BALLAST_BYTES = 64 << 20  # ~5 ms at the copy engine's 13.5 GB/s: every job completes that much after its grant


def _snapshot_step(s):
    """The chain in production order, then a copy of every destination on the decode stream, taken before any sync."""
    s.post()
    s.hit_wait()
    s.copy1()
    s.ack1()
    s.stream()
    s.ack2()
    s.copy_wait()
    s.finalize()
    snapshot = {n: s.dest[n].clone() for n in s.names}
    s.total()
    torch.cuda.synchronize()
    return snapshot


def _check(s, experts, snapshot):
    want = s.expected(experts)
    for lane, expert in enumerate(experts):
        for n in s.names:
            got = snapshot[n][lane].cpu().view(torch.uint8)
            assert torch.equal(got, want[expert][n].view(torch.uint8)), (lane, expert, n)


def _all_retired(s):
    c = s.counters()
    return c["leases_granted"] == c["leases_acked"] + c["leases_voided"] + c["leases_copied"]


@pytest.fixture
def ce(tmp_path):
    s = StreamService(tmp_path, copy_engine=True)
    try:
        yield s
    finally:
        s.close()


def test_every_lane_holds_its_row_under_host_slot_victim_reuse(ce):
    """Sixteen experts through eight pinned slots: the service evicts and reuses host slots, and every step rewrites
    the same six destination slots. Each step's snapshot must hold exactly the planned rows."""
    s = ce
    rng = random.Random(7)
    for _ in range(40):
        experts = rng.sample(range(EXPERTS), TOP_K)
        s.plan(experts)
        snapshot = _snapshot_step(s)
        assert s.keep.item() == 1.0, (experts, s.counters(), s.stats())
        _check(s, experts, snapshot)
        assert s.until(lambda: _all_retired(s)), s.counters()
    c = s.counters()
    assert c["copy_jobs"] > 10 and c["leases_copied"] > 30 and c["evictions"] > 0, c
    assert c["copy_errors"] == 0 and c["copy_generation_mismatches"] == 0 and c["lease_double_signal"] == 0, c
    assert s.stats()["copy_waits"] == c["copy_jobs"]


def test_a_delayed_completion_is_waited_for_and_the_lease_holds_until_it(ce):
    """The ballast delays every job's completion by ~5 ms: CW must spin for it, and the bytes after F must be right.
    Mutants: complete a job without querying its event, publish CopyDone at the grant, or a CW that does not wait --
    each red on the snapshot."""
    s = ce
    src = torch.empty(BALLAST_BYTES, dtype=torch.uint8).pin_memory()
    dst = torch.empty(BALLAST_BYTES, dtype=torch.uint8, device="cuda")
    s.plan(list(range(TOP_K)))
    s.step()  # resident, all through S
    assert s.until(lambda: _all_retired(s)), s.counters()
    s.host.copy_engine_ballast(dst, src)
    try:
        spun = s.stats()["copy_spun"]
        for shift in range(6):
            experts = [(e + shift) % TOP_K for e in range(TOP_K)]  # every lane a hit, each in a new lane order
            s.plan(experts)
            snapshot = _snapshot_step(s)
            assert s.keep.item() == 1.0, (s.counters(), s.stats())
            _check(s, experts, snapshot)
        assert s.stats()["copy_spun"] >= spun + 6, s.stats()
    finally:
        s.host.copy_engine_ballast(None, None)
    assert s.until(lambda: _all_retired(s)), s.counters()


def test_hits_go_to_the_copy_engine_while_s_streams_the_misses(ce):
    s = ce
    s.plan([0, 1, 2])
    s.step()
    assert s.until(lambda: _all_retired(s))
    before = s.counters()
    experts = [0, 9, 1, 10, 2, 11]
    s.plan(experts)
    snapshot = _snapshot_step(s)
    assert s.keep.item() == 1.0, s.counters()
    _check(s, experts, snapshot)
    assert int(s.dev.go_ce.item()) == 3 and int(s.dev.go_2.item()) == 3 and int(s.dev.go_1.item()) == 0
    assert s.until(lambda: _all_retired(s))
    after = s.counters()
    assert after["copy_jobs"] - before["copy_jobs"] == 1 and after["copy_lanes"] - before["copy_lanes"] == 3


def test_s_hands_a_copying_lane_w1_did_not_claim_to_the_copy_wait(tmp_path):
    """The service is paused while the chain is launched, so W1 (no budget) passes before any lane is published and
    claims nothing; on resume the lanes are published COPYING and S must hand them to CW rather than copy them or
    fail the request on them."""
    s = StreamService(tmp_path, copy_engine=True, hit_wait_ns=0)
    try:
        s.plan([3, 4, 5])
        s.step()
        assert s.until(lambda: _all_retired(s))
        for _ in range(5):
            s.plan([3, 4, 5])
            s.host.pause(5.0)
            s.post()
            s.hit_wait()
            s.copy1()
            s.ack1()
            s.stream()
            s.ack2()
            s.copy_wait()
            s.finalize()
            snapshot = {n: s.dest[n].clone() for n in s.names}
            s.total()
            time.sleep(0.01)  # W1 has long finished its one pass
            s.host.resume()
            torch.cuda.synchronize()
            assert s.dev.claimed[:3].tolist() == [0, 0, 0], "W1 claimed a lane published after it ran"
            assert s.keep.item() == 1.0, (s.counters(), s.stats())
            _check(s, [3, 4, 5], snapshot)
            assert int(s.dev.go_ce.item()) == 3 and int(s.dev.go_1.item()) + int(s.dev.go_2.item()) == 0
            assert s.until(lambda: _all_retired(s))
    finally:
        s.close()


def test_a_copy_wait_timeout_fails_closed_and_the_terminal_does_not_release_the_copying_lease(tmp_path):
    """A 256 MiB ballast (~20 ms) against a 10 ms deadline: CW times out, F fails the request and names the lane in
    its terminal. The lease must stay held until the copy completes, then be released by the copy thread alone."""
    s = StreamService(tmp_path, copy_engine=True, timeout_ms=10)
    src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    try:
        s.plan([3])
        s.step()
        assert s.until(lambda: _all_retired(s))
        slot = s.host.mapping(0)[3]
        voided = s.counters()["leases_voided"]
        s.host.copy_engine_ballast(dst, src)
        s.plan([3])
        s.post()
        s.hit_wait()
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        s.copy_wait()
        s.finalize()
        torch.cuda.current_stream().synchronize()  # the decode stream only: the copy is still in flight
        seq = s.seq()
        idx = (seq - 1) % lease.RING
        assert s.keep.item() == 0.0 and s.stats()["timeouts"] >= 1
        term = s.block.terminal(idx)
        assert term["word"] == (1 << 56) | s.generation(seq) and term["mask"] & 1, term
        assert s.host.slot_info(0)[slot][2] == 1, "the COPYING lease was released while its copy was in flight"
        assert s.host.lease_entry(idx)["lane_state"][0] == 1
        assert s.host.copy_engine_idle(10.0)
        c = s.counters()
        assert c["leases_voided"] == voided and c["lease_double_signal"] == 0, c
        assert s.host.slot_info(0)[slot][2] == 0 and s.host.lease_entry(idx)["lane_state"][0] == 2
    finally:
        s.host.copy_engine_ballast(None, None)
        s.close()


@pytest.mark.parametrize("fillers", [0, 3000])
def test_the_first_replay_of_a_copy_engine_graph_completes_armed(ce, fillers):
    """Both smokes that failed at startup (abba1-on, diag arm1-on) failed on a decode step that ran armed early in
    the server's life, where a graph variant may run for the first time. Here a freshly captured graph, with
    ``fillers`` kernels on each side of the chain as a stand-in for the decode graph's size, is replayed for the
    first time with the copy engine armed and every lane a hit: its copy must complete inside the deadline."""
    s = ce
    s.plan([0, 1, 2])
    s.step()  # resident, through S
    assert s.until(lambda: _all_retired(s))
    backend = s.make_backend()
    plan = s.make_plan()
    x = torch.zeros(1, device="cuda")
    with torch.cuda.stream(torch.cuda.Stream()):
        backend.post(0, plan)  # eager warm-up: loads every kernel, copy engine not allowed
    torch.cuda.synchronize()
    assert s.until(lambda: _all_retired(s))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(fillers):
            x.add_(1)
        backend.post(0, plan)
        for _ in range(fillers):
            x.add_(1)
    jobs = s.counters()["copy_jobs"]
    experts = [2, 0, 1]
    s.plan(experts)
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    replay_s = time.perf_counter() - t0
    assert s.keep.item() == 1.0, (replay_s, s.counters(), s.stats())
    assert replay_s < 1.0, replay_s
    assert s.counters()["copy_jobs"] == jobs + 1, s.counters()
    assert s.delivered(experts)
    assert s.until(lambda: _all_retired(s)), s.counters()


@pytest.mark.parametrize("mode", ["LAZY", "EAGER"])
def test_a_kernels_first_launch_while_cw_spins_fails_stop_under_lazy_loading_only(tmp_path, mode):
    """The soak's fail-stop (docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md), reduced: a JIT kernel,
    built and loaded but never launched, is launched for the first time while the copy thread's copies are being
    issued and CW waits for them. Under LAZY the launch loads the kernel, which waits for the device, which waits in CW
    for copies that cannot be issued: the request times out and the page goes fatal. Under EAGER the kernel loaded
    with its library and the launch returns at once. The service refuses the copy engine without EAGER; this pins the
    reason. Each mode runs in its own process: CUDA reads CUDA_MODULE_LOADING once, at initialisation."""
    scenario = Path(__file__).resolve().parent / "ce_lazy_load_scenario.py"
    env = dict(os.environ, CUDA_MODULE_LOADING=mode)
    r = subprocess.run([sys.executable, str(scenario), str(tmp_path)], env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    row = json.loads([line for line in r.stdout.splitlines() if line.startswith("{")][-1])
    if mode == "EAGER":
        assert row["keep"] == 1.0 and row["timeouts"] == 0 and row["fatal"] == 0, row
        assert row["launch_ms"] < 100, row
    else:
        assert row["keep"] == 0.0 and row["timeouts"] == 1 and row["fatal"] != 0, row
        assert row["launch_ms"] > 1000, row
    assert row["x"] == 1.0, row  # the kernel itself ran once either way


@pytest.mark.parametrize("waiter", ["h2d", "d2h", "kernel"])
def test_work_queued_behind_the_graph_on_other_streams_does_not_hold_the_copy_back(ce, waiter):
    """What the overlap scheduler does around a decode graph: the graph replays on a forward stream, and before it
    finishes the scheduler queues work behind it on other streams (schedule_stream waits on the forward for its next
    H2D input copies; copy_stream waits on it for the result's D2H copy). The recipe with --disable-overlap-schedule
    ran the diag that deadlocks with overlap on (diag arm1nooverlap-on). Here each kind of queued-behind work is put
    behind the first and later armed replays of a copy-engine graph: the copy must complete inside the deadline."""
    s = ce
    s.plan([0, 1, 2])
    s.step()
    assert s.until(lambda: _all_retired(s))
    backend = s.make_backend()
    plan = s.make_plan()
    forward = torch.cuda.Stream()
    behind = torch.cuda.Stream()
    host = torch.empty(4 << 20, dtype=torch.uint8).pin_memory()
    dev = torch.empty(4 << 20, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(forward):
        backend.post(0, plan)
    torch.cuda.synchronize()
    assert s.until(lambda: _all_retired(s))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=forward):
        backend.post(0, plan)
    for replay in range(3):
        experts = [(e + replay) % 3 for e in range(3)]
        s.plan(experts)
        torch.cuda.synchronize()
        jobs = s.counters()["copy_jobs"]
        t0 = time.perf_counter()
        with torch.cuda.stream(forward):
            graph.replay()
        done = torch.cuda.Event()
        done.record(forward)
        behind.wait_event(done)
        with torch.cuda.stream(behind):
            if waiter == "h2d":
                dev.copy_(host, non_blocking=True)
            elif waiter == "d2h":
                host.copy_(dev, non_blocking=True)
            else:
                dev.add_(1)
        done.synchronize()
        replay_s = time.perf_counter() - t0
        torch.cuda.synchronize()
        assert s.keep.item() == 1.0, (replay, replay_s, s.counters(), s.stats())
        assert replay_s < 1.0, (replay, replay_s)
        assert s.counters()["copy_jobs"] == jobs + 1, s.counters()
        assert s.delivered(experts)
        assert s.until(lambda: _all_retired(s)), s.counters()


@pytest.mark.skipif(cuda_drv is None, reason="needs cuda-python (cuda.bindings.driver)")
def test_the_captured_chain_waits_in_cw_and_replays_right_armed_and_unarmed(ce):
    """The captured backend chain is post -> W1 -> C1 -> A1 -> S -> A2 -> CW -> F -> add -> add; its replays deliver
    the planned rows both with the copy engine unarmed (the hits go READY through C1) and armed (COPYING)."""
    s = ce
    s.plan([])
    backend = s.make_backend()
    plan = s.make_plan()
    with torch.cuda.stream(torch.cuda.Stream()):
        backend.post(0, plan)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        backend.post(0, plan)
    assert s.dev.copy_engine_captured
    chain = _chain(graph.raw_cuda_graph())
    cw = _kernel_nodes(s.copy_wait)
    finalize = _kernel_nodes(lambda: s.dev.finalize(s.count, s.keep))
    assert len(cw) == 1 and len(finalize) == 1
    position = chain.index(cw[0])
    assert chain[position + 1] == finalize[0], chain
    rng = random.Random(11)
    for armed in (False, True, False, True):
        s.host.arm_copy_engine(armed)
        jobs = s.counters()["copy_jobs"]
        for _ in range(8):
            experts = rng.sample(range(CAPACITY), TOP_K)  # mostly resident after the first replays
            s.plan(experts)
            graph.replay()
            torch.cuda.synchronize()
            assert s.keep.item() == 1.0, (armed, s.counters(), s.stats())
            assert s.delivered(experts), (armed, experts)
            assert s.until(lambda: _all_retired(s))
        grew = s.counters()["copy_jobs"] > jobs
        assert grew == armed, (armed, s.counters())


# ---- SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES: CW reads the small tensors; the lease waits for its reads ----

CW_DELAY_CYCLES = 600_000_000  # ~250 ms at the 5090's clock: CW starts long after the trellis DMA has completed


def _victim_sequence(s, steps, seed):
    rng = random.Random(seed)
    snapshots = []
    for _ in range(steps):
        experts = rng.sample(range(EXPERTS), TOP_K)
        s.plan(experts)
        snapshot = _snapshot_step(s)
        assert s.keep.item() == 1.0, (experts, s.counters(), s.stats())
        _check(s, experts, snapshot)
        snapshots.append({n: t.cpu() for n, t in snapshot.items()})
        assert s.until(lambda: _all_retired(s)), s.counters()
    return snapshots


def test_sm_small_copies_deliver_the_same_six_tensors_as_the_six_copy_path(tmp_path):
    """Sixty steps of sixteen experts through eight pinned slots, host slots evicted and reused, on the six-copy path
    and on the SM path: every snapshot of every tensor must be byte-identical, and equal to the checkpoint's rows."""
    off = StreamService(tmp_path / "off", copy_engine=True)
    try:
        want = _victim_sequence(off, 60, seed=11)
    finally:
        off.close()
    on = StreamService(tmp_path / "on", copy_engine=True, sm_small=True)
    try:
        got = _victim_sequence(on, 60, seed=11)
        c = on.counters()
        assert c["copy_jobs"] > 20 and c["leases_copied"] > 60 and c["evictions"] > 0, c
        assert c["copy_errors"] == 0 and c["copy_generation_mismatches"] == 0 and c["lease_double_signal"] == 0, c
        # Only the two trellis tensors per lane went through the copy engine.
        trellis = sum(on.dest[n][0].numel() * on.dest[n].element_size() for n in on.names if n.endswith("_trellis"))
        assert c["copy_bytes"] == c["copy_lanes"] * trellis, (c["copy_bytes"], c["copy_lanes"], trellis)
    finally:
        on.close()
    for step, (a, b) in enumerate(zip(want, got)):
        for n in a:
            assert torch.equal(a[n].view(torch.uint8), b[n].view(torch.uint8)), (step, n)


def test_a_slab_row_rewritten_the_moment_its_lease_is_released_never_reaches_the_destination(tmp_path):
    """The hazard: CW reads the small tensors from the pinned slot, so the slot must stay leased until CW has read it,
    not merely until the DMA completed. CW is delayed ~250 ms behind a device sleep; a watcher rewrites every leased
    slot with a sentinel the instant its lease drops. The destination must hold the checkpoint's bytes.
    Mutant: release on the DMA's completion alone -- the sentinel lands before CW reads and the snapshot is red."""
    s = StreamService(tmp_path, copy_engine=True, sm_small=True)
    try:
        experts = list(range(TOP_K))
        s.plan(experts)
        s.step()  # resident
        assert s.until(lambda: _all_retired(s)), s.counters()
        slots = [s.host.mapping(s.row)[e] for e in experts]
        originals = {n: s.slabs[s.row][n].clone() for n in s.names}
        s.plan(experts)
        s.post()
        s.hit_wait()
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        torch.cuda._sleep(CW_DELAY_CYCLES)
        s.copy_wait()
        s.finalize()
        snapshot = {n: s.dest[n].clone() for n in s.names}
        s.total()
        done = torch.cuda.Event()
        done.record()
        before = s.counters()["leases_copied"]
        rewritten = False
        held_after_dma = None
        t0 = time.perf_counter()
        while not rewritten:
            info = s.host.slot_info(s.row)
            if held_after_dma is None and time.perf_counter() - t0 > 0.1:
                held_after_dma = all(info[slot][2] >= 1 for slot in slots) and not done.query()
            if all(info[slot][2] == 0 for slot in slots):
                for n in s.names:
                    s.slabs[s.row][n][slots].view(torch.uint8).fill_(0xAB)
                rewritten = True
            elif time.perf_counter() - t0 > 10:
                break
        torch.cuda.synchronize()
        assert rewritten, s.counters()
        assert s.counters()["leases_copied"] - before == TOP_K
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        _check(s, experts, snapshot)
        assert held_after_dma, "the leases dropped before CW ran"
    finally:
        for n in s.names:
            s.slabs[s.row][n].copy_(originals[n])
        s.close()
