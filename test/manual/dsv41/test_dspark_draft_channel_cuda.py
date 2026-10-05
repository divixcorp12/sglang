"""The draft channel's device half (draft_kernels.cuh): post, finish (close + wait node) and commit (GPU).

A Python thread stands in for the host half (Task 4's DraftCpuThread): it watches the head word through a numpy view
of the pinned channel, checks the record's seq, writes the stage's output rows, then completes the way the lease
channel's host `complete` does (done[G], then open the gate if it reads closed(G)). Python has no store->load fence,
so the stand-in keeps re-checking the gate until it reads open(G); the real host half fences.
"""

import os
import sys
import threading
import time

import numpy as np
import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, E, STAGES = 256, 16, 2


def _u32(buf, off):
    return int(buf[off : off + 4].view(np.uint32)[0])


class StandIn:
    """The host half, in Python. `answer(stage, rows)` gives the rows it writes; `delay_s` holds each answer back;
    `forge` opens the gate without storing done (the commit must trap)."""

    def __init__(self, areas, *, delay_s=0.0, forge=False):
        from sglang.kernels.ops.moe.dspark_draft_cpu import DraftWire

        self.w = DraftWire()
        self.areas, self.delay_s, self.forge = areas, delay_s, forge
        self.buf = areas.channel.numpy()
        self.next = 1
        self.served = []
        self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def value(stage, rows):
        return torch.arange(rows * H, dtype=torch.float32).reshape(rows, H) * 0.001 + 10.0 * (stage + 1) + rows

    def run(self):
        w = self.w
        while not self.stop:
            head = _u32(self.buf, w.head)
            if head == 0 or (head - self.next) & 0x80000000:
                time.sleep(0.0002)
                continue
            seq = self.next
            rec = w.ring + ((seq - 1) % w.records) * w.record_bytes
            assert _u32(self.buf, rec) == seq, "a torn or lapped record"
            stage = int(self.buf[rec + w.rec_stage : rec + w.rec_stage + 2].view(np.uint16)[0])
            rows, k = int(self.buf[rec + w.rec_rows]), int(self.buf[rec + w.rec_k])
            epoch = _u32(self.buf, rec + w.rec_epoch)
            self.served.append((seq, stage, rows, k))
            time.sleep(self.delay_s)
            self.areas.out[stage, :rows] = self.value(stage, rows)
            gen = (epoch << 32) | seq
            if not self.forge:
                self.buf[w.done + 8 * ((seq - 1) % w.records) :][:8].view(np.uint64)[0] = gen
            closed, opened = lease.gate_word(seq, "closed"), lease.gate_word(seq, "open")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not self.stop:
                g = _u32(self.buf, w.gate)
                if g == closed:
                    self.buf[w.gate : w.gate + 4].view(np.uint32)[0] = opened
                elif g == opened:
                    break
                time.sleep(0.0002)
            self.next = seq + 1

    def close(self):
        self.stop = True
        self.thread.join(timeout=5)


@pytest.fixture
def channel():
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas, DraftCpuDevice

    areas = DraftCpuAreas(stages=STAGES, hidden=H)
    on_cpu = torch.zeros(STAGES, E, dtype=torch.uint8)
    on_cpu[1, [2, 4]] = 1
    return areas, DraftCpuDevice(areas, on_cpu, torch.device("cuda"))


def _inputs(rows, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=g).to("cuda", torch.bfloat16)
    ids = torch.stack([torch.randperm(6, generator=g)[:k] for _ in range(rows)]).cuda()
    weights = torch.rand(rows, k, generator=g).cuda()
    return x, ids, weights


def test_the_post_stages_the_cpu_routes_and_publishes_one_record(channel):
    areas, dev = channel
    x, ids, w = _inputs(5)
    dev.post(1, x, ids, w)
    torch.cuda.synchronize()
    assert torch.equal(areas.x[1, :5], x.half().cpu())
    cpu = (ids == 2) | (ids == 4)
    assert torch.equal(areas.slots[1, :5, :3], torch.where(cpu, ids, -1).int().cpu())
    assert torch.equal(areas.weights[1, :5, :3], torch.where(cpu, w, 0.0).cpu())
    buf, wire = areas.channel.numpy(), dev.wire
    rec = wire.ring
    assert _u32(buf, rec) == 1 and int(buf[rec + wire.rec_stage]) == 1
    assert (int(buf[rec + wire.rec_rows]), int(buf[rec + wire.rec_k])) == (5, 3)
    assert _u32(buf, wire.head) == 1


def test_a_call_with_no_cpu_route_posts_nothing_and_finish_does_not_wait(channel):
    areas, dev = channel
    x, ids, w = _inputs(4)
    out = torch.ones(4, H, device="cuda")
    dev.post(0, x, ids, w)  # stage 0 owns no CPU expert
    dev.finish(0, out)
    torch.cuda.synchronize()  # nothing answers: a wait here would hang
    assert _u32(areas.channel.numpy(), dev.wire.head) == 0
    assert torch.equal(out, torch.ones(4, H, device="cuda"))


def test_finish_waits_for_the_host_and_adds_its_rows(channel):
    areas, dev = channel
    host = StandIn(areas, delay_s=0.05)
    try:
        x, ids, w = _inputs(5, seed=1)
        ids[0, 0] = 2  # at least one CPU route
        out = torch.full((5, H), 0.5, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        dev.post(1, x, ids, w)
        dev.finish(1, out)
        queued = time.perf_counter() - started
        torch.cuda.synchronize()
        waited = time.perf_counter() - started
        assert queued < 0.05 and waited >= 0.05, (queued, waited)
        assert torch.equal(out.cpu(), 0.5 + StandIn.value(1, 5))
        assert host.served == [(1, 1, 5, 3)]
    finally:
        host.close()


def test_a_captured_post_and_finish_replay_one_record_each(channel):
    areas, dev = channel
    host = StandIn(areas)
    try:
        x, ids, w = _inputs(5, seed=2)
        ids[:, 0] = 4
        out = torch.zeros(5, H, device="cuda")
        dev.post(1, x, ids, w)  # warm-up, eager
        dev.finish(1, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dev.post(1, x, ids, w)
            out.mul_(0.0)  # the GPU share's stand-in, between post and finish
            dev.finish(1, out)
        for replay in range(3):
            graph.replay()
            torch.cuda.synchronize()
            buf = areas.channel.numpy()
            assert _u32(buf, dev.wire.head) == 2 + replay
            assert _u32(buf, dev.wire.gate) == lease.gate_word(2 + replay, "open")
            assert torch.equal(out.cpu(), StandIn.value(1, 5))
    finally:
        host.close()


_TRAP_CHILD = """
import sys, threading, time
import torch
sys.path.insert(0, sys.argv[1])
from test_dspark_draft_channel_cuda import StandIn, _inputs, H, E, STAGES
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas, DraftCpuDevice
areas = DraftCpuAreas(stages=STAGES, hidden=H)
on_cpu = torch.zeros(STAGES, E, dtype=torch.uint8); on_cpu[1, 2] = 1
dev = DraftCpuDevice(areas, on_cpu, torch.device("cuda"))
host = StandIn(areas, forge=True)
x, ids, w = _inputs(2); ids[:, 0] = 2
out = torch.zeros(2, H, device="cuda")
dev.post(1, x, ids, w); dev.finish(1, out)
torch.cuda.synchronize()
print("reached")
"""


def test_the_commit_traps_when_the_gate_opens_without_done():
    from sglang.kernels.ops.moe import dspark_draft_cpu
    from sglang.test.dsv41_ram_miss_fixtures import NO_CORE_DUMP

    import subprocess

    dspark_draft_cpu.device_module()  # the child finds it built
    result = subprocess.run(
        [sys.executable, "-c", NO_CORE_DUMP + _TRAP_CHILD, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "reached" not in result.stdout, result.stdout
    assert result.returncode != 0
    assert "illegal instruction" in result.stderr or "unspecified launch failure" in result.stderr, result.stderr[-2000:]
