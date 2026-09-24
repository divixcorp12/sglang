"""Task 6 V1 (two-phase), device chain: the partial terminal mask a failed request publishes names only the
lanes no stage acknowledged, never the whole request (T5 of docs/superpowers/plans/task6-v1-checklist.md
section 5).

GPU. Copies the real-service end-to-end recipe of test_exl3_lease_kernels_cuda.py's Service/TestServiceEndToEnd,
with two-phase enabled and the chain driven stage by stage (post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F) instead
of through Exl3RamMissDevice.wait/ack.

Run on divix01 under gpu-run.sh (it holds cc-gpu.lock) with PYTHONPATH pointing at the tree under test.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import exl3_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe.exl3_ram_miss import (  # noqa: E402
    DEMAND_RECORDS,
    Exl3RamMissDevice,
    Exl3RamMissHost,
    new_page,
)

LAYERS, EXPERTS, CAPACITY = 2, 16, 8
TOP_K = 6
WORD_MASK = (1 << 64) - 1


def _close(host, slabs):
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    try:
        if host is not None:
            host.stop()
    finally:
        release_host_slabs([slab for names in slabs.values() for slab in names.values()])


def _terminal(host, seq):
    """Read a request's terminal record directly off the lease block, as the CPU LeaseSim does."""
    idx = (seq - 1) % DEMAND_RECORDS
    layout = host.lease_layout
    block = host.lease_block
    base = layout.d_offset + lease.TERMINAL + idx * lease.TERMINAL_BYTES
    f = lease.TERMINAL_FIELDS
    mask = int(block[base + f["skipped_mask"] : base + f["skipped_mask"] + 4].view(torch.int32)[0]) & 0xFFFFFFFF
    reason = int(block[base + f["reason"] : base + f["reason"] + 4].view(torch.int32)[0]) & 0xFFFFFFFF
    word = int(block[base + f["gen"] : base + f["gen"] + 8].view(torch.int64)[0]) & WORD_MASK
    return {"mask": mask, "reason": reason, "word": word}


class TwoPhaseService:
    """The real service thread, two-phase enabled, driven stage by stage. Copied from
    test_exl3_lease_kernels_cuda.py's Service, extended with the D1-D4 stage API
    (Exl3RamMissDevice.hit_wait/rest_wait/stage_ack/finalize)."""

    def __init__(self, tmp_path, *, timeout_ms=2000):
        from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
        from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
        from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        self.layout = build_exl3_expert_layout(str(tmp_path))
        self.fmt = Exl3ExpertFormat(self.layout, 0, direct=False)
        self.specs = {s.name: s for s in self.fmt.tensor_specs(None)}
        self.names = EXL3_STREAMED_NAMES
        self.slabs = {lid: {} for lid in range(LAYERS)}
        self.host = None
        try:
            for lid in range(LAYERS):
                for n in self.names:
                    self.slabs[lid][n] = allocate_host_slab(CAPACITY, self.specs[n].row_shape, self.specs[n].dtype, register=True)
            tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
            self.page = new_page(pin=True)
            slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
            self.slot_map = slot_map
            self.host = Exl3RamMissHost(tables, page=self.page, slot_map=slot_map, direct=False)
            self.host.enable_lease_mode()
            self.host.enable_two_phase()
            self.host.start_thread(fatal_wait_s=60.0)
            self.dev = Exl3RamMissDevice(
                self.page, slot_map, device="cuda", layers=LAYERS, timeout_ms=timeout_ms, advise=False,
                lease_block=self.host.lease_block, lease_layout=self.host.lease_layout,
            )
        except BaseException:
            _close(self.host, self.slabs)
            raise
        self.planned = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {
            n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda") for n in self.names
        }
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments

        self.segments = expert_row_segments([(self.slabs[0][n], self.dest[n]) for n in self.names])

    def close(self):
        _close(self.host, self.slabs)

    def plan(self, experts):
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: len(experts)] = torch.tensor(experts, dtype=torch.int64)

    def step(self, row=0, *, hit_wait_ns=100_000):
        """post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F, one request, one stream. Returns the request's seq."""
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        self.dev.post(row, self.planned, self.count, self.routes, -1)
        seq = int(self.dev.stats()["posted"]) & 0xFFFFFFFF
        self.dev.hit_wait(row, self.planned, self.count, self.dest_slots, hit_wait_ns)
        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_1, self.dev.dst_slots_1, self.dev.go_1)
        self.dev.stage_ack(1)
        self.dev.rest_wait(row, self.planned, self.count, self.dest_slots, self.ram_miss)
        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_2, self.dev.dst_slots_2, self.dev.go_2)
        self.dev.stage_ack(2)
        self.dev.finalize(self.count, self.keep)
        torch.cuda.synchronize()
        return seq

    def until(self, predicate, timeout_s=10.0):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.002)
        return False


@pytest.fixture
def service(tmp_path):
    s = TwoPhaseService(tmp_path)
    try:
        yield s
    finally:
        s.close()


def test_the_partial_terminal_mask_names_only_unacknowledged_lanes(service):
    """T5. Stage 1 copies and acknowledges a resident lane; stage 2 fails on an injected read failure. The
    finalize kernel's terminal must clear the hit lane's bit and set the miss lane's.

    Mutant: restore the whole-request mask `(1u << named) - 1u` in exl3_ram_miss.cuh's finalize kernel (the old
    single-stage formula), in place of the per-lane acknowledgement-word loop. Must go red on the mask bit.

    The checklist's second detector, kLeaseDoubleSignal, does NOT independently fire in this minimal setup and is
    not asserted as a kill signal here: retire_leases()'s early-out on lanes_outstanding_ == 0 means the double
    signal for this test's own hit lane is never evaluated once that lane's own ack has already retired it and
    nothing else is outstanding, whether or not the mask is later wrong. Reported to the lead rather than
    engineered around with a second held lease, which introduced its own unexplained side effects.
    """
    s = service
    s.plan([3])
    s.step()  # expert 3 is a genuine miss here: makes it resident for the request below
    assert s.until(lambda: s.host.counters()["leases_acked"] == 1), s.host.counters()

    # The delay is what makes go_1 below deterministic, and it is not optional: inject()'s sleep runs BEFORE
    # the fail_reads check, so it holds the request open after S2 has already published the hit lane. Without
    # it this test races the host's own fail-fast path -- measured 2026-09-22 failing 1 run in 12 at
    # 2e4f2f0e3c, with go_1 == 0 and leases_voided == 1, because the host reached demand_done before stage 1
    # ever polled and W1 then broke out of its poll on the "request served, nothing left to discover" exit.
    # That exit's premise does not hold for a FAILED request, whose hit lanes are voided rather than
    # published. Lengthening stage 1's wait does not help (verified at a 2000000-poll bound: still 1 in 12); only holding the
    # request open does.
    s.host.inject(delay_s=0.2, fail_reads=True)
    s.plan([3, 9])  # lane 0 hits (resident above), lane 1 misses and the read fails
    seq = s.step()

    assert s.keep.item() == 0.0, "a mixed request with a failed miss lane must not report served"
    assert s.dev.go_1.item() == 1, "stage 1 must still claim the hit lane despite the failed read"
    assert s.dev.go_2.item() == 0, "stage 2 must commit nothing: its only lane's read failed"
    assert s.until(lambda: s.host.counters()["leases_acked"] == 2), s.host.counters()  # 1 above + this hit lane

    # The miss lane (expert 9) was never granted a lease -- fail_reads trips before S3's grant, exactly as T4's
    # host-side failure path does -- so there is nothing for retire_leases to void for it; leases_voided is not
    # part of this claim. s.step() already synced the stream, so the finalize kernel's terminal store is visible.
    terminal = _terminal(s.host, seq)
    assert terminal["mask"] & 0b01 == 0, f"the hit lane's bit must be clear: {terminal}"
    assert terminal["mask"] & 0b10 != 0, f"the miss lane's bit must be set: {terminal}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
