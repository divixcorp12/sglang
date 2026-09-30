"""The lease chain on a real GPU against the real C++ service: a shared rig for the manual CUDA suites.

``Chain`` builds what the service's ``attach`` builds for each layer -- a host over row images read with O_DIRECT
into registered pinned slabs, one ``ExpertStreamDevice`` and one production ``Exl3RamMissRowBackend`` per row -- and
drives a gather with ``backend.post``: post -> W1 -> C1 -> S -> CW -> stream wait -> CC, in one stream.
"""

from __future__ import annotations

import time

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_stream_transport import (
    ExpertStreamDevice,
    ExpertStreamHost,
    new_page,
    stream_segment_map,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend, sm_copy_mask, sm_copy_table
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab, release_host_slabs
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan
from sglang.test.dsv41_fake_exl3 import write_fake_exl3
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, image_tables

LAYERS, EXPERTS, CAPACITY, TOP_K = 2, 16, 8, 6
DST_ROWS = TOP_K
READY = 2  # slot_info's state of a resident slot


class Chain:
    def __init__(
        self, tmp_path, *, capacity=CAPACITY, mirror_weights=None, timeout_ms=2000, lease_pdl=False,
        copy_engine=False, sm_small_copies=False, copy_wait_ms=2000, hit_wait_ns=100_000, start=True,
        variant="instr",
    ):
        write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=ROW_IMAGE_DIM,
                        inter=ROW_IMAGE_DIM, finite=True)
        self.layout = build_exl3_expert_layout(str(tmp_path))
        self.fmt = Exl3ExpertFormat(self.layout, 0, direct=False)
        self.specs = {s.name: s for s in self.fmt.tensor_specs(None)}
        self.names = EXL3_STREAMED_NAMES
        self.slabs = {row: {} for row in range(LAYERS)}
        self.host = None
        try:
            for row in range(LAYERS):
                for n in self.names:
                    spec = self.specs[n]
                    self.slabs[row][n] = allocate_host_slab(capacity, spec.row_shape, spec.dtype, register=True)
            self.tables, _ = image_tables(self.layout, self.fmt.segment_map(), self.slabs, tmp_path, mirror_weights)
            self.page = new_page(pin=True)
            self.host = ExpertStreamHost(
                self.tables, page=self.page,
                slot_map=torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory(), variant=variant,
            )
            self.dest = {
                row: {n: torch.zeros((DST_ROWS,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda")
                      for n in self.names}
                for row in range(LAYERS)
            }
            self.segments = {
                row: expert_row_segments([(self.slabs[row][n], self.dest[row][n]) for n in self.names])
                for row in range(LAYERS)
            }
            sm_mask = sm_copy_mask(self.names) if sm_small_copies else 0
            if copy_engine:
                self.host.enable_copy_engine(torch.cuda.current_device(), wait_timeout_ms=copy_wait_ms)
                for row in range(LAYERS):
                    self.host.set_copy_table(row, self.segments[row].table, DST_ROWS, sm_mask=sm_mask)
            if start:
                self.host.start_thread(fatal_wait_s=60.0)
            self.dev = ExpertStreamDevice(
                self.page, self.host.lease_block, device="cuda", layers=LAYERS, experts=EXPERTS,
                timeout_ms=timeout_ms, piece_runs=self.host.piece_runs(),
                row_capacities=[int(c) for c in self.tables.capacity], lease_pdl=lease_pdl,
            )
            host_row_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
            self.backends = {
                row: Exl3RamMissRowBackend(
                    {0: self.segments[row]}, host_row_map, self.dev, row, TOP_K,
                    {0: stream_segment_map(self.segments[row], self.tables, row)}, hit_wait_ns=hit_wait_ns,
                    copy_engine=copy_engine,
                    copy_sm_table=sm_copy_table(self.segments[row], sm_mask) if sm_mask else None,
                )
                for row in range(LAYERS)
            }
        except BaseException:
            self.close()
            raise
        self.plans = {
            row: ExpertRowPlan(
                torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda"),
                torch.arange(TOP_K, dtype=torch.int32, device="cuda"),
                torch.zeros(1, dtype=torch.int32, device="cuda"),
            )
            for row in range(LAYERS)
        }

    def close(self):
        try:
            torch.cuda.synchronize()
            if self.host is not None:
                self.host.stop()
        finally:
            release_host_slabs([slab for names in self.slabs.values() for slab in names.values()])

    def plan(self, experts, row=0):
        """Set ``row``'s plan (device copies, so a captured gather replays the new plan) and its protect routes."""
        plan, backend = self.plans[row], self.backends[row]
        ids = torch.full((TOP_K,), -1, dtype=torch.int64)
        ids[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        plan.expert_ids.copy_(ids)
        plan.count.fill_(len(experts))
        backend.routes.copy_(ids)

    def gather(self, row=0):
        """The production gather of ``row``'s current plan into its destination rows 0..count-1."""
        self.backends[row].post(0, self.plans[row])

    def snapshot(self, row=0):
        """Every destination tensor, copied on the gather's stream before anything synchronizes the device."""
        return {n: t.clone() for n, t in self.dest[row].items()}

    def expected(self, experts, row=0):
        layer = int(self.tables.layer_ids[row])
        out = {n: torch.empty((len(experts),) + self.specs[n].row_shape, dtype=self.specs[n].dtype) for n in self.names}
        Exl3ShardRowSource.for_layer(self.layout, layer, self.fmt.segment_map(), direct=False).read(
            torch.tensor(experts), out
        )
        return out

    def check(self, experts, snapshot, row=0):
        want = self.expected(experts, row)
        for lane, expert in enumerate(experts):
            for n in self.names:
                got = snapshot[n][lane].cpu().contiguous().view(torch.uint8)
                assert torch.equal(got, want[n][lane].contiguous().view(torch.uint8)), (lane, expert, n)

    def leases(self, row=0):
        return [info[2] for info in self.host.slot_info(row)]

    def resident(self, row=0):
        return {e for state, e, _ in self.host.slot_info(row) if state == READY and e >= 0}

    def retired(self, timeout_s=10.0):
        """Every lease of every row released (the service retires on Done, or on CopyDone for copied lanes)."""
        return until(lambda: all(not any(self.leases(row)) for row in range(LAYERS)), timeout_s)


def until(predicate, timeout_s=10.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False
