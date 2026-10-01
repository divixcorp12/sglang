"""The slot-map chain on a real GPU against the real C++ service: a shared rig for the manual CUDA suites.

``Chain`` builds what the service's ``attach`` builds for each layer -- a host over row images read with O_DIRECT
into registered pinned slabs, every row attached with ``staging`` staging slots, one ``ExpertStreamDevice`` and one
production ``Exl3RamMissRowBackend`` per row -- and drives a gather with ``backend.post``: post -> C1 -> S -> CW ->
stream wait -> CC, in one stream.
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
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, image_tables, paused

# Fourteen slots per row, six of them staging (one per lane a post can miss): eight mappable rows.
LAYERS, EXPERTS, CAPACITY, TOP_K = 2, 16, 14, 6
STAGING = TOP_K
DST_ROWS = TOP_K
READY = 2  # slot_info's state of a resident slot


class Chain:
    def __init__(
        self, tmp_path, *, capacity=CAPACITY, staging=STAGING, mirror_weights=None, timeout_ms=2000, lease_pdl=False,
        copy_engine=False, sm_small_copies=False, copy_wait_ms=2000, start=True, variant="instr", hit_copy="ce",
        cpu_misses=False,
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
            self.host.reserve_staging(staging)
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
                hit_copy=hit_copy, cpu_misses=cpu_misses,
            )
            if copy_engine:
                for row in range(LAYERS):
                    self.dev.set_row_copy(row, DST_ROWS)
            host_row_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
            self.backends = {
                row: Exl3RamMissRowBackend(
                    {0: self.segments[row]}, host_row_map, self.dev, row, TOP_K,
                    {0: stream_segment_map(self.segments[row], self.tables, row)},
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

    def chain(self, row=0, *, captured=False, before_c1=None, before_cw=None):
        """``backend.post``'s chain step by step, eagerly: ``captured`` sets the record flag an eager post never
        carries (so the post may type copy-engine and CPU lanes), and the hooks run on the stream between the steps."""
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        backend, plan, dev = self.backends[row], self.plans[row], self.dev
        backend._stage_planned(plan)
        dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=captured)
        if before_c1 is not None:
            before_c1()
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        if before_cw is not None:
            before_cw()
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)

    def kinds(self, count):
        """The lane kinds the last post typed (ram_slot_map.LaneKind values)."""
        return self.dev.lane_kind[:count].tolist()

    def device_map(self, row=0):
        return self.dev.map_bank["ram_slot"][row].tolist()

    def device_staging(self, row=0):
        return self.dev.map_bank["staging"][row].tolist()

    def sync_bulk(self):
        """after_host_use: the host's bulk delta onto the device (paused, or no thread)."""
        torch.cuda.synchronize()
        self.host.pause(10.0)
        try:
            bulk = self.host.take_bulk_delta()
            if bulk.numel():
                self.dev.map_bulk_apply(bulk)
                torch.cuda.synchronize()
        finally:
            self.host.resume()

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

    def resident(self, row=0):
        with paused(self.host):
            return {e for state, e, _ in self.host.slot_info(row) if state == READY and e >= 0}

    def handled(self, timeout_s=10.0):
        """The service has finished every record the device posted."""
        posted = int(self.dev.stats()["posted"]) & 0xFFFFFFFF
        return until(lambda: self.host.handled_through() == posted, timeout_s)


def until(predicate, timeout_s=10.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False
