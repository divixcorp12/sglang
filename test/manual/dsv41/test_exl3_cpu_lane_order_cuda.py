"""CPU experts' miss order end to end on a real GPU against the real C++ service: the fused plan sorts the miss lanes by
residency key, the post types the tail lanes CPU, and DIRECT's commit inserts only the copied ones.

Two RAM hits A (low key) and B (high key), routed A first, both VRAM misses, split[2] = 1. The plan puts B in lane 0
and A in lane 1, so the post types A kHitCpu and B kHitCopy; B is copied into the first shortlist victim (the coldest)
and A's victim keeps its expert.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import ctypes
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import EXPERTS, LAYERS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe import expert_stream_transport as ops  # noqa: E402
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertForward  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.test.dsv41_ram_miss_fixtures import paused  # noqa: E402

HIDDEN = 64
HANDLE = 7
READY_STATE, FREE_STATE = 3, 0  # expert_residency_gpu's _READY and _FREE


class _Forward:
    """Records (layer, slots, weights) and writes a zero partial."""

    def __init__(self):
        self.calls = []
        self.c = CpuExpertForward(self._run)

    def _run(self, call):
        c = call.contents
        k, out = c.k, c.out
        self.calls.append((c.layer, [c.slots[i] for i in range(k)], [c.weights[i] for i in range(k)]))
        if not c.accumulate:
            for j in range(HIDDEN):
                out[j] = 0.0
        return 0

    @property
    def address(self) -> int:
        return ctypes.cast(self.c, ctypes.c_void_p).value


def _until(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


PART_HITS, PART_MISSES = 1, 2  # CC's cpu_lanes[1] bits: which CPU output part holds a partial


@pytest.mark.parametrize("miss", [False, True], ids=["ram_hit", "nvme_miss"])
def test_the_low_scored_lane_goes_to_the_cpu_and_the_high_scored_one_takes_the_coldest_victim(tmp_path, miss):
    """``miss``: the low-key expert is not in the RAM tier either, and SGLANG_DSV41_CPU_EXPERTS_MISSES is on. It is
    typed kMissCpu, read into its staging slot, computed there by the CPU into part 1, inserted in the RAM tier by the
    record's delta, and still left out of VRAM by DIRECT's commit. Mutations: the miss is computed before its read
    landed (the forward sees the wrong slot's bytes -- here: the late job never runs, and the forward count is 0); CC
    flags part 0 for a miss-only record."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
    from sglang.kernels.ops.moe.expert_residency_direct_gather import (
        direct_commit_gather,
        direct_gather_destinations,
    )
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    row, low, high = 0, 3, 7
    forward = _Forward()
    split = [0] * (lease.wire_layout(8).lanes + 1)
    split[2] = 1
    c = Chain(tmp_path, copy_engine=True, start=False, cpu_misses=miss)
    try:
        x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, 2, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(forward.address, split, cores, x_rows, out_rows, threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        backend, plan, dev = c.backends[row], c.plans[row], c.dev

        # The RAM tier's experts through an eager gather, which the service serves without the copy engine.
        c.plan([high] if miss else [low, high], row)
        c.gather(row)
        torch.cuda.synchronize()
        assert c.handled()
        assert ({high} if miss else {low, high}) <= c.resident(row) and (low in c.resident(row)) != miss
        with paused(c.host):
            ram_slot = {e: s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if e >= 0}
        c.host.set_cpu_layer(row, HANDLE)
        c.host.arm_copy_engine()

        # VRAM: slots 0..5 hold experts 10..15, neither A nor B resident; the shortlist names slot 3 coldest, then 5.
        slots = TOP_K
        mapping = torch.full((EXPERTS + 1,), -1, dtype=torch.int64, device="cuda")
        slot_to_expert = torch.full((slots + 1,), -1, dtype=torch.int64, device="cuda")
        for slot in range(slots):
            mapping[10 + slot] = slot
            slot_to_expert[slot] = 10 + slot
        slot_state = torch.full((slots + 1,), READY_STATE, dtype=torch.uint8, device="cuda")
        slot_state[slots] = FREE_STATE
        generations = torch.zeros(slots + 1, dtype=torch.int64, device="cuda")
        victims = torch.tensor([3, 5, 0, 1, 2, 4], dtype=torch.int64, device="cuda")
        valid = torch.ones(slots, dtype=torch.bool, device="cuda")
        keys = torch.zeros(EXPERTS, dtype=torch.int64, device="cuda")
        keys[low], keys[high] = 1, 2

        ids = torch.tensor([low, high], dtype=torch.int64, device="cuda")
        plan.expert_ids.fill_(-1)
        remap = torch.empty(2, dtype=torch.int64, device="cuda")
        plan_unique_routes_cuda(
            ids, mapping[:EXPERTS], slots, plan.expert_ids[:2], torch.empty(2, dtype=torch.int32, device="cuda"),
            plan.count, remap, None, None, None, torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"), 0, None, keys,
        )
        destinations = torch.zeros(slots, dtype=torch.int64, device="cuda")
        live = torch.zeros(slots, dtype=torch.bool, device="cuda")
        remap_out = torch.empty(2, dtype=torch.int64, device="cuda")
        direct_gather_destinations(
            ids, mapping[:EXPERTS], victims, valid, plan.count, remap, slots, plan.slots, destinations, live, remap_out
        )
        backend.routes.fill_(-1)
        backend.routes[:2].copy_(ids)
        torch.cuda.synchronize()
        assert plan.expert_ids[:2].tolist() == [high, low], "the plan sorts the higher key first"
        assert plan.slots[:2].tolist() == [3, 5] and remap_out.tolist() == [5, 3]

        # The production chain of a captured post with CPU input, eagerly (Chain.chain plus cpu_input).
        x = torch.randn(1, HIDDEN, device="cuda").half()
        weights = torch.tensor([[0.25, 0.75]], device="cuda")
        backend._stage_planned(plan)
        dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=(x, weights))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        snapshot = c.snapshot(row)
        # The fake forward takes the GIL: let it run before anything blocks in the host.
        assert _until(lambda: len(forward.calls) == 1)
        torch.cuda.synchronize()

        low_slot = int(dev.lane_slot[1]) if miss else ram_slot[low]
        assert forward.calls == [(HANDLE, [low_slot], [0.25])], "the tail lane (the low key) is the CPU's"
        assert c.kinds(2) == [LaneKind.HIT_COPY, LaneKind.MISS_CPU if miss else LaneKind.HIT_CPU]
        assert dev.cpu_lanes.tolist() == [0b10, PART_MISSES if miss else PART_HITS]
        if miss:
            assert c.handled() and c.host.mapping(row)[low] == low_slot, "the CPU miss was not inserted in RAM"
        want = c.expected([high], row)
        for n in c.names:
            got = snapshot[n][3].cpu().contiguous().view(torch.uint8)
            assert torch.equal(got, want[n][0].contiguous().view(torch.uint8))
            assert not snapshot[n][5].view(torch.uint8).any(), f"{n}: the CPU lane's victim was copied into"

        insertions, evictions, truncated = (torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(3))
        direct_commit_gather(
            destinations, live, plan.expert_ids, mapping, slot_to_expert, slot_state, generations,
            insertions, evictions, truncated, plan.count, torch.ones(1, dtype=torch.float32, device="cuda"),
            plan.count, ready=READY_STATE, free_state=FREE_STATE, cpu_lanes=dev.cpu_lanes,
        )
        torch.cuda.synchronize()
        assert int(mapping[high]) == 3 and int(slot_to_expert[3]) == high, "the high key takes the coldest victim"
        assert int(mapping[13]) == -1, "the coldest victim's expert was evicted"
        assert int(mapping[low]) == -1 and int(slot_to_expert[5]) == 15 and int(mapping[15]) == 5, (
            "the CPU lane's victim keeps its expert"
        )
        assert (insertions.item(), evictions.item(), truncated.item()) == (1, 1, 0)
        assert c.handled()
    finally:
        c.close()


@pytest.mark.parametrize("lanes, count", [(16, 12), (32, 32)])
def test_the_cpu_tail_past_8_lanes_goes_through_the_real_chain_and_the_split_planner(lanes, count):
    """The 8-lane test above at 16 and 32 lanes: ``count`` RAM hits, split[count] = 4, so the last four lanes (the four
    lowest keys; at 32 lanes, lanes 28-31) are the CPU's. They go through the real post, CW and CC (ce_mask and
    cpu_lanes, whose u32 masks reach bit 31), the route plan's miss order and DIRECT's commit, which leaves their
    victims alone. A build still fixed at 8 lanes cannot host this chain at all."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
    from sglang.kernels.ops.moe.expert_residency_direct_gather import (
        direct_commit_gather,
        direct_gather_destinations,
    )
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    cpu, slots, experts, row = 4, lanes, 2 * lanes, 0
    copied, base = count - cpu, lanes
    routed = list(range(count))
    split = [0] * count + [cpu] + [0] * (lanes - count)
    forward = _Forward()
    with tempfile.TemporaryDirectory() as tmp:
        c = Chain(Path(tmp), copy_engine=True, start=False, lanes=lanes, experts=experts, top_k=lanes, dst_rows=lanes,
                  capacity=2 * lanes, staging=lanes)
        try:
            x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
            out_rows = torch.zeros((LAYERS, 2, HIDDEN), dtype=torch.float32).pin_memory()
            cores = sorted(os.sched_getaffinity(0))[:2]
            c.host.enable_cpu_experts(forward.address, split, cores, x_rows, out_rows, threads=2)
            c.host.start_thread(fatal_wait_s=60.0)
            c.dev.enable_cpu_experts(x_rows, out_rows)
            c.dev.set_row_cpu(row)
            backend, plan, dev = c.backends[row], c.plans[row], c.dev

            c.plan(routed, row)
            c.gather(row)
            torch.cuda.synchronize()
            assert c.handled() and set(routed) <= c.resident(row)
            with paused(c.host):
                ram_slot = {e: s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if e >= 0}
            c.host.set_cpu_layer(row, HANDLE)
            c.host.arm_copy_engine()

            # VRAM: every slot holds an expert from base..2*base-1, none of the routed ones; victims are taken from slot 15 down.
            mapping = torch.full((experts + 1,), -1, dtype=torch.int64, device="cuda")
            slot_to_expert = torch.full((slots + 1,), -1, dtype=torch.int64, device="cuda")
            for slot in range(slots):
                mapping[base + slot] = slot
                slot_to_expert[slot] = base + slot
            slot_state = torch.full((slots + 1,), READY_STATE, dtype=torch.uint8, device="cuda")
            slot_state[slots] = FREE_STATE
            generations = torch.zeros(slots + 1, dtype=torch.int64, device="cuda")
            victims = torch.arange(slots - 1, -1, -1, dtype=torch.int64, device="cuda")
            valid = torch.ones(slots, dtype=torch.bool, device="cuda")
            keys = torch.zeros(experts, dtype=torch.int64, device="cuda")
            keys[:count] = torch.arange(1, count + 1, device="cuda")

            ids = torch.tensor(routed, dtype=torch.int64, device="cuda")
            plan.expert_ids.fill_(-1)
            remap = torch.empty(count, dtype=torch.int64, device="cuda")
            plan_unique_routes_cuda(
                ids, mapping[:experts], slots, plan.expert_ids[:count], torch.empty(count, dtype=torch.int32, device="cuda"),
                plan.count, remap, None, None, None, torch.zeros(1, dtype=torch.int64, device="cuda"),
                torch.zeros(1, dtype=torch.int32, device="cuda"), 0, None, keys,
            )
            destinations = torch.zeros(slots, dtype=torch.int64, device="cuda")
            live = torch.zeros(slots, dtype=torch.bool, device="cuda")
            remap_out = torch.empty(count, dtype=torch.int64, device="cuda")
            direct_gather_destinations(
                ids, mapping[:experts], victims, valid, plan.count, remap, slots, plan.slots, destinations, live,
                remap_out,
            )
            backend.routes.fill_(-1)
            backend.routes[:count].copy_(ids)
            torch.cuda.synchronize()
            by_key = routed[::-1]
            assert plan.expert_ids[:count].tolist() == by_key, "the plan sorts the higher key first"
            assert plan.slots[:count].tolist() == list(range(slots - 1, slots - 1 - count, -1))

            x = torch.randn(1, HIDDEN, device="cuda").half()
            weights = torch.tensor([[(r + 1) / 64 for r in routed]], device="cuda")
            # The eager gather above filled destination rows 0..11: a CPU lane's victim must still hold those bytes.
            before = c.snapshot(row)
            backend._stage_planned(plan)
            dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=(x, weights))
            copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
            dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
            dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
            snapshot = c.snapshot(row)
            assert _until(lambda: len(forward.calls) == 1)
            torch.cuda.synchronize()

            cpu_experts = by_key[copied:]
            (call,) = forward.calls
            assert call[0] == HANDLE and sorted(call[1]) == sorted(ram_slot[e] for e in cpu_experts), (
                "the four lowest-keyed lanes are the CPU's"
            )
            assert c.kinds(count) == [LaneKind.HIT_COPY] * copied + [LaneKind.HIT_CPU] * cpu
            cpu_mask = ((1 << cpu) - 1) << copied
            assert dev.cpu_lanes.tolist() == [cpu_mask - (cpu_mask >> 31 << 32), PART_HITS]
            want = c.expected(by_key[:copied], row)
            for n in c.names:
                for lane in range(copied):
                    got = snapshot[n][slots - 1 - lane].cpu().contiguous().view(torch.uint8)
                    assert torch.equal(got, want[n][lane].contiguous().view(torch.uint8)), (n, lane)
                for lane in range(copied, count):
                    slot = slots - 1 - lane
                    assert torch.equal(snapshot[n][slot].view(torch.uint8), before[n][slot].view(torch.uint8)), (
                        f"{n}: a CPU lane's victim was copied into"
                    )

            insertions, evictions, truncated = (torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(3))
            direct_commit_gather(
                destinations, live, plan.expert_ids, mapping, slot_to_expert, slot_state, generations,
                insertions, evictions, truncated, plan.count, torch.ones(1, dtype=torch.float32, device="cuda"),
                plan.count, ready=READY_STATE, free_state=FREE_STATE, cpu_lanes=dev.cpu_lanes,
            )
            torch.cuda.synchronize()
            for lane in range(copied):
                slot = slots - 1 - lane
                assert int(mapping[by_key[lane]]) == slot and int(slot_to_expert[slot]) == by_key[lane]
                assert int(mapping[base + slot]) == -1, "a copied lane's victim was evicted"
            for lane in range(copied, count):
                slot = slots - 1 - lane
                assert int(slot_to_expert[slot]) == base + slot and int(mapping[base + slot]) == slot, (
                    "a CPU lane's victim keeps its expert"
                )
            assert (insertions.item(), evictions.item(), truncated.item()) == (copied, copied, 0)
        finally:
            c.close()


@pytest.mark.parametrize("miss_lane", [None, 11], ids=["hits", "last_lane_is_a_miss"])
def test_cpu_lanes_past_bit_7_reach_the_route_tables_and_the_direct_gather(miss_lane):
    """At 16 lanes the split sends the last 4 of 12 eligible lanes (8-11) to the CPU. CW and CC carry them in
    ce_mask and cpu_lanes[0], the route tables rank their routes past every column, and DIRECT's commit leaves their
    victims alone. Mutation: a mask narrowed to bits 0-7 drops all four lanes at each of the three."""
    from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_commit_gather

    lanes, count, slots, hidden = 16, 12, 14, 64
    w = lease.wire_layout(lanes)
    raw = torch.zeros(w.lease_block_bytes + w.block_align, dtype=torch.uint8, pin_memory=True)
    block = raw[(-raw.data_ptr()) % w.block_align :]
    block[w.copy_done : w.copy_done + 8].view(torch.int64)[0] = 1  # request 1's CopyDone: CW opens the gate itself

    kinds = [LaneKind.HIT_COPY] * 8 + [LaneKind.HIT_CPU] * 4
    parts = PART_HITS
    if miss_lane is not None:
        kinds[miss_lane] = LaneKind.MISS_CPU
        parts |= PART_MISSES
    lane_kind = torch.zeros(w.lanes, dtype=torch.int32, device="cuda")
    lane_kind[:count] = torch.tensor([int(k) for k in kinds], dtype=torch.int32)
    lane_slot = torch.arange(w.lanes, dtype=torch.int32, device="cuda")
    dst_slots = torch.arange(100, 100 + w.lanes, dtype=torch.int32, device="cuda")
    state = torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, device="cuda")
    state[ops.STATE_WORDS["pending"]] = 1
    ce_mask = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    cpu_lanes = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ops._device_module("exl3", lanes).expert_stream_lease_copy_wait(
        state, torch.tensor([count], dtype=torch.int32, device="cuda"), int(block.data_ptr()), lane_kind, lane_slot,
        dst_slots, 0, 0, ce_mask, cpu_lanes, 0,
    )
    torch.cuda.synchronize()
    assert ce_mask.tolist() == [0xFFF, 0xF00, parts]
    assert cpu_lanes.tolist() == [0xF00, parts]

    # Lane i is route i, routed to slot i, as the post's plan makes it for a BS1 remap.
    remap = torch.arange(count, dtype=torch.int64, device="cuda")
    weights = torch.ones(count, dtype=torch.float32, device="cuda")
    keep = torch.ones(1, dtype=torch.float32, device="cuda")
    x = torch.ones(1, hidden, dtype=torch.float32, device="cuda")
    partial = torch.full((2, hidden), 3.0, dtype=torch.float32).pin_memory()
    out = torch.full((1, hidden), 5.0, dtype=torch.float32, device="cuda")
    count_out = torch.full((slots + 1,), 99, dtype=torch.int64, device="cuda")
    inv = torch.full((count,), -3, dtype=torch.int64, device="cuda")
    ws = torch.zeros(count, dtype=torch.float16, device="cuda")
    det = torch.zeros(3, slots + 1, dtype=torch.int64, device="cuda")
    exl3_moe_route_tables(
        remap, weights, keep, x, torch.zeros_like(remap), torch.zeros(1, hidden, dtype=torch.float16, device="cuda"),
        out, count_out, inv, ws, det, cpu_lanes=cpu_lanes, dst_slots=remap.to(torch.int32),
        cpu_out=partial.data_ptr(), cpu_part_stride=hidden,
    )
    torch.cuda.synchronize()
    assert count_out.tolist() == [1] * 8 + [0] * (slots + 1 - 8), "lanes 8-11 must count 0 in every slot"
    assert inv.tolist() == list(range(count)), "lanes 8-11 rank past every column, after lanes 0-7"
    assert out.eq(3.0 * bin(parts).count("1")).all(), "the flagged parts' partial sums seed the output"

    # DIRECT's commit: the four CPU lanes keep their victims' experts; only lanes 0-7 are inserted.
    experts = 2 * count
    mapping = torch.full((experts + 1,), -1, dtype=torch.int64, device="cuda")
    slot_to_expert = torch.full((count + 1,), -1, dtype=torch.int64, device="cuda")
    for slot in range(count):
        mapping[experts - 1 - slot] = slot
        slot_to_expert[slot] = experts - 1 - slot
    slot_state = torch.full((count + 1,), READY_STATE, dtype=torch.uint8, device="cuda")
    slot_state[count] = FREE_STATE
    insertions, evictions, truncated = (torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(3))
    delivered = torch.tensor([count], dtype=torch.int32, device="cuda")
    direct_commit_gather(
        torch.arange(count, dtype=torch.int64, device="cuda"), torch.ones(count, dtype=torch.bool, device="cuda"),
        torch.arange(count, dtype=torch.int64, device="cuda"), mapping, slot_to_expert, slot_state,
        torch.zeros(count + 1, dtype=torch.int64, device="cuda"), insertions, evictions, truncated, delivered,
        torch.ones(1, dtype=torch.float32, device="cuda"), delivered, ready=READY_STATE, free_state=FREE_STATE,
        cpu_lanes=cpu_lanes,
    )
    torch.cuda.synchronize()
    assert slot_to_expert[:count].tolist() == list(range(8)) + [experts - 1 - s for s in range(8, count)]
    assert (insertions.item(), truncated.item()) == (8, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
