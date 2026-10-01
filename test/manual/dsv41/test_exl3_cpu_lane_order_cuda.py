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
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import EXPERTS, LAYERS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

HIDDEN = 64
HANDLE = 7
READY_STATE, FREE_STATE = 3, 0  # expert_residency_gpu's _READY and _FREE

_FORWARD = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
)


class _Forward:
    """Records (layer, slots, weights) and writes a zero partial."""

    def __init__(self):
        self.calls = []
        self.c = _FORWARD(self._run)

    def _run(self, layer, x, slots, weights, k, out, threads):
        self.calls.append((layer, [slots[i] for i in range(k)], [weights[i] for i in range(k)]))
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


def test_the_low_scored_ram_hit_goes_to_the_cpu_and_the_high_scored_one_takes_the_coldest_victim(tmp_path):
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
    from sglang.kernels.ops.moe.expert_residency_direct_gather import (
        direct_commit_gather,
        direct_gather_destinations,
    )
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    row, low, high = 0, 3, 7
    forward = _Forward()
    split = [0] * (lease.LANES + 1)
    split[2] = 1
    c = Chain(tmp_path, copy_engine=True, start=False)
    try:
        x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(forward.address, split, cores, x_rows, out_rows, threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        backend, plan, dev = c.backends[row], c.plans[row], c.dev

        # Both experts into the RAM tier through an eager gather, which the service serves without the copy engine.
        c.plan([low, high], row)
        c.gather(row)
        torch.cuda.synchronize()
        assert c.handled()
        assert {low, high} <= c.resident(row)
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

        assert forward.calls == [(HANDLE, [ram_slot[low]], [0.25])], "the tail lane (the low key) is the CPU's"
        assert c.kinds(2) == [LaneKind.HIT_COPY, LaneKind.HIT_CPU]
        assert int(dev.cpu_lanes.item()) == 0b10
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
