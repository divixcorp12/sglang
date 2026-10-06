"""DIRECT's kernels at a verify's width (33-64 lanes): the 64-thread destinations kernel equals the torch chain
(GpuResidencyUpdater.gather_destinations) on random shortlists, and the commit leaves out CPU lanes named by a 3-word
cpu_lanes, high word included (GPU, plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 2)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
WIDTH, SLOTS, EXPERTS = 36, 64, 96


def _chain(route_slots, victims, valid, miss_count, remap, base):
    """The torch chain of GpuResidencyUpdater.gather_destinations, without spill."""
    hazard = (route_slots.unsqueeze(1) == victims.unsqueeze(0)).any(dim=0)
    order = torch.argsort((hazard | ~valid).to(torch.uint8), stable=True)
    usable, usable_valid = victims.index_select(0, order), (valid & ~hazard).index_select(0, order)
    live = (torch.arange(victims.numel(), device="cuda") < miss_count) & usable_valid
    destinations = torch.where(live, usable, torch.zeros_like(usable))
    rank = (remap - base).clamp(min=0, max=victims.numel() - 1)
    return destinations, live, torch.where(remap >= base, destinations.index_select(0, rank), remap)


def test_the_wide_destinations_kernel_is_the_torch_chain():
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations

    gen = torch.Generator().manual_seed(36)
    for trial in range(200):
        ids = torch.randint(0, EXPERTS, (WIDTH,), generator=gen).cuda()
        expert_to_slot = torch.where(torch.rand(EXPERTS, generator=gen) < 0.4,
                                     torch.randint(0, SLOTS, (EXPERTS,), generator=gen), torch.full((EXPERTS,), -1)).cuda()
        victims = torch.randperm(SLOTS, generator=gen)[:WIDTH].cuda()
        valid = (torch.rand(WIDTH, generator=gen) < 0.8).cuda()
        miss_count = torch.randint(0, WIDTH + 1, (1,), generator=gen, dtype=torch.int32).cuda()
        remap = torch.where(torch.rand(WIDTH, generator=gen) < 0.5, torch.randint(0, SLOTS, (WIDTH,), generator=gen),
                            SLOTS + torch.randint(0, WIDTH, (WIDTH,), generator=gen)).cuda()
        want = _chain(expert_to_slot[ids], victims, valid, miss_count, remap, SLOTS)
        slots_out = torch.zeros(WIDTH, dtype=torch.int32, device="cuda")
        dest_out = torch.zeros(WIDTH, dtype=torch.int64, device="cuda")
        live_out = torch.zeros(WIDTH, dtype=torch.bool, device="cuda")
        remap_out = torch.zeros(WIDTH, dtype=torch.int64, device="cuda")
        direct_gather_destinations(ids, expert_to_slot, victims, valid, miss_count, remap, SLOTS, slots_out, dest_out,
                                   live_out, remap_out)
        assert torch.equal(dest_out, want[0]) and torch.equal(live_out, want[1]) and torch.equal(remap_out, want[2]), trial
        assert torch.equal(slots_out, want[0].to(torch.int32)), trial


def test_the_commit_leaves_out_cpu_lanes_past_32():
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_commit_gather

    cuda = dict(device="cuda")
    destinations = torch.arange(WIDTH, dtype=torch.int64, **cuda)
    live = torch.ones(WIDTH, dtype=torch.bool, **cuda)
    new_experts = torch.arange(10, 10 + WIDTH, dtype=torch.int64, **cuda)
    mapping = torch.full((EXPERTS + 1,), -1, dtype=torch.int64, **cuda)
    slot_to_expert = torch.full((SLOTS + 1,), -1, dtype=torch.int64, **cuda)
    slot_state = torch.zeros(SLOTS + 1, dtype=torch.uint8, **cuda)
    generations = torch.zeros(SLOTS + 1, dtype=torch.int64, **cuda)
    counters = [torch.zeros(1, dtype=torch.int64, **cuda) for _ in range(3)]
    cpu_lanes = torch.tensor([1, 1, 0b1010], dtype=torch.int32, **cuda)  # lanes 0, 33 and 35 are the CPU's
    direct_commit_gather(
        destinations, live, new_experts, mapping, slot_to_expert, slot_state, generations, *counters,
        torch.tensor([WIDTH], dtype=torch.int32, **cuda), torch.ones(1, **cuda), torch.tensor([WIDTH], dtype=torch.int32, **cuda),
        ready=3, free_state=0, cpu_lanes=cpu_lanes,
    )
    cpu = {0, 33, 35}
    assert [int(mapping[10 + j]) for j in range(WIDTH)] == [-1 if j in cpu else j for j in range(WIDTH)]
    assert int(counters[0].item()) == WIDTH - 3 and int(counters[2].item()) == 0  # insertions; nothing truncated
