"""The torch chains the EXL3 layer-fusion kernels replace, kept as bit-identity references for the tests.

``route_tables`` and ``unfused_run`` are ``Exl3FusedMoE``'s route tables and ``run`` without the fused route-tables
kernel; ``gather_destinations`` and ``commit_gather`` are ``GpuResidencyUpdater``'s DIRECT gather and commit without
the fused residency kernels. Each kernel must match its chain bit for bit.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.moe.expert_residency_gpu import _FREE, _READY


def route_tables(remap, expert_count, ones, weights, keep, token_sorted=None, top_k=1):
    """Fill ``expert_count`` from ``remap``; return (inv_order, weight_sorted fp16, det tables).

    Routes are ranked by slot, ties by route index (a stable sort: several tokens may route one slot); with
    ``token_sorted`` given, each rank's token, ``route // top_k``, is written into it.
    ``keep`` (fp32 [1]) scales every route weight and, when 0, empties ``expert_count``,
    so a dropped layer runs no expert.
    ``det`` is exllamav3's device-built deterministic table stack
    ``[expert_start, expert_start, count > 0]``.
    """
    expert_count.zero_().index_add_(0, remap, ones)
    order = torch.argsort(remap, stable=True)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(order.numel(), device=order.device)
    )
    if token_sorted is not None:
        torch.floor_divide(order, top_k, out=token_sorted)
    weight_sorted = (weights[order].float() * keep).to(torch.float16)
    # A dropped layer runs no expert: nothing reads rows that may be half written.
    expert_count.mul_((keep > 0).to(torch.int64))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    det = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
    return inv_order, weight_sorted, det


def route_tables_kernel(
    remap,
    weights,
    keep,
    x,
    remap64_out,
    x16_out,
    out_zero,
    expert_count,
    inv_order,
    weight_sorted,
    det,
    cpu_lanes=None,
    dst_slots=None,
    cpu_out=0,
    cpu_part_stride=0,
    token_sorted_out=None,
) -> None:
    """``exl3_moe_route_tables`` as torch ops, for tests that run ``Exl3FusedMoE`` on the CPU; no CPU-expert lanes."""
    if cpu_lanes is not None:
        raise NotImplementedError("the torch route tables have no CPU-expert lanes")
    remap64_out.copy_(remap)
    x16_out.copy_(x)
    out_zero.zero_()
    routes = remap.numel()
    ones = torch.ones(routes, dtype=torch.int64, device=remap.device)
    inv, ws, d = route_tables(
        remap64_out,
        expert_count,
        ones,
        weights,
        keep,
        token_sorted=token_sorted_out,
        top_k=routes // x.shape[0],
    )
    inv_order.copy_(inv)
    weight_sorted.copy_(ws)
    det.copy_(d)


def unfused_run(fused, x, topk_weights, remap, keep, act_limit: float) -> torch.Tensor:
    """``Exl3FusedMoE.run`` (no ``cpu``) with the torch route tables in place of the fused kernel; ``remap`` int64."""
    from sglang.srt.layers.quantization.exl3.fused_moe import (
        ACT_SILU,
        NUM_ACTIVE,
        ROW_TILE,
    )

    m = x.shape[0]
    if not 1 <= m <= fused.tokens:
        raise ValueError(f"exl3 in-graph MoE runs 1-{fused.tokens} tokens, not {m}")
    routes = m * fused.top_k
    x16, out = fused.x16[:m], fused.out[:m]
    token_sorted, scratch = fused.token_sorted[:routes], fused.scratch[:routes]
    x16.copy_(x)
    ones = torch.ones(routes, dtype=torch.int64, device=remap.device)
    inv_order, weight_sorted, det = route_tables(
        remap,
        fused.expert_count,
        ones,
        topk_weights,
        keep,
        token_sorted=token_sorted,
        top_k=fused.top_k,
    )
    out.zero_()
    t = fused.tables
    fused.ext.exl3_moe(
        x16,
        out,
        fused.expert_count,
        token_sorted,
        weight_sorted,
        fused.temp_state_g,
        fused.temp_state_u,
        fused.temp_intermediate_g,
        fused.temp_intermediate_u,
        ACT_SILU,
        fused.bits["gate"],
        fused.bits["up"],
        fused.bits["down"],
        t["gate_trellis"],
        t["gate_suh"],
        t["gate_svh"],
        t["up_trellis"],
        t["up_suh"],
        t["up_svh"],
        t["down_trellis"],
        t["down_suh"],
        t["down_svh"],
        False,
        True,
        False,
        True,
        False,
        True,
        float(act_limit),
        NUM_ACTIVE if m == 1 else -1,
        scratch,
        det[0],
        1,
        ROW_TILE,
        16,
    )
    fused.ext.exl3_moe_gather(
        out,
        scratch,
        remap,
        inv_order,
        det[1, : fused.slots],
        det[0, : fused.slots],
        det[2, : fused.slots],
        weight_sorted,
    )
    return out


def gather_destinations(updater, row: int, remap, route_slots, scratch_base: int) -> torch.Tensor:
    """``GpuResidencyUpdater.fused_gather_destinations`` as torch ops, given the routes' slots ``route_slots``.

    Writes the destinations into the gather's plan slots, records the pending commit, and returns the remap with
    every miss lane translated, in int64.
    """
    streamer = updater.streamers[row]
    victims, valid = updater.victims[row], updater.victim_valid[row]
    # A miss lane's slot is -1 and a padded column is clamped to the dump index, so
    # only genuine hits disqualify an entry.
    hazard = (route_slots.unsqueeze(1) == victims.unsqueeze(0)).any(dim=0)
    order = torch.argsort((hazard | ~valid).to(torch.uint8), stable=True)
    usable = victims.index_select(0, order)
    usable_valid = (valid & ~hazard).index_select(0, order)
    live = (updater.gather_lanes < streamer._graph_miss_count.long()) & usable_valid
    idle, idle_slot = updater._idle_destination()
    destinations = torch.where(live, usable, torch.full_like(usable, idle))
    # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
    streamer._graph_destination_slots[: updater.miss_rows].copy_(
        torch.where(live, destinations, torch.full_like(destinations, idle_slot)).to(torch.int32)
    )
    updater._pending_commit = (row, streamer, destinations, live)
    rank = (remap - scratch_base).clamp(min=0, max=updater.miss_rows - 1).long()
    return torch.where(remap >= scratch_base, destinations.index_select(0, rank), remap)


def commit_gather(updater) -> None:
    """``GpuResidencyUpdater.commit_gather`` as torch ops, for backends without CPU experts."""
    row, streamer, destinations, live = updater._pending_commit
    updater._pending_commit = None
    backend = streamer.row_backend
    if backend.leased:
        delivered = backend.delivered_count.long()
        live = live & (updater.gather_lanes < delivered) & (backend.keep[0] > 0)
    commit_live(updater, row, streamer, destinations, live)


def commit_live(updater, row: int, streamer, destinations, live) -> None:
    """The commit chain for an already narrowed ``live``: residency moves onto the rows the gather copied."""
    backend = streamer.row_backend
    experts, slot_dump = updater.num_experts, updater.max_capacity
    new_experts = streamer._graph_source_rows[: updater.miss_rows]
    slots = updater.slot_to_expert[row]
    targets = torch.where(live, destinations, slot_dump)
    evicted = live & (slots.gather(0, destinations) >= 0)
    mapping = updater.mapping[row]
    mapping.scatter_(0, torch.where(evicted, slots.gather(0, destinations), experts), -1)
    mapping.scatter_(0, torch.where(live, new_experts, experts), targets)
    slots.scatter_(0, targets, torch.where(live, new_experts, -1))
    updater.slot_state[row].scatter_(0, targets, _READY)
    updater.slot_generations[row].scatter_add_(0, targets, live.to(torch.long))
    updater.slot_state[row, slot_dump : slot_dump + 1].fill_(_FREE)
    slots[slot_dump : slot_dump + 1].fill_(-1)
    updater.slot_generations[row, slot_dump : slot_dump + 1].fill_(0)
    updater.gather_insertions[row].add_(live.sum())
    updater.gather_evictions[row].add_(evicted.sum())
    if backend.leased:
        delivered = backend.delivered_count.long()
        good = backend.keep[0] > 0
        updater.insertion_truncated[row].add_(((delivered > live.sum()) & good).long().sum())
    else:
        updater.insertion_truncated[row].add_(
            (streamer._graph_miss_count.long() > live.sum()).long().sum()
        )


def install_torch_chains(updater) -> None:
    """Make ``updater`` (one instance) run the torch chains in place of its fused gather and commit kernels."""

    def fused_gather_destinations(row, remap, flat, expert_to_slot, scratch_base, remap_dtype):
        route_slots = expert_to_slot.index_select(0, flat.long())
        return gather_destinations(updater, row, remap, route_slots, scratch_base).to(remap_dtype)

    updater.fused_gather_destinations = fused_gather_destinations
    updater.commit_gather = lambda: commit_gather(updater)
