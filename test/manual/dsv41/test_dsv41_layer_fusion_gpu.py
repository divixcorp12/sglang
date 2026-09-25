"""SGLANG_DSV41_ENABLE_LAYER_FUSION's three kernels against the torch chains they replace, bit for bit.

The references are the production methods themselves (``GpuResidencyUpdater.gather_destinations`` and
``commit_gather``, ``exl3_fused_moe.route_tables``), run on a clone of the same state. Every op in those chains is
integer bookkeeping or an exact conversion, so the comparison is exact equality of every output and every piece of
residency state, dump columns included. Shapes cover the production one (six routes, six lanes) and odd ones.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

READY, FREE = 3, 0


def _updater(width: int, experts: int, capacity: int, fused: bool, state: dict, streamer):
    from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

    u = object.__new__(GpuResidencyUpdater)
    u.num_layers, u.miss_rows, u.num_experts, u.max_capacity = 1, width, experts, capacity
    u.device = torch.device("cuda")
    u.streamers = [streamer]
    u.gather_lanes = torch.arange(width, dtype=torch.long, device="cuda")
    u._pending_commit = None
    for name, value in state.items():
        setattr(u, name, value.clone())
    u.layer_fusion = fused
    if fused:
        u._init_layer_fusion()
    return u


def _scenario(gen: torch.Generator, width: int, routes: int, experts: int, capacity: int):
    """A consistent residency row, a shortlist, this forward's routes and the planner's outputs."""
    dev = "cuda"

    def randint(lo, hi, n=()):
        return torch.randint(lo, hi, n if isinstance(n, tuple) else (n,), generator=gen)

    residents = int(randint(width, capacity + 1))
    expert_perm = torch.randperm(experts, generator=gen)
    slot_perm = torch.randperm(capacity, generator=gen)
    mapping = torch.full((experts + 1,), -1, dtype=torch.long)
    slot_to_expert = torch.full((capacity + 1,), -1, dtype=torch.long)
    slot_state = torch.zeros(capacity + 1, dtype=torch.uint8)
    for e, s in zip(expert_perm[:residents].tolist(), slot_perm[:residents].tolist()):
        mapping[e], slot_to_expert[s], slot_state[s] = s, e, READY
    # Dump columns hold garbage, so a wrong final dump value cannot pass by coincidence.
    mapping[experts] = int(randint(-5, 50))
    slot_to_expert[capacity] = int(randint(-5, 50))
    slot_state[capacity] = int(randint(0, 5))
    slot_generations = randint(0, 1000, capacity + 1)
    victims = torch.randperm(capacity, generator=gen)[:width]
    valid = torch.rand(width, generator=gen) < 0.85
    victims = torch.where(valid, victims, torch.full_like(victims, capacity))  # _rank_victims clamps invalid ones

    # Routes: distinct experts, mostly hits; misses first in the planner's source rows.
    hits = int(randint(0, routes + 1))
    resident_experts = expert_perm[:residents][torch.randperm(residents, generator=gen)[: min(hits, residents)]]
    others = expert_perm[residents:][torch.randperm(experts - residents, generator=gen)[: routes - resident_experts.numel()]]
    flat = torch.cat([resident_experts, others])[torch.randperm(routes, generator=gen)]
    slots = mapping[flat]
    miss = slots < 0
    rank = torch.cumsum(miss.long(), 0) - miss.long()
    scratch_base = capacity
    remap = torch.where(miss, scratch_base + rank, slots)
    if bool(torch.rand(1, generator=gen) < 0.2):
        remap = randint(0, scratch_base + width + 3, routes)  # out-of-range ranks exercise the clamp
    # Planner order (misses, then hits), padded with distinct experts no route names.
    unrouted = expert_perm[~torch.isin(expert_perm, flat)]
    source_rows = torch.cat([flat[miss], flat[~miss], unrouted[: width - routes]])
    misses = int(miss.sum())
    miss_count = misses if bool(torch.rand(1, generator=gen) < 0.8) else int(randint(0, width + 1))
    delivered = miss_count if bool(torch.rand(1, generator=gen) < 0.7) else int(randint(0, width + 1))
    keep = 1.0 if bool(torch.rand(1, generator=gen) < 0.8) else 0.0
    state = {
        "mapping": mapping.view(1, -1),
        "slot_to_expert": slot_to_expert.view(1, -1),
        "slot_state": slot_state.view(1, -1),
        "slot_generations": slot_generations.view(1, -1),
        "gather_insertions": randint(0, 9, 1),
        "gather_evictions": randint(0, 9, 1),
        "insertion_truncated": randint(0, 9, 1),
        "victims": victims.view(1, -1),
        "victim_valid": valid.view(1, -1),
    }
    return (
        {k: v.to(dev) for k, v in state.items()},
        flat.to(dev),
        remap.to(dev),
        scratch_base,
        source_rows.to(dev),
        miss_count,
        delivered,
        keep,
    )


def _streamer(source_rows, miss_count, delivered, keep, leased: bool, width: int):
    backend = (
        SimpleNamespace(
            name="exl3_ram_miss",
            delivered_count=torch.tensor([delivered], dtype=torch.int32, device="cuda"),
            keep=torch.tensor([keep], dtype=torch.float32, device="cuda"),
        )
        if leased
        else SimpleNamespace(name="in_graph")
    )
    return SimpleNamespace(
        _graph_miss_count=torch.tensor([miss_count], dtype=torch.int32, device="cuda"),
        _graph_source_rows=source_rows.clone(),
        _graph_destination_slots=torch.full((width,), -7, dtype=torch.int32, device="cuda"),
        row_backend=backend,
    )


STATE = ("mapping", "slot_to_expert", "slot_state", "slot_generations", "gather_insertions", "gather_evictions",
         "insertion_truncated")


@pytest.mark.parametrize(
    "width,routes,experts,capacity",
    [(6, 6, 256, 40), (6, 6, 257, 12), (1, 1, 9, 2), (7, 5, 33, 15), (13, 13, 101, 29), (32, 32, 400, 64)],
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("leased", [True, False])
def test_gather_and_commit_match_the_torch_chain(width, routes, experts, capacity, id_dtype, leased):
    gen = torch.Generator().manual_seed(width * 1000 + routes * 7 + capacity + (id_dtype == torch.int32))
    for trial in range(60):
        state, flat, remap, base, source_rows, miss_count, delivered, keep = _scenario(
            gen, width, routes, experts, capacity
        )
        flat, remap = flat.to(id_dtype), remap.to(id_dtype)
        ref_streamer = _streamer(source_rows, miss_count, delivered, keep, leased, width)
        fused_streamer = _streamer(source_rows, miss_count, delivered, keep, leased, width)
        ref = _updater(width, experts, capacity, False, state, ref_streamer)
        fused = _updater(width, experts, capacity, True, state, fused_streamer)
        expert_to_slot = state["mapping"][0, :experts].contiguous()

        want = ref.gather_destinations(0, remap, expert_to_slot.index_select(0, flat.long()), base)
        _, _, want_dest, want_live = ref._pending_commit
        for out_dtype in (id_dtype, torch.int64):
            got = fused.fused_gather_destinations(0, remap, flat, expert_to_slot, base, out_dtype)
            assert got.dtype == out_dtype
            assert torch.equal(got.long(), want.long()), f"trial {trial}: remap"
        _, _, got_dest, got_live = fused._pending_commit
        assert torch.equal(got_dest, want_dest), f"trial {trial}: destinations"
        assert torch.equal(got_live, want_live), f"trial {trial}: live"
        assert torch.equal(fused_streamer._graph_destination_slots, ref_streamer._graph_destination_slots)

        ref.commit_gather()
        fused.commit_gather()
        for name in STATE:
            assert torch.equal(getattr(fused, name), getattr(ref, name)), f"trial {trial}: {name}"


def test_captured_gather_and_commit_replay_new_inputs():
    """Both kernels inside a CUDA graph read their inputs on the device: a replay after the inputs change matches the
    torch chain run eagerly on the new inputs."""
    width, routes, experts, capacity = 6, 6, 256, 40
    gen = torch.Generator().manual_seed(7)
    state, flat, remap, base, source_rows, miss_count, delivered, keep = _scenario(gen, width, routes, experts, capacity)
    streamer = _streamer(source_rows, miss_count, delivered, keep, True, width)
    fused = _updater(width, experts, capacity, True, state, streamer)
    flat, remap = flat.to(torch.int32).clone(), remap.to(torch.int32).clone()
    expert_to_slot = fused.mapping[0, :experts]

    def step():
        fused.fused_gather_destinations(0, remap, flat, expert_to_slot, base, torch.int32)
        fused.commit_gather()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # compile the JIT modules outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
    torch.cuda.synchronize()
    for trial in range(20):
        state, new_flat, new_remap, _, new_rows, miss_count, delivered, keep = _scenario(
            gen, width, routes, experts, capacity
        )
        ref_streamer = _streamer(new_rows, miss_count, delivered, keep, True, width)
        ref = _updater(width, experts, capacity, False, state, ref_streamer)
        want = ref.gather_destinations(0, new_remap, state["mapping"][0, new_flat], base)
        ref.commit_gather()
        for name in (*STATE, "victims", "victim_valid"):
            getattr(fused, name).copy_(state[name])
        flat.copy_(new_flat)
        remap.copy_(new_remap)
        streamer._graph_source_rows.copy_(new_rows)
        streamer._graph_miss_count.fill_(miss_count)
        streamer.row_backend.delivered_count.fill_(delivered)
        streamer.row_backend.keep.fill_(keep)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(fused.fused_remaps[torch.int32][0].long(), want.long()), f"trial {trial}: remap"
        for name in STATE:
            assert torch.equal(getattr(fused, name), getattr(ref, name)), f"trial {trial}: {name}"


@pytest.mark.parametrize(
    "routes,slots,hidden",
    [(6, 46, 5120), (6, 12, 4096), (1, 3, 7), (5, 37, 1000), (13, 261, 5121), (32, 1025, 33)],
)
@pytest.mark.parametrize("remap_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_route_tables_match_the_torch_chain(routes, slots, hidden, remap_dtype, weight_dtype, x_dtype):
    from sglang.kernels.ops.moe.dsv41_layer_fusion import exl3_moe_route_tables
    from sglang.srt.layers.quantization.exl3_fused_moe import route_tables

    gen = torch.Generator().manual_seed(routes * 31 + slots + hidden)
    dev = "cuda"
    for trial in range(10):
        remap = torch.randperm(slots, generator=gen)[:routes].to(dev, remap_dtype)
        weights = (torch.rand(routes, generator=gen) * 3).to(dev, weight_dtype)
        keep = torch.tensor([[1.0, 0.0, 0.75][trial % 3]], device=dev)
        x = (torch.randn(1, hidden, generator=gen) * 40).to(dev, x_dtype)

        count_ref = torch.full((slots + 1,), 99, dtype=torch.int64, device=dev)
        ones = torch.ones(routes, dtype=torch.int64, device=dev)
        inv_ref, ws_ref, det_ref = route_tables(remap.long(), count_ref, ones, weights, keep)
        x16_ref = torch.empty(1, hidden, dtype=torch.float16, device=dev).copy_(x)

        remap64 = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        x16 = torch.full((1, hidden), 7.0, dtype=torch.float16, device=dev)
        out = torch.full((1, hidden), 5.0, dtype=torch.float32, device=dev)
        count = torch.full((slots + 1,), 99, dtype=torch.int64, device=dev)
        inv = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        ws = torch.full((routes,), 9.0, dtype=torch.float16, device=dev)
        det = torch.full((3, slots + 1), -3, dtype=torch.int64, device=dev)
        exl3_moe_route_tables(remap, weights, keep, x, remap64, x16, out, count, inv, ws, det)

        assert torch.equal(remap64, remap.long())
        assert torch.equal(x16.view(torch.int16), x16_ref.view(torch.int16)), f"trial {trial}: x16"
        assert torch.equal(out, torch.zeros_like(out))
        assert torch.equal(count, count_ref), f"trial {trial}: expert_count"
        assert torch.equal(inv, inv_ref), f"trial {trial}: inv_order"
        assert torch.equal(ws.view(torch.int16), ws_ref.view(torch.int16)), f"trial {trial}: weight_sorted"
        assert torch.equal(det, det_ref), f"trial {trial}: det"
