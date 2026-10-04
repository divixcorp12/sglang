"""End to end on the GPU (window): a captured EXL3 in-graph MoE whose RAM misses are
served by the lease chain.

The served replay is checked as Task 9's test_exl3_graph_apply_gpu.py checks P3: every
slot the gather names holds exactly the bytes a fresh checkpoint read of its expert
gives, the output meets the probe's bars (D7) against an fp32 reference over those
rows, and graph vs the eager streamed apply (exl3_moe_loop, the less accurate arm at
~1.5e-2 on these fake rows) stays within Task 9's 2.5e-2.
"""

import sys
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 16, 6
ACT_LIMIT = 10.0
REL_BOUND = 1.2e-2  # the probe's bar (D7), against the fp32 reference
LOOSE_BOUND = 2.5e-2  # graph vs loop (Task 9)


def _source_rows(tmp_path, layer_id=0, num_experts=EXPERTS):
    """Every expert's streamed rows of one layer, read afresh from the checkpoint."""
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, layer_id, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    source = {name: torch.empty((num_experts,) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    Exl3ShardRowSource.for_layer(layout, layer_id, fmt.segment_map(), direct=False).read(
        torch.arange(num_experts, dtype=torch.long), source
    )
    return source


def _reference(x, weights, slots, tensors):
    """fp32 routed output over the hot-cache rows at ``slots`` (the probe's reference)."""
    import torch.nn.functional as F

    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_linear_reference

    def view(prefix, slot, part):
        return Exl3Tensors(
            trellis=tensors[f"{prefix}_trellis"][slot, part],
            suh=tensors[f"{prefix}_suh"][slot, part],
            svh=tensors[f"{prefix}_svh"][slot, part],
            mul1=True,
        )

    x16 = x.to(torch.float16)
    out = torch.zeros((1, x.shape[1]), dtype=torch.float32, device=x.device)
    for k, slot in enumerate(slots.tolist()):
        gate = exl3_linear_reference(x16, view("w13", slot, 0)).clamp(max=ACT_LIMIT)
        up = exl3_linear_reference(x16, view("w13", slot, 1)).clamp(-ACT_LIMIT, ACT_LIMIT)
        h = F.silu(gate) * up * weights[k].float()
        out += exl3_linear_reference(h.to(torch.float16), view("w2", slot, 0))
    return out


def _rel(y, ref):
    return float((y.float() - ref.float()).norm() / ref.float().norm())


def _layers(tmp_path, timeout_ms=2000, num_layers=1):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import DirectUpdaterStandIn, service_row_images

    write_fake_exl3(str(tmp_path), num_layers=num_layers, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    # The staging slots the service reserves at start: the graph's misses, and the rest of the tier is mappable.
    service_module.Exl3RamMissService.get().plan_gather_width(TOP_K)
    with (
        # The service reads row images with O_DIRECT, in lease mode (always on), as in production.
        service_row_images(tmp_path),
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
        envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
        # Read when attach builds the device side, so attach runs inside this block.
        envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(timeout_ms),
    ):
        pairs, hots = [], []
        # Every pinned tier registers before the service starts, and a tier built after it is refused.
        for layer_id in range(num_layers):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            layer.top_k = TOP_K
            # No ``direct``: SGLANG_MOE_EXPERT_FILE_READER=uring_direct decides it, as in production.
            fmt = Exl3ExpertFormat(layout, layer_id, source_root=str(tmp_path))
            streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            # Eight mappable rows plus the TOP_K staging slots the service reserves for the graph's misses.
            ExpertPinnedHostCache(streamer, 8 + TOP_K, **fmt.pinned_tier_options(layer))
            pairs.append((layer, streamer))
        for layer, streamer in pairs:
            hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
            hot.reassign([0, 1, 2])
            streamer.enable_graph_gather(TOP_K)
            hots.append(hot)
        checks = []

        # The hot sets are static here: the DIRECT updater's bank, whose rows every post writes into its record.
        updater = DirectUpdaterStandIn(num_layers, 3, EXPERTS, device="cuda")
        for layer_id, hot in enumerate(hots):
            updater.slot_to_expert[layer_id, :3] = torch.tensor(hot.slot_to_expert, dtype=torch.int64, device="cuda")
        manager = type(
            "M", (), {"register_fail_stop_check": lambda self, f: checks.append(f), "gpu_residency": updater}
        )()
        for _, streamer in pairs:
            streamer.format.attach_hot_cache_manager(manager, streamer)
    service = service_module.Exl3RamMissService.get()
    assert service.device_side.lease_block is not None
    if num_layers == 1:
        return (*pairs[0], service, checks)  # the shape every single-layer test unpacks
    return pairs, service, checks


# The demand ring and the lease lanes are 16 deep (Wire::kDemandRecords): 4 layers fit, and 20 wrap them
# inside one replay, as the 40+ streamed layers of a real decode step do.
LAYER_COUNTS = pytest.mark.parametrize("layers", [4, 20], ids=["layers_4", "layers_20"])
REPLAY_STEPS = 4


@pytest.mark.parametrize("fused", [False, True], ids=["generic_routes", "fused_routes"])
def test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff(tmp_path, fused):
    """Runs only on divix01: actual EXL3 bytes, the native slot-map chain, and captured
    MoE output across a DIRECT insertion and an eager pinned-tier eviction."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import paused, service_row_images

    experts = 36
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=experts, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=experts)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    service.plan_gather_width(TOP_K)
    try:
        with (
            service_row_images(tmp_path),  # the service reads row images with O_DIRECT, as in production
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(fused),
        ):
            model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))  # direct from the reader env
            # A tiny synthetic checkpoint uses its routed width as the eager
            # staging bound; production leaves the format's 64-row bound.
            fmt.max_gather_rows = TOP_K
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, 3 * TOP_K, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=2 * TOP_K * streamer.bytes_per_expert,
                seed_path=None, dynamic=True, update_prefill_tokens=16,
                min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=1, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            assert streamer._fused_plan_enabled is fused
            assert (streamer._graph_fused_slots_scratch is not None) is fused
            assert (streamer._graph_fused_remaps is not None) is fused
            assert isinstance(streamer.row_backend, service_module.Exl3RamMissRowBackend)
            assert service.gpu_hot_enabled
            assert manager.caches[0].scratch_rows == 0
            assert manager.caches[0].capacity == 2 * TOP_K
            generator = torch.Generator(device="cpu").manual_seed(31)
            x = (torch.randn((1, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.full((1, TOP_K), 1.0 / TOP_K, device="cuda")
            ids = torch.tensor([list(range(12, 18))], device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()
            assert manager.gpu_residency.victims_fresh

            def replay(route):
                ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert streamer.row_backend.keep.item() == 1.0
                mapping = manager.gpu_residency.mapping[0, :experts].cpu()
                slots = [int(mapping[e]) for e in route]
                assert all(slot >= 0 for slot in slots)
                for expert, slot in zip(route, slots):
                    for name, rows in manager.caches[0].tensors.items():
                        assert torch.equal(rows[slot].cpu(), source[name][expert]), (expert, slot, name)
                ref = _reference(x, weights.reshape(-1), torch.tensor(slots), manager.caches[0].tensors)
                assert torch.isfinite(ref).all()
                assert ref.float().norm().item() > 0
                assert _rel(out, ref) <= REL_BOUND

            first = list(range(12, 18))
            replay(first)  # miss and direct copy
            delivered = service.host.counters()["rows_read"]
            replay(first)  # next replay is all GPU hits
            assert service.host.counters()["rows_read"] == delivered
            # The fused planner is a unique-ID path because production top-k
            # returns distinct experts; the generic planner covers duplicates.
            if not fused:
                replay([12, 12, 13, 13, 14, 14])  # duplicate routes remain hits
            top_slot = int(manager.gpu_residency.victims[0, 0].item())
            top_expert = int(manager.gpu_residency.slot_to_expert[0, top_slot].item())
            replay([top_expert, 18, 19, 20, 21, 22])  # routed hit is the top-ranked victim
            # The final insertion must remain in pinned RAM when eager prefill
            # admits cold rows before another same-layer graph post.
            protected = 19
            streamer.pinned_host_cache.ensure_rows(torch.tensor([30, 31, 32, 33, 34, 35]))
            with paused(service.host):
                assert protected in service.host.slot_to_expert(0)
            replay(
                list(range(24, 30)) if fused else [24, 24, 25, 25, 26, 26]
            )  # generic duplicate misses copy once per unique expert
            for start in (18, 24, 30):
                replay(list(range(start, start + TOP_K)))
            mapping = manager.gpu_residency.mapping[0, :experts].cpu()
            evicted = next(expert for expert in first if mapping[expert] < 0)
            replay([evicted, 30, 31, 32, 33, 34])  # refetch into GPU residency
            assert manager.gpu_residency.insertion_truncated[0].item() == 0
    finally:
        service.shutdown()


def test_ram_misses_inside_a_replay_are_served(tmp_path):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path)
    source = _source_rows(tmp_path)
    try:
        gen = torch.Generator(device="cpu").manual_seed(3)
        x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
        ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        for route in ([9, 10, 11, 0, 1, 12], [13, 14, 15, 2, 9, 4]):  # misses beyond the 8 pinned rows
            ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            got = out.float().clone()
            assert streamer.row_backend.keep.item() == 1.0
            assert streamer.row_backend.ram_miss.item() == 0
            for check in checks:
                check()
            # An eager call of the same routes repeats the replay bit for bit, so its gather
            # names the replay's slots; they hold the rows option C read, byte for byte.
            assert torch.equal(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT).float(), got)
            remap, tensors = streamer.gather(ids)
            slots = remap.reshape(-1).tolist()
            for k, expert in enumerate(route):
                for name, rows in tensors.items():
                    assert torch.equal(rows[slots[k]].cpu(), source[name][expert]), (route, k, expert, name)
            ref = _reference(x, weights.reshape(-1), remap.reshape(-1), tensors)
            loop = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), ACT_LIMIT)
            rel, rel_loop, rel_graph_loop = _rel(got, ref), _rel(loop, ref), _rel(got, loop)
            print(f"route {route}: rel {rel:.3e} rel_loop {rel_loop:.3e} graph_vs_loop {rel_graph_loop:.3e}")
            assert rel <= REL_BOUND and rel <= 2 * rel_loop + 1e-3, (route, rel, rel_loop)
            assert rel_graph_loop <= LOOSE_BOUND, (route, rel_graph_loop)
        assert service.host.counters()["rows_read"] >= 6
        # The service handled every record the replays posted, and none was lapped.
        assert _handled_all(service), service.host.counters()
        assert service.host.counters()["overruns"] == 0, service.host.counters()
    finally:
        service.shutdown()


def _handled_all(service, timeout_s=10.0):
    """The service has finished every record the device posted (handled_through reaches the device's posted count)."""
    posted = int(service.device_side.stats()["posted"]) & 0xFFFFFFFF
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if posted > 0 and service.host.handled_through() == posted:
            return True
        time.sleep(0.005)
    return False


def _step_route(layer_id, step):
    """Two VRAM-hot experts (0, 1) and four from the 13 others, walked around a cycle 4 further each step.

    The pinned tier holds 8 rows: 3 hot and 5 evictable. The 4 experts of a step are never among the 5 most
    recently loaded, so every layer misses RAM at every step, and the hot rows plus the route (7) fit the tier.
    """
    others = [3 + (4 * step + 3 * layer_id + i) % (EXPERTS - 3) for i in range(4)]
    return [others[0], 0, others[1], others[2], 1, others[3]]


def _capture_layers(pairs, seed):
    """Per-layer inputs; one eager pass, then every layer's ``_apply_graph`` captured into ONE graph in layer order."""
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    gen = torch.Generator(device="cpu").manual_seed(seed)
    inputs = []
    for layer_id in range(len(pairs)):
        x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
        ids = torch.tensor([_step_route(layer_id, -1)], device="cuda", dtype=torch.int32)
        inputs.append((x, weights, ids))

    def run():
        return [
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
            for (layer, streamer), (x, weights, ids) in zip(pairs, inputs)
        ]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    return inputs, graph, outs


def _replay_step(pairs, service, inputs, graph, step, checks):
    """Replay every layer on routes that miss its pinned tier; each layer's misses must be read and none dropped."""
    expected = []
    for layer_id, (_, streamer) in enumerate(pairs):
        # The published map, read without pausing: between replays every row the tier holds is READY.
        mapping = service.host.mapping(service.row_of(streamer.layer_id))
        hot, tier = streamer.hot_cache.slot_to_expert, [e for e, slot in enumerate(mapping) if slot >= 0]
        missing = [e for e in _step_route(layer_id, step) if e not in hot and e not in tier]
        # A layer with nothing to read would post an all-hit request and leave the RAM-miss path unexercised.
        assert missing, (layer_id, step)
        expected.append(len(missing))
    before = service.host.layer_rows()
    for layer_id, (_, _, ids) in enumerate(inputs):
        ids.copy_(torch.tensor([_step_route(layer_id, step)], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    for layer_id, (_, streamer) in enumerate(pairs):
        assert streamer.row_backend.keep.item() == 1.0, (layer_id, step, service.host.counters())
        assert streamer.row_backend.ram_miss.item() == 0, (layer_id, step)
    for check in checks:
        check()

    def served():
        rows = service.host.layer_rows()
        return [rows[service.row_of(n)] - before[service.row_of(n)] for n in range(len(pairs))]

    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline and served() != expected:
        time.sleep(0.005)
    assert served() == expected, (step, served(), expected, service.host.counters())


def _assert_device_map_is_hosts(service, layers):
    """The device's replica of every row's slot map and staging list equals the host's, once the last record's delta
    and the eager paths' bulk delta are applied (what the next post and after_host_use would do)."""
    torch.cuda.synchronize()
    service.host.pause(10.0)
    try:
        bulk = service.host.take_bulk_delta()
        service.device_side.map_bulk_apply(bulk)  # every row's pending decode delta first, then the bulk entries
        torch.cuda.synchronize()
        bank = service.device_side.map_bank
        for layer_id in range(layers):
            row = service.row_of(layer_id)
            assert bank["ram_slot"][row].tolist() == service.host.mapping(row), layer_id
            staging = {slot for slot, (state, _, _) in enumerate(service.host.slot_info(row)) if state == 3}
            assert {s for s in bank["staging"][row].tolist() if s >= 0} == staging, layer_id
    finally:
        service.host.resume()


def _assert_service_healthy(service):
    counters = service.host.counters()
    for name in ("overruns", "read_errors", "no_victim"):
        assert counters[name] == 0, (name, counters)


@LAYER_COUNTS
def test_many_layers_in_one_replay_are_served(tmp_path, layers):
    """Every layer of one graph posts its own RAM-miss request per replay, and each layer's output is right."""
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    pairs, service, checks = _layers(tmp_path, num_layers=layers)
    sources = [_source_rows(tmp_path, layer_id) for layer_id in range(layers)]
    try:
        inputs, graph, outs = _capture_layers(pairs, seed=5)
        for step in range(REPLAY_STEPS):
            _replay_step(pairs, service, inputs, graph, step, checks)
            for layer_id, ((layer, streamer), (x, weights, ids), out) in enumerate(zip(pairs, inputs, outs)):
                route = _step_route(layer_id, step)
                got = out.float().clone()
                # As in the single-layer test: an eager call repeats the replay bit for bit, so its gather names
                # the replay's slots. Here they must also hold this layer's rows, not a neighbour's.
                assert torch.equal(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT).float(), got), (layer_id, step)
                remap, tensors = streamer.gather(ids)
                slots = remap.reshape(-1).tolist()
                for k, expert in enumerate(route):
                    for name, rows in tensors.items():
                        assert torch.equal(rows[slots[k]].cpu(), sources[layer_id][name][expert]), (layer_id, step, k, expert, name)
                ref = _reference(x, weights.reshape(-1), remap.reshape(-1), tensors)
                loop = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), ACT_LIMIT)
                rel, rel_loop, rel_graph_loop = _rel(got, ref), _rel(loop, ref), _rel(got, loop)
                assert rel <= REL_BOUND and rel <= 2 * rel_loop + 1e-3, (layer_id, step, route, rel, rel_loop)
                assert rel_graph_loop <= LOOSE_BOUND, (layer_id, step, route, rel_graph_loop)
            # Replays and the eager calls above (admissions, applied as a bulk delta) leave one map on both sides.
            assert _handled_all(service), service.host.counters()
            _assert_device_map_is_hosts(service, layers)
        assert _handled_all(service), service.host.counters()
        _assert_service_healthy(service)
    finally:
        service.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
