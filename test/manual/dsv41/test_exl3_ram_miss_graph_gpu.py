"""End to end on the GPU (window): a captured EXL3 in-graph MoE whose RAM misses are
served by option C; and a forced timeout fail-stop.

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


def _source_rows(tmp_path):
    """Every expert's streamed rows, read afresh from the checkpoint."""
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    source = {name: torch.empty((EXPERTS,) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    Exl3ShardRowSource.for_layer(layout, 0, fmt.segment_map(), direct=False).read(
        torch.arange(EXPERTS, dtype=torch.long), source
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


def _layers(tmp_path, timeout_ms=2000, lease=False):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    with (
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
        envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
        # Read when attach builds the device side, so attach runs inside this block.
        envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(timeout_ms),
        # Read once, when the service starts (ensure_started, inside attach): the switch under test.
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(lease),
    ):
        layer = torch.nn.Module()
        layer.layer_id = 0
        layer.top_k = TOP_K
        fmt = Exl3ExpertFormat(layout, 0, direct=False)
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
        layer._nvfp4_expert_streamer = streamer
        pinned = ExpertPinnedHostCache(streamer, 8, **fmt.pinned_tier_options(layer))
        hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
        hot.reassign([0, 1, 2])
        streamer.enable_graph_gather(TOP_K)
        checks = []
        manager = type("M", (), {"register_fail_stop_check": lambda self, f: checks.append(f), "add_residency_listener": lambda self, f: f(0, list(hot.slot_to_expert))})()
        fmt.attach_hot_cache_manager(manager, streamer)
    service = service_module.Exl3RamMissService.get()
    assert service.lease_mode is lease and (service.device_side.lease_block is not None) is lease
    return layer, streamer, service, checks


LEASES = pytest.mark.parametrize("lease", [False, True], ids=["leases_off", "leases_on"])


@LEASES
def test_ram_misses_inside_a_replay_are_served(tmp_path, lease):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, lease=lease)
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
        if lease:
            # The replays' copies were leased and acknowledged, none violated, and the service retired every one.
            def retired():
                c = service.host.counters()
                return c["leases_granted"] > 0 and c["leases_acked"] == c["leases_granted"]

            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline and not retired():
                time.sleep(0.005)
            counters = service.host.counters()
            assert retired(), counters
            assert counters["leases_voided"] == 0 and counters["lease_double_signal"] == 0, counters
            assert service.host.fatal_seq() == 0
            assert all(info[2] == 0 for info in service.host.slot_info(0)), "no slot is left leased"
        else:
            assert service.host.counters()["leases_granted"] == 0
    finally:
        service.shutdown()


@LEASES
def test_a_forced_timeout_fails_stop_without_hanging(tmp_path, lease):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, timeout_ms=100, lease=lease)
    try:
        assert service.device_side.timeout_ns == 100_000_000  # the override reached attach
        x = torch.zeros((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
        weights = torch.full((1, TOP_K), 1.0 / TOP_K, device="cuda")
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        service.host.inject(delay_s=10.0)
        ids.copy_(torch.tensor([[13, 14, 15, 0, 1, 2]], device="cuda", dtype=torch.int32))
        base = streamer.row_planner.scratch_base
        scratch_before = {name: t[base : base + TOP_K].clone() for name, t in streamer.hot_cache.tensors.items()}
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        assert time.perf_counter() - started < 2.0
        assert streamer.row_backend.keep.item() == 0.0
        if lease:
            # A refused request commits nothing: the copy read nothing (the scratch rows are untouched) and
            # no acknowledgement was emitted.
            assert service.device_side.go_count.item() == 0
            for name, t in streamer.hot_cache.tensors.items():
                assert torch.equal(t[base : base + TOP_K].view(torch.uint8), scratch_before[name].view(torch.uint8)), name
        with pytest.raises(RuntimeError, match="exl3 RAM miss"):
            for check in checks:
                check()
    finally:
        service.host.inject(delay_s=0.0)
        service.shutdown()


def _replayed_outputs(tmp_path, lease, routes):
    """Warm up, capture, replay each route; the output bytes of every replay, and the service's ack count."""
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, lease=lease)
    try:
        gen = torch.Generator(device="cpu").manual_seed(11)
        x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
        ids = torch.tensor([routes[0]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        outputs = []
        for route in routes:
            ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            assert streamer.row_backend.keep.item() == 1.0, (lease, route, service.host.counters(), service.host.fatal_seq())
            for check in checks:
                check()
            outputs.append(out.clone())
        counters = service.host.counters()
        return outputs, counters
    finally:
        service.shutdown()
        service_module = sys.modules["sglang.srt.layers.moe.exl3_ram_miss"]
        service_module.Exl3RamMissService._instance = None


def test_lease_mode_output_is_byte_exact_against_off(tmp_path):
    """Same weights, same routes (misses, hits, repeats): the leased chain returns the very same bytes."""
    routes = [
        [9, 10, 11, 0, 1, 12],
        [13, 14, 15, 2, 9, 4],
        [13, 14, 15, 2, 9, 4],  # every row now resident: the all-hit handshake
        [0, 1, 2, 9, 10, 11],
        [4, 9, 10, 12, 0, 1],  # at most four misses a call: the 8-row tier plus the hot rows cannot hold six
    ]
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off, off_counters = _replayed_outputs(tmp_path / "off", False, routes)
    on, on_counters = _replayed_outputs(tmp_path / "on", True, routes)
    for n, (a, b) in enumerate(zip(off, on)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"replay {n} differs"
    assert off_counters["leases_granted"] == 0
    assert on_counters["leases_granted"] > 0 and on_counters["leases_voided"] == 0, on_counters


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
