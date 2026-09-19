"""End to end on the GPU (window): a captured EXL3 in-graph MoE whose RAM misses are
served by option C, against the eager streamed apply; and a forced timeout fail-stop."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 16, 6


def _layers(tmp_path, monkeypatch, timeout_ms=2000):
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
    monkeypatch.setenv("SGLANG_DSV41_RAM_MISS_TIMEOUT_MS", str(timeout_ms))
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
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
    return layer, streamer, service_module.Exl3RamMissService.get(), checks


def test_ram_misses_inside_a_replay_are_served(tmp_path, monkeypatch):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, monkeypatch)
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
            want = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), 10.0).float()
            rel = float((got - want).norm() / want.norm())
            assert rel <= 1.2e-2, (route, rel)
            assert streamer.row_backend.keep.item() == 1.0
            for check in checks:
                check()
        assert service.host.counters()["rows_read"] >= 6
    finally:
        service.shutdown()


def test_a_forced_timeout_fails_stop_without_hanging(tmp_path, monkeypatch):
    import time

    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, monkeypatch, timeout_ms=100)
    try:
        x = torch.zeros((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
        weights = torch.full((1, TOP_K), 1.0 / TOP_K, device="cuda")
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        service.host.inject(delay_s=10.0)
        ids.copy_(torch.tensor([[13, 14, 15, 0, 1, 2]], device="cuda", dtype=torch.int32))
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        assert time.perf_counter() - started < 2.0
        assert streamer.row_backend.keep.item() == 0.0
        with pytest.raises(RuntimeError, match="exl3 RAM miss"):
            for check in checks:
                check()
    finally:
        service.host.inject(delay_s=0.0)
        service.shutdown()
