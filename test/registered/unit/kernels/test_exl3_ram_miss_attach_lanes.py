"""Attaching an in-graph layer refuses a gather wider than the lanes the post kernel requests (CPU)."""

import faulthandler
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import MAX_IDS
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, service_row_images

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    """The service reads row images (service_row_images) with O_DIRECT, as in production. The attached layers' copy
    tables are stubs, so their piece maps are too: what is under test is the gather width."""
    monkeypatch.setattr(module, "stream_segment_map", lambda segments, tables, row: None)
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=ROW_IMAGE_DIM, inter=ROW_IMAGE_DIM)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    streamers = {}
    with service_row_images(tmp_path):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
            for layer_id in range(LAYERS):
                layer = torch.nn.Module()
                layer.layer_id = layer_id
                fmt = Exl3ExpertFormat(layout, layer_id, source_root=str(tmp_path))
                streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
                layer._nvfp4_expert_streamer = streamer
                ExpertPinnedHostCache(streamer, CAPACITY, device="cpu", **fmt.pinned_tier_options(layer))
                streamers[layer_id] = streamer
        service = module.Exl3RamMissService.get()
        yield service, streamers
        service.shutdown()
    module.Exl3RamMissService._instance = None


def _attach(service, streamer, rows):
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, add_residency_listener=lambda listener: None)
    streamer._graph_pinned_tier = True
    streamer.hot_cache = SimpleNamespace(device="cpu")
    streamer.graph_gather_rows = rows
    streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
    service.attach(manager, streamer)


@pytest.mark.parametrize("rows", [1, 6, MAX_IDS])
def test_a_gather_within_the_lanes_attaches(tiers, rows):
    service, streamers = tiers
    _attach(service, streamers[0], rows)
    assert service.routed_rows_per_step == rows
    assert streamers[0].row_backend.device_side is service.device_side


@pytest.mark.parametrize("rows", [MAX_IDS + 1, 2 * 6])  # 12: two tokens of a top-6 model
def test_a_gather_wider_than_the_lanes_is_refused_before_anything_is_built(tiers, rows):
    service, streamers = tiers
    with pytest.raises(ValueError, match=rf"gathers up to {rows} rows .* at most {MAX_IDS} lanes"):
        _attach(service, streamers[1], rows)
    assert service.device_side is None  # no device words were allocated for it
    assert service.routed_rows_per_step == 0
    assert not hasattr(streamers[1].row_backend, "device_side")  # the layer keeps its previous backend



def test_cpu_experts_refuse_a_manager_without_direct_residency(tiers):
    """The miss order CPU experts rely on is DIRECT's victim ranking; without the updater there is none to sort by."""
    service, streamers = tiers
    service.ensure_started()
    service.cpu_experts = object()
    try:
        with pytest.raises(RuntimeError, match="needs DIRECT residency"):
            _attach(service, streamers[0], 1)
    finally:
        service.cpu_experts = None


def test_cpu_experts_refuse_the_generic_route_plan():
    """Only the fused plan sorts the miss lanes; refused before the host is touched."""
    cfg = SimpleNamespace(enable_ram_miss_copy_engine=True, enable_layer_fusion=True)
    with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"), envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(False):
        with pytest.raises(RuntimeError, match="needs SGLANG_MOE_EXPERT_FUSED_PLAN"):
            module.Exl3RamMissService._start_cpu_experts(cfg, None, None, {}, False)

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
