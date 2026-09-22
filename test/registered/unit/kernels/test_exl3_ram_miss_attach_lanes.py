"""Attaching an in-graph layer refuses a gather wider than the lanes the post kernel requests (CPU)."""

import faulthandler
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import MAX_IDS
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tiers(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    streamers = {}
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
        for layer_id in range(LAYERS):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            fmt = Exl3ExpertFormat(layout, layer_id, direct=False, source_root=str(tmp_path))
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
