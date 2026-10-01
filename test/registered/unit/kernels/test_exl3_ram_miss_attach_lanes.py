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


def test_attach_unmaps_on_a_full_tier_reach_the_device_before_any_post(tiers, monkeypatch):
    """A full tier: attach evicts rows to make its staging slots. Those unmaps must reach the device's map before the
    service runs again -- a captured warm-up decode would otherwise type a hit on a slot the host now stages into, and
    the host would fail-stop. The device here is CPU-resident, so its bulk apply is recorded instead of launched."""
    service, streamers = tiers
    service.ensure_started()
    service.before_host_use()
    for row in range(LAYERS):
        for expert in range(CAPACITY):
            service.host.assign(row, expert)
    service.after_host_use()
    applied = []
    monkeypatch.setattr(module.ExpertStreamDevice, "map_bulk_apply", lambda self, bulk: applied.extend(bulk.tolist()))
    _attach(service, streamers[0], 1)
    at_attach = list(applied)  # what reached the device by the end of attach; the read below applies the rest
    service.before_host_use()
    try:
        mapping = service.host.mapping(service.row_of(0))
    finally:
        service.after_host_use()
    row = service.row_of(0)
    replica = {}
    for r, expert, slot in at_attach:
        if r == row:
            replica[expert] = slot
    evicted = [e for e in range(CAPACITY) if mapping[e] < 0]
    assert evicted, "the full tier gave attach nothing to evict"
    assert all(replica.get(e) == -1 for e in evicted), (evicted, replica)


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


class _DirectUpdater:
    """The DIRECT updater surface attach reads; ``enable_miss_order`` allocates the keys as the real one does."""

    insert_direct = True
    device = torch.device("cpu")
    layer_ids = list(range(LAYERS))

    def __init__(self):
        self.slot_to_expert = torch.full((LAYERS, CAPACITY + 1), -1, dtype=torch.int64)
        self.caches = [SimpleNamespace(capacity=CAPACITY) for _ in range(LAYERS)]
        self.miss_keys = None

    def enable_miss_order(self):
        self.miss_keys = torch.zeros((LAYERS, EXPERTS), dtype=torch.int64)


def test_cpu_experts_attach_gives_every_pinned_layer_its_row_of_the_miss_keys(tiers, monkeypatch):
    """Production wiring: the first attach turns the updater's miss order on.

    Each pinned-tier layer's fused plan sorts by that layer's own row of the keys.
    The row is a view, so each ranking's keys reach the captured plan.
    """
    service, streamers = tiers
    service.ensure_started()
    built = []
    monkeypatch.setattr(
        module, "Exl3RamMissRowBackend", lambda *args, **kwargs: built.append(kwargs) or SimpleNamespace()
    )
    service.cpu_experts = SimpleNamespace(
        x_rows=torch.zeros((LAYERS, 16), dtype=torch.uint8), out_rows=torch.zeros((LAYERS, 4), dtype=torch.float32),
        attach_device=lambda device_side: None,
    )
    updater = _DirectUpdater()
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, gpu_residency=updater)
    try:
        for layer_id, streamer in streamers.items():
            streamer._graph_pinned_tier = True
            streamer.hot_cache = SimpleNamespace(device="cpu", capacity=CAPACITY)
            streamer.graph_gather_rows = 1
            streamer.residency_row = LAYERS - 1 - layer_id  # not the service's row: the keys follow the updater's
            streamer.row_backend = SimpleNamespace(
                segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64)
            )
            service.attach(manager, streamer)
    finally:
        service.cpu_experts = None
    assert updater.miss_keys is not None
    for streamer in streamers.values():
        keys = streamer._plan_miss_keys
        assert keys.data_ptr() == updater.miss_keys[streamer.residency_row].data_ptr()
        assert keys.shape == (EXPERTS,)
    assert [kwargs["streamer_of"]() for kwargs in built] == list(streamers.values())
    assert all(kwargs["cpu_experts"] for kwargs in built)


class _CapturedPlan:
    expert_ids = torch.tensor([2], dtype=torch.int64)
    count = torch.ones(1, dtype=torch.int32)
    slots = torch.zeros(1, dtype=torch.int32)


class _Side:
    """The device side's chain, recorded: which stages a post launched, in order."""

    host_rows_1 = dst_slots_1 = go_1 = None

    def __init__(self):
        self.calls = []
        self.copy_engine_captured = False

    def __getattr__(self, name):
        if name in ("post", "stream", "copy_wait"):
            return lambda *args, **kwargs: self.calls.append((name, kwargs))
        raise AttributeError(name)


def _captured_backend(monkeypatch, streamer_of):
    side = _Side()
    backend = module.Exl3RamMissRowBackend(
        {0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), side, 0, 1, {0: None},
        copy_engine=True, cpu_experts=True, streamer_of=streamer_of,
    )
    backend.cpu_input = (torch.zeros(1, 16), torch.ones(1, 1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(module, "copy_expert_row_segments_gpu", lambda *args: side.calls.append(("copy", {})))
    return backend, side


def test_a_captured_cpu_expert_gather_with_the_miss_order_posts_its_input(monkeypatch):
    """The keys installed: the post runs the whole chain and stages the CPU input."""
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    backend.post(0, _CapturedPlan())
    assert [name for name, _ in side.calls] == ["post", "copy", "stream", "copy_wait"]
    assert side.calls[0][1]["captured"] and side.calls[0][1]["cpu_input"] is backend.cpu_input
    assert side.copy_engine_captured


def test_a_captured_cpu_expert_gather_without_the_miss_order_is_refused(monkeypatch):
    """enable_graph_gather resets a layer's keys; a capture after that would sort nothing, so the post refuses it."""
    streamer = SimpleNamespace(_plan_miss_keys=None)
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    with pytest.raises(RuntimeError, match="row 0's route plan has no miss order"):
        backend.post(0, _CapturedPlan())
    assert not side.calls


def test_a_captured_cpu_expert_gather_whose_streamer_is_gone_is_refused(monkeypatch):
    backend, side = _captured_backend(monkeypatch, lambda: None)
    with pytest.raises(RuntimeError, match="row 0's streamer is gone"):
        backend.post(0, _CapturedPlan())
    assert not side.calls


def test_cpu_experts_backend_needs_its_streamer():
    with pytest.raises(ValueError, match="pass streamer_of"):
        module.Exl3RamMissRowBackend(
            {0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), SimpleNamespace(), 0, 1, {0: None},
            copy_engine=True, cpu_experts=True,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
