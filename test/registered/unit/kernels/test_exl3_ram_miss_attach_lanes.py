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
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, DirectUpdaterStandIn, paused, service_row_images

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
        service.staging_slots = 1
        yield service, streamers
        service.shutdown()
    module.Exl3RamMissService._instance = None


def _manager(updater=None):
    return SimpleNamespace(
        register_fail_stop_check=lambda check: None,
        gpu_residency=updater if updater is not None else DirectUpdaterStandIn(LAYERS, CAPACITY, EXPERTS),
    )


def _attach(service, streamer, rows, manager=None):
    streamer._graph_pinned_tier = True
    streamer.hot_cache = SimpleNamespace(device="cpu", capacity=CAPACITY)
    streamer.graph_gather_rows = rows
    streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
    service.attach(manager if manager is not None else _manager(), streamer)


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


def test_a_full_eager_fill_after_start_maps_no_staging_slot(tiers):
    """The host reserves each row's staging at start, so a fill that fills the tier evicts for its mappable slots only,
    and no bulk-delta entry (what the device's map applies) names a staging slot. Mutation: staging is reserved after
    the fill, from the mapped rows."""
    service, _ = tiers
    service.ensure_started()
    host = service.host
    with paused(host):
        staging = {
            row: [s for s, (state, _, _) in enumerate(host.slot_info(row)) if state == 3] for row in range(LAYERS)
        }
    assert all(len(slots) == 1 for slots in staging.values())
    service.before_host_use()
    try:
        for row in range(LAYERS):
            for expert in range(EXPERTS):
                host.assign(row, expert)
        mapped = {row: {slot for slot in host.mapping(row) if slot >= 0} for row in range(LAYERS)}
        bulk = host.take_bulk_delta().tolist()
    finally:
        service.after_host_use()
    for row in range(LAYERS):
        assert len(mapped[row]) == CAPACITY - 1 and not mapped[row] & set(staging[row])
    assert bulk and all(slot not in staging[row] for row, _, slot in bulk if slot >= 0)


def test_eager_use_with_no_device_side_keeps_no_bulk_delta(tiers):
    """With no graph-pinned layer there is no device map, so nothing ever takes the bulk delta: kept, it would grow by
    every eager admission for the life of the server. Mutation: after_host_use keeps the bulk when there is no device."""
    service, _ = tiers
    service.ensure_started()
    assert service.device_side is None
    for _ in range(3):
        service.before_host_use()
        for row in range(LAYERS):
            for expert in range(EXPERTS):
                service.host.assign(row, expert)
        service.after_host_use()
    service.host.pause(5.0)
    try:
        assert service.host.take_bulk_delta().numel() == 0
    finally:
        service.host.resume()


def test_attach_refuses_a_manager_without_direct_residency(tiers):
    """Every record carries the DIRECT updater's hot set, and the CPU experts' miss order is its victim ranking: with no
    updater (or one that is not DIRECT) there is nothing to feed either. Mutation: attach registers a residency
    listener instead."""
    service, streamers = tiers
    service.ensure_started()
    off = SimpleNamespace(register_fail_stop_check=lambda check: None, gpu_residency=None)
    with pytest.raises(RuntimeError, match="needs DIRECT residency"):
        _attach(service, streamers[0], 1, off)
    with pytest.raises(RuntimeError, match="needs DIRECT residency"):
        _attach(service, streamers[0], 1, _manager(SimpleNamespace(insert_direct=False)))
    assert service.device_side is None and service._manager is None


def test_cpu_experts_refuse_the_generic_route_plan():
    """Only the fused plan sorts the miss lanes; refused before the host is touched."""
    cfg = SimpleNamespace(enable_ram_miss_copy_engine=True)
    with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"), envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(False):
        with pytest.raises(RuntimeError, match="needs SGLANG_MOE_EXPERT_FUSED_PLAN"):
            module.Exl3RamMissService._start_cpu_experts(cfg, None, None, {}, False)


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
    updater = DirectUpdaterStandIn(LAYERS, CAPACITY, EXPERTS)
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
