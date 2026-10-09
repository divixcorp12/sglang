"""Attaching an in-graph layer refuses a gather wider than the lanes the post kernel requests (CPU)."""

import faulthandler
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
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
        service.plan_gather_width(1)
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


@pytest.mark.parametrize("rows, lanes", [(1, 8), (6, 8), (8, 8), (12, 16), (16, 16), (32, 32), (36, 40), (40, 40)])
def test_a_gather_within_the_planned_lanes_attaches(tiers, rows, lanes):
    """The width is planned before the service starts (Exl3ExpertFormat.plan_graph_gather), and the build's lanes are
    that width rounded up to 8."""
    service, streamers = tiers
    service.plan_gather_width(rows)
    _attach(service, streamers[0], rows)
    assert service.routed_rows_per_step == rows and service.lanes == lanes
    assert service.device_side.wire.lanes == lanes
    assert streamers[0].row_backend.device_side is service.device_side


def test_a_gather_wider_than_64_is_refused_when_planned(tiers):
    service, streamers = tiers
    with pytest.raises(ValueError, match="1..64"):
        service.plan_gather_width(65)


@pytest.mark.parametrize("rows", [9, 2 * 6])  # 12: two tokens of a top-6 model
def test_a_gather_wider_than_the_built_lanes_is_refused_before_anything_is_built(tiers, rows):
    """A layer that never planned its width (the service built for 8 lanes) cannot post more lanes than it has."""
    service, streamers = tiers
    with pytest.raises(ValueError, match=rf"gathers up to {rows} rows .* at most 8 lanes"):
        _attach(service, streamers[1], rows)
    assert service.device_side is None  # no device words were allocated for it
    assert service.routed_rows_per_step == 0
    assert not hasattr(streamers[1].row_backend, "device_side")  # the layer keeps its previous backend


def test_a_verify_gather_attaches_by_its_miss_width_not_its_routes(tiers):
    """Six tokens of top-6 route 36 ids a layer, past the wire's 32; the gather serves 8 misses, so it builds 8 lanes."""
    service, streamers = tiers
    service.plan_gather_width(8)
    streamers[0].graph_miss_lanes = 8
    _attach(service, streamers[0], 36)
    assert service.lanes == 8 and service.routed_rows_per_step == 36


@pytest.mark.parametrize("staged, warns", [(8, False), (7, True)])
def test_a_verify_row_stages_its_miss_width_not_its_routes(tiers, monkeypatch, caplog, staged, warns):
    """A post requests at most the miss width, so a row staging W slots is enough though the gather routes 36 ids;
    fewer than W still warns."""
    service, streamers = tiers
    service.plan_gather_width(8)
    monkeypatch.setattr(service, "staging_for", lambda capacity: staged)
    streamers[0].graph_miss_lanes = 8
    with caplog.at_level("WARNING", logger=module.__name__):
        _attach(service, streamers[0], 36)
    assert any("stages" in r.getMessage() for r in caplog.records) is warns


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
    cfg = SimpleNamespace(enable_ram_miss_copy_engine=True, enable_layer_fusion=True)
    with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"), envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(False):
        with pytest.raises(RuntimeError, match="needs SGLANG_MOE_EXPERT_FUSED_PLAN"):
            module.Exl3RamMissService._start_cpu_experts(cfg, None, None, {}, False, None)


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


@pytest.mark.parametrize("lanes, capacity, width", [(8, 3, 8), (16, 5, 16), (16, 40, 40), (32, 12, 32)])
def test_the_backend_pads_its_plan_to_the_build_lanes(lanes, capacity, width):
    """The production backend (not only the service's helper) pads ``planned`` to max(capacity, lanes): an eager post
    of 9-16 lanes at 16 must fit the tensor. Mutation: padded_plan_width returns the capacity."""
    backend = module.Exl3RamMissRowBackend(
        {0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), SimpleNamespace(wire=lease.wire_layout(lanes)),
        0, capacity, {0: None},
    )
    assert backend.planned.numel() == width and backend.routes.numel() == capacity


class _CapturedPlan:
    expert_ids = torch.tensor([2], dtype=torch.int64)
    count = torch.ones(1, dtype=torch.int32)
    slots = torch.zeros(1, dtype=torch.int32)


class _Side:
    """The device side's chain, recorded: which stages a post launched, in order."""

    host_rows_1 = dst_slots_1 = go_1 = None
    wire = lease.wire_layout(8)

    def __init__(self):
        self.calls = []
        self.copy_engine_captured = False

    def __getattr__(self, name):
        if name in ("post", "stream", "copy_wait", "spec_score"):
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
            {0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), SimpleNamespace(wire=lease.wire_layout(8)), 0, 1,
            {0: None}, copy_engine=True, cpu_experts=True,
        )


def test_victim_lanes_stage_their_width_not_the_lanes(tiers):
    """Spill: the post types up to 36 lanes, but only the V victim lanes stage (a forced miss never does), so a row
    reserves V staging slots (Exl3ExpertFormat.plan_graph_gather plans it)."""
    service, streamers = tiers
    with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(2):
        streamers[0].format.plan_graph_gather(streamers[0], 36)
    assert service.resolved_lanes() == 40
    assert service.staging_width() == 2
    assert service.staging_for(CAPACITY) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def _cpu_stand_in():
    return SimpleNamespace(
        x_rows=torch.zeros((LAYERS, 16), dtype=torch.uint8), out_rows=torch.zeros((LAYERS, 4), dtype=torch.float32),
        attach_device=lambda device_side: None,
    )


def _spill_updater():
    updater = DirectUpdaterStandIn(LAYERS, CAPACITY, EXPERTS)
    updater.miss_rows, updater.victim_lanes = 36, 8
    updater.overflow_flag = torch.zeros(1, dtype=torch.int32)
    updater.gather_overflow = torch.zeros(LAYERS, dtype=torch.int64)
    return updater


def _attach_verify(service, streamer, updater):
    streamer._graph_pinned_tier = True
    streamer.hot_cache = SimpleNamespace(device="cpu", capacity=CAPACITY)
    streamer.graph_gather_rows, streamer.graph_miss_lanes = 36, 0  # a lane per route
    streamer.residency_row = 1
    streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
    service.attach(SimpleNamespace(register_fail_stop_check=lambda check: None, gpu_residency=updater), streamer)


def test_cpu_experts_attach_a_verify_gather_and_wire_its_spill_words(tiers, monkeypatch):
    """Six tokens of top-6 with a lane per route (36 on a 40-lane wire), 8 of them victim lanes: the layer attaches with
    CPU experts on, and its backend posts with DIRECT's overflow flag and its own row of the overflow counter (a view:
    the post's increment is the updater's). The room check is the next test's."""
    service, streamers = tiers
    service.plan_gather_width(36)
    service.ensure_started()
    monkeypatch.setattr(module, "Exl3RamMissRowBackend", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(service, "_check_spill_room", lambda row, streamer, width: None)
    service.cpu_experts = _cpu_stand_in()
    updater = _spill_updater()
    try:
        _attach_verify(service, streamers[0], updater)
    finally:
        service.cpu_experts = None
    flag, counter = streamers[0].row_backend.spill
    assert flag is updater.overflow_flag
    counter.add_(1)
    assert updater.gather_overflow.tolist() == [0, 1]


def test_spill_refuses_a_tier_without_a_victim_for_every_forced_miss(tiers, monkeypatch):
    """Review Focus 2 at start-up: a forced CPU miss is read into a RAM victim, so every node range of the layer must hold
    staging + 36 lanes + the VRAM-hot slots. These 3-slot tiers cannot: refused at attach, not fail-stopped mid-verify."""
    service, streamers = tiers
    service.plan_gather_width(36)
    service.ensure_started()
    monkeypatch.setattr(module, "Exl3RamMissRowBackend", lambda *args, **kwargs: SimpleNamespace())
    service.cpu_experts = _cpu_stand_in()
    try:
        with pytest.raises(ValueError, match="reads every forced CPU miss into a RAM victim"):
            _attach_verify(service, streamers[0], _spill_updater())
    finally:
        service.cpu_experts = None


@pytest.mark.parametrize(
    "ranges, short",
    [
        ([(0, 80), (80, 161)], []),  # the recipe: 80 and 81 slots, 8 + 36 + 24 = 68 needed
        ([(0, 60), (60, 161)], [(0, 60, 68)]),
        ([(0, 161)], []),  # one node
    ],
)
def test_spill_room_is_staging_plus_lanes_plus_hot_per_node(ranges, short):
    assert module.spill_room_shortfall(ranges, staging=8, lanes=36, hot=24) == short


def test_a_captured_cpu_expert_gather_posts_its_spill_words(monkeypatch):
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    assert backend.spill is None
    backend.spill = (torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64))
    backend.post(0, _CapturedPlan())
    assert side.calls[0][1]["spill"] is backend.spill


def test_cpu_rows_hold_the_verify_tokens():
    """A 6-token verify's service rows: x rows of cpu_row_bytes(hidden, 6, lanes), out rows [rows, 2 * nodes, 6,
    hidden]; one token keeps today's shapes."""
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    cores = sorted(os.sched_getaffinity(0))[:2]
    for tokens, out_shape in ((1, (2, 2, 64)), (6, (2, 2, 6, 64))):
        host = SimpleNamespace(
            wire=lease.wire_layout(40), nodes=1, enable_cpu_experts=lambda *a, **k: None, cpu_stats=lambda group: {},
        )
        trait = SimpleNamespace(check_environment=lambda: None, kernel_address=lambda: 1, name="t")
        service = CpuExpertService(
            host, trait, {0: {}, 1: {}}, hidden=64, cores=cores, threads=2, split=[0] * 41, pin=False, tokens=tokens,
        )
        assert tuple(service.out_rows.shape) == out_shape
        assert service.x_rows.shape[1] == lease.cpu_row_bytes(64, tokens, 40)


def test_the_speculative_pool_adds_its_share_to_the_spill_room():
    """The pool's kSpec slots are never victims, so a forced miss's room needs them on top (RAM prefetch)."""
    assert module.spill_room_shortfall([(0, 69), (69, 139)], staging=8, lanes=36, hot=24, pool=2) == [(0, 69, 70)]
    assert module.spill_room_shortfall([(0, 80), (80, 161)], staging=8, lanes=36, hot=24, pool=2) == []


def test_a_rows_reserved_slots_count_the_speculative_pool():
    """The pool's share per row and group is never assigned, so the eager LRU must not count on it either."""
    service = SimpleNamespace(staging_for=lambda capacity: 8, _spec_share=2, host=SimpleNamespace(nodes=2))
    service.pool_slots = lambda: module.Exl3RamMissService.pool_slots(service)
    assert module.NativePinnedSlotTable.reserved_rows.fget(SimpleNamespace(service=service, capacity=40)) == 12


@pytest.mark.parametrize(
    "ranges, width, short",
    [
        ([(0, 10), (10, 20)], 7, [(0, 0, 10, 7), (1, 10, 20, 7)]),  # each group stages 7, not 7 for the row
        ([(0, 4), (4, 20)], 1, [(0, 0, 4, 1)]),  # skewed: the small group runs out
        ([(0, 5), (5, 20)], 1, []),
        ([(0, 10), (10, 20)], 6, []),
    ],
)
def test_the_pool_room_is_checked_per_group_as_the_host_reserves_it(ranges, width, short):
    """Each group stages min(width, its slots - 1) and keeps 2 assignable slots past its pool (ram_tier.h
    reserve_staging, reserve_spec_pool); the row's totals would accept every case here."""
    capacity, nodes = ranges[-1][1], len(ranges)
    assert capacity - min(width, capacity - 1) - 2 * nodes >= 2 * nodes
    assert module.pool_room_shortfall(ranges, staging_width=width, share=2) == short


def test_a_row_with_a_target_scores_right_after_its_post(monkeypatch):
    """The scoring kernels follow the post on its stream, before C1: the host learns the record and its candidates
    while this layer still runs (spec 2026-10-09, Approach A)."""
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    entry = SimpleNamespace(target=1)
    backend.spec_expected, backend.spec_score = True, entry
    backend.post(0, _CapturedPlan())
    assert [name for name, _ in side.calls] == ["post", "spec_score", "copy", "stream", "copy_wait"]
    assert side.calls[1][1]["entry"] is entry and side.calls[1][1]["x"] is backend.cpu_input[0]


def test_a_row_whose_target_has_not_attached_is_refused_before_anything_posts(monkeypatch):
    """The table is built at attach, before capture; a post without its entry would freeze a graph that never scores."""
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    backend.spec_expected = True
    with pytest.raises(RuntimeError, match="row 0 posted before its target row attached"):
        backend.post(0, _CapturedPlan())
    assert not side.calls


def test_a_row_without_a_target_posts_the_chain_alone(monkeypatch):
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    backend.post(0, _CapturedPlan())
    assert [name for name, _ in side.calls] == ["post", "copy", "stream", "copy_wait"]
