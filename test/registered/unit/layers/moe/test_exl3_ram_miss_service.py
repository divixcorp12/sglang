"""Option C wiring on the host: the native slot table under the pinned tier, the fail-stop
check, residency pushes and the per-step graph trace (CPU; the thread runs, no device)."""

import faulthandler
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import sim_post, sim_wait
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


@pytest.fixture(autouse=True)
def hang_guard():
    # The service thread and its handshakes run in C++: a broken handshake or join hangs,
    # so dump every stack and exit instead (pytest-timeout could not interrupt it).
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    streamers, caches = {}, {}
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
        for layer_id in range(LAYERS):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            fmt = Exl3ExpertFormat(layout, layer_id, direct=False, source_root=str(tmp_path))
            streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            options = fmt.pinned_tier_options(layer)
            caches[layer_id] = ExpertPinnedHostCache(streamer, CAPACITY, device="cpu", **options)
            streamers[layer_id] = streamer
    service = module.Exl3RamMissService.get()
    yield service, streamers, caches
    service.shutdown()
    module.Exl3RamMissService._instance = None


def test_the_pinned_tier_runs_on_the_native_slot_table(tiers):
    service, streamers, caches = tiers
    cache = caches[1]
    assert isinstance(cache._lru, module.NativePinnedSlotTable)
    cache.ensure_rows(torch.tensor([4, 2]))
    row = service.row_of(1)
    assert service.host.contains(row, 4) and service.host.contains(row, 2)
    assert cache.expert_to_slot[4].item() == service.host.mapping(row)[4]
    assert service.slot_map[row, 4].item() == cache.expert_to_slot[4].item()


def test_the_service_reads_through_the_mirror_roots_the_env_names(tiers, tmp_path):
    service, streamers, caches = tiers
    roots = [tmp_path.parent / f"{tmp_path.name}_mirror{i}" for i in range(2)]
    for root in roots:
        shutil.copytree(tmp_path, root)
    with envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(os.pathsep.join(map(str, roots))):
        with envs.SGLANG_MOE_EXPERT_MIRROR_WEIGHTS.override("3:1"):
            caches[1].ensure_rows(torch.tensor([4, 2]))
    tables = service.host.tables
    assert tables.parts == 2
    assert all(path.startswith(str(root)) for path, root in zip(tables.paths, roots * len(tables.paths)))
    # Every row's parts are what the eager policy plans for 3:1.
    planned = StaticSplitPolicy((3.0, 1.0)).plan(int(tables.slot_bytes)).part_bytes
    assert bool((tables.extents[..., 2] == torch.tensor(planned)).all())
    row = service.row_of(1)
    assert service.host.contains(row, 4) and service.host.contains(row, 2)


def test_the_service_refuses_a_mirror_configuration_the_eager_source_refuses(tiers, tmp_path):
    service, streamers, caches = tiers
    with envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(str(tmp_path)):  # the checkpoint itself
        with pytest.raises(ValueError, match="not a mirror of it"):
            caches[1].ensure_rows(torch.tensor([4]))


def test_rows_the_thread_loads_reach_the_eager_map_on_next_host_use(tiers):
    service, streamers, caches = tiers
    service.ensure_started()
    row = service.row_of(0)
    assert sim_wait(service.page, sim_post(service.page, row, need=[5], protect=[5]), 10) == 1
    caches[0].lookup(torch.tensor([5]))  # before_host_use refreshes the device-side copy
    assert caches[0].expert_to_slot[5].item() == service.host.mapping(row)[5] >= 0


def test_nested_host_use_pauses_and_refreshes_only_at_the_outermost_level(tiers, monkeypatch):
    service, streamers, caches = tiers
    service.ensure_started()
    cache = caches[0]
    row = service.row_of(0)
    refreshes, pauses = [], []
    monkeypatch.setattr(cache, "_refresh_mapping", lambda: refreshes.append(service.host.version()))
    pause, resume = service.host.pause, service.host.resume
    monkeypatch.setattr(service.host, "pause", lambda timeout_s: (pauses.append("pause"), pause(timeout_s)))
    monkeypatch.setattr(service.host, "resume", lambda: (pauses.append("resume"), resume()))
    assert sim_wait(service.page, sim_post(service.page, row, need=[1], protect=[1]), 10) == 1
    with cache.host_use():
        with cache.host_use():  # nested: another version change must not refresh here
            service.host.assign(row, 3)
            with cache.host_use():
                pass
    assert len(refreshes) == 1 and pauses == ["pause", "resume"]
    with cache.host_use():  # the outer level sees the version the nested use left
        pass
    assert len(refreshes) == 2


def test_the_fail_stop_check_raises_once_fatal_is_set(tiers):
    service, streamers, caches = tiers
    service.ensure_started()
    service.fail_stop_check()  # nothing raised yet
    service.host.inject(fail_reads=True)
    row = service.row_of(0)
    assert sim_wait(service.page, sim_post(service.page, row, need=[1], protect=[1]), 10) == 2
    with pytest.raises(RuntimeError, match="exl3 RAM miss"):
        service.fail_stop_check()
    # fatal stays raised: stop now, before the watchdog's fatal-held rule can abort pytest.
    service.shutdown()


def test_attach_registers_once_and_pushes_residency(tiers):
    service, streamers, caches = tiers
    checks, listeners = [], []
    manager = SimpleNamespace(
        register_fail_stop_check=checks.append,
        add_residency_listener=listeners.append,
    )
    for streamer in streamers.values():
        streamer.format.attach_hot_cache_manager(manager, streamer)
    assert len(checks) == 1 and len(listeners) == 1
    listeners[0](1, [3, -1])  # expert 3 is hot in layer 1, and in layer 1 only
    for layer_id in (0, 1):
        # Capacity 3; 3 is loaded first, so plain LRU would evict it next.
        caches[layer_id].ensure_rows(torch.tensor([3]))
        caches[layer_id].ensure_rows(torch.tensor([0, 1]))
        caches[layer_id].ensure_rows(torch.tensor([2]))
    hot_row, cold_row = service.row_of(1), service.row_of(0)
    # The pushed hot map kept 3 in layer 1: the LRU-oldest non-hot row, 0, went instead.
    assert [service.host.contains(hot_row, e) for e in (3, 0, 1, 2)] == [True, False, True, True]
    # Layer 0 got no push, so its LRU-oldest row, 3, went.
    assert [service.host.contains(cold_row, e) for e in (3, 0, 1, 2)] == [False, True, True, True]


def test_gpu_hot_snapshot_handoff_keeps_seed_protection_then_uses_device_map(tiers):
    service, streamers, caches = tiers
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True):
        service.ensure_started()
    table = caches[0]._lru
    # The initial reassignment predates the updater: this path must use the
    # Python hot list until the GPU bank has been created.
    streamers[0].hot_cache = SimpleNamespace(slot_to_expert=[0, -1, -1])
    assert not service.gpu_hot_enabled
    table.before_host_use(caches[0])
    try:
        for expert in (0, 1, 2):
            service.host.assign(service.row_of(0), expert)
        assert service.host.victim_census(service.row_of(0)) == (0, 2, 0)
    finally:
        table.after_host_use(caches[0])

    bank = torch.tensor([[4, -1, -1], [-1, -1, -1]], dtype=torch.int64)
    updater = SimpleNamespace(
        device=torch.device("cpu"), slot_to_expert=bank,
        layer_ids=[0, 1], caches=[SimpleNamespace(capacity=3), SimpleNamespace(capacity=3)],
    )
    service._enable_gpu_hot(updater)
    assert service.hot_experts(0) == [4]
    bank[0, 0] = 2  # the next eager handoff must use this, not stale Python [0]
    table.before_host_use(caches[0])
    try:
        assert service.hot_experts(0) == [2]
        assert service.host.victim_census(service.row_of(0)) == (0, 2, 0)
        slot, evicted = service.host.assign(service.row_of(0), 3, protected_fallback=False)
        assert evicted == 0 and slot >= 0 and service.host.contains(service.row_of(0), 2)
    finally:
        table.after_host_use(caches[0])


def test_exl3_direct_startup_refuses_unsupported_modes_before_capture():
    from sglang.srt.layers.moe.expert_format import require_graph_gather_support
    from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

    source = torch.zeros(8, dtype=torch.int64)
    count = torch.zeros(1, dtype=torch.int32)
    fmt = SimpleNamespace(key="exl3", graph_source_kind="pinned_tier", supports_graph_gather=False)
    backend = object.__new__(module.Exl3RamMissRowBackend)
    streamer = SimpleNamespace(
        format=fmt, layer_id=0, pinned_host_cache=SimpleNamespace(capacity=16),
        has_spec_only_tensors=True, _graph_source_rows=source, _graph_miss_count=count,
        row_plan=SimpleNamespace(expert_ids=source, count=count), row_backend=backend,
    )
    # OFF/SCRATCH and a doorbell configuration retain the original EXL3 GPU
    # update rejection. Only the DIRECT caller opts into pinned-tier support.
    for stage in (0, 1):
        with pytest.raises(ValueError, match="does not support graph gather"):
            require_graph_gather_support([streamer], exl3_direct_ok=False)
    require_graph_gather_support([streamer], exl3_direct_ok=True)
    updater = object.__new__(GpuResidencyUpdater)
    updater.insert_direct = True
    updater.streamers = [streamer]
    with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"):
        with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(False):
            with pytest.raises(ValueError, match="ENABLE_RAM_MISS_LEASES=1"):
                updater.check_miss_plans()
        with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True):
            # Two-phase copies hit lanes into the same DIRECT destinations before the NVMe rows land,
            # and a failed request fail-stops the process, so it composes with DIRECT.
            with envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True):
                updater.check_miss_plans()
            with envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.override(True):
                with pytest.raises(ValueError, match="ENABLE_EXPERT_PREFETCH=0"):
                    updater.check_miss_plans()
            streamer.row_backend = object()
            with pytest.raises(ValueError, match="native EXL3 RAM-miss backend"):
                updater.check_miss_plans()
            streamer.row_backend = backend
            updater.check_miss_plans()
    with envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("always"):
        with pytest.raises(ValueError, match="PREFETCH_PULL_MODE=off"):
            updater.check_miss_plans()


def test_a_later_promotion_chunk_never_evicts_an_expert_an_earlier_chunk_made_hot(tiers):
    """Minor 1: the hot cache reserves chunk 1's experts before the residency listener
    pushes them to C++. Chunk 2's admission (its own host use) must still protect them,
    exactly as ``is_pinned`` does when it sizes the chunk."""
    service, streamers, caches = tiers
    cache = caches[0]
    service.ensure_started()
    row = service.row_of(0)
    cache.ensure_rows(torch.tensor([0]))
    cache.ensure_rows(torch.tensor([1, 2]))  # full (capacity 3); 0 is the LRU-oldest row
    # Chunk 1 promoted 0 into VRAM: the hot cache holds it; no listener push has run yet.
    streamers[0].hot_cache = SimpleNamespace(slot_to_expert=[0, -1])
    assert cache.evictable_rows() == 2  # is_pinned already protects 0
    cache.ensure_rows(torch.tensor([4]))  # chunk 2's admission
    assert [service.host.contains(row, e) for e in (0, 1, 2, 4)] == [True, False, True, True]


def test_expert_to_slot_is_rebuilt_only_when_the_slot_map_changes(tiers, monkeypatch):
    """Minor 7: promotions read ``_expert_to_slot`` once per promoted row."""
    service, streamers, caches = tiers
    cache = caches[0]
    cache.ensure_rows(torch.tensor([1, 2]))
    table = cache._lru
    builds = []
    mapping = service.host.mapping
    monkeypatch.setattr(service.host, "mapping", lambda row: (builds.append(row), mapping(row))[1])
    first = dict(cache._expert_to_slot)
    for _ in range(5):
        assert dict(cache._expert_to_slot) == first
    assert len(builds) == 1
    cache.ensure_rows(torch.tensor([4]))  # a version change: the next read rebuilds
    builds.clear()
    assert set(cache._expert_to_slot) == {1, 2, 4} and len(builds) == 1
    assert dict(table.expert_to_slot) == {e: s for e, s in enumerate(mapping(service.row_of(0))) if s >= 0}


def test_the_watchdog_wait_outlasts_the_wait_timeout_and_the_pause_bound(tiers, monkeypatch):
    """Minor 5: the watchdog's limit follows SGLANG_DSV41_RAM_MISS_TIMEOUT_MS, so a slow drive's
    demand fails stop through the device wait, not the watchdog's abort."""
    service, streamers, caches = tiers
    for timeout_ms in (50, 2000, 14_000, 40_000, 120_000):
        wait_s = module.watchdog_wait_s(timeout_ms)
        assert wait_s >= 30.0
        assert wait_s > 2 * timeout_ms / 1000 + 1.0  # the eager pause bound
    started = []
    start = module.Exl3RamMissHost.start_thread
    monkeypatch.setattr(
        module.Exl3RamMissHost, "start_thread", lambda self, **kw: (started.append(kw), start(self, **kw))[1]
    )
    with envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(40_000):
        service.ensure_started()
    assert started == [{"fatal_wait_s": module.watchdog_wait_s(40_000)}]
    assert started[0]["fatal_wait_s"] > 40.0 * 2 + 1.0


@pytest.mark.parametrize("prefetch, advise", [(True, 1), (None, 0)], ids=["env_on", "env_unset"])
def test_attach_builds_the_device_side_with_advise_from_the_prefetch_env(tiers, prefetch, advise):
    """The production hop: SGLANG_DSV41_ENABLE_EXPERT_PREFETCH -> prefetch_enabled() -> attach ->
    Exl3RamMissDevice(advise=...) -> the row backend's posts. Nothing else sets ``advise``."""
    service, streamers, caches = tiers
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, add_residency_listener=lambda listener: None)
    for streamer in streamers.values():
        # The graph-gather state the pinned-tier format sets up on a real streamer.
        streamer._graph_pinned_tier = True
        streamer.hot_cache = SimpleNamespace(device="cpu")
        streamer.graph_gather_rows = 6
        streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
    # The env is read while attaching, so it is overridden around the attach only.
    var = envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH
    with var.override(bool(prefetch)):  # restores whatever was there on exit
        if prefetch is None:
            var.clear()  # really unset, not set to False
        for streamer in streamers.values():
            streamer.format.attach_hot_cache_manager(manager, streamer)
    assert service.device_side.advise == advise
    # One bs-1 decode step routes graph_gather_rows rows in every in-graph layer.
    assert service.routed_rows_per_step == 6 * len(streamers)
    for layer_id, streamer in streamers.items():
        assert streamer.row_backend.device_side is service.device_side  # one device side, shared by every layer


def _attach_all(service, streamers, capacity=None):
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, add_residency_listener=lambda listener: None)
    for streamer in streamers.values():
        streamer._graph_pinned_tier = True
        streamer.hot_cache = SimpleNamespace(device="cpu", capacity=capacity)
        streamer.graph_gather_rows = 6
        streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
        streamer.format.attach_hot_cache_manager(manager, streamer)


@pytest.mark.parametrize("traced", [False, True], ids=["trace_off", "trace_on"])
def test_graph_routes_are_logged_only_when_the_stage_trace_is_on(tiers, monkeypatch, tmp_path, traced):
    """Trace off: no route log exists, so no backend adds a copy to the captured graph, no per-batch
    check reads one back, and no eager forward reads the graph count (each would cost a sync or a
    kernel per layer in production). Trace on: one log shared by every layer, which the backend's post
    fills with the routes _apply_graph copied in and the plan's miss count."""
    from sglang.srt.layers.moe import exl3_stream_trace

    service, streamers, caches = tiers
    trace = exl3_stream_trace.Exl3StreamTrace(str(tmp_path / "trace.jsonl") if traced else "")
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: trace)
    _attach_all(service, streamers, capacity=12)
    backends = [streamer.row_backend for streamer in streamers.values()]
    if not traced:
        assert service.route_log is None and trace.graph_seq_source is None
        assert all(backend.route_log is None for backend in backends)
        service.fail_stop_check()  # nothing to read back
        return
    log = service.route_log
    assert all(backend.route_log is log for backend in backends)
    assert log.layer_ids == sorted(streamers) and log.capacities == [12] * LAYERS
    assert trace.graph_seq_source == log.read_seq
    from sglang.srt.layers.moe import expert_row_plan

    monkeypatch.setattr(expert_row_plan, "copy_expert_row_segments_gpu", lambda *args: None)
    service.device_side = SimpleNamespace(lease_block=None, post=lambda *a: None, wait=lambda *a: None)
    for backend in backends:
        backend.device_side = service.device_side
    for row, backend in enumerate(backends):  # one forward, as _apply_graph then the gather leave it
        backend.routes.copy_(torch.tensor([row, 3, 5, -1, -1, -1][: backend.routes.numel()]))
        backend.post(0, SimpleNamespace(expert_ids=torch.tensor([3]), slots=torch.zeros(1, dtype=torch.int32),
                                        count=torch.tensor([row + 1], dtype=torch.int32)))
    assert int(log.seq[0]) == 1
    log.poll(trace)
    trace.close()
    (forward,) = tier_sim_load_forwards(tmp_path / "trace.jsonl")["forwards"]
    assert (forward["kind"], forward["seq"]) == ("graph", 0)
    assert forward["routes"] == {layer: [row, 3, 5] for row, layer in enumerate(sorted(streamers))}
    assert forward["misses"] == {layer: row + 1 for row, layer in enumerate(sorted(streamers))}


def test_router_capture_is_refused_without_the_stage_trace(tiers, monkeypatch, tmp_path):
    """The router rings live in the route log, which exists only with the stage trace: a router path alone
    would capture nothing, so it is refused before the service builds anything."""
    from sglang.srt.layers.moe import exl3_stream_trace

    service, streamers, caches = tiers
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: exl3_stream_trace.Exl3StreamTrace(""))
    with envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.override(str(tmp_path / "router")):
        with pytest.raises(RuntimeError, match="needs SGLANG_DSV41_EXPERT_TRACE_PATH"):
            service.ensure_started()
    assert service.host is None


@pytest.mark.parametrize("router", [False, True], ids=["router_off", "router_on"])
def test_router_capture_rides_the_route_log_only_when_its_path_is_set(tiers, monkeypatch, tmp_path, router):
    """Unset, the route log is exactly Track B's: 64 deep, no router rings. Set, the same log shared by every
    layer carries them, on a 32-deep ring so the router input costs ~13 MB of VRAM, not ~26."""
    from sglang.srt.layers.moe import exl3_stream_trace

    service, streamers, caches = tiers
    trace = exl3_stream_trace.Exl3StreamTrace(str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: trace)
    prefix = str(tmp_path / "router")
    with envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.override(prefix if router else ""):
        _attach_all(service, streamers, capacity=12)
    log = service.route_log
    assert all(streamer.row_backend.route_log is log for streamer in streamers.values())
    if not router:
        assert log.router is None and log.depth == 64 and len(log._ring()) == 4
        return
    assert log.router.prefix == prefix and log.depth == 32
    assert log.router_x is None  # sized by the first warmup forward's record_router


def _apply_graph_ops(monkeypatch, route_log, layer_fusion):
    """Every aten op _apply_graph issues around a gather that itself issues none (its post is not under test)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from sglang.srt.layers.quantization import exl3_fused_moe
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(exl3_fused_moe, "exl3_fused_moe_for",
                        lambda layer, streamer: SimpleNamespace(
                            layer_fusion=layer_fusion, run=lambda x, w, remap, keep, limit: x.float()))
    backend = module.Exl3RamMissRowBackend(
        {0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), SimpleNamespace(), 0, -1, 6, route_log=route_log
    )
    remap = torch.zeros(6, dtype=torch.int32)
    streamer = SimpleNamespace(row_backend=backend, gather=lambda topk_ids: (remap, None))
    x = torch.arange(16, dtype=torch.float32).to(torch.bfloat16).reshape(1, 16)
    weights = torch.full((1, 6), 0.25)
    with Ops() as mode:
        out = Exl3MoEMethod._apply_graph(SimpleNamespace(layer_id=0), streamer, x, weights,
                                         torch.arange(6).reshape(1, 6), 10.0)
    assert torch.equal(out, x)
    return mode.ops


@pytest.mark.parametrize("layer_fusion", [False, True], ids=["unfused", "fused"])
def test_apply_graph_adds_nothing_unless_router_capture_is_on(monkeypatch, tmp_path, layer_fusion):
    """Trace off (no route log) and trace on without router capture build the same captured chain; router
    capture adds exactly its two ring copies. A regression here puts kernels in the production graph."""
    from collections import Counter

    from sglang.srt.layers.moe.exl3_stream_trace import GraphRouteLog

    off = _apply_graph_ops(monkeypatch, None, layer_fusion)
    routes_only = _apply_graph_ops(monkeypatch, GraphRouteLog(1, 8, "cpu"), layer_fusion)
    assert routes_only == off
    log = GraphRouteLog(1, 8, "cpu", depth=32)
    log.enable_router(str(tmp_path / "router"))
    log.router_x = torch.zeros((32, 1, 16), dtype=torch.bfloat16)  # as the warmup forward sizes them
    log.router_w = torch.zeros((32, 1, 6))
    captured = _apply_graph_ops(monkeypatch, log, layer_fusion)
    added = Counter(captured) - Counter(off)
    assert added["aten.index_copy_.default"] == 2
    views = {"aten.select.int", "aten.slice.Tensor", "aten.view.default", "aten._unsafe_view.default",
             "aten.reshape.default", "aten.alias.default"}
    assert set(added) - {"aten.index_copy_.default"} <= views, added
    assert not Counter(off) - Counter(captured)
    assert log.router_x[0, 0].tolist() == list(range(16)) and log.router_w[0, 0].tolist() == [0.25] * 6


def tier_sim_load_forwards(path):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"))
    import tier_sim

    return tier_sim.load_forwards(str(path))


def test_the_lease_switch_defaults_off_and_the_device_is_built_without_a_lease_block(tiers, monkeypatch):
    service, streamers, caches = tiers
    enabled = []
    monkeypatch.setattr(module.Exl3RamMissHost, "enable_lease_mode", lambda self: enabled.append(self))
    assert envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.get() is False
    _attach_all(service, streamers)
    assert enabled == [] and service.lease_mode is False
    assert service.device_side.lease_block is None and service.device_side.go_count is None


def test_the_lease_switch_configures_the_host_before_its_thread_and_the_device_with_the_hosts_block(tiers, monkeypatch):
    """One env read (ensure_started) feeds both sides, so the device's arming and the service's leasing cannot disagree."""
    service, streamers, caches = tiers
    order = []
    enable, start = module.Exl3RamMissHost.enable_lease_mode, module.Exl3RamMissHost.start_thread
    monkeypatch.setattr(module.Exl3RamMissHost, "enable_lease_mode", lambda self: (order.append("lease"), enable(self))[1])
    monkeypatch.setattr(module.Exl3RamMissHost, "start_thread", lambda self, **kw: (order.append("thread"), start(self, **kw))[1])
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True):
        _attach_all(service, streamers)
    assert order == ["lease", "thread"] and service.lease_mode is True
    device = service.device_side
    assert device.lease_block is service.host.lease_block  # the block the host writes, not a second one
    assert device.go_count is not None and device.go_count.dtype == torch.int32 and device.go_count.shape == (1,)
    # The env is read once, when the service starts: flipping it afterwards changes nothing.
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(False):
        service.ensure_started()
    assert service.lease_mode is True


@pytest.mark.parametrize(
    "lease_mode,two_phase,pack_workers",
    [(False, False, 0), (True, False, 0), (True, True, 0)],
    ids=["no_lease", "no_two_phase", "no_pack_workers"],
)
def test_piece_stream_refuses_unless_two_phase_lease_and_pack_workers_all_hold(
    tiers, lease_mode, two_phase, pack_workers
):
    """Config/env refusal (piece-streaming plan Sec 4.4): the inline no-pool pack path has no publisher, so
    piece streaming needs two-phase, lease mode and pack_workers > 0 together, not any two of the three."""
    service, streamers, caches = tiers
    with (
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(lease_mode),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(two_phase),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(pack_workers),
    ):
        with pytest.raises(RuntimeError, match="PIECE_STREAM needs"):
            service.ensure_started()


def test_piece_stream_is_accepted_by_the_config_refusal_once_all_three_hold(tiers):
    service, streamers, caches = tiers
    with (
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(1),
    ):
        service.ensure_started()
    assert service.piece_stream is True


@pytest.mark.parametrize("piece_stream", [False, True])
def test_piece_stream_reaches_the_host_reader_before_its_thread_and_only_when_on(tiers, monkeypatch, piece_stream):
    service, streamers, caches = tiers
    order = []
    enable, start = module.Exl3RamMissHost.enable_piece_stream, module.Exl3RamMissHost.start_thread
    monkeypatch.setattr(module.Exl3RamMissHost, "enable_piece_stream", lambda self: (order.append("piece"), enable(self))[1])
    monkeypatch.setattr(module.Exl3RamMissHost, "start_thread", lambda self, **kw: (order.append("thread"), start(self, **kw))[1])
    with (
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM.override(piece_stream),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(1),
    ):
        service.ensure_started()
    assert order == (["piece", "thread"] if piece_stream else ["thread"])


def _attach_with_copy_tables(service, streamers, *, drop_name=None):
    """``_attach_all`` with a copy table per layer whose sources are that row's slabs, as a real graph gather's
    are: the stream kernel's segment map is built from them. ``drop_name`` leaves one streamed name out."""
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES

    service.ensure_started()
    tables = service.host.tables
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, add_residency_listener=lambda listener: None)
    for layer_id, streamer in streamers.items():
        row = service.row_of(layer_id)
        entries = [
            [int(tables.slabs[row, n]), 4096 * (n + 1), int(tables.row_bytes[n])]
            for n in range(len(EXL3_STREAMED_NAMES))
            if n != drop_name
        ]
        streamer._graph_pinned_tier = True
        streamer.hot_cache = SimpleNamespace(device="cpu")
        streamer.graph_gather_rows = 6
        streamer.row_backend = SimpleNamespace(
            segments={0: SimpleNamespace(table=torch.tensor(entries, dtype=torch.int64))},
            host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64),
        )
        streamer.format.attach_hot_cache_manager(manager, streamer)
    return tables


def _piece_stream_env():
    return (
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(1),
    )


def test_piece_stream_builds_the_stream_chain_on_the_device_side_and_every_backend(tiers):
    """Flag on, the device side carries the host's piece table and the stream kernel's words, and every layer's
    backend streams, with a segment map per copy table naming each row segment's entry. (The chain's kernels and
    graph shape are the GPU tests' business: test/manual/dsv41/test_exl3_piece_stream_cuda.py.)"""
    service, streamers, caches = tiers
    a, b, c, d = _piece_stream_env()
    with a, b, c, d:
        tables = _attach_with_copy_tables(service, streamers)
    dev = service.device_side
    assert dev.piece_stream is True
    assert torch.equal(dev.piece_runs, service.host.piece_runs())
    assert tuple(dev.piece_runs.shape) == (LAYERS, EXPERTS, 8, int(tables.segments.shape[0]), 2)
    assert dev.piece_runs.abs().sum() > 0, "the piece table must cut the rows, not be all empty"
    for words in (dev.stream_count, dev.stream_abort, dev.stream_fault):
        assert int(words.abs().sum()) == 0
    names = [int(n) for n in tables.segments[:, 0].tolist()]
    for streamer in streamers.values():
        backend = streamer.row_backend
        assert backend.piece_stream is True
        assert backend.stream_maps[0].tolist() == names + [0] * 6, "every entry is a named slab: none copied whole"


def test_piece_stream_refuses_a_copy_table_missing_a_streamed_name(tiers):
    service, streamers, caches = tiers
    a, b, c, d = _piece_stream_env()
    with a, b, c, d, pytest.raises(ValueError, match="no entry for streamed names"):
        _attach_with_copy_tables(service, streamers, drop_name=2)


def test_a_row_the_reader_cannot_cut_into_pieces_refuses_the_piece_table(tiers):
    """A refused row's runs are empty, so S would admit a READY hit of it, copy nothing and commit: the host must
    refuse the table rather than hand the device one with a hole."""
    service, streamers, caches = tiers
    service.ensure_started()
    host = service.host
    real = host._module

    class OneRowRefused:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def exl3_ram_miss_piece_runs(*args):
            real.exl3_ram_miss_piece_runs(*args)
            return 1

    host._module = OneRowRefused()
    try:
        with pytest.raises(RuntimeError, match="cannot cut 1"):
            host.piece_runs()
    finally:
        host._module = real
    assert host.piece_runs().abs().sum() > 0, "the real tables cut every row"


def test_attach_refuses_a_lease_block_of_another_abi_version(tiers, monkeypatch):
    """The device kernels read area D at the offsets of lease ABI 2 (StreamProbe appended); a block the host wrote
    under any other version is refused at attach, not read at the wrong offsets."""
    from sglang.kernels.ops.moe import exl3_lease_block as lease

    service, streamers, caches = tiers
    header = module.Exl3RamMissHost.lease_header
    monkeypatch.setattr(
        module.Exl3RamMissHost, "lease_header", lambda self: {**header(self), "abi_version": lease.ABI_VERSION + 1}
    )
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True), pytest.raises(RuntimeError, match="ABI version"):
        _attach_with_copy_tables(service, streamers)


def test_without_piece_stream_the_backends_do_not_stream(tiers):
    service, streamers, caches = tiers
    with (
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True),
    ):
        _attach_with_copy_tables(service, streamers)
    assert service.device_side.piece_stream is False and service.device_side.piece_runs is None
    assert all(s.row_backend.piece_stream is False and s.row_backend.stream_maps is None for s in streamers.values())


def _backend_calls(monkeypatch, *, lease):
    calls = []
    device_side = SimpleNamespace(
        lease_block=object() if lease else None,
        go_count="GO_COUNT",
        post=lambda *a: calls.append("post"),
        wait=lambda *a: calls.append("wait"),
        ack=lambda keep: calls.append(("ack", keep)),
    )
    from sglang.srt.layers.moe import expert_row_plan

    record = lambda segments, host_rows, slots, count: calls.append(("copy", count))  # noqa: E731
    monkeypatch.setattr(module, "copy_expert_row_segments_gpu", record)  # the lease post
    monkeypatch.setattr(expert_row_plan, "copy_expert_row_segments_gpu", record)  # the inherited post
    backend = module.Exl3RamMissRowBackend({0: "SEG"}, torch.full((EXPERTS,), -1, dtype=torch.int64), device_side, 0, -1, 6)
    plan = SimpleNamespace(
        expert_ids=torch.tensor([4, 2, 5, 0, 0, 0]),
        slots=torch.arange(6, dtype=torch.int32),
        count=torch.tensor([3], dtype=torch.int32),
    )
    backend.post(0, plan)
    return calls, backend, plan


def test_without_leases_the_row_backend_copies_plan_count_and_acknowledges_nothing(monkeypatch):
    calls, backend, plan = _backend_calls(monkeypatch, lease=False)
    assert calls == ["post", "wait", ("copy", plan.count)]


def test_with_leases_the_copy_takes_go_count_and_the_acknowledgement_follows_it(monkeypatch):
    calls, backend, plan = _backend_calls(monkeypatch, lease=True)
    assert calls == ["post", "wait", ("copy", "GO_COUNT"), ("ack", backend.keep)]
    assert calls[2][1] is not plan.count


def test_shutdown_stops_the_thread_before_releasing_the_tiers_slabs(tiers, monkeypatch):
    service, streamers, caches = tiers
    service.ensure_started()
    order = []
    stop = service.host.stop
    monkeypatch.setattr(service.host, "stop", lambda: (order.append("stop"), stop()))
    for layer_id, cache in caches.items():
        close = cache.close
        monkeypatch.setattr(cache, "close", lambda close=close, layer_id=layer_id: (order.append(layer_id), close()))
    service.shutdown()
    assert order == ["stop", 0, 1]
    service.shutdown()  # idempotent: the fixture calls it again
    assert order == ["stop", 0, 1]
    # A started service refuses every later use with a clear error, not the C++
    # "unknown handle" of a closed host; the per-batch check has nothing left to check.
    with pytest.raises(RuntimeError, match="option C service was shut down"):
        caches[0].lookup(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="option C service was shut down"):
        service.on_residency(0, [1])
    service.fail_stop_check()


def test_shutdown_freezes_native_reader_before_final_trace(monkeypatch):
    service = module.Exl3RamMissService()
    order = []

    class Host:
        rows = 1

        def close_admission(self):
            order.append("close_admission")

        def pause(self, timeout_s):
            order.append("pause")
            self.rows = 2  # an in-flight demand completes before pause returns

        def layer_rows(self):
            return [self.rows]

        def stop(self):
            order.append("stop")

    host = Host()
    service.host = host
    service._stages_traced = True
    monkeypatch.setattr(service, "_establish_gpu_completion", lambda: order.append("gpu_barrier") or None)
    monkeypatch.setattr(
        service, "_trace_step", lambda *, final=False: order.append(("trace", host.layer_rows(), final))
    )
    service.shutdown()
    assert order == ["close_admission", "gpu_barrier", "pause", ("trace", [2], True), "stop"]


def test_shutdown_quarantines_after_native_stop_failure(monkeypatch):
    service = module.Exl3RamMissService()
    service._stages_traced = True
    order = []

    class Host:
        def close_admission(self):
            order.append("close_admission")

        def pause(self, timeout_s):
            order.append("pause")

        def stop(self):
            order.append("stop")
            raise RuntimeError("join failed")

    service.host = Host()
    monkeypatch.setattr(service, "_establish_gpu_completion", lambda: None)
    monkeypatch.setattr(service, "_trace_step", lambda *, final=False: order.append("trace"))
    monkeypatch.setattr(service, "_quarantine", lambda reason: order.append(("quarantine", reason)))
    service.shutdown()
    assert order[:3] == ["close_admission", "pause", "trace"]
    assert order[3] == "stop"
    assert order[4][0] == "quarantine"
    assert "join failed" in order[4][1]


def test_graph_steps_are_traced_and_read_back_by_tier_sim(tmp_path):
    import os
    import sys

    from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"))
    import tier_sim

    path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(path))
    assert trace.enabled and not Exl3StreamTrace().enabled
    trace.record_graph_step(layer_rows_delta=[1, 0, 2], routed_rows=18, routed_misses=5)
    trace.record_graph_step(
        layer_rows_delta=[0, 1, 0], routed_rows=18, routed_misses=3, thread={"rows_read": 4, "advisory_rows": 1}
    )
    # Under overlap scheduling a per-batch read can lag a step: one line then holds two.
    trace.record_graph_step(layer_rows_delta=[1, 1, 0], routed_rows=36, routed_misses=4, steps=2)
    trace.close()
    assert trace.decode_tokens == 4 and trace.decode_vram_misses == 12 and trace.decode_ram_misses == 6
    # The thread's cumulative counters ride on the line: an Engine's scheduler is
    # SIGKILLed at shutdown, so the atexit counters line is never written there.
    lines = tier_sim.load_trace(str(path))
    assert "thread" not in lines[0] and lines[1]["thread"] == {"rows_read": 4, "advisory_rows": 1}
    assert [line["steps"] for line in lines] == [1, 1, 2]
    live = tier_sim.live_summary(lines, warmup=0)
    assert live["decode_tokens"] == 4 and live["G"] == 3.0 and live["f"] == 0.5
    # Warmup counts steps too: only the two-step line is past a 2-token warmup.
    late = tier_sim.live_summary([{"forward": 0, "layer": 0, "tokens": 256, "vram_miss": 0, "ram_miss": 0}] + lines, warmup=2)
    assert late["decode_tokens"] == 4 and late["f_after_warmup"] == 2 / 4


def test_the_trace_step_reads_the_manager_registers_before_they_are_lost(monkeypatch):
    # The forward observer (_accumulate_registers) zeroes the streamers' graph counters
    # before the scheduler's per-batch check runs; the step must still be recorded.
    from sglang.srt.layers.moe import exl3_stream_trace
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager._layer_ids = [0, 1]
    manager._graph_counters = torch.zeros((2, 2), dtype=torch.int64)
    manager._graph_unique_counters = torch.zeros((2, 2), dtype=torch.int64)
    manager._registers = {}
    manager._layer_index = {}
    manager._gathered_zero_masks = {}
    manager._collect_route_history = False
    manager._collect_affinity = False
    lines = []
    trace = SimpleNamespace(enabled=True, record_graph_step=lambda **kw: lines.append(kw))
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: trace)
    demand_rows = [[0, 0]]
    service = module.Exl3RamMissService()
    service._manager = manager
    service.routed_rows_per_step = 12  # 6 routed rows in each of the two layers
    counters = {"rows_read": 0}
    service.host = SimpleNamespace(
        fatal_seq=lambda: 0, layer_rows=lambda: demand_rows[0], counters=lambda: dict(counters)
    )

    manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])
    service.fail_stop_check()  # baseline: no line yet
    manager._graph_counters.copy_(torch.tensor([[6, 2], [6, 1]]))  # one replay's gathers
    manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])  # the observer
    assert int(manager._graph_counters.sum()) == 0
    demand_rows[0] = [1, 1]
    counters["rows_read"] = 3  # demand plus advisory rows, cumulative
    service.fail_stop_check()
    assert lines == [dict(layer_rows_delta=[1, 1], routed_rows=12, routed_misses=3, thread={"rows_read": 3}, steps=1)]

    # discard_graph_capture_routes zeroes the registers: no line, a new baseline.
    for register in manager._registers["decode"].values():
        register.zero_()
    service.fail_stop_check()
    assert len(lines) == 1
    manager._graph_counters.copy_(torch.tensor([[6, 1], [6, 0]]))
    manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])
    demand_rows[0] = [2, 1]
    service.fail_stop_check()
    assert lines[1:] == [dict(layer_rows_delta=[1, 0], routed_rows=12, routed_misses=1, thread={"rows_read": 3}, steps=1)]

    # Two replays before one check (the read lagged a step): one line, two steps.
    for _ in range(2):
        manager._graph_counters.copy_(torch.tensor([[6, 1], [6, 1]]))
        manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])
    service.fail_stop_check()
    assert lines[2:] == [dict(layer_rows_delta=[0, 0], routed_rows=24, routed_misses=4, thread={"rows_read": 3}, steps=2)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA snapshot needs a GPU")
def test_graph_trace_defers_cuda_readback_until_a_later_check(monkeypatch):
    from sglang.srt.layers.moe import exl3_stream_trace

    lines = []
    monkeypatch.setattr(
        exl3_stream_trace,
        "get_exl3_stream_trace",
        lambda: SimpleNamespace(enabled=True, record_graph_step=lambda **kw: lines.append(kw)),
    )
    graph_rows = torch.zeros((2, 2), dtype=torch.int64, device="cuda")
    service = module.Exl3RamMissService()
    service._manager = SimpleNamespace(_registers={"decode": {"graph_rows": graph_rows}})
    service.routed_rows_per_step = 12
    demand_rows = [[0, 0]]
    service.host = SimpleNamespace(
        fatal_seq=lambda: 0, layer_rows=lambda: demand_rows[0], counters=lambda: {"rows_read": 3}
    )

    service.fail_stop_check()  # enqueue baseline; no device readback on this call
    torch.cuda.synchronize()
    graph_rows.copy_(torch.tensor([[6, 2], [6, 1]], device="cuda"))
    demand_rows[0] = [1, 1]
    service.fail_stop_check()  # consume baseline, enqueue a fresh snapshot
    assert lines == []
    torch.cuda.synchronize()
    service.fail_stop_check()
    assert lines == [
        dict(layer_rows_delta=[1, 1], routed_rows=12, routed_misses=3, thread={"rows_read": 3}, steps=1)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA snapshot needs a GPU")
def test_graph_trace_coalesces_busy_slot_and_flushes_final_sample(monkeypatch):
    from sglang.srt.layers.moe import exl3_stream_trace

    lines = []
    monkeypatch.setattr(
        exl3_stream_trace,
        "get_exl3_stream_trace",
        lambda: SimpleNamespace(enabled=True, record_graph_step=lambda **kw: lines.append(kw)),
    )
    real_event = torch.cuda.Event

    class DelayedEvent:
        def __init__(self, **kwargs):
            self.event = real_event(**kwargs)
            self.allow_query = False

        def record(self, stream):
            self.event.record(stream)

        def query(self):
            return self.allow_query and self.event.query()

        def synchronize(self):
            self.event.synchronize()

    delayed = DelayedEvent(enable_timing=False)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: delayed)
    graph_rows = torch.zeros((2, 2), dtype=torch.int64, device="cuda")
    service = module.Exl3RamMissService()
    service._manager = SimpleNamespace(_registers={"decode": {"graph_rows": graph_rows}})
    service.routed_rows_per_step = 12
    demand_rows = [[0, 0]]
    service.host = SimpleNamespace(layer_rows=lambda: demand_rows[0], counters=lambda: {"rows_read": 7})

    service._trace_step()  # baseline snapshot is in flight
    for step in (1, 2):
        graph_rows.copy_(torch.tensor([[6 * step, 2 * step], [6 * step, step]], device="cuda"))
        demand_rows[0] = [step, step]
        service._trace_step()
    assert lines == []

    delayed.allow_query = True
    torch.cuda.synchronize()
    service._trace_step()  # consume old baseline; queue current cumulative totals
    torch.cuda.synchronize()
    service._trace_step()
    assert lines == [
        dict(layer_rows_delta=[2, 2], routed_rows=24, routed_misses=6, thread={"rows_read": 7}, steps=2)
    ]

    graph_rows.copy_(torch.tensor([[18, 6], [18, 3]], device="cuda"))
    demand_rows[0] = [3, 3]
    torch.cuda.synchronize()  # orderly shutdown establishes GPU completion first
    service._trace_step(final=True)
    assert lines[-1] == dict(
        layer_rows_delta=[1, 1], routed_rows=12, routed_misses=3, thread={"rows_read": 7}, steps=1
    )


def test_apply_graph_pads_the_routes_past_the_routed_ids_with_minus_one():
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    class Gathered(Exception):
        pass

    def gather(topk_ids):
        raise Gathered  # stop before the fused MoE: only the routes copy is under test

    backend = SimpleNamespace(routes=torch.full((6,), 7, dtype=torch.int64), keep=torch.ones(1))
    streamer = SimpleNamespace(row_backend=backend, gather=gather)
    ids = torch.tensor([[5, 2, 9, 0]], dtype=torch.int32)
    with pytest.raises(Gathered):
        Exl3MoEMethod._apply_graph(SimpleNamespace(layer_id=0), streamer, torch.zeros((1, 8)), torch.ones((1, 4)), ids, 10.0)
    assert backend.routes.tolist() == [5, 2, 9, 0, -1, -1]


def test_the_row_backend_hands_the_kernels_at_least_eight_planned_lanes():
    # The post kernel reads planned for min(count, 8) lanes and count lives on the
    # device, so a 6-lane plan (graph_gather_rows = top-6) must not be passed as is.
    from sglang.kernels.ops.moe.exl3_ram_miss import MAX_IDS

    calls = []
    device_side = SimpleNamespace(
        post=lambda row, planned, count, routes, next_row, hot_slots, hot_capacity: calls.append(("post", planned.clone())),
        wait=lambda row, planned, count, host_rows, keep, ram_miss: calls.append(("wait", planned.clone())),
    )
    backend = module.Exl3RamMissRowBackend({0: None}, torch.full((EXPERTS,), -1, dtype=torch.int64), device_side, 0, -1, 6)
    assert backend.planned.numel() >= MAX_IDS
    plan = SimpleNamespace(expert_ids=torch.tensor([4, 2, 5, 0, 0, 0]), count=torch.tensor([3], dtype=torch.int32))
    backend.translate(0, plan)
    for name, planned in calls:
        assert planned.numel() >= MAX_IDS, name
        assert planned.tolist() == [4, 2, 5, 0, 0, 0] + [-1] * (planned.numel() - 6), name
    assert [name for name, _ in calls] == ["post", "wait"]


def test_a_test_fault_spec_parses():
    assert module.parse_fault("") is None
    assert module.parse_fault("40:20") == (40, 20.0)


def test_apply_graph_refuses_the_p3_only_backend(tiers):
    from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    service, streamers, caches = tiers
    streamer = streamers[0]
    streamer.row_backend = PinnedTierRowBackend({0: None}, torch.full((6,), -1, dtype=torch.int64), 6)
    with pytest.raises(RuntimeError, match="option C"):
        Exl3MoEMethod._apply_graph(streamer.layer, streamer, torch.zeros((1, 8)), torch.ones((1, 6)), torch.zeros((1, 6), dtype=torch.long), 10.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def test_stage_records_are_drained_into_the_trace_only_when_traced(monkeypatch):
    from sglang.srt.layers.moe import exl3_stream_trace

    sent = []
    trace = SimpleNamespace(enabled=True, record_ram_miss_requests=lambda records, layer_ids: sent.append((records, layer_ids)))
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: trace)
    drained = []
    service = module.Exl3RamMissService()
    service._rows = {7: 0, 3: 1}  # layer id -> row
    service.host = SimpleNamespace(
        drain_trace=lambda: drained.append(1) or [{"row": 1}], trace_dropped=lambda: 0
    )
    service._trace_stages()  # not enabled on the host: it must not even drain
    assert drained == [] and sent == []
    service._stages_traced = True
    service._trace_stages()
    assert sent == [([{"row": 1}], [7, 3])]


def _copy_engine_service(monkeypatch):
    """A service with only what the copy engine's arming and guards read (LEASE_PROTOCOL.md 7.6)."""
    service = module.Exl3RamMissService()
    service.copy_engine = True
    service.device_side = SimpleNamespace(copy_engine_captured=True)
    armed = []
    service.host = SimpleNamespace(arm_copy_engine=lambda: armed.append(True))
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return service, armed, syncs


def _batch(decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode))


def test_the_copy_engine_arms_only_after_enough_decode_forwards_not_batches(monkeypatch):
    """Counting batches armed it inside the server's warm-up request, whose first decode steps then deadlocked;
    eager forwards must not count, and nothing counts before a copy-engine graph was captured."""
    service, armed, syncs = _copy_engine_service(monkeypatch)
    service.device_side.copy_engine_captured = False
    service._copy_engine_barrier(0, _batch(decode=True))
    assert service._copy_decodes == 0
    service.device_side.copy_engine_captured = True
    for _ in range(3 * module.COPY_ENGINE_ARM_DECODES):
        service._copy_engine_barrier(0, _batch(decode=False))
        service._arm_copy_engine()
    assert not armed and not syncs, "eager forwards armed the copy engine, or drained the device before arming"
    for _ in range(module.COPY_ENGINE_ARM_DECODES - 1):
        service._copy_engine_barrier(0, _batch(decode=True))
        service._arm_copy_engine()
    assert not armed
    service._copy_engine_barrier(0, _batch(decode=True))
    service._arm_copy_engine()
    service._arm_copy_engine()
    assert armed == [True] and service._copy_armed
    service._copy_engine_barrier(0, _batch(decode=True))
    assert not syncs, "a decode forward drained the device"
    service._copy_engine_barrier(0, _batch(decode=False))
    assert syncs == [True], "an eager forward did not drain the device once armed"


@pytest.mark.parametrize("value", [None, "", "LAZY", "eager", "DEFAULT"])
def test_the_copy_engine_refuses_any_module_loading_but_eager(value):
    """A kernel loaded lazily after arming fail-stopped the copy-engine soak on the same decode step every time, and
    never under EAGER. torch sets LAZY when the variable is unset, so unset is refused too."""
    environ = {} if value is None else {"CUDA_MODULE_LOADING": value}
    with pytest.raises(RuntimeError, match="CUDA_MODULE_LOADING=EAGER"):
        module.check_copy_engine_module_loading(environ)
    module.check_copy_engine_module_loading({"CUDA_MODULE_LOADING": "EAGER"})


def test_a_jit_library_load_drains_the_device_first_once_armed(monkeypatch):
    """Under EAGER, tvm-ffi's load_module (sglang's load_jit, flashinfer's JIT) loads a library's kernels at once: the
    same hazard as a Triton load, so the guard wraps it too."""
    import sys

    service, armed, syncs = _copy_engine_service(monkeypatch)
    order = []
    fake = SimpleNamespace(load_module=lambda path: order.append(("load", path)) or "module")
    monkeypatch.setitem(sys.modules, "tvm_ffi", fake)
    monkeypatch.setitem(sys.modules, "triton.runtime", None)  # no Triton: the guard must still wrap tvm-ffi
    service._install_copy_engine_module_load_guard()
    assert fake.load_module("a.so") == "module" and not syncs
    service._copy_armed = True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("sync"))
    assert fake.load_module("b.so") == "module"
    assert order[-2:] == ["sync", ("load", "b.so")]
    assert service.copy_engine_module_loads == 1


def test_a_triton_module_load_drains_the_device_first_once_armed(monkeypatch):
    """cuModuleLoadData holds the driver lock the copy thread needs and waits for the device, which may spin in a copy
    wait for that very copy (diag arm1eager83-on). The guard drains before the load, and never inside a capture."""
    service, armed, syncs = _copy_engine_service(monkeypatch)
    order = []
    load = service._copy_engine_module_load_guard(lambda *a, **k: order.append(("load", a, k)) or "handle")
    assert load("name", b"cubin", shared=0) == "handle" and not syncs
    service._copy_armed = True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("sync"))
    assert load("name", b"cubin") == "handle"
    assert order[-2:] == ["sync", ("load", ("name", b"cubin"), {})]
    assert service.copy_engine_module_loads == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    order.clear()
    load("name", b"cubin")
    assert order == [("load", ("name", b"cubin"), {})], "the guard synchronized during a capture"
