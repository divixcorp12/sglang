"""Option C wiring on the host: the native slot table under the pinned tier, the fail-stop
check, residency pushes and the per-step graph trace (CPU; the thread runs, no device)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import sim_post, sim_wait
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


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
            fmt = Exl3ExpertFormat(layout, layer_id, direct=False)
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


def test_a_later_promotion_chunk_never_evicts_an_expert_an_earlier_chunk_made_hot(tiers):
    """Minor 1: the hot cache reserves chunk 1's experts before the residency listener
    pushes them to C++. Chunk 2's admission (its own host use) must still protect them,
    exactly as ``is_pinned`` does when it sizes the chunk."""
    service, streamers, caches = tiers
    cache = caches[0]
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
        post=lambda row, planned, count, routes, next_row: calls.append(("post", planned.clone())),
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
