"""Graph gather over a partial pinned host tier: the row backend and the support check (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.expert_stream_requirements import ExpertStreamRequirements
from sglang.srt.layers.moe import expert_row_plan
from sglang.srt.layers.moe.expert_format import (
    STREAMER_ATTRIBUTE,
    graph_gather_needs_host_arena,
    graph_source_kind_of,
    require_graph_gather_support,
)
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan, PinnedTierRowBackend
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _plan(ids, count):
    return ExpertRowPlan(
        expert_ids=torch.tensor(ids, dtype=torch.int64),
        slots=torch.arange(len(ids), dtype=torch.int32),
        count=torch.tensor([count], dtype=torch.int32),
    )


@pytest.fixture
def copies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        expert_row_plan,
        "copy_expert_row_segments_gpu",
        lambda segments, rows, slots, count: calls.append((segments, rows.tolist(), slots.tolist(), int(count))),
    )
    return calls


def test_planned_experts_are_copied_from_their_pinned_slots(copies):
    host_map = torch.tensor([-1, 4, -1, 0, -1, 2, -1, -1], dtype=torch.int64)
    backend = PinnedTierRowBackend({0: "segments"}, host_map, 4)
    plan = _plan([3, 5, 1, 7], count=3)  # lane 3 is past the count: its -1 is not a miss
    backend.post(0, plan)
    assert copies == [("segments", [0, 2, 4, 0], [0, 1, 2, 3], 3)]
    assert backend.keep.tolist() == [1.0] and backend.ram_miss.tolist() == [0]
    assert backend.resolve(0, plan).delivered is None


def test_a_row_missing_from_ram_copies_slot_zero_and_drops_the_layer(copies):
    host_map = torch.tensor([-1, 4, -1, 0], dtype=torch.int64)
    backend = PinnedTierRowBackend({0: "segments"}, host_map, 2)
    backend.post(0, _plan([2, 1], count=2))
    assert copies[-1][1] == [0, 4]
    assert backend.keep.tolist() == [0.0] and backend.ram_miss.tolist() == [1]
    backend.post(0, _plan([1, 3], count=2))
    assert backend.keep.tolist() == [1.0] and backend.ram_miss.tolist() == [1]  # cumulative


def test_graph_source_kind_defaults_to_dense():
    assert graph_source_kind_of(SimpleNamespace()) == "dense"
    assert graph_source_kind_of(SimpleNamespace(graph_source_kind="pinned_tier")) == "pinned_tier"


def _streamer(kind, pinned):
    fmt = SimpleNamespace(key="k", supports_graph_gather=False, graph_source_kind=kind)
    return SimpleNamespace(format=fmt, has_spec_only_tensors=True, pinned_host_cache=pinned, layer_id=3)


def test_support_check_accepts_a_pinned_tier_format_only_when_asked():
    require_graph_gather_support([_streamer("pinned_tier", object())], pinned_tier_ok=True)
    with pytest.raises(ValueError, match="does not support graph gather"):
        require_graph_gather_support([_streamer("pinned_tier", object())])
    with pytest.raises(ValueError, match="pinned host tier"):
        require_graph_gather_support([_streamer("pinned_tier", None)], pinned_tier_ok=True)
    with pytest.raises(ValueError, match="does not support graph gather"):
        require_graph_gather_support([_streamer("dense", object())], pinned_tier_ok=True)


def test_requirements_default_to_the_host_arena():
    assert ExpertStreamRequirements("X", lambda cfg, budgets: None).graph_gather_host_source == "arena"


def test_requirements_refuse_an_unknown_host_source():
    ExpertStreamRequirements("X", lambda cfg, budgets: None, graph_gather_host_source="pinned_tier")
    with pytest.raises(ValueError, match="graph_gather_host_source"):
        ExpertStreamRequirements("X", lambda cfg, budgets: None, graph_gather_host_source="pinned-tier")


def _model(*kinds):
    model = torch.nn.Sequential(*(torch.nn.Module() for _ in kinds))
    for module, kind in zip(model, kinds):
        fmt = SimpleNamespace() if kind is None else SimpleNamespace(graph_source_kind=kind)
        setattr(module, STREAMER_ATTRIBUTE, SimpleNamespace(format=fmt))
    return model


def test_only_an_all_pinned_tier_model_skips_the_host_arena():
    assert not graph_gather_needs_host_arena(_model("pinned_tier", "pinned_tier"))
    assert graph_gather_needs_host_arena(_model("pinned_tier", None))
    assert graph_gather_needs_host_arena(_model("dense"))
    assert not graph_gather_needs_host_arena(torch.nn.Linear(2, 2))  # nothing streamed


def _graph_checked(slab, slot_map):
    pinned = SimpleNamespace(tensors={"w": slab}, expert_to_slot=slot_map)
    return SimpleNamespace(
        _graph_sources={"w": slab},
        _graph_pinned_tier=True,
        pinned_host_cache=pinned,
        row_backend=SimpleNamespace(host_row_map=slot_map),
    )


def test_a_rebound_pinned_slot_map_is_caught():
    streamer = _graph_checked(torch.zeros(2, 3), torch.full((4,), -1, dtype=torch.int64))
    ExpertStreamer._check_graph_sources(streamer)
    streamer.pinned_host_cache.expert_to_slot = torch.full((4,), -1, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="slot map"):
        ExpertStreamer._check_graph_sources(streamer)


def test_a_rebound_pinned_slab_is_caught():
    streamer = _graph_checked(torch.zeros(2, 3), torch.full((4,), -1, dtype=torch.int64))
    streamer.pinned_host_cache.tensors = {"w": torch.zeros(2, 3)}
    with pytest.raises(RuntimeError, match="moved after graph gather"):
        ExpertStreamer._check_graph_sources(streamer)


def test_the_hot_cache_manager_refuses_a_pinned_tier_under_residency_update_or_doorbell():
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
    from sglang.test.moe_expert_fakes import SpecOnlyFormat

    reference = {"w13_trellis": torch.arange(8 * 6, dtype=torch.int16).reshape(8, 6)}
    fmt = SpecOnlyFormat(reference)
    fmt.graph_source_kind = "pinned_tier"
    layer = torch.nn.Module()
    layer.layer_id = 0
    streamer = ExpertStreamer(layer, tuple(reference), format=fmt)
    ExpertPinnedHostCache(streamer, 2, device="cpu")
    assert streamer.pinned_host_cache is not None
    setattr(layer, STREAMER_ATTRIBUTE, streamer)
    common = dict(
        budget_bytes=1 << 20,
        seed_path=None,
        dynamic=False,
        update_prefill_tokens=16,
        min_residence_forwards=0,
        benefit_ratio=1.0,
    )
    for flags in (dict(gpu_residency_update=True), dict(expert_doorbell=True)):
        with pytest.raises(ValueError, match="does not support graph gather"):
            ExpertHotCacheManager.from_model(torch.nn.Sequential(layer), **common, **flags)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
