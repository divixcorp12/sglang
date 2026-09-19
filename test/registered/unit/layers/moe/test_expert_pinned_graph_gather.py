"""Graph gather over a partial pinned host tier: the row backend and the support check (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.expert_stream_requirements import ExpertStreamRequirements
from sglang.srt.layers.moe import expert_row_plan
from sglang.srt.layers.moe.expert_format import graph_source_kind_of, require_graph_gather_support
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan, PinnedTierRowBackend
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
