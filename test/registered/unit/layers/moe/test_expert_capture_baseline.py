"""discard_graph_capture_routes re-baselines the eager counters after CUDA-graph capture (CPU)."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_discarding_capture_routes_rebaselines_the_eager_counters():
    capture_stats = object()
    pinned = SimpleNamespace(stats=SimpleNamespace(populated_rows=7, evictions=3))
    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager.streamers = {
        2: SimpleNamespace(last_gather_stats=capture_stats, pinned_host_cache=pinned),
        5: SimpleNamespace(last_gather_stats=capture_stats, pinned_host_cache=None),
    }
    manager._graph_counters = None
    manager._registers = {}
    manager.residency_policies = {}
    manager._side_pull_snapshots = {}
    manager._last_side_pull_totals = {}
    manager._inflight_promotions = []
    manager._last_gather = {2: None, 5: None}
    manager._last_pinned_cache_stats = {2: (0, 0), 5: (0, 0)}

    manager.discard_graph_capture_routes()

    assert manager._last_gather == {2: capture_stats, 5: capture_stats}
    assert manager._last_pinned_cache_stats == {2: (7, 3), 5: (0, 0)}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
