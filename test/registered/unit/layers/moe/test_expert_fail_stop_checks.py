"""The hot-cache manager runs registered fail-stop checks and residency listeners (CPU)."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _manager():
    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager.doorbell = None
    return manager


def test_registered_checks_run_before_the_doorbell_check():
    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("a"))
    manager.register_fail_stop_check(lambda: calls.append("b"))
    assert manager.doorbell_fail_stop_check(synchronize=True) == 0.0
    assert calls == ["a", "b"]


def test_a_failing_check_raises_through_the_scheduler_hook():
    manager = _manager()

    def fail():
        raise RuntimeError("fail-stop")

    manager.register_fail_stop_check(fail)
    with pytest.raises(RuntimeError, match="fail-stop"):
        manager.doorbell_fail_stop_check(synchronize=True)


def test_a_manager_without_checks_still_returns():
    assert _manager().doorbell_fail_stop_check() == 0.0


def test_formats_are_offered_the_manager_then_residency_is_pushed():
    manager = _manager()
    manager.caches = {1: SimpleNamespace(slot_to_expert=[4])}
    events = []

    class _Format:
        def attach_hot_cache_manager(self, mgr, streamer):
            events.append(("attach", streamer.layer_id))
            mgr.add_residency_listener(
                lambda layer_id, experts: events.append(("hot", layer_id, experts))
            )

    manager.streamers = {
        1: SimpleNamespace(format=_Format(), layer_id=1),
        2: SimpleNamespace(format=SimpleNamespace(), layer_id=2),
    }
    manager._attach_formats()
    assert events == [("attach", 1), ("hot", 1, [4])]


def test_residency_listeners_get_every_layers_resident_experts():
    manager = _manager()
    manager.caches = {
        2: SimpleNamespace(slot_to_expert=[5, -1]),
        7: SimpleNamespace(slot_to_expert=[1]),
    }
    seen = []
    manager.add_residency_listener(
        lambda layer_id, experts: seen.append((layer_id, experts))
    )
    manager._notify_residency_listeners()
    assert sorted(seen) == [(2, [5, -1]), (7, [1])]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
