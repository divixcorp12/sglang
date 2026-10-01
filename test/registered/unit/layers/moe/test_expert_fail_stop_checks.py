"""The hot-cache manager runs registered fail-stop checks and offers formats the finished manager (CPU)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe import expert_hot_cache
from sglang.srt.layers.moe.expert_hot_cache import (
    ExpertHotCacheManager,
    HotCacheUpdateStats,
)
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _manager():
    return ExpertHotCacheManager.__new__(ExpertHotCacheManager)


def test_registered_checks_run_in_registration_order():
    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("a"))
    manager.register_fail_stop_check(lambda: calls.append("b"))
    manager.run_fail_stop_checks()
    assert calls == ["a", "b"]


def test_a_failing_check_raises_through_the_scheduler_hook():
    manager = _manager()

    def fail():
        raise RuntimeError("fail-stop")

    manager.register_fail_stop_check(fail)
    with pytest.raises(RuntimeError, match="fail-stop"):
        manager.run_fail_stop_checks()


def test_a_manager_without_checks_still_returns():
    assert _manager().run_fail_stop_checks() is None


def _scheduler_stub(manager):
    return SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(expert_hot_cache_manager=manager)
        )
    )


def test_the_scheduler_hook_runs_the_managers_checks():
    """The real, unbound Scheduler method on a stub. EXL3's RAM-miss service registers
    its per-batch fail-stop here (exl3_ram_miss.py), and this hook is its only caller."""
    from sglang.srt.managers.scheduler import Scheduler

    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("exl3"))
    Scheduler._run_expert_fail_stop_checks(_scheduler_stub(manager))
    assert calls == ["exl3"]


def test_the_scheduler_hook_does_nothing_without_a_manager():
    from sglang.srt.managers.scheduler import Scheduler

    Scheduler._run_expert_fail_stop_checks(_scheduler_stub(None))
    Scheduler._run_expert_fail_stop_checks(SimpleNamespace())


def test_every_batch_result_reaches_the_fail_stop_hook():
    """Wiring: process_batch_result must call the hook for every forward mode (Review Focus 1)."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    assert "self._run_expert_fail_stop_checks()" in inspect.getsource(
        Scheduler.process_batch_result
    )


def test_formats_are_offered_the_manager():
    manager = _manager()
    manager.caches = {1: SimpleNamespace(slot_to_expert=[4])}
    events = []

    class _Format:
        def attach_hot_cache_manager(self, mgr, streamer):
            events.append(("attach", streamer.layer_id))

    manager.streamers = {
        1: SimpleNamespace(format=_Format(), layer_id=1),
        2: SimpleNamespace(format=SimpleNamespace(), layer_id=2),
    }
    manager._attach_formats()
    assert events == [("attach", 1)]


def test_formats_are_attached_once():
    manager = _manager()
    manager.caches = {1: SimpleNamespace(slot_to_expert=[4])}
    attached = []

    class _Format:
        def attach_hot_cache_manager(self, mgr, streamer):
            attached.append(streamer.layer_id)

    manager.streamers = {1: SimpleNamespace(format=_Format(), layer_id=1)}
    manager._attach_formats()
    manager._attach_formats()
    assert attached == [1]


EXPERTS = 8


class _FakeHotCache:
    """Stands in for ExpertHotCache, which needs CUDA, in from_model."""

    def __init__(self, streamer, capacity, scratch_rows=0):
        self.streamer = streamer
        self.capacity = capacity
        self.device = torch.device("cpu")
        self.capacity_bytes = capacity * streamer.bytes_per_expert
        self.allocation_bytes = self.capacity_bytes
        self.scratch_bytes = 0
        self.prefetch_pull_bytes = 0
        self.last_copy_submission = None
        self.slot_to_expert = [-1] * capacity

    def reassign(self, expert_ids):
        experts = list(expert_ids)
        self.slot_to_expert = experts + [-1] * (self.capacity - len(experts))
        return HotCacheUpdateStats(len(experts), 0, 0)


class _AttachingFormat(SpecOnlyFormat):
    def __init__(self, reference, events):
        super().__init__(reference, tier_options={"device": "cpu"})
        self.events = events

    def attach_hot_cache_manager(self, manager, streamer):
        # The manager is finished: every cache exists.
        self.events.append(("attach", streamer.layer_id, sorted(manager.caches)))


def test_from_model_attaches_formats_once_every_cache_exists():
    events = []
    model = torch.nn.Module()
    for layer_id in range(2):
        reference = {
            "w13_trellis": torch.arange(EXPERTS * 6, dtype=torch.int16).reshape(
                EXPERTS, 1, 6
            )
        }
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(
            layer, tuple(reference), format=_AttachingFormat(reference, events)
        )
        ExpertPinnedHostCache(layer._nvfp4_expert_streamer, EXPERTS, device="cpu")
        model.add_module(str(layer_id), layer)
    streamer = model.get_submodule("0")._nvfp4_expert_streamer
    with (
        patch.object(expert_hot_cache, "ExpertHotCache", _FakeHotCache),
        patch("torch.cuda.memory_allocated", return_value=0),
        patch("torch.cuda.memory_reserved", return_value=0),
    ):
        manager = ExpertHotCacheManager.from_model(
            model,
            budget_bytes=4 * streamer.bytes_per_expert,
            seed_path=None,
            dynamic=False,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=1.0,
        )
    assert events == [("attach", 0, [0, 1]), ("attach", 1, [0, 1])]
    assert not hasattr(manager, "add_residency_listener")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
