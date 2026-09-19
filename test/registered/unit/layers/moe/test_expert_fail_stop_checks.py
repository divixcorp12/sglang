"""The hot-cache manager runs registered fail-stop checks and residency listeners (CPU)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe import expert_hot_cache
from sglang.srt.layers.moe.expert_hot_cache import (
    ExpertHotCacheManager,
    HotCacheUpdateStats,
)
from sglang.srt.layers.moe.expert_residency_clock import ResidencyBoundaryClock
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

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
    manager.doorbell = SimpleNamespace(
        fail_stop_check=lambda synchronize: (
            calls.append(("doorbell", synchronize)) or 1.5
        )
    )
    calls.clear()
    assert manager.doorbell_fail_stop_check(synchronize=True) == 1.5
    assert calls == ["a", "b", ("doorbell", True)]


def test_the_doorbell_check_still_runs_without_registered_checks():
    manager = _manager()
    calls = []
    manager.doorbell = SimpleNamespace(
        fail_stop_check=lambda synchronize: calls.append(synchronize) or 0.25
    )
    assert manager.doorbell_fail_stop_check(synchronize=False) == 0.25
    assert calls == [False]


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


def test_residency_listeners_are_refused_under_the_gpu_residency_updater():
    # The updater changes residency without _update_residency, so a listener
    # would never hear of it.
    manager = _manager()
    manager.caches = {1: SimpleNamespace(slot_to_expert=[4])}
    manager.gpu_residency = object()

    class _Format:
        def attach_hot_cache_manager(self, mgr, streamer):
            mgr.add_residency_listener(lambda layer_id, experts: None)

    manager.streamers = {1: SimpleNamespace(format=_Format(), layer_id=1)}
    with pytest.raises(ValueError, match="residency listener"):
        manager._attach_formats()


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
        # The manager is finished: every cache exists and the doorbell was decided.
        self.events.append(
            ("attach", streamer.layer_id, sorted(manager.caches), manager.doorbell)
        )
        if not getattr(manager, "residency_listeners", None):
            manager.add_residency_listener(
                lambda layer_id, experts: self.events.append(
                    ("hot", layer_id, list(experts))
                )
            )


def test_from_model_attaches_formats_then_pushes_residency():
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
    assert events[:2] == [("attach", 0, [0, 1], None), ("attach", 1, [0, 1], None)]
    assert events[2:] == [
        ("hot", layer_id, list(manager.caches[layer_id].slot_to_expert))
        for layer_id in manager.caches
    ]
    assert len(events) == 4


def _batch(mode, tokens=0):
    return SimpleNamespace(
        forward_mode=mode, extend_num_tokens=tokens, batch_size=1, spec_info=None
    )


def test_a_residency_boundary_notifies_listeners_after_the_update():
    manager = _manager()
    events = []
    manager._trace_telemetry = {}
    manager._inflight_promotions = []
    manager.streamers = {}
    manager._layer_ids = []
    manager._boundary_clock = ResidencyBoundaryClock(16, 0)
    manager._accumulate_registers = lambda mode, counts, gathered: None
    manager.gpu_residency = None
    manager.log_interval = 1 << 30
    manager.caches = {3: SimpleNamespace(slot_to_expert=[6, 2])}
    manager._update_residency = lambda tokens, mode: events.append("update")
    manager.add_residency_listener(
        lambda layer_id, experts: events.append(("hot", layer_id, experts))
    )
    counts = {"global_physical_count": torch.zeros(1, EXPERTS)}

    manager.on_expert_distribution(_batch(ForwardMode.DECODE), counts)
    assert events == []  # decode never qualifies without update_decode_forwards
    manager.on_expert_distribution(_batch(ForwardMode.EXTEND, tokens=4), counts)
    assert events == []  # a prefill below update_prefill_tokens
    manager.on_expert_distribution(_batch(ForwardMode.EXTEND, tokens=40), counts)
    assert events == ["update", ("hot", 3, [6, 2])]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
