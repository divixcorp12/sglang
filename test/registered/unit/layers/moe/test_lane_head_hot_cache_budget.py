"""The lane head (SGLANG_OPT_LANE_HEAD_PATH) with the startup hot-cache calibration (SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP)
on together (CPU; the deferred both-flags check, MERGE_PLAN_1016.md).

The two meet in the hot-cache budget: the head's device bytes (~57.5 MB at 40 layers) come out of
SGLANG_EXPERT_STREAM_HOT_GPU_MB before the cache is sized, and the head is allocated in from_model, at model load, so the
calibration's later free-memory measurement sees it. Calibration then resizes through set_hot_capacity, whose census
must not miss a holder of the capacity: the head keys experts, not slots, so it holds none, and it resets its
per-request state after capture, so the calibration's dummy forwards and trial capture leave nothing behind.
"""

import inspect
import re
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe import expert_hot_cache, lane_head
from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager, HotCacheUpdateStats
from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

EXPERTS = 8
HEAD_SLOTS = 2  # the head's bytes in hot-cache slots (57.5 MB is ~4.3 of the real 13.3 MB slots)


class _RecordingHotCache:
    """Stands in for ExpertHotCache (which needs CUDA) and records what from_model sized."""

    made: list = []

    def __init__(self, streamer, capacity, scratch_rows=0, upper_capacity=None):
        self.streamer = streamer
        self.capacity = capacity
        self.upper_capacity = upper_capacity
        self.device = torch.device("cpu")
        self.capacity_bytes = capacity * streamer.bytes_per_expert
        self.allocation_bytes = self.capacity_bytes
        self.scratch_bytes = 0
        self.prefetch_pull_bytes = 0
        self.last_copy_submission = None
        self.slot_to_expert = [-1] * capacity
        _RecordingHotCache.made.append(self)

    def reassign(self, expert_ids):
        experts = list(expert_ids)
        self.slot_to_expert = experts + [-1] * (self.capacity - len(experts))
        return HotCacheUpdateStats(len(experts), 0, 0)


def _model():
    model = torch.nn.Module()
    for layer_id in range(2):
        reference = {"w13_trellis": torch.arange(EXPERTS * 6, dtype=torch.int16).reshape(EXPERTS, 1, 6)}
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, tuple(reference), format=SpecOnlyFormat(reference))
        ExpertPinnedHostCache(layer._nvfp4_expert_streamer, EXPERTS, device="cpu")
        model.add_module(str(layer_id), layer)
    return model


def _sized_slots(monkeypatch, head: bool) -> tuple[int, list]:
    """The slots from_model gives the caches with SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP on, with or without a head."""
    monkeypatch.setenv("SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP", "1")
    model = _model()
    slot = model.get_submodule("0")._nvfp4_expert_streamer.bytes_per_expert
    if head:
        monkeypatch.setenv("SGLANG_OPT_LANE_HEAD_PATH", "/nonexistent/head.safetensors")
        monkeypatch.setattr(lane_head, "read_lane_head_meta", lambda path: object())
        monkeypatch.setattr(lane_head, "lane_head_device_bytes", lambda meta: HEAD_SLOTS * slot)
    else:
        monkeypatch.delenv("SGLANG_OPT_LANE_HEAD_PATH", raising=False)
    _RecordingHotCache.made = []
    with (
        patch.object(expert_hot_cache, "ExpertHotCache", _RecordingHotCache),
        # The CPU path of the upper bound (the pinned-tier limits); on a GPU box the bound also reads the launch's
        # --mem-fraction-static, which a unit test has not published.
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.memory_allocated", return_value=0),
        patch("torch.cuda.memory_reserved", return_value=0),
    ):
        call = lambda: ExpertHotCacheManager.from_model(  # noqa: E731
            model,
            budget_bytes=8 * slot,
            seed_path=None,
            dynamic=False,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=1.0,
        )
        if head:
            # This harness has no DIRECT residency, so the head itself is refused after the caches are sized; the
            # budget is what this test reads.
            with pytest.raises(ValueError, match="needs DIRECT residency"):
                call()
        else:
            call()
    return sum(cache.capacity for cache in _RecordingHotCache.made), list(_RecordingHotCache.made)


def test_with_both_flags_the_hot_cache_is_sized_from_the_budget_less_the_heads_bytes(monkeypatch):
    without, _ = _sized_slots(monkeypatch, head=False)
    with_head, caches = _sized_slots(monkeypatch, head=True)
    assert without > HEAD_SLOTS
    assert with_head == without - HEAD_SLOTS
    # Calibration may still grow each layer toward its upper bound from what it measures (the head included).
    assert all(cache.upper_capacity is not None and cache.upper_capacity >= cache.capacity for cache in caches)


def test_a_head_that_leaves_no_budget_is_refused(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_LANE_HEAD_PATH", "/nonexistent/head.safetensors")
    monkeypatch.setattr(lane_head, "read_lane_head_meta", lambda path: object())
    monkeypatch.setattr(lane_head, "lane_head_device_bytes", lambda meta: 1 << 40)
    with patch.object(expert_hot_cache, "ExpertHotCache", _RecordingHotCache):
        with pytest.raises(ValueError, match="leave no hot cache budget"):
            ExpertHotCacheManager.from_model(
                _model(),
                budget_bytes=1 << 20,
                seed_path=None,
                dynamic=False,
                update_prefill_tokens=16,
                min_residence_forwards=0,
                benefit_ratio=1.0,
            )


def test_the_head_holds_no_capacity_for_the_calibrations_resize_census():
    # set_hot_capacity lowers or raises every holder of a slot count; the head keys experts (layers x experts), so a
    # resize leaves it valid and the census has nothing of its to find.
    for cls in (lane_head.LaneHead, lane_head.LaneHeadGather):
        assert not re.search(r"capacity", inspect.getsource(cls)), cls.__name__


def test_capture_resets_the_heads_state_after_the_dummy_gathers():
    # Calibration's dummy forwards and trial capture run the head on dummy routes before the real capture; the real
    # capture's reset_after_capture (and every prefill) zeroes the per-request state they touched.
    source = inspect.getsource(GpuResidencyUpdater.reset_after_capture)
    assert "self.lane_head.reset()" in source
    reset = inspect.getsource(lane_head.LaneHead.reset)
    assert all(name in reset for name in ("self.score", "self.last", "self.verify_ct"))
