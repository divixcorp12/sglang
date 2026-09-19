"""The mHC stats side stream is re-forked into the current breakable segment (CPU).

A streamed-MoE eager break ends the capture segment the stats stream was forked into and
joins it. The FFN stats then launch on that stream after the break, so without a fresh
fork their kernels run outside the capture and the join back is dropped."""

import sys

import pytest

from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_stream(self, other):
        self.waited.append(other)


def test_under_breakable_capture_the_stats_stream_waits_for_the_current_stream(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "is_in_breakable_cuda_graph", lambda: True)
    stats, main = _Stream(), object()
    deepseek_v4._refork_stats_stream(stats, main)
    assert stats.waited == [main]


def test_outside_breakable_capture_nothing_is_added(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "is_in_breakable_cuda_graph", lambda: False)
    stats = _Stream()
    deepseek_v4._refork_stats_stream(stats, object())
    assert stats.waited == []


def test_no_stats_stream_is_a_no_op(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "is_in_breakable_cuda_graph", lambda: True)
    deepseek_v4._refork_stats_stream(None, object())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
