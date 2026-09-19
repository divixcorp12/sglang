"""An eager break nested in an eager break is transparent (CPU).

``--debug-cuda-graph`` wraps the whole forward as one eager break; the forward's own
breaks (the streamed MoE, the Engram lookup) must then run inline, not open a break of
their own inside the running one."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_capture():
    events = []
    capture = SimpleNamespace(
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
        _barrier_fn=None,
        cuda_graph=SimpleNamespace(_break_fns=[]),
    )
    return capture, events


@contextmanager
def _capturing(capture):
    token = bcg._current_capture_var.set(capture)
    try:
        yield
    finally:
        bcg._current_capture_var.reset(token)


def test_a_break_inside_a_break_runs_inline_and_records_one_break():
    capture, events = _fake_capture()
    calls = []

    @bcg.eager_on_graph(True)
    def inner_break(x):
        calls.append("inner")
        return x + 1

    @bcg.eager_on_graph(True)
    def outer_break(x):
        calls.append("outer")
        return inner_break(x) * 2

    with _capturing(capture):
        out = outer_break(torch.tensor([1.0]))
    assert out.tolist() == [4.0]
    assert calls == ["outer", "inner"]
    assert events == ["end", "begin"]  # one break: the outer one
    assert len(capture.cuda_graph._break_fns) == 1


def test_a_break_after_a_break_still_opens_its_own():
    capture, events = _fake_capture()

    @bcg.eager_on_graph(True)
    def first(x):
        return x + 1

    @bcg.eager_on_graph(True)
    def second(x):
        return x + 2

    with _capturing(capture):
        second(first(torch.tensor([0.0])))
    assert events == ["end", "begin", "end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 2


def test_the_nesting_flag_clears_when_the_break_body_raises():
    capture, events = _fake_capture()

    @bcg.eager_on_graph(True)
    def boom(x):
        raise RuntimeError("x")

    @bcg.eager_on_graph(True)
    def fine(x):
        return x

    with _capturing(capture):
        try:
            boom(torch.tensor([0.0]))
        except RuntimeError:
            pass
        fine(torch.tensor([0.0]))
    assert events == ["end", "end", "begin"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
