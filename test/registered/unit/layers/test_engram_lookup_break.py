"""Engram's file-table lookup is an eager break under a breakable graph capture (CPU)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import engram
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


DIM = 4


class _Table:
    """Returns ``2 * index`` on the first call and ``3 * index`` afterwards; raises on ids out of range."""

    dim = DIM
    rows = 10

    def __init__(self):
        self.calls = 0

    def lookup(self, indices):
        self.calls += 1
        assert int(indices.max()) < self.rows, "host lookup read an id outside the table"
        scale = 2 if self.calls == 1 else 3
        return (indices.float().unsqueeze(-1) * scale).expand(*indices.shape, DIM).to(torch.bfloat16)


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


def test_outside_capture_the_lookup_runs_directly():
    table = _Table()
    out = engram._engram_file_table_lookup(table, torch.tensor([[1, 2]]))
    assert table.calls == 1 and out.tolist() == [[[2.0] * DIM, [4.0] * DIM]]


def test_under_capture_the_stub_runs_not_the_table_and_a_replay_copies_into_out():
    capture, events = _fake_capture()
    table = _Table()
    # Ids the ended segment never computed: out of range for the table.
    garbage = torch.tensor([[10**9, -7]])
    with _capturing(capture):
        out = engram._engram_file_table_lookup(table, garbage)
    assert events == ["end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 1
    assert table.calls == 0  # the capture-time call never touches the table
    assert out.shape == (1, 2, DIM) and out.dtype == torch.bfloat16
    assert not out.any()
    # A replay re-runs the real lookup (the fake capture's inputs are live tensors) and copies into `out`.
    garbage.copy_(torch.tensor([[3, 5]]))
    capture.cuda_graph._break_fns[0]()
    assert table.calls == 1
    assert out.tolist() == [[[6.0] * DIM, [10.0] * DIM]]


def test_the_embedding_forward_goes_through_the_break():
    capture, events = _fake_capture()
    table = _Table()
    module = SimpleNamespace(file_table=table)
    with _capturing(capture):
        out = engram.EngramEmbedding.forward(module, torch.tensor([[5]]))
    assert events == ["end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 1
    assert table.calls == 0 and out.shape == (1, 1, DIM)


def test_layer14_without_native_route_keeps_the_eager_break_and_lookup():
    capture, events = _fake_capture()
    table = _Table()
    module = SimpleNamespace(
        file_table=table,
        layer_id=14,
        tp_size=1,
    )
    decode = SimpleNamespace(is_decode=lambda: True)
    batch = SimpleNamespace(forward_mode=decode)
    with _capturing(capture):
        out = engram.EngramEmbedding.forward(module, torch.tensor([[5]]), batch)
    assert events == ["end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 1
    assert table.calls == 0 and out.shape == (1, 1, DIM)

    eager = engram.EngramEmbedding.forward(module, torch.tensor([[5]]), batch)
    assert eager.tolist() == [[[10.0] * DIM]]
    assert table.calls == 1


def test_outside_capture_the_embedding_forward_looks_up():
    table = _Table()
    module = SimpleNamespace(file_table=table)
    out = engram.EngramEmbedding.forward(module, torch.tensor([[5]]))
    assert out.tolist() == [[[10.0] * DIM]] and table.calls == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
