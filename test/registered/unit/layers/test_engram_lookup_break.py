"""Engram's file-table lookup is an eager break under a breakable graph capture (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import engram
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Table:
    def __init__(self):
        self.calls = 0

    def lookup(self, indices):
        self.calls += 1
        return indices.float().unsqueeze(-1) * 2


def test_outside_capture_the_lookup_runs_directly():
    table = _Table()
    out = engram._engram_file_table_lookup(table, torch.tensor([[1, 2]]))
    assert table.calls == 1 and out.tolist() == [[[2.0], [4.0]]]


def test_under_capture_the_lookup_ends_the_segment_and_records_a_replay():
    events = []
    capture = SimpleNamespace(
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
        _barrier_fn=None,
        cuda_graph=SimpleNamespace(_break_fns=[]),
    )
    table = _Table()
    token = bcg._current_capture_var.set(capture)
    try:
        out = engram._engram_file_table_lookup(table, torch.tensor([[3]]))
    finally:
        bcg._current_capture_var.reset(token)
    assert events == ["end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 1
    assert table.calls == 1 and out.tolist() == [[[6.0]]]
    capture.cuda_graph._break_fns[0]()  # a replay re-runs the lookup and copies into `out`
    assert table.calls == 2


def test_the_embedding_forward_uses_the_break():
    table = _Table()
    module = SimpleNamespace(file_table=table)
    out = engram.EngramEmbedding.forward(module, torch.tensor([[5]]))
    assert out.tolist() == [[[10.0]]] and table.calls == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
