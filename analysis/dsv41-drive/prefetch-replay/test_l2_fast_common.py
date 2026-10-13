"""Tests for l2_fast_common: output guard, atomic JSON, rotating-graph timing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_fast_common as common  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("protected", ["f109-l2-cap1-20261013", "b1-r1-20261012"])
def test_guard_refuses_protected_run_dirs(protected):
    """Catches a fast-path output landing in the F109 or R1 result directories."""
    with pytest.raises(common.RefusedOut):
        common.guard_out(os.path.join(common.OUT_ROOT, protected, "x"))


def test_guard_refuses_path_outside_root():
    """Catches an --out under /tmp or the home directory slipping past the root check."""
    with pytest.raises(common.RefusedOut):
        common.guard_out("/tmp/f123-out")


def test_guard_refuses_inside_run_dir():
    """Catches --out nested in RUN_DIR, which is read-only input."""
    run_dir = os.path.join(common.OUT_ROOT, "some-run")
    with pytest.raises(common.RefusedOut):
        common.guard_out(os.path.join(run_dir, "out"), run_dir=run_dir)


def test_guard_accepts_new_dir_under_root():
    """Catches an over-eager guard that rejects the default OUT."""
    out = os.path.join(common.OUT_ROOT, "f123-l2-fast-test")
    assert common.guard_out(out, run_dir=os.path.join(common.OUT_ROOT, "other")) == os.path.realpath(out)


def test_write_json_is_atomic(tmp_path):
    """Catches a leftover .tmp file or a partial result.json after a write."""
    path = tmp_path / "result.json"
    common.write_json(str(path), {"a": 1})
    assert path.read_text().strip().startswith("{")
    assert sorted(os.listdir(tmp_path)) == ["result.json"]


@needs_cuda
def test_replay_stats_round_robin():
    """Catches replay_stats dropping the graph count or percentiles."""
    graphs = []
    bufs = [torch.zeros(1024, device="cuda") for _ in range(3)]
    for buf in bufs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buf.add_(1.0)
        graphs.append(graph)
    stats = common.replay_stats(graphs, runs=30, warmup=3)
    assert stats["graphs"] == 3 and stats["runs"] == 30
    assert 0.0 <= stats["median_ms"] <= stats["p99_ms"]
    counts = {int(b[0].item()) for b in bufs}  # a redone chunk replays all three graphs equally again
    assert len(counts) == 1 and counts.pop() >= 11



def test_implied_bandwidth_flags_above_spec():
    """Catches an L2-resident number (above the card's DRAM spec) passing as plausible."""
    assert common.implied_bandwidth(moved_bytes=10**9, median_ms=1.0)["plausible"]
    assert not common.implied_bandwidth(moved_bytes=3 * 10**9, median_ms=1.0)["plausible"]


@needs_cuda
def test_graph_nodes_counts_kernels():
    """Catches graph_nodes returning 0 on a DOT layout it cannot parse."""
    a = torch.ones(256, device="cuda")

    def step():
        a.add_(1.0)
        a.mul_(2.0)
        a.sub_(1.0)

    assert common.graph_nodes(step) >= 3
