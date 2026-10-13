"""Shared helpers for the l2_fast CLIs: output guard, atomic JSON, measurement labels, rotating-graph timing.

Inputs: none; imported by l2_fast_profile, l2_fast_cost, l2_fast_interference, l2_fast_parity and l2_fast_report.
Output: no files of its own; write_json writes atomically wherever a caller says.
Nothing is written outside OUT_ROOT, never into a protected run directory, and never inside RUN_DIR.

Usage: import l2_fast_common as common; out = common.guard_out(args.out, run_dir=args.run_dir)
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_stage  # noqa: E402

OUT_ROOT = "/mnt/nvme1/nvfp4-work"
# Copied from l2_cap1_eval.FORBIDDEN (importing it pulls in the whole eval stack), plus the F109 result dir.
FORBIDDEN = (
    "b1-run-20261012",
    "b1-r1-20261012",
    "b1-r2-20261012",
    "l2-ab-20261012",
    "l2-ab-20261012-failed-generation",
    "f109-l2-cap1-20261013",
)
OBSERVED = "observed offline"
HYPOTHESIS = "hypothesis"
BAR_MS_PER_LAYER = 0.1
SLOTS = 38
WIDTHS = (1, 2, 4, 6)

_NODE_LINE = re.compile(r'^\s*"([^"]+)"\s*\[', re.MULTILINE)


class RefusedOut(ValueError):
    pass


def guard_out(out: str, *, run_dir: str | None = None) -> str:
    """The real --out; refuses a path outside OUT_ROOT, in a protected run dir, or equal to or inside run_dir."""
    try:
        real = r1_stage.guard_out_root(out, root=OUT_ROOT, forbidden=FORBIDDEN)
    except ValueError as e:
        raise RefusedOut(str(e)) from e
    if run_dir is not None:
        real_run = os.path.realpath(run_dir)
        if real == real_run or real.startswith(real_run + os.sep):
            raise RefusedOut(f"--out {out!r} is or lies inside RUN_DIR {run_dir!r}, which is read-only")
    return real


def write_json(path: str, obj: object) -> None:
    """Write obj as JSON to path through a tmp file in the same directory and os.replace."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def replay_stats(graphs: list[torch.cuda.CUDAGraph], *, runs: int, warmup: int) -> dict:
    """Replay graphs round-robin (graph i % len) and time each replay with CUDA events."""
    n = len(graphs)
    for i in range(warmup):
        graphs[i % n].replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for i in range(runs):
        starts[i].record()
        graphs[i % n].replay()
        stops[i].record()
    torch.cuda.synchronize()
    ms = np.array([a.elapsed_time(b) for a, b in zip(starts, stops)], dtype=np.float64)
    return {
        "median_ms": float(np.median(ms)),
        "p90_ms": float(np.percentile(ms, 90)),
        "p99_ms": float(np.percentile(ms, 99)),
        "runs": runs,
        "graphs": n,
    }


def graph_nodes(step) -> int:
    """Number of nodes in the CUDA graph that capturing step() produces.

    Relies on torch.cuda.CUDAGraph.enable_debug_mode() + debug_dump(path), which writes the graph as a DOT file;
    keep_graph=True keeps the cudaGraph_t alive for the dump.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        dot = os.path.join(tmp, "graph.dot")
        graph.debug_dump(dot)
        with open(dot) as f:
            text = f.read()
    return len(set(_NODE_LINE.findall(text)))
