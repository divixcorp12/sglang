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
from collections.abc import Callable

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
L2_BYTES = 96 * 2**20
# RTX 5090 spec: 512-bit GDDR7 at 28 Gbps.
DRAM_GBPS = 1792.0
# Arbitrary; 50 launches plus their events stay well inside the CUDA launch queue.
QUEUE_CHUNK = 50
# Arbitrary generous first guess at host time per timed launch; doubled per chunk the host fails to queue in time.
QUEUE_MS_PER_LAUNCH = 0.1
QUEUE_MAX_REDOS = 8
_CALIBRATE_CYCLES = 20_000_000

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
    """Replay graphs round-robin (graph i % len), each timed with CUDA events behind a pre-queued GPU delay."""
    n = len(graphs)
    for i in range(warmup):
        graphs[i % n].replay()
    torch.cuda.synchronize()

    def launch(i: int, start: torch.cuda.Event, stop: torch.cuda.Event) -> None:
        start.record()
        graphs[i % n].replay()
        stop.record()

    ms, queue = queued_times(launch, runs=runs)
    return {**summarize(ms), "runs": runs, "graphs": n, "queue": queue}


def summarize(ms: np.ndarray) -> dict:
    return {"median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
            "p99_ms": float(np.percentile(ms, 99))}


def queued_times(launch: Callable[[int, torch.cuda.Event, torch.cuda.Event], None], *, runs: int,
                 chunk: int = QUEUE_CHUNK) -> tuple[np.ndarray, dict]:
    """ms between each launch's (start, stop) events, launched in chunks queued behind a GPU sleep.

    Without the sleep the GPU idles while the host launches, so each sample would include host launch latency.
    A chunk whose sleep ended before its last launch was queued is redone with a doubled sleep.
    """
    cycles_per_ms = _cycles_per_ms()
    sleep_ms = chunk * QUEUE_MS_PER_LAUNCH
    ms, redone = [], 0
    for lo in range(0, runs, chunk):
        while True:
            got = _queued_chunk(launch, idx=range(lo, min(runs, lo + chunk)), cycles=int(sleep_ms * cycles_per_ms))
            if got is not None:
                break
            redone += 1
            if redone > QUEUE_MAX_REDOS:
                raise RuntimeError(f"host could not queue {chunk} launches inside a {sleep_ms:.1f} ms GPU sleep")
            sleep_ms *= 2
        ms.extend(got)
    return np.array(ms, dtype=np.float64), {"chunk": chunk, "sleep_ms": sleep_ms, "redone_chunks": redone}


def _queued_chunk(launch, *, idx: range, cycles: int) -> list[float] | None:
    """Event ms of one chunk queued behind a sleep; None if the GPU caught up with the host."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in idx]
    stops = [torch.cuda.Event(enable_timing=True) for _ in idx]
    torch.cuda._sleep(cycles)
    awake = torch.cuda.Event()
    awake.record()
    for i, start, stop in zip(idx, starts, stops):
        launch(i, start, stop)
    caught_up = awake.query()
    torch.cuda.synchronize()
    if caught_up:
        return None
    return [a.elapsed_time(b) for a, b in zip(starts, stops)]


def _cycles_per_ms() -> float:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(_CALIBRATE_CYCLES)
    stop.record()
    stop.synchronize()
    return _CALIBRATE_CYCLES / start.elapsed_time(stop)


def implied_bandwidth(*, moved_bytes: int, median_ms: float) -> dict:
    """GB/s of ``moved_bytes`` over ``median_ms`` and whether it is below the RTX 5090's DRAM spec."""
    gbps = moved_bytes / (median_ms * 1e-3) / 1e9
    return {"gbps": gbps, "ceiling_gbps": DRAM_GBPS, "plausible": gbps <= DRAM_GBPS}


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
