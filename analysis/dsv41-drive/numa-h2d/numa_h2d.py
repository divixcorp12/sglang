"""Host-to-GPU bandwidth of 13.3 MB expert rows by source NUMA node, idle and under host memory load.

Answers: does the pinned tier's node-1 share (40 of 100 GiB) reach the GPU (on node 0) slower than node 0's, and
does other memory traffic on the socket (disk DMA, CPU copies) throttle the link? Two read paths are measured, since
serving uses both:
  ce  cudaMemcpyAsync per row (copy engine)
  zc  an SM kernel reading the registered host pointer directly (zero copy, the way C1 reads a row)
The load is `load.py` processes, each copying a 256 MiB buffer to another in a loop, bound to one node's cores and
memory with numactl. Rows are bound with host_numa (mbind), like the serving tier, and checked with page_nodes.

Writes one JSON line per (source node, load, path) to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.srt.layers.moe.host_numa import page_nodes

ROW_BYTES = 13_315_584  # one expert-layer row (exl3_row_image.image_bytes)
ROWS = 48  # 639 MB per node: far past L2 and any host cache, and a full sweep is ~50 ms at link speed
LOAD_CORES = {0: "0-7", 1: "18-25"}  # physical cores of each node; never 64-71 (core 71 is production's spin core)
HERE = os.path.dirname(os.path.abspath(__file__))


class _HostView:
    """A CUDA-array-interface view of registered host memory, so an SM kernel reads it over PCIe."""

    def __init__(self, ptr: int, shape: tuple[int, int]):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "<i4", "data": (ptr, False), "version": 3}


def sweep_seconds(fn, reps: int) -> list[float]:
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for r in range(ROWS):
            fn(r)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return times


def start_load(node: int, procs: int, seconds: float, python: str) -> list[subprocess.Popen]:
    return [
        subprocess.Popen(
            ["numactl", f"--physcpubind={LOAD_CORES[node]}", f"--membind={node}", python, f"{HERE}/load.py", str(seconds)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(procs)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.init()

    slabs = {}
    for node in (0, 1):
        slab = allocate_host_slab(ROWS, (ROW_BYTES,), torch.uint8, register=True, placement=((node, ROWS * ROW_BYTES),))
        for r in range(ROWS):
            slab[r].fill_(r + 16 * node + 1)
        where = dict(page_nodes(slab, samples=256))
        assert set(where) == {node}, f"rows meant for node {node} landed on {where}"
        slabs[node] = slab
        print(json.dumps({"slab": node, "pages": where}), flush=True)

    dst = torch.empty((ROWS, ROW_BYTES), dtype=torch.uint8, device=dev)
    dst32 = dst.view(torch.int32)
    views = {
        node: torch.as_tensor(_HostView(slab.data_ptr(), (ROWS, ROW_BYTES // 4)), device=dev)
        for node, slab in slabs.items()
    }
    for node, slab in slabs.items():
        torch.bitwise_or(views[node][3], 0, out=dst32[3])
        torch.cuda.synchronize()
        assert torch.equal(dst[3, :4096].cpu(), slab[3, :4096]), f"zero-copy read of node {node} returned wrong bytes"

    paths = {
        "ce": lambda node: (lambda r: dst[r].copy_(slabs[node][r], non_blocking=True)),
        "zc": lambda node: (lambda r: torch.bitwise_or(views[node][r], 0, out=dst32[r])),
    }
    # (source node, load node, load processes)
    conditions = [(0, None, 0), (1, None, 0), (0, 0, 4), (1, 0, 4), (0, 0, 8), (1, 0, 8), (1, 1, 8), (0, 1, 8)]
    for path in paths:
        for src in (0, 1):
            sweep_seconds(paths[path](src), 3)  # warm
    for src, load_node, procs in conditions:
        load = start_load(load_node, procs, 30.0, sys.executable) if procs else []
        if load:
            time.sleep(2.0)
        for path, make in paths.items():
            times = sweep_seconds(make(src), args.reps)
            gbs = [ROWS * ROW_BYTES / t / 1e9 for t in times]
            print(
                json.dumps(
                    {
                        "src_node": src,
                        "load_node": load_node,
                        "load_procs": procs,
                        "path": path,
                        "gbs_p50": round(statistics.median(gbs), 2),
                        "gbs_min": round(min(gbs), 2),
                        "gbs_max": round(max(gbs), 2),
                        "row_ms_p50": round(statistics.median(times) / ROWS * 1e3, 3),
                    }
                ),
                flush=True,
            )
        load_gbs = []
        for p in load:
            p.terminate()
        for p in load:
            out, _ = p.communicate(timeout=30)
            if out.strip():
                load_gbs.append(json.loads(out.strip().splitlines()[-1])["gbs"])
        if load:
            print(json.dumps({"load_node": load_node, "load_procs": procs, "load_gbs_total": round(sum(load_gbs), 1)}), flush=True)


if __name__ == "__main__":
    main()
