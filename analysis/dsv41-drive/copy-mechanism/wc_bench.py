#!/usr/bin/env python3
"""Does a write-combined pinned source slab (GPU reads may skip the CPU snoop) raise H2D over the ordinary slab?

    PYTHONPATH=<repo>/python python wc_bench.py --repo <repo> --probe gen3/probe.json --out gen3/<host>-wc.jsonl
    python3 wc_report.py gen3/<host>-wc.jsonl

Three source slabs, same rows and segments as mech_bench's: `pinned` (allocate_host_slab, the sweep's), `hostalloc`
(cudaHostAlloc Mapped: the allocator without WC, so a difference is attributable to WC) and `wc` (cudaHostAlloc
Mapped | WriteCombined). Both cudaHostAlloc slabs are allocated and first written with this thread on the GPU's NUMA
node. Every shape runs once per slab per round, the slabs interleaved, for --rounds rounds, so drift cancels.
The fresh check runs against a WC source (the host rewrite is fenced with sfence before the flag release).

A speedup here does NOT clear WC for production: NVMe O_DIRECT writes may land in the LLC (DDIO), and a no-snoop GPU
read could miss them. That needs a separate NVMe-write check.
"""
import argparse
import ctypes
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mech_bench as mb  # noqa: E402

CUDA_HOST_ALLOC_MAPPED, CUDA_HOST_ALLOC_WRITE_COMBINED = 0x02, 0x04
mb.WRAPPERS += ["mech_host_alloc", "mech_host_free", "mech_host_write_ns", "mech_host_read_ns"]
HOST_WRITE_BYTES, HOST_READ_BYTES = 256 << 20, 32 << 20  # CPU reads of WC memory are uncached: keep that one short


def node_cpus(node: int) -> set[int]:
    text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    cpus = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def view(ptr: int, rows: int, row_bytes: int) -> torch.Tensor:
    """A CPU tensor over a cudaHostAlloc'd block (the kernels take the raw pointer from data_ptr())."""
    return torch.from_numpy(np.ctypeslib.as_array((ctypes.c_uint8 * (rows * row_bytes)).from_address(ptr))).view(
        rows, row_bytes)


def hostalloc_rows(mod, base: "mb.Rows", flags: int, node: int):
    """A Rows whose sources are cudaHostAlloc(flags) slabs, first written on `node`; destinations shared with base."""
    keep = os.sched_getaffinity(0)
    os.sched_setaffinity(0, node_cpus(node) & keep or node_cpus(node))
    try:
        ptrs, src = [], []
        for b in mb.SEGMENTS:
            ptr = mod.mech_host_alloc(mb.ROWS * b, flags)
            ptrs.append(ptr)
            t = view(ptr, mb.ROWS, b)
            mod.mech_host_write_ns(ptr, mb.ROWS * b, 0)  # first touch: pages land on this node
            for r in range(mb.ROWS):
                t[r, :512].fill_(r % 251)
            src.append(t)
    finally:
        os.sched_setaffinity(0, keep)
    rows = mb.Rows.__new__(mb.Rows)
    rows.src, rows.dst, rows.row, rows.slot = src, base.dst, 0, 0
    return rows, ptrs


def shapes(mod, probe):
    yield mb.Cell("sm_cv16", 8, 4, 16, 8 * 256 * 4 * 16, "plateau", mb.N_ROWS, mb.ALL_SEGMENTS,
                  lambda cpu, dev: mod.mech_copy(dev, mb.KIND_CV, 8, 4, 16))
    yield mb.Cell("sm_cv16", 1, 4, 16, 16384, "16k", mb.N_ROWS, mb.ALL_SEGMENTS,
                  lambda cpu, dev: mod.mech_copy(dev, mb.KIND_CV, 1, 4, 16))
    yield mb.Cell("ce_each", 0, 4, 0, 0, None, 4, mb.ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_each(cpu, dev))
    if mb.probe_ok(probe, "batch"):
        yield mb.Cell("ce_batch", 0, 4, 0, 0, None, 4, mb.ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_batch(cpu, dev))


def fresh_wc(mod, kind, grid, a, b) -> bool:
    """mb.fresh with a write-combined source."""
    n = mb.FRESH_BYTES
    ptr = mod.mech_host_alloc(n, CUDA_HOST_ALLOC_MAPPED | CUDA_HOST_ALLOC_WRITE_COMBINED)
    try:
        src = view(ptr, 1, n)[0]
        src.fill_(0x11)
        pattern = torch.full((n,), 0x22, dtype=torch.uint8)
        dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
        words = torch.zeros(2, dtype=torch.int32).pin_memory()  # protocol words stay in ordinary pinned memory
        jobs = torch.tensor([[ptr, dst.data_ptr(), n]], dtype=torch.int64, device="cuda")
        mod.mech_fresh(jobs, words, src, pattern, kind, grid, a, b)
        torch.cuda.current_stream().synchronize()
        return bool((dst.cpu() == 0x22).all())
    finally:
        torch.cuda.synchronize()
        mod.mech_host_free(ptr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", type=int, default=0, help="NUMA node of every source slab (the GPU's, on divix01)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    probe = json.loads(Path(a.probe).read_text())
    torch.cuda.set_stream(torch.cuda.Stream())  # cudaMemcpyBatchAsync rejects the legacy NULL stream
    mod = mb.load(repo, probe)
    pinned = mb.Rows(a.node)
    hostalloc, ha_ptrs = hostalloc_rows(mod, pinned, CUDA_HOST_ALLOC_MAPPED, a.node)
    wc, wc_ptrs = hostalloc_rows(mod, pinned, CUDA_HOST_ALLOC_MAPPED | CUDA_HOST_ALLOC_WRITE_COMBINED, a.node)
    slabs = {"pinned": pinned, "hostalloc": hostalloc, "wc": wc}
    out = open(a.out, "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")

    emit({"meta": True, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "node": a.node,
          "sglang": sglang.__file__, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "rounds": a.rounds, "reps": a.reps})
    emit({"kind": "fresh", "method": "sm_cv16@wc", "fresh": fresh_wc(mod, mb.KIND_CV, 8, 1, 16)})
    emit({"kind": "fresh", "method": "nc_control@wc", "fresh": fresh_wc(mod, mb.KIND_NC, 8, 1, 16)})
    for rnd in range(a.rounds):
        for cell in shapes(mod, probe):
            for name, rows in slabs.items():
                emit({**mb.measure(cell, rows, a.reps), "slab": name, "round": rnd})
    # Host rates last: the writes overwrite the row markers the byte checks read.
    for name, ptr in (("pinned", pinned.src[0].data_ptr()), ("hostalloc", ha_ptrs[0]), ("wc", wc_ptrs[0])):
        emit({"kind": "host_time", "slab": name, "write_bytes": HOST_WRITE_BYTES,
              "write_ns": mod.mech_host_write_ns(ptr, HOST_WRITE_BYTES, 0x33), "read_bytes": HOST_READ_BYTES,
              "read_ns": mod.mech_host_read_ns(ptr, HOST_READ_BYTES)})
    out.close()
    torch.cuda.synchronize()
    for ptr in ha_ptrs + wc_ptrs:
        mod.mech_host_free(ptr)
    import wc_report

    records = [json.loads(line) for line in open(a.out) if line.strip()]
    above = [r for r in records if r.get("kind") == "cell" and r["gbs"] > 15.75]
    stale = [r for r in records if r.get("method") == "sm_cv16@wc" and not r["fresh"]]
    print(json.dumps({"above_ceiling": above, "wc_stale": bool(stale), "pairs": wc_report.pairs(records)}))
    return 1 if above or stale else 0


if __name__ == "__main__":
    sys.exit(main())
