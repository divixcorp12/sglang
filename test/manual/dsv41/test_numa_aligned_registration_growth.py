"""io_uring registration of a NUMA-split tier grows linearly once node changes sit on 2 MiB boundaries (plan
2026-09-28-reader-crtp-uring-registration, Task 10). divix01 only: real mbind across nodes 0 and 1, the real 6.12
kernel, THP=always.

Task 9's R3 arm hung in ``register_resources``: ``allocate_bound`` split each slab's VMA on page-rounded row
boundaries, leaving 4 KiB pages among the THP folios, so every registered chunk spanning a split failed
``io_try_coalesce_buffer`` and ``headpage_already_acct`` walked every page of every earlier chunk (about +1.13 s per
earlier GiB). This builds the tier the way production does (one ``allocate_host_slab_arena`` per layer, the dsv41
EXL3 slabs and row sizes, rows split 5:4 across nodes 0 and 1 and bound by ``allocate_bound``), touches it, and
registers every named slab with the production ``RegisteredBufferTable`` in row-aligned chunks of at most 1 GiB,
one ``update_tag`` per chunk, timing each (the logic of Task 9's ``regtime.cpp`` harness, kept here).

Recorded per run: the per-chunk times and ``growth``, a least-squares line of each big chunk's ms per GiB against the
GiB registered before it (its slope is the quadratic term: ~1,000 uncoalesced, a few ms coalesced), the
THP coverage (``AnonHugePages`` against the tier), the ``thp_fault_*`` deltas across allocation and touch, and the
chunks lying in a VMA with non-THP pages (an upper bound: which pages inside a VMA are 4 KiB is root-only here).

pytest: 8, 16 and 32 GiB at the new alignment, and an 8 GiB control with the pre-Task-10 page-rounded binding
(``--old-alignment``), which must reproduce the quadratic growth. The 90 GiB tier-scale run is a command, gated and
run under the locks by hand (task-10 report), not a pytest case:

    flock rowimg-disk.lock flock cc-gpu.lock taskset -c 0-63 python test_numa_aligned_registration_growth.py \
        --tier-90g --gate [--collapse]

``--collapse`` is a measurement only: ``madvise(MADV_COLLAPSE)`` per slab after the touch, before registration.
Production code does not do this.
"""

import argparse
import ctypes
import json
import math
import mmap
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(os.uname().nodename != "divix01", reason="needs divix01's two nodes and 6.12 kernel")

GIB, MIB = 1 << 30, 1 << 20
# The dsv41 EXL3 streamed slabs, in EXL3_STREAMED_NAMES order, and their row bytes (DeepSeek-V4.1-Flash-EXL3-3.0bpw).
EXL3_ROWS = (
    ("w13_trellis", 8_847_360),
    ("w13_suh", 20_480),
    ("w13_svh", 9_216),
    ("w2_trellis", 4_423_680),
    ("w2_suh", 4_608),
    ("w2_svh", 10_240),
)
ROW_BYTES = sum(row for _, row in EXL3_ROWS)  # 13,315,584
LAYERS_90G, ROWS_PER_LAYER = 40, 181  # the dsv41 tier: 40 streamed layers, ~181 rows each at 90 GiB
TIER_90G = ((0, 51200 * MIB), (1, 40960 * MIB))
GATE_90G_MIB = {0: 70656, 1: 55056}  # free + page cache needed right before the 90 GiB run
MADV_COLLAPSE = 25

_SOURCE = r"""
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "io/registered_buffers.h"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Plans every region into row-aligned <= 1 GiB chunks (production plan_chunks), sparse-inits one table slot per
// chunk, then registers the chunks one update_tag each, in order, timing each. Stops after abort_ms in total.
// Returns the chunks registered; out_* receive each planned chunk (max_chunks at most) and its time.
extern "C" int register_regions(unsigned regions, const uint64_t* base, const uint64_t* bytes, const uint64_t* row,
                                unsigned max_chunks, uint64_t* out_base, uint64_t* out_len, double* out_ms,
                                unsigned* planned, double abort_ms, char* error, unsigned error_len) {
  try {
    std::vector<sglang::io::ChunkPlan> chunks;
    std::vector<uint64_t> chunk_row;
    for (unsigned r = 0; r < regions; ++r)
      for (const auto& c : sglang::io::plan_chunks(base[r], bytes[r], row[r], sglang::io::kMaxRegisteredBufferBytes)) {
        chunks.push_back(c);
        chunk_row.push_back(row[r]);
      }
    *planned = chunks.size();
    if (chunks.size() > max_chunks) {
      std::snprintf(error, error_len, "%zu chunks > %u", chunks.size(), max_chunks);
      return -1;
    }
    io_uring ring;
    int rc = io_uring_queue_init(8, &ring, 0);
    if (rc) {
      std::snprintf(error, error_len, "ring: %s", std::strerror(-rc));
      return -1;
    }
    sglang::io::RegisteredBufferTable table;
    if (!table.init(&ring, chunks.size())) {
      std::snprintf(error, error_len, "sparse init: %s", table.last_error_context().c_str());
      io_uring_queue_exit(&ring);
      return -1;
    }
    const auto all = Clock::now();
    unsigned done = 0;
    for (; done < chunks.size(); ++done) {
      out_base[done] = chunks[done].base;
      out_len[done] = chunks[done].length;
      const auto t = Clock::now();
      if (table.add(chunks[done].base, chunks[done].length, chunk_row[done]) != 1) {
        std::snprintf(error, error_len, "chunk %u refused: %s", done, table.last_error_context().c_str());
        break;
      }
      out_ms[done] = ms_since(t);
      std::fprintf(stderr, "chunk %u len=%llu ms=%.1f\n", done, (unsigned long long)chunks[done].length, out_ms[done]);
      std::fflush(stderr);
      if (ms_since(all) > abort_ms) {
        ++done;
        std::snprintf(error, error_len, "aborted after %.0f ms", ms_since(all));
        break;
      }
    }
    io_uring_queue_exit(&ring);
    return done;
  } catch (const std::exception& e) {
    std::snprintf(error, error_len, "%s", e.what());
    return -1;
  }
}
"""


def build_library(directory: Path) -> ctypes.CDLL:
    compiler = shutil.which("c++")
    assert compiler is not None, "the harness needs a C++ compiler"
    source, library = directory / "regtime.cpp", directory / "libregtime.so"
    source.write_text(_SOURCE)
    built = subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", "-I", str(ROOT / "python/sglang/kernels/jit/csrc"),
         str(source), "-luring", "-o", str(library)],
        capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    lib = ctypes.CDLL(str(library))
    u64p, dp = ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double)
    lib.register_regions.argtypes = [ctypes.c_uint, u64p, u64p, u64p, ctypes.c_uint, u64p, u64p, dp,
                                     ctypes.POINTER(ctypes.c_uint), ctypes.c_double, ctypes.c_char_p, ctypes.c_uint]
    lib.register_regions.restype = ctypes.c_int
    return lib


def page_rounded_allocate_bound(nbytes, runs, row_bytes):
    """The pre-Task-10 allocate_bound (40452dacc2^), for the control: a page-aligned mapping, one mbind per run,
    each run's first page rounded down, so the node changes sit on page (not 2 MiB) boundaries."""
    from sglang.srt.layers.moe import host_numa

    if nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    page = host_numa.PAGE_BYTES
    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    tensor = torch.frombuffer(mapping, dtype=torch.uint8)
    base = tensor.data_ptr()
    for node, first, count in runs:
        lo = (first * row_bytes) // page * page
        hi = min(nbytes, (first + count) * row_bytes)
        if hi > lo:
            host_numa._mbind(base + lo, -(-(hi - lo) // page) * page, node)
    return tensor


def vmstat() -> dict:
    with open("/proc/vmstat") as f:
        return {k: int(v) for k, v in (line.split() for line in f) if k.startswith(("thp_", "compact_"))}


def anon_huge_kib() -> int:
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith("AnonHugePages:"):
                return int(line.split()[1])
    return 0


def vmas() -> list[tuple[int, int, int, int]]:
    """(start, end, Rss KiB, AnonHugePages KiB) of every mapping of this process."""
    out, current = [], None
    with open("/proc/self/smaps") as f:
        for line in f:
            head = line.split()
            if "-" in head[0] and not head[0].endswith(":"):
                start, end = (int(x, 16) for x in head[0].split("-"))
                current = [start, end, 0, 0]
                out.append(current)
            elif head[0] == "Rss:":
                current[2] = int(head[1])
            elif head[0] == "AnonHugePages:":
                current[3] = int(head[1])
    return [tuple(v) for v in out]


def node_available_mib(node: int) -> int:
    from sglang.srt.layers.moe.host_numa import node_memory

    memory = node_memory(node)
    return (memory["free"] + memory["reclaimable"]) >> 20


def build_tier(total_bytes: int, placement, layers: int):
    """One allocate_host_slab_arena per layer, rows spread as the manager does (differ by at most one)."""
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab_arena

    rows = total_bytes // ROW_BYTES
    specs = {name: ((row,), torch.uint8) for name, row in EXL3_ROWS}
    return [
        allocate_host_slab_arena(rows // layers + (layer < rows % layers), specs, register=False, placement=placement)
        for layer in range(layers)
    ]


def fit(ms: list[float], xs: list[float] | None = None) -> dict:
    """Least squares y = intercept + slope * x (x: the chunk index unless given)."""
    xs = list(range(len(ms))) if xs is None else xs
    n = len(ms)
    if n < 2:
        return {"slope": 0.0, "intercept": ms[0] if ms else 0.0}
    mean_x, mean_y = sum(xs) / n, sum(ms) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ms)) / sum((x - mean_x) ** 2 for x in xs)
    return {"slope": slope, "intercept": mean_y - slope * mean_x}


def growth(ms: list[float], lengths: list[int]) -> dict:
    """Cost per GiB of chunk against the GiB registered before it, over the chunks of at least 256 MiB.

    headpage_already_acct walks every bvec of every earlier buffer for each new head page, so a chunk's cost per GiB
    rises with the bytes registered before it. The slope (ms per GiB of chunk, per earlier GiB) is small for
    coalesced chunks (512 bvecs per earlier GiB) and ~1,000 ms for uncoalesced ones (Task 9: +1.13 s per earlier GiB).
    """
    before, xs, ys = 0, [], []
    for chunk_ms, length in zip(ms, lengths):
        if length >= 256 * MIB:
            xs.append(before / GIB)
            ys.append(chunk_ms / (length / GIB))
        before += length
    result = fit(ys, xs)
    return {"ms_per_gib_per_earlier_gib": result["slope"], "ms_per_gib_at_start": result["intercept"],
            "chunks": len(xs)}


def run(total_bytes: int, placement, layers: int, *, old_alignment=False, collapse=False, abort_s=300.0,
        workdir: Path) -> dict:
    import sglang
    from sglang.srt.layers.moe import host_numa

    lib = build_library(workdir)
    result = {"sglang": sglang.__file__, "pid": os.getpid(), "total_bytes": total_bytes, "layers": layers,
              "placement": [[n, b >> 20] for n, b in placement], "old_alignment": old_alignment}
    huge0, stat0 = anon_huge_kib(), vmstat()
    saved = host_numa.allocate_bound
    if old_alignment:
        host_numa.allocate_bound = page_rounded_allocate_bound
    try:
        t = time.monotonic()
        tier = build_tier(total_bytes, placement, layers)
        result["allocate_s"] = time.monotonic() - t
    finally:
        host_numa.allocate_bound = saved
    owners = [next(iter(slabs.values()))._expert_stream_slab_arena for slabs in tier]
    tier_bytes = sum(owner.nbytes for owner in owners)
    result["tier_bytes"] = tier_bytes
    result["base_2mib_aligned"] = sum(owner.data_ptr() % host_numa.HUGE_BYTES == 0 for owner in owners)
    bound = {}
    for owner in owners:
        for node, nbytes in getattr(owner, "_numa_bound_bytes", {}).items():
            bound[node] = bound.get(node, 0) + nbytes
    result["bound_bytes"] = bound
    t = time.monotonic()
    for owner in owners:
        owner.fill_(1)
    result["touch_s"] = time.monotonic() - t
    stat1 = vmstat()
    result["vmstat_alloc_touch"] = {k: stat1[k] - stat0.get(k, 0) for k in stat1 if stat1[k] != stat0.get(k, 0)}
    huge1 = anon_huge_kib()
    result["anon_huge_kib_after_touch"] = huge1 - huge0
    result["anon_huge_fraction"] = (huge1 - huge0) * 1024 / tier_bytes

    if collapse:
        calls, t_all = [], time.monotonic()
        madvise = ctypes.CDLL(None, use_errno=True).madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        for slabs in tier:
            for name, slab in slabs.items():
                start = slab.data_ptr() // mmap.PAGESIZE * mmap.PAGESIZE
                end = slab.data_ptr() + slab.nbytes
                pieces = [(start, end)]
                t = time.monotonic()
                rc = madvise(start, end - start, MADV_COLLAPSE)
                err = ctypes.get_errno() if rc else 0
                if rc and err in (11, 12):  # EAGAIN, ENOMEM: retry in <= 1 GiB pieces
                    pieces = [(a, min(a + GIB, end)) for a in range(start, end, GIB)]
                    for a, b in pieces:
                        prc = madvise(a, b - a, MADV_COLLAPSE)
                        calls.append({"slab": name, "bytes": b - a, "rc": prc,
                                      "errno": ctypes.get_errno() if prc else 0, "piece": True})
                calls.append({"slab": name, "bytes": end - start, "rc": rc, "errno": err,
                              "s": time.monotonic() - t})
        stat2 = vmstat()
        huge2 = anon_huge_kib()
        result["collapse"] = {
            "s": time.monotonic() - t_all,
            "s_per_gib": (time.monotonic() - t_all) / (tier_bytes / GIB),
            "anon_huge_kib_before": huge1 - huge0,
            "anon_huge_kib_after": huge2 - huge0,
            "vmstat": {k: stat2[k] - stat1.get(k, 0) for k in stat2 if stat2[k] != stat1.get(k, 0)},
            "calls_failed": [c for c in calls if c["rc"]],
            "calls": len(calls),
            "errnos": sorted({c["errno"] for c in calls}),
        }

    # Regions as _table_buffer_regions yields them: every named slab, layer-major, its true row size.
    regions = [(slab.data_ptr(), slab.nbytes, slab.nbytes // slab.shape[0]) for slabs in tier for slab in slabs.values()
               if slab.nbytes]
    n = len(regions)
    arr = lambda values: (ctypes.c_uint64 * n)(*values)  # noqa: E731
    cap = 1 << 14
    out_base, out_len, out_ms = (ctypes.c_uint64 * cap)(), (ctypes.c_uint64 * cap)(), (ctypes.c_double * cap)()
    planned, error = ctypes.c_uint(), ctypes.create_string_buffer(512)
    t = time.monotonic()
    done = lib.register_regions(n, arr(r[0] for r in regions), arr(r[1] for r in regions), arr(r[2] for r in regions),
                                cap, out_base, out_len, out_ms, ctypes.byref(planned), abort_s * 1000.0, error, 512)
    result["register_total_ms"] = (time.monotonic() - t) * 1000.0
    result["chunks_planned"], result["chunks_registered"] = planned.value, done
    result["register_error"] = error.value.decode()
    ms = [out_ms[i] for i in range(max(done, 0))]
    result["chunk_ms"] = [round(x, 1) for x in ms]
    result["chunk_len"] = [out_len[i] for i in range(max(done, 0))]
    result["register_sum_ms"] = sum(ms)
    result["fit_ms_per_chunk_index"] = fit(ms)
    result["growth"] = growth(ms, result["chunk_len"])

    # Chunks lying in a VMA with non-THP resident pages (upper bound on chunks holding a 4 KiB page).
    maps = [v for v in vmas() if v[2] > v[3]]
    flagged = 0
    for i in range(max(done, 0)):
        lo, hi = out_base[i], out_base[i] + out_len[i]
        flagged += any(start < hi and lo < end for start, end, _, _ in maps)
    result["chunks_in_a_vma_with_4k_pages"] = flagged
    lo_tier = min(o.data_ptr() for o in owners)
    hi_tier = max(o.data_ptr() + o.nbytes for o in owners)
    in_tier = [v for v in vmas() if v[0] < hi_tier and lo_tier < v[1]]
    result["tier_vmas"] = len(in_tier)
    result["tier_non_thp_kib"] = sum(v[2] - v[3] for v in in_tier)
    return result


def _gib_run(gib: int, tmp_path: Path, old: bool = False) -> dict:
    total = gib * GIB
    layers = max(1, math.ceil(total // ROW_BYTES / ROWS_PER_LAYER))
    placement = ((0, 5 * total // 9), (1, total - 5 * total // 9))  # the 90 GiB tier's 5:4
    result = run(total, placement, layers, old_alignment=old, workdir=tmp_path)
    print(json.dumps(result))
    return result


@pytest.mark.parametrize("gib", [8, 16, 32])
def test_aligned_registration_grows_linearly(gib, tmp_path):
    result = _gib_run(gib, tmp_path)
    assert result["register_error"] == "", result["register_error"]
    assert result["chunks_registered"] == result["chunks_planned"]
    assert result["base_2mib_aligned"] == result["layers"]
    # Task 9 measured +1,130 ms per GiB per earlier GiB for uncoalesced chunks and ~+3 ms for pure THP.
    assert result["growth"]["ms_per_gib_per_earlier_gib"] < 20, result["growth"]


def test_old_alignment_control_reproduces_the_quadratic_growth(tmp_path):
    result = _gib_run(8, tmp_path, old=True)
    assert result["register_error"] == "", result["register_error"]
    assert result["growth"]["ms_per_gib_per_earlier_gib"] > 250, result["growth"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, help="total GiB, split 5:4 across nodes 0 and 1")
    parser.add_argument("--tier-90g", action="store_true", help="node 0 51200 MiB, node 1 40960 MiB, 40 layers")
    parser.add_argument("--old-alignment", action="store_true")
    parser.add_argument("--collapse", action="store_true", help="measurement only: MADV_COLLAPSE before registering")
    parser.add_argument("--gate", action="store_true", help="refuse unless both nodes clear the 90 GiB gate")
    parser.add_argument("--abort-s", type=float, default=300.0)
    parser.add_argument("--workdir", default="/mnt/nvme1/numa-regtime")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if args.gate:
        available = {node: node_available_mib(node) for node in GATE_90G_MIB}
        print(json.dumps({"gate_available_mib": available, "gate_need_mib": GATE_90G_MIB}), flush=True)
        if any(available[node] < need for node, need in GATE_90G_MIB.items()):
            print("GATE REFUSED", flush=True)
            return 3
    if args.tier_90g:
        total = sum(nbytes for _, nbytes in TIER_90G)
        result = run(total, TIER_90G, LAYERS_90G, old_alignment=args.old_alignment, collapse=args.collapse,
                     abort_s=args.abort_s, workdir=workdir)
    else:
        total = args.gib * GIB
        layers = max(1, math.ceil(total // ROW_BYTES / ROWS_PER_LAYER))
        placement = ((0, 5 * total // 9), (1, total - 5 * total // 9))
        result = run(total, placement, layers, old_alignment=args.old_alignment, collapse=args.collapse,
                     abort_s=args.abort_s, workdir=workdir)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
