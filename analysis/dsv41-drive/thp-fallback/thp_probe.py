"""THP fallback in the pinned host tier, and what it costs io_uring buffer registration (divix01, 6.12 EL10).

Builds the tier the way production does (``test/manual/dsv41/test_numa_aligned_registration_growth.py``'s
``build_tier``: one ``allocate_host_slab_arena`` per layer, rows split 5:4 over nodes 0 and 1, bound by
``host_numa.allocate_bound``), faults it, classifies every page as PMD-mapped THP or not with the unprivileged
``PAGEMAP_SCAN`` ioctl (``PAGE_IS_HUGE``), and registers every named slab with the production
``RegisteredBufferTable`` in row-aligned <= 1 GiB chunks, timing each chunk.

Per chunk it records whether it can coalesce (every page it covers lies in a PMD-mapped THP) and a predicted count of
the struct-page visits ``io_buffer_account_pin`` -> ``headpage_already_acct`` makes for it:

    visits(c) = sum over new head pages h in c of index(h in c's page array)       (the scan of c's own array)
              + H(c) * sum over earlier chunks c' of bvecs(c')                        (the scan of every earlier buffer)

where bvecs is one per folio for a coalesced chunk and one per 4 KiB page otherwise, and H(c) is c's THP count. A
least-squares fit of chunk ms against visits gives the cost per visit, i.e. the scaling law.

Options (all measurements; nothing here is production code):
  --madvise hugepage   madvise(MADV_HUGEPAGE) on each mapping before the fault (defrag=madvise then compacts)
  --madvise nohugepage madvise(MADV_NOHUGEPAGE) on each mapping before the fault: 4 KiB pages only
  --strategies ...     re-register the same faulted tier other ways (see strategy_items): order, clone,
                       clone-rowsplit, prod
  --inject K           MADV_NOHUGEPAGE on one 2 MiB piece in each of K evenly spaced chunks: K mixed chunks
  --repair refault     after the fault, MADV_DONTNEED + re-touch every non-THP 2 MiB range (with MADV_HUGEPAGE)
  --repair collapse    after the fault, MADV_COLLAPSE every non-THP 2 MiB range
  --fault touch|none   how the tier is faulted before registration (none: registration's own pin faults it)
  --no-register        classify only
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GIB, MIB, HUGE, PAGE = 1 << 30, 1 << 20, 2 << 20, 4096

_spec = importlib.util.spec_from_file_location(
    "growth", ROOT / "test/manual/dsv41/test_numa_aligned_registration_growth.py"
)
growth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(growth)

MADV_DONTNEED, MADV_HUGEPAGE, MADV_NOHUGEPAGE, MADV_COLLAPSE = 4, 14, 15, 25
PAGE_IS_PRESENT, PAGE_IS_HUGE = 1 << 3, 1 << 6
PAGEMAP_SCAN = 0xC0606610  # _IOWR('f', 16, struct pm_scan_arg), sizeof == 96

_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]


class PageRegion(ctypes.Structure):
    _fields_ = [("start", ctypes.c_uint64), ("end", ctypes.c_uint64), ("categories", ctypes.c_uint64)]


class PmScanArg(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "size", "flags", "start", "end", "walk_end", "vec", "vec_len", "max_pages",
        "category_inverted", "category_mask", "category_anyof_mask", "return_mask")]


def madvise(start: int, length: int, advice: int) -> int:
    rc = _libc.madvise(start, length, advice)
    return ctypes.get_errno() if rc else 0


def scan(start: int, end: int) -> list[tuple[int, int, bool]]:
    """Present ranges of [start, end) as (lo, hi, huge), merged; unprivileged PAGEMAP_SCAN."""
    fd = os.open("/proc/self/pagemap", os.O_RDONLY)
    try:
        out, vec = [], (PageRegion * 4096)()
        at = start
        while at < end:
            arg = PmScanArg(size=ctypes.sizeof(PmScanArg), start=at, end=end, vec=ctypes.addressof(vec),
                            vec_len=len(vec), category_mask=PAGE_IS_PRESENT,
                            return_mask=PAGE_IS_PRESENT | PAGE_IS_HUGE)
            n = _libc.ioctl(fd, PAGEMAP_SCAN, ctypes.byref(arg))
            if n < 0:
                err = ctypes.get_errno()
                raise OSError(err, f"PAGEMAP_SCAN: {os.strerror(err)}")
            for r in vec[:n]:
                huge = bool(r.categories & PAGE_IS_HUGE)
                if out and out[-1][1] == r.start and out[-1][2] == huge:
                    out[-1] = (out[-1][0], r.end, huge)
                else:
                    out.append((r.start, r.end, huge))
            if arg.walk_end <= at:
                break
            at = arg.walk_end
        return out
    finally:
        os.close(fd)


def owner_span(owner) -> tuple[int, int]:
    base = owner.data_ptr()
    return base, base + -(-owner.nbytes // HUGE) * HUGE


def plan(regions, cap: int = GIB) -> list[tuple[int, int]]:
    """(base, length) of every registered chunk, as plan_chunks cuts them (row-aligned, <= cap)."""
    chunks = []
    for base, nbytes, row in regions:
        step = (cap // row) * row
        chunks += [(base + at, min(step, nbytes - at)) for at in range(0, nbytes, step)]
    return chunks


def chunk_model(chunks, present) -> list[dict]:
    """Per chunk: pages, THP folios, coalescible, bvecs and predicted headpage_already_acct visits."""
    present = sorted(present)  # owners are mapped top-down, so their scans do not come in address order
    huge_ranges = [(lo, hi) for lo, hi, h in present if h]
    small = [(lo, hi) for lo, hi, h in present if not h]
    import bisect

    hstarts = [lo for lo, _ in huge_ranges]
    sstarts = [lo for lo, _ in small]
    rows, earlier = [], 0
    for base, length in chunks:
        first, last = base // PAGE * PAGE, -(-(base + length) // PAGE) * PAGE
        pages = (last - first) // PAGE
        # THP folios covered: 2 MiB frames intersecting the chunk inside a huge range.
        heads = []
        i = max(0, bisect.bisect_right(hstarts, first) - 1)
        while i < len(huge_ranges) and huge_ranges[i][0] < last:
            lo, hi = huge_ranges[i]
            a, b = max(lo, first), min(hi, last)
            if a < b:
                f = a // HUGE * HUGE
                while f < b:
                    heads.append(f)
                    f += HUGE
            i += 1
        small_pages = 0
        j = max(0, bisect.bisect_right(sstarts, first) - 1)
        while j < len(small) and small[j][0] < last:
            a, b = max(small[j][0], first), min(small[j][1], last)
            if a < b:
                small_pages += (b - a) // PAGE
            j += 1
        coalesced = small_pages == 0 and len(heads) > 1
        bvecs = len(heads) if coalesced else pages
        if coalesced:
            self_visits = len(heads) * (len(heads) - 1) // 2
        else:
            self_visits = sum(max(h, first) - first for h in heads) // PAGE
        rows.append({"base": base, "length": length, "pages": pages, "heads": len(heads), "small_pages": small_pages,
                     "coalesced": coalesced, "bvecs": bvecs, "self": self_visits,
                     "visits": self_visits + len(heads) * earlier})
        earlier += bvecs
    return rows


_STRATEGY_SOURCE = r"""
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include "io/registered_buffers.h"

using Clock = std::chrono::steady_clock;

// Registers items (base, len) as one buffer each, in the given order, into a sparse table of n slots.
// clone == 0: RegisteredBufferTable(false)::add straight into the ring (production's call before the clone fix).
// clone == 1: each item is registered alone into slot 0 of a scratch ring, whose pin accounting then scans only
// itself, and cloned into slot i of the ring (IORING_REGISTER_CLONE_BUFFERS, DST_REPLACE); the scratch slot is
// emptied again. Returns the items registered; out_ms gets each one's time.
extern "C" int register_items(unsigned n, const uint64_t* base, const uint64_t* len, int clone, double* out_ms,
                              double abort_ms, char* error, unsigned error_len) {
  try {
    io_uring ring, scratch;
    int rc = io_uring_queue_init(8, &ring, 0);
    if (rc) { std::snprintf(error, error_len, "ring: %s", std::strerror(-rc)); return -1; }
    sglang::io::RegisteredBufferTable table(false);  // clone == 1 does its own cloning below
    if (!table.init(&ring, n)) {
      std::snprintf(error, error_len, "sparse: %s", table.last_error_context().c_str());
      io_uring_queue_exit(&ring);
      return -1;
    }
    if (clone) {
      rc = io_uring_queue_init(8, &scratch, 0);
      if (!rc) rc = io_uring_register_buffers_sparse(&scratch, 1);
      if (rc) { std::snprintf(error, error_len, "scratch: %s", std::strerror(-rc)); io_uring_queue_exit(&ring); return -1; }
    }
    const auto all = Clock::now();
    unsigned done = 0;
    for (; done < n; ++done) {
      const auto t = Clock::now();
      if (!clone) {
        if (table.add(base[done], len[done], len[done]) != 1) {
          std::snprintf(error, error_len, "item %u: %s", done, table.last_error_context().c_str());
          break;
        }
      } else {
        struct iovec v { reinterpret_cast<void*>(base[done]), static_cast<size_t>(len[done]) };
        __u64 tag = 0;
        rc = io_uring_register_buffers_update_tag(&scratch, 0, &v, &tag, 1);
        if (rc != 1) { std::snprintf(error, error_len, "item %u scratch: %d", done, rc); break; }
        rc = io_uring_clone_buffers_offset(&ring, &scratch, done, 0, 1, IORING_REGISTER_DST_REPLACE);
        if (rc < 0) { std::snprintf(error, error_len, "item %u clone: %s", done, std::strerror(-rc)); break; }
        struct iovec empty { nullptr, 0 };
        rc = io_uring_register_buffers_update_tag(&scratch, 0, &empty, &tag, 1);
        if (rc != 1) { std::snprintf(error, error_len, "item %u unscratch: %d", done, rc); break; }
      }
      out_ms[done] = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
      if (std::chrono::duration<double, std::milli>(Clock::now() - all).count() > abort_ms) {
        ++done;
        std::snprintf(error, error_len, "aborted");
        break;
      }
    }
    if (clone) io_uring_queue_exit(&scratch);
    io_uring_queue_exit(&ring);
    return done;
  } catch (const std::exception& e) {
    std::snprintf(error, error_len, "%s", e.what());
    return -1;
  }
}
"""


def build_strategy_library(directory: Path) -> ctypes.CDLL:
    import shutil
    import subprocess

    source, library = directory / "strategy.cpp", directory / "libstrategy.so"
    source.write_text(_STRATEGY_SOURCE)
    built = subprocess.run([shutil.which("c++"), "-std=c++20", "-O2", "-shared", "-fPIC", "-I",
                            str(ROOT / "python/sglang/kernels/jit/csrc"), str(source), "-luring", "-o", str(library)],
                           capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    lib = ctypes.CDLL(str(library))
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.register_items.argtypes = [ctypes.c_uint, u64p, u64p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_double, ctypes.c_char_p, ctypes.c_uint]
    lib.register_items.restype = ctypes.c_int
    return lib


def strategy_items(name: str, chunks_rows, model) -> list[tuple[int, int]]:
    """The buffers a strategy registers, in its order. chunks_rows: (base, length, row bytes) per planned chunk."""
    if name in ("prod", "clone"):
        return [(b, n) for b, n, _ in chunks_rows]
    if name == "order":  # coalesced chunks first, then the rest by THP count, most first (Smith's rule, equal P)
        keyed = sorted(range(len(chunks_rows)), key=lambda k: (not model[k]["coalesced"], -model[k]["heads"], k))
        return [chunks_rows[k][:2] for k in keyed]
    if name == "clone-rowsplit":  # every chunk holding 4 KiB pages registered row by row
        items = []
        for (b, n, row), m in zip(chunks_rows, model):
            if m["small_pages"]:
                items += [(b + at, row) for at in range(0, n, row)]
            else:
                items.append((b, n))
        return items
    if name.startswith(("cap", "clone-cap")):  # every region cut at a smaller row-aligned cap (MiB), THP-blind
        cap = int(name.rsplit("cap", 1)[1]) * MIB
        return [item for b, n, row in chunks_rows for item in plan([(b, n, row)], cap)]
    raise ValueError(name)


def lsq(xs, ys) -> dict:
    n = len(xs)
    if n < 2:
        return {}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    tot = sum((y - my) ** 2 for y in ys)
    return {"ms_per_gvisit": slope * 1e9, "intercept_ms": my - slope * mx, "r2": 1 - res / tot if tot else 1.0}


def summarize(present, tier_bytes) -> dict:
    huge = sum(hi - lo for lo, hi, h in present if h)
    small = sum(hi - lo for lo, hi, h in present if not h)
    return {"present_gib": (huge + small) / GIB, "thp_gib": huge / GIB, "small_gib": small / GIB,
            "small_fraction": small / max(1, huge + small), "tier_gib": tier_bytes / GIB,
            "small_runs": sum(1 for _, _, h in present if not h)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=4)
    ap.add_argument("--placement", default="", help="node:MiB,... (default: --gib split 5:4 over nodes 0 and 1)")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--madvise", choices=["none", "hugepage", "nohugepage"], default="none")
    ap.add_argument("--inject", type=int, default=0)
    ap.add_argument("--repair", choices=["none", "refault", "collapse"], default="none")
    ap.add_argument("--fault", choices=["touch", "none"], default="touch")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--direct", action="store_true", help="register directly (the table's pre-clone path)")
    ap.add_argument("--abort-s", type=float, default=600)
    ap.add_argument("--workdir", default="/mnt/nvme1/thp-fallback")
    ap.add_argument("--label", default="")
    ap.add_argument("--strategies", default="", help="comma list of order,clone,clone-rowsplit,prod: each registers "
                    "the same faulted tier again on a fresh ring, after the production-harness registration")
    args = ap.parse_args()

    import sglang
    import torch
    from sglang.srt.layers.moe import host_numa

    if args.placement:
        placement = host_numa.parse_placement(args.placement)
        total = sum(b for _, b in placement)
    else:
        total = int(args.gib * GIB)
        placement = ((0, 5 * total // 9 // MIB * MIB), (1, total - 5 * total // 9 // MIB * MIB))
    layers = args.layers or max(1, math.ceil(total // growth.ROW_BYTES / growth.ROWS_PER_LAYER))
    result = {"label": args.label, "sglang": sglang.__file__, "args": vars(args),
              "placement_mib": [[n, b >> 20] for n, b in placement], "layers": layers,
              "node_available_mib_before": {n: growth.node_available_mib(n) for n, _ in placement}}
    buddy = Path("/proc/buddyinfo").read_text()
    result["buddyinfo_before"] = [l for l in buddy.splitlines() if "Normal" in l]

    stat0 = growth.vmstat()
    real_allocate = host_numa.allocate_bound

    def allocate(nbytes, runs, row_bytes):
        tensor = real_allocate(nbytes, runs, row_bytes)
        if args.madvise != "none" and nbytes:
            lo, hi = owner_span(tensor)
            err = madvise(lo, hi - lo, MADV_HUGEPAGE if args.madvise == "hugepage" else MADV_NOHUGEPAGE)
            assert err == 0, os.strerror(err)
        return tensor

    host_numa.allocate_bound = allocate
    try:
        tier = growth.build_tier(total, placement, layers)
    finally:
        host_numa.allocate_bound = real_allocate
    owners = growth.owners_of(tier)
    tier_bytes = sum(o.nbytes for o in owners)
    regions = [(s.data_ptr(), s.nbytes, s.nbytes // s.shape[0]) for slabs in tier for s in slabs.values() if s.nbytes]
    chunks = plan(regions)
    result["chunks"] = len(chunks)

    if args.inject:
        big = [k for k, (_, length) in enumerate(chunks) if length >= 256 * MIB]
        picks = sorted({big[round(i * (len(big) - 1) / max(1, args.inject - 1))] for i in range(args.inject)}) \
            if args.inject > 1 else [big[len(big) // 2]]
        injected = []
        for k in picks:
            base, length = chunks[k]
            piece = (base + length // 2) // HUGE * HUGE
            if piece >= base and piece + HUGE <= base + length:
                assert madvise(piece, HUGE, MADV_NOHUGEPAGE) == 0
                injected.append(k)
        result["injected_chunks"] = injected

    t = time.monotonic()
    if args.fault == "touch":
        for owner in owners:
            owner.fill_(1)
    result["fault_s"] = time.monotonic() - t
    stat1 = growth.vmstat()
    result["vmstat_fault"] = {k: stat1[k] - stat0.get(k, 0) for k in stat1 if stat1[k] != stat0.get(k, 0)}

    def present_now():
        out = []
        for owner in owners:
            lo, _ = owner_span(owner)
            out += scan(lo, lo + owner.nbytes)
        return out

    if args.fault == "touch":
        present = present_now()
        result["after_fault"] = summarize(present, tier_bytes)
        result["after_fault"]["mixed_chunks"] = sum(r["small_pages"] > 0 for r in chunk_model(chunks, present))

        if args.repair != "none":
            spans = [owner_span(o) for o in owners]
            frames = sorted({f for lo, hi, h in present if not h for f in range(lo // HUGE * HUGE, hi, HUGE)})
            t, failed = time.monotonic(), 0
            for f in frames:
                if not any(a <= f and f + HUGE <= b for a, b in spans):
                    failed += 1
                    continue
                if args.repair == "refault":
                    assert madvise(f, HUGE, MADV_HUGEPAGE) == 0
                    assert madvise(f, HUGE, MADV_DONTNEED) == 0
                    ctypes.memset(f, 1, HUGE)
                else:
                    failed += madvise(f, HUGE, MADV_COLLAPSE) != 0
            stat2 = growth.vmstat()
            present = present_now()
            result["repair"] = {"frames": len(frames), "failed_calls": failed, "s": time.monotonic() - t,
                                "vmstat": {k: stat2[k] - stat1.get(k, 0) for k in stat2 if stat2[k] != stat1.get(k, 0)},
                                "after": summarize(present, tier_bytes)}
            result["repair"]["after"]["mixed_chunks"] = sum(r["small_pages"] > 0 for r in chunk_model(chunks, present))

    if not args.no_register:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        lib = growth.build_library(workdir)
        n, cap = len(regions), 1 << 14
        arr = lambda values: (ctypes.c_uint64 * n)(*values)  # noqa: E731
        ob, ol, oms = (ctypes.c_uint64 * cap)(), (ctypes.c_uint64 * cap)(), (ctypes.c_double * cap)()
        planned, error = ctypes.c_uint(), ctypes.create_string_buffer(512)
        stat2 = growth.vmstat()
        t = time.monotonic()
        done = lib.register_regions(n, arr(r[0] for r in regions), arr(r[1] for r in regions),
                                    arr(r[2] for r in regions), cap, ob, ol, oms, ctypes.byref(planned),
                                    args.abort_s * 1000.0, error, 512, int(not args.direct))
        result["register_s"] = time.monotonic() - t
        result["register_error"] = error.value.decode()
        result["chunks_registered"] = done
        stat3 = growth.vmstat()
        result["vmstat_register"] = {k: stat3[k] - stat2.get(k, 0) for k in stat3 if stat3[k] != stat2.get(k, 0)}
        assert [(ob[i], ol[i]) for i in range(done)] == chunks[:done], "chunk plan differs from plan_chunks"
        present = present_now()  # registration pinned (so faulted) everything
        result["after_register"] = summarize(present, tier_bytes)
        model = chunk_model(chunks, present)[:done]
        ms = [oms[i] for i in range(done)]
        result["mixed_chunks"] = sum(r["small_pages"] > 0 for r in model)
        result["uncoalesced_chunks"] = sum(not r["coalesced"] for r in model)
        result["fit"] = lsq([r["visits"] for r in model], ms)
        result["visits_total_g"] = sum(r["visits"] for r in model) / 1e9
        result["ms_mixed_chunks"] = sum(m for m, r in zip(ms, model) if not r["coalesced"])
        result["ms_coalesced_chunks"] = sum(m for m, r in zip(ms, model) if r["coalesced"])
        result["per_chunk"] = [[round(m, 1), r["coalesced"], r["small_pages"], r["heads"], r["visits"]]
                               for m, r in zip(ms, model)]
    if args.strategies and not args.no_register:
        slib = build_strategy_library(Path(args.workdir))
        chunks_rows = [(b, n, row) for b_, n_, row in regions
                       for b, n in plan([(b_, n_, row)])]
        present = present_now()
        model = chunk_model(chunks, present)
        result["strategies"] = {}
        for name in args.strategies.split(","):
            items = strategy_items(name, chunks_rows, model)
            n = len(items)
            if n > (1 << 14):
                result["strategies"][name] = {"error": f"{n} items > 16384 slots"}
                continue
            bases = (ctypes.c_uint64 * n)(*[b for b, _ in items])
            lens = (ctypes.c_uint64 * n)(*[l for _, l in items])
            oms, error = (ctypes.c_double * n)(), ctypes.create_string_buffer(512)
            t = time.monotonic()
            done = slib.register_items(n, bases, lens, int(name.startswith("clone")), oms, args.abort_s * 1000.0,
                                       error, 512)
            result["strategies"][name] = {"items": n, "done": done, "s": time.monotonic() - t,
                                          "error": error.value.decode(),
                                          "max_item_ms": max((oms[i] for i in range(max(done, 0))), default=0)}
            print(name, json.dumps(result["strategies"][name]), file=sys.stderr, flush=True)
        result["self_visits_g"] = sum(m["self"] for m in model) / 1e9
    result["node_available_mib_after"] = {n: growth.node_available_mib(n) for n, _ in placement}
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
