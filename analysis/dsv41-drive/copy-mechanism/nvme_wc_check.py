#!/usr/bin/env python3
"""Does a GPU read of a write-combined slab return stale DRAM after an NVMe O_DIRECT read into it (DDIO)?

    PYTHONPATH=<repo>/python python nvme_wc_check.py --repo <repo> --probe gen3/probe.json \\
        --root /mnt/nvme0/dsv41_flash --out gen3/<host>-nvme-wc.jsonl
    python3 nvme_report.py gen3/<host>-nvme-wc.jsonl

Needs rowimg-disk.lock, then cc-gpu.lock (run protocol lock order). The row-image files are opened read-only.

Per trial, on each slab (`pinned`: allocate_host_slab; `hostalloc`: cudaHostAlloc Mapped; `wc`: cudaHostAlloc
Mapped | WriteCombined; interleaved, rotating which goes first): the CPU fills the slab with pattern A and sfences;
a kernel waits on a host flag; the slab is filled by pread(O_DIRECT) from a row image; the flag is released and the
slab is read at once three ways into device buffers -- .cv loads, weak loads after the flag's acquire, and
cudaMemcpyAsync -- with no CPU touch in between. Each is compared on the GPU against a reference of the same file
region, loaded once through buffered I/O into ordinary memory (never from a slab). Page-cache residency of the
regions is checked with mincore before and after.
"""
import argparse
import ctypes
import json
import mmap
import os
import socket
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mech_bench as mb  # noqa: E402
import nvme_report as report  # noqa: E402
import wc_bench as wcb  # noqa: E402  (its import already registered the host-alloc wrappers)

mb.WRAPPERS += ["mech_nvme_trial"]
SIZES = {65_536: 300, 1_662_976: 300, 13_312_000: 100}  # bytes -> trials per slab (64 KiB, one piece, one row)
SLAB_BYTES = max(SIZES)
GRID = 16  # 8 blocks of weak loads + 8 of .cv: S's grid each
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
POSIX_FADV_DONTNEED = 4


def resident_pages(path: str, offset: int, size: int) -> int:
    """Pages of [offset, offset+size) of `path` in the page cache: mincore over a read-only mapping (faults nothing)."""
    page = mmap.PAGESIZE
    start = offset - offset % page
    length = offset + size - start
    mm = LIBC.mmap
    mm.restype = ctypes.c_void_p
    mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = mm(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, start)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap for mincore")
        try:
            vec = (ctypes.c_ubyte * ((length + page - 1) // page))()
            if LIBC.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(length), vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            return sum(v & 1 for v in vec)
        finally:
            LIBC.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(length))
    finally:
        os.close(fd)


def load_reference(path: str, offset: int, size: int) -> torch.Tensor:
    """The file region through buffered I/O into ordinary memory, then onto the GPU; its pages then dropped."""
    fd = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(fd, size, offset)
        LIBC.posix_fadvise(fd, ctypes.c_long(offset), ctypes.c_long(size), POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if len(data) != size:
        raise SystemExit(f"short buffered read of {path} at {offset}")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--root", required=True, help="a mirror root holding exl3_row_images/layer-NNN.rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", type=int, default=0)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    probe = json.loads(Path(a.probe).read_text())
    torch.cuda.set_stream(torch.cuda.Stream())
    mod = mb.load(repo, probe)
    images = sorted((Path(a.root) / "exl3_row_images").glob("layer-*.rows"))
    if not images:
        raise SystemExit(f"no row images under {a.root}")
    file_bytes = min(p.stat().st_size for p in images)
    files = [str(p) for p in images]

    # Slabs: every one page-aligned, as O_DIRECT requires; the cudaHostAlloc ones allocated on the GPU's node.
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    pinned = allocate_host_slab(1, (SLAB_BYTES,), torch.uint8, register=True, placement=((a.node, SLAB_BYTES),))
    slabs = {"pinned": pinned.data_ptr()}  # `pinned` stays referenced until the end: it owns the registration
    keep = os.sched_getaffinity(0)
    os.sched_setaffinity(0, wcb.node_cpus(a.node) & keep or wcb.node_cpus(a.node))
    try:
        slabs["hostalloc"] = mod.mech_host_alloc(SLAB_BYTES, wcb.CUDA_HOST_ALLOC_MAPPED)
        slabs["wc"] = mod.mech_host_alloc(SLAB_BYTES, wcb.CUDA_HOST_ALLOC_MAPPED | wcb.CUDA_HOST_ALLOC_WRITE_COMBINED)
        for name in ("hostalloc", "wc"):
            mod.mech_host_write_ns(slabs[name], SLAB_BYTES, 0)
    finally:
        os.sched_setaffinity(0, keep)
    for name, ptr in slabs.items():
        if ptr % 4096:
            raise SystemExit(f"{name} slab is not page-aligned: {ptr:#x}")

    out = open(a.out, "a")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    emit({"meta": True, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "root": a.root,
          "sglang": sglang.__file__, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "sizes": SIZES, "grid": GRID})
    words = torch.zeros(2, dtype=torch.int32).pin_memory()
    fds = {p: os.open(p, os.O_RDONLY | os.O_DIRECT) for p in files}
    trial = 0
    try:
        for size, n in SIZES.items():
            regs = report.regions(n, size, file_bytes=file_bytes, files=files)
            refs = [load_reference(p, off, size) for p, off in regs]
            before = sum(resident_pages(p, off, size) for p, off in regs)
            emit({"kind": "page_cache", "size": size, "when": "before", "resident_pages": before,
                  "pages": n * -(-size // 4096)})
            dst = {m: torch.empty(size, dtype=torch.uint8, device="cuda") for m in report.METHODS}
            order = list(slabs)
            for i, ((path, off), ref) in enumerate(zip(regs, refs)):
                ref64 = ref.view(torch.int64)
                for k in range(len(order)):
                    slab = order[(i + k) % len(order)]  # rotate which slab reads the region first
                    pattern = report.pattern_word(trial)
                    in_file = int((ref64 == pattern).sum())
                    if in_file:
                        emit({"kind": "pattern_in_file", "trial": trial, "words": in_file})
                    for d in dst.values():
                        d.fill_(0)
                    torch.cuda.synchronize()
                    ns = mod.mech_nvme_trial(words, slabs[slab], size, fds[path], off, pattern, dst["weak"],
                                             dst["sm_cv16"], dst["ce"], GRID)
                    rec = {"kind": "trial", "trial": trial, "slab": slab, "size": size, "file": Path(path).name,
                           "offset": off, "first": k == 0, "pread_ns": ns, "stale": {}, "wrong": {}}
                    for m in report.METHODS:
                        got = dst[m].view(torch.int64)
                        rec["stale"][m] = int((got == pattern).sum())
                        rec["wrong"][m] = int((got != ref64).sum())
                    emit(rec)
                    trial += 1
            after = sum(resident_pages(p, off, size) for p, off in regs)
            emit({"kind": "page_cache", "size": size, "when": "after", "resident_pages": after,
                  "pages": n * -(-size // 4096)})
            del refs
            torch.cuda.empty_cache()
    finally:
        for fd in fds.values():
            os.close(fd)
        out.close()
        torch.cuda.synchronize()
        for name in ("hostalloc", "wc"):
            mod.mech_host_free(slabs[name])
        del pinned
    records = [json.loads(line) for line in open(a.out) if line.strip()]
    s = report.summarize(records)
    print(json.dumps({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in s.items()}))
    return 1 if any(c["wrong_words"] for c in s.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
