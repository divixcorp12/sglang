"""Convert one Engram shard's split weight/scale tables to the page-packed format (engram_packed.py).

    python scripts/dsv41/engram_relayout.py --source /mnt/nvme2/DeepSeek-V4.1-Flash/model-00047-of-00048.safetensors \
        --layer 1 --out /mnt/nvme0/engram-packed/engram-layer1-packed.safetensors

Reads the source's weight and scale tensors sequentially (O_DIRECT, so the page cache is untouched), packs 15 rows
of 264 B per 4 KiB page and writes them to ``<out>.partial`` on another drive (O_DIRECT). The data region's
checksum is sha256 over the sha256 of each SHA256_PIECE_BYTES piece, hashed on --hash-threads threads (one sha256
stream runs at ~0.45 GiB/s on divix01's Xeon 6154, which has no SHA extensions). The header is written last, with
the checksum, and the file renamed to ``<out>``: a reader refuses a header without one, so an interrupted
conversion never opens.

Then a verify pass (on by default) reads the written file back and the source again, and checks every row's 264 B
against the source's weight and scale bytes, every padding byte zero, and the checksum. A ``<out>.manifest.json``
records the source, geometry, checksum, timings and the verify result. The source is only read.

Disk use: holds --lock (rowimg-disk.lock) for the whole run and refuses while a served capture (run_capture_pair)
is running, so it never shares the drives with a capture.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import mmap
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from sglang.srt.layers import engram_packed
from sglang.srt.layers.engram_packed import PAGE_BYTES, PackedGeometry

SHA256_PIECE_BYTES = 8 << 20
DEFAULT_LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"


def _open(path: str, flags: int, direct: bool) -> tuple[int, bool]:
    """(fd, whether O_DIRECT took); tmpfs refuses O_DIRECT with EINVAL, which falls back to buffered."""
    if direct:
        try:
            return os.open(path, flags | os.O_DIRECT, 0o644), True
        except OSError as exc:
            if exc.errno != 22:
                raise
    return os.open(path, flags, 0o644), False


def _aligned(nbytes: int) -> mmap.mmap:
    return mmap.mmap(-1, max(PAGE_BYTES, -(-nbytes // PAGE_BYTES) * PAGE_BYTES))


def read_range(fd: int, offset: int, length: int, buf: mmap.mmap) -> np.ndarray:
    """Bytes [offset, offset + length) of fd through page-aligned reads into ``buf``."""
    start = offset & ~(PAGE_BYTES - 1)
    want = offset + length - start
    span = -(-want // PAGE_BYTES) * PAGE_BYTES
    view = memoryview(buf)
    got = 0
    while got < want:
        n = os.preadv(fd, [view[got:span]], start + got)
        if n <= 0:
            raise EOFError(f"read at {start + got}: end of file before {offset + length}")
        got += n
    return np.frombuffer(buf, np.uint8, length, offset - start)


def hash_pieces(pool, data: np.ndarray) -> list[concurrent.futures.Future]:
    """One sha256 per SHA256_PIECE_BYTES piece of ``data``, on the pool (hashlib releases the GIL)."""
    return [
        pool.submit(lambda piece: hashlib.sha256(piece).digest(), data[i : i + SHA256_PIECE_BYTES])
        for i in range(0, data.size, SHA256_PIECE_BYTES)
    ]


def capture_running() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "run_capture_pair"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "pgrep" not in line and "engram_relayout" not in line]


class Source:
    """The split shard: weight [rows, dim] F8_E4M3 and scale [rows, dim/32] F8_E8M0."""

    def __init__(self, path: str, layer: int, rows: int | None, direct: bool):
        data_start, header = engram_packed.read_header(path)
        w = header[f"layers.{layer}.engram.embed.weight"]
        s = header[f"layers.{layer}.engram.embed.scale"]
        if (w["dtype"], s["dtype"]) != ("F8_E4M3", "F8_E8M0") or w["shape"][0] != s["shape"][0]:
            raise ValueError(f"{path}: weight {w['dtype']} {w['shape']}, scale {s['dtype']} {s['shape']}")
        self.path, self.total_rows = path, w["shape"][0]
        self.rows = self.total_rows if rows is None else rows
        if not 0 < self.rows <= self.total_rows:
            raise ValueError(f"--rows {self.rows} outside 1..{self.total_rows}")
        self.dim, self.scale_dim = w["shape"][1], s["shape"][1]
        self.weight_offset = data_start + w["data_offsets"][0]
        self.scale_offset = data_start + s["data_offsets"][0]
        self.fd, self.direct = _open(path, os.O_RDONLY, direct)
        st = os.fstat(self.fd)
        self.size, self.mtime_ns = st.st_size, st.st_mtime_ns

    def read(self, row: int, n: int, wbuf: mmap.mmap, sbuf: mmap.mmap) -> tuple[np.ndarray, np.ndarray]:
        weight = read_range(self.fd, self.weight_offset + row * self.dim, n * self.dim, wbuf)
        scale = read_range(self.fd, self.scale_offset + row * self.scale_dim, n * self.scale_dim, sbuf)
        return weight.reshape(n, self.dim), scale.reshape(n, self.scale_dim)


class Throttle:
    def __init__(self, mib_s: float):
        self.rate, self.start, self.done = mib_s * (1 << 20), time.monotonic(), 0

    def __call__(self, nbytes: int) -> None:
        self.done += nbytes
        if self.rate > 0:
            ahead = self.done / self.rate - (time.monotonic() - self.start)
            if ahead > 0:
                time.sleep(ahead)


def convert(src: Source, layer: int, out: str, chunk_pages: int, pool, direct: bool, throttle: Throttle) -> dict:
    geometry = PackedGeometry(src.rows, src.dim, src.scale_dim)
    rpp = geometry.rows_per_page
    partial = out + ".partial"
    fd, out_direct = _open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, direct)
    meta = {
        "source": os.path.basename(src.path),
        "source_bytes": str(src.size),
        "source_rows": str(src.total_rows),
        "sha256_piece_bytes": str(SHA256_PIECE_BYTES),
        "data_sha256": engram_packed.UNSET_SHA256,
    }
    header = _aligned(PAGE_BYTES)
    header[:] = engram_packed.packed_header(layer, geometry, meta)
    os.pwrite(fd, header, 0)
    chunk_rows = chunk_pages * rpp
    ready: queue.Queue = queue.Queue(maxsize=1)
    free: queue.Queue = queue.Queue()
    for _ in range(2):
        free.put(_aligned(chunk_pages * PAGE_BYTES))
    wbuf, sbuf = _aligned(chunk_rows * src.dim + 2 * PAGE_BYTES), _aligned(chunk_rows * src.scale_dim + 2 * PAGE_BYTES)
    timings = {"read_pack_s": 0.0, "write_s": 0.0, "hash_s": 0.0}

    def reader() -> None:
        try:
            for row in range(0, src.rows, chunk_rows):
                n = min(chunk_rows, src.rows - row)
                t = time.monotonic()
                weight, scale = src.read(row, n, wbuf, sbuf)
                buf = free.get()
                pages = engram_packed.pack_rows(
                    weight, scale, np.frombuffer(buf, np.uint8).reshape(chunk_pages, PAGE_BYTES)
                )
                timings["read_pack_s"] += time.monotonic() - t
                throttle(n * geometry.row_bytes)
                ready.put((row // rpp, pages.shape[0], buf))
            ready.put(None)
        except BaseException as exc:  # handed to the writer, which raises it
            ready.put(exc)

    thread = threading.Thread(target=reader, name="engram-relayout-read", daemon=True)
    thread.start()
    digests: list[bytes] = []
    t0 = time.monotonic()
    while (item := ready.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        page, npages, buf = item
        data = np.frombuffer(buf, np.uint8, npages * PAGE_BYTES)
        t = time.monotonic()
        hashed = hash_pieces(pool, data)
        view = memoryview(buf)[: npages * PAGE_BYTES]
        written = 0
        while written < view.nbytes:
            written += os.pwrite(fd, view[written:], PAGE_BYTES + page * PAGE_BYTES + written)
        timings["write_s"] += time.monotonic() - t
        t = time.monotonic()
        digests += [f.result() for f in hashed]
        timings["hash_s"] += time.monotonic() - t
        free.put(buf)
    thread.join()
    os.fsync(fd)
    data_sha = hashlib.sha256(b"".join(digests)).hexdigest()
    header[:] = engram_packed.packed_header(layer, geometry, {**meta, "data_sha256": data_sha})
    os.pwrite(fd, header, 0)
    os.fsync(fd)
    os.close(fd)
    os.rename(partial, out)
    dfd = os.open(os.path.dirname(os.path.abspath(out)), os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)
    wall = time.monotonic() - t0
    return {
        "data_sha256": data_sha, "wall_s": round(wall, 2), "direct_write": out_direct,
        "gb_s_written": round((geometry.data_bytes + PAGE_BYTES) / wall / 1e9, 3),
        **{k: round(v, 2) for k, v in timings.items()},
    }


def verify(src: Source, layer: int, out: str, chunk_pages: int, pool, direct: bool, throttle: Throttle) -> dict:
    """Every packed row against the source's weight and scale bytes, every padding byte zero, and the checksum."""
    weight_key = f"layers.{layer}.engram.embed.weight"
    offset, geometry = engram_packed.read_packed_geometry(out, weight_key, src.rows, src.dim, src.dim // src.scale_dim)
    meta = engram_packed.read_header(out)[1]["__metadata__"]
    fd, out_direct = _open(out, os.O_RDONLY, direct)
    rpp, dim = geometry.rows_per_page, src.dim
    chunk_rows = chunk_pages * rpp
    pbuf = _aligned(chunk_pages * PAGE_BYTES + PAGE_BYTES)
    wbuf, sbuf = _aligned(chunk_rows * dim + 2 * PAGE_BYTES), _aligned(chunk_rows * src.scale_dim + 2 * PAGE_BYTES)
    digests: list[bytes] = []
    bad_rows = 0
    t0 = time.monotonic()
    for row in range(0, src.rows, chunk_rows):
        n = min(chunk_rows, src.rows - row)
        npages = -(-n // rpp)
        packed = read_range(fd, offset + (row // rpp) * PAGE_BYTES, npages * PAGE_BYTES, pbuf)
        hashed = hash_pieces(pool, packed)
        weight, scale = src.read(row, n, wbuf, sbuf)
        throttle(n * geometry.row_bytes)
        slots = engram_packed.packed_rows_view(packed.reshape(npages, PAGE_BYTES), dim, src.scale_dim)
        full = n // rpp
        ok = np.ones(n, bool)
        ok[: full * rpp] &= (slots[:full, :, :dim].reshape(-1, dim) == weight[: full * rpp]).all(1)
        ok[: full * rpp] &= (slots[:full, :, dim:].reshape(-1, src.scale_dim) == scale[: full * rpp]).all(1)
        if full < npages:
            ok[full * rpp :] &= (slots[full, : n - full * rpp, :dim] == weight[full * rpp :]).all(1)
            ok[full * rpp :] &= (slots[full, : n - full * rpp, dim:] == scale[full * rpp :]).all(1)
            if slots[full, n - full * rpp :].any():
                raise ValueError(f"{out}: nonzero empty slot in page {row // rpp + full}")
        if packed.reshape(npages, PAGE_BYTES)[:, rpp * geometry.row_bytes :].any():
            raise ValueError(f"{out}: nonzero padding in pages {row // rpp}..{row // rpp + npages - 1}")
        bad_rows += int((~ok).sum())
        digests += [f.result() for f in hashed]
    os.close(fd)
    sha = hashlib.sha256(b"".join(digests)).hexdigest()
    return {
        "rows_compared": src.rows, "rows_differing": bad_rows, "data_sha256_read_back": sha,
        "checksum_matches": sha == meta["data_sha256"], "direct_read": out_direct,
        "wall_s": round(time.monotonic() - t0, 2),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, help="convert only the first N rows (a prototype slice)")
    p.add_argument("--chunk-pages", type=int, default=16384, help="pages per read/write (default 64 MiB)")
    p.add_argument("--hash-threads", type=int, default=6)
    p.add_argument("--max-mib-s", type=float, default=0.0, help="cap the packed bytes per second (0: no cap)")
    p.add_argument("--no-direct", action="store_true", help="buffered I/O (tests on tmpfs)")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-only", action="store_true", help="verify an existing --out against --source")
    p.add_argument("--same-drive", action="store_true", help="allow --out on the source's drive")
    p.add_argument("--lock", default=DEFAULT_LOCK, help="flock held for the run ('' for none)")
    a = p.parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    if not a.verify_only:
        for path in (a.out, a.out + ".partial"):
            if os.path.exists(path):
                sys.exit(f"{path} exists; this tool never overwrites (remove it by hand)")
        if os.stat(a.source).st_dev == os.stat(out_dir).st_dev and not a.same_drive:
            sys.exit(f"{a.out} is on the source's drive; pass --same-drive to allow it")
    if (running := capture_running()):
        sys.exit("a served capture is running; not converting:\n" + "\n".join(running))
    lock = None
    if a.lock:
        lock = open(a.lock, "w")
        print(f"waiting for {a.lock}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"{a.lock} held", flush=True)

    src = Source(a.source, a.layer, a.rows, not a.no_direct)
    geometry = PackedGeometry(src.rows, src.dim, src.scale_dim)
    if not a.verify_only:
        st = os.statvfs(out_dir)
        need = PAGE_BYTES + geometry.data_bytes
        if st.f_bavail * st.f_frsize < need + (1 << 30):
            sys.exit(f"{out_dir}: {st.f_bavail * st.f_frsize} bytes free, need {need} + 1 GiB")
    print(f"layer {a.layer}: {src.rows} rows -> {geometry.pages} pages, {geometry.data_bytes / 1e9:.3f} GB", flush=True)
    manifest = {
        "host": socket.gethostname(), "source": os.path.abspath(a.source), "source_bytes": src.size,
        "source_mtime_ns": src.mtime_ns, "source_rows": src.total_rows, "layer": a.layer, "out": os.path.abspath(a.out),
        "rows": src.rows, "dim": src.dim, "scale_dim": src.scale_dim, "rows_per_page": geometry.rows_per_page,
        "pages": geometry.pages, "data_bytes": geometry.data_bytes, "file_bytes": PAGE_BYTES + geometry.data_bytes,
        "format": engram_packed.FORMAT, "version": engram_packed.VERSION, "direct_read": src.direct,
        "chunk_pages": a.chunk_pages, "hash_threads": a.hash_threads, "max_mib_s": a.max_mib_s,
    }
    rc = 0
    with concurrent.futures.ThreadPoolExecutor(a.hash_threads) as pool:
        if not a.verify_only:
            manifest["convert"] = convert(src, a.layer, a.out, a.chunk_pages, pool, not a.no_direct, Throttle(a.max_mib_s))
            print(f"converted: {manifest['convert']}", flush=True)
        if not a.no_verify:
            manifest["verify"] = v = verify(src, a.layer, a.out, a.chunk_pages, pool, not a.no_direct, Throttle(a.max_mib_s))
            print(f"verified: {v}", flush=True)
            rc = 0 if v["rows_differing"] == 0 and v["checksum_matches"] else 1
    os.close(src.fd)
    suffix = ".verify.json" if a.verify_only else ".manifest.json"
    with open(a.out + suffix, "w") as f:
        json.dump(manifest, f, indent=1)
    if lock is not None:
        lock.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
