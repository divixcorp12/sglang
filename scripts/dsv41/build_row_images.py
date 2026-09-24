"""Build (or re-check) EXL3 row images on the mirror roots.

A row image is an expert row re-laid so the RAM-miss reader can ``readv`` it straight into the pinned slabs; the
format, the byte mapping, the digests and the manifest all belong to
``python/sglang/srt/layers/moe/exl3_row_image.py``, and this tool only drives them over a real checkpoint.

A build, for the chosen layers:

1. Refuses a root that is the checkpoint directory (or holds its shards), one that already holds a manifest for a
   different source or image layout, and (after step 2) one without room for what it must write.
2. Resumes: a layer whose ``layer-LLL.rows`` and digest sidecar (``layer-LLL.rows.digests.json``) both exist on a
   root is re-read with O_DIRECT and kept only if every row's digest matches the sidecar.
3. Reads every other layer's source rows once (O_DIRECT) and writes each row's image to every root that needs it,
   into ``layer-LLL.rows.tmp`` (O_DIRECT), then fsyncs, renames it into place and writes its sidecar.
4. Reads back every layer it wrote (O_DIRECT) and compares each row's digest with the one computed from the source.
5. Only then writes ``manifest.json`` on each root (listing exactly the chosen layers), and opens the result with
   ``open_row_images`` as a last check. A build removes any earlier manifest before it changes anything, so an
   interrupted or failed run leaves a root with no manifest, which every reader refuses.

``--verify`` writes nothing: it opens the set with ``open_row_images`` (manifest, layout, source, sizes, cross-root
digests) and reads every row back with O_DIRECT against the manifest's digests.

Every read and write uses O_DIRECT, so the page cache never holds the ~205 GB a root receives.

Exit status: 0 a build that finished, or a --verify of every layer that passed; 2 a verification failure;
3 a --verify that passed but covered only some layers; 1 a usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from verify_expert_mirror import parse_layers

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_format import _row_schema, parse_mirror_roots
from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    build_exl3_expert_layout,
)

# Rows per task: one O_DIRECT write of 4 x 13.3 MB per root on the real checkpoint.
CHUNK_ROWS = 4
DEFAULT_THREADS = 16
# Free space a root must keep beyond what the build writes to it.
SPACE_MARGIN = 1 << 30
SIDECAR_SUFFIX = ".digests.json"
TMP_SUFFIX = ".tmp"
# Mismatching rows printed per root; the count is always given.
SHOWN_MISMATCHES = 20

EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_NOT_COMPLETE = 0, 1, 2, 3


@dataclass(frozen=True)
class Source:
    """The checkpoint the images are built from, and everything derived from it."""

    model_dir: str
    layout: Exl3ExpertLayout
    segments: tuple
    image: ri.RowImageLayout
    fingerprint: dict

    @classmethod
    def load(cls, model_dir: str) -> "Source":
        layout = build_exl3_expert_layout(model_dir)
        _, segments = _row_schema(layout)
        return cls(
            model_dir=model_dir,
            layout=layout,
            segments=segments,
            image=ri.row_image_layout(segments),
            fingerprint=ri.source_fingerprint(layout, model_dir),
        )

    @property
    def layer_bytes(self) -> int:
        return self.layout.num_experts * self.image.row_stride

    def sidecar(self, layer: int, digests: Sequence[str]) -> dict:
        return {
            "format": ri.FORMAT,
            "version": ri.VERSION,
            "layer": layer,
            "file": ri.layer_file_name(layer),
            "layout": self.image.to_json(),
            "source": self.fingerprint,
            "digests": list(digests),
        }


@dataclass(frozen=True)
class Mismatch:
    root: str
    layer: int
    expert: int
    expected: str
    found: str


def layer_path(root: str, layer: int) -> str:
    return os.path.join(ri.row_image_dir(root), ri.layer_file_name(layer))


def sidecar_path(root: str, layer: int) -> str:
    return layer_path(root, layer) + SIDECAR_SUFFIX


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path))


def _remove(path: str) -> bool:
    try:
        os.remove(path)
    except FileNotFoundError:
        return False
    return True


def _read_at_least(fd: int, view: memoryview, offset: int, need: int) -> None:
    """Fill ``view`` from ``offset`` until at least ``need`` bytes arrived (the tail may lie past EOF)."""
    got = 0
    while got < need:
        n = os.preadv(fd, [view[got:]], offset + got)
        if n == 0:
            raise OSError(f"short read: {got} of {need} B at offset {offset}")
        got += n


def _write_all(fd: int, view: memoryview, offset: int) -> None:
    done = 0
    while done < len(view):
        done += os.pwritev(fd, [view[done:]], offset + done)


class _Buffers(threading.local):
    """Page-aligned (mmap) buffers per thread, as O_DIRECT needs; grown on demand, zero-filled when made."""

    def get(self, name: str, nbytes: int) -> memoryview:
        buf = self.__dict__.get(name)
        if buf is None or len(buf) < nbytes:
            buf = mmap.mmap(-1, max(nbytes, ri.PAGE))
            self.__dict__[name] = buf
        return memoryview(buf)


_BUFFERS = _Buffers()


def _run(threads: int, tasks: Sequence[Callable[[], object]]) -> list:
    """Run ``tasks`` on a pool; on the first failure cancel the rest and raise it."""
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(task) for task in tasks]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            if future.done() and future.exception() is not None:
                for other in pending:
                    other.cancel()
                raise future.exception()
        return [future.result() for future in futures]


def _chunks(num_experts: int) -> list[tuple[int, int]]:
    return [(first, min(CHUNK_ROWS, num_experts - first)) for first in range(0, num_experts, CHUNK_ROWS)]


def _gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f} GB"


class _Progress:
    """Per-layer progress lines of one phase, with throughput and an ETA."""

    def __init__(self, log: Callable[[str], None], phase: str, total_units: int, unit_bytes: int) -> None:
        self.log, self.phase = log, phase
        self.total, self.unit_bytes = total_units, unit_bytes
        self.done = 0
        self.start = time.monotonic()
        self.lock = threading.Lock()

    def unit_done(self, what: str) -> None:
        with self.lock:
            self.done += 1
            elapsed = time.monotonic() - self.start
            moved = self.done * self.unit_bytes
            eta = elapsed / self.done * (self.total - self.done)
            self.log(
                f"[{self.phase}] {what}: {self.done}/{self.total}, {_gb(moved)} in {elapsed:.0f} s "
                f"({moved / max(elapsed, 1e-9) / 1e9:.2f} GB/s), eta {eta:.0f} s"
            )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


# --- Reading images back ----------------------------------------------------------------------------------------


def _readback_chunk(src: Source, root: str, layer: int, first: int, count: int, expected: Sequence[str]) -> list[Mismatch]:
    """O_DIRECT read of experts ``first..first+count`` of a layer file; the rows whose digest is not ``expected``."""
    stride, image_bytes = src.image.row_stride, src.image.image_bytes
    view = _BUFFERS.get("back", CHUNK_ROWS * stride)[: count * stride]
    fd = os.open(layer_path(root, layer), os.O_RDONLY | os.O_DIRECT)
    try:
        _read_at_least(fd, view, first * stride, len(view))
    finally:
        os.close(fd)
    bad = []
    for i in range(count):
        found = ri.row_digest(view[i * stride : i * stride + image_bytes])
        if found != expected[first + i]:
            bad.append(Mismatch(root, layer, first + i, expected[first + i], found))
    return bad


def _readback(
    src: Source,
    pairs: Sequence[tuple[int, str, Sequence[str]]],
    threads: int,
    log: Callable[[str], None],
    phase: str,
) -> dict[tuple[int, str], list[Mismatch]]:
    """Read back every (layer, root) of ``pairs`` against its expected digests; mismatches per pair."""
    progress = _Progress(log, phase, len(pairs), src.layer_bytes)
    chunks = _chunks(src.layout.num_experts)
    remaining = {(layer, root): len(chunks) for layer, root, _ in pairs}
    found: dict[tuple[int, str], list[Mismatch]] = {(layer, root): [] for layer, root, _ in pairs}
    lock = threading.Lock()

    def task(layer, root, expected, first, count):
        def run():
            bad = _readback_chunk(src, root, layer, first, count, expected)
            with lock:
                found[(layer, root)] += bad
                remaining[(layer, root)] -= 1
                last = remaining[(layer, root)] == 0
            if last:
                progress.unit_done(f"layer {layer} on {root}")

        return run

    _run(threads, [task(layer, root, expected, first, count) for layer, root, expected in pairs for first, count in chunks])
    return found


# --- Building ---------------------------------------------------------------------------------------------------


class _LayerJob:
    """One layer's build: its source rows read once, their images written to ``roots``."""

    def __init__(self, src: Source, layer: int, roots: Sequence[str]) -> None:
        self.src, self.layer, self.roots = src, layer, tuple(roots)
        self.digests: list[Optional[str]] = [None] * src.layout.num_experts
        self.remaining = len(_chunks(src.layout.num_experts))
        self.fds: dict[str, int] = {}
        self.lock = threading.Lock()

    def open_files(self) -> None:
        """Create the tmp files (once): drop the old sidecar first, so an old one never vouches for a new file."""
        with self.lock:
            if self.fds:
                return
            for root in self.roots:
                _remove(sidecar_path(root, self.layer))
                _fsync_dir(ri.row_image_dir(root))
                fd = os.open(
                    layer_path(root, self.layer) + TMP_SUFFIX,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT,
                    0o644,
                )
                os.posix_fallocate(fd, 0, self.src.layer_bytes)
                self.fds[root] = fd

    def chunk_done(self) -> bool:
        with self.lock:
            self.remaining -= 1
            return self.remaining == 0


def _finish_layer(job: _LayerJob) -> None:
    """Make a fully written layer durable and visible: fsync, rename into place, then its sidecar."""
    for root in list(job.fds):
        fd = job.fds.pop(root)
        os.fsync(fd)
        os.close(fd)
        os.replace(layer_path(root, job.layer) + TMP_SUFFIX, layer_path(root, job.layer))
        _fsync_dir(ri.row_image_dir(root))
        _write_json_atomic(sidecar_path(root, job.layer), job.src.sidecar(job.layer, job.digests))


def _build_chunk(job: _LayerJob, first: int, count: int, src_fds: dict[str, int], progress: _Progress) -> None:
    src, layer = job.src, job.layer
    stride, image_bytes = src.image.row_stride, src.image.image_bytes
    job.open_files()
    raw = _BUFFERS.get("src", src.layout.row_bytes + 2 * ri.PAGE)
    out = _BUFFERS.get("out", CHUNK_ROWS * stride)
    for i in range(count):
        record = src.layout.records[(layer, first + i)]
        start, length, lead = record.aligned_read(ri.PAGE)
        _read_at_least(src_fds[record.path], raw[:length], start, lead + record.nbytes)
        image = ri.image_of_row(src.image, raw[lead : lead + record.nbytes])
        job.digests[first + i] = ri.row_digest(image)
        # The padding after each image in ``out`` is never written, so it stays the zeros mmap gave it.
        out[i * stride : i * stride + image_bytes] = image
    for fd in job.fds.values():
        _write_all(fd, out[: count * stride], first * stride)
    if job.chunk_done():
        _finish_layer(job)
        progress.unit_done(f"layer {layer} -> {len(job.roots)} root(s)")


def _build_layers(src: Source, todo: dict[int, list[str]], threads: int, log: Callable[[str], None]) -> tuple[dict[int, _LayerJob], float]:
    jobs = {layer: _LayerJob(src, layer, roots) for layer, roots in sorted(todo.items()) if roots}
    if not jobs:
        return {}, 0.0
    shard_paths = sorted({src.layout.records[(layer, e)].path for layer in jobs for e in range(src.layout.num_experts)})
    src_fds = {path: os.open(path, os.O_RDONLY | os.O_DIRECT) for path in shard_paths}
    progress = _Progress(log, "build", len(jobs), src.layer_bytes)
    try:
        tasks = [
            (lambda job=job, first=first, count=count: _build_chunk(job, first, count, src_fds, progress))
            for job in jobs.values()
            for first, count in _chunks(src.layout.num_experts)
        ]
        _run(threads, tasks)
    finally:
        for fd in src_fds.values():
            os.close(fd)
        for job in jobs.values():
            for fd in job.fds.values():
                os.close(fd)
    return jobs, progress.elapsed


# --- Checks -----------------------------------------------------------------------------------------------------


def check_roots(src: Source, roots: Sequence[str]) -> None:
    """Refuse a root the build must not write: the source itself, or images of another source or layout."""
    shards = {r.path for r in src.layout.records.values()}
    forbidden = {os.path.realpath(src.model_dir)} | {os.path.dirname(os.path.realpath(p)) for p in shards}
    want_layout = src.image.to_json()
    for root in roots:
        if os.path.realpath(root) in forbidden:
            raise ValueError(f"{root!r} is the source checkpoint directory (or holds its shards); images go on a mirror root")
        if not os.access(root, os.W_OK):
            raise ValueError(f"{root!r} is not writable")
        manifest = os.path.join(ri.row_image_dir(root), ri.MANIFEST)
        if not os.path.exists(manifest):
            continue
        try:
            m = ri.read_manifest(root)
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest} is not a readable manifest ({error}); remove it to rebuild") from None
        if (m.get("format"), m.get("version")) != (ri.FORMAT, ri.VERSION) or m.get("layout") != want_layout:
            raise ValueError(f"{manifest} describes another format or image layout; remove {ri.row_image_dir(root)} to rebuild")
        if m.get("source") != src.fingerprint:
            raise ValueError(
                f"{manifest} holds row images of a different source than {src.model_dir}; "
                f"remove {ri.row_image_dir(root)} to rebuild"
            )


def check_space(src: Source, todo: dict[int, list[str]], roots: Sequence[str]) -> None:
    for root in roots:
        need = sum(src.layer_bytes for layer_roots in todo.values() if root in layer_roots)
        if need == 0:
            continue
        free = _free_bytes(ri.row_image_dir(root))
        if free < need + SPACE_MARGIN:
            raise ValueError(
                f"{root!r} has {_gb(free)} free; the build writes {_gb(need)} there and keeps {_gb(SPACE_MARGIN)} spare"
            )


def _valid_sidecar(src: Source, root: str, layer: int) -> Optional[list[str]]:
    """The digests of a finished layer file on ``root``, if its sidecar describes this source and layout."""
    path, side = layer_path(root, layer), sidecar_path(root, layer)
    if not (os.path.exists(path) and os.path.exists(side)) or os.path.getsize(path) != src.layer_bytes:
        return None
    try:
        with open(side) as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    want = src.sidecar(layer, s.get("digests", []))
    if s != want or len(want["digests"]) != src.layout.num_experts:
        return None
    return want["digests"]


# --- Commands ---------------------------------------------------------------------------------------------------


def _report_mismatches(found: dict[tuple[int, str], list[Mismatch]], log: Callable[[str], None]) -> int:
    bad = [m for ms in found.values() for m in ms]
    for root in sorted({m.root for m in bad}):
        rows = sorted((m for m in bad if m.root == root), key=lambda m: (m.layer, m.expert))
        log(f"MISMATCH on {root}: {len(rows)} row(s) differ from their expected digest")
        for m in rows[:SHOWN_MISMATCHES]:
            log(f"    layer {m.layer} expert {m.expert}: expected {m.expected}, read back {m.found}")
    return len(bad)


def build(src: Source, roots: Sequence[str], layers: Sequence[int], threads: int, log: Callable[[str], None]) -> int:
    t0 = time.monotonic()
    check_roots(src, roots)
    for root in roots:
        directory = ri.row_image_dir(root)
        os.makedirs(directory, exist_ok=True)
        # The set is unreadable from here until this run's manifest lands.
        if _remove(os.path.join(directory, ri.MANIFEST)):
            log(f"removed the existing manifest of {directory}; it is rewritten when this run completes")
        _remove(os.path.join(directory, ri.MANIFEST + TMP_SUFFIX))
        _fsync_dir(directory)
    log(
        f"source {src.model_dir}: {src.layout.num_layers} layers x {src.layout.num_experts} experts; image "
        f"{src.image.image_bytes} B, stride {src.image.row_stride} B, {_gb(src.layer_bytes)} per layer per root"
    )

    candidates = [(layer, root, d) for layer in layers for root in roots if (d := _valid_sidecar(src, root, layer)) is not None]
    kept: dict[tuple[int, str], list[str]] = {}
    if candidates:
        log(f"resume: re-reading {len(candidates)} finished layer file(s) against their sidecars")
        found = _readback(src, candidates, threads, log, "resume")
        for layer, root, digests in candidates:
            if found[(layer, root)]:
                log(f"resume: layer {layer} on {root}: {len(found[(layer, root)])} row(s) differ from its sidecar; rebuilding it")
            else:
                kept[(layer, root)] = digests
    todo = {layer: [root for root in roots if (layer, root) not in kept] for layer in layers}
    check_space(src, todo, roots)
    rebuilt = sum(len(r) for r in todo.values())
    log(f"build: {rebuilt} layer file(s) to write, {len(kept)} kept")

    jobs, build_s = _build_layers(src, todo, threads, log)
    written = sum(len(job.roots) for job in jobs.values()) * src.layer_bytes
    if jobs:
        read = len(jobs) * src.layer_bytes
        log(
            f"build: {len(jobs)} layer(s), {_gb(read)} of images from the source, {_gb(written)} written in "
            f"{build_s:.0f} s ({read / build_s / 1e9:.2f} GB/s of rows, {written / build_s / 1e9:.2f} GB/s written)"
        )
        pairs = [(layer, root, job.digests) for layer, job in jobs.items() for root in job.roots]
        verify_start = time.monotonic()
        found = _readback(src, pairs, threads, log, "verify")
        verify_s = time.monotonic() - verify_start
        log(f"verify: {_gb(written)} read back in {verify_s:.0f} s ({written / verify_s / 1e9:.2f} GB/s)")
        if _report_mismatches(found, log):
            for (layer, root), ms in found.items():
                if ms:
                    _remove(sidecar_path(root, layer))  # a rerun rebuilds it
            log("FAIL: read-back mismatches; no manifest written")
            return EXIT_MISMATCH
        for job in jobs.values():
            for root in job.roots:
                kept[(job.layer, root)] = list(job.digests)

    for layer in layers:
        lists = {root: kept[(layer, root)] for root in roots}
        if len({tuple(d) for d in lists.values()}) != 1:
            log(f"FAIL: layer {layer}'s row digests differ between roots; no manifest written")
            return EXIT_MISMATCH
    for root in roots:
        ri.write_manifest(root, ri.manifest_json(src.image, src.fingerprint, {layer: kept[(layer, root)] for layer in layers}))
    ri.open_row_images(roots, src.layout, src.segments, src.model_dir, layers)
    for root in roots:
        log(f"manifest: {os.path.join(ri.row_image_dir(root), ri.MANIFEST)} ({len(layers)} layers)")
    log(f"done in {time.monotonic() - t0:.0f} s")
    return EXIT_OK


def verify(src: Source, roots: Sequence[str], layers: Sequence[int], threads: int, log: Callable[[str], None]) -> int:
    t0 = time.monotonic()
    try:
        ri.open_row_images(roots, src.layout, src.segments, src.model_dir, layers)
    except ValueError as error:
        log(f"FAIL: {error}")
        return EXIT_MISMATCH
    pairs = []
    for root in roots:
        m = ri.read_manifest(root)
        pairs += [(layer, root, m["layers"][str(layer)]["digests"]) for layer in layers]
    found = _readback(src, pairs, threads, log, "verify")
    elapsed = time.monotonic() - t0
    total = len(pairs) * src.layer_bytes
    log(f"verify: {_gb(total)} read back with O_DIRECT in {elapsed:.0f} s ({total / elapsed / 1e9:.2f} GB/s)")
    if _report_mismatches(found, log):
        log("FAIL")
        return EXIT_MISMATCH
    if len(layers) != src.layout.num_layers:
        log(f"PASS (PARTIAL): {len(layers)} of {src.layout.num_layers} layers on {len(roots)} root(s)")
        return EXIT_NOT_COMPLETE
    log(f"PASS: every row of all {len(layers)} layers on {len(roots)} root(s) matches its manifest digest")
    return EXIT_OK


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", default=None, help="the source checkpoint (default: SGLANG_DSV41_EXPERT_DIR)")
    parser.add_argument(
        "--roots",
        default=None,
        help=f"{os.pathsep!r}-separated mirror roots (default: SGLANG_MOE_EXPERT_MIRROR_DIRS)",
    )
    parser.add_argument("--layers", default="all", help="'all' (default), or layers such as 0,3,10-12")
    parser.add_argument("--verify", action="store_true", help="re-check an existing image set; write nothing")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"I/O threads (default {DEFAULT_THREADS})")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log = lambda line: print(line, flush=True)  # noqa: E731
    try:
        model_dir = args.model_dir or envs.SGLANG_DSV41_EXPERT_DIR.get()
        if not model_dir:
            raise ValueError("no source given: pass --model-dir or set SGLANG_DSV41_EXPERT_DIR")
        roots_spec = args.roots or envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.get()
        if not roots_spec:
            raise ValueError("no roots given: pass --roots or set SGLANG_MOE_EXPERT_MIRROR_DIRS")
        roots = parse_mirror_roots(roots_spec)
        if args.threads < 1:
            raise ValueError("--threads must be at least 1")
        src = Source.load(model_dir)
        layers = parse_layers(args.layers)
        layers = list(range(src.layout.num_layers)) if layers is None else layers
        bad = [layer for layer in layers if layer >= src.layout.num_layers]
        if bad:
            raise ValueError(f"layers {bad} are outside the checkpoint's {src.layout.num_layers}")
        run = verify if args.verify else build
        return run(src, roots, layers, args.threads, log)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
