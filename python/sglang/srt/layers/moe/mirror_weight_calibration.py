"""Mirror read weights measured at startup (SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION).

A mirrored row is read from every root at once, each root its weight's share of the row
(SGLANG_MOE_EXPERT_MIRROR_WEIGHTS, StaticSplitPolicy). With equal shares a slower drive
finishes last and the row waits for it: on divix01 nvme3 (max_sectors_kb 256) was busy
104.5 s on demand reads against 86.6 / 85.6 s on nvme0 / nvme2 for equal bytes
(cores-ab-20261014/splitmodel-stream-mem-tiebreak). Shares in proportion to each
root's read throughput give every drive the same busy time.

With the flag on and SGLANG_MOE_EXPERT_MIRROR_WEIGHTS unset, ``probed_weights`` reads
each root's row images before the tables are built: PROBE_DEPTH threads per root, all
roots at once as serving reads them, issue O_DIRECT reads of one sub-read each (a row's
share of a root over its pieces, so the drive's request splitting is what serving sees)
at random rows of the images, for PROBE_ROUNDS rounds of PROBE_SECONDS. A root's weight
is its mean throughput over the mean of every root's, clamped to WEIGHT_RANGE.

Guardrails: a root that read nothing, or whose rounds disagree by more than NOISE_LIMIT of
its mean (a drive busy with something else, a probe too short to settle), fails the
probe; so does a missing manifest or file. A failure warns and keeps the static weights
(equal shares). An explicit SGLANG_MOE_EXPERT_MIRROR_WEIGHTS always wins and is not
probed. The result is computed once per process and roots (every caller of
exl3_mirror_config sees the same weights) and logged with the throughputs.
"""

from __future__ import annotations

import logging
import mmap
import os
import random
import threading
import time
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

PROBE_DEPTH = 4  # reads in flight per root
PROBE_ROUNDS = 3
PROBE_SECONDS = 0.15  # per round
PIECES = 8  # the reader's pieces per row (expert_stream_transport.STAGE_PIECES)
ALIGN = 4096
NOISE_LIMIT = 0.25
# A weight over the mean, at most this far either way: past it the probe more likely measured a drive's transient
# state than its speed.
WEIGHT_RANGE = (0.5, 2.0)

_CACHE: dict = {}


class MirrorProbeError(RuntimeError):
    """The probe could not measure the roots, or measured noise."""


def weights_from_rates(rates: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], list]:
    """(weights, clamped root indices) from rates[round][root] in bytes/s: each root's mean over every root's mean,
    clamped to WEIGHT_RANGE. Raises MirrorProbeError for a root with no throughput or rounds that disagree by more than
    NOISE_LIMIT of the root's mean."""
    if not rates or not rates[0]:
        raise MirrorProbeError("no probe rounds")
    roots = len(rates[0])
    means = []
    for root in range(roots):
        values = [r[root] for r in rates]
        mean = sum(values) / len(values)
        if mean <= 0:
            raise MirrorProbeError(f"root {root} read nothing")
        spread = (max(values) - min(values)) / mean
        if spread > NOISE_LIMIT:
            raise MirrorProbeError(
                f"root {root}'s rounds disagree by {spread:.0%} of its mean ({', '.join(f'{v / 1e9:.2f}' for v in values)}"
                f" GB/s), more than {NOISE_LIMIT:.0%}"
            )
        means.append(mean)
    overall = sum(means) / roots
    weights, clamped = [], []
    lo, hi = WEIGHT_RANGE
    for root, mean in enumerate(means):
        w = mean / overall
        if not lo <= w <= hi:
            clamped.append(root)
        weights.append(round(min(max(w, lo), hi), 4))
    return tuple(weights), clamped


def root_files(root: str, layers: int = 4) -> tuple[list, int]:
    """Up to ``layers`` of ``root``'s row-image files and their row stride (file bytes over experts)."""
    from sglang.srt.layers.moe.exl3_row_image import read_manifest, row_image_dir

    m = read_manifest(root)
    entries = sorted(m.get("layers", {}).items(), key=lambda kv: int(kv[0]))[:layers]
    if not entries:
        raise MirrorProbeError(f"{root} has no row images")
    files = []
    for _, entry in entries:
        path = os.path.join(row_image_dir(root), entry["file"])
        experts = len(entry["digests"])
        files.append((path, os.path.getsize(path), experts))
    stride = files[0][1] // files[0][2]
    return [p for p, _, _ in files], stride


def chunk_bytes(stride: int, roots: int) -> int:
    """One sub-read: a row's share of a root over the reader's pieces, ALIGN-rounded."""
    return max(ALIGN, -(-stride // (roots * PIECES * ALIGN)) * ALIGN)


def read_rate(paths: Sequence[str], chunk: int, seconds: float, *, depth: int = PROBE_DEPTH, direct: bool = True,
              seed: int = 0, start: Optional[threading.Barrier] = None) -> float:
    """Bytes/s of ``depth`` threads reading ``chunk``-byte, chunk-aligned blocks at random offsets of ``paths`` for
    ``seconds`` (O_DIRECT when ``direct``). ``start`` lines the threads up with other roots' probes."""
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fds = [os.open(p, flags) for p in paths]
    sizes = [os.fstat(fd).st_size // chunk for fd in fds]
    if not all(sizes):
        raise MirrorProbeError(f"a row-image file is smaller than one {chunk}-byte read")
    total = [0] * depth
    deadline = [0.0]
    ready = threading.Barrier(depth + 1)
    errors = []

    def worker(i: int) -> None:
        rng = random.Random(seed * 1000 + i)
        buf = mmap.mmap(-1, chunk)  # page-aligned, as O_DIRECT needs
        try:
            ready.wait()
            while time.monotonic() < deadline[0]:
                f = rng.randrange(len(fds))
                n = os.preadv(fds[f], [buf], rng.randrange(sizes[f]) * chunk)
                if n <= 0:
                    raise MirrorProbeError(f"a read of {paths[f]} returned {n}")
                total[i] += n
        except Exception as e:  # noqa: BLE001 - reported by the caller
            errors.append(e)
        finally:
            buf.close()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(depth)]
    for t in threads:
        t.start()
    try:
        if start is not None:
            start.wait()
        began = time.monotonic()
        deadline[0] = began + seconds
        ready.wait()
    except threading.BrokenBarrierError:
        ready.abort()  # another root's probe failed: release this root's threads unread
        raise MirrorProbeError("another root's probe failed") from None
    finally:
        for t in threads:
            t.join()
        for fd in fds:
            os.close(fd)
    elapsed = time.monotonic() - began
    if errors:
        raise MirrorProbeError(f"a probe read failed: {errors[0]!r}")
    return sum(total) / elapsed


def probe_rates(roots: Sequence[str], *, rounds: int = PROBE_ROUNDS, seconds: float = PROBE_SECONDS,
                direct: bool = True) -> tuple[list, int]:
    """rates[round][root] (bytes/s), every root read at once each round, and the sub-read size."""
    files = [root_files(r) for r in roots]
    chunk = chunk_bytes(files[0][1], len(roots))
    rates = []
    for r in range(rounds):
        out = [0.0] * len(roots)
        errors = []
        start = threading.Barrier(len(roots))

        def one(i: int) -> None:
            try:
                out[i] = read_rate(files[i][0], chunk, seconds, direct=direct, seed=r, start=start)
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)
                start.abort()

        threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(roots))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise MirrorProbeError(f"probe round {r} failed: {errors[0]!r}")
        rates.append(out)
    return rates, chunk


def probed_weights(roots: Sequence[str], probe: Optional[Callable] = None) -> tuple[float, ...]:
    """The mirror weights for ``roots`` from the probe (module doc), once per process; equal weights, with a warning,
    when it fails. ``probe`` replaces probe_rates (tests)."""
    key = tuple(roots)
    if key in _CACHE:
        return _CACHE[key]
    began = time.monotonic()
    try:
        rates, chunk = (probe or probe_rates)(key)
        weights, clamped = weights_from_rates(rates)
    except (MirrorProbeError, OSError, ValueError, KeyError) as error:
        weights = (1.0,) * len(key)
        logger.warning("mirror weight calibration failed (%s); keeping the static weights %s", error,
                       ":".join(f"{w:g}" for w in weights))
    else:
        means = [sum(r[i] for r in rates) / len(rates) / 1e9 for i in range(len(key))]
        logger.info(
            "mirror weight calibration in %.2f s, %d KiB reads x %d deep, %d rounds: %s -> weights %s%s",
            time.monotonic() - began, chunk // 1024, PROBE_DEPTH, len(rates),
            ", ".join(f"{root} {gb:.2f} GB/s" for root, gb in zip(key, means)),
            ":".join(f"{w:g}" for w in weights),
            f" (clamped to {WEIGHT_RANGE[0]:g}..{WEIGHT_RANGE[1]:g} of the mean: roots {clamped})" if clamped else "",
        )
    _CACHE[key] = weights
    return weights
