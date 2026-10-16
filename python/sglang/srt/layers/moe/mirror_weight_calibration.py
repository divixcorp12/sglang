"""Mirror read weights measured at startup (SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION).

A mirrored row is read from every root at once, each root its weight's share of the row
(SGLANG_MOE_EXPERT_MIRROR_WEIGHTS, StaticSplitPolicy). With equal shares a slower drive
finishes last and the row waits for it: on divix01 the SPCC drive (nvme4 in
arm_env.EXPERT_MIRROR_DIRS, max_sectors_kb 256) was busy 104.5 s on demand reads against
86.6 / 85.6 s on the two Samsungs for equal bytes (cores-ab-20261014/splitmodel-stream-mem-tiebreak),
about 1.21x slower, so weights near 1.056 : 0.875 : 1.069 would even out the busy time.

The probe reproduces served demand, not a saturating benchmark. A first probe (4 random
reads in flight per root, back to back for 0.15 s) measured the SPCC 1.7x slower
(2.04 against 3.48 / 3.41 GB/s, split-calib-20261015/mirror1) and its weights would have
over-corrected: sustained queued reads flatter the Samsungs far more than demand does.
Demand arrives in bursts: a row's sub-reads (PIECES on every root, one row's share of
that root each) issued together, then nothing until the next row, so each drive starts
every burst idle. ``burst_rates`` does exactly that: BURSTS bursts a round, each at a
random row of the images, every root's PIECES sub-reads in flight at once (the served
queue depth of one row), timed from the burst's start until that root's last sub-read
lands, then an idle gap of IDLE_RATIO times the burst. A root's rate in a round is its
share's bytes over its median burst time; its weight is its mean rate over every root's
mean, clamped to WEIGHT_RANGE, so each root finishes a row's share at the same time.

Guardrails: a root that read nothing, or whose rounds disagree by more than NOISE_LIMIT of
its mean (a drive busy with something else), fails the probe; so does a missing manifest
or file. A failure warns and keeps the static weights (equal shares). An explicit
SGLANG_MOE_EXPERT_MIRROR_WEIGHTS always wins and is not probed. The result is computed
once per process and roots (every caller of exl3_mirror_config sees the same weights) and
logged with the rates. The flag stays off until a served start checks the weights
against the busy-time target above.
"""

from __future__ import annotations

import logging
import mmap
import os
import random
import statistics
import threading
import time
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

PIECES = 8  # the reader's pieces per row (expert_stream_transport.STAGE_PIECES): one row's sub-reads per root
PROBE_ROUNDS = 3
BURSTS = 40  # per round
# The idle gap after a burst, as a multiple of the burst's time: drives start each row idle, as in decode, where a
# layer's reads wait for its routing.
IDLE_RATIO = 1.0
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


def _preadv(fd: int, buf, offset: int) -> int:
    return os.preadv(fd, [buf], offset)


def burst_rates(files: Sequence[Sequence[str]], stride: int, *, rounds: int = PROBE_ROUNDS, bursts: int = BURSTS,
                idle_ratio: float = IDLE_RATIO, direct: bool = True, seed: int = 0,
                read: Callable = _preadv) -> tuple[list, int]:
    """rates[round][root] (bytes/s) of row bursts over ``files[root]`` (row-image files of one row ``stride``), and the
    sub-read size: the module doc's procedure. ``read(fd, buf, offset)`` replaces os.preadv (tests)."""
    roots = len(files)
    chunk = chunk_bytes(stride, roots)
    share = PIECES * chunk  # one root's share of a row
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fds = [[os.open(p, flags) for p in paths] for paths in files]
    rows = [[os.fstat(fd).st_size // stride for fd in root_fds] for root_fds in fds]
    if any(n == 0 for root_rows in rows for n in root_rows):
        raise MirrorProbeError(f"a row-image file is smaller than one {stride}-byte row")
    workers = roots * PIECES
    go, done = threading.Barrier(workers + 1), threading.Barrier(workers + 1)
    burst = {"file": 0, "row": 0, "stop": False}
    ends = [0.0] * workers
    errors = []

    def worker(w: int) -> None:
        root, piece = divmod(w, PIECES)
        buf = mmap.mmap(-1, chunk)  # page-aligned, as O_DIRECT needs
        try:
            while True:
                go.wait()
                if burst["stop"]:
                    return
                f = burst["file"] % len(fds[root])
                row = burst["row"] % rows[root][f]
                # Root p reads its share of the row, as the static split lays the shares out in root order.
                offset = row * stride + min(root * share + piece * chunk, stride - chunk)
                n = read(fds[root][f], buf, offset)
                if n <= 0:
                    raise MirrorProbeError(f"a read of {files[root][f]} returned {n}")
                ends[w] = time.monotonic()
                done.wait()
        except threading.BrokenBarrierError:
            return
        except Exception as e:  # noqa: BLE001 - reported by the caller
            errors.append(e)
            go.abort()
            done.abort()
        finally:
            buf.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(workers)]
    for t in threads:
        t.start()
    rng = random.Random(seed)
    rates = []
    try:
        for _ in range(rounds):
            times = [[] for _ in range(roots)]
            for _ in range(bursts):
                burst["file"], burst["row"] = rng.randrange(1 << 30), rng.randrange(1 << 30)
                start = time.monotonic()
                go.wait()
                done.wait()
                for root in range(roots):
                    times[root].append(max(ends[root * PIECES : (root + 1) * PIECES]) - start)
                time.sleep((time.monotonic() - start) * idle_ratio)
            rates.append([share / statistics.median(t) for t in times])
        burst["stop"] = True
        go.wait()
    except threading.BrokenBarrierError:
        raise MirrorProbeError(f"a probe read failed: {errors[0]!r}" if errors else "the probe broke") from None
    finally:
        go.abort()
        done.abort()
        for t in threads:
            t.join()
        for root_fds in fds:
            for fd in root_fds:
                os.close(fd)
    if errors:
        raise MirrorProbeError(f"a probe read failed: {errors[0]!r}")
    return rates, chunk


def probe_rates(roots: Sequence[str], *, direct: bool = True, **kwargs) -> tuple[list, int]:
    """burst_rates over each root's row images."""
    found = [root_files(r) for r in roots]
    return burst_rates([f for f, _ in found], found[0][1], direct=direct, **kwargs)


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
            "mirror weight calibration in %.2f s, row bursts of %d x %d KiB a root, %d rounds: %s -> weights %s%s",
            time.monotonic() - began, PIECES, chunk // 1024, len(rates),
            ", ".join(f"{root} {gb:.2f} GB/s" for root, gb in zip(key, means)),
            ":".join(f"{w:g}" for w in weights),
            f" (clamped to {WEIGHT_RANGE[0]:g}..{WEIGHT_RANGE[1]:g} of the mean: roots {clamped})" if clamped else "",
        )
    _CACHE[key] = weights
    return weights
