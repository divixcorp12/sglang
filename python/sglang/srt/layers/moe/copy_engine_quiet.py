"""Wait for the box to go quiet before the EXL3 RAM-miss copy engine's calibration (CPU split, split model).

Those calibrations time CPU expert work, PCIe copies and drive reads; any other CPU, PCIe or NVMe traffic in their
window skews the costs they publish. ``wait_for_quiet`` polls host-side probes (no device read) until every one is
idle at the same poll, or a deadline passes, and reports which probes were busy and for how long.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Iterable, Optional

import msgspec

# Bounds the wait for quiet, the scheduler's idle included; past it the calibration runs anyway.
QUIET_TIMEOUT_S = 30.0
QUIET_POLL_S = 0.01


class QuietReport(msgspec.Struct, frozen=True):
    quiet: bool
    waited_s: float
    # Probe name -> seconds it was seen busy, for every probe that was busy at least once.
    busy_s: dict[str, float]
    # The probes still busy at the deadline (empty when quiet).
    still_busy: tuple[str, ...]
    probes: tuple[str, ...]

    def line(self) -> str:
        busy = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in self.busy_s.items()) or "none"
        checks = ", ".join(self.probes)
        if self.quiet:
            return f"quiet after {self.waited_s:.2f} s (busy: {busy}; checks: {checks})"
        return (
            f"NOT quiet after {self.waited_s:.2f} s, calibrating anyway: still busy {', '.join(self.still_busy)} "
            f"(busy: {busy}; checks: {checks})"
        )


def wait_for_quiet(
    probes: dict[str, Callable[[], bool]],
    *,
    deadline: float,
    busy_s: Optional[dict[str, float]] = None,
    poll_s: float = QUIET_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> QuietReport:
    """Poll ``probes`` (name -> is busy) until none is busy or ``clock()`` passes ``deadline``.

    ``busy_s`` carries busy time measured earlier (the scheduler's idle wait) into the report.
    """
    start = clock()
    busy_s = dict(busy_s or {})
    last = start
    while True:
        busy = [name for name, probe in probes.items() if probe()]
        now = clock()
        for name in busy:
            busy_s[name] = busy_s.get(name, 0.0) + max(now - last, poll_s)
        last = now
        if not busy or now >= deadline:
            return QuietReport(
                quiet=not busy,
                waited_s=now - start,
                busy_s=busy_s,
                still_busy=tuple(busy),
                probes=tuple(probes),
            )
        sleep(poll_s)


def block_device_inflight_paths(dirs: Iterable[str], *, sys_root: str = "/sys") -> dict[str, str]:
    """The ``inflight`` file of the whole disk under each existing directory in ``dirs``, by disk name.

    A partition's own counters miss the disk's other partitions, so the disk's are read. A directory that does not
    exist, or whose device has no sysfs entry (tmpfs, a network mount), is skipped.
    """
    paths = {}
    for directory in dirs:
        try:
            dev = os.stat(directory).st_dev
        except OSError:
            continue
        node = os.path.realpath(os.path.join(sys_root, "dev", "block", f"{os.major(dev)}:{os.minor(dev)}"))
        if os.path.exists(os.path.join(node, "partition")):
            node = os.path.dirname(node)
        inflight = os.path.join(node, "inflight")
        if os.path.exists(inflight):
            paths[os.path.basename(node)] = inflight
    return paths


def inflight_busy(path: str) -> bool:
    """Whether the block device has reads or writes in flight (its sysfs ``inflight``: reads, writes)."""
    try:
        with open(path) as f:
            return any(int(value) for value in f.read().split())
    except (OSError, ValueError):
        return False


def cuda_module_loading_env() -> dict[str, str]:
    """The process's ``CUDA_MODULE_LOADING`` alone, as the mapping the expert engine's copy-engine launch check takes.

    Read here, outside the engine package, which reads no process environment.
    """
    value = os.environ.get("CUDA_MODULE_LOADING")
    return {} if value is None else {"CUDA_MODULE_LOADING": value}
