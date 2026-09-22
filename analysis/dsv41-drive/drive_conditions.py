"""Drive identity, block-request model and run conditions for the row benchmarks.

Everything here reads /proc and /sys (roots are parameters, so tests can supply
fake trees) and issues no I/O against a data drive. A figure recorded without
these conditions cannot be compared with another: the two mirrors differ in
block-layer limits, filesystem and page-cache residency, and only a resolved
device name says which drive a byte counter belongs to.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

SECTOR_BYTES = 512  # /proc/diskstats sector unit, independent of the device's logical size
PAGE_BYTES = 4096


class DriveInfo(NamedTuple):
    path: str
    device: str  # kernel block-device name, e.g. nvme3n1 for a mount named nvme4
    major: int
    minor: int
    fs_type: str
    max_sectors_kb: int  # block-layer request size cap; a larger extent becomes several requests
    max_hw_sectors_kb: int
    max_segments: int
    chunk_sectors: int  # 0 unless the device asks for requests not to cross a boundary
    model: str

    @property
    def max_request_bytes(self) -> int:
        return self.max_sectors_kb * 1024


class DiskCounters(NamedTuple):
    reads_completed: int
    reads_merged: int
    sectors_read: int
    read_ms: int
    io_in_progress: int
    io_ms: int  # time the device had at least one request in flight
    weighted_io_ms: int  # in-flight requests integrated over time


class DiskDelta(NamedTuple):
    device: str
    read_bytes: int
    reads_completed: int
    reads_merged: int
    read_ms: int
    mean_queue_depth: Optional[float]  # weighted_io_ms / wall_ms; None without a wall time
    busy_fraction: Optional[float]  # io_ms / wall_ms
    depth_when_busy: Optional[float]  # weighted_io_ms / io_ms: in-flight requests while not idle
    weighted_io_ms: int
    io_ms: int


def _read_text(path: Path) -> str:
    return path.read_text().strip()


def _read_int(path: Path, default: int = 0) -> int:
    try:
        return int(_read_text(path))
    except (OSError, ValueError):
        return default


def block_device_of(path: str) -> tuple[int, int]:
    st = os.stat(path).st_dev
    return os.major(st), os.minor(st)


def _diskstats_rows(proc_root: str) -> dict:
    rows = {}
    with open(Path(proc_root) / "diskstats") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 14:
                rows[(int(fields[0]), int(fields[1]))] = (fields[2], fields[3:])
    return rows


def filesystem_type(path: str, *, proc_root: str = "/proc") -> str:
    """Type of the mount that holds ``path``: the longest mount point that prefixes it."""
    real = os.path.realpath(path)
    best, fs = "", "unknown"
    with open(Path(proc_root) / "self" / "mountinfo") as f:
        for line in f:
            left, _, right = line.partition(" - ")
            mount = left.split()[4].replace("\\040", " ")
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) >= len(best):
                best, fs = mount, right.split()[0]
    return fs


def _queue_dir(sys_root: str, major: int, minor: int) -> Optional[Path]:
    """The queue/ directory of the disk under (major, minor); a partition keeps it on its parent."""
    node = Path(sys_root) / "dev" / "block" / f"{major}:{minor}"
    for candidate in (node, node.resolve().parent):
        if (candidate / "queue").is_dir():
            return candidate / "queue"
    return None


def resolve_drive(path: str, *, proc_root: str = "/proc", sys_root: str = "/sys") -> DriveInfo:
    """Resolve a directory to its block device through st_dev, never through its name.

    Raises when the device has no /proc/diskstats row (tmpfs, overlay, a name that
    does not exist), because a counter keyed on a guessed name silently reads nothing.
    """
    major, minor = block_device_of(path)
    rows = _diskstats_rows(proc_root)
    if (major, minor) not in rows:
        raise ValueError(
            f"{path}: device {major}:{minor} has no {proc_root}/diskstats row; "
            "its I/O cannot be attributed (tmpfs, overlay or a network mount?)"
        )
    queue = _queue_dir(sys_root, major, minor)
    if queue is None:
        raise ValueError(f"{path}: no {sys_root} queue directory for {major}:{minor}")
    model_file = queue.parent / "device" / "model"
    return DriveInfo(
        path=path,
        device=rows[(major, minor)][0],
        major=major,
        minor=minor,
        fs_type=filesystem_type(path, proc_root=proc_root),
        max_sectors_kb=_read_int(queue / "max_sectors_kb"),
        max_hw_sectors_kb=_read_int(queue / "max_hw_sectors_kb"),
        max_segments=_read_int(queue / "max_segments"),
        chunk_sectors=_read_int(queue / "chunk_sectors"),
        model=_read_text(model_file) if model_file.is_file() else "unknown",
    )


def block_requests(*, length: int, max_request_bytes: int) -> int:
    """Block requests one extent becomes when the block layer only splits at the size cap.

    A model, not a count: the real number also depends on segment merging and on
    how the filesystem builds bios, and is measured from /proc/diskstats.
    """
    if length <= 0:
        return 0
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    return math.ceil(length / max_request_bytes)


def counters_of(device: str, *, proc_root: str = "/proc") -> DiskCounters:
    for name, fields in _diskstats_rows(proc_root).values():
        if name == device:
            v = [int(x) for x in fields]
            return DiskCounters(v[0], v[1], v[2], v[3], v[8], v[9], v[10])
    raise ValueError(f"{device} vanished from {proc_root}/diskstats")


def read_counters(drives: Sequence[DriveInfo], *, proc_root: str = "/proc") -> list[DiskCounters]:
    return [counters_of(d.device, proc_root=proc_root) for d in drives]


def counters_delta(
    drive: DriveInfo, before: DiskCounters, after: DiskCounters, *, wall_s: Optional[float]
) -> DiskDelta:
    wall_ms = wall_s * 1000.0 if wall_s else None
    return DiskDelta(
        device=drive.device,
        read_bytes=(after.sectors_read - before.sectors_read) * SECTOR_BYTES,
        reads_completed=after.reads_completed - before.reads_completed,
        reads_merged=after.reads_merged - before.reads_merged,
        read_ms=after.read_ms - before.read_ms,
        mean_queue_depth=(after.weighted_io_ms - before.weighted_io_ms) / wall_ms if wall_ms else None,
        busy_fraction=(after.io_ms - before.io_ms) / wall_ms if wall_ms else None,
        depth_when_busy=(
            (after.weighted_io_ms - before.weighted_io_ms) / (after.io_ms - before.io_ms)
            if after.io_ms > before.io_ms else None
        ),
        weighted_io_ms=after.weighted_io_ms - before.weighted_io_ms,
        io_ms=after.io_ms - before.io_ms,
    )


def load_average(*, proc_root: str = "/proc") -> list[float]:
    return [float(x) for x in _read_text(Path(proc_root) / "loadavg").split()[:3]]


class Residency(NamedTuple):
    resident_bytes: int
    total_bytes: int


def residency(directory: str, *, pattern: str = "*.safetensors") -> Optional[Residency]:
    """Page-cache bytes of the shards under ``directory`` (mincore via fincore; reads no data).

    None when fincore is missing or fails: an unmeasured residency must not read as zero.
    """
    exe = shutil.which("fincore")
    if exe is None:
        return None
    files = sorted(str(p) for p in Path(directory).rglob(pattern) if p.is_file())
    resident = 0
    for start in range(0, len(files), 512):
        run = subprocess.run(
            [exe, "-b", "-n", "--raw", "-o", "RES", *files[start : start + 512]],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if run.returncode != 0:
            return None
        resident += sum(int(x) for x in run.stdout.split())
    return Residency(resident, sum(os.path.getsize(f) for f in files))


class CpuTimes(NamedTuple):
    busy_all: int  # jiffies, every cpu
    busy_low: int  # jiffies, cpus 0..low_cpus-1 (the cores a benchmark may use)
    own: int  # this process's user+system jiffies
    total_all: int


def _cpu_lines(proc_root: str) -> list[list[int]]:
    with open(Path(proc_root) / "stat") as f:
        return [[int(x) for x in line.split()[1:]] for line in f if line.startswith("cpu")]


def _busy(fields: Sequence[int]) -> int:
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    return sum(fields[:8]) - idle


def read_cpu_times(*, low_cpus: int = 64, proc_root: str = "/proc") -> CpuTimes:
    lines = _cpu_lines(proc_root)
    aggregate, per_cpu = lines[0], lines[1:]
    stat = _read_text(Path(proc_root) / "self" / "stat")
    fields = stat[stat.rindex(")") + 2 :].split()  # after "pid (comm)": state is field 0
    return CpuTimes(
        busy_all=_busy(aggregate),
        busy_low=sum(_busy(c) for c in per_cpu[:low_cpus]),
        own=int(fields[11]) + int(fields[12]),
        total_all=sum(aggregate[:8]),
    )


def foreign_cpu_cores(before: CpuTimes, after: CpuTimes, *, wall_s: float, hz: int) -> dict:
    """Cores' worth of CPU used by everything except this process between two samples."""
    if wall_s <= 0:
        return {"foreign_cores_all": None, "foreign_cores_low": None}
    scale = hz * wall_s
    own = (after.own - before.own) / scale
    return {
        "foreign_cores_all": max(0.0, (after.busy_all - before.busy_all) / scale - own),
        "foreign_cores_low": max(0.0, (after.busy_low - before.busy_low) / scale - own),
    }


class ProbeStart(NamedTuple):
    residency: list
    load: list
    counters: list
    cpu: CpuTimes
    began: float


class ConditionProbe:
    """Conditions around one measured block: bracket it with ``start`` and ``stop``.

    Residency is sampled outside the timed interval (fincore is slow); the
    counters, load and CPU times bracket it tightly. Sampling costs the run
    nothing that any batch timing sees.
    """

    def __init__(
        self, drives: Sequence[DriveInfo], *, low_cpus: int = 64,
        proc_root: str = "/proc", residency_fn=residency,
    ) -> None:  # fmt: skip
        self.drives = tuple(drives)
        self.low_cpus = low_cpus
        self.proc_root = proc_root
        self.residency_fn = residency_fn
        self.hz = os.sysconf("SC_CLK_TCK")

    def start(self) -> ProbeStart:
        cache = [self.residency_fn(d.path) for d in self.drives]
        return ProbeStart(
            cache,
            load_average(proc_root=self.proc_root),
            read_counters(self.drives, proc_root=self.proc_root),
            read_cpu_times(low_cpus=self.low_cpus, proc_root=self.proc_root),
            time.perf_counter(),
        )

    def stop(self, start: ProbeStart) -> dict:
        wall_s = time.perf_counter() - start.began
        counters = read_counters(self.drives, proc_root=self.proc_root)
        cpu = read_cpu_times(low_cpus=self.low_cpus, proc_root=self.proc_root)
        load = load_average(proc_root=self.proc_root)
        cache = [self.residency_fn(d.path) for d in self.drives]
        return {
            "wall_s": wall_s,
            "drives": [
                {
                    "path": d.path, "device": d.device,
                    **counters_delta(d, before, after, wall_s=wall_s)._asdict(),
                    "residency_before": None if r0 is None else r0._asdict(),
                    "residency_after": None if r1 is None else r1._asdict(),
                    "residency_delta_bytes": (
                        None if r0 is None or r1 is None
                        else r1.resident_bytes - r0.resident_bytes
                    ),
                }
                for d, before, after, r0, r1 in zip(
                    self.drives, start.counters, counters, start.residency, cache
                )
            ],
            "load_average_before": start.load,
            "load_average_after": load,
            **foreign_cpu_cores(start.cpu, cpu, wall_s=wall_s, hz=self.hz),
        }  # fmt: skip
