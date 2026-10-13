"""The quick run's quiet-box preflight: GPU compute pids and non-mirror drive users (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
MOUNTS = ("/mnt/nvme0", "/mnt/nvme2", "/mnt/nvme4")
ROOTS = tuple(f"{m}/dsv41_flash" for m in MOUNTS)

LSOF_HEAD = "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"


def _preflight():
    spec = importlib.util.spec_from_file_location(
        "l2_t3_preflight", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "l2_t3_preflight.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lsof(*rows):
    return LSOF_HEAD + "".join(f"{cmd} {pid} u 5r REG 259,1 100 7 {path}\n" for cmd, pid, path in rows)


def test_any_gpu_compute_pid_is_reported():
    """Mutant: gpu_compute_pids drops rows or the header-less csv's first column -- red."""
    smi = "1234, python\n 5678, /usr/bin/Xorg\n"
    assert _preflight().gpu_compute_pids(smi) == [1234, 5678]
    assert _preflight().gpu_compute_pids("") == []


def test_mirror_only_reader_passes():
    """Mutant: files under the mirror roots are counted as drive users -- red."""
    text = _lsof(("py", 10, "/mnt/nvme4/dsv41_flash/layer3/expert.bin"), ("py", 10, "/mnt/nvme0/dsv41_flash/a"))
    assert _preflight().drive_users(text, mounts=MOUNTS, allowed_roots=ROOTS) == {}


def test_file_elsewhere_on_a_mount_refuses():
    """Mutant: the allow-list matches by mount prefix instead of the mirror root -- red."""
    text = _lsof(
        ("py", 10, "/mnt/nvme4/dsv41_flash/layer3/expert.bin"),
        ("py", 10, "/mnt/nvme4/scratch/other.bin"),
        ("dd", 11, "/mnt/nvme2/dsv41_flash2/x"),
    )
    users = _preflight().drive_users(text, mounts=MOUNTS, allowed_roots=ROOTS)
    assert users == {10: ["/mnt/nvme4/scratch/other.bin"], 11: ["/mnt/nvme2/dsv41_flash2/x"]}


def _diskstats(**sectors):
    # /proc/diskstats: major minor name reads merged sectors_read ms writes merged sectors_written ms ...
    return "".join(f" 259 1 {d} 1 0 {r} 0 1 0 {w} 0 0 0 0\n" for d, (r, w) in sectors.items())


def test_drive_busy_counts_read_and_written_sectors_of_named_devices_only():
    """Mutant: only reads are counted, sectors are not scaled by 512, or an unnamed device leaks in -- red."""
    pf = _preflight()
    before = pf.diskstat_bytes(_diskstats(nvme0n1p1=(0, 0), nvme3n1p1=(0, 0)), devices={"nvme0n1p1"})
    after = pf.diskstat_bytes(_diskstats(nvme0n1p1=(4000, 6000), nvme3n1p1=(10**9, 0)), devices={"nvme0n1p1"})
    assert after == {"nvme0n1p1": 10000 * 512}
    assert pf.busy_devices(before, after, seconds=1.0) == {"nvme0n1p1": 10000 * 512 / 1e6}
    assert pf.busy_devices(before, after, seconds=10.0) == {}


def test_path_with_spaces_is_kept_whole():
    """Mutant: NAME is cut at the first space, hiding a non-mirror path -- red."""
    text = _lsof(("py", 12, "/mnt/nvme0/my files/a b.bin"))
    users = _preflight().drive_users(text, mounts=MOUNTS, allowed_roots=ROOTS)
    assert users == {12: ["/mnt/nvme0/my files/a b.bin"]}
