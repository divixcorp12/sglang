"""``drive_conditions.py`` against fake /proc and /sys trees; no drive is read.

    OMP_NUM_THREADS=4 taskset -c 0-63 python -m pytest \\
        analysis/dsv41-drive/test_drive_conditions.py -p no:cacheprovider
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_conditions as dc  # noqa: E402

MIB = 1 << 20


def _diskstats_line(major, minor, name, *, reads=0, merged=0, sectors=0, read_ms=0,
                    in_flight=0, io_ms=0, weighted=0):  # fmt: skip
    return (
        f"{major:>4} {minor:>7} {name} {reads} {merged} {sectors} {read_ms} "
        f"0 0 0 0 {in_flight} {io_ms} {weighted} 0 0 0 0 0 0\n"
    )


def _fake_tree(tmp_path, mount_dir, *, name="nvme3n1p1", disk="nvme3n1", max_sectors_kb=256,
               fs="ext4", row=True, model="SPCC M.2 PCIe SSD", **stats):  # fmt: skip
    """/proc and /sys for a directory whose mount is called nvme4 but whose device is nvme3n1p1."""
    major, minor = dc.block_device_of(str(mount_dir))
    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    (proc / "self").mkdir(parents=True, exist_ok=True)
    (proc / "diskstats").write_text(
        _diskstats_line(major, minor, name, **stats) if row else _diskstats_line(1, 1, "other")
    )
    (proc / "self" / "mountinfo").write_text(
        f"36 25 {major}:{minor} / / rw - xfs /dev/root rw\n"
        f"40 36 {major}:{minor} / {os.path.realpath(mount_dir)} rw - {fs} /dev/{name} rw\n"
    )
    disk_dir = sys_root / "devices" / disk
    part_dir = disk_dir / name
    (disk_dir / "queue").mkdir(parents=True, exist_ok=True)
    (disk_dir / "device").mkdir(exist_ok=True)
    part_dir.mkdir(exist_ok=True)
    (disk_dir / "device" / "model").write_text(model + "\n")
    for key, value in (("max_sectors_kb", max_sectors_kb), ("max_hw_sectors_kb", max_sectors_kb),
                       ("max_segments", 65), ("chunk_sectors", 0)):  # fmt: skip
        (disk_dir / "queue" / key).write_text(f"{value}\n")
    (sys_root / "dev" / "block").mkdir(parents=True, exist_ok=True)
    link = sys_root / "dev" / "block" / f"{major}:{minor}"
    if not link.exists():
        link.symlink_to(part_dir)
    return str(proc), str(sys_root)


def test_a_mount_named_nvme4_resolves_to_the_device_it_is_on(tmp_path):
    mount = tmp_path / "nvme4"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount)
    drive = dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)
    assert drive.device == "nvme3n1p1"  # not "nvme4", which has no diskstats row anywhere
    assert (drive.max_sectors_kb, drive.max_segments, drive.fs_type) == (256, 65, "ext4")
    assert drive.max_request_bytes == 256 * 1024
    assert drive.model == "SPCC M.2 PCIe SSD"


def test_a_device_without_a_diskstats_row_is_refused_not_read_as_zero(tmp_path):
    mount = tmp_path / "tmpfs_like"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount, row=False)
    with pytest.raises(ValueError, match="no .*diskstats row"):
        dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)


def test_a_missing_queue_directory_is_refused(tmp_path):
    mount = tmp_path / "m"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount)
    shutil.rmtree(Path(sys_root) / "devices" / "nvme3n1" / "queue")
    with pytest.raises(ValueError, match="queue directory"):
        dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)


def test_filesystem_type_takes_the_longest_mount_prefix(tmp_path):
    mount = tmp_path / "data"
    (mount / "deep").mkdir(parents=True)
    proc, _ = _fake_tree(tmp_path, mount, fs="xfs")
    assert dc.filesystem_type(str(mount / "deep"), proc_root=proc) == "xfs"
    assert dc.filesystem_type(str(tmp_path), proc_root=proc) == "xfs"  # only the root entry prefixes it


def test_one_extent_is_twice_the_requests_on_a_256k_cap_as_on_a_512k_cap():
    extent = 6_660_096  # a measured extent: one whole row on one drive
    assert dc.block_requests(length=extent, max_request_bytes=512 * 1024) == 13
    assert dc.block_requests(length=extent, max_request_bytes=256 * 1024) == 26


def test_block_requests_edges():
    cap = 256 * 1024
    assert dc.block_requests(length=0, max_request_bytes=cap) == 0
    assert dc.block_requests(length=cap, max_request_bytes=cap) == 1
    assert dc.block_requests(length=cap + 1, max_request_bytes=cap) == 2
    with pytest.raises(ValueError):
        dc.block_requests(length=1, max_request_bytes=0)


def test_counters_and_depth_come_from_the_right_diskstats_fields(tmp_path):
    mount = tmp_path / "m"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount, reads=100, merged=7, sectors=1000, read_ms=50,
                                io_ms=200, weighted=1000)  # fmt: skip
    drive = dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)
    before = dc.counters_of(drive.device, proc_root=proc)
    major, minor = drive.major, drive.minor
    Path(proc, "diskstats").write_text(
        _diskstats_line(major, minor, drive.device, reads=126, merged=9, sectors=1000 + 2048,
                        read_ms=90, io_ms=500, weighted=4000)  # fmt: skip
    )
    after = dc.counters_of(drive.device, proc_root=proc)
    delta = dc.counters_delta(drive, before, after, wall_s=1.0)
    assert delta.reads_completed == 26 and delta.reads_merged == 2
    assert delta.read_bytes == 2048 * 512  # diskstats sectors are 512 B whatever the device's size
    assert delta.mean_queue_depth == pytest.approx(3.0)  # 3000 weighted ms over 1000 ms
    assert delta.depth_when_busy == pytest.approx(10.0)  # 3000 weighted ms over 300 busy ms
    assert delta.busy_fraction == pytest.approx(0.3)


def test_depth_when_busy_is_none_for_an_idle_drive(tmp_path):
    mount = tmp_path / "m"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount)
    drive = dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)
    snap = dc.counters_of(drive.device, proc_root=proc)
    assert dc.counters_delta(drive, snap, snap, wall_s=1.0).depth_when_busy is None


def test_a_vanished_device_raises(tmp_path):
    mount = tmp_path / "m"
    mount.mkdir()
    proc, _ = _fake_tree(tmp_path, mount)
    with pytest.raises(ValueError, match="vanished"):
        dc.counters_of("nvme9n1", proc_root=proc)


def test_load_average(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "loadavg").write_text("4.60 3.10 2.00 5/900 12345\n")
    assert dc.load_average(proc_root=str(proc)) == [4.6, 3.1, 2.0]


def _fake_stat(tmp_path, per_cpu, own_utime=0, own_stime=0):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True, exist_ok=True)
    total = [sum(c[i] for c in per_cpu) for i in range(len(per_cpu[0]))]
    rows = ["cpu " + " ".join(map(str, total))] + [
        f"cpu{i} " + " ".join(map(str, c)) for i, c in enumerate(per_cpu)
    ]
    (proc / "stat").write_text("\n".join(rows) + "\nintr 0\n")
    # pid (comm with) spaces) state ppid ... utime is field 14 and stime 15 counting from 1
    fields = ["S"] + ["0"] * 10 + [str(own_utime), str(own_stime)] + ["0"] * 10
    (proc / "self" / "stat").write_text("1 (a b) " + " ".join(fields) + "\n")
    return str(proc)


def test_foreign_cpu_excludes_this_process_and_splits_low_cores(tmp_path):
    # columns: user nice system idle iowait irq softirq steal
    cpus0 = [[100, 0, 0, 900, 0, 0, 0, 0], [0, 0, 0, 1000, 0, 0, 0, 0]]
    cpus1 = [[300, 0, 0, 900, 0, 0, 0, 0], [200, 0, 0, 1000, 0, 0, 0, 0]]
    before = dc.read_cpu_times(low_cpus=1, proc_root=_fake_stat(tmp_path, cpus0, 10, 5))
    after = dc.read_cpu_times(low_cpus=1, proc_root=_fake_stat(tmp_path, cpus1, 110, 55))
    assert (after.busy_all - before.busy_all, after.busy_low - before.busy_low) == (400, 200)
    assert after.own - before.own == 150
    foreign = dc.foreign_cpu_cores(before, after, wall_s=2.0, hz=100)
    # 400 busy jiffies over 200 = 2.0 cores, minus our own 150 / 200 = 0.75
    assert foreign["foreign_cores_all"] == pytest.approx(1.25)
    assert foreign["foreign_cores_low"] == pytest.approx(0.25)


def test_foreign_cpu_without_a_wall_time_is_unknown():
    zero = dc.CpuTimes(0, 0, 0, 0)
    assert dc.foreign_cpu_cores(zero, zero, wall_s=0.0, hz=100)["foreign_cores_all"] is None


@pytest.mark.skipif(shutil.which("fincore") is None, reason="fincore not installed")
def test_residency_counts_resident_shard_bytes_only(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * (2 * MIB))
    (tmp_path / "notes.txt").write_bytes(b"y" * MIB)
    got = dc.residency(str(tmp_path))
    assert got.total_bytes == 2 * MIB
    assert 0 <= got.resident_bytes <= 2 * MIB


def test_residency_is_none_not_zero_when_it_cannot_be_measured(tmp_path, monkeypatch):
    (tmp_path / "a.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    assert dc.residency(str(tmp_path)) is None


def test_probe_reports_every_condition_for_every_drive(tmp_path):
    mount = tmp_path / "nvme4"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount, reads=10, sectors=100, io_ms=10, weighted=20)
    (Path(proc) / "loadavg").write_text("4.6 3.0 2.0 1/2 3\n")
    _fake_stat(tmp_path, [[0, 0, 0, 100, 0, 0, 0, 0]])
    drive = dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)
    seen = iter([dc.Residency(15 * MIB, 200 * MIB), dc.Residency(15 * MIB + 8 * MIB, 200 * MIB)])
    probe = dc.ConditionProbe([drive], low_cpus=1, proc_root=proc, residency_fn=lambda _: next(seen))
    started = probe.start()
    Path(proc, "diskstats").write_text(
        _diskstats_line(drive.major, drive.minor, drive.device, reads=36, sectors=100 + 128,
                        io_ms=110, weighted=520)  # fmt: skip
    )
    out = probe.stop(started)
    cell = out["drives"][0]
    assert cell["device"] == "nvme3n1p1" and cell["path"] == str(mount)
    assert cell["reads_completed"] == 26 and cell["read_bytes"] == 128 * 512
    assert cell["depth_when_busy"] == pytest.approx(5.0)
    assert cell["residency_delta_bytes"] == 8 * MIB
    assert cell["residency_before"] == {"resident_bytes": 15 * MIB, "total_bytes": 200 * MIB}
    assert out["load_average_before"] == [4.6, 3.0, 2.0]
    assert set(out) >= {"wall_s", "foreign_cores_all", "foreign_cores_low", "load_average_after"}


def test_probe_marks_unmeasured_residency_as_none(tmp_path):
    mount = tmp_path / "m"
    mount.mkdir()
    proc, sys_root = _fake_tree(tmp_path, mount)
    (Path(proc) / "loadavg").write_text("1 1 1 1/1 1\n")
    _fake_stat(tmp_path, [[0, 0, 0, 100, 0, 0, 0, 0]])
    drive = dc.resolve_drive(str(mount), proc_root=proc, sys_root=sys_root)
    probe = dc.ConditionProbe([drive], low_cpus=1, proc_root=proc, residency_fn=lambda _: None)
    cell = probe.stop(probe.start())["drives"][0]
    assert cell["residency_before"] is None and cell["residency_delta_bytes"] is None
