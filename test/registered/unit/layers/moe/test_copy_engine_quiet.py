"""The quiet wait before the EXL3 copy engine's calibration: its busy accounting, its timeout, and the drive probes
read from sysfs (CPU)."""

import os

from sglang.srt.layers.moe.copy_engine_quiet import (
    block_device_inflight_paths,
    inflight_busy,
    wait_for_quiet,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def busy_for(polls):
    left = [polls]

    def probe():
        left[0] -= 1
        return left[0] >= 0

    return probe


def test_the_wait_ends_at_the_first_poll_where_every_probe_is_idle_and_reports_busy_time():
    clock = FakeClock()
    report = wait_for_quiet(
        {"ram_miss_thread": busy_for(3), "nvme0n1_inflight": busy_for(1), "copy_engine_jobs": busy_for(0)},
        deadline=clock() + 30.0,
        busy_s={"scheduler_requests": 2.0},
        poll_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )
    assert report.quiet and report.still_busy == ()
    assert abs(report.waited_s - 0.3) < 1e-9
    assert set(report.busy_s) == {"scheduler_requests", "ram_miss_thread", "nvme0n1_inflight"}
    assert abs(report.busy_s["ram_miss_thread"] - 0.3) < 1e-9 and report.busy_s["scheduler_requests"] == 2.0
    assert report.line().startswith("quiet after 0.30 s (busy: scheduler_requests 2.00 s, ram_miss_thread 0.30 s")
    assert report.line().endswith("checks: ram_miss_thread, nvme0n1_inflight, copy_engine_jobs)")


def test_the_wait_gives_up_at_the_deadline_and_names_what_was_still_busy():
    clock = FakeClock()
    report = wait_for_quiet(
        {"nvme1n1_inflight": lambda: True, "ram_miss_thread": busy_for(2)},
        deadline=clock() + 1.0,
        poll_s=0.25,
        clock=clock,
        sleep=clock.sleep,
    )
    assert not report.quiet and report.still_busy == ("nvme1n1_inflight",)
    assert report.waited_s >= 1.0
    assert "NOT quiet after 1.00 s, calibrating anyway: still busy nvme1n1_inflight" in report.line()


def test_the_drive_probe_reads_the_whole_disk_under_each_directory(tmp_path):
    dev = os.stat(tmp_path).st_dev
    sys_root = tmp_path / "sys"
    disk = sys_root / "devices" / "nvme" / "nvme7n1"
    part = disk / "nvme7n1p1"
    part.mkdir(parents=True)
    (part / "partition").write_text("1\n")
    (part / "inflight").write_text("       0        0\n")
    (disk / "inflight").write_text("       2        1\n")
    (sys_root / "dev" / "block").mkdir(parents=True)
    os.symlink(part, sys_root / "dev" / "block" / f"{os.major(dev)}:{os.minor(dev)}")
    paths = block_device_inflight_paths([str(tmp_path), str(tmp_path / "missing")], sys_root=str(sys_root))
    assert paths == {"nvme7n1": str(disk / "inflight")}
    assert inflight_busy(paths["nvme7n1"])
    (disk / "inflight").write_text("       0        0\n")
    assert not inflight_busy(paths["nvme7n1"])
    assert not inflight_busy(str(tmp_path / "no-such-file"))
