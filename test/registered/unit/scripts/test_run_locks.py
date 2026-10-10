"""run_locks.take: a lock the caller's own ancestor holds is not waited on; any other holder is waited on, by name."""

import os
import subprocess
import sys
import time

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs flock(1) and /proc/locks")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TAKER = (
    "import sys; sys.path.insert(0, {lib!r}); import run_locks\n"
    "f = run_locks.take({path!r})\n"
    "print('TAKEN', flush=True)\n"
)


def _taker(path):
    lib = os.path.join(ROOT, "benchmarks", "dsv41_baseline")
    return [sys.executable, "-c", TAKER.format(lib=lib, path=str(path))]


def test_a_lock_the_parent_already_holds_is_taken_without_waiting(tmp_path):
    lock = tmp_path / "disk.lock"
    # bash in between: the holder is a grandparent, as under `flock LOCK bash run.sh`.
    out = subprocess.run(["flock", str(lock), "bash", "-c", '"$@"; exit $?', "bash", *_taker(lock)],
                         capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    assert "TAKEN" in out.stdout
    assert "held by our parent" in out.stdout


def test_a_lock_another_process_holds_is_waited_on_and_the_holder_named(tmp_path):
    lock = tmp_path / "disk.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "300"])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and str(holder.pid) not in open("/proc/locks").read():
            time.sleep(0.05)
        taker = subprocess.Popen(_taker(lock), stdout=subprocess.PIPE, text=True)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                taker.wait(timeout=2)
            holder.kill()
            holder.wait()
            out, _ = taker.communicate(timeout=10)
        finally:
            taker.kill()
        assert f"waiting for {lock} held by pid {holder.pid}" in out
        assert out.rstrip().endswith("TAKEN")
    finally:
        holder.kill()


def test_a_free_lock_is_taken_silently(tmp_path):
    out = subprocess.run(_taker(tmp_path / "free.lock"), capture_output=True, text=True, timeout=20)
    assert (out.returncode, out.stdout) == (0, "TAKEN\n"), out.stderr


DRIVERS = (
    "analysis/dsv41-drive/dspark/both_cpu_ab.py",
    "analysis/dsv41-drive/dspark/spec_margin_capture.py",
    "analysis/dsv41-drive/prefetch-replay/capture_verify.py",
    "benchmarks/dsv41_baseline/run_stall_capture.py",
)


@pytest.mark.parametrize("driver", DRIVERS)
def test_no_driver_waits_on_a_run_lock_directly(driver):
    src = open(os.path.join(ROOT, driver)).read()
    blocking = [line for line in src.splitlines() if "fcntl.flock(" in line and "LOCK_EX" in line and "LOCK_NB" not in line]
    assert blocking == [], f"{driver} should take its locks through run_locks.take: {blocking}"
