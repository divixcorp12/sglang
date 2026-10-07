"""launch_prod.sh stop / restart, against stand-in processes: never the live server.

Every test points the script at a port and a lock file of its own (LAUNCH_PROD_PORT, LAUNCH_PROD_GPU_LOCK), so a run
on divix01 while production serves on 7867 cannot touch it. Linux only (pgrep, flock, /proc).
"""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs pgrep, flock and /proc")

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_prod.sh")
PORT = 47867  # never production's 7867


def _stand_in(port, *, ignore_term=False):
    """A process whose command line reads like a server on `port`."""
    trap = "trap '' TERM; " if ignore_term else ""
    argv0 = f"/data/models/slang/.venv/bin/python -m sglang.launch_server --model-path m --port {port} --host 0.0.0.0"
    proc = subprocess.Popen(["bash", "-c", f'{trap}exec -a "{argv0}" sleep 300'], start_new_session=True)
    for _ in range(50):
        with open(f"/proc/{proc.pid}/cmdline", "rb") as f:
            if b"sglang.launch_server" in f.read():
                return proc
        time.sleep(0.05)
    raise AssertionError("the stand-in never took its command line")


def _run(tmp_path, *args, **env):
    full = dict(os.environ, LAUNCH_PROD_PORT=str(PORT), LAUNCH_PROD_GPU_LOCK=str(tmp_path / "gpu.lock"))
    full.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", SCRIPT, *args], env=full, capture_output=True, text=True, timeout=120)


def _gone(proc):
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return False
    return True


def _kill(*procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_stop_ends_the_server_on_its_port_and_no_other(tmp_path):
    prod, other = _stand_in(PORT), _stand_in(PORT + 1)
    try:
        r = _run(tmp_path, "stop")
        assert r.returncode == 0, r.stderr
        assert _gone(prod)
        assert other.poll() is None, "a server on another port was stopped"
        assert f"stopped production (pid {prod.pid})" in r.stdout
    finally:
        _kill(prod, other)


def test_stop_kills_a_server_that_ignores_sigterm_after_the_timeout(tmp_path):
    prod = _stand_in(PORT, ignore_term=True)
    try:
        r = _run(tmp_path, "stop", LAUNCH_PROD_STOP_TIMEOUT_S=2)
        assert r.returncode == 0, r.stderr
        assert _gone(prod)
        assert "SIGKILL" in r.stderr
    finally:
        _kill(prod)


def test_stop_with_nothing_running_succeeds(tmp_path):
    r = _run(tmp_path, "stop")
    assert r.returncode == 0, r.stderr
    assert "production is not running" in r.stdout


def test_stop_fails_while_another_job_still_holds_the_gpu_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "300"])
    prod = _stand_in(PORT)
    try:
        time.sleep(0.3)
        r = _run(tmp_path, "stop", LAUNCH_PROD_LOCK_WAIT_S=1)
        assert _gone(prod)
        assert r.returncode != 0
        assert "cc-gpu.lock" in r.stderr
    finally:
        _kill(holder, prod)


def test_a_dry_run_restart_stops_and_starts_nothing(tmp_path):
    prod = _stand_in(PORT)
    try:
        r = _run(tmp_path, "restart", DRY_RUN=1, LAUNCH_PROD_LOG_ROOT=tmp_path / "servers")
        assert r.returncode == 0, r.stderr
        assert f"would stop production (pid {prod.pid})" in r.stdout
        assert "would start production, log " + str(tmp_path / "servers") in r.stdout
        assert prod.poll() is None
        assert not (tmp_path / "servers").exists()
    finally:
        _kill(prod)


def test_an_unknown_command_is_refused_with_the_usage(tmp_path):
    r = _run(tmp_path, "reboot")
    assert r.returncode == 2
    assert "usage: launch_prod.sh [start|stop|restart]" in r.stderr
