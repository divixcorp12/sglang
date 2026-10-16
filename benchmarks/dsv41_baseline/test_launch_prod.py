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
        r = _run(tmp_path, "restart", DRY_RUN=1, LAUNCH_PROD_LOG_ROOT=tmp_path / "servers",
                 PRELAUNCH_NODE_SYSFS=_fake_nodes(tmp_path, 86.0, 84.0))
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


def _fake_nodes(tmp_path, free0_gib, free1_gib, file1_gib=2.0):
    """A sysfs node directory whose meminfo reads as the given free memory (PRELAUNCH_NODE_SYSFS)."""
    root = tmp_path / "sysfs"
    for node, free, file_gib in ((0, free0_gib, 1.0), (1, free1_gib, file1_gib)):
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "meminfo").write_text(
            f"Node {node} MemTotal:       100000000 kB\n"
            f"Node {node} MemFree:        {int(free * 2**20)} kB\n"
            f"Node {node} FilePages:      {int(file_gib * 2**20)} kB\n"
            f"Node {node} Shmem:          0 kB\n"
        )
    return root


def _floor_env(tmp_path, free0_gib, free1_gib):
    holders = tmp_path / "holders"
    (holders / "big-model").mkdir(parents=True)
    (holders / "big-model" / "shard.bin").write_bytes(os.urandom(1 << 20))  # just written: in the page cache
    evictable = tmp_path / "evictable"
    evictable.mkdir()
    (evictable / "rows").write_bytes(b"0" * 8192)
    return dict(
        PRELAUNCH_NODE_SYSFS=_fake_nodes(tmp_path, free0_gib, free1_gib),
        PRELAUNCH_EVICT_PATHS=evictable,
        LAUNCH_PROD_PAGE_CACHE_ROOTS=holders,
        LAUNCH_PROD_HOLDERS_BUDGET_S=10,
        LAUNCH_PROD_LOG_ROOT=tmp_path / "servers",
    )


def test_a_dry_run_start_reports_the_memory_floor_and_would_start_above_it(tmp_path):
    r = _run(tmp_path, "start", DRY_RUN=1, **_floor_env(tmp_path, 86.0, 84.0))
    assert r.returncode == 0, r.stderr
    # arm_env.PROD_NODE1_*: HiCache 12.67 + arm A's rest 51.24 + 1 GiB margin.
    assert "node-1 floor 64.91 GiB" in r.stdout
    assert "evicted 0 files" in r.stdout  # a dry run evicts nothing
    assert "would start: node 1 is above its floor" in r.stdout
    assert not (tmp_path / "servers").exists()


def test_a_dry_run_start_below_the_floor_names_the_nodes_the_floor_and_the_page_cache_holders(tmp_path):
    r = _run(tmp_path, "start", DRY_RUN=1, **_floor_env(tmp_path, 70.0, 20.0))
    assert r.returncode == 0, r.stderr  # a dry run decides, it does not fail
    assert "would refuse: node 1 has 20.0 GiB free" in r.stderr
    assert "below its floor of 64.91 GiB; node 0 has 70.0 GiB free" in r.stderr
    assert "WARNING node 0 has 70.0 GiB free" in r.stderr  # under PROD_NODE0_WARN_GIB: a warning, not a refusal
    assert str(tmp_path / "holders" / "big-model") in r.stderr
    assert "drop_caches" in r.stderr


def test_the_floor_check_runs_under_the_lock_and_before_the_server_starts():
    text = open(SCRIPT).read()
    start = text.index("flock --nonblock 9")
    check = text.index('if ! prelaunch "$prelaunch_log"', start)
    assert start < check < text.index("exec taskset", check)
