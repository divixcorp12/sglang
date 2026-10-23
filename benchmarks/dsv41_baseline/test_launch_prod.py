"""launch_prod.sh stop / restart, against stand-in processes: never the live server.

Every test points the script at a port and a lock file of its own (LAUNCH_PROD_PORT, LAUNCH_PROD_GPU_LOCK), so a run
on divix01 while production serves on 7867 cannot touch it. Linux only (pgrep, flock, /proc).
"""

import hashlib
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


def _head(tmp_path, content=b"a stand-in prefetch head", name="head.safetensors"):
    """A head file and its sha256, for LAUNCH_PROD_HEAD / LAUNCH_PROD_HEAD_SHA256."""
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _run(tmp_path, *args, **env):
    head, sha = _head(tmp_path, name="default-head.safetensors")  # every run checks a head; tests may name another
    sts, sts_sha = _head(tmp_path, b"a stand-in sts file", "default-sts.json")  # and the cut's STS file
    full = dict(os.environ, LAUNCH_PROD_PORT=str(PORT), LAUNCH_PROD_GPU_LOCK=str(tmp_path / "gpu.lock"),
                LAUNCH_PROD_HEAD=str(head), LAUNCH_PROD_HEAD_SHA256=sha,
                LAUNCH_PROD_STS=str(sts), LAUNCH_PROD_STS_SHA256=sts_sha)
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


def test_the_script_runs_arm_envs_interpreter():
    sys.path.insert(0, os.path.dirname(SCRIPT))
    import arm_env

    with open(SCRIPT) as f:
        assignments = [line.strip() for line in f if line.startswith("py=")]
    assert assignments == [f"py={arm_env.PYTHON}"]


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
                 PRELAUNCH_NODE_SYSFS=_fake_nodes(tmp_path, 150.0, 140.0))
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
    r = _run(tmp_path, "start", DRY_RUN=1, **_floor_env(tmp_path, 150.0, 140.0))
    assert r.returncode == 0, r.stderr
    # arm_env.PROD_NODE1_*: HiCache 12.67 + arm A's rest 51.24 + production's layer-major StateStore 5.0 + the pinned
    # tier beyond arm A's 48 GiB on node 1 (94,450 - 49,152 MiB = 44.24 GiB) + 1 GiB margin.
    assert "node-1 floor 114.15 GiB" in r.stdout
    assert "+ layer-major StateStore 5.0 + pinned tier beyond arm A 44.24 + margin 1.0" in r.stdout
    assert "evicted 0 files" in r.stdout  # a dry run evicts nothing
    assert "would start: every checked node is above its floor" in r.stdout
    assert not (tmp_path / "servers").exists()


def test_a_dry_run_start_below_the_floor_names_the_nodes_the_floor_and_the_page_cache_holders(tmp_path):
    r = _run(tmp_path, "start", DRY_RUN=1, **_floor_env(tmp_path, 70.0, 20.0))
    assert r.returncode == 0, r.stderr  # a dry run decides, it does not fail
    assert "would refuse: node 1 has 20.0 GiB free" in r.stderr
    assert "below its floor of 114.15 GiB" in r.stderr
    assert "node 0 has 70.0 GiB free" in r.stderr  # each node's free memory, the short one first (prelaunch-cache)
    assert "WARNING node 0 has 70.0 GiB free" in r.stderr  # under PROD_NODE0_WARN_GIB: a warning, not a refusal
    assert str(tmp_path / "holders" / "big-model") in r.stderr
    assert "drop_caches" in r.stderr


def test_a_dry_run_start_verifies_the_prefetch_heads_checksum(tmp_path):
    head, sha = _head(tmp_path, b"the head production loads")
    r = _run(tmp_path, "start", DRY_RUN=1, LAUNCH_PROD_HEAD=head, LAUNCH_PROD_HEAD_SHA256=sha,
             **_floor_env(tmp_path, 150.0, 140.0))
    assert r.returncode == 0, r.stderr
    assert f"prefetch head verified: {head} sha256 {sha}" in r.stdout


def test_a_start_refuses_a_prefetch_head_that_does_not_match_or_is_missing(tmp_path):
    head, sha = _head(tmp_path, b"the head production loads")
    wrong = hashlib.sha256(b"another head").hexdigest()
    # Dry runs only: a real start that got past a broken check would launch production.
    for cmd in ("start", "restart"):
        (tmp_path / cmd).mkdir()
        r = _run(tmp_path, cmd, DRY_RUN=1, LAUNCH_PROD_HEAD=head, LAUNCH_PROD_HEAD_SHA256=wrong,
                 **_floor_env(tmp_path / cmd, 150.0, 140.0))
        assert r.returncode == 1, (cmd, r.stdout, r.stderr)
        assert f"has sha256 {sha}, not the pinned {wrong}" in r.stderr
        assert "prelaunch:" not in r.stdout  # refused before the memory floor
    r = _run(tmp_path, "start", DRY_RUN=1, LAUNCH_PROD_HEAD=tmp_path / "missing.safetensors",
             LAUNCH_PROD_HEAD_SHA256=sha)
    assert r.returncode == 1 and "is missing" in r.stderr
    # The real start runs the same check_assets before anything else (test_the_real_start_checks_the_assets_...).


def test_a_dry_run_start_verifies_the_confidence_sts_files_checksum_and_refuses_a_wrong_or_missing_one(tmp_path):
    sts, sha = _head(tmp_path, b"the sts file production loads", "sts.json")
    r = _run(tmp_path, "start", DRY_RUN=1, LAUNCH_PROD_STS=sts, LAUNCH_PROD_STS_SHA256=sha,
             **_floor_env(tmp_path, 150.0, 140.0))
    assert r.returncode == 0, r.stderr
    assert f"confidence STS file verified: {sts} sha256 {sha}" in r.stdout
    wrong = hashlib.sha256(b"another file").hexdigest()
    for cmd in ("start", "restart"):
        (tmp_path / cmd).mkdir()
        r = _run(tmp_path, cmd, DRY_RUN=1, LAUNCH_PROD_STS=sts, LAUNCH_PROD_STS_SHA256=wrong,
                 **_floor_env(tmp_path / cmd, 150.0, 140.0))
        assert r.returncode == 1, (cmd, r.stdout, r.stderr)
        assert f"has sha256 {sha}, not the pinned {wrong}" in r.stderr and "prelaunch:" not in r.stdout
    r = _run(tmp_path, "start", DRY_RUN=1, LAUNCH_PROD_STS=tmp_path / "missing.json", LAUNCH_PROD_STS_SHA256=sha)
    assert r.returncode == 1 and "STS file" in r.stderr and "is missing" in r.stderr


def test_the_real_start_checks_the_assets_before_the_lock():
    text = open(SCRIPT).read()
    lock = text.index('exec 9>"$gpu_lock"')
    assert text.rindex("check_assets || exit 1", 0, lock) > text.index('if [ "$dry" = 1 ]; then\n    echo "checkout')


def test_the_floors_state_store_term_follows_the_envs_layer_major_threshold():
    sys.path.insert(0, os.path.dirname(SCRIPT))
    import arm_env

    # 131072 tokens x (4 x 5120 bf16 + 4 fp32) = 5.00 GiB, for production (layer-major on) and nothing for the arms.
    assert arm_env.LAYER_MAJOR_STATE_BYTES_PER_TOKEN == 40976
    assert arm_env.layer_major_state_gib(arm_env.prod_env()) == 5.0
    assert arm_env.layer_major_state_gib(arm_env.arm_env(arm_env.dspark_env())) == 0.0
    assert arm_env.layer_major_state_gib({}) == 0.0


def test_the_floor_check_runs_under_the_lock_and_before_the_server_starts():
    text = open(SCRIPT).read()
    start = text.index("flock --nonblock 9")
    check = text.index('if ! prelaunch "$prelaunch_log"', start)
    assert start < check < text.index("exec taskset", check)
