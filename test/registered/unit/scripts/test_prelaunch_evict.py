"""prelaunch_evict.py: the node-1 floor from an arm's previous memsample, else the arm-A default, and the eviction
itself on a scratch file (CPU)."""

import importlib.util
import json
import os
import socket
import subprocess
import sys

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark", "prelaunch_evict.py"
)
_spec = importlib.util.spec_from_file_location("prelaunch_evict", _PATH)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)

GIB = 2**20  # kB


def _sample(free1_gib, shmem1_gib):
    return json.dumps({"nodes": {"0": {"MemFree": 0}, "1": {"MemFree": free1_gib * GIB, "Shmem": shmem1_gib * GIB}}})


def test_the_rest_is_the_previous_runs_node1_need_less_its_hicache_pools(tmp_path):
    # 60 GiB free at start, 4 at the lowest: a 56 GiB need, of which 12 GiB of Shmem appeared (HiCache on node 1).
    lines = [_sample(60, 0.4), _sample(30, 0.4), _sample(4, 12.4), _sample(10, 12.4), "{\"numastat\": \"x\"}"]
    rest, pools = pe.rest_from_samples(lines)
    assert (round(rest, 3), round(pools, 3)) == (44.0, 12.0)
    path = tmp_path / "prev.memsample.jsonl"
    path.write_text("\n".join(lines) + "\n")
    floor = pe.node1_floor(str(path), 11.75, 1.0)
    assert floor["floor_gib"] == 56.75 and floor["rest_source"] == str(path)


def test_no_or_an_unusable_previous_sample_falls_back_to_arm_a(tmp_path):
    floor = pe.node1_floor("", 11.75, 1.0)
    assert floor["rest_gib"] == round(64.69 - 1.70 - 11.75, 2) and floor["rest_source"].startswith("default")
    assert floor["floor_gib"] == round(11.75 + floor["rest_gib"] + 1.0, 2)
    short = tmp_path / "short.jsonl"
    short.write_text(_sample(60, 0) + "\n")
    floor = pe.node1_floor(str(short), 0.0, 1.0)
    assert floor["rest_source"].startswith("default") and "unusable" in floor["rest_source"]


def test_the_extra_gib_raises_the_floor_and_is_recorded(tmp_path):
    """--extra-gib (both_cpu_ab.node1_extra_gib: the Engram scales on node 1) adds to the floor: 66 GiB free holds the
    default 63.99 GiB floor, not the 69.71 GiB one with the scales' 5.72."""
    floor = pe.node1_floor("", 11.75, 1.0, 5.72)
    assert floor["extra_gib"] == 5.72 and floor["floor_gib"] == round(11.75 + floor["rest_gib"] + 5.72 + 1.0, 2)
    assert pe.node1_floor("", 11.75, 1.0)["extra_gib"] == 0.0
    r, record = _main(tmp_path, 66.0)
    assert r.returncode == 0 and record["floor_gib"] == 63.99 and record["extra_gib"] == 0.0
    r, record = _main(tmp_path, 66.0, "--extra-gib", "5.72")
    assert r.returncode == 3 and record["floor_gib"] == 69.71 and "+ extra 5.72 +" in r.stdout


def test_evict_drops_each_file_and_counts_it(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.rows").write_bytes(b"0" * 8192)
    (tmp_path / "y.safetensors").write_bytes(b"1" * 4096)
    result = pe.evict([str(tmp_path), str(tmp_path / "missing")])
    assert result == {"files": 2, "bytes": 12288, "failed": 0}


def _nodes(tmp_path, free0_gib, free1_gib):
    root = tmp_path / "sysfs"
    for node, free in ((0, free0_gib), (1, free1_gib)):
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "meminfo").write_text(
            f"Node {node} MemFree:        {int(free * GIB)} kB\nNode {node} FilePages:      {GIB} kB\n"
            f"Node {node} Shmem:          0 kB\n"
        )
    return root


def _main(tmp_path, free1_gib, *args):
    evictable = tmp_path / "evictable"
    evictable.mkdir(exist_ok=True)
    (evictable / "rows").write_bytes(b"0" * 8192)
    env = dict(os.environ, PRELAUNCH_NODE_SYSFS=str(_nodes(tmp_path, 80.0, free1_gib)),
               PRELAUNCH_EVICT_PATHS=str(evictable))
    log = tmp_path / "prelaunch.json"
    r = subprocess.run([sys.executable, _PATH, "--log", str(log), "--wait-min", "0", *args], env=env,
                       capture_output=True, text=True, timeout=60)
    return r, json.loads(log.read_text())


def test_a_start_below_the_floor_evicts_once_then_refuses_with_the_holders(tmp_path):
    holders = tmp_path / "holders"
    (holders / "co-tenant").mkdir(parents=True)
    (holders / "co-tenant" / "w.bin").write_bytes(os.urandom(1 << 20))
    r, record = _main(tmp_path, 20.0, "--holders", str(holders))
    assert r.returncode == 3
    assert record["ready"] is False and record["rounds"][0]["evicted"]["files"] == 1
    assert "refusing to start: node 1 has 20.0 GiB free" in r.stderr
    assert record["page_cache_holders"]["holders"][0]["path"] == str(holders / "co-tenant")


def test_above_the_floor_it_is_ready_and_names_no_holders(tmp_path):
    r, record = _main(tmp_path, 80.0)
    assert r.returncode == 0 and record["ready"] is True and "page_cache_holders" not in record


def test_a_listening_port_skips_the_eviction_but_not_the_check(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        r, record = _main(tmp_path, 20.0, "--skip-evict-port", str(port))
    assert r.returncode == 3 and record["evict_skipped_for_ports"] == [port]
    assert record["rounds"][0]["evicted"]["files"] == 0


def test_the_holders_skip_files_this_user_may_not_write(tmp_path):
    # The kernel reports every page of such a file as resident (a 248 GiB root-owned file read as fully cached).
    if os.geteuid() == 0:
        import pytest

        pytest.skip("root may write every file")
    (tmp_path / "ours").mkdir()
    (tmp_path / "theirs").mkdir()
    (tmp_path / "ours" / "a.bin").write_bytes(os.urandom(1 << 20))
    theirs = tmp_path / "theirs" / "b.bin"
    theirs.write_bytes(os.urandom(1 << 20))
    theirs.chmod(0o444)
    holders = pe.page_cache_holders([str(tmp_path)], budget_s=10)
    assert [h["path"] for h in holders["holders"]] == [str(tmp_path / "ours")]


def test_cached_bytes_counts_a_just_written_file_and_less_once_it_is_dropped(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(os.urandom(1 << 20))
    assert pe.cached_bytes(str(path)) == 1 << 20
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    assert pe.cached_bytes(str(path)) < 1 << 20
