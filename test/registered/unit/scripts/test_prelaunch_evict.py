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


def test_layer_major_state_store_raises_the_default_floor_and_the_log_names_it(tmp_path):
    # Arm A's default rest predates layer-major's 5 GiB node-1 StateStore: added to it, and named on the floor's line.
    floor = pe.node1_floor("", 12.67, 1.0, 5.0)
    assert floor["floor_gib"] == round(12.67 + 51.24 + 5.0 + 1.0, 2) and floor["layer_major_state_gib"] == 5.0
    # 66.0 GiB free passes 64.91 without the StateStore and is short of 69.91 with it.
    r, record = _main(tmp_path, 66.0, "--hicache-gib", "12.67", "--layer-major-state-gib", "5.0")
    assert r.returncode == 3 and record["floor_gib"] == 69.91
    assert "node-1 floor 69.91 GiB = HiCache 12.67 + rest 51.24 (default (arm A, 2026-10-15)) " \
           "+ layer-major StateStore 5.0 + margin 1.0" in r.stdout
    (tmp_path / "off").mkdir()
    r, record = _main(tmp_path / "off", 66.0, "--hicache-gib", "12.67")
    assert r.returncode == 0 and record["layer_major_state_gib"] == 0.0 and "StateStore" not in r.stdout


def test_a_previous_memsample_already_holds_the_state_store(tmp_path):
    lines = [_sample(60, 0.4), _sample(4, 12.4)]
    path = tmp_path / "prev.memsample.jsonl"
    path.write_text("\n".join(lines) + "\n")
    floor = pe.node1_floor(str(path), 11.75, 1.0, 5.0)
    assert floor["layer_major_state_gib"] == 0.0 and floor["floor_gib"] == 56.75


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


def test_a_retry_below_the_floor_says_so_and_names_the_holders_once(tmp_path, monkeypatch, capsys):
    holders = tmp_path / "holders"
    (holders / "co-tenant").mkdir(parents=True)
    (holders / "co-tenant" / "w.bin").write_bytes(os.urandom(1 << 20))
    evictable = tmp_path / "evictable"
    evictable.mkdir()
    free1 = iter([20.0, 20.0, 20.0, 80.0])  # the first read, two rounds below the floor, then ready
    meminfo = lambda: {n: {"MemFree": 80.0, "FilePages": 1.0, "SReclaimable": 0.0} for n in "01"}
    monkeypatch.setattr(pe, "node_meminfo",
                        lambda: {**meminfo(), "1": {"MemFree": next(free1), "FilePages": 1.0, "SReclaimable": 0.0}})
    monkeypatch.setattr(pe, "evict", lambda: {"files": 0, "bytes": 0, "failed": 0})
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    log = tmp_path / "prelaunch.json"
    monkeypatch.setattr(sys, "argv", ["prelaunch_evict.py", "--log", str(log), "--wait-min", "5", "--holders", str(holders)])
    assert pe.main() == 0
    out = capsys.readouterr().out
    assert out.count("evicting again in 60 s") == 2
    assert out.count("largest page-cache holders") == 1 and str(holders / "co-tenant") in out
    record = json.loads(log.read_text())
    assert record["ready"] is True and record["page_cache_holders_first"]["holders"][0]["path"] == str(holders / "co-tenant")


def _sysfs(tmp_path, free1, slab1, file1=1.0):
    root = tmp_path / "sysfs"
    for node, free, slab in ((0, 80.0, 0.0), (1, free1, slab1)):
        (root / f"node{node}").mkdir(parents=True, exist_ok=True)
        (root / f"node{node}" / "meminfo").write_text(
            f"Node {node} MemFree:        {int(free * GIB)} kB\nNode {node} FilePages:      {int(file1 * GIB)} kB\n"
            f"Node {node} Shmem:          0 kB\nNode {node} SReclaimable:   {int(slab * GIB)} kB\n"
        )
    return root


def _run_main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(sys, "argv", ["prelaunch_evict.py", "--log", str(tmp_path / "log.json"), "--wait-min", "0",
                                      "--hicache-gib", "12.67", *args])
    monkeypatch.setattr(pe, "evict", lambda: {"files": 0, "bytes": 0, "failed": 0})
    return pe.main()


def test_reclaimable_slab_counts_toward_the_floor(tmp_path, monkeypatch, capsys):
    # 61 GiB free alone is below the 64.91 GiB floor; with 10.7 GiB of reclaimable slab it is above it.
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_sysfs(tmp_path, 61.0, 10.7)))
    assert _run_main(monkeypatch, tmp_path) == 0
    assert json.loads((tmp_path / "log.json").read_text())["ready"] is True


def test_the_refusal_shows_the_split_and_which_part_is_short(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_sysfs(tmp_path, 20.0, 5.0, file1=50.0)))
    assert _run_main(monkeypatch, tmp_path) == 3
    err = capsys.readouterr().err
    assert "refusing to start: node 1 has 20.0 GiB free + 5.0 GiB reclaimable slab = 25.0 available" in err
    assert "short by 39.91 GiB" in err and "enough to cover it if evicted" in err


def test_a_retry_line_shows_free_plus_slab_equals_available_against_the_floor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_sysfs(tmp_path, 20.0, 5.0)))
    monkeypatch.setattr(pe.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("slept")))
    monkeypatch.setattr(sys, "argv", ["prelaunch_evict.py", "--log", str(tmp_path / "log.json"), "--wait-min", "5",
                                      "--hicache-gib", "12.67"])
    monkeypatch.setattr(pe, "evict", lambda: {"files": 0, "bytes": 0, "failed": 0})
    try:
        pe.main()
    except RuntimeError:
        pass
    assert "node 1 free 20.0 + reclaimable slab 5.0 = available 25.0 GiB against the floor 64.91 GiB" in capsys.readouterr().out


def _holder_tree(tmp_path):
    root = tmp_path / "root"
    big, small = root / "big", root / "small"
    big.mkdir(parents=True)
    small.mkdir()
    (big / "a.bin").write_bytes(b"0" * 4096)
    (big / "b.bin").write_bytes(b"0" * 4096)
    (small / "c.bin").write_bytes(b"0" * 4096)
    return root, big, small


def test_evict_holders_drops_only_holders_above_the_threshold_under_the_roots(tmp_path, monkeypatch):
    root, big, small = _holder_tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "d.bin").write_bytes(b"0" * 4096)
    seen = []
    monkeypatch.setattr(pe.os, "posix_fadvise", lambda fd, off, n, adv: seen.append(os.readlink(f"/proc/self/fd/{fd}")))
    holders = [{"path": str(big), "gib": 1.0}, {"path": str(small), "gib": 0.1}, {"path": str(outside), "gib": 2.0}]
    done = pe.evict_holders(holders, [str(root)], 256.0)
    assert done == [{"path": str(big), "gib": 1.0, "files": 2}]
    assert sorted(seen) == [str(big / "a.bin"), str(big / "b.bin")]


def test_evict_holders_skips_files_open_for_writing_and_the_production_checkout(tmp_path, monkeypatch):
    root, big, _ = _holder_tree(tmp_path)
    prod = root / "cc-expert-prediction" / "dsv41-direct-prod"
    prod.mkdir(parents=True)
    (prod / "p.bin").write_bytes(b"0" * 4096)
    seen = []
    monkeypatch.setattr(pe.os, "posix_fadvise", lambda fd, off, n, adv: seen.append(os.readlink(f"/proc/self/fd/{fd}")))
    with open(big / "a.bin", "ab"):
        done = pe.evict_holders([{"path": str(big), "gib": 1.0}, {"path": str(prod), "gib": 1.0}], [str(root)], 256.0)
    assert seen == [str(big / "b.bin")] and [d["path"] for d in done] == [str(big)]


def test_main_evicts_holders_only_when_asked(tmp_path, monkeypatch, capsys):
    root, big, _ = _holder_tree(tmp_path)
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_sysfs(tmp_path, 20.0, 0.0)))
    monkeypatch.setattr(pe.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("slept")))
    monkeypatch.setattr(pe, "page_cache_holders", lambda roots, budget: {"complete": True, "holders": [{"path": str(big), "gib": 1.0}]})
    for flag, expected in (([], 0), (["--evict-holders"], 2)):
        seen = []
        monkeypatch.setattr(pe.os, "posix_fadvise", lambda fd, off, n, adv: seen.append(fd))
        monkeypatch.setattr(sys, "argv", ["prelaunch_evict.py", "--log", str(tmp_path / "log.json"), "--wait-min", "0.05",
                                          "--hicache-gib", "12.67", "--holders", str(root), *flag])
        monkeypatch.setattr(pe, "evict", lambda: {"files": 0, "bytes": 0, "failed": 0})
        monkeypatch.setattr(pe.time, "time", iter_time())
        try:
            pe.main()
        except RuntimeError:
            pass
        assert len(seen) == expected, flag


def iter_time():
    """A clock that advances 1 s per read, so a short --wait-min expires after a few rounds."""
    state = {"t": 1000.0}

    def now():
        state["t"] += 1.0
        return state["t"]

    return now


def _two_nodes(tmp_path, free0, slab0, free1, slab1, file0=2.0):
    root = tmp_path / "sysfs"
    for node, free, slab, file in ((0, free0, slab0, file0), (1, free1, slab1, 1.0)):
        (root / f"node{node}").mkdir(parents=True, exist_ok=True)
        (root / f"node{node}" / "meminfo").write_text(
            f"Node {node} MemFree:        {int(free * GIB)} kB\nNode {node} FilePages:      {int(file * GIB)} kB\n"
            f"Node {node} Shmem:          0 kB\nNode {node} SReclaimable:   {int(slab * GIB)} kB\n"
        )
    return root


def test_both_nodes_above_their_floors_passes_and_shows_node_0s_file_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_two_nodes(tmp_path, 85.0, 0.0, 80.0, 0.0, file0=3.5)))
    assert _run_main(monkeypatch, tmp_path, "--node0-floor-gib", "80.6") == 0
    out = capsys.readouterr().out
    assert "node 0 free 85.0 + slab 0.0 = 85.0 File 3.5 GiB" in out and "node-0 floor 80.6" in out


def test_node_0_short_refuses_naming_node_0_and_slab_counts_there(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_two_nodes(tmp_path, 70.0, 5.0, 80.0, 0.0)))
    assert _run_main(monkeypatch, tmp_path, "--node0-floor-gib", "80.6") == 3
    err = capsys.readouterr().err
    assert "refusing to start: node 0 has 70.0 GiB free + 5.0 GiB reclaimable slab = 75.0 available" in err
    assert "below its floor of 80.6 GiB, short by 5.6 GiB" in err
    assert "node 1 has 80.0 GiB free" in err and "refusing to start: node 1" not in err
    # slab alone closes the gap on node 0 as it does on node 1
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_two_nodes(tmp_path, 70.0, 11.0, 80.0, 0.0)))
    assert _run_main(monkeypatch, tmp_path, "--node0-floor-gib", "80.6") == 0


def test_node_0_check_is_off_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_two_nodes(tmp_path, 1.0, 0.0, 80.0, 0.0)))
    assert _run_main(monkeypatch, tmp_path) == 0


def test_a_node_0_retry_names_node_0_and_scans_the_holders(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "NODE_SYSFS", str(_two_nodes(tmp_path, 70.0, 5.0, 80.0, 0.0)))
    monkeypatch.setattr(pe.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("slept")))
    monkeypatch.setattr(pe, "page_cache_holders", lambda roots, budget: {"complete": True, "holders": [{"path": "/x/y", "gib": 9.0}]})
    monkeypatch.setattr(sys, "argv", ["prelaunch_evict.py", "--log", str(tmp_path / "log.json"), "--wait-min", "5",
                                      "--hicache-gib", "12.67", "--node0-floor-gib", "80.6", "--holders", str(tmp_path)])
    monkeypatch.setattr(pe, "evict", lambda: {"files": 0, "bytes": 0, "failed": 0})
    try:
        pe.main()
    except RuntimeError:
        pass
    out = capsys.readouterr().out
    assert "node 0 free 70.0 + reclaimable slab 5.0 = available 75.0 GiB against the floor 80.6 GiB" in out
    assert "against the floor 64.91" not in out and "/x/y" in out
