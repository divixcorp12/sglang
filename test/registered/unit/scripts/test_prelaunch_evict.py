"""prelaunch_evict.py: the node-1 floor from an arm's previous memsample, else the arm-A default, and the eviction
itself on a scratch file (CPU)."""

import importlib.util
import json
import os

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
