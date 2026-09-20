"""Synthetic proof that ``bench_mirror_rows.py`` measures what it says it does.

Everything runs against a few-MB fake checkpoint and two copies of it under
``tmp_path`` (O_DIRECT needs a real filesystem: run with ``--basetemp`` on the
disk, not on tmpfs). No test reads or writes the real drives.

    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 0-63 \\
        python -m pytest analysis/dsv41-drive/test_bench_mirror_rows.py -p no:cacheprovider
"""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bench_mirror_rows", Path(__file__).with_name("bench_mirror_rows.py")
)
bench = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_mirror_rows"] = bench
_SPEC.loader.exec_module(bench)  # puts this worktree's python/ on sys.path

from sglang.srt.layers.moe.expert_row_source import RowReadStats  # noqa: E402
from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402

NUM_EXPERTS = 12


def _make(tmp_path, experts_per_shard=NUM_EXPERTS, num_roots=2):
    """A fake checkpoint (one shard per layer by default) and identical copies."""
    source = tmp_path / "source" / "ckpt"
    source.mkdir(parents=True)
    write_fake_exl3(
        str(source),
        num_layers=2,
        num_experts=NUM_EXPERTS,
        experts_per_shard=experts_per_shard,
        hidden=512,
        inter=512,
    )
    roots = []
    for i in range(num_roots):
        root = tmp_path / f"drive{i}" / "copy"
        shutil.copytree(source, root)
        roots.append(str(root))
    return str(source), roots


def _args(source, roots, out, *extra, layers=("0",), reps="3", rows="8"):
    return [
        "--source", source,
        "--roots", *roots,
        "--layers", *layers,
        "--reps", reps,
        "--rows", rows,
        "--output", str(out),
        *extra,
    ]  # fmt: skip


def test_four_arms_first_reads_apart_and_bytes_agree(tmp_path, capsys):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    assert bench.main(_args(source, roots, out)) == 0
    report = json.loads(out.read_text())
    arms = report["results"]
    assert list(arms) == [
        "source baseline",
        "drive0 only",
        "drive1 only",
        "mirrored 1:1",
    ]
    for arm in arms.values():
        # 3 reps x 8 rows in one shard: the first read of the arm is set apart.
        assert arm["n"] == 3 * 8 - 1
        assert len(arm["first_reads"]) == 1
        assert len(arm["rep_p50_ms"]) == 3
        assert 0 < arm["p50_ms"] <= arm["p90_ms"] <= arm["p99_ms"] <= arm["max_ms"]
        assert arm["mb_per_s"] > 0
    # Same shard, same rows: every arm moves the same bytes per row.
    assert len({a["bytes_per_row"] for a in arms.values()}) == 1
    assert report["bytes_match"] is True
    assert len(report["samples_ns"]["mirrored 1:1"]) == 3 * 8
    assert report["layers"][0]["shards"] == ["model-00001.safetensors"]
    table = capsys.readouterr().out
    for needle in ("p50", "p90", "p99", "MB/s", "per-rep", "first read", "GATE"):
        assert needle in table


def test_every_read_is_one_row_and_direct(tmp_path, monkeypatch):
    source, roots = _make(tmp_path)
    calls = []
    original = bench.Exl3ShardRowSource.read

    def spy(self, rows, destinations, destination_rows=None):
        calls.append(rows.numel())
        assert self.reader._direct is True
        return original(self, rows, destinations, destination_rows)

    monkeypatch.setattr(bench.Exl3ShardRowSource, "read", spy)
    assert bench.main(_args(source, roots, tmp_path / "r.json")) == 0
    # 4 arms x 3 reps x 8 experts, each timed as its own call.
    assert calls == [1] * (4 * 3 * 8)


def test_same_experts_in_every_arm_and_distinct_within_a_rep(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    assert bench.main(_args(source, roots, out, rows="10")) == 0
    samples = json.loads(out.read_text())["samples_ns"]
    per_arm = {
        name: [[s["expert"] for s in rows if s["rep"] == rep] for rep in range(3)]
        for name, rows in samples.items()
    }
    reference = next(iter(per_arm.values()))
    for experts in per_arm.values():
        assert experts == reference
    assert all(len(set(rep)) == 10 for rep in reference)
    import random

    rng = random.Random(1234)
    assert reference == [rng.sample(range(NUM_EXPERTS), 10) for _ in range(3)]


def test_arm_order_rotates_so_every_arm_takes_every_position():
    arms = [bench.Arm(n, "single") for n in "abcd"]
    orders = [[a.name for a in bench.arm_order(arms, b)] for b in range(4)]
    assert orders[0] == list("abcd") and orders[1] == list("bcda")
    for position in range(4):
        assert {order[position] for order in orders} == set("abcd")


def test_poisoned_mirror_is_reported_not_timed_as_a_win(tmp_path, capsys):
    source, roots = _make(tmp_path)
    for name in os.listdir(roots[1]):
        if name.endswith(".safetensors"):
            path = os.path.join(roots[1], name)
            data = Path(path).read_bytes()
            Path(path).write_bytes(bytes(b ^ 0xFF for b in data))
    out = tmp_path / "r.json"
    assert bench.main(_args(source, roots, out)) == 2
    report = json.loads(out.read_text())
    assert report["bytes_match"] is False
    bad = {m["arm"] for m in report["mismatches"]}
    assert bad == {"drive1 only", "mirrored 1:1"}  # the clean copy still agrees
    assert "BYTE MISMATCH" in capsys.readouterr().out


def test_a_reader_that_transfers_nothing_cannot_pass(tmp_path, monkeypatch):
    source, roots = _make(tmp_path)
    # A read that returns instantly and writes no destination. One rep: the
    # baseline runs first and fills the buffers with these very experts, so
    # unless the buffers are zeroed before each arm the mirror arms would
    # "return" the baseline's bytes and pass.
    monkeypatch.setattr(
        bench.Exl3MirrorRowSource,
        "read",
        lambda self, rows, destinations, destination_rows=None: RowReadStats(),
    )
    out = tmp_path / "r.json"
    assert bench.main(_args(source, roots, out, reps="1")) == 2
    bad = {m["arm"] for m in json.loads(out.read_text())["mismatches"]}
    assert bad == {"drive0 only", "drive1 only", "mirrored 1:1"}


def test_layer_spanning_shards_is_refused_unless_allowed(tmp_path):
    source, roots = _make(tmp_path, experts_per_shard=5)
    out = tmp_path / "r.json"
    with pytest.raises(SystemExit, match="span 3 shard"):
        bench.main(_args(source, roots, out))
    assert not out.exists()
    assert bench.main(_args(source, roots, out, "--allow-multi-shard", rows="12")) == 0
    arm = json.loads(out.read_text())["results"]["mirrored 1:1"]
    assert len(arm["first_reads"]) == 3  # one open per shard, none counted
    assert arm["n"] == 3 * 12 - 3


def test_zero_weight_and_arm_selection_for_the_contention_test(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    args = _args(source, roots, out, "--weights", "1:0", "--arms", "single", "mirrored")
    assert bench.main(args) == 0
    report = json.loads(out.read_text())
    assert list(report["results"]) == ["drive0 only", "drive1 only", "mirrored 1:0"]
    assert report["weights"] == [1.0, 0.0]
    assert report["bytes_match"] is True


def test_two_layers_are_sampled_per_rep_and_first_read_is_per_shard(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    assert bench.main(_args(source, roots, out, layers=("0", "1"))) == 0
    arm = json.loads(out.read_text())["results"]["drive0 only"]
    assert len(arm["first_reads"]) == 2  # layer 0 and layer 1 are different shards
    assert arm["n"] == 2 * 3 * 8 - 2


def test_bad_configuration_is_refused_before_any_read(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    with pytest.raises(SystemExit, match="duplicates"):
        bench.main(_args(source, [roots[0], roots[0]], out))
    with pytest.raises(SystemExit, match="1 entries for 2 roots"):
        bench.main(_args(source, roots, out, "--weights", "1"))
    with pytest.raises(SystemExit, match="source checkpoint itself"):
        bench.main(_args(source, [roots[0], source], out))
    with pytest.raises(SystemExit, match="outside"):
        bench.main(_args(source, roots, out, layers=("7",)))
    with pytest.raises(SystemExit, match="exceeds"):
        bench.main(_args(source, roots, out, rows="99"))
    assert not out.exists()


def test_statistics_on_known_values():
    def sample(rep, ms, first=False):
        return bench.Sample(
            "a", 0, rep, 0, 0, int(ms * 1e6), 1_000_000, 0, 0, "s", first
        )

    samples = [sample(0, 999, first=True)] + [
        sample(rep, ms) for rep, ms in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    ]
    s = bench.summarize_arm(samples, reps=3)
    assert s["n"] == 5 and s["first_reads"][0]["ms"] == 999
    assert s["p50_ms"] == 3 and s["mean_ms"] == 3 and s["max_ms"] == 5
    assert s["p90_ms"] == pytest.approx(4.6) and s["p99_ms"] == pytest.approx(4.96)
    assert s["rep_p50_ms"] == [1.5, 3.5, 5]
    assert s["mb_per_s"] == pytest.approx(5 * 1.0 / (15e-3))
    # Rep medians falling to under 85% of rep 0's is what a warming cache looks like.
    assert bench.cache_suspects({"a": {"rep_p50_ms": [10, 9, 5]}}, 3) == ["a"]
    assert bench.cache_suspects({"a": {"rep_p50_ms": [10, 9.5, 9]}}, 3) == []
