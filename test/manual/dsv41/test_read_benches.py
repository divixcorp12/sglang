"""The expert read benches run on a fake checkpoint (buffered, CPU)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

import bench_expert_reads  # noqa: E402
import bench_split_vs_repack  # noqa: E402

from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402


def test_bench_reports_throughput(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=4)
    out = bench_expert_reads.run(str(tmp_path), reads=16, batch=4, direct=False)
    assert out["reads"] == 16 and out["batch"] == 4
    assert out["gb_per_s"] > 0 and out["ms_per_expert"] > 0


def test_split_vs_repack_measures_three_modes_and_cleans_up(tmp_path):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=2, num_experts=5)
    rows = bench_split_vs_repack.run(str(ckpt), 1, 3, [1, 2], str(work), direct=False, max_read_gb=1.0)
    assert [(r["mode"], r["batch"]) for r in rows] == [
        (mode, batch)
        for batch in (1, 2)
        for mode in ("superset_raw", "superset_split", "repacked_per_name")
    ]
    assert all(r["rows"] == 3 and r["ms_per_row"] > 0 and r["gb_per_s"] > 0 for r in rows)
    assert all(r["verified"] for r in rows if r["mode"] != "superset_raw")
    assert all(0.0 <= r["split_share"] <= 1.0 for r in rows if r["mode"] == "superset_split")
    assert not work.exists()


def test_split_vs_repack_refuses_a_read_over_its_budget(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=5)
    with pytest.raises(SystemExit, match="max-read-gb"):
        bench_split_vs_repack.run(str(tmp_path), 0, 5, [1], str(tmp_path / "w"), direct=False, max_read_gb=1e-6)
    with pytest.raises(SystemExit, match="only 5 experts"):
        bench_split_vs_repack.run(str(tmp_path), 0, 6, [1], str(tmp_path / "w"), direct=False)
    assert not (tmp_path / "w").exists()


def test_planned_reads_count_both_copies():
    # 12 rows, batches 1/4/8: 7 superset passes of the shards, 3 per-name passes of the copy.
    planned = bench_split_vs_repack.planned_read_bytes(13_320_192, 13_315_584, 12, [1, 4, 8])
    assert planned == 12 * (13_320_192 * 7 + 13_315_584 * 3)
    assert 3.1e9 < 2 * planned < 3.3e9  # queue depths 32 and 128: about 3.2 GB


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
