"""The expert read benches run on a fake checkpoint (buffered, CPU)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

import bench_expert_reads  # noqa: E402
import bench_split_vs_repack  # noqa: E402

from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402


def test_bench_reports_throughput(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=4)
    out = bench_expert_reads.run(str(tmp_path), reads=16, batch=4, direct=False)
    assert out["reads"] == 16 and out["batch"] == 4
    assert out["gb_per_s"] > 0 and out["ms_per_expert"] > 0
    assert out["file_gb_per_s"] >= out["gb_per_s"] > 0
    assert out["direct"] is False and out["queue_depth"] > 0
    assert out["bytes_basis"] == "row_payload"


def test_bench_reports_throughput_refuses_empty_runs(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=2)
    with pytest.raises(SystemExit, match="reads"):
        bench_expert_reads.run(str(tmp_path), reads=0, batch=1, direct=False)
    with pytest.raises(SystemExit, match="batch"):
        bench_expert_reads.run(str(tmp_path), reads=4, batch=0, direct=False)


def test_split_vs_repack_measures_three_modes_and_cleans_up(tmp_path):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=2, num_experts=5)
    rows = bench_split_vs_repack.run(
        str(ckpt), 1, 3, [1, 2], str(work), direct=False, max_read_gb=1.0, require_same_drive=False
    )
    assert [(r["mode"], r["batch"]) for r in rows] == [
        (mode, batch)
        for batch in (1, 2)
        for mode in ("superset_raw", "superset_split", "repacked_per_name")
    ]
    assert all(r["rows"] == 3 and r["ms_per_row"] > 0 and r["gb_per_s"] > 0 for r in rows)
    assert all(r["verified"] for r in rows if r["mode"] == "repacked_per_name")
    assert all(r["deterministic"] for r in rows if r["mode"] == "superset_split")
    assert not any("verified" in r for r in rows if r["mode"] != "repacked_per_name")
    assert all(0.0 <= r["split_share"] <= 1.0 for r in rows if r["mode"] == "superset_split")
    assert not work.exists()


def test_rows_record_provenance_and_one_byte_basis(tmp_path):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=2, num_experts=4)
    rows = bench_split_vs_repack.run(
        str(ckpt), 1, 2, [1], str(work), direct=False, max_read_gb=1.0, require_same_drive=False
    )
    for r in rows:
        assert r["layer"] == 1 and r["queue_depth"] > 0 and r["direct"] is False
        assert r["bytes_basis"] == "streamed_payload"
        assert r["file_bytes"] > 0 and r["file_gb_per_s"] > 0
    by_mode = {r["mode"]: r for r in rows}
    # One payload basis: the same bytes per row on every arm.
    assert by_mode["superset_raw"]["payload_bytes"] == by_mode["repacked_per_name"]["payload_bytes"]
    # A repacked copy holds exactly the streamed bytes; a superset read moves at least that.
    assert by_mode["repacked_per_name"]["file_bytes"] == by_mode["repacked_per_name"]["payload_bytes"]
    assert by_mode["superset_raw"]["file_bytes"] >= by_mode["superset_raw"]["payload_bytes"]
    # Buffered runs report every name as buffered.
    assert by_mode["repacked_per_name"]["buffered_names"] == sorted(
        bench_split_vs_repack.EXL3_STREAMED_NAMES
    )


def test_a_page_multiple_name_stays_direct_when_direct_is_asked():
    # buffered_names is the names whose repacked file cannot be read with O_DIRECT.
    class Source:
        def __init__(self, direct):
            self.direct = direct

    slab = torch.empty(8192 + 4096, dtype=torch.uint8)
    aligned = slab[(-slab.data_ptr()) % 4096 :][:4096]
    misaligned = slab[((-slab.data_ptr()) % 4096) + 1 :][:4096]
    names = bench_split_vs_repack._buffered_names(
        {"a": Source(True), "b": Source(False), "c": Source(True)},
        {"a": aligned, "b": aligned, "c": misaligned},
    )
    assert names == ["b", "c"]


def test_split_vs_repack_refuses_a_work_dir_off_the_shards_drive(tmp_path, monkeypatch):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=4)

    def device_of(path):
        return 7 if str(path).startswith(str(work)) or str(path) == str(tmp_path) else 9

    monkeypatch.setattr(bench_split_vs_repack, "_device_of", device_of)
    monkeypatch.setattr(bench_split_vs_repack, "_fs_type", lambda path: "ext4")
    with pytest.raises(SystemExit, match="not on the drive"):
        bench_split_vs_repack.run(str(ckpt), 0, 2, [1], str(work), direct=False)
    assert not work.exists()


@pytest.mark.parametrize("fs_type", ["tmpfs", "overlay", "ramfs"])
def test_split_vs_repack_refuses_a_ram_backed_work_dir(tmp_path, monkeypatch, fs_type):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=4)
    monkeypatch.setattr(bench_split_vs_repack, "_fs_type", lambda path: fs_type)
    with pytest.raises(SystemExit, match=fs_type):
        bench_split_vs_repack.run(str(ckpt), 0, 2, [1], str(work), direct=False)
    assert not work.exists()


def test_split_vs_repack_accepts_a_work_dir_on_the_shards_drive(tmp_path, monkeypatch):
    # Same device (both under tmp_path); only the filesystem type is pinned.
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=4)
    monkeypatch.setattr(bench_split_vs_repack, "_fs_type", lambda path: "ext4")
    rows = bench_split_vs_repack.run(str(ckpt), 0, 2, [1], str(work), direct=False, max_read_gb=1.0)
    assert len(rows) == 3 and not work.exists()


def test_a_preexisting_work_dir_is_kept(tmp_path):
    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    work.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=4)
    bench_split_vs_repack.run(
        str(ckpt), 0, 2, [1], str(work), direct=False, max_read_gb=1.0, require_same_drive=False
    )
    assert work.is_dir() and not list(work.iterdir())


def test_split_vs_repack_refuses_degenerate_runs(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=4)
    work = str(tmp_path / "w")
    for rows, batches, message in ((0, [1], "rows"), (2, [0], "batch"), (2, [3], "batch")):
        with pytest.raises(SystemExit, match=message):
            bench_split_vs_repack.run(
                str(tmp_path), 0, rows, batches, work, direct=False, require_same_drive=False
            )
    assert not (tmp_path / "w").exists()


def test_split_vs_repack_refuses_a_write_over_its_budget(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=4)
    with pytest.raises(SystemExit, match="max-write-gb"):
        bench_split_vs_repack.run(
            str(tmp_path), 0, 2, [1], str(tmp_path / "w"), direct=False,
            max_write_gb=1e-9, require_same_drive=False,
        )
    assert not (tmp_path / "w").exists()


def test_the_default_read_budget_is_the_plan_cap():
    import inspect

    default = inspect.signature(bench_split_vs_repack.run).parameters["max_read_gb"].default
    assert default == 3.5


def test_warmup_reads_are_in_the_planned_budget():
    base = bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 2])
    warm = bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 2], warmup_rows=1)
    assert warm - base == 2 * (2 * 100 + 90)  # per batch: two superset arms and one per-name arm


def test_split_vs_repack_refuses_a_read_over_its_budget(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=5)
    with pytest.raises(SystemExit, match="max-read-gb"):
        bench_split_vs_repack.run(
            str(tmp_path), 0, 5, [1], str(tmp_path / "w"), direct=False, max_read_gb=1e-6, require_same_drive=False
        )
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
