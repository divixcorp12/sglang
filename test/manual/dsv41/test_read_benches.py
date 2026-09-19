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


def test_default_run_fits_the_read_cap_and_gates_batch_8_on_full_submissions():
    bench = bench_split_vs_repack
    assert bench.GATE_BATCH == 8 and bench.DEFAULT_REPEATS == 5
    assert bench.DEFAULT_ROWS % bench.GATE_BATCH == 0  # batch 8 is whole 8-row submissions
    assert bench.GATE_BATCH in bench.DEFAULT_BATCHES
    planned = bench.planned_read_bytes(
        13_320_192, 13_315_584, bench.DEFAULT_ROWS, bench.DEFAULT_BATCHES,
        warmup_rows=1, repeats=bench.DEFAULT_REPEATS,
    )
    assert planned <= 3.5e9
    # Why the default batches are not 1/4/8: five repeats at the gate would not fit.
    assert bench.planned_read_bytes(
        13_320_192, 13_315_584, bench.DEFAULT_ROWS, [1, 4, 8], warmup_rows=1, repeats=5
    ) > 3.5e9


def test_repeats_are_counted_at_the_gate_batch_only():
    base = bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 8])
    five = bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 8], repeats=5)
    assert five - base == 4 * 4 * (100 + 90)  # 4 extra passes of split (buffer) and repacked (streamed)
    no_gate = bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 4], repeats=5)
    assert no_gate == bench_split_vs_repack.planned_read_bytes(100, 90, 4, [1, 4])


def test_the_gate_batch_alternates_split_and_repacked_passes(tmp_path, monkeypatch):
    bench = bench_split_vs_repack
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=8)
    events = []
    real_read = bench.Exl3ShardRowSource.read
    real_plans = bench.read_plans

    def read(self, rows, *args, **kwargs):
        events.append(("S", rows.numel()))
        return real_read(self, rows, *args, **kwargs)

    def plans(reader, plan_list):
        events.append(("R", int(plan_list[0].file_ids.numel())))
        return real_plans(reader, plan_list)

    monkeypatch.setattr(bench.Exl3ShardRowSource, "read", read)
    monkeypatch.setattr(bench, "read_plans", plans)
    bench.run(
        str(ckpt), 0, 4, [2], str(tmp_path / "work"), direct=False, max_read_gb=1.0,
        require_same_drive=False, repeats=3, gate_batch=2,
    )
    # The repack pass's read of the sample, one warm read per arm, then three
    # (split over 4 rows, repacked as two 2-row submissions) pairs in turn.
    assert events == [("S", 4), ("S", 1), ("R", 1)] + [("S", 4), ("R", 2), ("R", 2)] * 3


def test_the_gate_batch_reports_all_samples_median_min_and_the_ratio(tmp_path):
    bench = bench_split_vs_repack
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=8)
    rows = bench.run(
        str(ckpt), 0, 4, [1, 2], str(tmp_path / "work"), direct=False, max_read_gb=1.0,
        require_same_drive=False, repeats=3, gate_batch=2,
    )
    by = {(r["mode"], r["batch"]): r for r in rows}
    for mode in ("superset_split", "repacked_per_name"):
        gate, other = by[(mode, 2)], by[(mode, 1)]
        assert gate["repeats"] == 3 and len(gate["samples_ms_per_row"]) == 3
        assert other["repeats"] == 1 and len(other["samples_ms_per_row"]) == 1
        samples = sorted(gate["samples_ms_per_row"])
        assert gate["median_ms_per_row"] == samples[1] and gate["min_ms_per_row"] == samples[0]
        assert gate["ms_per_row"] == gate["median_ms_per_row"]
    assert by[("superset_raw", 2)]["repeats"] == 1
    repacked, split = by[("repacked_per_name", 2)], by[("superset_split", 2)]
    assert repacked["ratio_median"] == pytest.approx(
        repacked["median_ms_per_row"] / split["median_ms_per_row"]
    )
    assert split["deterministic"] and repacked["verified"]


def test_repeats_that_overrun_the_read_budget_fail_fast(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=8)
    with pytest.raises(SystemExit, match="repeats"):
        bench_split_vs_repack.run(
            str(tmp_path), 0, 8, [8], str(tmp_path / "w"), direct=False,
            max_read_gb=1.0, require_same_drive=False, repeats=10**9,
        )
    with pytest.raises(SystemExit, match="repeats"):
        bench_split_vs_repack.run(
            str(tmp_path), 0, 4, [1], str(tmp_path / "w"), direct=False,
            require_same_drive=False, repeats=0,
        )
    assert not (tmp_path / "w").exists()


def test_the_repacked_copy_is_sparse_at_expert_id_offsets(tmp_path):
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout

    ckpt, work = tmp_path / "ckpt", tmp_path / "work"
    ckpt.mkdir()
    work.mkdir()
    write_fake_exl3(str(ckpt), num_layers=1, num_experts=6)
    layout = build_exl3_expert_layout(str(ckpt))
    specs = Exl3ExpertFormat(layout, 0, direct=False).tensor_specs(None)
    experts = [1, 4]
    truth = {}
    for spec in specs:
        pattern = (torch.arange(2 * spec.row_bytes, dtype=torch.int64) % 251 + 1).to(torch.uint8)
        truth[spec.name] = pattern.view(spec.dtype).view((2,) + spec.row_shape)
    paths = bench_split_vs_repack._write_repacked(str(work), 0, specs, truth, experts, layout.num_experts)
    for spec in specs:
        with open(paths[spec.name], "rb") as f:
            data = f.read()
        assert len(data) == layout.num_experts * spec.row_bytes
        raw = truth[spec.name].view(torch.uint8).reshape(2, -1)
        for i, expert in enumerate(experts):
            at = expert * spec.row_bytes
            assert data[at : at + spec.row_bytes] == raw[i].numpy().tobytes()
        holes = b"".join(
            data[e * spec.row_bytes : (e + 1) * spec.row_bytes]
            for e in range(layout.num_experts)
            if e not in experts
        )
        assert holes == bytes(len(holes))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
