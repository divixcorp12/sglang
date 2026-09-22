"""Synthetic proof that ``bench_row_scheduling.py`` replays one fixed request
sequence for every arm and accounts rows, extents and bytes correctly.

Runs on a few-MB fake checkpoint and copies of it under ``tmp_path`` (O_DIRECT
needs a real filesystem: pass ``--basetemp`` on the disk, not tmpfs). No test
touches the real drives.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 taskset -c 0-63 \\
        python -m pytest analysis/dsv41-drive/test_bench_row_scheduling.py -p no:cacheprovider
"""

import importlib.util
import os
import json
import shutil
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bench_row_scheduling", Path(__file__).with_name("bench_row_scheduling.py")
)
sched = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_row_scheduling"] = sched
_SPEC.loader.exec_module(sched)

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout  # noqa: E402
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy  # noqa: E402
from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402

NUM_EXPERTS = 12
SIZES = ("1", "2", "4", "6")


def _make(tmp_path, num_roots=2):
    source = tmp_path / "source" / "ckpt"
    source.mkdir(parents=True)
    write_fake_exl3(
        str(source), num_layers=2, num_experts=NUM_EXPERTS,
        experts_per_shard=NUM_EXPERTS, hidden=512, inter=512,
    )  # fmt: skip
    roots = []
    for i in range(num_roots):
        root = tmp_path / f"drive{i}" / "copy"
        shutil.copytree(source, root)
        roots.append(str(root))
    return str(source), roots


def _args(source, roots, out, *extra, sizes=SIZES, rows="8", reps="2"):
    return [
        "--source", source, "--roots", *roots, "--layers", "0",
        "--sizes", *sizes, "--rows-per-size", rows, "--reps", reps,
        "--warmup-batches", "1", "--calib-batches", "1", "--verify-rows", "6",
        "--weights", "3:1", "--output", str(out), *extra,
    ]  # fmt: skip


def _run(tmp_path, *extra, **kw):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    code = sched.main(_args(source, roots, out, *extra, **kw))
    return code, json.loads(out.read_text()), source, roots


# ------------------------------------------------------------------- replay


def test_replay_is_deterministic_and_batches_are_distinct_single_layer():
    a = sched.build_replay(NUM_EXPERTS, [0, 1], [1, 2, 4], 8, seed=7)
    b = sched.build_replay(NUM_EXPERTS, [0, 1], [1, 2, 4], 8, seed=7)
    c = sched.build_replay(NUM_EXPERTS, [0, 1], [1, 2, 4], 8, seed=8)
    assert a == b and sched.replay_digest(a) == sched.replay_digest(b)
    assert sched.replay_digest(a) != sched.replay_digest(c)
    assert [x.size for x in a] == [1] * 8 + [2] * 4 + [4] * 2
    for batch in a:
        assert len({r.expert for r in batch.rows}) == batch.size
        assert len({r.layer for r in batch.rows}) == 1
    assert [x.index for x in a] == list(range(len(a)))


def test_replay_never_has_fewer_than_two_batches_per_size():
    replay = sched.build_replay(NUM_EXPERTS, [0], [8], 8, seed=1)
    assert len(replay) == 2


def test_replay_does_not_depend_on_the_arms(tmp_path):
    code, report, *_ = _run(tmp_path)
    assert code == 0
    again = sched.build_replay(NUM_EXPERTS, [0], [1, 2, 4, 6], 8, seed=1234)
    assert report["replay"]["digest"] == sched.replay_digest(again)
    solo, *_ = _run(tmp_path / "solo", "--arms", "whole_row")
    assert solo == 0


# ---------------------------------------------------------------- extents


def _geoms(tmp_path, experts):
    source, _ = _make(tmp_path)
    layout = build_exl3_expert_layout(source)
    rows = [sched.RowRequest(0, e) for e in experts]
    return layout, [sched.row_geometry(layout, r) for r in rows]


def test_within_row_is_two_extents_per_row_and_one_root_is_one(tmp_path):
    _, geoms = _geoms(tmp_path, [3, 5, 7])
    both = sched.SplitPlanner((1.0, 1.0)).plan(geoms)
    assert len(both) == 2 * 3
    assert {e.row for e in both} == {0, 1, 2}
    for row, g in enumerate(geoms):
        mine = [e for e in both if e.row == row]
        assert sorted(e.root for e in mine) == [0, 1]
        assert sum(e.length for e in mine) == g.length
        first = min(mine, key=lambda e: e.dest_offset)
        assert first.dest_offset == 0 and first.offset == g.offset
    only = sched.SplitPlanner((0.0, 1.0)).plan(geoms)
    assert len(only) == 3 and {e.root for e in only} == {1}
    assert [e.length for e in only] == [g.length for g in geoms]


def test_split_plan_matches_what_production_read_split_submits(tmp_path):
    source, roots = _make(tmp_path)
    layout = build_exl3_expert_layout(source)
    reader = sched.shared_row_reader(layout, True, source)
    seen = []
    original = reader._submit

    def spy(file_ids, offsets, destinations, lengths, expected):
        seen.append((list(offsets), list(destinations), list(lengths), expected))
        return original(file_ids, offsets, destinations, lengths, expected)

    reader._submit = spy
    harness = sched.Harness.build(layout, source, roots, [0], 4)
    bases = [harness.bounce[i].data_ptr() for i in range(4)]
    experts = [2, 9, 4, 0]
    rows = [sched.RowRequest(0, e) for e in experts]
    geoms = [sched.row_geometry(layout, r) for r in rows]
    for weights in ((1.0, 1.0), (3.0, 1.0), (1.0, 0.0)):
        seen.clear()
        reader.read_split(
            [(0, e) for e in experts], bases, roots=roots, policy=StaticSplitPolicy(weights)
        )
        prod_offsets, prod_dests, prod_lengths, prod_expected = seen[0]
        ours = sched.SplitPlanner(weights).plan(geoms)
        assert [e.offset for e in ours] == prod_offsets
        assert [bases[e.row] + e.dest_offset for e in ours] == prod_dests
        assert [e.length for e in ours] == prod_lengths
        seen.clear()
        assert harness.submit(geoms, ours) == prod_expected
        assert seen[0][0] == prod_offsets and seen[0][1] == prod_dests


def test_whole_row_reads_each_row_once_from_one_root_and_balances_a_batch(tmp_path):
    _, geoms = _geoms(tmp_path, list(range(8)))
    planner = sched.WholeRowPlanner(2)
    extents = planner.plan(geoms)
    assert [e.row for e in extents] == list(range(8))
    assert all(e.dest_offset == 0 for e in extents)
    assert [e.length for e in extents] == [g.length for g in geoms]
    per_root = [sum(e.length for e in extents if e.root == r) for r in range(2)]
    assert abs(per_root[0] - per_root[1]) <= max(g.length for g in geoms)
    assert {e.root for e in extents} == {0, 1}


def test_whole_row_state_makes_single_row_batches_alternate_roots(tmp_path):
    _, geoms = _geoms(tmp_path, [1, 2, 3, 4])
    planner = sched.WholeRowPlanner(2)
    roots = [planner.plan([g])[0].root for g in geoms]
    assert roots == [0, 1, 0, 1]
    planner.reset()
    assert [planner.plan([g])[0].root for g in geoms] == roots  # reset replays it
    stateless = [sched.WholeRowPlanner(2).plan([g])[0].root for g in geoms]
    assert stateless == [0, 0, 0, 0]  # without carried state a batch of 1 is one-root


def test_whole_row_assign_uses_outstanding_bytes_then_history():
    planner = sched.WholeRowPlanner(3)
    assert planner.assign([10, 10, 10]) == [0, 1, 2]
    assert planner.assign([10]) == [0]
    planner.served = [30, 10, 20]
    assert planner.assign([5, 5, 5, 5]) == [1, 2, 0, 1]


# ------------------------------------------------------------ whole run


def test_every_arm_requests_identical_useful_and_aligned_bytes(tmp_path):
    code, report, *_ = _run(tmp_path)
    assert code == 0
    assert report["accounting_errors"] == []
    assert [a["name"] for a in report["arms"]] == [
        "within-row 1:1", "whole-row", "one-root drive0", "weighted 3:1",
    ]  # fmt: skip
    plans = {a["name"]: a for a in report["arms"]}
    assert len({a["useful_bytes"] for a in plans.values()}) == 1
    assert len({a["requested_bytes"] for a in plans.values()}) == 1
    assert len({a["rows"] for a in plans.values()}) == 1
    for name, per_size in report["results"].items():
        for size, r in per_size.items():
            assert r["app_rows_per_batch"] == int(size)
            useful = {
                report["results"][n][size]["useful_bytes_per_batch"]
                for n in report["results"]
            }
            req = {
                report["results"][n][size]["requested_bytes_per_batch"]
                for n in report["results"]
            }
            assert len(useful) == 1 and len(req) == 1
            assert sum(r["per_drive_bytes_mean"]) == pytest.approx(
                r["requested_bytes_per_batch"]
            )
            assert r["n_batches"] == sum(
                1 for s in report["batch_samples"] if s["arm"] == name and s["size"] == int(size)
            )


def test_extents_rows_and_per_drive_bytes_are_reported_apart(tmp_path):
    code, report, *_ = _run(tmp_path)
    res = report["results"]
    for size in SIZES:
        n = int(size)
        within = res["within-row 1:1"][size]
        whole = res["whole-row"][size]
        one = res["one-root drive0"][size]
        for r in (within, whole, one):
            assert r["app_rows_per_batch"] == n
        assert within["extents_per_batch"] == 2 * n  # one row, two extents
        assert whole["extents_per_batch"] == n
        assert one["extents_per_batch"] == n
        assert one["per_drive_bytes_mean"][1] == 0
        assert one["drive_share"][0] == pytest.approx(1.0)
        assert one["per_drive_extents_max"] == [n, 0]
        assert within["per_drive_extents_max"] == [n, n]
    # 1 row per batch: whole-row alternates the roots, so neither drive is idle.
    share = res["whole-row"]["1"]["drive_share"]
    assert 0.3 < share[0] < 0.7 and share[0] + share[1] == pytest.approx(1.0)
    weighted = res["weighted 3:1"]["4"]["drive_share"]
    assert weighted[0] == pytest.approx(0.75, abs=0.01)


def test_ring_depth_and_max_extents_are_recorded(tmp_path):
    _, report, *_ = _run(tmp_path)
    assert report["ring_depth"] >= report["results"]["within-row 1:1"]["6"]["max_batch_extents"]
    assert report["results"]["within-row 1:1"]["6"]["max_batch_extents"] == 12


def test_percentiles_counts_pack_and_read_are_present(tmp_path, capsys):
    _, report, *_ = _run(tmp_path)
    r = report["results"]["whole-row"]["4"]
    assert r["n_batches"] == 2 * 2  # 2 batches per rep x 2 reps
    assert set(r["read_ms"]) == {"p50", "p90", "p95", "p99"}
    assert 0 < r["read_ms"]["p50"] <= r["read_ms"]["p90"] <= r["read_ms"]["p95"] <= r["read_ms"]["p99"]
    assert r["pack_ms_mean"] > 0 and r["read_ms_mean"] > 0
    assert r["tail_reliable"] is False
    out = capsys.readouterr().out
    for needle in ("p95", "pack", "per-drive", "application rows", "planned I/O"):
        assert needle in out


def test_poisoned_mirror_is_caught_against_the_source(tmp_path, capsys):
    source, roots = _make(tmp_path)
    for path in Path(roots[1]).glob("*.safetensors"):
        path.write_bytes(bytes(b ^ 0xFF for b in path.read_bytes()))
    out = tmp_path / "r.json"
    assert sched.main(_args(source, roots, out)) == 2
    report = json.loads(out.read_text())
    assert report["bytes_match"] is False
    bad = {m["arm"] for m in report["source_mismatches"]}
    assert "within-row 1:1" in bad and "whole-row" in bad
    assert "one-root drive0" not in bad  # the clean copy still matches
    assert "BYTE MISMATCH" in capsys.readouterr().out


def test_a_submit_that_transfers_nothing_cannot_pass(tmp_path, monkeypatch):
    source, roots = _make(tmp_path)
    layout_reader = sched.Exl3RowReader
    monkeypatch.setattr(
        layout_reader, "_submit", lambda self, f, o, d, l, expected: None
    )
    out = tmp_path / "r.json"
    assert sched.main(_args(source, roots, out, reps="1")) == 2
    assert json.loads(out.read_text())["source_mismatches"]


def test_unequal_work_between_arms_is_flagged():
    def sample(arm, rows):
        return sched.BatchSample(
            arm, 0, rows, 0, 1, 1, 1, rows, rows, [20, 0], [rows, 0], 10, 20, 20
        )

    specs = [sched.ArmSpec("a", "one_root", (1.0, 0.0)), sched.ArmSpec("b", "whole_row")]
    assert sched.check_equal_work([sample("a", 2), sample("b", 2)], specs) == []
    errors = sched.check_equal_work([sample("a", 2), sample("b", 3)], specs)
    assert any("different work" in e for e in errors)


def test_identical_plans_are_called_out(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    args = _args(source, roots, out)
    args[args.index("3:1")] = "1:1"
    assert sched.main(args) == 0
    notes = json.loads(out.read_text())["identical_plans"]
    assert any("weighted 1:1" in n and "within-row 1:1" in n for n in notes)


def test_auto_weights_are_measured_and_recorded(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    args = _args(source, roots, out)
    args[args.index("3:1")] = "auto"
    assert sched.main(args) == 0
    report = json.loads(out.read_text())
    calib = report["calibration"]
    assert len(calib["weights"]) == 2 and max(calib["weights"]) == 1.0
    assert all(m > 0 for m in calib["mb_per_s"]) and calib["bytes_moved"] > 0
    assert report["arms"][-1]["name"].startswith("weighted ")


def test_byte_volume_is_planned_capped_and_dry_run_reads_nothing(tmp_path, capsys, monkeypatch):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    args = _args(source, roots, out)
    with pytest.raises(SystemExit, match="exceeds --max-gib"):
        sched.main(args + ["--max-gib", "0.0001"])
    monkeypatch.setattr(
        sched.Exl3RowReader, "_submit", lambda *a, **k: pytest.fail("dry run read")
    )
    assert sched.main(args + ["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "planned I/O" in text and "per-drive GiB" in text
    assert not out.exists()


def test_io_accounting_matches_planned_bytes(tmp_path):
    _, report, *_ = _run(tmp_path)
    moved = sum(report["io"]["harness_bytes_per_root"].values())
    arms = len(report["arms"])
    per_rep = report["replay"]["requested_bytes_per_rep"]
    warm = report["io"]["warmup_bytes_per_arm"]
    calib = report["calibration"]
    expected = arms * (2 * per_rep + warm) + (calib["bytes_moved"] if calib else 0)
    assert moved == expected


def test_bad_configuration_is_refused_before_any_read(tmp_path):
    source, roots = _make(tmp_path)
    out = tmp_path / "r.json"
    with pytest.raises(SystemExit, match="duplicates"):
        sched.main(_args(source, [roots[0], roots[0]], out))
    with pytest.raises(SystemExit, match="at least two roots"):
        sched.main(_args(source, roots[:1], out))
    with pytest.raises(SystemExit, match="source checkpoint itself"):
        sched.main(_args(source, [roots[0], source], out))
    with pytest.raises(SystemExit, match="outside"):
        sched.main(_args(source, roots, out) + ["--layers", "9"])
    assert not out.exists()


def test_paired_ratios_compare_the_same_batch_in_the_same_rep():
    def sample(arm, rep, batch, ns):
        return sched.BatchSample(arm, rep, 2, batch, ns, 1, 1, 2, 2, [1, 1], [1, 1], 1, 2, 2)

    specs = [sched.ArmSpec("ref", "within_row", (1.0, 1.0)), sched.ArmSpec("x", "whole_row")]
    samples = [
        sample(arm, rep, batch, ns * scale)
        for arm, scale in (("ref", 1.0), ("x", 0.5))
        for rep in (0, 1)
        for batch, ns in ((0, 100), (1, 200), (2, 400))
    ]
    paired = sched.paired_vs_first(samples, specs)["arms"]["x"]["2"]
    assert paired["all_reps"]["pairs"] == 6 and paired["after_rep0"]["pairs"] == 3
    assert paired["all_reps"]["median_ratio"] == pytest.approx(0.5)
    assert paired["after_rep0"]["faster_than_ref"] == 3
    assert paired["after_rep0"]["sign_test_p"] == pytest.approx(0.25)


def test_report_carries_paired_comparison_for_every_non_reference_arm(tmp_path):
    _, report, *_ = _run(tmp_path)
    assert report["paired"]["reference"] == "within-row 1:1"
    assert set(report["paired"]["arms"]) == {"whole-row", "one-root drive0", "weighted 3:1"}
    one = report["paired"]["arms"]["one-root drive0"]["4"]["after_rep0"]
    assert one["pairs"] == 2 and one["median_ratio"] > 0  # both fake roots share one disk


# ------------------------------------------------- request model and conditions


def _drive(path, name, max_sectors_kb):
    return sched.dc.DriveInfo(path, name, 259, 1, "xfs", max_sectors_kb, max_sectors_kb, 128, 0, "fake")


def _model(tmp_path, kinds, drives, sizes=("1", "2", "4")):
    source, _ = _make(tmp_path)
    layout = build_exl3_expert_layout(source)
    replay = sched.build_replay(NUM_EXPERTS, [0], [int(s) for s in sizes], 8, 1234)
    specs = sched.build_arm_specs(kinds, 2, ["a", "b"], 0, (3.0, 1.0))
    return layout, replay, sched.request_model(specs, replay, layout, drives)


def test_request_model_counts_extents_against_each_drives_own_cap(tmp_path):
    drives = [_drive("/a", "nvme0n1p1", 4), _drive("/b", "nvme3n1p1", 2)]
    layout, replay, model = _model(tmp_path, ["within_row"], drives)
    planner = sched.SplitPlanner((1.0, 1.0))
    for size, cell in model["within-row 1:1"].items():
        group = [b for b in replay if b.size == int(size)]
        expected = [0, 0]
        for batch in group:
            for e in planner.plan([sched.row_geometry(layout, r) for r in batch.rows]):
                expected[e.root] += -(-e.length // (drives[e.root].max_sectors_kb * 1024))
        assert cell["requests_per_batch"] == [x / len(group) for x in expected]
        assert cell["requests_per_batch"][1] >= cell["requests_per_batch"][0]  # the 2 KiB cap splits more
        assert cell["bytes_per_batch"][0] == cell["bytes_per_batch"][1] or abs(
            cell["bytes_per_batch"][0] - cell["bytes_per_batch"][1]
        ) <= 4096 * int(size)  # within-row 1:1 is byte-balanced; the requests are what differ


def test_request_model_resets_whole_row_state_per_size_group_like_the_run(tmp_path):
    drives = [_drive("/a", "d0", 4), _drive("/b", "d1", 4)]
    _, _, model = _model(tmp_path, ["whole_row"], drives)
    one = model["whole-row"]["1"]
    assert one["extents_per_batch"] == [0.5, 0.5]  # single-row batches alternate roots from a reset start
    assert model["whole-row"]["2"]["extents_per_batch"] == [1.0, 1.0]


def test_request_model_reports_one_root_as_infinite_imbalance_not_a_crash(tmp_path):
    drives = [_drive("/a", "d0", 4), _drive("/b", "d1", 4)]
    _, _, model = _model(tmp_path, ["one_root"], drives)
    cell = model["one-root a"]["2"]
    assert cell["requests_per_batch"][1] == 0
    assert cell["busiest_over_quietest_requests"] is None
    assert cell["request_share"] == [1.0, 0.0]


def test_model_requests_mode_reads_nothing_and_records_its_conditions(tmp_path, capsys, monkeypatch):
    source, roots = _make(tmp_path)
    out = tmp_path / "model.json"
    monkeypatch.setattr(
        sched.Exl3RowReader, "_submit", lambda *a, **k: pytest.fail("model mode read")
    )
    assert sched.main(_args(source, roots, out, "--model-requests")) == 0
    data = json.loads(out.read_text())
    assert [d["path"] for d in data["drives"]] == [os.path.realpath(r) for r in roots]
    assert all(d["device"] and d["max_sectors_kb"] > 0 for d in data["drives"])
    assert len(data["load_average"]) == 3 and "model" in data
    text = capsys.readouterr().out
    assert "max_sectors_kb" in text and "not measured" in text


def test_model_requests_needs_explicit_weights_for_the_weighted_arm(tmp_path):
    source, roots = _make(tmp_path)
    args = _args(source, roots, tmp_path / "m.json", "--model-requests")
    args[args.index("--weights") + 1] = "auto"
    with pytest.raises(SystemExit, match="explicit --weights"):
        sched.main(args)


def test_a_root_whose_device_cannot_be_attributed_stops_the_run_before_any_read(tmp_path, monkeypatch):
    source, roots = _make(tmp_path)

    def refuse(path, **_):
        raise ValueError(f"{path}: no diskstats row")

    monkeypatch.setattr(sched.dc, "resolve_drive", refuse)
    monkeypatch.setattr(
        sched.Exl3RowReader, "_submit", lambda *a, **k: pytest.fail("read before attribution")
    )
    with pytest.raises(ValueError, match="no diskstats row"):
        sched.main(_args(source, roots, tmp_path / "r.json"))


def test_every_block_carries_device_request_cache_and_load_conditions(tmp_path):
    _, report, _, roots = _run(tmp_path)
    arms, sizes, reps = len(report["arms"]), len(SIZES), 2
    assert len(report["condition_blocks"]) == arms * sizes * reps
    assert [d["path"] for d in report["drives"]] == [os.path.realpath(r) for r in roots]
    assert report["load_average_at_start"] and report["load_average_at_end"]
    lo, hi, count = report["cpus_allowed"]
    assert lo <= hi and count >= 1
    for block in report["condition_blocks"]:
        assert set(block) >= {"arm", "rep", "size", "batches", "wall_s", "load_average_before",
                              "foreign_cores_all", "drives"}  # fmt: skip
        assert len(block["drives"]) == 2
        assert all("residency_before" in d and "reads_completed" in d for d in block["drives"])
    cell = report["conditions"]["within-row 1:1"]["4"]
    samples = [s for s in report["batch_samples"] if s["arm"] == "within-row 1:1" and s["size"] == 4]
    assert [d["predicted_requests"] for d in cell["drives"]] == [
        sum(s["drive_requests_model"][r] for s in samples) for r in (0, 1)
    ]
    assert all(d["asked_bytes"] > 0 for d in cell["drives"])


def _block(arm, size, *, reads, before, after, measured_bytes=0):
    drive = {
        "device": "d0", "read_bytes": measured_bytes, "reads_completed": reads,
        "reads_merged": 0, "io_ms": 10, "weighted_io_ms": 40,
        "residency_before": None if before is None else {"resident_bytes": before, "total_bytes": 1 << 40},
        "residency_delta_bytes": None if before is None or after is None else after - before,
    }  # fmt: skip
    return {"arm": arm, "size": size, "rep": 0, "batches": 1, "drives": [drive],
            "load_average_before": [4.0, 0, 0], "load_average_after": [5.0, 0, 0],
            "foreign_cores_low": 0.5, "foreign_cores_all": 1.5}  # fmt: skip


def _sample(arm, size, requests):
    return sched.BatchSample(arm, 0, size, 0, 1, 1, 1, size, 1, [100], [1], 1, 100, 100, [requests])


def test_conditions_summary_flags_request_disagreement_and_cache_movement():
    samples = [_sample("a", 4, 100)]
    ok = sched.conditions_summary([_block("a", 4, reads=110, before=1 << 30, after=(1 << 30) + 1)], samples, 1)
    cell = ok["a"]["4"]["drives"][0]
    assert cell["requests_disagree"] is False and cell["residency_moved"] is False
    assert cell["depth_when_busy"] == 4.0
    assert ok["a"]["4"]["load_average_1m"] == [4.0, 5.0]
    bad = sched.conditions_summary(
        [_block("a", 4, reads=200, before=1 << 30, after=(1 << 30) + 100 * (1 << 20))], samples, 1
    )["a"]["4"]["drives"][0]
    assert bad["requests_disagree"] is True and bad["residency_moved"] is True


def test_unmeasured_residency_counts_as_moved_not_as_cold():
    cell = sched.conditions_summary([_block("a", 4, reads=100, before=None, after=None)],
                                    [_sample("a", 4, 100)], 1)["a"]["4"]["drives"][0]  # fmt: skip
    assert cell["residency_unmeasured"] is True and cell["residency_moved"] is True
    assert cell["residency_before_min"] is None


def test_conditions_table_prints_model_measurement_and_flags():
    summary = sched.conditions_summary(
        [_block("a", 4, reads=200, before=1 << 30, after=(1 << 30) + 100 * (1 << 20))],
        [_sample("a", 4, 100)], 1,
    )  # fmt: skip
    text = sched.format_conditions(summary)
    assert "100/200?" in text and "1.0-1.0!" in text and "d0" in text
