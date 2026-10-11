"""Per-session drive load from run_arm.sh's boundary samples, and the slow-session flag (CPU, synthetic counters)."""

import importlib.util
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GB = 10**9


def _module():
    spec = importlib.util.spec_from_file_location(
        "drive_sessions", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "drive_sessions.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _c(device="nvme2n1p1", reads=0, read_bytes=0, writes=0, written_bytes=0, busy_ms=0, queue_ms=0):
    return {"device": device, "reads": reads, "sectors_read": read_bytes // 512, "writes": writes,
            "sectors_written": written_bytes // 512, "io_ticks_ms": busy_ms, "time_in_queue_ms": queue_ms}  # fmt: skip


def _row(label, t, session_id=None, **drives):
    row = {"label": label, "utc": "x", "monotonic": t, "drive_counters": drives}
    return row if session_id is None else {**row, "session_id": session_id}


def _session(sid, t0, t1, start, end):
    return [_row("session_start", t0, sid, **start), _row("session_end", t1, sid, **end)]


def test_one_sessions_deltas_give_gb_busy_s_per_gb_mean_read_size_and_mb_written():
    ds = _module()
    start = {"nvme4": _c(reads=100, read_bytes=GB, writes=5, written_bytes=1_048_576, busy_ms=1000, queue_ms=5000)}
    end = {"nvme4": _c(reads=1100, read_bytes=3 * GB, writes=9, written_bytes=2_072_576, busy_ms=1640, queue_ms=7000)}
    load = ds.session_drive_load(_session("s0", 10.0, 30.0, start, end))
    assert load["notes"] == [] and load["warmup"] is None
    [session] = load["sessions"]
    assert session["session_id"] == "s0" and session["wall_s"] == 20.0
    d = session["drives"]["nvme4"]
    assert d["device"] == "nvme2n1p1"
    assert d["gb_read"] == pytest.approx(2.0)
    assert d["busy_s"] == pytest.approx(0.64)
    assert d["busy_s_per_gb"] == pytest.approx(0.32)
    assert d["reads"] == 1000 and d["mean_bytes_per_read"] == pytest.approx(2.0e6)
    assert d["mb_written"] == pytest.approx(1.024)
    assert d["queue_s"] == pytest.approx(2.0) and d["busy_fraction"] == pytest.approx(0.032)


def test_sessions_keep_their_order_and_the_warm_up_is_reported_apart():
    ds = _module()
    a, b, c = (_c(read_bytes=n * GB, busy_ms=n * 300, reads=n) for n in (1, 2, 4))
    rows = [
        {"label": "before_server", "diskstats_sectors": {}},  # older boundary rows carry no drive_counters
        _row("warmup_start", 0.0, nvme0=a), _row("warmup_end", 5.0, nvme0=b),
        {"label": "server_ready"},
        *_session("s1", 6.0, 7.0, {"nvme0": b}, {"nvme0": c}),
        {"label": "session_s1"},
        *_session("s0", 8.0, 9.0, {"nvme0": a}, {"nvme0": c}),
    ]  # fmt: skip
    load = ds.session_drive_load(rows)
    assert load["warmup"]["drives"]["nvme0"]["gb_read"] == pytest.approx(1.0)
    assert [s["session_id"] for s in load["sessions"]] == ["s1", "s0"]
    assert load["sessions"][1]["drives"]["nvme0"]["gb_read"] == pytest.approx(3.0)


def test_a_counter_that_went_backwards_skips_that_drive_with_a_note_and_keeps_the_others():
    ds = _module()
    start = {"nvme0": _c(read_bytes=GB, busy_ms=500), "nvme4": _c(read_bytes=GB, busy_ms=4_294_967_000)}
    end = {"nvme0": _c(read_bytes=2 * GB, busy_ms=800), "nvme4": _c(read_bytes=2 * GB, busy_ms=200)}  # io_ticks wrapped
    load = ds.session_drive_load(_session("s0", 0.0, 1.0, start, end))
    drives = load["sessions"][0]["drives"]
    assert drives["nvme4"] is None and drives["nvme0"]["busy_s"] == pytest.approx(0.3)
    [note] = load["notes"]
    assert "s0" in note and "nvme4" in note and "io_ticks_ms" in note


def test_a_drive_that_changed_device_or_vanished_is_skipped_with_a_note():
    ds = _module()
    start = {"nvme0": _c(), "nvme4": _c(device="nvme2n1p1")}
    end = {"nvme0": _c(device="nvme0n1p1"), "nvme2": _c()}
    load = ds.session_drive_load(_session("s0", 0.0, 1.0, start, end))
    drives = load["sessions"][0]["drives"]
    assert drives == {"nvme0": None, "nvme2": None, "nvme4": None}
    assert len(load["notes"]) == 3 and all("s0" in n for n in load["notes"])


def test_unreadable_counters_or_a_missing_end_sample_give_notes_not_a_crash():
    ds = _module()
    rows = [
        {"label": "session_start", "session_id": "s0", "monotonic": 0.0, "drive_counters": None,
         "drive_counters_unavailable": "ValueError: no row"},
        _row("session_end", 1.0, "s0", nvme0=_c()),
        _row("session_start", 2.0, "s1", nvme0=_c()),  # the session failed; no end sample
        _row("warmup_end", 3.0, nvme0=_c()),
    ]  # fmt: skip
    load = ds.session_drive_load(rows)
    assert load["sessions"] == [] and load["warmup"] is None
    text = " | ".join(load["notes"])
    assert "s0" in text and "ValueError: no row" in text and "s1" in text and "warm-up" in text


def test_no_samples_at_all_is_one_note():
    ds = _module()
    load = ds.session_drive_load([{"label": "before_server", "diskstats_sectors": {"nvme0": 1}}])
    assert load["sessions"] == [] and len(load["notes"]) == 1 and "no per-session drive counters" in load["notes"][0]


def _load_with(ds, per_session):
    """per_session: [(session_id, {label: (gb, busy_s)})]"""
    rows = []
    for t, (sid, drives) in enumerate(per_session):
        start = {k: _c() for k in drives}
        end = {k: _c(read_bytes=int(gb * GB), busy_ms=int(round(busy * 1000)), reads=1) for k, (gb, busy) in drives.items()}
        rows += _session(sid, float(t), t + 0.5, start, end)
    return ds.session_drive_load(rows)


def test_a_session_is_flagged_over_15_percent_above_the_drives_median_over_both_arms():
    ds = _module()
    loads = {
        "A": _load_with(ds, [("s0", {"nvme4": (1, 0.30), "nvme0": (1, 0.30)}),
                             ("s1", {"nvme4": (1, 0.32), "nvme0": (1, 0.40)})]),
        "B": _load_with(ds, [("s0", {"nvme4": (1, 0.40), "nvme0": (1, 0.30)}),
                             ("s1", {"nvme4": (2, 0.62), "nvme0": (0, 0.0)})]),
    }  # fmt: skip
    # nvme4 per GB: 0.30, 0.32, 0.40, 0.31 -> median 0.315; 0.40 is +27%, 0.32 is +1.6%.
    # nvme0 per GB: 0.30, 0.40, 0.30 (the 0-GB session has no ratio) -> median 0.30; 0.40 is +33%.
    flags = ds.flag_sessions(loads)
    assert [(f["session_id"], f["drive"]) for f in flags["A"]] == [("s1", "nvme0")]
    [b] = flags["B"]
    assert (b["session_id"], b["drive"]) == ("s0", "nvme4")
    assert b["busy_s_per_gb"] == pytest.approx(0.40) and b["median_busy_s_per_gb"] == pytest.approx(0.315)
    assert b["excess_pct"] == pytest.approx(100 * (0.40 / 0.315 - 1))
    assert b["device"] == "nvme2n1p1"


def test_exactly_15_percent_over_the_median_is_not_flagged():
    ds = _module()
    loads = {"A": _load_with(ds, [("s0", {"nvme4": (1, 0.20)}), ("s1", {"nvme4": (1, 0.20)}), ("s2", {"nvme4": (1, 0.23)})])}
    assert ds.flag_sessions(loads) == {"A": []}


def test_the_table_lists_every_arms_sessions_and_marks_the_flagged_cells():
    ds = _module()
    loads = {
        "A": _load_with(ds, [("cfq-train-Single_CDW/2015/page_35.pdf-2", {"nvme4": (1, 0.30)})]),
        "B": _load_with(ds, [("s0", {"nvme4": (1, 0.60)}), ("s1", {"nvme4": (1, 0.31)})]),
    }
    flags = ds.flag_sessions(loads)
    text = ds.format_table(loads, flags)
    lines = text.splitlines()
    assert "busy s per GB" in lines[0]
    assert any(line.startswith("A") and "0.30" in line and "1.00 GB" in line for line in lines)
    assert any(line.startswith("B") and "0.60*" in line for line in lines)
    assert not any("0.31*" in line for line in lines)
