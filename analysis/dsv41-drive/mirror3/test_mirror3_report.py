import datetime
import json

import mirror3_report as report


def _sample(mono, per_dev):
    fields = dict(reads=0, sectors_read=0, io_ticks_ms=0)
    return {"mono": mono, **{d: {**fields, **v} for d, v in per_dev.items()}}


def test_drive_split_takes_the_deltas_inside_the_window_and_shares_the_bytes():
    samples = [
        _sample(0.0, {"a": dict(reads=0, sectors_read=0), "b": dict(reads=0, sectors_read=0)}),
        _sample(10.0, {"a": dict(reads=100, sectors_read=2000), "b": dict(reads=50, sectors_read=2000)}),
        _sample(20.0, {"a": dict(reads=300, sectors_read=6000, io_ticks_ms=5000), "b": dict(reads=150, sectors_read=4000)}),
        _sample(99.0, {"a": dict(reads=999, sectors_read=99999), "b": dict(reads=999, sectors_read=99999)}),
    ]
    out = report.drive_split(samples, 5.0, 25.0, ["a", "b"])  # the window holds the samples at 10 and 20
    assert out["seconds"] == 10.0
    a, b = out["drives"]["a"], out["drives"]["b"]
    assert a["read_MB"] == 4000 * 512 / 1e6 and b["read_MB"] == 2000 * 512 / 1e6
    assert a["iops"] == 20.0 and a["req_kB"] == 4000 * 512 / 1e3 / 200
    assert round(a["share_pct"] + b["share_pct"], 9) == 100.0 and a["share_pct"] > b["share_pct"]
    assert a["util_pct"] == 50.0


def test_identity_compares_reasoning_and_content_per_turn(tmp_path):
    for name, text in (("ref", "x"), ("new", "y")):
        d = tmp_path / name
        d.mkdir()
        rows = [dict(session_id="s1", turn=0, reasoning="r", content="c", decode_tokens_per_sec=10.0),
                dict(session_id="s2", turn=0, reasoning="r", content=text, decode_tokens_per_sec=8.0)]
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert report.identity(tmp_path / "ref", tmp_path / "new") == {"s1/t0": True, "s2/t0": False}
    per_turn, median = report.ms_per_token(tmp_path / "ref")
    assert per_turn == {"s1/t0": 100.0, "s2/t0": 125.0} and median == 112.5


def test_timed_window_runs_from_server_ready_to_the_last_session(tmp_path):
    rows = [dict(label="before_server", monotonic=1.0), dict(label="server_ready", monotonic=5.0),
            dict(label="session_a", monotonic=9.0), dict(label="session_b", monotonic=12.0)]
    (tmp_path / "boundary-samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert report.timed_window(tmp_path) == (5.0, 12.0)


def test_decode_pools_seconds_over_tokens_per_session_and_arm(tmp_path):
    rows = [dict(session_id="s1", turn=0, completion_tokens=101, decode_tokens_per_sec=10.0, ttft=2.0),  # 100 tok, 10 s
            dict(session_id="s1", turn=1, completion_tokens=51, decode_tokens_per_sec=5.0, ttft=4.0),  # 50 tok, 10 s
            dict(session_id="s2", turn=0, completion_tokens=11, decode_tokens_per_sec=1.0, ttft=6.0),  # 10 tok, 10 s
            dict(session_id="s3", turn=0, error="boom")]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = report.decode(tmp_path)
    assert out["sessions"]["s1"]["pooled_ms_per_token"] == 1000 * 20 / 150
    assert out["sessions"]["s1"]["ttft_s"] == {"s1/t0": 2.0, "s1/t1": 4.0}
    assert out["sessions"]["s2"]["pooled_ms_per_token"] == 1000.0
    assert "s3" not in out["sessions"]
    assert out["pooled_ms_per_token"] == 1000 * 30 / 160 and out["decode_tokens"] == 160
    assert out["median_ttft_s"] == 4.0 and out["total_ttft_s"] == 12.0


def test_clock_summary_keeps_the_samples_inside_the_window(tmp_path):
    t0 = datetime.datetime(2026, 9, 28, 12, 0, 0, tzinfo=datetime.timezone.utc)
    lines = ["timestamp, clocks.current.sm [MHz], clocks.max.sm [MHz]"]
    for i, mhz in enumerate((225, 2500, 2600, 2700, 225)):
        local = (t0 + datetime.timedelta(seconds=i)).astimezone().replace(tzinfo=None)
        lines.append(f"{local.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, {mhz} MHz, 3135 MHz")
    (tmp_path / "c.csv").write_text("\n".join(lines) + "\n")
    out = report.clock_summary(tmp_path / "c.csv", t0 + datetime.timedelta(seconds=0.5), t0 + datetime.timedelta(seconds=3.5))
    assert out == {"samples": 3, "sm_min_mhz": 2500, "sm_median_mhz": 2600, "sm_max_mhz": 2700, "max_sm_mhz": [3135]}


def test_ram_miss_reads_the_last_shutdown_counters(tmp_path):
    thread = {"served": 7, "rows_read": 9, "read_errors": 0, "copy_errors": 0, "copy_fallbacks": 0, "overruns": 0,
              "late_after_fatal": 0, "piece_stream_refused": 0, "piece_publish_refused": 1, "evictions": 5}
    (tmp_path / "server.log").write_text(
        "[2026-09-28 04:10:25] exl3 expert stream: " + json.dumps({"vram_misses": 3, "ram_misses": 2, "read_ms": 1.5}) + "\n"
        + "exl3 RAM miss thread counters " + json.dumps({**thread, "served": 1}) + "\n"
        + "exl3 RAM miss thread counters " + json.dumps(thread) + "\n")
    (tmp_path / "results-warmup-1.jsonl").write_text("")
    (tmp_path / "results-warmup-2.jsonl").write_text("")
    out = report.ram_miss(tmp_path)
    assert out["warmup_rounds"] == 2
    assert out["thread"] == {k: v for k, v in thread.items() if k != "evictions"}
    assert out["stream"] == {"vram_misses": 3, "ram_misses": 2, "read_ms": 1.5}


def test_ram_miss_is_none_when_the_server_never_wrote_its_counters(tmp_path):
    (tmp_path / "server.log").write_text("killed\n")
    out = report.ram_miss(tmp_path)
    assert out["thread"] is None and out["stream"] is None
