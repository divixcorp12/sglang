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
