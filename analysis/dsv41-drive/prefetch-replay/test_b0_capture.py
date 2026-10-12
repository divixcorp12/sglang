"""b0_capture.py: the server env, the rid map and the resume/retry bookkeeping. CPU only."""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_capture as b0  # noqa: E402


def _session(sid, turns=2):
    return {"session_id": sid, "turns": [f"q{i}" for i in range(turns)]}


def test_the_server_env_is_the_misscut_arm_with_prefetch_off_and_the_capture_files(tmp_path):
    env = b0.server_env(str(tmp_path))
    assert env["SGLANG_DSV41_RAM_PREFETCH"] == "0"
    assert env["SGLANG_DSV41_CPU_SPLIT_MISS_CUT"] == "1"
    assert env["SGLANG_DSV41_RESIDENCY_CAPTURE"] == "1"
    assert env["SGLANG_DSV41_ROUTER_CAPTURE_PATH"] == str(tmp_path / "router")
    assert env["SGLANG_DSV41_EXPERT_TRACE_PATH"] == str(tmp_path / "trace.jsonl")
    assert env["SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH"] == str(tmp_path / "verify-accept")


def test_rids_map_to_session_turn_and_split():
    out = b0.rid_map(_session("s1", turns=2), split_name="dev")
    assert out == {
        "s1-t0": {"session_id": "s1", "turn": 0, "split": "dev"},
        "s1-t1": {"session_id": "s1", "turn": 1, "split": "dev"},
    }


def _results(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_session_with_an_unanswered_turn_or_an_error_record_is_incomplete(tmp_path):
    session = _session("s1", turns=2)
    _results(tmp_path / "results.jsonl", [{"turn": 0, "rid": "s1-t0"}, {"turn": 1, "rid": "s1-t1"}])
    b0.results_complete(str(tmp_path), session=session)
    _results(tmp_path / "results.jsonl", [{"turn": 0, "rid": "s1-t0"}, {"turn": 1, "rid": "s1-t1", "error": "x"}])
    with pytest.raises(RuntimeError, match="s1-t1"):
        b0.results_complete(str(tmp_path), session=session)


def _serve_ok(session, sdir):
    Path(sdir, "payload").write_text(session["session_id"])


def test_a_failed_session_is_reserved_once_and_then_passes(tmp_path):
    calls = []

    def serve(session, sdir):
        calls.append(os.listdir(sdir))  # a retry starts from a wiped dir
        _serve_ok(session, sdir)

    def check(session, sdir):
        if len(calls) == 1:
            raise RuntimeError("seq gap")
        return {"verify_forwards": 3}

    out = b0.capture_all(out_dir=str(tmp_path), sessions=[_session("a")], split={"train": ["a"]}, serve=serve,
                         check=check)
    assert out == {"done": ["a"], "failed": []} and calls == [[], []]
    done = json.loads((tmp_path / "a" / b0.DONE_NAME).read_text())
    assert done["attempts"] == 2 and done["summary"] == {"verify_forwards": 3}
    assert json.loads((tmp_path / "a" / "rids.json").read_text())["a-t0"]["split"] == "train"


def test_a_session_failing_twice_is_listed_and_leaves_the_corpus_root(tmp_path):
    def check(session, sdir):
        raise RuntimeError("missing router record")

    out = b0.capture_all(out_dir=str(tmp_path), sessions=[_session("a")], split={"train": ["a"]}, serve=_serve_ok,
                         check=check)
    assert out == {"done": [], "failed": ["a"]}
    assert not (tmp_path / "a").exists()
    failed = json.loads((tmp_path / b0.FAILED_DIR / "a" / b0.FAILED_NAME).read_text())
    assert failed["attempts"] == b0.ATTEMPTS and len(failed["reasons"]) == b0.ATTEMPTS


def test_a_rerun_skips_done_and_failed_sessions_and_serves_the_rest(tmp_path):
    split = {"train": ["a", "b", "c"]}
    sessions = [_session(s) for s in "abc"]
    first = b0.capture_all(
        out_dir=str(tmp_path), sessions=sessions[:2], split=split, serve=_serve_ok,
        check=lambda s, d: {} if s["session_id"] == "a" else (_ for _ in ()).throw(RuntimeError("bad")),
    )
    assert first == {"done": ["a"], "failed": ["b"]}
    served = []
    second = b0.capture_all(
        out_dir=str(tmp_path), sessions=sessions, split=split,
        serve=lambda s, d: (served.append(s["session_id"]), _serve_ok(s, d)), check=lambda s, d: {},
    )
    assert served == ["c"] and second == {"done": ["a", "c"], "failed": ["b"]}


def test_the_run_stops_after_consecutive_failures(tmp_path):
    ids = [f"s{i}" for i in range(b0.MAX_CONSECUTIVE_FAILURES + 3)]
    out = b0.capture_all(
        out_dir=str(tmp_path), sessions=[_session(i) for i in ids], split={"train": ids}, serve=_serve_ok,
        check=lambda s, d: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    assert len(out["failed"]) == b0.MAX_CONSECUTIVE_FAILURES


def _sample(label, sid, mono, *, sectors, ticks):
    drive = {"device": "nvme1n1", "reads": 10, "sectors_read": sectors, "writes": 0, "sectors_written": 0,
             "io_ticks_ms": ticks, "time_in_queue_ms": 0}
    return {"label": label, "session_id": sid, "monotonic": mono, "drive_counters": {"d0": drive}}


def test_a_session_with_a_slow_drive_is_flagged(tmp_path):
    gb = 2_000_000  # sectors of 512 B: 1.024 GB
    for sid, ticks in (("a", 1000), ("b", 1000), ("c", 2000)):
        d = tmp_path / sid
        d.mkdir()
        rows = [_sample("session_start", sid, 0.0, sectors=0, ticks=0),
                _sample("session_end", sid, 10.0, sectors=gb, ticks=ticks)]
        (d / "drive-samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    flags = b0.drive_flags(out_dir=str(tmp_path), ids=["a", "b", "c"])
    assert [f["session_id"] for f in flags] == ["c"]
