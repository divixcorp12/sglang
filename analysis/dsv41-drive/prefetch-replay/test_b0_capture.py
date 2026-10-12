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


def _serve_ok(handle, session, sdir, block_dir):
    assert os.listdir(sdir) == []  # every attempt serves into a clean session dir
    Path(sdir, "payload").write_text(session["session_id"])
    with open(Path(block_dir, "trace.jsonl"), "a") as f:
        f.write(session["session_id"] + "\n")
    handle["served"].append(session["session_id"])


def _raise(exc):
    raise exc


def _blocks(tmp_path, sessions, *, block_size=b0.DEFAULT_BLOCK, split=None, start=None, serve_session=_serve_ok,
            check_block=None, check_session=None, log=None):
    """capture_blocks with fakes; ``log`` collects ("start", block), ("stop", served sids) events."""
    log = [] if log is None else log

    def default_start(block_dir):
        log.append(("start", os.path.basename(block_dir)))
        return {"served": []}

    def stop(handle):
        log.append(("stop", handle["served"]))

    ids = [s["session_id"] for s in sessions]
    return b0.capture_blocks(
        out_dir=str(tmp_path), sessions=sessions, split=split or {"train": ids}, block_size=block_size,
        start=start or default_start, serve_session=serve_session, stop=stop,
        check_block=check_block or (lambda d: {"verify_forwards": 3}),
        check_session=check_session or (lambda s, d: None),
    )


def _done(tmp_path, sid):
    return json.loads((tmp_path / sid / b0.DONE_NAME).read_text())


def test_a_failed_session_is_reserved_once_and_then_passes(tmp_path):
    seen = []

    def check_session(session, sdir):
        seen.append(session["session_id"])
        if seen.count("a") == 1 and session["session_id"] == "a":
            raise RuntimeError("seq gap")

    out = _blocks(tmp_path, [_session("a"), _session("b")], check_session=check_session)
    assert out == {"done": ["a", "b"], "failed": [], "blocks": {"a": "b001", "b": "b000"}, "warm_start": ["a"]}
    done = _done(tmp_path, "a")
    assert done == {"session_id": "a", "block": "b001", "warm_start": True, "attempts": 2,
                    "summary": {"verify_forwards": 3}}
    assert _done(tmp_path, "b")["warm_start"] is False
    assert json.loads((tmp_path / "a" / "rids.json").read_text())["a-t0"]["split"] == "train"
    # b000 holds b's capture, so it stays in place.
    assert (tmp_path / b0.BLOCKS_DIR / "b000" / "trace.jsonl").read_text() == "a\nb\n"


def test_a_session_failing_twice_is_listed_and_leaves_the_corpus_root(tmp_path):
    out = _blocks(tmp_path, [_session("a")], check_block=lambda d: _raise(RuntimeError("missing router record")))
    assert out == {"done": [], "failed": ["a"], "blocks": {}, "warm_start": []}
    assert not (tmp_path / "a").exists()
    failed = json.loads((tmp_path / b0.FAILED_DIR / "a" / b0.FAILED_NAME).read_text())
    assert failed["attempts"] == b0.ATTEMPTS and len(failed["reasons"]) == b0.ATTEMPTS
    assert "missing router record" in failed["reasons"][0]
    assert sorted(os.listdir(tmp_path / b0.FAILED_DIR / b0.BLOCKS_DIR)) == ["b000", "b001"]
    assert os.listdir(tmp_path / b0.BLOCKS_DIR) == []


def test_a_rerun_skips_done_and_failed_sessions_and_serves_the_rest(tmp_path):
    split = {"train": ["a", "b", "c"]}
    sessions = [_session(s) for s in "abc"]
    first = _blocks(tmp_path, sessions[:2], split=split,
                    check_session=lambda s, d: None if s["session_id"] == "a" else _raise(RuntimeError("bad")))
    assert first["done"] == ["a"] and first["failed"] == ["b"]
    log = []
    second = _blocks(tmp_path, sessions, split=split, log=log)
    assert log == [("start", "b002"), ("stop", ["c"])]
    assert second == {"done": ["a", "c"], "failed": ["b"], "blocks": {"a": "b000", "c": "b002"},
                      "warm_start": ["a", "c"]}


def test_the_run_stops_after_consecutive_failures(tmp_path):
    ids = [f"s{i}" for i in range(6)]
    log = []
    out = _blocks(tmp_path, [_session(i) for i in ids], block_size=2, log=log,
                  check_block=lambda d: _raise(RuntimeError("bad")))
    assert out["failed"] == ids[: 2 * b0.MAX_CONSECUTIVE_FAILED_BLOCKS]
    assert [e for e in log if e[0] == "start"] == [("start", f"b{i:03d}") for i in range(4)]


def test_block_sessions_share_one_server_start_and_the_first_is_warm(tmp_path):
    ids = [f"s{i}" for i in range(5)]
    log = []
    out = _blocks(tmp_path, [_session(i) for i in ids], block_size=2, log=log)
    assert log == [("start", "b000"), ("stop", ["s0", "s1"]), ("start", "b001"), ("stop", ["s2", "s3"]),
                   ("start", "b002"), ("stop", ["s4"])]
    assert out["done"] == ids and out["warm_start"] == ["s0", "s2", "s4"]
    assert out["blocks"] == {"s0": "b000", "s1": "b000", "s2": "b001", "s3": "b001", "s4": "b002"}
    rids = json.loads((tmp_path / b0.BLOCKS_DIR / "b001" / "rids.json").read_text())
    assert sorted(rids) == ["s2-t0", "s2-t1", "s3-t0", "s3-t1"]


def test_block_server_death_reserves_the_block_and_retires_its_dir(tmp_path):
    starts = []

    def serve_session(handle, session, sdir, block_dir):
        if len(starts) == 1 and session["session_id"] != "a":
            raise RuntimeError("runner exited 1")  # the server died after a
        _serve_ok(handle, session, sdir, block_dir)

    def start(block_dir):
        starts.append(block_dir)
        return {"served": []}

    out = _blocks(tmp_path, [_session(s) for s in "abc"], start=start, serve_session=serve_session,
                  check_block=lambda d: _raise(OSError("router ring short")) if len(starts) == 1 else {})
    assert out["done"] == ["a", "b", "c"] and out["warm_start"] == ["a"]
    assert set(out["blocks"].values()) == {"b001"}
    assert os.listdir(tmp_path / b0.FAILED_DIR / b0.BLOCKS_DIR) == ["b000"]
    assert _done(tmp_path, "a")["attempts"] == 2


def test_block_unhealthy_start_is_reserved_and_never_stopped(tmp_path):
    log = []

    def start(block_dir):
        if not log:
            log.append(("start-failed", os.path.basename(block_dir)))
            raise RuntimeError("the server never became healthy")
        log.append(("start", os.path.basename(block_dir)))
        return {"served": []}

    out = _blocks(tmp_path, [_session("a")], start=start, log=log)
    assert log == [("start-failed", "b000"), ("start", "b001"), ("stop", ["a"])]
    assert out["blocks"] == {"a": "b001"} and out["warm_start"] == ["a"]


def test_block_a_failing_session_does_not_fail_the_rest_of_its_block(tmp_path):
    out = _blocks(tmp_path, [_session(s) for s in "abc"],
                  check_session=lambda s, d: _raise(RuntimeError("turn 1 unanswered")) if s["session_id"] == "b"
                  else None)
    assert out["done"] == ["a", "c"] and out["failed"] == ["b"]
    assert out["blocks"] == {"a": "b000", "c": "b000"}
    failed = json.loads((tmp_path / b0.FAILED_DIR / "b" / b0.FAILED_NAME).read_text())
    assert [r.split(":")[0] for r in failed["reasons"]] == ["attempt 1 (block b000)", "attempt 2 (block b001)"]
    # b000 keeps a and c's capture; b001 served only b and got no DONE.
    assert os.listdir(tmp_path / b0.BLOCKS_DIR) == ["b000"]
    assert os.listdir(tmp_path / b0.FAILED_DIR / b0.BLOCKS_DIR) == ["b001"]


def test_block_ids_skip_existing_dirs_in_blocks_and_failed_blocks(tmp_path):
    (tmp_path / b0.BLOCKS_DIR / "b000").mkdir(parents=True)
    (tmp_path / b0.FAILED_DIR / b0.BLOCKS_DIR / "b003").mkdir(parents=True)
    assert b0._next_block_id(str(tmp_path)) == "b004"
    assert b0._next_block_id(str(tmp_path)) == "b005"


def test_block_plan_groups_in_order_up_to_the_block_size():
    sessions = [_session(f"s{i}") for i in range(5)]
    plan = b0.plan_blocks(sessions=sessions, block_size=2)
    assert [[s["session_id"] for s in block] for block in plan] == [["s0", "s1"], ["s2", "s3"], ["s4"]]
    assert b0.plan_blocks(sessions=[], block_size=2) == []
    with pytest.raises(ValueError):
        b0.plan_blocks(sessions=sessions, block_size=0)


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
