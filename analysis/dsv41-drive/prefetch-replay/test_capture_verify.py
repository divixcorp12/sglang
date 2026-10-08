"""capture_verify.py: the sessions file, the server env and the capture check. CPU only."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture_verify as cv  # noqa: E402


def test_the_sessions_file_skips_the_suite_and_the_warmup_row(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"session_id": f"s{i}"}) + "\n" for i in range(20)))
    held_out = cv.write_sessions(str(corpus), str(tmp_path / "sessions.jsonl"), skip=9, n=8)
    assert [json.loads(l)["session_id"] for l in open(held_out)] == [f"s{i}" for i in range(9, 17)]
    suite = cv.write_sessions(str(corpus), str(tmp_path / "suite.jsonl"), skip=0, n=8)
    assert [json.loads(l)["session_id"] for l in open(suite)] == [f"s{i}" for i in range(8)]


def test_the_server_env_names_the_trace_and_router_capture_under_the_out_dir(tmp_path):
    env = cv.server_env(str(tmp_path))
    assert env["SGLANG_DSV41_EXPERT_TRACE_PATH"] == str(tmp_path / "trace.jsonl")
    assert env["SGLANG_DSV41_ROUTER_CAPTURE_PATH"] == str(tmp_path / "router")
    assert env["SGLANG_MOE_HOT_METRICS_FILE"] == str(tmp_path / "metrics.jsonl")
    assert env["SGLANG_DSV41_CPU_EXPERTS"] == "1" and env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"


def test_the_driver_refuses_a_capture_with_dropped_forwards(tmp_path):
    trace = tmp_path / "trace.jsonl"
    lines = [
        {"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": [0], "hot_capacity": [1]},
        {"kind": "graph_routes", "schema": 3, "seq": 0, "forward_pass_id": 1, "phase": "target_verify", "rids": ["a"],
         "forward_tokens": 6, "routes": [[1]], "misses": [0], "router": 0},
        {"kind": "graph_routes", "schema": 3, "seq": 5, "forward_pass_id": 2, "phase": "target_verify", "rids": ["a"],
         "forward_tokens": 6, "routes": [[1]], "misses": [0], "router": 1, "dropped_before": 4},
    ]
    trace.write_text("".join(json.dumps(l) + "\n" for l in lines))
    (tmp_path / "router.json").write_text(json.dumps({"schema": 2, "layer_ids": [0], "tokens": 6, "hidden": 4, "topk": 2}))
    with pytest.raises(RuntimeError, match="lost 4 graph forwards"):
        cv.check_capture(str(tmp_path))


def _capture(tmp_path, forward_tokens, live, routers=None, M=6):
    """A schema-2 capture of len(forward_tokens) verify forwards: trace lines and router side files."""
    import numpy as np

    n = len(forward_tokens)
    routers = list(range(n)) if routers is None else routers
    lines = [{"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": [0], "hot_capacity": [1]}]
    lines += [{"kind": "graph_routes", "schema": 3, "seq": s, "forward_pass_id": s + 1, "phase": "target_verify",
               "rids": ["a"], "forward_tokens": t, "routes": [[1]], "misses": [0], "router": r}
              for s, (t, r) in enumerate(zip(forward_tokens, routers))]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    prefix = str(tmp_path / "router")
    (tmp_path / "router.json").write_text(json.dumps({"schema": 2, "layer_ids": [0], "tokens": M, "hidden": 4, "topk": 2}))
    np.zeros((n, 1, M, 4), dtype=np.uint16).tofile(prefix + ".x.bin")
    np.zeros((n, 1, M, 2), dtype=np.int32).tofile(prefix + ".ids.bin")
    np.zeros((n, 1, M, 2), dtype=np.float32).tofile(prefix + ".w.bin")
    np.stack([np.arange(n), np.arange(n) + 1, np.asarray(live)], axis=1).astype(np.int64).tofile(prefix + ".seq.bin")


def test_check_capture_accepts_a_capture_whose_live_counts_agree(tmp_path):
    _capture(tmp_path, [6, 4], [6, 4])
    assert cv.check_capture(str(tmp_path))["verify_forwards"] == 2


def test_check_capture_refuses_a_verify_without_its_live_token_count(tmp_path):
    """A verify logged with 0 tokens (no live count reached the meta) cannot be ranked or budgeted."""
    _capture(tmp_path, [6, 0], [6, 0])
    with pytest.raises(RuntimeError, match="live token count"):
        cv.check_capture(str(tmp_path))


def test_check_capture_refuses_a_trace_and_side_files_that_disagree(tmp_path):
    _capture(tmp_path, [6, 4], [6, 5])
    with pytest.raises(RuntimeError, match="live token count"):
        cv.check_capture(str(tmp_path))
    _capture(tmp_path, [6, 4], [6, 4], routers=[0, 2])
    with pytest.raises(RuntimeError, match="router record"):
        cv.check_capture(str(tmp_path))


def test_stop_server_signals_the_launcher_alone_before_cleaning_up_its_group(tmp_path):
    """SIGTERM to the whole group would kill the scheduler before its shutdown runs the route log's final read;
    only the launcher is signalled, and whatever outlives it is killed afterwards."""
    import os
    import subprocess
    import textwrap
    import time

    child = textwrap.dedent(f"""
        import os, signal, time
        signal.signal(signal.SIGTERM, lambda *a: (open({str(tmp_path / 'child-term')!r}, 'w').close(), os._exit(0)))
        time.sleep(60)
    """)
    leader = textwrap.dedent(f"""
        import os, signal, subprocess, sys, time
        kid = subprocess.Popen([sys.executable, "-c", {child!r}])
        open({str(tmp_path / 'child.pid')!r}, "w").write(str(kid.pid))
        signal.signal(signal.SIGTERM, lambda *a: (time.sleep(0.3), os._exit(0)))
        time.sleep(60)
    """)
    proc = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
    pid_file = tmp_path / "child.pid"
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.3)  # the child's handler is installed
    kid = int(pid_file.read_text())
    cv.stop_server(proc, timeout=10)
    assert proc.returncode == 0
    assert not (tmp_path / "child-term").exists()
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(kid, 0)
