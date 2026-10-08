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
