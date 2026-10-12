"""Tests for r1_stage: atomic DONE records, resume matching, out-root guard, invocation log."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_stage  # noqa: E402


def _ctx(out: str) -> r1_stage.StageCtx:
    return r1_stage.StageCtx(
        out=out, dataset="d", corpus="c", model_path="m", b1_head="h", device="cpu", smoke=True, plan_sha=None
    )


def test_sha256_json_is_order_independent(tmp_path):
    assert r1_stage.sha256_json({"a": 1, "b": [1, 2]}) == r1_stage.sha256_json({"b": [1, 2], "a": 1})
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert r1_stage.sha256_file(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_done_state_missing_match_stale(tmp_path):
    ctx = _ctx(str(tmp_path))
    ins = {"x": "1"}
    assert r1_stage.done_state(ctx, "s1", inputs=ins) == "missing"
    r1_stage.write_done(ctx, "s1", inputs=ins, outputs={"o": "2"}, info={"n": 3})
    assert os.path.isfile(os.path.join(r1_stage.stage_dir(ctx, "s1"), "DONE.json"))
    assert r1_stage.done_state(ctx, "s1", inputs=ins) == "match"
    assert r1_stage.done_state(ctx, "s1", inputs={"x": "9"}) == "stale"
    assert not [n for n in os.listdir(r1_stage.stage_dir(ctx, "s1")) if ".tmp" in n]


def test_guard_out_root():
    root = "/mnt/nvme1/nvfp4-work"
    assert r1_stage.guard_out_root(root + "/b1-r1-20261013", root=root) == root + "/b1-r1-20261013"
    for bad in ("/tmp/x", root, root + "/b1-run-20261012", root + "/b1-run-20261012/sub", root + "/../nvme0/x"):
        with pytest.raises(ValueError):
            r1_stage.guard_out_root(bad, root=root)


def test_guard_blocks_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", root / "link")
    with pytest.raises(ValueError):
        r1_stage.guard_out_root(str(root / "link"), root=str(root))


def test_invocation_log_appends(tmp_path):
    out = str(tmp_path / "run")
    r1_stage.log_invocation(out, argv=["a", "stage1"], exit_code=0, started="t0", ended="t1")
    r1_stage.log_invocation(out, argv=["a", "audit"], exit_code=2, started="t2", ended="t3")
    with open(os.path.join(out, "invocations.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    assert [r["exit_code"] for r in rows] == [0, 2]
    assert rows[1]["argv"] == ["a", "audit"]
