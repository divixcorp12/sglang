"""b1_dataset.py main on test_b0_integrity's synthetic block: manifest counts, shard round-trip, the --out guard.
CPU only."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_dataset  # noqa: E402
import test_b0_integrity as t0  # noqa: E402
from b1_rows import read_groups, read_x  # noqa: E402


def _setup(tmp_path, monkeypatch):
    block = t0._block(tmp_path / "b000")
    manifest = {"split": {"train": ["s1"], "dev": ["s2"], "calib": [], "locked_test": []},
                "warm_start": ["s2"], "flagged": ["s2"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    cell = {"seq": 0, "router_record": 0, "layer_id": 19, "token": 0, "computed": [], "captured": []}
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps({"per_session": [{"out_dir": block, "mismatches": [cell]}]}))
    monkeypatch.setattr(b1_dataset.b0, "_gates", lambda model_path, rows: t0._gates())
    return block, manifest_path, integrity_path


def _argv(tmp_path, block, manifest_path, integrity_path, *extra):
    return ["model", block, "--manifest", str(manifest_path), "--integrity", str(integrity_path),
            "--out", str(tmp_path / "out"), *extra]


def test_main_writes_shards_and_a_manifest_with_the_hand_counted_rows(tmp_path, monkeypatch):
    block, manifest_path, integrity_path = _setup(tmp_path, monkeypatch)
    assert b1_dataset.main(_argv(tmp_path, block, manifest_path, integrity_path, "--allow-any-out")) == 0
    out = tmp_path / "out"
    result = json.loads((out / "dataset.json").read_text())
    counts = result["counts"]
    # s1 (train): 2 live tokens, token 0 dropped; s2 (dev): 1 live token. Routed misses are experts 5 and 4.
    assert counts["train"]["1"] == {"rows": 1, "groups": 1, "eligible_rows": 2, "dropped_cells": 1}
    assert counts["dev"]["1"] == {"rows": 1, "groups": 1, "eligible_rows": 2, "dropped_cells": 0}
    for split in ("train", "dev"):
        for h in ("2", "4"):
            assert counts[split][h] == {"rows": 0, "groups": 0, "eligible_rows": 0, "dropped_cells": 0}
    assert counts["calib"]["1"]["groups"] == counts["locked_test"]["4"]["groups"] == 0
    assert result["warmup_forwards"] == 0 and result["dropped_cells"] == 1
    assert result["sessions_by_split"]["train"] == ["s1"] and result["flagged"] == ["s2"]
    assert result["manifest_sha256"] == b1_dataset._sha256(str(manifest_path))
    (g,) = read_groups(out, split="train", horizon=1)
    assert g.session == "s1" and g.tokens.tolist() == [1] and not g.flagged
    assert read_x(out, split="train", session="s1").shape == (1, t0.HIDDEN)
    (g,) = read_groups(out, split="dev", horizon=1)
    assert g.warm_start and g.flagged


def test_out_off_nvme_is_refused_without_allow_any_out(tmp_path, monkeypatch):
    block, manifest_path, integrity_path = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        b1_dataset.main(_argv(tmp_path, block, manifest_path, integrity_path))
    assert not (tmp_path / "out").exists()


def test_a_session_missing_from_the_manifest_is_refused(tmp_path, monkeypatch):
    block, manifest_path, integrity_path = _setup(tmp_path, monkeypatch)
    manifest_path.write_text(json.dumps({"split": {"train": ["s1"]}, "warm_start": [], "flagged": []}))
    with pytest.raises(ValueError, match="s2"):
        b1_dataset.main(_argv(tmp_path, block, manifest_path, integrity_path, "--allow-any-out"))
