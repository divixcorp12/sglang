"""b0_manifest.py: block layout, session -> block mapping, header identity, sizes and checksums. CPU only."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_manifest  # noqa: E402

ROWS = [
    {"layer_id": 3, "stratum": "early", "gate_has_bias": True},
    {"layer_id": 4, "stratum": "early", "gate_has_bias": False},
]
RULE = "invalid: no target row"


def _block(corpus: Path, block_id: str, *, rows=ROWS) -> bytes:
    d = corpus / "blocks" / block_id
    d.mkdir(parents=True)
    (d / "residency.json").write_text(json.dumps({"rows": rows, "validity_rule": RULE}))
    payload = block_id.encode() * 5
    (d / "residency.snap.bin").write_bytes(payload)
    return payload


def _session(corpus: Path, sid: str) -> None:
    d = corpus / sid
    d.mkdir()
    (d / "results.jsonl").write_text("{}\n")
    (d / "DONE.json").write_text("{}")


def _write(tmp_path, session_block, warm_start=()):
    return b0_manifest.write_manifest(
        str(tmp_path),
        commit="abc",
        recipe={"A": "1"},
        split={"train": ["s1"], "dev": ["s2"]},
        failed=["s9"],
        flagged=["s8"],
        session_block=session_block,
        warm_start=list(warm_start),
    )


def test_manifest_maps_sessions_to_blocks_and_lists_sizes_and_sha256(tmp_path):
    payload = _block(tmp_path, "b0")
    _block(tmp_path, "b1")
    for sid in ("s1", "s2", "s3"):
        _session(tmp_path, sid)
    m = _write(tmp_path, {"s2": "b1", "s1": "b0", "s3": "b0"}, warm_start=["s1", "s2"])
    assert m == json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema"] == 2
    assert m["sessions"] == ["s1", "s2", "s3"]
    assert m["blocks"] == {"b0": ["s1", "s3"], "b1": ["s2"]}
    assert m["session_block"] == {"s1": "b0", "s2": "b1", "s3": "b0"}
    assert m["warm_start"] == ["s1", "s2"]
    assert m["rows"] == ROWS and m["validity_rule"] == RULE
    assert (m["failed"], m["flagged"], m["commit"]) == (["s9"], ["s8"], "abc")
    entry = m["files"]["blocks/b0/residency.snap.bin"]
    assert entry == {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert "s1/DONE.json" in m["files"]


def test_manifest_omits_block_dirs_without_a_done_session(tmp_path):
    _block(tmp_path, "b0")
    _block(tmp_path, "b1")
    _session(tmp_path, "s1")
    m = _write(tmp_path, {"s1": "b0"})
    assert list(m["blocks"]) == ["b0"]
    assert not any(k.startswith("blocks/b1/") for k in m["files"])


def test_manifest_refuses_blocks_with_different_row_metadata(tmp_path):
    _block(tmp_path, "b0")
    _block(tmp_path, "b1", rows=ROWS[:1])
    for sid in ("s1", "s2"):
        _session(tmp_path, sid)
    with pytest.raises(ValueError, match="differs"):
        _write(tmp_path, {"s1": "b0", "s2": "b1"})
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_refuses_an_empty_mapping(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        _write(tmp_path, {})
