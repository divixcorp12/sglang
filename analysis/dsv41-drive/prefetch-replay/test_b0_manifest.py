"""b0_manifest.py: header identity across sessions, sizes and checksums. CPU only."""

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


def _session(corpus: Path, name: str, *, rows=ROWS) -> bytes:
    d = corpus / name
    d.mkdir()
    (d / "residency.json").write_text(json.dumps({"rows": rows, "validity_rule": RULE}))
    payload = name.encode() * 5
    (d / "residency.snap.bin").write_bytes(payload)
    return payload


def test_manifest_copies_header_and_lists_sizes_and_sha256(tmp_path):
    payload = _session(tmp_path, "s1")
    _session(tmp_path, "s2")
    m = b0_manifest.write_manifest(
        str(tmp_path),
        commit="abc",
        recipe={"A": "1"},
        split={"train": ["s1"], "dev": ["s2"]},
        failed=["s9"],
        flagged=["s8"],
    )
    assert m == json.loads((tmp_path / "manifest.json").read_text())
    assert m["sessions"] == ["s1", "s2"]
    assert m["rows"] == ROWS and m["validity_rule"] == RULE
    assert (m["failed"], m["flagged"], m["commit"]) == (["s9"], ["s8"], "abc")
    entry = m["files"]["s1/residency.snap.bin"]
    assert entry == {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_manifest_refuses_sessions_with_different_row_metadata(tmp_path):
    _session(tmp_path, "s1")
    _session(tmp_path, "s2", rows=ROWS[:1])
    with pytest.raises(ValueError, match="differs"):
        b0_manifest.write_manifest(str(tmp_path), commit="abc", recipe={}, split={}, failed=[], flagged=[])
    assert not (tmp_path / "manifest.json").exists()
