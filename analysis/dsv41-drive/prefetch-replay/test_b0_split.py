"""b0_split.py: the seeded sample and the whole-session split. CPU only."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_split  # noqa: E402


def test_the_split_is_deterministic_180_45_30_45_and_shares_no_session():
    ids = [f"s{i}" for i in range(300)]
    a = b0_split.split_sessions(ids, seed=7)
    assert a == b0_split.split_sessions(list(reversed(ids)), seed=7)
    assert [len(a[k]) for k in ("train", "dev", "calib", "locked_test")] == [180, 45, 30, 45]
    flat = [s for part in a.values() for s in part]
    assert sorted(flat) == sorted(ids)


def test_the_sample_is_seeded_and_drawn_from_the_file(tmp_path):
    path = tmp_path / "sessions.jsonl"
    path.write_text("".join(json.dumps({"session_id": f"s{i}"}) + "\n" for i in range(50)))
    a = b0_split.sample_sessions(str(path), n=10, seed=1)
    assert a == b0_split.sample_sessions(str(path), n=10, seed=1)
    assert len({r["session_id"] for r in a}) == 10
    assert a != b0_split.sample_sessions(str(path), n=10, seed=2)
