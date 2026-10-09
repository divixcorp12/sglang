"""The RAM prefetch's hit rate by gate rank and margin, from InstrBuild job-trace events (CPU)."""

import importlib.util
import json
import os
import struct

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _module():
    spec = importlib.util.spec_from_file_location(
        "spec_margin", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gen(rank: int, margin: float) -> int:
    return (rank << 32) | struct.unpack("<I", struct.pack("<f", margin))[0]


def _event(kind, row, seq, expert, gen=0, c=0, group=0):
    return {"event": kind, "ns": 1, "row": row, "gen": gen, "seq": seq, "group": group, "a": expert, "b": 0, "c": c}


def _write(path, events, dropped=0):
    with open(path, "w") as f:
        f.write(json.dumps({"schema": 1}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({"dropped": dropped}) + "\n")


def test_unpack_gen_round_trips_rank_and_negative_margin():
    rank, margin = _module().unpack_gen(_gen(7, -0.375))
    assert (rank, margin) == (7, -0.375)


def test_a_use_joins_its_submit_by_target_row_source_seq_and_expert(tmp_path):
    m = _module()
    events = [
        _event("spec_submit", row=5, seq=40, expert=11, gen=_gen(2, 0.5)),
        _event("spec_land", row=5, seq=40, expert=11),
        _event("spec_submit", row=5, seq=41, expert=11, gen=_gen(8, -0.3)),
        _event("spec_land", row=5, seq=41, expert=11),
        # The demand record's seq is 44; c carries the entry's source seq 40.
        _event("spec_use", row=5, seq=44, expert=11, c=40),
    ]
    _write(tmp_path / "e.1.spec.jsonl", events)
    loaded, dropped = m.load([str(tmp_path / "e.1.spec.jsonl")])
    picks = sorted(m.join(loaded), key=lambda p: p["rank"])
    assert dropped == 0
    assert [(p["rank"], p["landed"], p["used"]) for p in picks] == [(2, True, True), (8, True, False)]
    result = m.bins(picks, top_k=6)
    assert result["inside_top_k"]["precision"] == 1.0
    assert result["outside_top_k"]["precision"] == 0.0


def test_main_refuses_a_trace_that_dropped_events(tmp_path, monkeypatch, capsys):
    m = _module()
    _write(tmp_path / "e.1.spec.jsonl", [_event("spec_submit", 1, 1, 1, gen=_gen(0, 1.0))], dropped=3)
    monkeypatch.setattr("sys.argv", ["spec_margin.py", str(tmp_path / "e.*.jsonl")])
    assert m.main() == 1
    assert "dropped" in capsys.readouterr().err


def test_timing_reports_the_read_the_lead_and_the_slack_of_used_reads():
    m = _module()
    picks = [
        {"submit_ns": 0, "land_ns": 2_000_000, "use_ns": 5_000_000},
        {"submit_ns": 0, "land_ns": 3_000_000, "use_ns": None},
        {"submit_ns": 0, "land_ns": None, "use_ns": None},
    ]
    t = m.timing(picks)
    assert t["read_ms (landed)"]["n"] == 2
    assert t["lead_ms submit->use (used)"] == {"n": 1, "p10": 5.0, "p50": 5.0, "p90": 5.0, "p99": 5.0, "min": 5.0}
    assert t["slack_ms land->use (used)"]["p50"] == 3.0
