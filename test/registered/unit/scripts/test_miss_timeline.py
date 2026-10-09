import importlib.util
import json
import os
import sys

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark", "miss_timeline.py"
)
_spec = importlib.util.spec_from_file_location("miss_timeline", _PATH)
mt = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.dirname(_PATH))
_spec.loader.exec_module(mt)

MS = 1_000_000


def _ev(kind, ms, row=3, gen=0, seq=0, group=-1, a=0, b=0, c=0):
    return {"event": kind, "ns": int(ms * MS), "row": row, "gen": gen, "seq": seq, "group": group, "a": a, "b": b, "c": c}


def _write(path, events):
    with open(path, "w") as f:
        f.write(json.dumps({"schema": 1}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")


@pytest.fixture
def trace(tmp_path):
    # One record on group 0: one CPU hit (seq 10) and two NVMe misses that land at 2 and 2.5 ms, run as two jobs
    # (seq 11, 12). The hit job runs until 3 ms, so the first miss job waits 1 ms behind it. A draft job runs 5-5.5.
    _write(tmp_path / "events.1.exl3-copy-eng.0.jsonl", [
        _ev("copy_submit", 0, gen=7, seq=10, group=0, a=12, b=2, c=0b1),
        _ev("copy_dma_observed", 1, gen=7, seq=10, group=0),
        _ev("group_done", 5.2, gen=7, seq=10, group=0, a=12),
    ])
    _write(tmp_path / "events.1.exl3-cpu-exp0.1.jsonl", [
        _ev("cpu_submit", 0, seq=10, a=0, b=1), _ev("cpu_start", 0.1, seq=10, a=0, b=1),
        _ev("cpu_end", 3, seq=10, a=0, b=1),
        _ev("cpu_submit", 2, seq=11, a=1, b=1), _ev("cpu_start", 3, seq=11, a=1, b=1),
        _ev("cpu_end", 4, seq=11, a=1, b=1),
        _ev("cpu_submit", 2.5, seq=12, a=1, b=1), _ev("cpu_start", 4, seq=12, a=1, b=1),
        _ev("cpu_end", 5, seq=12, a=1, b=1),
        _ev("draft_start", 5, seq=1), _ev("draft_end", 5.5, seq=1),
    ])
    return tmp_path


def test_record_timeline(trace):
    recs = mt.records(str(trace))
    assert len(recs) == 1
    r = recs[0]
    assert r["misses"] == 2 and r["batches"] == 2 and r["hits"] == 1
    assert r["land_first_ms"] == pytest.approx(2.0)
    assert r["land_last_ms"] == pytest.approx(2.5)
    assert r["first_wait_ms"] == pytest.approx(1.0)  # submitted at 2, started at 3
    assert r["hit_blocked_ms"] == pytest.approx(1.0)  # the hit job ran past the first landing by 1 ms
    assert r["idle_before_land_ms"] == pytest.approx(0.1)  # busy 0.1-2 with the hit job; idle 0-0.1
    assert r["tail_ms"] == pytest.approx(2.5)  # last miss job ends 5, last row landed 2.5
    assert r["miss_ms_per_lane"] == pytest.approx(1.0)
    assert r["last"] == "cpu"  # the miss chain (5) ends after the DMA (1)
    # The last row landed at 2.5 and its job started at 4: 0.5 ms behind the hit job, 1 ms behind the first miss.
    assert r["last_wait_ms"] == pytest.approx(1.5)
    assert r["last_wait_hit_ms"] == pytest.approx(0.5)
    assert r["last_wait_draft_ms"] == pytest.approx(0.0)


def test_layer_gain_takes_the_later_group(trace):
    recs = mt.records(str(trace))
    # One group: the layer ends 0.5 ms sooner when the hit job's share of the last wait goes.
    assert mt.layer_gain_ms(recs) == {7: pytest.approx(0.5)}


def test_summary_counts_together_landings(trace):
    s = mt.summarize(mt.records(str(trace)))
    assert s["records"] == 1
    assert s["multi_miss_one_batch_share"] == 0.0
    assert s["hit_blocked_share"] == 1.0
    assert s["last_side"] == {"cpu": 1.0}


def test_busy_union_merges_overlaps():
    assert mt.busy_within([(0, 2), (1, 3), (5, 6)], 0, 10) == 4
    assert mt.busy_within([(0, 2), (5, 6)], 1, 5.5) == pytest.approx(1.5)
