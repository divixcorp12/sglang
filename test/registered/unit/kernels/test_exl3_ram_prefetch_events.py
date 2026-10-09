"""The RAM prefetch's InstrBuild observability (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Observability"):
spec_submit, spec_land and spec_use job-trace events that join on the source record's seq, and the scorer's records
scored and scoring time (CPU). spec_submit and spec_land: row = target row, seq = source record seq, c = source row.
spec_use: row, gen, seq = the demand record that swapped the entry in, c = the entry's source seq (for_seq)."""

import collections
import json

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, load, prefetch_rig, trigger

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _events(tmp_path):
    files = list(tmp_path.glob("jobs.*.exl3-spec0.*.jsonl"))
    assert len(files) == 1
    return list(map(json.loads, files[0].read_text().splitlines()[1:-1]))


def _counts(events):
    return dict(collections.Counter(e["event"] for e in events))


def test_a_speculative_row_leaves_submit_land_and_use_events_and_the_scorer_counts(tmp_path, monkeypatch):
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        # Read when enable_ram_prefetch builds each group's trace; the rig's other engines were built without it.
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        enable(rig, LOGITS)
        trig = trigger(rig)
        assert rig.host.spec_pump(0)
        slot = next(e["slot"] for e in rig.host.spec_pool(1) if e["expert"] == 2)
        use = forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["spec_scored"] == 1 and c["spec_score_ns"] > 0
    finally:
        rig.host.stop()
    events = _events(tmp_path)
    assert _counts(events) == {"spec_submit": 1, "spec_land": 1, "spec_use": 1}
    by_kind = {e["event"]: e for e in events}
    for kind in ("spec_submit", "spec_land"):
        e = by_kind[kind]
        assert (e["row"], e["seq"], e["group"], e["a"], e["b"], e["c"]) == (1, trig.seq, 0, 2, slot, 0)
    e = by_kind["spec_use"]
    assert (e["row"], e["seq"], e["gen"], e["a"], e["b"], e["c"]) == (1, use.seq, use.gen, 2, slot, trig.seq)


def test_the_events_still_join_when_another_rows_record_posts_between_source_and_demand(tmp_path, monkeypatch):
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig", rows=3)
    try:
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        enable(rig, LOGITS)
        trig = trigger(rig)
        other = rig.sim.post(2, [])
        assert rig.host.pump() == 1 and rig.sim.wait_handled(other)
        assert rig.host.spec_pump(0)
        use = forced(rig, 1, [2])
    finally:
        rig.host.stop()
    events = _events(tmp_path)
    assert _counts(events) == {"spec_submit": 1, "spec_land": 1, "spec_use": 1}
    by_kind = {e["event"]: e for e in events}
    assert by_kind["spec_submit"]["seq"] == by_kind["spec_land"]["seq"] == trig.seq
    assert by_kind["spec_use"]["c"] == trig.seq and by_kind["spec_use"]["seq"] == use.seq


def test_a_failed_speculative_read_emits_submit_but_no_land(tmp_path, monkeypatch):
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        enable(rig, LOGITS)
        load(rig, 0, [5])
        rig.host.inject_fault(part=0, part_error=5)
        trig = trigger(rig)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_failed"], c["spec_landed"]) == (1, 1, 0)
    finally:
        rig.host.stop()
    events = _events(tmp_path)
    assert _counts(events) == {"spec_submit": 1}
    assert (events[0]["row"], events[0]["seq"], events[0]["c"]) == (1, trig.seq, 0)
