"""The RAM prefetch's InstrBuild observability (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Observability"):
spec_submit, spec_land and spec_use job-trace events that join on the source record's seq, and the scorer's records
scored and scoring time (CPU). spec_submit and spec_land: row = target row, seq = source record seq, c = source row.
spec_use: row, gen, seq = the demand record that swapped the entry in, c = the entry's source seq (for_seq)."""

import collections
import json
import math
import struct

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


def test_spec_submit_carries_the_picks_rank_and_margin_in_gen(tmp_path, monkeypatch):
    """gen = rank << 32 | the margin's fp32 bits, for spec_margin.py. LOGITS' top_k 2 ranks 2 (40) then 4 (35): the
    pick is 2, rank 0, margin sqrt(40) - sqrt(35) in fp32. Mutant: emit 0 for gen -- red."""
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        enable(rig, LOGITS)
        trigger(rig)
        assert rig.host.spec_pump(0)
    finally:
        rig.host.stop()
    submit = next(e for e in _events(tmp_path) if e["event"] == "spec_submit")
    rank = submit["gen"] >> 32
    margin = struct.unpack("<f", struct.pack("<I", submit["gen"] & 0xFFFFFFFF))[0]
    f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
    expected = f32(f32(math.sqrt(40.0)) - f32(math.sqrt(35.0)))
    assert (submit["a"], rank) == (2, 0)
    assert margin == expected


def test_every_forced_miss_leaves_a_miss_expert_event_on_its_group(tmp_path, monkeypatch):
    """miss_expert: row, gen and seq of the record, a = the expert, b = its lane in the record. A hit leaves none. The
    group's trace is built with the tier, so the prefix is set before the rig."""
    monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig", capacity=9, share=1)  # room for two victims
    try:
        load(rig, 1, [4])
        req = forced(rig, 1, [2, 4, 5])
    finally:
        rig.host.stop()
    files = list(tmp_path.glob("jobs.*.exl3-tier0.*.jsonl"))
    assert len(files) == 1
    events = [e for e in map(json.loads, files[0].read_text().splitlines()[1:-1]) if e["event"] == "miss_expert"]
    assert sorted((e["row"], e["gen"], e["seq"], e["a"], e["b"]) for e in events) == [
        (1, req.gen, req.seq, 2, 0), (1, req.gen, req.seq, 5, 2)]


def _tier_events(tmp_path, kinds):
    files = list(tmp_path.glob("jobs.*.exl3-tier0.*.jsonl"))
    assert len(files) == 1
    return [e for e in map(json.loads, files[0].read_text().splitlines()[1:-1]) if e["event"] in kinds]


def _read_record(rig):
    (record,) = [r for r in rig.host.drain_trace() if r["kind"] == "demand" and r["status"] == "served" and r["rows"]]
    return record


def test_each_read_miss_leaves_a_miss_land_event_with_its_expert_and_the_pool_state_it_saw(tmp_path, monkeypatch):
    """miss_land (stage trace on): a = the expert, b = when its row landed, within the record's read, c = 1 for a miss
    whose row had landed in the pool when planned (pool serve would have copied it), 0 for one the pool lacked.
    Mutant: drop the pool state -- red on c."""
    monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        rig.host.enable_trace()
        rig.host.spec_place(1, 4)
        req = load(rig, 1, [2, 4])
        record = _read_record(rig)
    finally:
        rig.host.stop()
    events = _tier_events(tmp_path, {"miss_land"})
    assert sorted((e["row"], e["gen"], e["seq"], e["a"], e["c"]) for e in events) == [
        (1, req.gen, req.seq, 2, 0), (1, req.gen, req.seq, 4, 1)]
    for e in events:
        assert record["submit"] < e["b"] <= record["done"], e


def test_with_pool_serve_a_copied_miss_leaves_pool_copy_events_and_the_read_one_its_own_land(tmp_path, monkeypatch):
    """A copied miss is not read: pool_copy_start and pool_copy_end (a = expert, b = lane, c = miss index) instead of a
    miss_land; the read miss's miss_land carries its own row's landing although its read ordinal (0) is not its miss
    index (1). Mutant: take the stage record's row by miss index -- red on b."""
    monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        enable(rig, LOGITS, pool_serve=True)
        rig.host.enable_trace()
        rig.host.spec_place(1, 4)
        req = load(rig, 1, [4, 2])
        record = _read_record(rig)
    finally:
        rig.host.stop()
    events = _tier_events(tmp_path, {"miss_land", "pool_copy_start", "pool_copy_end"})
    by_kind = collections.defaultdict(list)
    for e in events:
        by_kind[e["event"]].append(e)
    (land,) = by_kind["miss_land"]
    assert (land["seq"], land["a"], land["c"]) == (req.seq, 2, 0)
    assert record["submit"] < land["b"] <= record["done"]
    (start,), (end,) = by_kind["pool_copy_start"], by_kind["pool_copy_end"]
    for e in (start, end):
        assert (e["row"], e["gen"], e["seq"], e["a"], e["b"], e["c"]) == (1, req.gen, req.seq, 4, 0, 0)
    assert start["ns"] <= end["ns"]
