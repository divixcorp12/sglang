"""The RAM prefetch's InstrBuild observability (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Observability"):
spec_submit, spec_land and spec_use job-trace events that join on the source record's seq, and the scorer's records
scored and scoring time (CPU). spec_submit and spec_land: row = target row, seq = source record seq, c = source row.
spec_use: row, gen, seq = the demand record that swapped the entry in, c = the entry's source seq (for_seq)."""

import collections
import json
import math
import struct

from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, _served, enable, forced, load, prefetch_rig, trigger

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
        monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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
        monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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
        monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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
        monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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


def test_each_lane_leaves_its_kind_and_each_record_its_routed_and_lane_counts(tmp_path, monkeypatch):
    """lane_kind: a = the expert, b = its kind | 256 when spilled (destination -1), c = its lane. svc_routed, also for a
    lane-less record: a = the group's routed experts, b = all routed, c = all lanes, so expert 0 (routed, no lane) is
    VRAM-hot. Mutants: report the slot as b, drop the spill bit, or skip svc_routed on a lane-less record -- red."""
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig", capacity=9, share=1)  # room for the two misses' victims
    try:
        first = load(rig, 1, [4])
        req = rig.sim.post(
            1, [4, 2, 5], protect=[0, 4, 2, 5], dst=[0, 1, -1], captured=True, cpu_on=True, cpu_misses=True,
            forced_from=2,
        )
        _served(rig, req)
        touch = rig.sim.post(1, [], protect=[0])
        assert rig.host.pump() == 1 and rig.sim.wait_handled(touch)
    finally:
        rig.host.stop()
    events = _tier_events(tmp_path, {"lane_kind", "svc_routed"})
    lanes = sorted((e["seq"], e["a"], e["b"], e["c"]) for e in events if e["event"] == "lane_kind")
    assert lanes == [
        (first.seq, 4, LaneKind.MISS_GPU, 0),
        (req.seq, 2, LaneKind.MISS_CPU, 1),
        (req.seq, 4, LaneKind.HIT_CPU, 0),
        (req.seq, 5, LaneKind.MISS_CPU | 256, 2),
    ]
    routed = sorted((e["seq"], e["row"], e["a"], e["b"], e["c"]) for e in events if e["event"] == "svc_routed")
    assert routed == [(first.seq, 1, 1, 1, 1), (req.seq, 1, 4, 4, 3), (touch.seq, 1, 1, 1, 0)]


def _read_record(rig):
    (record,) = [r for r in rig.host.drain_trace() if r["kind"] == "demand" and r["status"] == "served" and r["rows"]]
    return record


def test_each_read_miss_leaves_a_miss_land_event_with_its_expert_and_the_pool_state_it_saw(tmp_path, monkeypatch):
    """miss_land (stage trace on): a = the expert, b = when its row landed, within the record's read, c = 1 for a miss
    whose row had landed in the pool when planned (pool serve would have copied it), 0 for one the pool lacked.
    Mutant: drop the pool state -- red on c."""
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
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


def test_with_pool_serve_a_pooled_miss_leaves_a_pool_point_and_the_read_one_its_own_land(tmp_path, monkeypatch):
    """A pool-served miss is not read: one pool_point (a = expert, b = the pool slot S streams from, c = the victim
    pooled in its place) instead of a miss_land; the read miss's miss_land carries its own row's landing although its
    read ordinal (0) is not its miss index (1). Mutant: take the stage record's row by miss index -- red on b; report
    the staging slot as b -- red on the slot."""
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        enable(rig, LOGITS, pool_serve=True)
        rig.host.enable_trace()
        slot = rig.host.spec_place(1, 4)
        before = rig.host.slot_info(1)
        req = load(rig, 1, [4, 2])
        record = _read_record(rig)
        spec = 4  # slot_info's kSpec
        (victim,) = [s for s, (b, a) in enumerate(zip(before, rig.host.slot_info(1))) if b[0] != spec and a[0] == spec]
    finally:
        rig.host.stop()
    events = _tier_events(tmp_path, {"miss_land", "pool_point", "svc_record"})
    by_kind = collections.defaultdict(list)
    for e in events:
        by_kind[e["event"]].append(e)
    (land,) = by_kind["miss_land"]
    assert (land["seq"], land["a"], land["c"]) == (req.seq, 2, 0)
    assert record["submit"] < land["b"] <= record["done"]
    (point,) = by_kind["pool_point"]
    assert (point["row"], point["gen"], point["seq"], point["a"], point["b"]) == (1, req.gen, req.seq, 4, slot)
    assert point["c"] == victim
    (svc,) = [e for e in by_kind["svc_record"] if e["seq"] == req.seq]
    assert 0 < svc["c"] <= svc["a"]


def test_a_demand_miss_on_a_full_tier_evicts_the_least_recently_used_expert_and_says_so(tmp_path, monkeypatch):
    """evict: a = the evicted expert (-1: a free slot), b = its age in use ticks, c = the expert moving in. Loading one
    expert at a time, the evictions take them in load order. Mutant: report the incoming expert as a -- red."""
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", json.dumps({"job": {"prefix": str(tmp_path / "jobs")}}))
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    order = [0, 1, 2, 3, 5]
    try:
        for expert in order:
            load(rig, 1, [expert])
    finally:
        rig.host.stop()
    events = _tier_events(tmp_path, {"evict"})
    assert [e["c"] for e in events] == order
    evicted = [(e["a"], e["c"]) for e in events if e["a"] >= 0]
    assert evicted, "the tier never filled"
    first = len(order) - len(evicted)
    assert evicted == list(zip(order, order[first:]))
    assert all(e["b"] > 0 for e in events if e["a"] >= 0)
    assert all(e["b"] == -1 for e in events if e["a"] < 0)
