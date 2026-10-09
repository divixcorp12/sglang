"""The RAM prefetch's InstrBuild observability (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Observability"):
spec_submit, spec_land and spec_use job-trace events carrying the record's seq and row, so the job-trace joins
attribute them, and the scorer's records scored and scoring time (CPU)."""

import json

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, prefetch_rig, trigger

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


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
    files = list(tmp_path.glob("jobs.*.exl3-spec0.*.jsonl"))
    assert len(files) == 1
    events = {e["event"]: e for e in map(json.loads, files[0].read_text().splitlines()[1:-1])}
    target = (trig.seq + 1) & 0xFFFFFFFF
    for kind in ("spec_submit", "spec_land"):
        e = events[kind]
        assert (e["row"], e["seq"], e["group"], e["a"], e["b"]) == (1, target, 0, 2, slot)
    e = events["spec_use"]
    assert (e["row"], e["seq"], e["gen"], e["a"], e["b"]) == (1, use.seq, use.gen, 2, slot)
    assert use.seq == target
