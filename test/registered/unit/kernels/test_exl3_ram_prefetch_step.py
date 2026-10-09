"""The speculative step (spec 2026-10-08-dsv41-ram-prefetch-design, "The speculative thread"), driven on the
test's thread with spec_pump: a record that staged a CPU input feeds each group's ring; the step scores the target
row's gate, skips hot, mapped and earlier-pooled experts, keeps the layer's budget over both groups, reads its own
group's picks into the pool, and drops a candidate whose target record was served first (CPU, ChainSim)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup, same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import (
    HALVES,
    LOGITS,
    enable,
    forced,
    gate,
    load,
    prefetch_rig,
    trigger,
    write_x,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _landed(host, row, group=None):
    return sorted(
        e["expert"] for e in host.spec_pool(row) if e["state"] == "landed" and (group is None or e["group"] == group)
    )


def test_a_cpu_record_reads_the_next_rows_top_pick_into_the_pool_and_the_target_swaps_it(tmp_path):
    """Mutant: the speculative read publishing the mirror -- red on mapping(1)[2] before the swap."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        assert rig.host.spec_pump(0) and not rig.host.spec_pump(0)
        entry = next(e for e in rig.host.spec_pool(1) if e["state"] == "landed")
        assert entry["expert"] == 2
        assert rig.host.mapping(1)[2] == -1 and rig.sim.delta(1)[2] == []
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_landed"], c["spec_dropped"], c["spec_failed"]) == (1, 1, 0, 0)
        got, ref = rig.sim.read_slot(1, entry["slot"]), rig.setup.reference(1, [2])
        assert all(same_bytes(got[n], ref[n][0]) for n in got)
        rows = c["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["spec_used"] == 1 and c["rows_read"] == rows and rig.host.mapping(1)[2] == entry["slot"]
    finally:
        rig.host.stop()


def test_hot_mapped_and_earlier_pooled_experts_are_skipped(tmp_path):
    """2 is VRAM-hot, 4 mapped, 0 pooled for an earlier record: the pick is 1. Mutant: drop the hot skip -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        load(rig, 1, [4])
        rig.host.set_hot(1, [2])
        rig.host.spec_place(1, 0)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [0, 1]
    finally:
        rig.host.stop()


@pytest.mark.parametrize("per_layer, picks", [(1, [2]), (2, [2, 4])])
def test_the_layer_reads_per_layer_rows_in_margin_order(tmp_path, per_layer, picks):
    """Two candidates per token, budget 1 or 2. Mutant (gate_scorer.h): per_layer + 1 -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS, per_token=2, per_layer=per_layer)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == picks and rig.host.counters()["spec_issued"] == len(picks)
    finally:
        rig.host.stop()


@pytest.mark.parametrize("live, picks", [(2, [0, 2]), (1, [2])])
def test_each_live_token_adds_its_own_pick(tmp_path, live, picks):
    """A two-token row; token 0 prefers 2, token 1 prefers 0. With the table counting one live token,
    token 1's stale input is not scored."""
    rig = prefetch_rig(tmp_path, tokens=2)
    try:
        enable(rig, [LOGITS[0], [41, 21, 22, 23, 24, 25]], per_layer=2)
        trigger(rig, tokens=live)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == picks
    finally:
        rig.host.stop()


def test_a_candidate_whose_target_record_was_served_first_is_dropped(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        target = rig.sim.post(1, [])  # row 1's record, served before the step runs
        assert rig.host.pump() == 1 and rig.sim.wait_handled(target)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_dropped"], c["spec_issued"]) == (1, 0) and _landed(rig.host, 1) == []
    finally:
        rig.host.stop()


def test_a_record_of_another_row_or_an_older_target_record_leaves_the_job_live(tmp_path):
    """Staleness is the target row's own last record: row 1's record before the source, and row 0's after it, keep the
    job. Mutant: stale once the group handled the source's seq + 1 -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        older = rig.sim.post(1, [])
        assert rig.host.pump() == 1 and rig.sim.wait_handled(older)
        trigger(rig)
        other = rig.sim.post(0, [])
        assert rig.host.pump() == 1 and rig.sim.wait_handled(other)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_dropped"], c["spec_issued"]) == (0, 1) and _landed(rig.host, 1) == [2]
    finally:
        rig.host.stop()


def test_a_group_that_skips_the_record_still_gets_its_job(tmp_path):
    """Group 1 serves the trigger's only CPU lane; group 0 then finds the record's hot record lapped and skips it, yet
    scores it alike and reads its own pick. Mutant: offer the job only for a record the group serves -- red."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1, hot=True)
    try:
        enable(rig, [[30, 25, 40, 39, 22, 21, 23, 24]], per_token=2, per_layer=2)
        load(rig, 0, [5])
        write_x(rig, 0, 1)
        req = rig.sim.post(0, [5], captured=True, cpu_on=True)
        assert rig.host.pump_group(1) == 1
        rig.sim._write_hot(req.seq, (), req.seq + 16)  # a later post's hot record in its place
        overruns = rig.host.group_counters(0)["overruns"]
        assert rig.host.pump_group(0) == 1 and rig.host.group_counters(0)["overruns"] == overruns + 1
        assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 0) == [2] and _landed(rig.host, 1, 1) == [3]
    finally:
        rig.host.stop()


def test_a_failed_speculative_read_empties_its_entry_and_the_demand_reads_the_row(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        rig.host.inject_spec(fail=True)
        trigger(rig)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_failed"], c["spec_landed"], c["read_errors"]) == (1, 1, 0, 0)
        assert all(e["state"] == "empty" for e in rig.host.spec_pool(1))
        rows = c["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["rows_read"] == rows + 1 and c["spec_used"] == 0
    finally:
        rig.host.stop()


def test_a_reclaimed_entry_is_never_swapped_in_for_its_old_expert(tmp_path):
    """One entry, holding 0 for an earlier record; the step's pick 2 reclaims it. A forced miss on 0
    then reads 0's row."""
    rig = prefetch_rig(tmp_path, share=1)
    try:
        enable(rig, LOGITS)
        rig.host.spec_place(1, 0)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [2]
        rows = rig.host.counters()["rows_read"]
        forced(rig, 1, [0])
        c = rig.host.counters()
        assert c["rows_read"] == rows + 1 and c["spec_used"] == 0
        got, ref = rig.sim.read_slot(1, rig.host.mapping(1)[0]), rig.setup.reference(1, [0])
        assert all(same_bytes(got[n], ref[n][0]) for n in got)
    finally:
        rig.host.stop()


def test_a_full_ring_drops_jobs_and_never_blocks_the_service(tmp_path):
    """70 CPU records with no step run; the ring holds 64."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        for _ in range(70):
            trigger(rig)
        assert rig.host.counters()["spec_dropped"] == 6
    finally:
        rig.host.stop()


@pytest.mark.parametrize("per_layer, by_group", [(2, {0: [2], 1: [3]}), (1, {0: [2], 1: []})])
def test_both_groups_rank_alike_and_each_reads_only_its_own_experts(tmp_path, per_layer, by_group):
    """The trigger's only CPU lane is expert 5, group 1's, yet both groups get the job. Expert 2 is
    group 0's, 3 group 1's. Mutants: read every pick on any group -- red; skip an expert pooled for this same record
    (pooled_before ignoring the seq) -- red at budget 1, since group 1 then reads 3."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        enable(rig, [[30, 25, 40, 39, 22, 21, 23, 24]], per_token=2, per_layer=per_layer)
        trigger(rig, resident=5)
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        for g in (0, 1):
            assert _landed(rig.host, 1, g) == by_group[g]
            assert rig.host.group_counters(g)["spec_issued"] == len(by_group[g])
            lo, hi = HALVES[g][1]
            assert all(lo <= e["slot"] < hi for e in rig.host.spec_pool(1) if e["group"] == g)
    finally:
        rig.host.stop()


def test_records_without_a_cpu_lane_or_without_a_target_feed_no_job(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        load(rig, 0, [1])  # uncaptured: no CPU lane, nothing staged
        assert not rig.host.spec_pump(0)
        load(rig, 1, [3])
        write_x(rig, 1, 1)
        req = rig.sim.post(1, [3], captured=True, cpu_on=True)  # a CPU lane on row 1, which targets nothing
        assert rig.host.pump() == 1 and rig.sim.copy_wait(req)
        assert not rig.host.spec_pump(0)
    finally:
        rig.host.stop()


def test_enable_refuses_what_the_step_could_not_serve(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    rig = prefetch_rig(tmp_path / "a", pool=False)
    try:
        with pytest.raises(RuntimeError, match="reserve_spec_pool first"):
            enable(rig, LOGITS)
    finally:
        rig.host.stop()
    rig = prefetch_rig(tmp_path / "b")
    try:
        for kw, why in [
            (dict(per_layer=0), "per_layer must be in 1..8"),
            (dict(per_token=13), "per_token must be in 1..12"),
            (dict(top_k=7), "top_k must be in 1..6"),
        ]:
            with pytest.raises(RuntimeError, match=why):
                enable(rig, LOGITS, **kw)
        w, bias = gate(LOGITS)
        with pytest.raises(RuntimeError, match="one entry per streamed row"):
            rig.host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1], [-1, -1]]), w, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
        wide = torch.zeros((1, 6, 16), dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="hidden size 16 is not the CPU rows' 8"):
            rig.host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1]]), wide, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
        with pytest.raises(ValueError, match="one core list per NUMA group"):
            enable(rig, LOGITS, cores=[[], []])
        enable(rig, LOGITS)
        with pytest.raises(RuntimeError, match="already enabled"):
            enable(rig, LOGITS)
    finally:
        rig.host.stop()
    s = ram_miss_setup(tmp_path / "c", capacity=7)
    from sglang.kernels.ops.moe.expert_lease_block import wire_layout
    from sglang.kernels.ops.moe.expert_stream_transport import new_page

    host = attached_host(s, new_page(pin=False, wire=wire_layout(8)), k=3)
    try:
        host.reserve_spec_pool(2)
        w, bias = gate(LOGITS)
        with pytest.raises(RuntimeError, match="CPU experts on group 0"):
            host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1]]), w, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
    finally:
        host.stop()
