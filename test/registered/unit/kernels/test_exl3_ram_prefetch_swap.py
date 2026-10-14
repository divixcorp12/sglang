"""Swap-on-use (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Swap on use"): a forced CPU miss whose expert
landed in the row's pool takes its victim as today, maps the pool slot, gives the victim to the pool, reads nothing and
goes to the CPU at once. spec_place lands a pool row as a speculative read would (CPU, ChainSim)."""

import time

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import piece_word
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ALL_PIECES
from sglang.test.dsv41_ram_miss_fixtures import same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, load, prefetch_rig, trigger

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FREE, READY, SPEC = 0, 2, 4

PROMOTE_COUNTERS = (
    "spec_promoted_early",
    "spec_promote_no_delta",
    "spec_promote_no_room",
    "spec_promote_no_victim",
    "spec_promote_victim_routed",
    "spec_promote_used",
    "spec_promote_reading",
    "spec_publish_backstop",
    "spec_promote_row_busy",
)
EIGHT_LOGITS = [[30, 25, 40, 22, 35, 21, 28, 27]]


def _entry(host, row, expert):
    return next((e for e in host.spec_pool(row) if e["expert"] == expert), None)


def _pooled_victim(before, after):
    """The one slot a swap gave the pool: not SPEC in `before` (slot_info), SPEC in `after`."""
    (victim,) = [s for s, (b, a) in enumerate(zip(before, after)) if b[0] != SPEC and a[0] == SPEC]
    return victim


def _same_row(rig, row, slot, expert):
    got, ref = rig.sim.read_slot(row, slot), rig.setup.reference(row, [expert])
    return all(same_bytes(got[name], ref[name][0]) for name in got)


def test_a_pool_row_is_invisible_to_the_device_until_a_forced_miss_swaps_it_in(tmp_path):
    """No read, the pool slot mapped by the record's delta, the free victim pooled in its place, the CPU job at once,
    bytes equal to a fresh read. Mutant: spec_place publishing the mirror -- red on the first mapping check."""
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
        assert rig.host.mapping(1)[4] == -1 and rig.sim.delta(1)[2] == []
        rows, layer_rows = rig.host.counters()["rows_read"], rig.host.layer_rows()
        before = rig.host.slot_info(1)
        req = forced(rig, 1, [4])
        assert req.kinds == [LaneKind.MISS_CPU] and req.slots == [-1]
        counters = rig.host.counters()
        assert counters["spec_used"] == 1
        assert counters["rows_read"] == rows and rig.host.layer_rows() == layer_rows, "a pooled miss was read"
        assert rig.host.mapping(1)[4] == slot and rig.sim.delta(1)[2] == [(4, slot)]
        assert rig.host.slot_info(1)[slot][0] == READY
        victim = _pooled_victim(before, rig.host.slot_info(1))
        assert before[victim][0] == FREE, "the victim is a free slot (nothing loaded)"
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        calls = rig.host.test_kernel_calls()
        assert calls[-1]["slots"] == [slot] and calls[-1]["accumulate"] is False
        assert _same_row(rig, 1, slot, 4)
    finally:
        rig.host.stop()


def test_a_record_whose_misses_all_came_from_the_pool_counts_served_and_reads_nothing(tmp_path):
    """The stage record says served, so the counters do too; rows_read still counts only real reads."""
    rig = prefetch_rig(tmp_path)
    try:
        rig.host.spec_place(1, 4)
        before = rig.host.counters()
        forced(rig, 1, [4])
        after = rig.host.counters()
        assert after["served"] == before["served"] + 1 and after["touch_only"] == before["touch_only"]
        assert after["rows_read"] == before["rows_read"]
    finally:
        rig.host.stop()


def test_a_record_served_wholly_from_the_pool_stamps_a_zero_width_read(tmp_path):
    """No read ran, but the stage record carries its read stamps at the reservation, in order. Mutant: leave them 0 --
    red."""
    rig = prefetch_rig(tmp_path)
    try:
        rig.host.enable_trace()
        rig.host.spec_place(1, 4)
        forced(rig, 1, [4])
        (record,) = [r for r in rig.host.drain_trace() if r["kind"] == "demand" and r["status"] == "served"]
        assert record["rows"] == 0
        assert record["reserved"] > 0
        for stamp in ("submit", "first_cqe", "last_cqe", "pack_start", "pack_end"):
            assert record[stamp] == record["reserved"], stamp
        assert record["observed"] <= record["reserved"] <= record["mapped"] <= record["done"]
    finally:
        rig.host.stop()


def test_a_record_served_wholly_from_the_pool_counts_as_a_demand_read(tmp_path):
    """inject(delay_after_demands=1) delays the second demand read: the pooled record is the first. Mutant: count only
    reads that ran -- the later read is not delayed (red)."""
    rig = prefetch_rig(tmp_path)
    try:
        rig.host.spec_place(1, 4)
        forced(rig, 1, [4])
        rig.host.inject(delay_s=0.4, delay_after_demands=1)
        start = time.monotonic()
        load(rig, 0, [5])
        assert time.monotonic() - start >= 0.3
    finally:
        rig.host.stop()


def test_the_swap_still_evicts_its_victim_in_the_delta(tmp_path):
    """Mutant: skip the victim's eviction entry in the forced loop -- red on the delta."""
    rig = prefetch_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        victim = rig.host.mapping(1)[0]  # 0 is the LRU of the two
        before, evictions = rig.host.slot_info(1), rig.host.counters()["evictions"]
        forced(rig, 1, [4])
        assert rig.sim.delta(1)[2] == [(0, -1), (4, slot)]
        assert rig.host.mapping(1)[0] == -1 and rig.host.mapping(1)[4] == slot
        assert _pooled_victim(before, rig.host.slot_info(1)) == victim
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        assert rig.host.counters()["evictions"] == evictions + 1
    finally:
        rig.host.stop()


def test_a_record_mixing_a_pooled_and_a_read_forced_miss_sends_the_pooled_one_first(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        rows = rig.host.counters()["rows_read"]
        req = forced(rig, 1, [4, 2])
        assert req.kinds == [LaneKind.MISS_CPU, LaneKind.MISS_CPU]
        calls = rig.host.test_kernel_calls()[-2:]
        assert calls[0]["slots"] == [slot] and calls[0]["accumulate"] is False
        assert calls[1]["slots"] == [rig.host.mapping(1)[2]] and calls[1]["accumulate"] is True
        assert rig.host.counters()["rows_read"] == rows + 1 and rig.host.counters()["spec_used"] == 1
        assert _same_row(rig, 1, rig.host.mapping(1)[2], 2) and _same_row(rig, 1, slot, 4)
    finally:
        rig.host.stop()


def test_a_gpu_miss_on_a_pooled_expert_is_read_and_the_pool_row_stays(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        req = load(rig, 1, [4])
        assert req.kinds == [LaneKind.MISS_GPU]
        assert rig.host.mapping(1)[4] not in (-1, slot)
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
        assert rig.host.counters()["spec_used"] == 0
    finally:
        rig.host.stop()


def test_a_two_group_swap_stays_in_the_experts_home_range(tmp_path):
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        slot = rig.host.spec_place(1, 3)  # expert 3 is group 1's
        assert 4 <= slot < 8
        forced(rig, 1, [3])
        assert rig.host.mapping(1)[3] == slot
        assert rig.host.group_counters(1)["spec_used"] == 1 and rig.host.group_counters(0)["spec_used"] == 0
    finally:
        rig.host.stop()


def test_a_gpu_miss_beside_a_pooled_forced_miss_publishes_its_pieces(tmp_path):
    """forced_from=1 on [GPU miss, pooled miss]: only the GPU miss is read, its lane's PieceMask word carries the
    record's generation with every piece (the compacted read's lane list), and the pooled lane gets no word."""
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        rows = rig.host.counters()["rows_read"]
        req = rig.sim.post(1, [2, 4], captured=True, cpu_on=True, forced_from=1)
        assert req.kinds == [LaneKind.MISS_GPU, LaneKind.MISS_CPU] and req.slots[1] == -1
        empty = rig.sim.piece_word(req, 1)  # the pooled lane's word as posted, before any read
        rig.host.pump()
        assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)
        assert rig.sim.piece_word(req, 0) == piece_word(req.gen, ALL_PIECES)
        assert rig.sim.piece_word(req, 1) == empty
        counters = rig.host.counters()
        assert counters["rows_read"] == rows + 1 and counters["spec_used"] == 1
        assert rig.host.mapping(1)[4] == slot and _same_row(rig, 1, slot, 4)
        assert _same_row(rig, 1, req.slots[0], 2), "the GPU miss's staging slot holds its row"
        assert rig.host.test_kernel_calls()[-1]["slots"] == [slot]
    finally:
        rig.host.stop()


def test_a_swapped_victim_joins_the_pool_only_once_the_rows_delta_is_published(tmp_path):
    """Two groups report the record: group 1 swaps first, but its victim (expert 1's slot, evicted in the delta) stays
    out of the pool's claimable entries until group 0, the last reporter, publishes the delta. Mutant: release the
    victim in the swap itself -- red on the first pool check."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        load(rig, 1, [1])
        load(rig, 1, [5])
        slot = rig.host.spec_place(1, 3)
        victim = rig.host.mapping(1)[1]  # group 1's LRU
        tag = rig.sim.delta(1)[0]
        req = forced(rig, 1, [3, 2], serve=False)
        assert rig.host.pump_group(1) == 1
        assert rig.sim.delta(1)[0] == tag, "group 1 alone published the delta"
        pool = [e for e in rig.host.spec_pool(1) if e["group"] == 1]
        assert pool == [{"group": 1, "slot": victim, "state": "swapped", "expert": -1}]
        assert rig.host.pump_group(0) == 1
        assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)
        assert rig.sim.delta(1)[0] != tag and (1, -1) in rig.sim.delta(1)[2] and (3, slot) in rig.sim.delta(1)[2]
        pool = [e for e in rig.host.spec_pool(1) if e["group"] == 1]
        assert pool == [{"group": 1, "slot": victim, "state": "empty", "expert": -1}]
    finally:
        rig.host.stop()


def _promote_rig(tmp_path, logits=LOGITS, **kwargs):
    rig = prefetch_rig(tmp_path, **kwargs)
    try:
        enable(rig, logits, promote=True)
    except BaseException:
        rig.host.stop()
        raise
    return rig


def _promoted(rig):
    """Row 1 holds 0 and 1 with its delta pending, 4 lands in its pool, then row 0's record claims row 1's delta.
    Returns (pool slot of 4, victim slot of expert 0, the delta tag before the claim)."""
    load(rig, 1, [0, 1])
    slot = rig.host.spec_place(1, 4)
    victim = rig.host.mapping(1)[0]  # 0 is the LRU of the two
    tag = rig.sim.delta(1)[0]
    load(rig, 0, [5])
    return slot, victim, tag


def test_a_landed_row_is_mapped_into_the_pending_delta_and_the_pool_entry_is_released(tmp_path):
    """Row 1's delta stays unwritten until row 0's next record, which maps the landed 4 and evicts the LRU victim, then
    the swapped entry empties and the victim slot is pooled. Mutant: write the delta at the end of row 1's own record
    -- red on the unchanged tag; skip release_swapped after a promotion -- red on the pool state."""
    rig = _promote_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        victim = rig.host.mapping(1)[0]
        tag = rig.sim.delta(1)[0]
        with pytest.raises(AssertionError):
            rig.sim.post(1, [2])  # the device would wait: row 1's delta is not written yet
        load(rig, 0, [5])
        got_tag, _, entries = rig.sim.delta(1)
        assert got_tag != tag and entries[-2:] == [(0, -1), (4, slot)]
        assert rig.host.mapping(1)[4] == slot and rig.host.mapping(1)[0] == -1
        assert rig.host.slot_info(1)[slot][0] == READY
        assert _entry(rig.host, 1, 4) is None
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        assert rig.host.counters()["spec_promoted_early"] == 1
    finally:
        rig.host.stop()


def test_a_promoted_row_types_as_a_hit_on_row_rs_next_post_and_is_not_read(tmp_path):
    """The post after the claim types 4 as an unforced hit and reads nothing. Mutant: leave the promoted expert out of
    the delta's entries -- MISS_GPU and one more row read."""
    rig = _promote_rig(tmp_path)
    try:
        slot, _, _ = _promoted(rig)
        rows, layer_rows = rig.host.counters()["rows_read"], rig.host.layer_rows()
        req = load(rig, 1, [4])
        assert req.kinds[0] in (LaneKind.HIT_COPY, LaneKind.HIT_SM)
        assert rig.host.counters()["rows_read"] == rows and rig.host.layer_rows() == layer_rows
        assert _same_row(rig, 1, slot, 4)
    finally:
        rig.host.stop()


def test_a_promoted_row_is_a_cpu_hit_on_a_forced_lane_and_is_not_read(tmp_path):
    """A forced lane that would be a CPU miss is a CPU hit at the promoted slot, with no read. Mutant: the pool bypass
    at the unforced-miss path left in place for a promoted expert -- red on MISS_CPU or rows_read."""
    rig = _promote_rig(tmp_path)
    try:
        slot, _, _ = _promoted(rig)
        rows = rig.host.counters()["rows_read"]
        req = forced(rig, 1, [4])
        assert req.kinds == [LaneKind.HIT_CPU] and req.slots == [slot]
        assert rig.host.counters()["rows_read"] == rows
        assert rig.host.test_kernel_calls()[-1]["slots"] == [slot]
    finally:
        rig.host.stop()


def test_row_rs_next_record_counts_the_promotion_used_and_the_victim_routed(tmp_path):
    """A record routing the promoted 4 and its evicted 0: 4 is a hit (used), 0 a miss (victim routed), each once.
    Mutant: judge in the claim instead of at row r's record -- both zero; keep the lists past the record -- the next
    record counts again."""
    rig = _promote_rig(tmp_path)
    try:
        _promoted(rig)
        req = load(rig, 1, [4, 0])
        assert req.kinds[0] in (LaneKind.HIT_COPY, LaneKind.HIT_SM) and req.kinds[1] == LaneKind.MISS_GPU
        c = rig.host.counters()
        assert (c["spec_promote_used"], c["spec_promote_victim_routed"]) == (1, 1)
        load(rig, 0, [5])  # claims the delta row 1's miss on 0 left pending
        load(rig, 1, [4])
        c = rig.host.counters()
        assert (c["spec_promote_used"], c["spec_promote_victim_routed"]) == (1, 1)
    finally:
        rig.host.stop()


def test_a_landed_row_with_no_delta_waiting_counts_no_delta_and_stays_landed(tmp_path):
    """Row 1 had no miss, so no delta waits for row 0's record to carry it. Mutant: promote without a pending delta --
    red on the entry state; not counting the landed row -- red on the count."""
    rig = _promote_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        load(rig, 0, [5])
        c = rig.host.counters()
        assert (c["spec_promote_no_delta"], c["spec_promoted_early"]) == (1, 0)
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
    finally:
        rig.host.stop()


def test_two_groups_count_each_landed_row_with_no_delta_once(tmp_path):
    """One landed row in each group's pool, no delta pending, both groups serve row 0's record: no_delta is 2. Mutant:
    each group counts every group's entries -- red on 4."""
    rig = _promote_rig(tmp_path, logits=EIGHT_LOGITS, nodes=2, share=1)
    try:
        rig.host.spec_place(1, 4)
        rig.host.spec_place(1, 3)
        load(rig, 0, [1])
        assert rig.host.counters()["spec_promote_no_delta"] == 2
    finally:
        rig.host.stop()


def test_the_group_that_loses_the_claim_does_not_count_the_leftovers_as_no_delta(tmp_path):
    """Two groups serve row 0's record; the winner claims row 1's delta and leaves 4 landed (no victim), the loser finds
    the word empty. Mutant: count no_delta on every lost claim -- red on no_delta 1."""
    rig = _promote_rig(tmp_path, logits=EIGHT_LOGITS, nodes=2, share=1)
    try:
        load(rig, 1, [0])  # with 2, both of group 0's cache slots (one staging slot: one miss a record)
        load(rig, 0, [1])
        load(rig, 1, [2])
        rig.host.set_hot(1, [0, 2])
        rig.host.spec_place(1, 4)
        tag = rig.sim.delta(1)[0]
        load(rig, 0, [3])
        c = rig.host.counters()
        assert (c["spec_promote_no_victim"], c["spec_promote_no_delta"]) == (1, 0)
        assert rig.sim.delta(1)[0] != tag
    finally:
        rig.host.stop()


def test_a_landed_row_with_no_evictable_victim_counts_no_victim_and_stays_landed(tmp_path):
    """Both RAM slots of row 1 hold hot experts: take_victim finds none, the landed 4 stays in the pool and the delta
    is still written. Mutant: promote into a staging slot -- red on the entry state."""
    rig = _promote_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])
        rig.host.set_hot(1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        tag = rig.sim.delta(1)[0]
        load(rig, 0, [5])
        c = rig.host.counters()
        assert (c["spec_promote_no_victim"], c["spec_promoted_early"]) == (1, 0)
        assert rig.sim.delta(1)[0] != tag
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
    finally:
        rig.host.stop()


def test_a_busy_row_lock_at_the_claim_counts_row_busy_and_leaves_the_row_landed(tmp_path):
    """Group 0's row lock of row 1 is held while row 0's record claims row 1's delta: the delta is written without the
    promotion and the landed 4 stays landed. Mutant: skip a busy group without counting -- red on row_busy 0."""
    rig = _promote_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        tag = rig.sim.delta(1)[0]
        rig.host.hold_row(1, 0)
        try:
            load(rig, 0, [5])
        finally:
            rig.host.hold_row(1, 0, hold=False)
        c = rig.host.counters()
        assert (c["spec_promote_row_busy"], c["spec_promoted_early"]) == (1, 0)
        assert rig.sim.delta(1)[0] != tag and rig.host.mapping(1)[4] == -1
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
    finally:
        rig.host.stop()


def test_a_row_still_reading_at_the_claim_counts_reading_and_is_left_alone(tmp_path):
    """The speculative read of 2 is held 0.6 s; row 0's next record claims row 1's delta meanwhile. Mutant: treat a
    reading entry as landed -- red (a promotion of an unlanded row)."""
    rig = prefetch_rig(tmp_path)
    try:
        # An idle-drive read sleeps off the demand reader's turn, so row 1's record is served while it is in flight.
        enable(rig, LOGITS, promote=True, idle_deadline_s=5.0)
        rig.host.inject_spec(delay_s=0.6)
        rig.host.start_thread(fatal_wait_s=5.0)
        trigger(rig)
        deadline = time.monotonic() + 5.0
        while rig.host.counters()["spec_issued"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert rig.host.counters()["spec_issued"] == 1
        load(rig, 1, [3])
        load(rig, 0, [5])
        c = rig.host.counters()
        assert (c["spec_promote_reading"], c["spec_promoted_early"]) == (1, 0)
    finally:
        rig.host.stop()


def test_a_promotion_past_the_deltas_entry_limit_counts_no_room_and_stays_landed(tmp_path):
    """Seven misses each evicting a resident expert leave row 1's delta at 14 of kDeltaMaxEntries (16): the first
    landed row fits (16), the second counts no_room and stays landed. Mutant: check room for one entry instead of the
    eviction and the insert -- the second promotes (an oversized delta) and no_room stays 0."""
    experts = 14
    rig = prefetch_rig(tmp_path, experts=experts, capacity=17, staging=8)  # 7 cache slots beside staging and the pool
    try:
        enable(rig, [[21 + e for e in range(experts)]], promote=True)
        load(rig, 1, range(7))
        load(rig, 0, [0])  # claims the first record's delta
        load(rig, 1, range(7, 14))
        assert len(rig.sim.delta(1)[2]) == 7, "row 1's second delta is still pending"
        rig.host.spec_place(1, 0)
        rig.host.spec_place(1, 1)
        load(rig, 0, [0])
        c = rig.host.counters()
        assert (c["spec_promoted_early"], c["spec_promote_no_room"]) == (1, 1)
        assert len(rig.sim.delta(1)[2]) == 16
        assert [e["state"] for e in rig.host.spec_pool(1)].count("landed") == 1
    finally:
        rig.host.stop()


def test_another_groups_serve_of_the_stashing_record_does_not_backstop_its_delta(tmp_path):
    """Group 0 (expert 0's home) stashes row 1's delta, then group 1 serves the same record: the delta must wait for
    row 0's next record, which promotes the landed 2 into it. Mutant: backstop the record's own row whatever its
    forward -- group 1 writes the delta, red on backstop 1 and no promotion."""
    rig = _promote_rig(tmp_path, logits=EIGHT_LOGITS, nodes=2, share=1)
    try:
        load(rig, 1, [0])
        slot = rig.host.spec_place(1, 2)
        assert 0 <= slot < 4
        load(rig, 0, [1])
        c = rig.host.counters()
        assert (c["spec_publish_backstop"], c["spec_promoted_early"]) == (0, 1)
        assert rig.host.mapping(1)[2] == slot and (2, slot) in rig.sim.delta(1)[2]
    finally:
        rig.host.stop()


def test_a_two_group_promotion_lands_in_the_experts_home_range(tmp_path):
    """Expert 3 is group 1's: its promotion maps the pool slot of group 1's range and pools group 1's victim. Mutant:
    take the victim from the claimer's group -- red on the range."""
    rig = _promote_rig(tmp_path, logits=EIGHT_LOGITS, nodes=2, share=1)
    try:
        load(rig, 1, [1])
        slot = rig.host.spec_place(1, 3)
        assert 4 <= slot < 8
        load(rig, 0, [0])
        assert rig.host.mapping(1)[3] == slot and (3, slot) in rig.sim.delta(1)[2]
        pool = [e for e in rig.host.spec_pool(1) if e["group"] == 1]
        assert len(pool) == 1 and pool[0]["state"] == "empty" and 4 <= pool[0]["slot"] < 8
        assert rig.host.counters()["spec_promoted_early"] == 1
    finally:
        rig.host.stop()


def test_with_the_flag_off_the_delta_is_written_at_once_and_no_promote_counter_moves(tmp_path):
    """Row 1's post needs no row 0 record in between, a pooled row is swapped in as before, and all eight counters
    stay 0. Mutant: count a promotion or a backstop on the flag-off path -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        load(rig, 1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        load(rig, 1, [2])  # raises unless the previous record's delta is already written
        forced(rig, 1, [4])
        c = rig.host.counters()
        assert c["spec_used"] == 1 and rig.host.mapping(1)[4] == slot
        assert [c[name] for name in PROMOTE_COUNTERS] == [0] * len(PROMOTE_COUNTERS)
    finally:
        rig.host.stop()
