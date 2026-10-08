"""Swap-on-use (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Swap on use"): a forced CPU miss whose expert
landed in the row's pool takes its victim as today, maps the pool slot, gives the victim to the pool, reads nothing and
goes to the CPU at once. spec_place lands a pool row as a speculative read would (CPU, ChainSim)."""

from sglang.kernels.ops.moe.expert_stream_transport import piece_word
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ALL_PIECES
from sglang.test.dsv41_ram_miss_fixtures import same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import forced, load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FREE, READY, SPEC = 0, 2, 4


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
        rig.host.pump()
        assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)
        assert rig.sim.piece_word(req, 0) == piece_word(req.gen, ALL_PIECES)
        assert rig.sim.piece_word(req, 1) != piece_word(req.gen, ALL_PIECES)
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
