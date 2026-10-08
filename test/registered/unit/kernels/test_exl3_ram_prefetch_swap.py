"""Swap-on-use (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Swap on use"): a forced CPU miss whose expert
landed in the row's pool takes its victim as today, maps the pool slot, gives the victim to the pool, reads nothing and
goes to the CPU at once. spec_place lands a pool row as a speculative read would (CPU, ChainSim)."""

from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import forced, load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

SPEC, READY = 4, 2


def _entry(host, row, expert):
    return next((e for e in host.spec_pool(row) if e["expert"] == expert), None)


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
        req = forced(rig, 1, [4])
        assert req.kinds == [LaneKind.MISS_CPU] and req.slots == [-1]
        counters = rig.host.counters()
        assert counters["spec_used"] == 1
        assert counters["rows_read"] == rows and rig.host.layer_rows() == layer_rows, "a pooled miss was read"
        assert rig.host.mapping(1)[4] == slot and rig.sim.delta(1)[2] == [(4, slot)]
        assert rig.host.slot_info(1)[slot][0] == READY
        victim = 5  # the row's first free slot, now pooled
        assert rig.host.slot_info(1)[victim][0] == SPEC
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
        forced(rig, 1, [4])
        assert rig.sim.delta(1)[2] == [(0, -1), (4, slot)]
        assert rig.host.mapping(1)[0] == -1 and rig.host.mapping(1)[4] == slot
        assert rig.host.slot_info(1)[victim][0] == SPEC
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        assert rig.host.counters()["evictions"] >= 1
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
