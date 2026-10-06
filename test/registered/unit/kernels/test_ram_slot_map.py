"""The device slot map's Python reference: lane typing and delta application (CPU)."""

import pytest

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.srt.layers.moe.ram_slot_map import LaneKind, LaneOverflow, MapReplica, type_lanes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]
NO_STAGING = [-1] * 8


def _type(experts, ram_slot, staging=NO_STAGING, **kw):
    args = dict(split=SPLIT, captured=True, copy_armed=True, hit_copy="ce", cpu_on=False, cpu_misses=False, lanes=8)
    args.update(kw)
    return type_lanes(experts, ram_slot=ram_slot, staging=staging, **args)


def test_hits_and_misses_take_map_and_staging_slots_in_order():
    kinds, slots = _type([5, 9, 7], ram_slot=[-1] * 5 + [11, -1, -1, -1, -1], staging=[40, 41] + [-1] * 6)
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU, LaneKind.MISS_GPU]
    assert slots == [11, 40, 41]


def test_cpu_tail_hits_only_and_with_misses():
    ram = [-1] * 10
    ram[1], ram[3] = 21, 23
    staging = [50, 51, 52] + [-1] * 5
    kinds, _ = _type([1, 2, 3, 4], ram, staging, cpu_on=True)  # hits are lanes 0 and 2: n = 2, split[2] = 1
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU, LaneKind.HIT_CPU, LaneKind.MISS_GPU]
    kinds, _ = _type([1, 2, 3, 4], ram, staging, cpu_on=True, cpu_misses=True)  # n = 4, split[4] = 3: lanes 1-3
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_CPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU]


def test_eager_or_unarmed_posts_use_sm_and_no_cpu():
    for captured, armed in ((False, True), (True, False)):
        kinds, _ = _type([0, 1], [7, -1], [3] + [-1] * 7, captured=captured, copy_armed=armed,
                         cpu_on=True, cpu_misses=True)
        assert kinds == [LaneKind.HIT_SM, LaneKind.MISS_GPU]


def test_sm_mode_hits_never_copy_engine():
    kinds, _ = _type([0], [4], hit_copy="sm")
    assert kinds == [LaneKind.HIT_SM]


def test_an_ineligible_row_or_destination_falls_back_to_sm_and_gpu():
    kinds, _ = _type([0], [4], ce_ok=False)
    assert kinds == [LaneKind.HIT_SM]
    kinds, _ = _type([0, 1], [4, 5], dst_ok=[True, False])
    assert kinds == [LaneKind.HIT_COPY, LaneKind.HIT_SM]
    kinds, _ = _type([0, 1], [4, -1], [9] + [-1] * 7, cpu_on=True, cpu_misses=True, cpu_ok=False)
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU]


def test_refuses_duplicates_too_many_lanes_and_a_miss_without_staging():
    with pytest.raises(ValueError, match="twice"):
        _type([3, 3], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="at most 8"):
        _type(list(range(9)), [1] * 9, lanes=8)
    with pytest.raises(ValueError, match="staging"):
        _type([0], [-1])


def test_a_16_lane_request_takes_16_experts_and_refuses_17():
    """The bound is the build's lane count, not 8: lanes 8-15 take their own staging slots in order."""
    experts = list(range(16))
    kinds, slots = _type(experts, ram_slot=[-1] * 17, staging=list(range(40, 56)), lanes=16, split=[0] * 17)
    assert kinds == [LaneKind.MISS_GPU] * 16 and slots == list(range(40, 56))
    with pytest.raises(ValueError, match="at most 16"):
        _type(list(range(17)), [1] * 17, lanes=16, split=[0] * 17)
    with pytest.raises(ValueError, match="staging"):
        _type(experts, [-1] * 17, list(range(40, 55)) + [-1], lanes=16, split=[0] * 17)


def test_the_cpu_tail_reaches_lanes_past_8_at_16_lanes():
    ram = list(range(100, 112)) + [-1] * 4
    kinds, _ = _type(list(range(12)), ram, [-1] * 16, lanes=16, cpu_on=True, split=[0] * 12 + [4] + [0] * 4)
    assert kinds == [LaneKind.HIT_COPY] * 8 + [LaneKind.HIT_CPU] * 4


def test_replica_applies_the_attach_delta_then_a_decode_delta_once():
    m = MapReplica(rows=1, experts=4, lanes=8)
    assert m.map_chain == [1] and m.map_applied == [0]
    m.apply_delta(0, tag=1, staging=[9] + [-1] * 7, entries=[])
    assert m.staging[0][0] == 9 and m.map_applied[0] == 1
    m.apply_delta(0, tag=2, staging=[8] + [-1] * 7, entries=[(2, 5), (1, -1)])
    m.ram_slot[0][2] = 6  # a bulk entry applied later must survive a second application attempt
    m.apply_delta(0, tag=2, staging=[8] + [-1] * 7, entries=[(2, 5), (1, -1)])
    assert m.ram_slot[0][2] == 6 and m.map_applied[0] == 2


def test_a_replica_keeps_one_staging_slot_per_lane_of_its_build():
    assert [len(row) for row in MapReplica(rows=2, experts=3, lanes=16).staging] == [16, 16]


def test_replica_bulk_writes_entries_across_rows():
    m = MapReplica(rows=2, experts=3, lanes=8)
    m.apply_bulk([(0, 1, 4), (1, 2, 7), (0, 1, -1)])
    assert m.ram_slot == [[-1, -1, -1], [-1, -1, 7]]


def _staging(nodes, lanes, counts):
    """Node-major lists: node n's k-th staging slot is 100 * (n + 1) + k, its first counts[n] of them valid."""
    return [100 * (n + 1) + k if k < counts[n] else -1 for n in range(nodes) for k in range(lanes)]


def test_each_miss_takes_the_next_staging_slot_of_its_home_node():
    ram = [-1] * 16
    ram[2], ram[5] = 7, 9
    kinds, slots = type_lanes(
        [1, 2, 3, 4, 5, 6], ram, _staging(2, 8, [8, 8]), [0] * 18, lanes=8, captured=False, copy_armed=False,
        hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2,
    )
    # experts 1 and 3 are node 1's misses, 4 and 6 node 0's; 2 and 5 are hits
    assert slots == [200, 7, 201, 100, 9, 101]
    assert kinds == [LaneKind.MISS_GPU, LaneKind.HIT_SM, LaneKind.MISS_GPU, LaneKind.MISS_GPU, LaneKind.HIT_SM,
                     LaneKind.MISS_GPU]


def test_misses_homed_on_one_node_draw_only_on_that_nodes_list():
    """Review Focus 1, the reference: all misses on node 1 use node 1's list and leave node 0's untouched; one miss
    past node 1's list raises even though node 0 has slots."""
    staging = _staging(2, 8, [8, 2])
    _, slots = type_lanes([1, 3], [-1] * 16, staging, [0] * 18, lanes=8, captured=False, copy_armed=False,
                          hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2)
    assert slots == [200, 201]
    with pytest.raises(ValueError, match="no staging slot on node 1"):
        type_lanes([1, 3, 5], [-1] * 16, staging, [0] * 18, lanes=8, captured=False, copy_armed=False,
                   hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2)


def test_the_cpu_split_is_per_node_and_a_zero_split_leaves_a_node_on_the_gpu():
    lanes = 8
    ram = list(range(16))  # every expert a hit
    split = [0] * (2 * (lanes + 1))
    split[0 * (lanes + 1) + 3] = 2  # node 0: 2 of its 3 eligible lanes on the CPU
    kinds, _ = type_lanes(
        [0, 1, 2, 3, 4, 5], ram, _staging(2, lanes, [8, 8]), split, lanes=lanes, captured=True, copy_armed=True,
        hit_copy="sm", cpu_on=True, cpu_misses=False, nodes=2,
    )
    # node 0's lanes are 0, 2, 4: the last two go to the CPU; node 1's split is all zero
    assert kinds == [LaneKind.HIT_SM, LaneKind.HIT_SM, LaneKind.HIT_CPU, LaneKind.HIT_SM, LaneKind.HIT_CPU,
                     LaneKind.HIT_SM]


def test_one_node_is_the_flat_lists_of_today():
    replica = MapReplica(2, 16, 8)
    assert len(replica.staging[0]) == 8
    assert len(MapReplica(2, 16, 8, nodes=2).staging[0]) == 16
    assert wire_layout(8, 2).home(5) == 1


def test_forced_lanes_are_cpu_lanes_outside_the_split():
    """Lanes 0-1 found VRAM victims, lanes 2-3 did not (spill). The split sees only lane 0, the one eligible unforced
    lane (split[1] = 1), so it is the CPU's; lane 1, an unforced miss, stays on the GPU in staging slot 9. The forced
    hit is a CPU lane at its RAM slot; the forced miss is a CPU miss with slot -1, though staging slot 10 is free: the
    host reads it into a RAM victim (Task 9), so staging only ever holds live misses."""
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    staging = [9, 10] + [-1] * 6
    kinds, slots = _type([1, 0, 2, 7], ram, staging, cpu_on=True, forced_from=2)
    assert kinds == [LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU]
    assert slots == [4, 9, 5, -1]


def test_36_forced_misses_on_one_node_need_no_staging():
    """Review Focus 2 at the record's full width: 36 lanes on a 40-lane, 2-node wire, 8 of them with VRAM victims,
    every lane an NVMe miss homed on node 0 (even experts). The 8 live misses take node 0's 8 staging slots; the 28
    forced ones take none. No LaneOverflow, whatever one node holds."""
    lanes, nodes = 40, 2
    experts = [2 * e for e in range(36)]
    staging = list(range(100, 108)) + [-1] * (lanes - 8) + [-1] * lanes  # node 0's list, then node 1's
    kinds, slots = type_lanes(
        experts, [-1] * 80, staging, [0] * (nodes * (lanes + 1)), lanes=lanes, captured=True, copy_armed=True,
        hit_copy="ce", cpu_on=True, cpu_misses=False, nodes=nodes, forced_from=8,
    )
    assert kinds == [LaneKind.MISS_GPU] * 8 + [LaneKind.MISS_CPU] * 28
    assert slots == list(range(100, 108)) + [-1] * 28


def test_forced_from_the_count_forces_nothing():
    ram = [-1] * 16
    ram[1] = 4
    staging = [9, 10] + [-1] * 6
    assert _type([1, 0], ram, staging, cpu_on=True, forced_from=2) == _type([1, 0], ram, staging, cpu_on=True)


@pytest.mark.parametrize(
    "changes",
    [{"copy_armed": False}, {"captured": False}, {"cpu_on": False}, {"cpu_ok": False}],
    ids=["unarmed", "eager", "cpu-off", "no-cpu-layer"],
)
def test_a_forced_lane_that_cannot_be_the_cpus_overflows(changes):
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    with pytest.raises(LaneOverflow):
        _type([1, 2], ram, forced_from=1, **{"cpu_on": True, **changes})


def test_an_unforced_miss_without_staging_still_raises_plainly():
    """A live miss always has a staging slot (live <= victim lanes = staging per node): its absence is a broken
    invariant, a plain ValueError (the device traps), never an overflow."""
    with pytest.raises(ValueError) as refused:
        _type([0, 7], [-1] * 16, NO_STAGING, cpu_on=True, forced_from=1)
    assert not isinstance(refused.value, LaneOverflow)
