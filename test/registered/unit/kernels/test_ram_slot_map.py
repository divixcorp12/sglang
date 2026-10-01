"""The device slot map's Python reference: lane typing and delta application (CPU)."""

import pytest

from sglang.srt.layers.moe.ram_slot_map import LaneKind, MapReplica, type_lanes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]
NO_STAGING = [-1] * 8


def _type(experts, ram_slot, staging=NO_STAGING, **kw):
    args = dict(split=SPLIT, captured=True, copy_armed=True, hit_copy="ce", cpu_on=False, cpu_misses=False)
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
        _type(list(range(9)), [1] * 9)
    with pytest.raises(ValueError, match="staging"):
        _type([0], [-1])


def test_replica_applies_the_attach_delta_then_a_decode_delta_once():
    m = MapReplica(rows=1, experts=4)
    assert m.map_chain == [1] and m.map_applied == [0]
    m.apply_delta(0, tag=1, staging=[9] + [-1] * 7, entries=[])
    assert m.staging[0][0] == 9 and m.map_applied[0] == 1
    m.apply_delta(0, tag=2, staging=[8] + [-1] * 7, entries=[(2, 5), (1, -1)])
    m.ram_slot[0][2] = 6  # a bulk entry applied later must survive a second application attempt
    m.apply_delta(0, tag=2, staging=[8] + [-1] * 7, entries=[(2, 5), (1, -1)])
    assert m.ram_slot[0][2] == 6 and m.map_applied[0] == 2


def test_replica_bulk_writes_entries_across_rows():
    m = MapReplica(rows=2, experts=3)
    m.apply_bulk([(0, 1, 4), (1, 2, 7), (0, 1, -1)])
    assert m.ram_slot == [[-1, -1, -1], [-1, -1, 7]]
