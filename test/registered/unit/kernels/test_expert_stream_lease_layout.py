"""LeaseLayout<NumLanes, NumNodes> (lease_layout.h) and its Python mirror wire_layout agree, and (8, 1) is wire v2."""

import pytest

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

GRID = [(lanes, nodes) for lanes in (1, 6, 8, 13, 32) for nodes in (1, 2)]


@pytest.mark.parametrize("lanes, nodes", GRID)
def test_the_python_layout_is_the_cpp_trait(lanes, nodes):
    assert lease.wire_layout(lanes, nodes).cpp_constants() == lease.wire_probe(lanes, nodes)


def test_eight_lanes_on_one_node_is_wire_v2():
    w = lease.wire_layout(8)
    assert (w.record_bytes, w.page_bytes, w.lease_block_bytes) == (128, 2176, 20480)
    assert w.record_fields == {
        "seq": 0, "row": 4, "counts": 6, "flags": 7, "chain": 8, "epoch": 16, "kinds": 20,
        "protect": 32, "lane_expert": 48, "lane_slot": 64, "lane_dst": 80, "lane_weight": 96,
    }
    assert (w.copy_done, w.copy_gate, w.copy_armed, w.split) == (0x4000, 0x4080, 0x4100, 16768)
    assert (w.delta_fields, w.delta_max_entries, w.delta_stride) == (
        {"tag": 0, "count": 8, "staging": 16, "entries": 32}, 16, 256,
    )
    assert w.packed_counts


@pytest.mark.parametrize("lanes, rounded", [(1, 8), (8, 8), (9, 16), (13, 16), (17, 24), (32, 32)])
def test_lanes_round_up_to_eight(lanes, rounded):
    assert lease.wire_layout(lanes).lanes == rounded


@pytest.mark.parametrize("lanes", [0, 33])
def test_a_lane_count_outside_1_to_32_is_refused(lanes):
    with pytest.raises(ValueError, match="1..32"):
        lease.wire_layout(lanes)


def test_wider_records_round_to_whole_line_pairs():
    assert [lease.wire_layout(n).record_bytes for n in (16, 24, 32)] == [256, 384, 512]
    assert not lease.wire_layout(16).packed_counts
