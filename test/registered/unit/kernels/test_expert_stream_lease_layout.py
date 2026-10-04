"""LeaseLayout<NumLanes, NumNodes> (lease_layout.h) and its Python mirror wire_layout agree, and (8, 1) is wire v2."""

import re

import pytest

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import device_sources, host_sources, wire_header

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

GRID = [(lanes, nodes) for lanes in (1, 6, 8, 13, 24, 32) for nodes in (1, 2)]


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


def test_the_transport_and_lease_modules_carry_the_trait_values():
    """What the transport and lease modules still carry as constants, the wire layout every runtime object is built
    from (``wire_layout(8)``, ``new_page``) and LaneKind equal lease_layout.h's v2."""
    from sglang.srt.layers.moe.ram_slot_map import LaneKind

    wire = lease.wire_probe(8, 1)
    w = lease.wire_layout(8)
    f, d = w.record_fields, w.delta_fields
    python = [
        ("kDemandHead", transport.WORDS["demand_head"]),
        ("kDemandRing", w.demand_ring),
        ("kDemandRecords", w.demand_records),
        ("kRecordBytes", w.record_bytes),
        ("kLanes", w.lanes),
        ("kRecSeq", f["seq"]),
        ("kRecRow", f["row"]),
        ("kRecCounts", f["counts"]),
        ("kRecFlags", f["flags"]),
        ("kRecFlagCaptured", transport.RECORD_FLAG_CAPTURED),
        ("kRecChain", f["chain"]),
        ("kRecEpoch", f["epoch"]),
        ("kRecKinds", f["kinds"]),
        ("kRecProtect", f["protect"]),
        ("kRecLaneExpert", f["lane_expert"]),
        ("kRecLaneSlot", f["lane_slot"]),
        ("kRecLaneDst", f["lane_dst"]),
        ("kRecLaneWeight", f["lane_weight"]),
        ("kRecIdMax", transport.RECORD_ID_MAX),
        ("kPageBytes", transport.new_page(pin=False, wire=w).numel()),
        ("kKindHitCopy", LaneKind.HIT_COPY),
        ("kKindHitSm", LaneKind.HIT_SM),
        ("kKindHitCpu", LaneKind.HIT_CPU),
        ("kKindMissGpu", LaneKind.MISS_GPU),
        ("kKindMissCpu", LaneKind.MISS_CPU),
        ("kHotHeaderBytes", transport.HOT_HEADER_BYTES),
        ("kHotAlignment", transport.HOT_ALIGNMENT),
        ("kHotRecords", transport.HOT_RECORDS),
        ("kLeaseBlockAlign", w.block_align),
        ("kLeasePieceMask", w.piece_mask),
        ("kLeasePieceMaskLineBytes", w.piece_mask_line_bytes),
        ("kLeaseCopyDone", w.copy_done),
        ("kLeaseCopyDoneBytes", w.copy_done_bytes),
        ("kLeaseCopyGate", w.copy_gate),
        ("kLeaseGateClosed", lease.GATE["closed"]),
        ("kLeaseGateOpen", lease.GATE["open"]),
        ("kLeaseGateSeqShift", lease.GATE_SEQ_SHIFT),
        ("kLeaseGateSeqMask", lease.GATE_SEQ_MASK),
        ("kCopyArmed", w.copy_armed),
        ("kSplit", w.split),
        ("kLeaseBlockBytes", w.lease_block_bytes),
        ("kDeltaBase", w.lease_block_bytes),
        ("kDeltaStride", w.delta_stride),
        ("kDeltaTag", d["tag"]),
        ("kDeltaCount", d["count"]),
        ("kDeltaStaging", d["staging"]),
        ("kDeltaEntries", d["entries"]),
        ("kDeltaMaxEntries", w.delta_max_entries),
    ]
    assert [(name, wire[name]) for name, _ in python] == [(name, int(value)) for name, value in python]


FREE_NAME = re.compile(r"(?<![\w:])k(MaxIds|LeaseLanes|LeaseRing|RecordBytes|PageBytes|RecLane\w+|Delta\w+|Split)\b")


def test_no_source_uses_a_free_wire_name():
    """Every wire offset is spelled Wire::k..., so a build's lane count reaches every use."""
    for path in (*host_sources(), *device_sources()):
        text = re.sub(r"//[^\n]*", "", path.read_text())
        assert not FREE_NAME.search(text), f"{path.name} still uses a free wire name"
    assert "inline constexpr auto" not in wire_header().read_text()
