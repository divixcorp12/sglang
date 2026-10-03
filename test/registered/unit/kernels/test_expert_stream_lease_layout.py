"""LeaseLayout<NumLanes, NumNodes> (lease_layout.h) and its Python mirror wire_layout agree, and (8, 1) is wire v2."""

import re

import pytest

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import device_sources, host_sources, wire_header

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


def test_the_transport_and_lease_modules_carry_the_trait_values():
    """expert_stream_transport and expert_lease_block's module constants, and LaneKind, equal lease_layout.h's v2."""
    from sglang.srt.layers.moe.ram_slot_map import LaneKind

    wire = lease.wire_probe(8, 1)
    python = [
        ("kDemandHead", transport.WORDS["demand_head"]),
        ("kDemandRing", transport.DEMAND_RING),
        ("kDemandRecords", transport.DEMAND_RECORDS),
        ("kRecordBytes", transport.RECORD_BYTES),
        ("kLanes", transport.MAX_IDS),
        ("kRecSeq", transport.RECORD_FIELDS["seq"]),
        ("kRecRow", transport.RECORD_FIELDS["row"]),
        ("kRecCounts", transport.RECORD_FIELDS["counts"]),
        ("kRecFlags", transport.RECORD_FIELDS["flags"]),
        ("kRecFlagCaptured", transport.RECORD_FLAG_CAPTURED),
        ("kRecChain", transport.RECORD_FIELDS["chain"]),
        ("kRecEpoch", transport.RECORD_FIELDS["epoch"]),
        ("kRecKinds", transport.RECORD_FIELDS["kinds"]),
        ("kRecProtect", transport.RECORD_FIELDS["protect"]),
        ("kRecLaneExpert", transport.RECORD_FIELDS["lane_expert"]),
        ("kRecLaneSlot", transport.RECORD_FIELDS["lane_slot"]),
        ("kRecLaneDst", transport.RECORD_FIELDS["lane_dst"]),
        ("kRecLaneWeight", transport.RECORD_FIELDS["lane_weight"]),
        ("kRecIdMax", transport.RECORD_ID_MAX),
        ("kPageBytes", transport.PAGE_BYTES),
        ("kKindHitCopy", LaneKind.HIT_COPY),
        ("kKindHitSm", LaneKind.HIT_SM),
        ("kKindHitCpu", LaneKind.HIT_CPU),
        ("kKindMissGpu", LaneKind.MISS_GPU),
        ("kKindMissCpu", LaneKind.MISS_CPU),
        ("kHotHeaderBytes", transport.HOT_HEADER_BYTES),
        ("kHotAlignment", transport.HOT_ALIGNMENT),
        ("kHotRecords", transport.HOT_RECORDS),
        ("kDemandRecords", lease.RING),
        ("kLanes", lease.LANES),
        ("kLeaseBlockAlign", lease.BLOCK_ALIGN),
        ("kLeasePieceMask", lease.PIECE_MASK),
        ("kLeasePieceMaskLineBytes", lease.PIECE_MASK_LINE_BYTES),
        ("kLeaseCopyDone", lease.COPY_DONE),
        ("kLeaseCopyDoneBytes", lease.COPY_DONE_BYTES),
        ("kLeaseCopyGate", lease.COPY_GATE),
        ("kLeaseGateClosed", lease.GATE["closed"]),
        ("kLeaseGateOpen", lease.GATE["open"]),
        ("kLeaseGateSeqShift", lease.GATE_SEQ_SHIFT),
        ("kLeaseGateSeqMask", lease.GATE_SEQ_MASK),
        ("kCopyArmed", lease.COPY_ARMED),
        ("kSplit", lease.SPLIT),
        ("kLeaseBlockBytes", lease.BLOCK_BYTES),
        ("kDeltaBase", lease.DELTA_BASE),
        ("kDeltaStride", lease.DELTA_STRIDE),
        ("kDeltaTag", lease.DELTA_FIELDS["tag"]),
        ("kDeltaCount", lease.DELTA_FIELDS["count"]),
        ("kDeltaStaging", lease.DELTA_FIELDS["staging"]),
        ("kDeltaEntries", lease.DELTA_FIELDS["entries"]),
        ("kDeltaMaxEntries", lease.DELTA_MAX_ENTRIES),
    ]
    assert [(name, wire[name]) for name, _ in python] == [(name, int(value)) for name, value in python]


FREE_NAME = re.compile(r"(?<![\w:])k(MaxIds|LeaseLanes|LeaseRing|RecordBytes|PageBytes|RecLane\w+|Delta\w+|Split)\b")


def test_no_source_uses_a_free_wire_name():
    """Every wire offset is spelled Wire::k..., so a build's lane count reaches every use."""
    for path in (*host_sources(), *device_sources()):
        text = re.sub(r"//[^\n]*", "", path.read_text())
        assert not FREE_NAME.search(text), f"{path.name} still uses a free wire name"
    assert "inline constexpr auto" not in wire_header().read_text()
