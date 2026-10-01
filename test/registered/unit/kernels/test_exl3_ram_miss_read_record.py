"""The service's record reader against records written byte by byte (CPU): every field round-trips, the kinds of
unused lanes are never judged, and a torn or malformed record is reported as such. read_record_fields is the
instrumented build's view of read_record (host/tier_protocol.h)."""

import random
import struct

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import (
    DEMAND_RECORDS,
    DEMAND_RING,
    PAGE_BYTES,
    RECORD_BYTES,
    RECORD_FIELDS,
    RECORD_FLAG_CAPTURED,
    RECORD_ID_MAX,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LANES = 8
KINDS = [int(k) for k in (LaneKind.HIT_COPY, LaneKind.HIT_SM, LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.MISS_CPU)]


def _put(record: torch.Tensor, field: str, fmt: str, *values) -> None:
    data = struct.pack("<" + fmt, *values)
    at = RECORD_FIELDS[field]
    record[at : at + len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8)


def write_record(*, seq, row=0, captured=False, chain=0, epoch=0, protect=(), lanes=(), kinds=None, counts=None):
    """One record as the post kernel's write_record lays it out. lanes: (expert, slot, dst, weight, kind) tuples;
    ``kinds`` and ``counts`` override the packed words (a record the device never writes)."""
    record = torch.zeros(RECORD_BYTES, dtype=torch.uint8)
    if kinds is None:
        kinds = sum(lane[4] << (4 * j) for j, lane in enumerate(lanes))
    if counts is None:
        counts = len(lanes) | len(protect) << 4

    def ids(values):
        return list(values) + [-1] * (LANES - len(values))

    _put(record, "seq", "I", seq)
    _put(record, "row", "H", row)
    _put(record, "counts", "B", counts)
    _put(record, "flags", "B", RECORD_FLAG_CAPTURED if captured else 0)
    _put(record, "chain", "Q", chain)
    _put(record, "epoch", "I", epoch)
    _put(record, "kinds", "I", kinds)
    _put(record, "protect", "8h", *ids(protect))
    _put(record, "lane_expert", "8h", *ids([lane[0] for lane in lanes]))
    _put(record, "lane_slot", "8h", *ids([lane[1] for lane in lanes]))
    _put(record, "lane_dst", "8h", *ids([lane[2] for lane in lanes]))
    _put(record, "lane_weight", "8f", *([lane[3] for lane in lanes] + [0.0] * (LANES - len(lanes))))
    return record


def read(record, expected):
    return ops.read_record_fields(record, expected, variant="instr")


@pytest.mark.parametrize("lane_count, protect_count", [(0, 0), (1, 8), (6, 6), (8, 0), (8, 8)])
def test_every_field_of_a_whole_record_round_trips(lane_count, protect_count):
    rng = random.Random(9 * lane_count + protect_count)

    def ids(n):
        return rng.sample([0, 1, RECORD_ID_MAX] + list(range(2, 64)), n)

    lanes = [
        (e, s, d, rng.choice([0.25, -1.5, 0.5]), rng.choice(KINDS))
        for e, s, d in zip(ids(lane_count), ids(lane_count), ids(lane_count))
    ]
    protect = ids(protect_count)
    record = write_record(seq=37, row=RECORD_ID_MAX, captured=True, chain=(1 << 40) + 5, epoch=9, protect=protect,
                          lanes=lanes)
    got = read(record, 37)
    assert got["status"] == "ok"
    assert (got["row"], got["captured"], got["chain"], got["gen"]) == (RECORD_ID_MAX, True, (1 << 40) + 5, 9 << 32 | 37)
    assert got["protect"] == protect
    assert [(l["expert"], l["slot"], l["dst"], l["weight"], l["kind"]) for l in got["lanes"]] == lanes


def test_the_kinds_of_unused_lanes_are_never_judged():
    lanes = [(1, 2, 3, 0.5, int(LaneKind.HIT_SM)), (4, 5, 6, 0.25, int(LaneKind.MISS_GPU))]
    kinds = int(LaneKind.HIT_SM) | int(LaneKind.MISS_GPU) << 4 | 0xFFFFFF00
    got = read(write_record(seq=5, lanes=lanes, kinds=kinds), 5)
    assert got["status"] == "ok" and [l["kind"] for l in got["lanes"]] == [LaneKind.HIT_SM, LaneKind.MISS_GPU]


@pytest.mark.parametrize("case", ["lane_count", "kind_zero", "kind_six"])
def test_a_record_the_device_never_writes_is_malformed(case):
    override = {"lane_count": {"counts": 9}, "kind_zero": {"kinds": 0}, "kind_six": {"kinds": 6}}[case]
    record = write_record(seq=5, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))], **override)
    assert read(record, 5)["status"] == "malformed"


def test_a_record_whose_seq_is_not_the_expected_one_is_torn():
    record = write_record(seq=5 + 16, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))])
    assert read(record, 5)["status"] == "torn"


def test_a_record_is_one_128_byte_prefetch_pair():
    """The record's two cache lines are one 128-byte-aligned block, so an L2 miss on the first makes the adjacent-line
    prefetcher fetch the second; the ring starts on such a block."""
    assert RECORD_BYTES == 128 and DEMAND_RING % 128 == 0
    assert RECORD_FIELDS["lane_weight"] + 4 * LANES == RECORD_BYTES
    assert PAGE_BYTES == DEMAND_RING + DEMAND_RECORDS * RECORD_BYTES


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
