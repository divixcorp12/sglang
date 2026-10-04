"""The service's record reader against records written byte by byte (CPU): every field round-trips, the kinds of
unused lanes are never judged, and a torn or malformed record is reported as such. read_record_fields is the
instrumented build's view of read_record (host/tier_protocol.h)."""

import random
import struct

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import RECORD_FLAG_CAPTURED, RECORD_ID_MAX
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LANES = 8
KINDS = [int(k) for k in (LaneKind.HIT_COPY, LaneKind.HIT_SM, LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.MISS_CPU)]


def encode(lanes, *, seq, row=3, chain=0, epoch=0, flags=0, protect=(), kinds=(), experts=(), slots=(), dsts=(),
           weights=(), kind_words=None, counts=None):
    """A record as the post writes it, for LeaseLayout<lanes>. ``kind_words`` and ``counts`` override the packed words
    (a record the device never writes)."""
    w = lease.wire_layout(lanes)
    f, n = w.record_fields, w.lanes
    raw = bytearray(w.record_bytes)
    count = len(kinds)
    struct.pack_into("<IH", raw, f["seq"], seq, row)
    if counts is not None:
        raw[f["counts"]] = counts
    elif w.packed_counts:
        raw[f["counts"]] = count | len(protect) << 4
    else:
        raw[f["counts"]] = count
        raw[w.protect_count] = len(protect)
    raw[f["flags"]] = flags
    struct.pack_into("<QI", raw, f["chain"], chain, epoch)
    if kind_words is None:
        kind_words = [0] * w.kind_words
        for j, kind in enumerate(kinds):
            kind_words[j // 8] |= kind << (4 * (j % 8))
    struct.pack_into(f"<{w.kind_words}I", raw, f["kinds"], *kind_words)
    pad = lambda xs: list(xs) + [-1] * (n - len(xs))
    struct.pack_into(f"<{n}h", raw, f["protect"], *pad(protect))
    struct.pack_into(f"<{n}h", raw, f["lane_expert"], *pad(experts))
    struct.pack_into(f"<{n}h", raw, f["lane_slot"], *pad(slots))
    struct.pack_into(f"<{n}h", raw, f["lane_dst"], *pad(dsts))
    struct.pack_into(f"<{n}f", raw, f["lane_weight"], *(list(weights) + [0.0] * (n - len(weights))))
    return torch.tensor(list(raw), dtype=torch.uint8)


def write_record(*, seq, row=0, captured=False, chain=0, epoch=0, protect=(), lanes=(), kinds=None, counts=None):
    """An 8-lane record from (expert, slot, dst, weight, kind) tuples; ``kinds`` is the single kinds word."""
    return encode(
        8, seq=seq, row=row, chain=chain, epoch=epoch, flags=RECORD_FLAG_CAPTURED if captured else 0,
        protect=protect, kinds=[lane[4] for lane in lanes], experts=[lane[0] for lane in lanes],
        slots=[lane[1] for lane in lanes], dsts=[lane[2] for lane in lanes], weights=[lane[3] for lane in lanes],
        kind_words=None if kinds is None else [kinds], counts=counts,
    )


def read(record, expected, lanes=8):
    return ops.read_record_fields(record, expected, variant="instr", lanes=lanes)


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


@pytest.mark.parametrize("case", ["lane_count", "protect_count", "kind_zero", "kind_six"])
def test_a_record_the_device_never_writes_is_malformed(case):
    override = {
        "lane_count": {"counts": 9},
        "protect_count": {"counts": 1 | 9 << 4},
        "kind_zero": {"kinds": 0},
        "kind_six": {"kinds": 6},
    }[case]
    record = write_record(seq=5, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))], **override)
    assert read(record, 5)["status"] == "malformed"


def test_a_record_whose_seq_is_not_the_expected_one_is_torn():
    record = write_record(seq=5 + 16, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))])
    assert read(record, 5)["status"] == "torn"


def test_a_record_is_one_128_byte_prefetch_pair():
    """The record's two cache lines are one 128-byte-aligned block, so an L2 miss on the first makes the adjacent-line
    prefetcher fetch the second; the ring starts on such a block."""
    w = lease.wire_layout(8)
    assert w.record_bytes == 128 and w.demand_ring % 128 == 0
    assert w.record_fields["lane_weight"] + 4 * LANES == w.record_bytes
    assert ops.new_page(pin=False, wire=w).numel() == w.demand_ring + w.demand_records * w.record_bytes


@pytest.mark.parametrize(
    "lanes, count, protect", [(8, 8, 0), (8, 1, 8), (16, 16, 16), (16, 9, 3), (32, 32, 32), (32, 17, 0)]
)
def test_read_record_round_trips_every_lane(lanes, count, protect):
    kinds = [KINDS[j % 5] for j in range(count)]
    rec = encode(lanes, seq=7, protect=range(protect), kinds=kinds, experts=range(100, 100 + count),
                 slots=range(count), dsts=range(50, 50 + count), weights=[0.5 + j for j in range(count)])
    got = read(rec, 7, lanes)
    assert got["status"] == "ok"
    assert got["lanes"] == [
        {"expert": 100 + j, "slot": j, "dst": 50 + j, "weight": 0.5 + j, "kind": kinds[j]} for j in range(count)
    ]
    assert got["protect"] == list(range(protect))


@pytest.mark.parametrize("lanes", [16, 32])
def test_a_count_past_the_lane_width_is_malformed(lanes):
    rec = encode(lanes, seq=7, kinds=[1])
    rec[lease.wire_layout(lanes).record_fields["counts"]] = lanes + 1
    assert read(rec, 7, lanes)["status"] == "malformed"


@pytest.mark.parametrize("lanes", [16, 32])
def test_a_protect_count_past_the_lane_width_is_malformed(lanes):
    rec = encode(lanes, seq=7, kinds=[1])
    rec[lease.wire_layout(lanes).protect_count] = lanes + 1
    assert read(rec, 7, lanes)["status"] == "malformed"


@pytest.mark.parametrize("lanes", [8, 32])
def test_no_torn_record_is_accepted(lanes):
    accepted, torn = ops.seqlock_stress(2.0, variant="instr", lanes=lanes)
    assert accepted > 0, (accepted, torn)
    assert torn == 0, (accepted, torn)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
