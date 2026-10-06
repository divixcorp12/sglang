"""Layout and allocation of the lease block: the completion and map delta blocks.

The lease block is one pinned allocation that sits beside the request page and is
read and written by the RAM-miss service thread, the copy thread and the device
kernels (through UVA). It holds two regions:

  * the completion block (``[0, wire.lease_block_bytes)``): the per-request piece masks, the
    ``CopyDone`` words, the copy-wait gate, the copy-engine armed flag and the CPU
    split table;
  * the map delta block (from ``wire.lease_block_bytes``): one ``wire.delta_stride`` record per
    streamed row, through which the service hands map changes to the device.

The service writes both regions, except the gate, which the copy-wait kernel (CW)
closes. Python writes only the attach-time split, and only through the service's
export.

``WireLayout`` mirrors ``csrc/moe/expert_stream/lease_layout.h`` (under
``python/sglang/kernels/jit/``); ``test_expert_stream_lease_layout`` checks that they agree.
See ``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "Wire (v2)".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch


def _round_up(value: int, align: int) -> int:
    return -(-value // align) * align


@dataclass(frozen=True)
class WireLayout:
    """lease_layout.h's LeaseLayout<lanes, nodes>: the request page, completion block and map delta offsets."""

    lanes: int
    nodes: int

    # Request page.
    demand_ring = 128
    demand_records = 16
    record_id_max = 32767

    @property
    def kind_words(self) -> int:
        return self.lanes // 8

    @property
    def wide_lanes(self) -> bool:
        """Lane masks are u64 (LeaseLayout::kWideLanes): more than 32 lanes."""
        return self.lanes > 32

    @property
    def ce_mask_words(self) -> int:
        return 5 if self.wide_lanes else 3

    @property
    def cpu_lane_words(self) -> int:
        return 3 if self.wide_lanes else 2

    @property
    def packed_counts(self) -> bool:
        return self.lanes == 8

    @property
    def protect_count(self) -> int:
        return 20 + 4 * self.kind_words

    @property
    def header_bytes(self) -> int:
        return _round_up(self.protect_count + (0 if self.packed_counts else 1), 16)

    @property
    def record_fields(self) -> dict[str, int]:
        h, n = self.header_bytes, self.lanes
        return {
            "seq": 0, "row": 4, "counts": 6, "flags": 7, "chain": 8, "epoch": 16, "kinds": 20,
            "protect": h, "lane_expert": h + 2 * n, "lane_slot": h + 4 * n, "lane_dst": h + 6 * n,
            "lane_weight": h + 8 * n,
        }

    @property
    def record_bytes(self) -> int:
        return _round_up(self.record_fields["lane_weight"] + 4 * self.lanes, 128)

    @property
    def page_bytes(self) -> int:
        return self.demand_ring + self.demand_records * self.record_bytes

    # Completion block.
    block_align = 4096
    piece_mask = 0
    piece_mask_line_bytes = 128
    copy_done_bytes = 8

    @property
    def copy_done(self) -> int:
        return self.piece_mask + self.demand_records * self.lanes * self.piece_mask_line_bytes

    @property
    def copy_gate(self) -> int:
        return self.copy_done + 128

    @property
    def copy_armed(self) -> int:
        return self.copy_gate + 128

    @property
    def split(self) -> int:
        return self.copy_armed + 128

    @property
    def split_stride(self) -> int:
        return _round_up(4 * (self.lanes + 1), 16)

    @property
    def lease_block_bytes(self) -> int:
        return _round_up(self.split + self.nodes * self.split_stride, self.block_align)

    # Map delta.
    @property
    def delta_fields(self) -> dict[str, int]:
        return {"tag": 0, "count": 8, "staging": 16, "entries": _round_up(16 + 2 * self.nodes * self.lanes, 16)}

    @property
    def delta_max_entries(self) -> int:
        return 2 * self.lanes

    @property
    def delta_stride(self) -> int:
        return _round_up(self.delta_fields["entries"] + 4 * self.delta_max_entries, 256)

    def home(self, expert: int) -> int:
        """The node whose group serves ``expert``: LeaseLayout::home, the one home rule."""
        return expert % self.nodes

    def cpp_constants(self) -> dict[str, int]:
        """The trait's members by their C++ names, as lease_layout_probe prints them."""
        from sglang.srt.layers.moe.ram_slot_map import LaneKind

        f, d = self.record_fields, self.delta_fields
        return {
            "kLanes": self.lanes, "kNodes": self.nodes, "kDemandHead": 0, "kDemandRing": self.demand_ring,
            "kDemandRecords": self.demand_records, "kRecSeq": f["seq"], "kRecRow": f["row"],
            "kRecCounts": f["counts"], "kRecFlags": f["flags"], "kRecFlagCaptured": 1, "kRecChain": f["chain"],
            "kRecEpoch": f["epoch"], "kRecKinds": f["kinds"], "kKindWords": self.kind_words,
            "kPackedCounts": int(self.packed_counts), "kRecProtectCount": self.protect_count,
            "kRecHeaderBytes": self.header_bytes, "kRecProtect": f["protect"], "kRecLaneExpert": f["lane_expert"],
            "kRecLaneSlot": f["lane_slot"], "kRecLaneDst": f["lane_dst"], "kRecLaneWeight": f["lane_weight"],
            "kRecPayloadEnd": f["lane_weight"] + 4 * self.lanes, "kRecordBytes": self.record_bytes,
            "kRecIdMax": self.record_id_max, "kPageBytes": self.page_bytes, "kKindHitCopy": int(LaneKind.HIT_COPY),
            "kKindHitSm": int(LaneKind.HIT_SM), "kKindHitCpu": int(LaneKind.HIT_CPU),
            "kKindMissGpu": int(LaneKind.MISS_GPU), "kKindMissCpu": int(LaneKind.MISS_CPU), "kHotHeaderBytes": 8, "kHotAlignment": 64,
            "kHotRecords": self.demand_records, "kLeaseBlockAlign": self.block_align,
            "kLeasePieceMask": self.piece_mask, "kLeasePieceMaskLineBytes": self.piece_mask_line_bytes,
            "kLeaseCopyDone": self.copy_done, "kLeaseCopyDoneBytes": self.copy_done_bytes,
            "kLeaseCopyGate": self.copy_gate, "kLeaseGateClosed": GATE["closed"], "kLeaseGateOpen": GATE["open"],
            "kLeaseGateSeqShift": GATE_SEQ_SHIFT, "kLeaseGateSeqMask": GATE_SEQ_MASK, "kCopyArmed": self.copy_armed,
            "kSplit": self.split, "kSplitStride": self.split_stride, "kLeaseBlockBytes": self.lease_block_bytes,
            "kDeltaBase": self.lease_block_bytes, "kDeltaTag": d["tag"], "kDeltaCount": d["count"],
            "kDeltaStaging": d["staging"], "kDeltaEntries": d["entries"],
            "kDeltaMaxEntries": self.delta_max_entries, "kDeltaStride": self.delta_stride, "home7": self.home(7),
        }


MAX_LANES = 64


# The CPU experts' input rows (cpu_token_table.h): CpuTokenTable::kHeaderBytes and kMaxTokens.
CPU_TOKEN_TABLE_HEADER = 16
CPU_TOKENS_MAX = 32


def cpu_row_bytes(hidden: int, tokens: int, lanes: int) -> int:
    """Bytes of one CPU experts input row: ``tokens`` fp16 inputs, each padded to 16 bytes, then, for more than one
    token, the token table: a 16-byte header, a u32 mask per lane and an fp32 weight per token and lane."""
    x = -(-2 * hidden // 16) * 16
    return tokens * x + (CPU_TOKEN_TABLE_HEADER + 4 * lanes + 4 * tokens * lanes if tokens > 1 else 0)


@functools.cache
def wire_layout(lanes: int, nodes: int = 1) -> WireLayout:
    """Return the wire layout for ``lanes`` (rounded up to 8) on ``nodes`` NUMA nodes."""
    if not 1 <= lanes <= MAX_LANES:
        raise ValueError(f"a demand record carries 1..{MAX_LANES} lanes, not {lanes}")
    if nodes < 1:
        raise ValueError(f"the wire needs at least one node, not {nodes}")
    return WireLayout(_round_up(lanes, 8), nodes)


def wire_probe(lanes: int, nodes: int) -> dict[str, int]:
    """Test only: LeaseLayout<lanes, nodes>'s members as the C++ compiler computes them."""
    from sglang.kernels.jit.utils import load_jit

    module = load_jit(
        "expert_stream_lease_layout_probe",
        cpp_files=["moe/expert_stream/lease_layout_probe.cpp"],
        header_only=False,
    )
    text = str(module.expert_stream_lease_layout_probe(lanes, nodes))
    if not text:
        raise ValueError(f"the probe has no LeaseLayout<{lanes}, {nodes}> instantiation")
    return {name: int(value) for name, value in (line.split("=") for line in text.split("\n"))}


def channel_probe(lanes: int, nodes: int) -> dict[str, int]:
    """Test only: the target's lease channel (lease_channel_layout.h, TargetChannelOf<LeaseLayout<lanes, nodes>>) and
    the shared gate encoding, as the C++ compiler computes them."""
    from sglang.kernels.jit.utils import load_jit

    module = load_jit(
        "expert_stream_lease_layout_probe",
        cpp_files=["moe/expert_stream/lease_layout_probe.cpp"],
        header_only=False,
    )
    text = str(module.expert_stream_lease_channel_probe(lanes, nodes))
    if not text:
        raise ValueError(f"the probe has no channel for LeaseLayout<{lanes}, {nodes}>")
    return {name: int(value) for name, value in (line.split("=") for line in text.split("\n"))}


GATE = {"closed": 0x80000001, "open": 1}
GATE_SEQ_SHIFT = 2
GATE_SEQ_MASK = 0x1FFFFFFF

# The map delta block starts at ``wire.lease_block_bytes`` of the same allocation: one ``wire.delta_stride`` record per
# row. The service stores the tag last, with a release, and a zero tag is never a written delta, so a zero-filled record
# cannot be mistaken for a published one.


def gate_word(seq: int, low: str) -> int:
    """Return the copy-wait gate word for request ``seq``; ``low`` is a ``GATE`` key."""
    return ((seq & GATE_SEQ_MASK) << GATE_SEQ_SHIFT) | GATE[low]


def lease_block_bytes(rows: int, *, wire: WireLayout) -> int:
    """Return the lease block size: the completion block plus ``rows`` delta records.

    The delta region starts at ``wire.lease_block_bytes`` and is rounded up to whole pages.
    """
    if rows < 1:
        raise ValueError(f"the lease block needs at least one row, got {rows}")
    return wire.lease_block_bytes + _round_up(rows * wire.delta_stride, wire.block_align)


def new_lease_block(rows: int, *, pin: bool, wire: WireLayout) -> torch.Tensor:
    """Allocate a zeroed, aligned uint8 block of ``lease_block_bytes(rows, wire=wire)`` bytes.

    The block is pinned when ``pin`` is set (a real device). Neither ``torch.zeros``
    nor the pinned allocator promises 4096-byte alignment, so the allocation carries
    one page of slack and the returned view starts at the aligned address; the view
    keeps the storage alive.
    """
    size = lease_block_bytes(rows, wire=wire)
    raw = torch.zeros(size + wire.block_align, dtype=torch.uint8, pin_memory=pin)
    start = (-raw.data_ptr()) % wire.block_align
    block = raw[start : start + size]
    check_lease_block(block, rows, need_pinned=pin, wire=wire)
    return block


def check_lease_block(
    block: torch.Tensor, rows: int, *, need_pinned: bool, wire: WireLayout
) -> None:
    """Raise ValueError for a block the kernels and the service cannot address."""
    if block.dtype != torch.uint8 or block.device.type != "cpu" or block.dim() != 1:
        raise ValueError("the lease block must be a 1-D CPU uint8 tensor")
    if block.numel() != lease_block_bytes(rows, wire=wire):
        raise ValueError(
            f"the lease block has {block.numel()} bytes, not {lease_block_bytes(rows, wire=wire)} for {rows} rows"
        )
    if not block.is_contiguous():
        raise ValueError("the lease block must be contiguous")
    if block.data_ptr() % wire.block_align != 0:
        raise ValueError(
            f"the lease block must be {wire.block_align}-byte aligned, its address is {block.data_ptr():#x}"
        )
    if need_pinned and not block.is_pinned():
        raise ValueError(
            "the lease block must be pinned for a CUDA device: the kernels read it through UVA"
        )
