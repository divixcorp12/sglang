"""Layout and allocation of the lease block, the pinned region beside the option C request page.

The layout is fixed by analysis/dsv41-drive/LEASE_PROTOCOL.md section 4. It is written three times: here,
in exl3_ram_miss_host.cpp and in exl3_ram_miss.cuh, and test_exl3_lease_block checks that the constants agree.
Every 128-byte line has one writer: the service writes areas H and S, the device writes area D, Python writes
nothing. The tag encoding (tag << 56 | generation) is code, not a constant, so the C++ constexpr parser of the
layout test can stay + - *.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Fixed by the request page: the ring is kDemandRecords, the lane count kMaxIds.
RING = 16
LANES = 8

BLOCK_ALIGN = 4096
HEADER_BYTES = 128
# Offsets of the header words (u32), all service-written.
HEADER = {
    "magic": 0,
    "abi_version": 4,
    "ring": 8,
    "lanes": 12,
    "rows": 16,
    "shutdown": 20,
    "slot_gen_offset": 32,
    "d_offset": 36,
    "piece_offset": 40,
    "copy_offset": 44,
}
MAGIC = 0x4C534531  # "LSE1"
# 2: StreamProbe appended to area D (piece streaming). 3: LaneRequest grown to 128 bytes (destination slots and
# flags) and area C (CopyDone) appended (copy engine). A block of another version is refused at attach.
ABI_VERSION = 3

ROW_TABLE = HEADER_BYTES  # rows * {u32 slot_gen_base; u32 capacity}
ROW_TABLE_ENTRY_BYTES = 8
MAX_ROWS = (BLOCK_ALIGN - ROW_TABLE) // ROW_TABLE_ENTRY_BYTES

# Area S, service-written: RowResult[RING][LANES], then SlotGen[] (u32 per (row, slot)).
ROW_RESULT = BLOCK_ALIGN
ROW_RESULT_BYTES = 32
# Field offsets inside one RowResult.
ROW_RESULT_FIELDS = {"ready": 0, "slot_generation": 8, "host_slot": 12, "expert": 16, "row": 20, "lane": 22}
SLOT_GEN = ROW_RESULT + RING * LANES * ROW_RESULT_BYTES

# Area D, device-written, at d_offset: LaneRequest[RING], LaneAck[RING][LANES], Terminal[RING], StreamProbe[RING].
LANE_REQUEST = 0
LANE_REQUEST_BYTES = 128
# expert[LANES] and dst_slot[LANES] are int32 per lane; flags is a u32 of LANE_REQUEST_FLAGS bits.
LANE_REQUEST_FIELDS = {"gen": 0, "count": 8, "row": 12, "expert": 16, "dst_slot": 48, "flags": 80}
# The service may copy this request's resident lanes with its copy engine (the device passes it only when capturing).
LANE_REQUEST_FLAG_COPY_ENGINE = 1
LANE_ACK = LANE_REQUEST + RING * LANE_REQUEST_BYTES
LANE_ACK_BYTES = 8
TERMINAL = LANE_ACK + RING * LANES * LANE_ACK_BYTES
TERMINAL_BYTES = 16
TERMINAL_FIELDS = {"skipped_mask": 0, "reason": 4, "gen": 8}
# tagged(STREAM_PROBE_TAG, generation) once the stream kernel has copied a piece of that request: piece streaming's
# one device-to-host progress word (the kernels' state words live in device memory).
STREAM_PROBE = TERMINAL + RING * TERMINAL_BYTES
STREAM_PROBE_BYTES = 8
AREA_D_BYTES = STREAM_PROBE + RING * STREAM_PROBE_BYTES

# Area P, service-written, at a new header offset (piece_offset): PieceMask[RING][LANES], a per-lane
# generation-tagged 8-bit readiness bitmask (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment). Each
# word gets its own 128 B line rather than packing densely, so the device's per-lane poll never shares a
# line with a lane it did not ask for.
PIECE_MASK_LINE_BYTES = 128
PIECE_MASK_BYTES = 8  # one uint64 per word
AREA_PIECE_MASK_BYTES = RING * LANES * PIECE_MASK_LINE_BYTES

# Area C, service-written, at copy_offset (copy engine, LEASE_PROTOCOL.md 7.6): CopyDone[RING], the lane mask, then
# tagged(COPIED, generation) stored last once the service observed the request's copy-engine copies complete.
COPY_DONE_BYTES = 16
COPY_DONE_FIELDS = {"mask": 0, "gen": 8}
AREA_COPY_DONE_BYTES = RING * COPY_DONE_BYTES

# Tags of the 8-bit field above the 56-bit generation.
READY = 1  # RowResult.ready
LOADING = 2  # RowResult.ready: leased, still loading (piece-streaming plan; task 1)
COPYING = 3  # RowResult.ready: leased, the service's copy engine writes the lane's destination slot
COPIED = 1  # CopyDone.gen
CONSUMED, VIOLATED = 1, 2  # LaneAck
DEMAND_TAG = 1  # LaneRequest.gen, written by the post kernel
TERMINAL_TAG = 1  # Terminal.gen, written by the wait kernel
STREAM_PROBE_TAG = 1  # StreamProbe, written by the stream kernel
# Terminal.reason, written by the wait kernel (the service does not interpret it; it is for the trace and the tests).
TERMINAL_REASONS = {"timeout": 1, "aborted": 2, "failed": 3, "identity": 4, "count": 5}
TAG_SHIFT = 56
GENERATION_MASK = (1 << TAG_SHIFT) - 1


def _align(value: int) -> int:
    return (value + BLOCK_ALIGN - 1) // BLOCK_ALIGN * BLOCK_ALIGN


def tagged(tag: int, generation: int) -> int:
    """The u64 publication word: tag in the top byte, request generation (epoch << 32 | seq) below it."""
    if not 0 < generation <= GENERATION_MASK:
        raise ValueError(f"generation {generation} does not fit 56 bits or is zero (zero is never a valid generation)")
    return (tag << TAG_SHIFT) | generation


def untag(word: int) -> tuple[int, int]:
    """(tag, generation) of a publication word."""
    return (word >> TAG_SHIFT) & 0xFF, word & GENERATION_MASK


@dataclass(frozen=True)
class LeaseLayout:
    rows: int
    capacities: tuple[int, ...]
    slot_gen_base: tuple[int, ...]  # first SlotGen word of each row, in words
    slot_gen_offset: int  # bytes from the block base
    d_offset: int  # bytes from the block base to area D
    piece_offset: int  # bytes from the block base to area P (PieceMask)
    copy_offset: int  # bytes from the block base to area C (CopyDone)
    total_bytes: int


def lease_layout(capacities: Sequence[int]) -> LeaseLayout:
    """Offsets of a lease block whose row ``r`` has ``capacities[r]`` pinned slots."""
    capacities = tuple(int(c) for c in capacities)
    if not 0 < len(capacities) <= MAX_ROWS:
        raise ValueError(f"the row table holds 1..{MAX_ROWS} rows, got {len(capacities)}")
    if any(c < 1 for c in capacities):
        raise ValueError(f"every row needs at least one slot, got {capacities}")
    bases, total_slots = [], 0
    for capacity in capacities:
        bases.append(total_slots)
        total_slots += capacity
    d_offset = _align(SLOT_GEN + 4 * total_slots)
    piece_offset = _align(d_offset + AREA_D_BYTES)
    copy_offset = _align(piece_offset + AREA_PIECE_MASK_BYTES)
    return LeaseLayout(
        rows=len(capacities),
        capacities=capacities,
        slot_gen_base=tuple(bases),
        slot_gen_offset=SLOT_GEN,
        d_offset=d_offset,
        piece_offset=piece_offset,
        copy_offset=copy_offset,
        total_bytes=_align(copy_offset + AREA_COPY_DONE_BYTES),
    )


def new_lease_block(layout: LeaseLayout, *, pin: bool) -> torch.Tensor:
    """A zeroed, 4096-aligned uint8 block of ``layout.total_bytes``; pinned for a real device.

    The allocator is asked for a page of slack and the view is sliced to the aligned start, since neither
    torch.zeros nor the pinned allocator promises 4096 alignment. The slice keeps the storage alive.
    """
    raw = torch.zeros(layout.total_bytes + BLOCK_ALIGN, dtype=torch.uint8, pin_memory=pin)
    start = (-raw.data_ptr()) % BLOCK_ALIGN
    block = raw[start : start + layout.total_bytes]
    check_lease_block(block, layout, need_pinned=pin)
    return block


def check_lease_block(block: torch.Tensor, layout: LeaseLayout, *, need_pinned: bool) -> None:
    """Refuse a block the kernels and the service cannot address. There is no alignment check on the request page."""
    if block.dtype != torch.uint8 or block.device.type != "cpu" or block.dim() != 1:
        raise ValueError("the lease block must be a 1-D CPU uint8 tensor")
    if block.numel() != layout.total_bytes:
        raise ValueError(f"the lease block has {block.numel()} bytes, its layout needs {layout.total_bytes}")
    if not block.is_contiguous():
        raise ValueError("the lease block must be contiguous")
    if block.data_ptr() % BLOCK_ALIGN != 0:
        raise ValueError(f"the lease block must be {BLOCK_ALIGN}-byte aligned, its address is {block.data_ptr():#x}")
    if need_pinned and not block.is_pinned():
        raise ValueError("the lease block must be pinned for a CUDA device: the kernels read it through UVA")
