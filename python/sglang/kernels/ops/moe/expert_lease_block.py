"""Layout and allocation of the lease block, the pinned region beside the RAM-miss request page.

Mirrors ``csrc/moe/expert_stream/lease_layout.h`` (analysis/dsv41-drive/LEASE_PROTOCOL.md); test_exl3_lease_block and
test_exl3_ram_miss_device_args check that the constants agree. The service writes areas S, P and C, the device area D
and the gate; Python writes nothing. The tag encoding (tag << 56 | generation) is code, not a constant, so the C++
constexpr parser of the layout test can stay + - *.
"""

from __future__ import annotations

import torch

# Fixed by the request page: the ring is kDemandRecords, the lane count kMaxIds.
RING = 16
LANES = 8

BLOCK_ALIGN = 4096
BLOCK_BYTES = 28672

# Area S, service-written: RowResult[RING][LANES], {u64 ready = tag << 56 | G; i32 host_slot}.
ROW_RESULT = 0
ROW_RESULT_BYTES = 16
ROW_RESULT_FIELDS = {"ready": 0, "host_slot": 8}

# Area P, service-written: PieceMask[RING][LANES], u64 G << 8 | 8 piece bits, one 128-byte line each.
PIECE_MASK = 4096
PIECE_MASK_LINE_BYTES = 128

# Area C, service-written: CopyDone[RING] (u64 G), then the copy wait's gate on its own line.
COPY_DONE = PIECE_MASK + RING * LANES * PIECE_MASK_LINE_BYTES
COPY_DONE_BYTES = 8
COPY_GATE = COPY_DONE + 128
GATE = {"closed": 0x80000001, "open": 1}
GATE_SEQ_SHIFT = 2
GATE_SEQ_MASK = 0x1FFFFFFF

# Area D, device-written: LaneRequest[RING], then Done[RING] (u64 G: no kernel of G reads a leased slot after it).
LANE_REQUEST = 24576
LANE_REQUEST_BYTES = 128
# expert[LANES] and dst_slot[LANES] are int32 per lane; weight[LANES] each lane expert's fp32 routing weight.
LANE_REQUEST_FIELDS = {"gen": 0, "count": 8, "flags": 12, "expert": 16, "dst_slot": 48, "weight": 80}
# Posted from a captured graph: the service may copy the hits itself, and compute CPU lanes (the input is staged).
LANE_REQUEST_FLAG_CAPTURED = 1
DONE = LANE_REQUEST + RING * LANE_REQUEST_BYTES
DONE_BYTES = 8

# RowResult tags, the byte above the 56-bit generation.
READY = 1
LOADING = 2  # being read: the lane's PieceMask word says which pieces are final
COPYING = 3  # the service's copy engine writes the lane's destination slot
CPU = 4  # the CPU expert thread computes the lane; nothing writes its destination slot
TAG_SHIFT = 56
GENERATION_MASK = (1 << TAG_SHIFT) - 1


def tagged(tag: int, generation: int) -> int:
    """A RowResult ready word: tag in the top byte, request generation (epoch << 32 | seq) below it."""
    if not 0 < generation <= GENERATION_MASK:
        raise ValueError(f"generation {generation} does not fit 56 bits or is zero (zero is never a valid generation)")
    return (tag << TAG_SHIFT) | generation


def gate_word(seq: int, low: str) -> int:
    """The copy-wait gate word for request ``seq``: ``low`` is a GATE key."""
    return ((seq & GATE_SEQ_MASK) << GATE_SEQ_SHIFT) | GATE[low]


def untag(word: int) -> tuple[int, int]:
    """(tag, generation) of a ready word."""
    return (word >> TAG_SHIFT) & 0xFF, word & GENERATION_MASK


def new_lease_block(*, pin: bool) -> torch.Tensor:
    """A zeroed, 4096-aligned uint8 block of BLOCK_BYTES; pinned for a real device.

    The allocator is asked for a page of slack and the view is sliced to the aligned start, since neither
    torch.zeros nor the pinned allocator promises 4096 alignment. The slice keeps the storage alive.
    """
    raw = torch.zeros(BLOCK_BYTES + BLOCK_ALIGN, dtype=torch.uint8, pin_memory=pin)
    start = (-raw.data_ptr()) % BLOCK_ALIGN
    block = raw[start : start + BLOCK_BYTES]
    check_lease_block(block, need_pinned=pin)
    return block


def check_lease_block(block: torch.Tensor, *, need_pinned: bool) -> None:
    """Refuse a block the kernels and the service cannot address."""
    if block.dtype != torch.uint8 or block.device.type != "cpu" or block.dim() != 1:
        raise ValueError("the lease block must be a 1-D CPU uint8 tensor")
    if block.numel() != BLOCK_BYTES:
        raise ValueError(f"the lease block has {block.numel()} bytes, not {BLOCK_BYTES}")
    if not block.is_contiguous():
        raise ValueError("the lease block must be contiguous")
    if block.data_ptr() % BLOCK_ALIGN != 0:
        raise ValueError(f"the lease block must be {BLOCK_ALIGN}-byte aligned, its address is {block.data_ptr():#x}")
    if need_pinned and not block.is_pinned():
        raise ValueError("the lease block must be pinned for a CUDA device: the kernels read it through UVA")
