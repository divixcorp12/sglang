"""Layout and allocation of the completion block and the map delta block, the pinned regions beside the request page.

Mirrors ``csrc/moe/expert_stream/lease_layout.h`` (analysis/dsv41-drive/LEASE_PROTOCOL.md); test_exl3_lease_block and
test_exl3_ram_miss_device_args check that the constants agree. The service writes both blocks except the gate, which CW
closes; Python writes the attach-time split only through the service's export.
"""

from __future__ import annotations

import torch

# Fixed by the request page: the ring is kDemandRecords, the lane count kMaxIds.
RING = 16
LANES = 8

BLOCK_ALIGN = 4096
BLOCK_BYTES = 20480

# PieceMask[RING][LANES], u64 G << 8 | 8 piece bits, one 128-byte line each.
PIECE_MASK = 0
PIECE_MASK_LINE_BYTES = 128

# CopyDone[RING] (u64 G), then the copy wait's gate on its own line.
COPY_DONE = PIECE_MASK + RING * LANES * PIECE_MASK_LINE_BYTES
COPY_DONE_BYTES = 8
COPY_GATE = COPY_DONE + 128
GATE = {"closed": 0x80000001, "open": 1}
GATE_SEQ_SHIFT = 2
GATE_SEQ_MASK = 0x1FFFFFFF
COPY_ARMED = COPY_GATE + 128  # u32: 1 once the service armed its copy engine
SPLIT = COPY_ARMED + 128  # i32[LANES + 1]: CPU lanes per n eligible lanes

# The map delta block: one DELTA_STRIDE record per row, {u64 tag; u32 count; i32 staging[LANES]; {i32 expert,
# i32 slot}[DELTA_MAX_ENTRIES]}. The tag is stored last with a release; a zero tag is never a written delta.
DELTA_STRIDE = 256
DELTA_FIELDS = {"tag": 0, "count": 8, "staging": 16, "entries": 48}
DELTA_MAX_ENTRIES = 16


def gate_word(seq: int, low: str) -> int:
    """The copy-wait gate word for request ``seq``: ``low`` is a GATE key."""
    return ((seq & GATE_SEQ_MASK) << GATE_SEQ_SHIFT) | GATE[low]


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


def new_delta_block(rows: int, *, pin: bool) -> torch.Tensor:
    """A zeroed, 4096-aligned uint8 block of rows * DELTA_STRIDE (whole pages); pinned for a real device."""
    if rows < 1:
        raise ValueError(f"the delta block needs at least one row, got {rows}")
    size = -(-rows * DELTA_STRIDE // BLOCK_ALIGN) * BLOCK_ALIGN
    raw = torch.zeros(size + BLOCK_ALIGN, dtype=torch.uint8, pin_memory=pin)
    start = (-raw.data_ptr()) % BLOCK_ALIGN
    return raw[start : start + size]
