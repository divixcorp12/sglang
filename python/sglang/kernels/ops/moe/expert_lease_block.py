"""Layout and allocation of the lease block: the completion and map delta blocks.

The lease block is one pinned allocation that sits beside the request page and is
read and written by the RAM-miss service thread, the copy thread and the device
kernels (through UVA). It holds two regions:

  * the completion block (``[0, BLOCK_BYTES)``): the per-request piece masks, the
    ``CopyDone`` words, the copy-wait gate, the copy-engine armed flag and the CPU
    split table;
  * the map delta block (from ``DELTA_BASE``): one ``DELTA_STRIDE`` record per
    streamed row, through which the service hands map changes to the device.

The service writes both regions, except the gate, which the copy-wait kernel (CW)
closes. Python writes only the attach-time split, and only through the service's
export.

The constants mirror ``csrc/moe/expert_stream/lease_layout.h`` (under
``python/sglang/kernels/jit/``); ``test_exl3_lease_block`` and
``test_exl3_ram_miss_device_args`` check that they agree.
See ``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "Wire (v2)".
"""

from __future__ import annotations

import torch

# Fixed by the request page: the ring is kDemandRecords, the lane count kMaxIds.
RING = 16
LANES = 8

BLOCK_ALIGN = 4096
BLOCK_BYTES = 20480

# Completion block, in offset order. Each word below sits on its own 128-byte line.
#
# PieceMask[RING][LANES]: u64 ``generation << 8 | 8 piece bits``, one line each.
PIECE_MASK = 0
PIECE_MASK_LINE_BYTES = 128

# CopyDone[RING]: u64 generation of the last request whose host lanes all landed.
COPY_DONE = PIECE_MASK + RING * LANES * PIECE_MASK_LINE_BYTES
COPY_DONE_BYTES = 8
# The copy-wait gate: closed while the decode stream waits for the copy thread.
COPY_GATE = COPY_DONE + 128
GATE = {"closed": 0x80000001, "open": 1}
GATE_SEQ_SHIFT = 2
GATE_SEQ_MASK = 0x1FFFFFFF
COPY_ARMED = COPY_GATE + 128  # u32: 1 once the service armed its copy engine
SPLIT = COPY_ARMED + 128  # i32[LANES + 1]: CPU lanes per n eligible lanes

# The map delta block, at DELTA_BASE of the same allocation: one DELTA_STRIDE record
# per row, laid out as
#   {u64 tag; u32 count; i16 staging[LANES] @16;
#    {i16 expert, i16 slot}[DELTA_MAX_ENTRIES] @32}.
# The service stores the tag last, with a release, and a zero tag is never a written
# delta, so a zero-filled record cannot be mistaken for a published one.
DELTA_BASE = BLOCK_BYTES
DELTA_STRIDE = 256
DELTA_FIELDS = {"tag": 0, "count": 8, "staging": 16, "entries": 32}
DELTA_MAX_ENTRIES = 16


def gate_word(seq: int, low: str) -> int:
    """Return the copy-wait gate word for request ``seq``; ``low`` is a ``GATE`` key."""
    return ((seq & GATE_SEQ_MASK) << GATE_SEQ_SHIFT) | GATE[low]


def lease_block_bytes(rows: int) -> int:
    """Return the lease block size: the completion block plus ``rows`` delta records.

    The delta region starts at ``DELTA_BASE`` and is rounded up to whole pages.
    """
    if rows < 1:
        raise ValueError(f"the lease block needs at least one row, got {rows}")
    return BLOCK_BYTES + -(-rows * DELTA_STRIDE // BLOCK_ALIGN) * BLOCK_ALIGN


def new_lease_block(rows: int, *, pin: bool) -> torch.Tensor:
    """Allocate a zeroed, aligned uint8 block of ``lease_block_bytes(rows)`` bytes.

    The block is pinned when ``pin`` is set (a real device). Neither ``torch.zeros``
    nor the pinned allocator promises 4096-byte alignment, so the allocation carries
    one page of slack and the returned view starts at the aligned address; the view
    keeps the storage alive.
    """
    size = lease_block_bytes(rows)
    raw = torch.zeros(size + BLOCK_ALIGN, dtype=torch.uint8, pin_memory=pin)
    start = (-raw.data_ptr()) % BLOCK_ALIGN
    block = raw[start : start + size]
    check_lease_block(block, rows, need_pinned=pin)
    return block


def check_lease_block(block: torch.Tensor, rows: int, *, need_pinned: bool) -> None:
    """Raise ValueError for a block the kernels and the service cannot address."""
    if block.dtype != torch.uint8 or block.device.type != "cpu" or block.dim() != 1:
        raise ValueError("the lease block must be a 1-D CPU uint8 tensor")
    if block.numel() != lease_block_bytes(rows):
        raise ValueError(
            f"the lease block has {block.numel()} bytes, not {lease_block_bytes(rows)} for {rows} rows"
        )
    if not block.is_contiguous():
        raise ValueError("the lease block must be contiguous")
    if block.data_ptr() % BLOCK_ALIGN != 0:
        raise ValueError(
            f"the lease block must be {BLOCK_ALIGN}-byte aligned, its address is {block.data_ptr():#x}"
        )
    if need_pinned and not block.is_pinned():
        raise ValueError(
            "the lease block must be pinned for a CUDA device: the kernels read it through UVA"
        )
