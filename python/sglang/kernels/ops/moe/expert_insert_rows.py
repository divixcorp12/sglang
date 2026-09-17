"""Masked device-to-device row insert for the MoE hot cache's residency boundary.

The boundary copies a fixed ``miss_rows`` lanes per tensor per layer whatever
the insertion count, because every decision is gate-masked so that no count
crosses to the host. The eager form of that is
``rows.index_copy_(0, destinations, rows.index_select(0, sources))``, which
makes two passes over every lane -- a full gather into a temporary, then a full
scatter out of it -- and pays both passes for lanes that are inserting nothing.

This kernel does the same work in one pass, and reads the per-lane active flag
*inside* the kernel so an inactive lane exits before issuing a single load.
Traffic is therefore proportional to the real insertion count while the launch
shape stays fixed and no count is ever read on the host, which is what keeps it
capturable in the decode graph.

Safety contract (the caller must hold it; see ``insert_expert_rows``):

* ``rows`` is both source and destination, so the source rows and the
  destination rows of the *active* lanes must be disjoint sets, and no two
  active lanes may name the same destination. The boundary satisfies this
  because its sources are scratch rows and its destinations are cache slots.
* Inactive lanes are never read, so their source/destination entries may hold
  anything in range.

Why this is not a persistent kernel. ``sglang.srt.layers.hc_mix_triton`` fuses
its chain into one persistent kernel and caps the grid at one CTA per SM, which
it must do because its CTAs synchronise through a software grid barrier and a
non-resident CTA would deadlock it. That pattern was considered here and does
not apply: these lanes share nothing -- disjoint reads, disjoint writes, no
ordering between them -- so there is no device-wide barrier to need, and the
occupancy cap the barrier forces would only starve a kernel whose entire job is
to saturate HBM. Many small CTAs is the right shape for this one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Bytes per program in the row dimension. Measured on an RTX 5090 against a
# 4.9 GiB working set (48 layer tensors, so no L2 reuse between layers); see
# the regime note in `sglang.srt.layers.moe.expert_residency_gpu`.
INSERT_ROW_BLOCK = 4096


@triton.jit
def _insert_expert_rows_kernel(
    rows_ptr,
    source_ids,
    destination_ids,
    active_ptr,
    row_bytes,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)
    # Before any load: an idle lane costs a launched program and one byte read,
    # not two passes over a row.
    if tl.load(active_ptr + lane) != 0:
        source_row = tl.load(source_ids + lane).to(tl.int64)
        destination_row = tl.load(destination_ids + lane).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < row_bytes
        values = tl.load(
            rows_ptr + source_row * row_bytes + offsets, mask=mask, other=0
        )
        tl.store(
            rows_ptr + destination_row * row_bytes + offsets, values, mask=mask
        )


def insert_expert_rows(
    rows: torch.Tensor,
    source_ids: torch.Tensor,
    destination_ids: torch.Tensor,
    active: torch.Tensor,
    block: int = INSERT_ROW_BLOCK,
) -> None:
    """Copy ``rows[source_ids[i]] -> rows[destination_ids[i]]`` for active lanes.

    ``rows`` is a 2D contiguous byte view of one cache tensor; ``source_ids``,
    ``destination_ids`` and ``active`` are 1D device tensors of one entry per
    lane. ``active`` is read on the device, so the grid does not depend on it
    and this is safe to capture inside a CUDA graph.

    Checks are metadata-only and never sync. They are explicit raises rather
    than asserts so they survive ``python -O`` and a bad plan fails loudly
    instead of corrupting a cache row.
    """
    if rows.dim() != 2 or rows.stride(1) != 1 or rows.stride(0) != rows.shape[1]:
        raise ValueError(
            "insert_expert_rows needs a contiguous 2D row view, got shape "
            f"{tuple(rows.shape)} stride {tuple(rows.stride())}"
        )
    lanes = source_ids.numel()
    if destination_ids.numel() != lanes or active.numel() != lanes:
        raise ValueError(
            "source_ids, destination_ids and active must name the same lanes, got "
            f"{lanes}, {destination_ids.numel()}, {active.numel()}"
        )
    for name, tensor in (
        ("source_ids", source_ids),
        ("destination_ids", destination_ids),
        ("active", active),
    ):
        if tensor.device != rows.device:
            raise ValueError(f"{name} must live on the same device as rows")
        if tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must be an integer tensor, got {tensor.dtype}")
    if lanes == 0:
        return
    row_bytes = rows.shape[1]
    _insert_expert_rows_kernel[(lanes, triton.cdiv(row_bytes, block))](
        rows,
        source_ids,
        destination_ids,
        active,
        row_bytes,
        BLOCK=block,
    )
