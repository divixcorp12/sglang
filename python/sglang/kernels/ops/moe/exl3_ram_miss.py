"""Option C RAM-miss service for EXL3 streamed experts: the C++ host module's wrappers.

Page layout and memory ordering are the plan's Design decisions D10-D11. The
device kernels (Task 13) and the host simulator (Task 11) speak the same protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

# Rows per io_uring batch: the C++ bounce holds kBounceRows = 8 rows.
BOUNCE_ROWS = 8

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _host_module() -> Module:
    return load_jit(
        "exl3_ram_miss_host",
        cpp_files=["moe/exl3_ram_miss_host.cpp"],
        extra_ldflags=["-luring", "-lpthread"],
        header_only=False,
    )


def _ids(values: Iterable[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64)


def _checked_rows(tables, row: int, experts, slots) -> tuple[torch.Tensor, torch.Tensor]:
    """``experts`` and ``slots`` as int64 tensors, after the bounds checks the C++ hot path skips."""
    expert_ids, slot_ids = _ids(experts), _ids(slots)
    if expert_ids.numel() != slot_ids.numel():
        raise ValueError(f"{expert_ids.numel()} experts but {slot_ids.numel()} slots")
    if not 0 <= row < len(tables.layer_ids):
        raise ValueError(f"streamed row {row} is outside [0, {len(tables.layer_ids)})")
    num_experts = int(tables.reads.shape[1])
    if expert_ids.numel() and not (0 <= int(expert_ids.min()) and int(expert_ids.max()) < num_experts):
        raise ValueError(f"expert ids {expert_ids.tolist()} are outside [0, {num_experts})")
    capacity = int(tables.capacity[row])
    if slot_ids.numel() and not (0 <= int(slot_ids.min()) and int(slot_ids.max()) < capacity):
        raise ValueError(f"slots {slot_ids.tolist()} are outside [0, {capacity})")
    return expert_ids, slot_ids


def _table_args(tables, direct: bool) -> tuple:
    return (
        tables.reads,
        tables.file_sizes,
        tables.segments,
        tables.slabs,
        tables.row_bytes,
        "\n".join(tables.paths),
        tables.slot_bytes,
        int(direct),
    )


def read_rows_once(tables, row: int, experts, slots, *, direct: bool, step: int = BOUNCE_ROWS) -> int:
    """Read ``experts`` of streamed row ``row`` into pinned ``slots`` in C++: 1 ok, 0 failed.

    ``step`` rows go to io_uring per batch (at most ``BOUNCE_ROWS``).
    """
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    return int(
        _host_module().exl3_ram_miss_read_rows(
            *_table_args(tables, direct), row, expert_ids, slot_ids, int(step)
        )
    )


def read_rows_with_fault(
    tables,
    row: int,
    experts,
    slots,
    then_experts,
    then_slots,
    *,
    direct: bool,
    submit_error: int = 0,
    submit_call: int = 0,
    submit_first: bool = False,
    cqe_error: int = 0,
    cqe_call: int = 0,
) -> tuple[int, int]:
    """Test only: on one C++ reader, read with an injected io_uring fault, then read cleanly.

    ``submit_error`` (an errno) replaces the result of the ``submit_call``-th submit-and-wait
    (after submitting the prepared reads when ``submit_first``); ``cqe_error`` replaces the
    ``cqe_call``-th completion's result. Returns both reads' results (1 ok, 0 failed).
    """
    first = _checked_rows(tables, row, experts, slots)
    then = _checked_rows(tables, row, then_experts, then_slots)
    fault = torch.tensor([submit_error, submit_call, int(submit_first), cqe_error, cqe_call], dtype=torch.int64)
    results = torch.zeros(2, dtype=torch.int64)
    _host_module().exl3_ram_miss_read_rows_faulted(
        *_table_args(tables, direct), row, *first, *then, fault, results
    )
    return int(results[0]), int(results[1])
