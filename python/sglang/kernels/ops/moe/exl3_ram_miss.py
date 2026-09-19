"""Option C RAM-miss service for EXL3 streamed experts: the C++ host module's wrappers.

Page layout and memory ordering are the plan's Design decisions D10-D11. The
device kernels (Task 13) and the host simulator (Task 11) speak the same protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

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


def read_rows_once(tables, row: int, experts, slots, *, direct: bool) -> int:
    """Read ``experts`` of streamed row ``row`` into pinned ``slots`` in C++: 1 ok, 0 failed."""
    return int(
        _host_module().exl3_ram_miss_read_rows(
            tables.reads,
            tables.file_sizes,
            tables.segments,
            tables.slabs,
            tables.row_bytes,
            "\n".join(tables.paths),
            tables.slot_bytes,
            int(direct),
            row,
            _ids(experts),
            _ids(slots),
        )
    )
