"""Option C RAM-miss service for EXL3 streamed experts: the C++ host module's wrappers.

Page layout and memory ordering are the plan's Design decisions D10-D11. The
device kernels (Task 13) and the host simulator (Task 11) speak the same protocol.
"""

from __future__ import annotations

import atexit
import json
import sys
import weakref
from typing import TYPE_CHECKING, Iterable, Optional

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


PAGE_BYTES = 10304
RECORD_BYTES = 128
DEMAND_RING = 64
DEMAND_RECORDS = 16
ADVISE_RING = DEMAND_RING + DEMAND_RECORDS * RECORD_BYTES
ADVISE_RECORDS = 64
MAX_IDS = 8
WORDS = {
    "demand_head": 0,
    "demand_done": 4,
    "fatal": 8,
    "stop": 12,
    "advise_head": 16,
    "advise_done": 20,
    "busy_seq": 24,
    "heartbeat": 28,
}
STATUS = {"pending": 0, "served": 1, "failed": 2}
COUNTERS = (
    "served",
    "touch_only",
    "rows_read",
    "read_errors",
    "evictions",
    "overruns",
    "advisories",
    "advisories_skipped",
    "advisory_rows",
    "late_after_fatal",
    "no_victim",
    "version",
    "running",
    "spin_cpu",
)


def new_page(pin: bool) -> torch.Tensor:
    """A zeroed request page; pinned (device-readable through UVA) for a real device."""
    return torch.zeros(PAGE_BYTES, dtype=torch.uint8, pin_memory=pin)


def page_word(page: torch.Tensor, name: str) -> int:
    offset = WORDS[name]
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


def sim_post(page, row: int, need, protect, *, advisory: bool = False, after: int = 0) -> int:
    """Post a record as the device post kernel does; returns its sequence."""
    return int(_host_module().exl3_ram_miss_sim_post(page, row, _ids(need), _ids(protect), int(advisory), after))


def sim_wait(page, seq: int, timeout_s: float) -> int:
    """Wait as the device wait kernel does: 1 served, 2 failed, 0 timed out, 3 fatal already raised."""
    return int(_host_module().exl3_ram_miss_sim_wait(page, seq, int(timeout_s * 1e9)))


_LIVE: weakref.WeakSet[Exl3RamMissHost] = weakref.WeakSet()


@atexit.register
def _stop_live() -> None:
    for host in list(_LIVE):
        host.stop()


class Exl3RamMissHost:
    """The C++-owned pinned-slot bookkeeping of every streamed layer and its request service.

    ``tables``: ``Exl3RamMissTables``; ``page``: a ``new_page`` tensor; ``slot_map``:
    int32 ``[layers, experts]`` filled with -1 (pinned for a real device). Row ``r`` is
    streamed layer ``tables.layer_ids[r]``. Without a thread (Task 12) requests are
    served only by ``pump()``.
    """

    def __init__(self, tables, *, page: torch.Tensor, slot_map: torch.Tensor, direct: bool) -> None:
        if page.numel() != PAGE_BYTES or page.dtype != torch.uint8 or page.device.type != "cpu":
            raise ValueError("page must be a CPU uint8 tensor of PAGE_BYTES")
        if slot_map.dtype != torch.int32 or tuple(slot_map.shape) != tuple(tables.reads.shape[:2]):
            raise ValueError("slot_map must be int32 [layers, experts]")
        self._module = _host_module()
        # The C++ service writes through raw addresses of the page, the slot map and the
        # slabs (``tables.keepalive``): this object holds all three, and the finalizer
        # below closes the service before they can be released.
        self.tables = tables
        self.page = page
        self.slot_map = slot_map
        self.layers, self.experts = tables.reads.shape[:2]
        self.handle = int(
            self._module.exl3_ram_miss_open(
                page, slot_map, tables.reads, tables.file_sizes, tables.segments, tables.slabs,
                tables.row_bytes, tables.capacity, "\n".join(tables.paths), tables.slot_bytes, int(direct),
            )
        )
        if self.handle < 0:
            raise RuntimeError("exl3 RAM miss service failed to open (files, io_uring or bounce)")
        self._close = weakref.finalize(self, self._module.exl3_ram_miss_close, self.handle)
        self._close.atexit = False  # _stop_live closes live hosts at exit, logging counters first
        _LIVE.add(self)

    def _check(self, row: int, expert: Optional[int] = None, slot: Optional[int] = None) -> None:
        """The bounds the C++ bookkeeping does not check."""
        if not 0 <= row < self.layers:
            raise ValueError(f"streamed row {row} is outside [0, {self.layers})")
        if expert is not None and not 0 <= expert < self.experts:
            raise ValueError(f"expert {expert} is outside [0, {self.experts})")
        capacity = int(self.tables.capacity[row])
        if slot is not None and not 0 <= slot < capacity:
            raise ValueError(f"slot {slot} is outside [0, {capacity})")

    def pump(self) -> int:
        return int(self._module.exl3_ram_miss_pump(self.handle))

    def contains(self, row: int, expert: int) -> bool:
        self._check(row, expert)
        return bool(self._module.exl3_ram_miss_contains(self.handle, row, expert))

    def touch(self, row: int, expert: int) -> None:
        self._check(row, expert)
        self._module.exl3_ram_miss_touch(self.handle, row, expert)

    def assign(self, row: int, expert: int, protected: Iterable[int] = (), protected_fallback: bool = True) -> tuple[int, Optional[int]]:
        self._check(row, expert)
        out = torch.zeros(2, dtype=torch.int64)
        self._module.exl3_ram_miss_assign(self.handle, row, expert, _ids(protected), int(protected_fallback), out)
        slot, evicted = int(out[0]), int(out[1])
        if evicted == -2:
            raise ValueError(f"expert {expert} already holds a pinned slot")
        if slot < 0:
            raise RuntimeError("every pinned host slot holds a protected expert")
        return slot, (None if evicted < 0 else evicted)

    def release(self, row: int, slot: int) -> None:
        self._check(row, slot=slot)
        self._module.exl3_ram_miss_release(self.handle, row, slot)

    def mapping(self, row: int) -> list[int]:
        self._check(row)
        out = torch.empty(self.experts, dtype=torch.int64)
        self._module.exl3_ram_miss_mapping(self.handle, row, out)
        return out.tolist()

    def slot_to_expert(self, row: int) -> list[int]:
        self._check(row)
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        self._module.exl3_ram_miss_slot_to_expert(self.handle, row, out)
        return out.tolist()

    def lru_order(self, row: int) -> list[int]:
        self._check(row)
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        count = int(self._module.exl3_ram_miss_lru_order(self.handle, row, out))
        return out[:count].tolist()

    def set_hot(self, row: int, experts: Iterable[int]) -> None:
        self._check(row)
        self._module.exl3_ram_miss_set_hot(self.handle, row, _ids(e for e in experts if e >= 0))

    def version(self) -> int:
        return self.counters()["version"]

    def inject(self, delay_s: float = 0.0, fail_reads: bool = False, delay_after_demands: int = 0) -> None:
        """Test-only faults (see RamTier::inject)."""
        self._module.exl3_ram_miss_inject(self.handle, int(delay_s * 1e9), int(fail_reads), delay_after_demands)

    def counters(self) -> dict[str, int]:
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.exl3_ram_miss_counters(self.handle, out)
        return dict(zip(COUNTERS, out.tolist()))

    def layer_rows(self) -> list[int]:
        """Rows read for demands, per streamed layer: the RAM misses behind ``f``."""
        out = torch.zeros(self.layers, dtype=torch.int64)
        self._module.exl3_ram_miss_layer_rows(self.handle, 0, out)
        return out.tolist()

    def layer_advisory_rows(self) -> list[int]:
        out = torch.zeros(self.layers, dtype=torch.int64)
        self._module.exl3_ram_miss_layer_rows(self.handle, 1, out)
        return out.tolist()

    def fatal_seq(self) -> int:
        return page_word(self.page, "fatal")

    def stop(self) -> None:
        close = getattr(self, "_close", None)
        if close is not None and close.alive:
            # One line for the window's records (the corpus arms grep it).
            sys.stderr.write("exl3 RAM miss thread counters " + json.dumps(self.counters()) + "\n")
            close()
