"""Option C RAM-miss service for EXL3 streamed experts: the C++ host module's and the device kernels' wrappers.

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
    num_experts = int(tables.starts.shape[1])
    if expert_ids.numel() and not (0 <= int(expert_ids.min()) and int(expert_ids.max()) < num_experts):
        raise ValueError(f"expert ids {expert_ids.tolist()} are outside [0, {num_experts})")
    capacity = int(tables.capacity[row])
    if slot_ids.numel() and not (0 <= int(slot_ids.min()) and int(slot_ids.max()) < capacity):
        raise ValueError(f"slots {slot_ids.tolist()} are outside [0, {capacity})")
    return expert_ids, slot_ids


def _table_args(tables, direct: bool) -> tuple:
    return (
        tables.extents,
        tables.starts,
        tables.file_sizes,
        tables.segments,
        tables.slabs,
        tables.row_bytes,
        "\n".join(tables.paths),
        "\n".join(tables.source_paths),
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


def read_rows_traced(
    tables,
    row: int,
    experts,
    slots,
    *,
    direct: bool,
    step: int = BOUNCE_ROWS,
    submit_error: int = 0,
    submit_call: int = 0,
    submit_first: bool = False,
    cqe_error: int = 0,
    cqe_call: int = 0,
    part: int = -1,
    part_error: int = 0,
    part_short: int = 0,
    reverse_cqes: bool = False,
    max_outstanding: int = 0,
) -> tuple[int, dict]:
    """Test only: ``read_rows_once`` (with the fault arguments of ``read_rows_with_fault``) that also
    returns the reader's stage record, decoded by ``stage_records``. The reader-side stages only:
    the request-side ones (observed, reserved, mapped, done) are the tier's and stay 0."""
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    fault = torch.tensor(
        [
            submit_error,
            submit_call,
            int(submit_first),
            cqe_error,
            cqe_call,
            part,
            part_error,
            part_short,
            int(reverse_cqes),
            max_outstanding,
        ],
        dtype=torch.int64,
    )
    record = torch.zeros(_stage_words(), dtype=torch.int64)
    result = int(
        _host_module().exl3_ram_miss_read_rows_traced(
            *_table_args(tables, direct), row, expert_ids, slot_ids, int(step), fault, record
        )
    )
    return result, stage_records(record.unsqueeze(0))[0]


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
    part: int = -1,
    part_error: int = 0,
    part_short: int = 0,
    reverse_cqes: bool = False,
    max_outstanding: int = 0,
    cqes: Optional[list[int]] = None,
) -> tuple[int, int]:
    """Test only: on one C++ reader, read with an injected io_uring fault, then read cleanly.

    ``submit_error`` (an errno) replaces the result of the ``submit_call``-th submit-and-wait
    (after submitting the prepared reads when ``submit_first``); ``cqe_error`` replaces the
    ``cqe_call``-th completion's result. ``part`` (a part index) targets the first completion of a
    part-``part`` extent: ``part_error`` (an errno) replaces its result, ``part_short`` (a block
    multiple) caps the bytes it reports, so only that extent is resubmitted. ``reverse_cqes``
    processes each reaped batch of completions back to front, which must not change any result:
    the reader keys every extent by its own ``user_data`` and the kernel orders nothing.
    ``max_outstanding`` caps outstanding reads below the ring's depth, so a batch needs several
    refill rounds; production constants keep a batch inside the ring, so credit never binds
    there. Returns
    both reads' results (1 ok, 0 failed); ``cqes``, if given, receives the completions reaped
    after each read.
    """
    first = _checked_rows(tables, row, experts, slots)
    then = _checked_rows(tables, row, then_experts, then_slots)
    fault = torch.tensor(
        [
            submit_error,
            submit_call,
            int(submit_first),
            cqe_error,
            cqe_call,
            part,
            part_error,
            part_short,
            int(reverse_cqes),
            max_outstanding,
        ],
        dtype=torch.int64,
    )
    results = torch.zeros(4, dtype=torch.int64)
    _host_module().exl3_ram_miss_read_rows_faulted(
        *_table_args(tables, direct), row, *first, *then, fault, results
    )
    if cqes is not None:
        cqes[:] = [int(results[2]), int(results[3])]
    return int(results[0]), int(results[1])


# One request's stage record: the C++ StageRecord's int64 words, in order. Every time is the host's
# CLOCK_MONOTONIC in ns (time.monotonic() reads the same clock); a stage never reached is 0.
# submit..pack_start are the first io_uring batch's, pack_end the last's (see StageRecord).
# The byte split, terminal status, per-row packing and per-extent CQE stamps are defined at StageRecord.
# STAGE_TRACE_ROWS / STAGE_TRACE_EXTENTS are its kTraceRows / kTraceExtents.
STAGE_DRIVES = 4
STAGE_TRACE_ROWS = 16
STAGE_TRACE_EXTENTS = 32
STAGE_FIELDS = (
    "seq", "kind", "row", "ok", "rows", "batches", "backlog", "prev_done",
    "observed", "reserved", "submit", "first_cqe", "last_cqe", "pack_start", "pack_end", "mapped", "done",
    "submit_to_first_cqe_ns", "first_to_last_cqe_ns", "pack_ns", "bytes", "extents",
    *(f"drive_dev_{d}" for d in range(STAGE_DRIVES)),
    *(f"drive_bytes_{d}" for d in range(STAGE_DRIVES)),
    *(f"drive_extents_{d}" for d in range(STAGE_DRIVES)),
    "status", "rows_asked", "useful_bytes", "submitted_bytes", "retried_bytes", "cancelled_bytes",
    "rows_untraced", "extents_untraced",
    *(f"row_pack_start_{k}" for k in range(STAGE_TRACE_ROWS)),
    *(f"row_pack_end_{k}" for k in range(STAGE_TRACE_ROWS)),
    *(f"extent_id_{k}" for k in range(STAGE_TRACE_EXTENTS)),
    *(f"extent_cqe_{k}" for k in range(STAGE_TRACE_EXTENTS)),
)
STAGE_KINDS = ("demand", "advisory", "touch")
# Index 0 is a record that never finished: the service never pushes one.
STAGE_STATUSES = ("none", "served", "no_read", "failed", "cancelled", "touch")
# The order of the time stamps within a request: the non-zero ones never decrease along it.
STAGE_ORDER = (
    "observed", "reserved", "submit", "first_cqe", "last_cqe", "pack_start", "pack_end", "mapped", "done",
)


def _stage_words() -> int:
    words = int(_host_module().exl3_ram_miss_trace_words())
    if words != len(STAGE_FIELDS):
        raise RuntimeError(f"C++ StageRecord has {words} words, STAGE_FIELDS {len(STAGE_FIELDS)}")
    return words


def stage_records(words: torch.Tensor) -> list[dict]:
    """Decode int64 ``[n, len(STAGE_FIELDS)]`` rows into dicts, with a ``drives`` list of the
    drives that served an extent: ``{"dev", "bytes", "extents"}`` (``dev`` -1: several folded).

    ``status`` names how the request ended and ``missing_stages`` the STAGE_ORDER stamps it never
    reached. ``row_pack`` has one ``{"row", "start", "end"}`` per row asked for (first
    ``STAGE_TRACE_ROWS``), 0/0 for a row that never packed; ``extent_cqe`` one ``{"row", "part",
    "cqe"}`` per extent issued (first ``STAGE_TRACE_EXTENTS``), ``cqe`` 0 for one that never completed.
    ``bytes`` is the completed total; the rest of the split is ``useful/submitted/retried/cancelled_bytes``.
    """
    out = []
    for row in words.tolist():
        record = dict(zip(STAGE_FIELDS, row))
        record["kind"] = STAGE_KINDS[record["kind"]]
        record["status"] = STAGE_STATUSES[record["status"]]
        record["missing_stages"] = [name for name in STAGE_ORDER if not record[name]]
        record["row_pack"] = [
            {
                "row": k,
                "start": record[f"row_pack_start_{k}"],
                "end": record[f"row_pack_end_{k}"],
            }
            for k in range(min(record["rows_asked"], STAGE_TRACE_ROWS))
        ]
        record["extent_cqe"] = [
            {
                "row": record[f"extent_id_{k}"] >> 16,
                "part": record[f"extent_id_{k}"] & 0xFFFF,
                "cqe": record[f"extent_cqe_{k}"],
            }
            for k in range(min(record["extents"], STAGE_TRACE_EXTENTS))
        ]
        for k in range(STAGE_TRACE_ROWS):
            del record[f"row_pack_start_{k}"], record[f"row_pack_end_{k}"]
        for k in range(STAGE_TRACE_EXTENTS):
            del record[f"extent_id_{k}"], record[f"extent_cqe_{k}"]
        record["drives"] = [
            {
                "dev": record.pop(f"drive_dev_{d}"),
                "bytes": record.pop(f"drive_bytes_{d}"),
                "extents": record.pop(f"drive_extents_{d}"),
            }
            for d in range(STAGE_DRIVES)
        ]
        record["drives"] = [drive for drive in record["drives"] if drive["extents"]]
        out.append(record)
    return out


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
# Order of the C++ counters. ``rows_read`` counts every row read, demand AND advisory
# (``advisory_rows`` is the advisory part); it is not the RAM-miss count behind ``f``. Demand
# rows come only from ``Exl3RamMissHost.layer_rows()``: per streamed layer, demand-only, and
# read under the host's lock. Do not derive them as ``rows_read - advisory_rows``: the two
# counters are separate atomics bumped after the rows are published, so a read can tear.
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


def sim_post(page, row: int, need, protect, *, advisory: bool = False, after: int = 0, armed: bool = True) -> int:
    """Post a record as the device post kernel does; returns its sequence.

    ``armed``: a device waits on the record. The post kernel arms a demand record when
    its need is non-empty or advisories are on; the thread only touches for an unarmed one.
    """
    return int(
        _host_module().exl3_ram_miss_sim_post(page, row, _ids(need), _ids(protect), int(advisory), after, int(armed))
    )


def sim_wait(page, seq: int, timeout_s: float) -> int:
    """Wait as the device wait kernel does: 1 served, 2 failed, 0 timed out, 3 fatal already raised."""
    return int(_host_module().exl3_ram_miss_sim_wait(page, seq, int(timeout_s * 1e9)))


def seqlock_stress(seconds: float) -> tuple[int, int]:
    """Test only: read one record while a C++ thread rewrites it; (accepted, torn accepted)."""
    out = torch.zeros(2, dtype=torch.int64)
    _host_module().exl3_ram_miss_seqlock_stress(int(seconds * 1e9), out)
    return int(out[0]), int(out[1])


_LIVE: weakref.WeakSet[Exl3RamMissHost] = weakref.WeakSet()


@atexit.register
def _stop_live() -> None:
    for host in list(_LIVE):
        try:
            host.stop()
        except Exception as error:  # noqa: BLE001 - one host's failure must not leave the rest open
            sys.stderr.write(f"exl3 RAM miss: stopping a host failed: {error!r}\n")


class Exl3RamMissHost:
    """The C++-owned pinned-slot bookkeeping of every streamed layer and its request service.

    ``tables``: ``Exl3RamMissTables``; ``page``: a ``new_page`` tensor; ``slot_map``:
    int32 ``[layers, experts]`` filled with -1 (pinned for a real device). Row ``r`` is
    streamed layer ``tables.layer_ids[r]``. Requests are served by ``pump()`` until
    ``start_thread()``, then by the C++ service thread until ``stop()``.
    """

    def __init__(self, tables, *, page: torch.Tensor, slot_map: torch.Tensor, direct: bool) -> None:
        if page.numel() != PAGE_BYTES or page.dtype != torch.uint8 or page.device.type != "cpu":
            raise ValueError("page must be a CPU uint8 tensor of PAGE_BYTES")
        if slot_map.dtype != torch.int32 or tuple(slot_map.shape) != tuple(tables.starts.shape):
            raise ValueError("slot_map must be int32 [layers, experts]")
        # C++ indexes both through raw addresses and starts with every slot FREE.
        if not page.is_contiguous() or not slot_map.is_contiguous() or slot_map.device.type != "cpu":
            raise ValueError("page and slot_map must be contiguous CPU tensors")
        if not bool((slot_map == -1).all()):
            raise ValueError("slot_map must start filled with -1 (the C++ tiers start empty)")
        self._module = _host_module()
        self.threaded = False
        # The C++ service writes through raw addresses of the page, the slot map and the
        # slabs (``tables.keepalive``): this object holds all three, and the finalizer
        # below closes the service before they can be released.
        self.tables = tables
        self.page = page
        self.slot_map = slot_map
        self.layers, self.experts = tables.starts.shape
        self.handle = int(
            self._module.exl3_ram_miss_open(
                page, slot_map, tables.extents, tables.starts, tables.file_sizes, tables.segments,
                tables.slabs, tables.row_bytes, tables.capacity, "\n".join(tables.paths),
                "\n".join(tables.source_paths), tables.slot_bytes, int(direct),
            )
        )
        if self.handle < 0:
            raise RuntimeError("exl3 RAM miss service failed to open (files, io_uring or bounce)")
        # exl3_ram_miss_close also stops and joins the service thread, if one runs.
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

    def start_thread(self, *, cpu_core: int = -1, fatal_wait_s: float = 30.0, spin_us: int = 5000) -> None:
        """Serve requests on a C++ thread (no more ``pump()``), with the fail-stop watchdog.

        ``cpu_core`` -1 inherits the caller's affinity; cores 64-71 are reserved (D19).
        """
        if 64 <= cpu_core <= 71:
            raise ValueError(f"cpu_core {cpu_core}: cores 64-71 are reserved (71 is production's doorbell core)")
        self._module.exl3_ram_miss_start_thread(self.handle, cpu_core, int(fatal_wait_s * 1e9), int(spin_us * 1e3))
        self.threaded = True

    def pause(self, timeout_s: float) -> None:
        """Hand the slots to the caller: returns once the thread is between two requests.

        The caller must have synchronized the device stream, so no demand is pending (D12).
        Not reentrant: one owner (the slot table's depth counter) pauses and resumes.
        """
        if not self.threaded:
            return
        if not self._module.exl3_ram_miss_pause(self.handle, int(timeout_s * 1e9)):
            raise RuntimeError(f"exl3 RAM miss thread did not pause within {timeout_s} s")

    def resume(self) -> None:
        if self.threaded:
            self._module.exl3_ram_miss_resume(self.handle)

    def pump(self) -> int:
        return int(self._module.exl3_ram_miss_pump(self.handle))

    def contains(self, row: int, expert: int) -> bool:
        """True once the expert holds a slot in ``row``: from the moment the thread claims the
        slot, before its read has finished. It does not mean the bytes are in RAM; the read is
        done when ``layer_rows()`` / ``layer_advisory_rows()`` count the row."""
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

    def enable_trace(self, capacity: int = 8192) -> None:
        """Record one stage record per served request, up to ``capacity`` undrained (more are dropped
        and counted). Before ``start_thread``; with it off the service takes no timestamps."""
        _stage_words()
        self._module.exl3_ram_miss_trace_enable(self.handle, int(capacity))

    def drain_trace(self, limit: int = 4096) -> list[dict]:
        """The stage records not yet drained, oldest first, as ``stage_records`` decodes them."""
        out = []
        while True:
            words = torch.empty((limit, len(STAGE_FIELDS)), dtype=torch.int64)
            count = int(self._module.exl3_ram_miss_trace_drain(self.handle, words))
            out.extend(stage_records(words[:count]))
            if count < limit:
                return out

    def trace_dropped(self) -> int:
        return int(self._module.exl3_ram_miss_trace_dropped(self.handle))

    def counters(self) -> dict[str, int]:
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.exl3_ram_miss_counters(self.handle, out)
        return dict(zip(COUNTERS, out.tolist()))

    def layer_rows(self) -> list[int]:
        """Rows read for demands only (not advisories), per streamed layer: the RAM misses behind ``f``.

        ``counters()["rows_read"]`` is demand plus advisory rows.
        """
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
            try:
                if self.threaded:
                    self._module.exl3_ram_miss_stop_thread(self.handle)
                    self.threaded = False
                # One line for the window's records (the corpus arms grep it).
                sys.stderr.write("exl3 RAM miss thread counters " + json.dumps(self.counters()) + "\n")
            finally:
                close()


STATE_WORDS = {
    "posted": 0,
    "pending": 1,
    "timeouts": 2,
    "failures": 3,
    "waits": 4,
    "polls": 5,
    "sticky": 6,
    "advised": 7,
    "unserved_misses": 8,
}


@cache_once
def _device_module() -> Module:
    names = ("exl3_ram_miss_post", "exl3_ram_miss_wait")
    return load_jit(
        "exl3_ram_miss",
        cuda_files=["moe/exl3_ram_miss.cuh"],
        cuda_wrappers=[(name, name) for name in names],
    )


class Exl3RamMissDevice:
    """The post and wait kernels of option C, capturable in a CUDA graph.

    ``page`` and ``slot_map`` are the host's pinned tensors (device-readable
    through UVA). ``state`` holds the device words ``STATE_WORDS``;
    ``last_routes`` int32 ``[layers, MAX_IDS]`` the previous token's routes per
    layer for advisories (``advise``). ``timeout_ms`` bounds each wait.
    """

    def __init__(self, page, slot_map, *, device, layers: int, timeout_ms: int, advise: bool) -> None:
        if page.numel() != PAGE_BYTES or page.dtype != torch.uint8:
            raise ValueError("page must be a uint8 tensor of PAGE_BYTES")
        if slot_map.dtype != torch.int32 or slot_map.dim() != 2 or slot_map.shape[0] != layers:
            raise ValueError("slot_map must be int32 [layers, experts]")
        if timeout_ms <= 0:
            raise ValueError("the RAM-miss wait timeout must be positive")
        if page.device.type != "cpu" or slot_map.device.type != "cpu":
            raise ValueError("page and slot_map must be host tensors")
        if not page.is_contiguous() or not slot_map.is_contiguous():
            raise ValueError("page and slot_map must be contiguous")
        # Checked before any CUDA call: the kernels read both through UVA, and an unpinned
        # address faults inside the captured graph.
        if torch.device(device).type == "cuda" and not (page.is_pinned() and slot_map.is_pinned()):
            raise ValueError("page and slot_map must be pinned for a CUDA device")
        self.page = page
        self.slot_map = slot_map
        self.layers = layers
        self.timeout_ns = int(timeout_ms * 1_000_000)
        self.advise = int(bool(advise))
        state = torch.zeros(len(STATE_WORDS), dtype=torch.int32)
        # Continue from the page's heads: the thread serves demand_done + 1 next, so a device
        # restarting at 1 over a used page would never be served.
        for word, head in (("posted", "demand_head"), ("advised", "advise_head")):
            state[STATE_WORDS[word]] = page[WORDS[head] : WORDS[head] + 4].view(torch.int32)[0]
        self.state = state.to(device)
        self.last_routes = torch.full((layers, MAX_IDS), -1, dtype=torch.int32, device=device)
        self._module = None

    def _kernels(self):
        if self._module is None:
            self._module = _device_module()
        return self._module

    def _check_row(self, name: str, row: int, *, allow_none: bool = False) -> None:
        low = -1 if allow_none else 0
        if not low <= row < self.layers:
            raise ValueError(f"{name} {row} is outside [{low}, {self.layers})")

    def _check_buffers(self, **buffers) -> None:
        """The kernels cast each buffer's data pointer to one fixed type and read ``[0, lanes)``."""
        for name, (tensor, dtype) in buffers.items():
            if tensor.dtype != dtype or tensor.device != self.state.device or not tensor.is_contiguous() or tensor.numel() < 1:
                raise ValueError(f"{name} must be a non-empty contiguous {dtype} tensor on {self.state.device}")

    def post(self, row: int, planned, count, routes, next_row: int) -> None:
        self._check_row("row", row)
        self._check_row("next_row", next_row, allow_none=True)
        self._check_buffers(planned=(planned, torch.int64), count=(count, torch.int32), routes=(routes, torch.int64))
        self._kernels().exl3_ram_miss_post(
            self.page, self.state, self.slot_map, planned, count, routes, row, self.advise, self.last_routes, next_row
        )

    def wait(self, row: int, planned, count, host_rows, keep, ram_miss) -> None:
        self._check_row("row", row)
        self._check_buffers(
            planned=(planned, torch.int64),
            count=(count, torch.int32),
            host_rows=(host_rows, torch.int64),
            keep=(keep, torch.float32),
            ram_miss=(ram_miss, torch.int64),
        )
        if planned.numel() < host_rows.numel():
            raise ValueError(f"planned has {planned.numel()} lanes but host_rows {host_rows.numel()}: the wait reads planned per lane")
        self._kernels().exl3_ram_miss_wait(
            self.page, self.state, self.slot_map, planned, count, row, host_rows, keep, ram_miss, self.timeout_ns
        )

    def stats(self) -> dict[str, int]:
        values = self.state.cpu().tolist()
        return {name: values[index] for name, index in STATE_WORDS.items()}
