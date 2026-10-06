"""Python wrappers for the RAM-miss expert-stream transport of EXL3 streamed experts.

Streamed experts live in files and in pinned host RAM; the device kernels fetch the
ones a layer needs through a request/response protocol with a C++ service thread.
This module wraps both halves:

  * the C++ host module: ``ExpertStreamHost`` (the service handle: slot bookkeeping,
    the service, copy and CPU-expert threads), the row reader's test entry points
    (``read_rows_*``) and the stage-trace decoder (``stage_records``);
  * the device kernels: ``ExpertStreamDevice``, one capturable chain per layer,
    ``post -> C1 (row copy) -> S (stream) -> CW (copy wait) -> stream wait -> CC
    (commit)``;
  * the byte layouts the two halves share: the request page, the lease block
    (``expert_lease_block``) and the counters.

The layouts mirror ``csrc/moe/expert_stream/lease_layout.h`` (under
``python/sglang/kernels/jit/``): ``expert_lease_block.wire_layout(lanes)`` gives the byte
offsets of a build with ``lanes`` lanes. See ``analysis/dsv41-drive/LEASE_PROTOCOL.md`` for
the protocol.
"""

from __future__ import annotations

import atexit
import functools
import json
import struct
import sys
import weakref
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.ops.moe import expert_lease_block
from sglang.srt.environ import envs

# Rows per io_uring batch and per bounce bank. The C++ reader has kBanks = 2 banks of
# kBounceRows = 8 row slots each; a bank is reused only after every row read into it
# has packed.
BOUNCE_ROWS = 8

if TYPE_CHECKING:
    from tvm_ffi.module import Module


class TransportBuild(msgspec.Struct, frozen=True):
    """One instantiation of the transport for a row format.

    ``host_sources`` maps each build variant (``VARIANTS``) to the host translation
    unit that binds the format's C++ layout and file reader. ``device_source`` is the
    translation unit ``RowCopyKernel`` is compiled in, and ``device_layout`` the
    layout type it is instantiated with.
    """

    host_sources: dict[str, str]
    device_source: str
    device_layout: str


# One entry per format the transport is built for. Adding a format adds an entry, its
# instantiation files (``host_sources``, ``device_source``) and the layout type they
# define.
LAYOUTS = {
    "exl3": TransportBuild(
        host_sources={
            "prod": "moe/exl3/exl3_ram_miss_host.cpp",
            "instr": "moe/exl3/exl3_ram_miss_host_instr.cpp",
        },
        device_source="moe/exl3/exl3_ram_miss.cuh",
        device_layout="sglang::exl3::Exl3RowLayout",
    )
}

# The two host builds: production (ProdBuild) and instrumented (InstrBuild), selected
# by ``csrc/moe/expert_stream/host/build_policy.h``. Each is its own module exporting
# the same entry points.
VARIANTS = ("prod", "instr")
# Tests set this (``sglang.test.expert_stream_variant``, through the conftest.py of
# test/registered/unit/{kernels,layers/moe}) to load the instrumented build, whose
# test-only entry points (faults, the stage trace) most of them use. None:
# ``host_variant()`` decides.
_DEFAULT_VARIANT: Optional[str] = None


def host_variant() -> str:
    """Return the host build a new service loads.

    The instrumented build when this process writes a stream trace
    (``SGLANG_DSV41_EXPERT_TRACE_PATH``) or injects a RAM-miss fault
    (``SGLANG_TEST_DSV41_RAM_MISS_FAULT``); production otherwise.
    """
    if _DEFAULT_VARIANT is not None:
        return _DEFAULT_VARIANT
    if (
        envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get()
        or envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get()
    ):
        return "instr"
    return "prod"


# The test-only entry points. Each exists only in the instrumented build, so on
# production both its Python wrapper (before building any tensor) and its C++ export
# raise RuntimeError("<name> is test-only: it exists in the instrumented host build").
# The stage trace (enable_trace, drain_trace, trace_dropped) also refuses on
# production, from C++, naming the same build. This tuple is documentation only.
TEST_ONLY_EXPORTS: tuple[str, ...] = (
    "read_rows_faulted",
    "read_rows_sqes",
    "inject",
    "inject_fault",
    "inject_group_stall",
    "copy_engine_fail",
    "copy_engine_ballast",
    "trace_clock_reads",
    "seqlock_stress",
    "pause_ns",
    "read_record_fields",
    "pump_group",
    "test_kernel_address",
    "test_kernel_calls",
    "test_kernel_hold",
    "test_keep_warm_calls",
    "test_keep_warm_core",
    "draft_test_post",
    "draft_test_tear",
    "draft_test_finish_close",
    "draft_test_poll_pause",
)


def _layer_tensors(spec) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``CpuExpertLayerSpec``'s slab table (int64 ``[n, 2]`` of {address, slot bytes}) and params (uint8 ``[m]``),
    as the host's set_cpu_layer and kernel_layer take them. ``reshape(n, 2)``: torch refuses ``reshape(-1, 2)`` of a
    zero-slab table (the fake kernel's layers have none)."""
    flat = [int(v) for pair in spec.slabs for v in pair]
    slabs = torch.tensor(flat, dtype=torch.int64).reshape(len(spec.slabs), 2)
    params = torch.tensor(list(spec.params), dtype=torch.uint8)
    return slabs, params


def _refuse_test_only(name: str, variant: Optional[str]) -> None:
    """Raise, as the C++ export would, if ``name`` (a ``TEST_ONLY_EXPORTS`` entry) is
    called on the production build."""
    if (host_variant() if variant is None else variant) == "prod":
        raise RuntimeError(
            f"{name} is test-only: it exists in the instrumented host build"
        )


# cache_once keys f(), f("exl3") and f(layout="exl3") apart, so each cached loader
# below is called only positionally, through a wrapper: a layout and variant then have
# exactly one module whatever the call form.
def _suffix(lanes: int, nodes: int) -> str:
    """A build's module-name suffix: one node keeps the Phase 1 names, so its modules and caches are unchanged."""
    return f"_l{lanes}" if nodes == 1 else f"_l{lanes}_n{nodes}"


def _host_module(
    layout: str = "exl3",
    variant: Optional[str] = None,
    lanes: int = 8,
    nodes: int = 1,
) -> Module:
    """Return the cached host module for ``layout``, ``variant`` (default build), ``lanes`` and ``nodes``."""
    variant = host_variant() if variant is None else variant
    wire = expert_lease_block.wire_layout(lanes, nodes)
    lanes, nodes = wire.lanes, wire.nodes
    if variant == "instr_tsan" and _ALLOW_TSAN:
        return _host_module_tsan(layout, lanes, nodes)
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown host build variant {variant!r}; expected one of {VARIANTS}"
        )
    if variant not in LAYOUTS[layout].host_sources:
        raise ValueError(f"layout {layout!r} has no {variant!r} host build variant")
    return _host_module_cached(layout, variant, lanes, nodes)


@cache_once
def _host_module_cached(layout: str, variant: str, lanes: int, nodes: int) -> Module:
    # Hidden visibility keeps HostExports' registries and members private to each
    # module's .so; only the TVM_FFI_DLL_EXPORT entry points are exported.
    return load_jit(
        f"expert_stream_host_{layout}_{variant}{_suffix(lanes, nodes)}",
        cpp_files=[LAYOUTS[layout].host_sources[variant]],
        extra_cflags=[
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
        extra_ldflags=["-luring", "-lpthread", "-ldl"],
        header_only=False,
    )


# The instrumented build under ThreadSanitizer, for the manual test
# test/manual/dsv41/test_expert_stream_hotpath_tsan.py only:
# ``ExpertStreamHost(..., variant="instr_tsan")`` loads it when that test sets
# ``_ALLOW_TSAN`` and is an unknown variant otherwise. The process must preload the
# compiler's TSan runtime.
_ALLOW_TSAN = False


@cache_once
def _host_module_tsan(layout: str, lanes: int, nodes: int) -> Module:
    return load_jit(
        f"expert_stream_host_{layout}_instr_tsan{_suffix(lanes, nodes)}",
        cpp_files=[LAYOUTS[layout].host_sources["instr"]],
        extra_cflags=[
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-fsanitize=thread",
            "-O1",
            "-g",
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
        extra_ldflags=["-luring", "-lpthread", "-ldl", "-fsanitize=thread"],
        header_only=False,
    )


def host_layout(layout: str = "exl3", lanes: int = 8) -> tuple[tuple[str, ...], int]:
    """Return the host module's row layout.

    That is its tensor names in copy-table order and a bit mask of the ones the SMs
    can read. ``lanes`` picks the build whose module is read, so a launch reads the
    one it already builds instead of compiling another.
    """
    return _host_layout_cached(layout, expert_lease_block.wire_layout(lanes).lanes)


@cache_once
def _host_layout_cached(layout: str, lanes: int) -> tuple[tuple[str, ...], int]:
    # The default variant is enough: every variant of a layout has the same layout.
    module = _host_module(layout, lanes=lanes)
    return tuple(str(module.expert_stream_layout_names()).split("\n")), int(
        module.expert_stream_layout_small_mask()
    )


def _ids(values: Iterable[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64)


def _checked_rows(
    tables, row: int, experts, slots
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``experts`` and ``slots`` as int64 tensors, after the bounds checks the
    C++ hot path skips."""
    expert_ids, slot_ids = _ids(experts), _ids(slots)
    if expert_ids.numel() != slot_ids.numel():
        raise ValueError(f"{expert_ids.numel()} experts but {slot_ids.numel()} slots")
    if not 0 <= row < len(tables.layer_ids):
        raise ValueError(f"streamed row {row} is outside [0, {len(tables.layer_ids)})")
    num_experts = int(tables.starts.shape[1])
    if expert_ids.numel() and not (
        0 <= int(expert_ids.min()) and int(expert_ids.max()) < num_experts
    ):
        raise ValueError(
            f"expert ids {expert_ids.tolist()} are outside [0, {num_experts})"
        )
    capacity = int(tables.capacity[row])
    if slot_ids.numel() and not (
        0 <= int(slot_ids.min()) and int(slot_ids.max()) < capacity
    ):
        raise ValueError(f"slots {slot_ids.tolist()} are outside [0, {capacity})")
    return expert_ids, slot_ids


def _table_buffer_regions(tables) -> torch.Tensor:
    """Return one registration region per slab tensor, with its row size.

    The reader cuts registration on row boundaries (``plan_chunks``), so the slab
    allocation itself stays one contiguous tensor. Arena owners are not registered
    whole: a 2.69 GB arena is past io_uring's 1 GiB per-buffer limit, and the
    registration is refused.
    """
    regions: dict[int, tuple[int, int, int]] = {}
    for slab in getattr(tables, "keepalive", ()):
        if not isinstance(slab, torch.Tensor):
            raise ValueError("I/O buffer owners must be tensors")
        if slab.device.type != "cpu" or not slab.is_contiguous():
            raise ValueError("I/O buffer slabs must be contiguous CPU tensors")
        nbytes = slab.numel() * slab.element_size()
        if nbytes and slab.dim() >= 1 and slab.shape[0] > 0:
            regions.setdefault(
                slab.data_ptr(), (slab.data_ptr(), nbytes, nbytes // slab.shape[0])
            )
    return torch.tensor(
        list(regions.values()), dtype=torch.int64, device="cpu"
    ).reshape(-1, 3)


def _table_args(tables, direct: bool = True) -> tuple:
    """Return the table arguments shared by every C++ reader entry.

    ``direct`` stays internal: the public helpers always pass 1, because the reader
    reads row images with O_DIRECT only. A test hands C++ ``direct=False`` to check
    that it refuses a buffered read.
    """
    return (
        tables.extents,
        tables.starts,
        tables.file_sizes,
        tables.segments,
        tables.slabs,
        tables.row_bytes,
        _table_buffer_regions(tables),
        "\n".join(tables.paths),
        "\n".join(tables.source_paths),
        tables.slot_bytes,
        # getattr: duck-typed test tables predate the field.
        int(getattr(tables, "row_images", False)),
        int(direct),
    )


def read_rows_once(
    tables,
    row: int,
    experts,
    slots,
    *,
    step: int = BOUNCE_ROWS,
    layout: str = "exl3",
    variant: Optional[str] = None,
) -> int:
    """Read ``experts`` of streamed row ``row`` into pinned ``slots`` in C++.

    Returns 1 if every row landed, 0 on failure. ``step`` rows go to io_uring per
    batch (at most ``BOUNCE_ROWS``).
    """
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    return int(
        _host_module(layout, variant).expert_stream_read_rows(
            *_table_args(tables), row, expert_ids, slot_ids, int(step)
        )
    )


def _fault_tensor(
    *,
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
    pack_delay_ns: int = 0,
    poison: bool = False,
    stale_cqe_call: int = 0,
    generation_start: int = 0,
    submit_short_call: int = 0,
    ordinal: int = -1,
    hold_ordinal: int = -1,
    abandon_after: int = 0,
    step: int = 0,
    hold_rest: bool = False,
    piece_stream: bool = False,
    sub: int = -1,
    publish_twice: int = 0,
    short_is_eof: bool = False,
    last_publish_delay_ns: int = 0,
    fixed_chunk_cap: int = 0,
    leg: int = -1,
    ring_reset_fail: bool = False,
    leg_cut_cap: int = 0,
    nop_flush_refused: bool = False,
) -> torch.Tensor:
    """Pack fault keywords into the C++ fault tensor (``kFaultWords`` int64 words).

    The word order must stay in step with ``fault_from()`` in
    ``exl3_ram_miss_host.cpp``; words 19, 20 and 26 are reserved. The keywords are
    documented at ``read_rows_with_fault``.
    """
    return torch.tensor(
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
            pack_delay_ns,
            int(poison),
            stale_cqe_call,
            generation_start,
            submit_short_call,
            ordinal,
            hold_ordinal,
            abandon_after,
            step,
            0,  # word 19: reserved (was pack_workers; the packed path is gone)
            0,  # word 20: reserved (was pack_split)
            int(hold_rest),
            int(piece_stream),
            sub,
            publish_twice,
            int(short_is_eof),
            0,  # word 26: reserved
            last_publish_delay_ns,
            fixed_chunk_cap,
            leg,
            (1 if (nop_flush_refused or ring_reset_fail) else 0)
            | (2 if ring_reset_fail else 0),
            leg_cut_cap,
        ],
        dtype=torch.int64,
    )


def read_rows_traced(
    tables,
    row: int,
    experts,
    slots,
    *,
    step: int = BOUNCE_ROWS,
    owner_core: int = -1,
    layout: str = "exl3",
    variant: Optional[str] = None,
    **faults,
) -> tuple[int, dict]:
    """Test only: ``read_rows_once`` that also returns the reader's stage record.

    Takes the fault arguments of ``read_rows_with_fault``; the record is decoded by
    ``stage_records``. Only the reader-side stages are filled: the request-side ones
    (observed, reserved, mapped, done) belong to the tier and stay 0.

    ``owner_core`` is a test-only owner-pinning scaffold (see
    ``analysis/dsv41-drive/PACK_WORKERS.md``). -1, the default, leaves the reader
    byte-for-byte what it is without this argument; >= 0 pins the calling thread to
    that core before ``open()``. Nothing wires it to the real service.

    The result is 1 (every row landed), 0 (failed) or -1 (abandoned: ``abandon_after``
    batches were admitted, the rows admitted were still read and packed, the rest
    never read).

    Available in both builds. On production (``variant="prod"``) the record's stages
    stay 0 (it has no trace; only ``ok`` and ``status`` are set), and a fault
    argument that injects a fault is refused with a RuntimeError naming the
    instrumented build. ``piece_stream``, ``abandon_after``, ``step``,
    ``fixed_chunk_cap`` and ``leg_cut_cap`` are not faults and work in both.
    """
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    fault = _fault_tensor(**faults)
    record = torch.zeros(_stage_words(layout, variant), dtype=torch.int64)
    result = int(
        _host_module(layout, variant).expert_stream_read_rows_traced(
            *_table_args(tables),
            row,
            expert_ids,
            slot_ids,
            int(step),
            fault,
            record,
            int(owner_core),
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
    cqes: Optional[list[int]] = None,
    stats: Optional[dict] = None,
    layout: str = "exl3",
    variant: Optional[str] = None,
    **faults,
) -> tuple[int, int]:
    """Test only: on one C++ reader, read with an injected fault, then read cleanly.

    Instrumented build only (``TEST_ONLY_EXPORTS``): production raises before building
    anything. Every fault keyword defaults to off.

    Submit and completion faults:

    - ``submit_error`` (an errno) replaces the result of the ``submit_call``-th
      submit, after submitting the prepared reads when ``submit_first``.
    - ``cqe_error`` replaces the ``cqe_call``-th completion's result.
    - ``part`` (a part index) targets the first completion of a part-``part`` extent
      (of row ``ordinal`` of the request, when given). ``part_error`` (an errno)
      replaces its result; ``part_short`` (a block multiple) caps the bytes it
      reports, so only that extent is resubmitted.
    - ``reverse_cqes`` processes each reaped batch of completions back to front. It
      must not change any result: the reader keys every extent by its own descriptor
      and generation, and the kernel orders nothing.
    - ``max_outstanding`` caps outstanding reads below the ring's depth, so a batch
      needs several refill rounds. Production constants keep a batch inside the
      ring, so credit never binds there.
    - ``nop_flush_refused`` (with a ``submit_error`` that leaves SQEs unconsumed)
      refuses the drain's NOP submission of those SQEs: a fixed read mode then raises
      "refused the NOP drain" and closes the reader, any other mode resets its ring.
      ``ring_reset_fail`` also makes that reset fail: the first read then raises
      "io_uring ring reset failed" (in a fixed read mode, the NOP-drain error) and no
      second read runs.

    Pipeline faults:

    - ``pack_delay_ns`` sleeps inside every row's packing.
    - ``poison`` fills bounce slots and scribbles retired descriptors.
    - ``stale_cqe_call`` redelivers the k-th retired extent's completion after its
      descriptor is recycled; the read must fail and publish nothing.
    - ``generation_start`` seeds the generation counter (near 2**32 it wraps).
    - ``hold_ordinal`` withholds the completions of that row of the request from the
      reader until every other row is done. This models a slow drive: buffered reads
      of cached data complete inside submit, so nothing else can hold an extent
      outstanding while other rows pack. With ``hold_rest``, every row from that
      ordinal on is withheld and released together, so they become ready in one reap.
    - ``submit_short_call`` makes that submit consume nothing and report success.
    - ``abandon_after`` stops admitting batches after that many.
    - ``step`` is the faulted read's rows per batch (default ``BOUNCE_ROWS``).

    Piece streaming and read cuts:

    - ``piece_stream`` reads each part as sub-reads and vets rows piece by piece;
      ``sub`` then narrows the ``part`` faults, and ``hold_ordinal``, to that sub-read
      of the part.
    - ``publish_twice`` publishes the k-th piece the reader publishes a second time;
      the readiness word must refuse it.
    - ``short_is_eof`` makes the ``part_short`` completion the end of its sub-read, as
      a file ending there would.
    - ``fixed_chunk_cap`` (bytes, 0: 1 GiB) caps the registered-buffer chunks of a
      fixed read mode, so small slabs register as many chunks. ``leg`` narrows the
      ``part``, ``cqe_error`` and ``hold_ordinal`` faults to that leg of a fanned-out
      fixed read (-1: any).
    - ``leg_cut_cap`` (bytes, 0: ``READ_CUTS`` and the device limits) cuts every read
      into legs of at most that many bytes on a 4 KiB boundary, whatever
      ``READ_CUTS`` says (see ``analysis/dsv41-drive/iopoll-cuts/results.md``);
      ``leg`` then narrows the faults to a cut leg as well.

    Returns both reads' results (1 ok, 0 failed, -1 abandoned). ``cqes``, if given,
    receives the completions reaped after each read. ``stats``, if given, receives
    the reader's ``stale_cqes``, ``generation_wraps``, ``unfinished_jobs`` and
    ``pack_workers`` (always 0: the packed path is gone), and after the first read
    ``fixed_cuts`` (reads fanned out to more than one leg), ``fanout_sqes`` (their
    SQEs), ``cut_reads`` (reads cut into more than one device-sized run) and
    ``gap_cuts`` (runs a boundary gap opened).
    """
    _refuse_test_only("read_rows_faulted", variant)
    first = _checked_rows(tables, row, experts, slots)
    then = _checked_rows(tables, row, then_experts, then_slots)
    fault = _fault_tensor(**faults)
    results = torch.zeros(12, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_rows_faulted(
        *_table_args(tables), row, *first, *then, fault, results
    )
    if cqes is not None:
        cqes[:] = [int(results[2]), int(results[3])]
    if stats is not None:
        stats.update(
            stale_cqes=int(results[4]),
            generation_wraps=int(results[5]),
            unfinished_jobs=int(results[6]),
            pack_workers=int(results[7]),
            fixed_cuts=int(results[8]),
            fanout_sqes=int(results[9]),
            cut_reads=int(results[10]),
            gap_cuts=int(results[11]),
        )
    return int(results[0]), int(results[1])


def read_rows_sqes(
    tables,
    row: int,
    experts,
    slots,
    *,
    step: int = BOUNCE_ROWS,
    max_sqes: int = 4096,
    layout: str = "exl3",
    variant: Optional[str] = None,
    **faults,
) -> tuple[int, list[tuple[int, int, int, int]], dict, dict]:
    """Test only: ``read_rows_traced``'s read, also returning every SQE it prepared.

    Returns ``(result, sqes, info, record)``. ``sqes`` lists the SQEs in order as
    ``(file, offset, length, bounce_offset)``. ``info`` holds ``sqes`` (the count),
    ``descriptors``, ``credit`` (the ring's), ``cqes``, ``fixed_cuts`` (reads fanned
    out to more than one leg), ``fanout_sqes`` (their SQEs, first attempts),
    ``cut_reads``, ``gap_cuts``, ``min_cut_bytes`` (the smallest cut in force, 0: cuts
    off) and ``leg_stride`` (the legs a read may have). Faults and ``piece_stream``
    are as in ``read_rows_with_fault``. Instrumented build only
    (``TEST_ONLY_EXPORTS``).
    """
    _refuse_test_only("read_rows_sqes", variant)
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    fault = _fault_tensor(**faults)
    record = torch.zeros(_stage_words(layout, variant), dtype=torch.int64)
    sqes = torch.zeros((max_sqes, 4), dtype=torch.int64)
    info = torch.zeros(11, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_rows_sqes(
        *_table_args(tables),
        row,
        expert_ids,
        slot_ids,
        int(step),
        fault,
        record,
        sqes,
        info,
    )
    (
        result,
        count,
        descriptors,
        credit,
        cqes,
        fixed_cuts,
        fanout_sqes,
        cut_reads,
        gap_cuts,
        min_cut_bytes,
        leg_stride,
    ) = info.tolist()
    if count > max_sqes:
        raise RuntimeError(f"{count} SQEs, more than max_sqes {max_sqes}")
    log = [tuple(entry) for entry in sqes[:count].tolist()]
    info = dict(
        sqes=count,
        descriptors=descriptors,
        credit=credit,
        cqes=cqes,
        fixed_cuts=fixed_cuts,
        fanout_sqes=fanout_sqes,
        cut_reads=cut_reads,
        gap_cuts=gap_cuts,
        min_cut_bytes=min_cut_bytes,
        leg_stride=leg_stride,
    )
    return result, log, info, stage_records(record.unsqueeze(0))[0]


def publish_piece(
    word: int,
    generation: int,
    bit: int,
    *,
    layout: str = "exl3",
    variant: Optional[str] = None,
) -> tuple[bool, int]:
    """Test only: the reader owner's publish primitive on one readiness word.

    ``word`` is ``generation << 8 | bits``. Returns whether the primitive set ``bit``
    and the word afterwards.
    """
    cell = torch.tensor(
        [word - (1 << 64) if word >= 1 << 63 else word], dtype=torch.int64
    )
    done = int(
        _host_module(layout, variant).expert_stream_publish_piece(
            cell, int(generation), int(bit)
        )
    )
    return bool(done), int(cell[0]) & 0xFFFFFFFFFFFFFFFF


def piece_word(generation: int, bits: int = 0) -> int:
    """Return a readiness word: the generation's low 56 bits over an 8-bit piece mask.

    This is the layout of one ``PieceMask`` word of the lease block.
    """
    return ((generation & ((1 << 56) - 1)) << 8) | bits


def read_rows_pieces(
    tables,
    row: int,
    experts,
    slots,
    *,
    generation: int,
    masks: Optional[torch.Tensor] = None,
    reference: Optional[torch.Tensor] = None,
    ref_slots=None,
    step: int = BOUNCE_ROWS,
    layout: str = "exl3",
    variant: Optional[str] = None,
    **faults,
) -> tuple[int, dict, torch.Tensor, dict]:
    """Test only: ``read_rows_traced``'s read, with piece streaming's publishing.

    Row ordinal o's pieces are published into ``masks[o]`` (int64 ``[rows, lanes]``;
    default one word per row, initialised to ``piece_word(generation)``). With
    ``reference`` (a slab pointer table like ``tables.slabs``, holding row o at
    ``ref_slots[o]``) a C++ thread checks, while the read runs, the destination bytes
    behind every bit it sees set on each row's first word.

    Returns ``(result, record, masks, info)``. ``info`` holds ``refused`` (the
    reader's refused publishes), ``checked`` and ``differed`` (pieces the checker
    compared, and those whose bytes were not the reference's) and ``early`` (bits it
    saw before the read returned).

    Available in both builds, as ``read_rows_traced``: on production the record's
    stages stay 0, ``refused`` is 0 (a metric), and a fault argument that injects a
    fault is refused.
    """
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    if masks is None:
        masks = torch.full(
            (len(expert_ids), 1), piece_word(generation), dtype=torch.int64
        )
    fault = _fault_tensor(**faults)
    record = torch.zeros(_stage_words(layout, variant), dtype=torch.int64)
    info = torch.zeros(5, dtype=torch.int64)
    ref = reference if reference is not None else torch.zeros(0, dtype=torch.int64)
    ref_ids = (
        _ids(ref_slots) if ref_slots is not None else torch.zeros(0, dtype=torch.int64)
    )
    _host_module(layout, variant).expert_stream_read_rows_pieces(
        *_table_args(tables),
        row,
        expert_ids,
        slot_ids,
        int(step),
        fault,
        record,
        masks,
        int(generation),
        ref,
        ref_ids,
        info,
    )
    result, refused, checked, differed, early = info.tolist()
    return (
        result,
        stage_records(record.unsqueeze(0))[0],
        masks,
        dict(refused=refused, checked=checked, differed=differed, early=early),
    )


def piece_geometry(
    tables,
    row: int,
    expert: int,
    *,
    layout: str = "exl3",
    variant: Optional[str] = None,
) -> Optional[tuple[list[dict], list[dict]]]:
    """Test only: how the C++ reader cuts one expert under piece streaming.

    Returns ``(sub_reads, pieces)`` for expert ``expert`` of streamed row ``row``, or
    None when the reader refuses the row. Sub-reads, in file order, are
    ``{file, offset, length, dest, part, k}``. Pieces (``STAGE_PIECES``) are
    ``{deps, runs}``: ``deps`` is the bitmask of sub-reads the piece depends on and
    ``runs`` holds one ``(dst_lo, dst_hi)`` per segment, in segment destination
    coordinates.
    """
    segments = int(tables.segments.shape[0])
    subs = torch.zeros((STAGE_PIECES, 6), dtype=torch.int64)
    pieces = torch.zeros((STAGE_PIECES, 1 + 2 * segments), dtype=torch.int64)
    count = int(
        _host_module(layout, variant).expert_stream_piece_geometry(
            *_table_args(tables)[:-1], row, expert, subs, pieces
        )
    )
    if count < 0:
        return None
    keys = ("file", "offset", "length", "dest", "part", "k")
    sub_reads = [dict(zip(keys, line)) for line in subs[:count].tolist()]
    out = []
    for line in pieces.tolist():
        out.append(
            {
                "deps": line[0],
                "runs": [(line[1 + 2 * i], line[2 + 2 * i]) for i in range(segments)],
            }
        )
    return sub_reads, out


# One request's stage record: the C++ ``StageRecord``'s int64 words, in order.
#
# Every time is the host's CLOCK_MONOTONIC in ns (``time.monotonic()`` reads the same
# clock); a stage never reached is 0. ``submit``/``first_cqe``/``last_cqe`` span the
# whole read, while ``pack_start``..``pack_end`` run from the first row's packing to
# the last row's and overlap the reads. The byte split, terminal status, per-row
# packing and per-extent CQE stamps are defined at ``StageRecord``;
# ``stage_trace_rows(lanes)`` is the build's ``kTraceRows`` and twice that its ``kTraceExtents``.
#
# Row images, the only reader since the direct mode became the sole mode, copy
# nothing: the drive writes the slab rows. The pack stamps then mean publish time.
# With piece streaming ``row_pack_start``/``row_pack_end`` are the clocks of the
# row's first and last piece publish; without it both are the clock at which the
# row's reads were vetted and the row finished. ``pack_start``, ``pack_end`` and
# ``pack_ns`` are built from them as for packing. ``pack_workers`` and ``pack_split``
# are 0, and ``useful_bytes`` still counts the segment bytes that landed in the slabs.
STAGE_DRIVES = 4


def stage_trace_rows(lanes: int = 8) -> int:
    """kTraceRows: the need and protect ids a request can read, two per lane."""
    return 2 * expert_lease_block.wire_layout(lanes).lanes


def calibration_shape(lanes: int = 8) -> tuple[int, int]:
    """The calibration grid (kCalibRows, kCalibCols)."""
    n = expert_lease_block.wire_layout(lanes).lanes
    return (n + 2, n + 1)


# The C++ ``kPieces``: pieces per row, and the most sub-reads a row issues under piece
# streaming.
STAGE_PIECES = 8
_STAGE_HEAD = (
    "seq",
    "kind",
    "row",
    "ok",
    "rows",
    "batches",
    "backlog",
    "prev_done",
    "observed",
    "reserved",
    "submit",
    "first_cqe",
    "last_cqe",
    "pack_start",
    "pack_end",
    "mapped",
    "done",
    "submit_to_first_cqe_ns",
    "first_to_last_cqe_ns",
    "pack_ns",
    "bytes",
    "extents",
    *(f"drive_dev_{d}" for d in range(STAGE_DRIVES)),
    *(f"drive_bytes_{d}" for d in range(STAGE_DRIVES)),
    *(f"drive_extents_{d}" for d in range(STAGE_DRIVES)),
    "status",
    "rows_asked",
    "useful_bytes",
    "submitted_bytes",
    "retried_bytes",
    "cancelled_bytes",
    "rows_untraced",
    "extents_untraced",
    "rows_reading_max",
    "pending_max",
    "bank_stalls",
)


@functools.cache
def stage_fields(lanes: int = 8) -> tuple[str, ...]:
    """The C++ ``StageRecord``'s int64 words of a ``lanes``-lane build, in order."""
    rows = stage_trace_rows(lanes)
    extents = 2 * rows
    return (
        *_STAGE_HEAD,
        *(f"row_pack_start_{k}" for k in range(rows)),
        *(f"row_pack_end_{k}" for k in range(rows)),
        *(f"extent_id_{k}" for k in range(extents)),
        *(f"extent_cqe_{k}" for k in range(extents)),
        "dropped_before",
        *(f"row_admit_{k}" for k in range(rows)),
        *(f"extent_submit_{k}" for k in range(extents)),
        *(f"extent_attempts_{k}" for k in range(extents)),
        "lanes",
        "pack_workers",
        "pack_split",
        "piece_stream",
        "pieces_vetted",
        *(f"sub_land_seq_{k}_{j}" for k in range(rows) for j in range(STAGE_PIECES)),
        *(f"piece_cqe_{k}_{j}" for k in range(rows) for j in range(STAGE_PIECES)),
        *(f"piece_seq_{k}_{j}" for k in range(rows) for j in range(STAGE_PIECES)),
        *(f"piece_publish_{k}_{j}" for k in range(rows) for j in range(STAGE_PIECES)),
        "pieces_published",
        "pieces_out_of_order",
        "piece_publish_refused",
    )


STAGE_KINDS = ("demand", "touch")
# Index 0 is a record that never finished: the service never pushes one.
STAGE_STATUSES = ("none", "served", "no_read", "failed", "cancelled", "touch")
# The order of the time stamps within a request. The non-zero ones never decrease
# along it except ``last_cqe`` against ``pack_start``: a row packs as soon as its own
# extents landed, so packing starts before the last completion when reads and packing
# overlap. What holds instead is ``first_cqe <= pack_start`` and
# ``last_cqe <= pack_end`` (see ``StageRecord``).
STAGE_ORDER = (
    "observed",
    "reserved",
    "submit",
    "first_cqe",
    "last_cqe",
    "pack_start",
    "pack_end",
    "mapped",
    "done",
)


def _stage_words(
    layout: str = "exl3", variant: Optional[str] = None, lanes: int = 8
) -> int:
    """Return the C++ ``StageRecord`` word count of a ``lanes``-lane build, checked against ``stage_fields``."""
    words = int(_host_module(layout, variant, lanes).expert_stream_trace_words())
    if words != len(stage_fields(lanes)):
        raise RuntimeError(
            f"C++ StageRecord has {words} words, stage_fields({lanes}) {len(stage_fields(lanes))}"
        )
    return words


def stage_records(words: torch.Tensor, lanes: int = 8) -> list[dict]:
    """Decode int64 ``[n, len(stage_fields(lanes))]`` stage-record rows of a ``lanes``-lane build into dicts.

    Beyond the ``stage_fields`` scalars, each dict has:

    - ``status``: how the request ended; ``missing_stages``: the ``STAGE_ORDER``
      stamps it never reached.
    - ``drives``: the drives that served an extent, as ``{"dev", "bytes", "extents"}``
      (``dev`` -1: several folded).
    - ``row_pack``: one ``{"row", "admit", "start", "end"}`` per row asked for (first
      ``stage_trace_rows(lanes)``); 0/0 for a row that never packed, ``admit`` 0 for one
      never admitted.
    - ``extent_cqe``: one ``{"row", "part", "sub", "submit", "attempts", "cqe"}`` per
      extent issued (first twice ``stage_trace_rows(lanes)``). ``cqe`` is 0 for an extent that
      never completed, and otherwise the time the wait that reaped it returned:
      io_uring gives no per-completion time. ``attempts`` counts resubmissions and
      ``sub`` is the extent's sub-read within its part (0 unless ``piece_stream``).
    - ``pieces``: empty unless ``piece_stream``; otherwise one
      ``{"row", "sub_seq", "seq", "cqe", "publish"}`` per row asked for (first
      ``stage_trace_rows(lanes)``), each field a list over ``STAGE_PIECES``. ``sub_seq`` is
      when sub-read j (row file order) landed, ``seq`` when piece j was vetted, and
      ``publish`` when it was published, as sequence numbers shared by all three
      (1-based, 0 never). ``cqe`` is the vetting's clock.

    Scalars worth knowing: ``bytes`` is the completed total and the rest of the split
    is ``useful_bytes``, ``submitted_bytes``, ``retried_bytes`` and
    ``cancelled_bytes``; ``dropped_before`` counts the records the ring dropped just
    before this one; ``lanes`` is the planned lane count the device posted with the
    request; ``pieces_published``, ``pieces_out_of_order`` (published after a
    higher-numbered piece of their row) and ``piece_publish_refused`` describe the
    read's publishing. ``pack_workers`` and ``pack_split`` are the packing mode the
    reader ran the request in: 0 workers is the inline reader, and a worker-mode
    record's pack stamps are not comparable with an inline one's (see
    ``StageRecord``).
    """
    fields = stage_fields(lanes)
    trace_rows = stage_trace_rows(lanes)
    trace_extents = 2 * trace_rows
    out = []
    for row in words.tolist():
        record = dict(zip(fields, row))
        record["kind"] = STAGE_KINDS[record["kind"]]
        record["status"] = STAGE_STATUSES[record["status"]]
        record["missing_stages"] = [name for name in STAGE_ORDER if not record[name]]
        record["row_pack"] = [
            {
                "row": k,
                "admit": record[f"row_admit_{k}"],
                "start": record[f"row_pack_start_{k}"],
                "end": record[f"row_pack_end_{k}"],
            }
            for k in range(min(record["rows_asked"], trace_rows))
        ]
        record["extent_cqe"] = [
            {
                "row": record[f"extent_id_{k}"] >> 16,
                "part": record[f"extent_id_{k}"] & 0xFF,
                "sub": (record[f"extent_id_{k}"] >> 8) & 0xFF,
                "submit": record[f"extent_submit_{k}"],
                "attempts": record[f"extent_attempts_{k}"],
                "cqe": record[f"extent_cqe_{k}"],
            }
            for k in range(min(record["extents"], trace_extents))
        ]
        for k in range(trace_rows):
            del (
                record[f"row_pack_start_{k}"],
                record[f"row_pack_end_{k}"],
                record[f"row_admit_{k}"],
            )
        record["pieces"] = (
            [
                {
                    "row": k,
                    "sub_seq": [
                        record[f"sub_land_seq_{k}_{j}"] for j in range(STAGE_PIECES)
                    ],
                    "seq": [record[f"piece_seq_{k}_{j}"] for j in range(STAGE_PIECES)],
                    "cqe": [record[f"piece_cqe_{k}_{j}"] for j in range(STAGE_PIECES)],
                    "publish": [
                        record[f"piece_publish_{k}_{j}"] for j in range(STAGE_PIECES)
                    ],
                }
                for k in range(min(record["rows_asked"], trace_rows))
            ]
            if record["piece_stream"]
            else []
        )
        for k in range(trace_extents):
            del record[f"extent_id_{k}"], record[f"extent_cqe_{k}"]
            del record[f"extent_submit_{k}"], record[f"extent_attempts_{k}"]
        for k in range(trace_rows):
            for j in range(STAGE_PIECES):
                del (
                    record[f"sub_land_seq_{k}_{j}"],
                    record[f"piece_seq_{k}_{j}"],
                    record[f"piece_cqe_{k}_{j}"],
                )
                del record[f"piece_publish_{k}_{j}"]
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


# The request page (``lease_layout.h``): ``demand_head``, then ``wire.demand_records`` records
# (``wire.record_fields``). A record's lanes are one i16 array per id and an f32 weight
# array; "counts" holds the lane count (bits 0-3) and the protect-id count (bits 4-7) on
# an 8-lane wire, and "kinds" a ``ram_slot_map.LaneKind`` nibble per lane.
RECORD_FLAG_CAPTURED = 1
# Experts, slots and destinations are i16 in the record and the map delta.
RECORD_ID_MAX = 32767
HOT_HEADER_BYTES = 8
HOT_ALIGNMENT = 64
HOT_RECORDS = expert_lease_block.wire_layout(8).demand_records


def hot_record_bytes(experts: int) -> int:
    """Return the size of one hot-bitmap record: header plus one bit per expert."""
    if experts <= 0 or experts > 65535:
        raise ValueError(
            f"EXL3 hot bitmap expert count {experts} is outside [1, 65535]"
        )
    return (
        (HOT_HEADER_BYTES + (experts + 7) // 8 + HOT_ALIGNMENT - 1) // HOT_ALIGNMENT
    ) * HOT_ALIGNMENT


def new_hot_page(experts: int, *, pin: bool = True) -> torch.Tensor:
    """Allocate a zeroed hot page: ``HOT_RECORDS`` hot-bitmap records."""
    return torch.zeros(
        HOT_RECORDS * hot_record_bytes(experts), dtype=torch.uint8, pin_memory=pin
    )


WORDS = {"demand_head": 0}
# Order of the C++ counters (``host/tier_protocol.h``). Demand rows per layer come only
# from ``ExpertStreamHost.layer_rows()``: one word per layer written by the tier's
# owner.
COUNTERS = (
    "served",
    "touch_only",
    "rows_read",
    "read_errors",
    "evictions",
    "overruns",
    "no_victim",
    "version",
    "running",
    "spin_cpu",
    "ram_insert_skipped",
    "piece_publish_refused",
    # Copy engine.
    "copy_jobs",
    "copy_lanes",
    "copy_bytes",
    "copy_issue_ns",
    "copy_latency_ns",
    "copy_latency_max_ns",
    # CPU experts.
    "cpu_jobs",
    "cpu_lanes",
)

# The counters a production host keeps: the shutdown line's served, rows and errors,
# the admission policy's outcomes, and the functional version, in ``COUNTERS`` order.
# ``is_core_counter`` in ``host/tier_protocol.h`` defines the set and
# ``expert_stream_core_counter_mask`` checks it. Every other counter is a metric the
# production build does not compile: its ``counters()`` has no such key.
CORE_COUNTERS = (
    "served",
    "touch_only",
    "rows_read",
    "read_errors",
    "evictions",
    "overruns",
    "no_victim",
    "version",
    "running",
    "spin_cpu",
    "ram_insert_skipped",
)
assert CORE_COUNTERS == tuple(sorted(CORE_COUNTERS, key=COUNTERS.index))


def seqlock_stress(
    seconds: float,
    *,
    layout: str = "exl3",
    variant: Optional[str] = None,
    lanes: int = 8,
) -> tuple[int, int]:
    """Test only: read one record while a C++ thread rewrites it for ``seconds``.

    Returns ``(accepted, torn_accepted)``. Instrumented build only.
    """
    _refuse_test_only("seqlock_stress", variant)
    out = torch.zeros(2, dtype=torch.int64)
    _host_module(layout, variant, lanes).expert_stream_seqlock_stress(
        int(seconds * 1e9), out
    )
    return int(out[0]), int(out[1])


def read_record_words(lanes: int = 8) -> int:
    """Words of the C++ read_record result: 6 scalars, the protect ids, the lane count and 5 words per lane."""
    n = expert_lease_block.wire_layout(lanes).lanes
    return 6 + n + 1 + 5 * n


def read_record_fields(
    record: torch.Tensor,
    expected: int,
    *,
    layout: str = "exl3",
    variant: Optional[str] = None,
    lanes: int = 8,
) -> dict:
    """Test only: the service's ``read_record`` over one record of ``lanes`` lanes.

    ``expected`` is the sequence number the record must carry. Instrumented build only.
    """
    _refuse_test_only("read_record_fields", variant)
    out = torch.zeros(read_record_words(lanes), dtype=torch.int64)
    _host_module(layout, variant, lanes).expert_stream_read_record_fields(
        record, int(expected), out
    )
    w = out.tolist()
    n = expert_lease_block.wire_layout(lanes).lanes
    protect, lane_count = w[5], w[6 + n]
    base = 7 + n
    return {
        "status": ("ok", "torn", "malformed")[w[0]],
        "row": w[1],
        "captured": bool(w[2]),
        "chain": w[3] & 0xFFFFFFFFFFFFFFFF,
        "gen": w[4] & 0xFFFFFFFFFFFFFFFF,
        "protect": w[6 : 6 + protect],
        "lanes": [
            {
                "expert": w[base + 5 * j],
                "slot": w[base + 5 * j + 1],
                "dst": w[base + 5 * j + 2],
                "kind": w[base + 5 * j + 3],
                "weight": struct.unpack("<f", struct.pack("<i", w[base + 5 * j + 4]))[
                    0
                ],
            }
            for j in range(lane_count)
        ],
    }


# kernel_layer's layers' spec.keep, by layer id: the slabs each layer reads stay alive until kernel_drop.
_kernel_layer_keep: dict[int, tuple] = {}


def kernel_layer(kernel: int, spec, *, layout: str = "exl3", variant: Optional[str] = None) -> int:
    """Kernel ``kernel``'s make_layer over ``spec`` (a ``CpuExpertLayerSpec``); returns the layer's id.

    ``spec.keep`` is kept alive until :func:`kernel_drop`. Both builds: the DSpark draft's CPU experts
    (``cpu_experts/draft.py``) run through it, and tests.
    """
    slabs, params = _layer_tensors(spec)
    layer = int(
        _host_module(layout, variant).expert_stream_kernel_layer(
            int(kernel),
            slabs,
            int(spec.capacity),
            int(spec.hidden),
            int(spec.intermediate),
            int(spec.activation),
            float(spec.act_limit),
            params,
        )
    )
    _kernel_layer_keep[layer] = spec.keep
    return layer


def kernel_forward(
    layer: int,
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    *,
    threads: int,
    cores: Sequence[int] = (),
    accumulate: bool = False,
    layout: str = "exl3",
    variant: Optional[str] = None,
) -> tuple[int, str]:
    """One forward of :func:`kernel_layer`'s ``layer`` over ``rows`` token rows (x ``[rows, H]``, slots and weights
    ``[rows, k]``).

    Returns (0, "") or the kernel's refusal: (2, why) for a bad call, (1, why) for a failure; ``out`` is untouched
    then. Pins the calling thread to ``cores[0]``. Both builds.
    """
    module = _host_module(layout, variant)
    status = int(
        module.expert_stream_kernel_forward(
            int(layer),
            x.to(torch.float16).contiguous(),
            slots.to(torch.int32).contiguous(),
            weights.to(torch.float32).contiguous(),
            out,
            int(threads),
            torch.tensor(list(cores), dtype=torch.int64),
            int(bool(accumulate)),
        )
    )
    # The message is this thread's (a thread-local in the module), read on the same thread.
    return status, (str(module.expert_stream_kernel_error()) if status else "")


def kernel_drop(layer: int, *, layout: str = "exl3", variant: Optional[str] = None) -> None:
    """Release :func:`kernel_layer`'s ``layer``. Both builds."""
    _host_module(layout, variant).expert_stream_kernel_drop(int(layer))
    _kernel_layer_keep.pop(int(layer), None)


def pause_ns(*, layout: str = "exl3", variant: Optional[str] = None) -> float:
    """Test only: ns per ``_mm_pause`` on this core.

    That is the RAM-miss service's idle-poll quantum. Instrumented build only.
    """
    _refuse_test_only("pause_ns", variant)
    return float(_host_module(layout, variant).expert_stream_pause_ns())


def new_page(*, pin: bool, wire: expert_lease_block.WireLayout) -> torch.Tensor:
    """A zeroed request page of ``wire``; pinned (device-readable through UVA) for a real device."""
    return torch.zeros(wire.page_bytes, dtype=torch.uint8, pin_memory=pin)


def page_word(page: torch.Tensor, name: str) -> int:
    """Return the u32 request-page word ``name`` (a ``WORDS`` key)."""
    offset = WORDS[name]
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


_LIVE: weakref.WeakSet[ExpertStreamHost] = weakref.WeakSet()


@atexit.register
def _stop_live() -> None:
    """Stop every live host at exit, logging its counters first."""
    for host in list(_LIVE):
        try:
            host.stop()
        except Exception as error:  # noqa: BLE001 - keep stopping the other hosts
            sys.stderr.write(f"exl3 RAM miss: stopping a host failed: {error!r}\n")


def _spin_ns(spin_us: int) -> int:
    """An idle spin budget in ns for the C++ side; a negative budget (never sleep) stays -1."""
    return -1 if spin_us < 0 else int(spin_us * 1e3)


class ExpertStreamHost:
    """The C++-owned pinned-slot bookkeeping of every streamed layer and its request
    service.

    ``tables`` is an ``Exl3RamMissTables``; ``page`` a ``new_page`` tensor of the
    ``lanes``-lane wire (kept as ``self.wire``, which every lane-sized tensor follows); ``slot_map``
    an int32 ``[layers, experts]`` tensor filled with -1 (pinned for a real device).
    Row ``r`` is streamed layer ``tables.layer_ids[r]``. ``variant`` is the host build
    to load (``VARIANTS``), by default ``host_variant()``'s choice, and is kept as
    ``self.variant``.

    Threads and ownership. Requests are served by ``pump()`` until ``start_thread()``,
    then by the C++ service thread until ``stop()``. The tier has one owner at a time:
    the service thread while it runs, the caller between ``pause()`` and ``resume()``,
    and the caller of ``pump()`` when there is no thread. While the thread runs
    unpaused, every call that reads or writes tier state (``contains``, ``touch``,
    ``assign``, ``release``, ``fill_begin``, ``set_hot``, ``slot_info``,
    ``slot_to_expert``, ``lru_order``, ``victim_census``) raises ``RuntimeError``
    ("... needs the service thread paused"). ``mapping``, ``counters``,
    ``busy_episode`` and ``layer_rows`` read published words without waiting. Paused,
    or with no thread, every call runs at once.

    Lifetime. The C++ service reads and writes through the raw addresses of the page,
    the slot map, the lease block and the slabs (``tables.keepalive``); this object
    holds all of them, and a finalizer closes the service (stopping and joining its
    thread) before they can be released. Live hosts are also stopped at interpreter
    exit by ``_stop_live``, which logs their counters first.
    """

    def __init__(
        self,
        tables,
        *,
        page: torch.Tensor,
        slot_map: torch.Tensor,
        lease_block: Optional[torch.Tensor] = None,
        hot_page: Optional[torch.Tensor] = None,
        layout: str = "exl3",
        variant: Optional[str] = None,
        lanes: int = 8,
        node_ranges: Optional[Sequence[Sequence[tuple[int, int]]]] = None,
        sq_thread_cpus: Optional[Sequence[int]] = None,
    ) -> None:
        # NUMA groups: node_ranges[g][row] = (lo, hi), group g's slots of the row; one group owning every slot by
        # default. sq_thread_cpus: each group's SQPOLL core (-1 unpinned); None keeps the uring env's.
        capacity = [int(c) for c in tables.capacity.tolist()]
        if node_ranges is None:
            node_ranges = [[(0, c) for c in capacity]]
        self.nodes = len(node_ranges)
        self.node_ranges = [[(int(lo), int(hi)) for lo, hi in rows] for rows in node_ranges]
        for g, rows in enumerate(self.node_ranges):
            if len(rows) != len(capacity):
                raise ValueError(f"node_ranges[{g}] has {len(rows)} rows, the tables {len(capacity)}")
            for row, (lo, hi) in enumerate(rows):
                if not 0 <= lo <= hi <= capacity[row]:
                    raise ValueError(f"group {g}'s slots [{lo}, {hi}) of row {row} are outside its {capacity[row]} slots")
                for other in range(g):
                    olo, ohi = self.node_ranges[other][row]
                    if lo < ohi and olo < hi:
                        raise ValueError(f"groups {other} and {g} overlap in row {row}")
        self.wire = expert_lease_block.wire_layout(lanes, self.nodes)
        if (
            page.numel() != self.wire.page_bytes
            or page.dtype != torch.uint8
            or page.device.type != "cpu"
        ):
            raise ValueError(
                f"page must be a CPU uint8 tensor of {self.wire.page_bytes} bytes (new_page of this wire)"
            )
        if slot_map.dtype != torch.int32 or tuple(slot_map.shape) != tuple(
            tables.starts.shape
        ):
            raise ValueError("slot_map must be int32 [layers, experts]")
        # C++ indexes both through raw addresses and starts with every slot FREE.
        if (
            not page.is_contiguous()
            or not slot_map.is_contiguous()
            or slot_map.device.type != "cpu"
        ):
            raise ValueError("page and slot_map must be contiguous CPU tensors")
        if not bool((slot_map == -1).all()):
            raise ValueError(
                "slot_map must start filled with -1 (the C++ tiers start empty)"
            )
        self._layout = layout
        # The host build is chosen once here: every later call goes to this module.
        self.variant = host_variant() if variant is None else variant
        self._module = _host_module(self._layout, self.variant, self.wire.lanes, self.nodes)
        self.threaded = False
        # set_cpu_layer's specs' tensors, kept alive while the host's layers read them.
        self._cpu_layer_keep: dict[int, tuple] = {}
        # Allocated here when the caller passes none.
        if lease_block is None:
            lease_block = expert_lease_block.new_lease_block(
                int(tables.starts.shape[0]), pin=page.is_pinned(), wire=self.wire
            )
        else:
            expert_lease_block.check_lease_block(
                lease_block,
                int(tables.starts.shape[0]),
                need_pinned=page.is_pinned(),
                wire=self.wire,
            )
        self.lease_block = lease_block
        self.hot_page = hot_page
        if hot_page is not None:
            stride = hot_record_bytes(tables.starts.shape[1])
            if (
                hot_page.dtype != torch.uint8
                or hot_page.device.type != "cpu"
                or not hot_page.is_contiguous()
                or hot_page.numel() != HOT_RECORDS * stride
                or (page.is_pinned() and not hot_page.is_pinned())
            ):
                raise ValueError(
                    f"hot_page must be a contiguous pinned CPU uint8 tensor of {HOT_RECORDS * stride} bytes"
                )
        self.tables = tables
        self.page = page
        self.slot_map = slot_map
        self.layers, self.experts = tables.starts.shape
        self.handle = int(
            self._module.expert_stream_open(
                page,
                slot_map,
                tables.extents,
                tables.starts,
                tables.file_sizes,
                tables.segments,
                tables.slabs,
                tables.row_bytes,
                _table_buffer_regions(tables),
                tables.capacity,
                "\n".join(tables.paths),
                "\n".join(tables.source_paths),
                tables.slot_bytes,
                int(tables.row_images),
                1,
                self.lease_block,
                self.hot_page
                if self.hot_page is not None
                else torch.empty(0, dtype=torch.uint8),
                torch.tensor(self.node_ranges, dtype=torch.int64).reshape(self.nodes, len(capacity), 2),
                torch.tensor(
                    [-2] * self.nodes if sq_thread_cpus is None else [int(c) for c in sq_thread_cpus],
                    dtype=torch.int64,
                ),
            )
        )
        if self.handle < 0:
            raise RuntimeError(
                "exl3 RAM miss service failed to open (files, io_uring or bounce)"
            )
        self.layout_names, self.small_mask = host_layout(self._layout)
        self._close = weakref.finalize(
            self, self._module.expert_stream_close, self.handle
        )
        # _stop_live closes live hosts at exit, logging counters first.
        self._close.atexit = False
        _LIVE.add(self)

    def _check(
        self, row: int, expert: Optional[int] = None, slot: Optional[int] = None
    ) -> None:
        """Raise ValueError for a row, expert or slot outside the bounds the C++
        bookkeeping does not check."""
        if not 0 <= row < self.layers:
            raise ValueError(f"streamed row {row} is outside [0, {self.layers})")
        if expert is not None and not 0 <= expert < self.experts:
            raise ValueError(f"expert {expert} is outside [0, {self.experts})")
        capacity = int(self.tables.capacity[row])
        if slot is not None and not 0 <= slot < capacity:
            raise ValueError(f"slot {slot} is outside [0, {capacity})")

    def start_thread(
        self,
        *,
        cpu_core: int | Sequence[int] = -1,
        fatal_wait_s: float = 30.0,
        spin_us: int = -1,
        busy_poll: bool = False,
    ) -> None:
        """Serve requests on one C++ thread per NUMA group (no more ``pump()``), with the watchdog.

        ``cpu_core`` is one core per group (a sequence), or one int for a single group; -1 inherits the caller's
        affinity. Cores 64-71 are reserved for NVMe completion interrupts. An idle service thread spins with PAUSE,
        or with ``busy_poll`` with no PAUSE (which the C++ side refuses unless each physical core is its service's
        alone), for ``spin_us``, then sleeps 50 us between polls; parked, it sleeps 20 us between checks. -1 never
        sleeps (``SGLANG_DSV41_RAM_MISS_IDLE_SPIN_US`` in a server).
        """
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        cores = [int(cpu_core)] * self.nodes if isinstance(cpu_core, int) else [int(c) for c in cpu_core]
        if len(cores) != self.nodes or (self.nodes > 1 and isinstance(cpu_core, int) and cpu_core >= 0):
            raise ValueError(f"start_thread takes one core per NUMA group ({self.nodes}), got {cpu_core!r}")
        for core in cores:
            check_not_reserved(core)
        self._module.expert_stream_start_thread(
            self.handle,
            torch.tensor(cores, dtype=torch.int64),
            int(fatal_wait_s * 1e9),
            _spin_ns(spin_us),
            int(busy_poll),
        )
        self.threaded = True

    def pause(self, timeout_s: float) -> None:
        """Hand the slots to the caller: return once the thread is between two requests.

        The caller must have synchronized the device stream, so no demand is pending.
        Not reentrant: one owner (the slot table's depth counter) pauses and resumes.
        Raises RuntimeError if the copy thread still has a job or the thread does not
        pause within ``timeout_s``.
        """
        if not self.threaded:
            return
        outcome = int(
            self._module.expert_stream_pause(self.handle, int(timeout_s * 1e9))
        )
        if outcome == 2:
            raise RuntimeError(
                "exl3 RAM miss: not paused, the copy thread still has a job (unsynchronized stream)"
            )
        if outcome != 1:
            raise RuntimeError(
                f"exl3 RAM miss thread did not pause within {timeout_s} s"
            )

    def resume(self) -> None:
        """Give the slots back to the service thread."""
        if self.threaded:
            self._module.expert_stream_resume(self.handle)

    def pump(self) -> int:
        """Serve the pending requests on the calling thread (no service thread)."""
        return int(self._module.expert_stream_pump(self.handle))

    def pump_group(self, group: int) -> int:
        """Test only: serve NUMA group ``group``'s next pending request alone, on the calling thread."""
        _refuse_test_only("pump_group", self.variant)
        return int(self._module.expert_stream_pump_group(self.handle, int(group)))

    def contains(self, row: int, expert: int) -> bool:
        """Return True once the expert holds a slot in ``row``.

        That is from the moment the thread claims the slot, before its read has
        finished: it does not mean the bytes are in RAM. The read is done when
        ``layer_rows()`` counts the row. Needs the thread paused (or no thread).
        """
        self._check(row, expert)
        return bool(self._module.expert_stream_contains(self.handle, row, expert))

    def touch(self, row: int, expert: int) -> None:
        """Refresh the recency stamp of a resident expert of ``row`` (a hit)."""
        self._check(row, expert)
        self._module.expert_stream_touch(self.handle, row, expert)

    def assign(
        self,
        row: int,
        expert: int,
        protected: Iterable[int] = (),
        protected_fallback: bool = True,
    ) -> tuple[int, Optional[int]]:
        """Claim a pinned slot for ``expert`` in ``row``.

        Returns ``(slot, evicted_expert_or_None)``. A free slot is taken first; else
        the least recently used victim that is not hot or ``protected``, and with
        ``protected_fallback`` the least recently used protected one when no other
        victim exists. Raises ValueError if the expert already holds a slot and
        RuntimeError if every slot is protected or leased.
        """
        self._check(row, expert)
        out = torch.zeros(2, dtype=torch.int64)
        self._module.expert_stream_assign(
            self.handle, row, expert, _ids(protected), int(protected_fallback), out
        )
        slot, evicted = int(out[0]), int(out[1])
        if evicted == -2:
            raise ValueError(f"expert {expert} already holds a pinned slot")
        if slot < 0:
            raise RuntimeError(
                "every pinned host slot holds a protected or leased expert"
            )
        return slot, (None if evicted < 0 else evicted)

    def release(self, row: int, slot: int) -> None:
        """Free ``slot`` of ``row``."""
        self._check(row, slot=slot)
        self._module.expert_stream_release(self.handle, row, slot)

    def fill_begin(
        self,
        row: int,
        experts: Sequence[int],
        protected: Iterable[int] = (),
        fallback: bool = False,
    ) -> tuple[list[int], int]:
        """Claim slots for ``experts`` of ``row`` and read them on a helper thread.

        Used by prefill fills. Needs the service thread paused (or no thread), and every
        expert must hold no slot. Claiming stops at the first expert with no victim.
        Returns the claimed prefix's slots and the rows evicted for them.
        """
        self._check(row)
        experts = [int(expert) for expert in experts]
        for expert in experts:
            self._check(row, expert)
        out = torch.zeros(len(experts) + 1, dtype=torch.int64)
        claimed = int(
            self._module.expert_stream_fill_begin(
                self.handle, row, _ids(experts), _ids(protected), int(fallback), out
            )
        )
        values = out.tolist()
        return values[:claimed], values[len(experts)]

    def fill_wait(self, rows: int, timeout_s: float) -> None:
        """Return once the first ``rows`` claimed rows of the running fill have landed.

        Raises RuntimeError if the fill failed first or ``timeout_s`` passes.
        """
        outcome = int(
            self._module.expert_stream_fill_wait(
                self.handle, int(rows), int(timeout_s * 1e9)
            )
        )
        if outcome == 0:
            raise RuntimeError(
                f"exl3 RAM miss: a prefill fill failed before its first {rows} rows landed"
            )
        if outcome != 1:
            raise RuntimeError(
                f"exl3 RAM miss: a prefill fill did not land {rows} rows within {timeout_s} s"
            )

    def fill_landed(self) -> int:
        """Return how many claimed rows have landed so far (a claim-order prefix).

        Never blocks.
        """
        return int(self._module.expert_stream_fill_landed(self.handle))

    def fill_end(self) -> bool:
        """Join the fill; False if it failed (rows that did not land are released)."""
        return bool(self._module.expert_stream_fill_end(self.handle))

    def slot_info(self, row: int) -> list[tuple[int, int, int]]:
        """Return ``(state, expert, stamp)`` per slot; state 0 FREE, 2 READY, 3 STAGING.

        Paused or pumping only.
        """
        self._check(row)
        out = torch.empty(int(self.tables.capacity[row]) * 3, dtype=torch.int64)
        self._module.expert_stream_slot_info(self.handle, row, out)
        values = out.tolist()
        return [tuple(values[i : i + 3]) for i in range(0, len(values), 3)]

    def reserve_staging(self, k: Optional[int] = None) -> None:
        """Reserve every row's staging slots and publish its tag-1 map delta.

        The staging slots are the first ``min(k, capacity - 1)`` free slots of each row
        (``k`` defaults to the wire's lanes);
        no slot is ever taken from them afterwards. Call once, before the thread starts
        (or paused), with every tier empty. A row with fewer than 2 slots raises. See
        ``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "Deltas and the bulk delta".
        """
        k = self.wire.lanes if k is None else k
        self._module.expert_stream_reserve_staging(self.handle, int(k))

    def take_bulk_delta(self) -> torch.Tensor:
        """Return the eager paths' map changes since the last call.

        The result is int32 ``[n, 3]`` of (row, expert, slot). Paused only.
        """
        out = torch.empty(
            (int(self._module.expert_stream_bulk_delta_count(self.handle)), 3),
            dtype=torch.int32,
        )
        self._module.expert_stream_take_bulk_delta(self.handle, out)
        return out

    def handled_through(self) -> int:
        """Test only: return the last request seq the service finished."""
        return int(self._module.expert_stream_handled_through(self.handle)) & 0xFFFFFFFF

    def victim_census(self, row: int, wanted: Iterable[int] = ()) -> tuple[int, int]:
        """Return ``(free, evictable)`` slots a request wanting ``wanted`` could take.

        Counted without taking any. Paused or pumping only.
        """
        self._check(row)
        out = torch.empty(2, dtype=torch.int64)
        self._module.expert_stream_victim_census(self.handle, row, _ids(wanted), out)
        return tuple(out.tolist())

    def close_admission(self) -> None:
        """Shutdown step, once the device is synchronized: serve nothing new."""
        self._module.expert_stream_close_admission(self.handle)

    def busy_episode(self) -> int:
        """Return the busy episode of the request (or fill) now in service, 0 when none.

        Each episode gets a new value; the watchdog times it on its own thread (its
        stuck rule).
        """
        return int(self._module.expert_stream_busy_episode(self.handle))

    def set_prefill_share(self, share: int) -> None:
        """Set the rows a prefill may own per layer before it evicts its own rows.

        Past this share, a prefill's admissions evict its own rows instead of decode's;
        0 is off.
        """
        self._module.expert_stream_set_prefill_share(self.handle, int(share))

    def enable_copy_engine(
        self, device: int, *, spin_us: int = -1, wait_timeout_ms: int = 2000, cpus: Sequence[int] = ()
    ) -> None:
        """Start the copy-engine thread on CUDA ``device`` (-1: the CPU test backend).

        Call before the service thread starts. It copies nothing until
        :meth:`arm_copy_engine`, and then only rows :meth:`set_copy_table` registered.
        ``wait_timeout_ms`` bounds an armed copy wait: the service watchdog aborts the
        process once a closed gate has held the decode stream that long
        (``SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`` in a server). The idle copy thread polls for
        ``spin_us``, then sleeps on its doorbell until the next submit; -1 never sleeps
        (``SGLANG_DSV41_RAM_MISS_COPY_IDLE_SPIN_US`` in a server). ``cpus`` is its core
        (``ThreadingConfig.copy_cpus``); empty inherits the caller's affinity.
        """
        self._module.expert_stream_enable_copy_engine(
            self.handle,
            int(device),
            _spin_ns(spin_us),
            int(wait_timeout_ms * 1e6),
            torch.tensor([int(c) for c in cpus], dtype=torch.int64),
        )

    def set_copy_table(
        self, row: int, table: torch.Tensor, dst_rows: int, *, sm_mask: int = 0
    ) -> None:
        """Register row ``row``'s copy table.

        ``table`` is int64 ``[n, 3]`` of (source slab, destination tensor, row bytes)
        addresses, as in ``ExpertRowSegments.table``; every destination tensor holds
        ``dst_rows`` rows. Bit i of ``sm_mask`` leaves entry i to the copy wait's SM
        reads.
        """
        self._check(row)
        entries = table.detach().to("cpu", torch.int64).contiguous()
        if entries.dim() != 2 or entries.shape[1] != 3 or entries.shape[0] < 1:
            raise ValueError(
                f"a copy table is int64 [n, 3], not {tuple(entries.shape)}"
            )
        if dst_rows < 1:
            raise ValueError("a copy table needs at least one destination row")
        self._module.expert_stream_set_copy_table(
            self.handle, row, entries, int(dst_rows), int(sm_mask)
        )

    def arm_copy_engine(self, on: bool = True) -> None:
        """Let the service type resident hits as copy-engine lanes and copy them itself.

        Applies to requests whose post allows it.
        """
        self._module.expert_stream_arm_copy_engine(self.handle, int(bool(on)))

    def enable_cpu_experts(
        self,
        kernel: int,
        split: Sequence[int],
        cores: Sequence[int],
        x_rows: torch.Tensor,
        out_rows: torch.Tensor,
        *,
        threads: int,
        group: int = 0,
        spin_us: int = -1,
        keep_warm_us: int = 0,
    ) -> None:
        """Start NUMA group ``group``'s CPU expert thread (after the copy engine, before the service).

        ``kernel`` is the format's ``CpuExpertKernel`` address (the trait's ``kernel_address()``);
        its library must stay loaded while the host runs. Each row joins once its layer is set
        (:meth:`set_cpu_layer`). Of a captured
        post's n eligible lanes, ``split[n]`` are computed on the CPU (n = 0..``wire.lanes``;
        the device reads the table). ``x_rows`` (uint8 ``[rows, stride]``) and ``out_rows``
        (float32 ``[rows, hidden]`` at one node, otherwise ``[rows, 2 * nodes, hidden]``:
        group g's CPU-hit and CPU-miss partial sums are parts ``2g`` and ``2g + 1``) are
        pinned host rows: the post kernel stages a row's input in the first, the CPU
        writes its partial sums to the second and the device reads them. The host keeps
        references to both. The idle thread holds its workers in the kernel's keep-warm, in
        register work for ``keep_warm_us`` after each job and in PAUSE for ``spin_us`` after
        that, then releases them and sleeps until the next submit; -1 holds them until the
        next submit (``SGLANG_DSV41_CPU_EXPERTS_IDLE_SPIN_US`` in a server).
        """
        lanes = self.wire.lanes
        if len(split) != lanes + 1:
            raise ValueError(
                f"the CPU split table has {lanes + 1} entries (n = 0..{lanes}), not {len(split)}"
            )
        if (
            x_rows.dtype != torch.uint8
            or x_rows.dim() != 2
            or x_rows.device.type != "cpu"
            or not x_rows.is_contiguous()
        ):
            raise ValueError("x_rows must be a contiguous host uint8 [rows, n] tensor")
        if (
            out_rows.dtype != torch.float32
            or out_rows.device.type != "cpu"
            or not out_rows.is_contiguous()
            or not (
                (out_rows.dim() == 2 and self.nodes == 1)
                or (out_rows.dim() == 3 and out_rows.shape[1] == 2 * self.nodes)
            )
        ):
            raise ValueError(
                "out_rows must be a contiguous host float32 [rows, hidden] at one node or [rows, 2 * nodes, hidden] tensor"
            )
        parts = 1 if out_rows.dim() == 2 else 2
        hidden = int(out_rows.shape[-1])
        self._module.expert_stream_enable_cpu_experts(
            self.handle,
            int(group),
            int(kernel),
            torch.tensor(list(split), dtype=torch.int64),
            torch.tensor(list(cores), dtype=torch.int64),
            x_rows,
            out_rows.view(out_rows.shape[0], -1),
            hidden,
            parts,
            int(threads),
            _spin_ns(spin_us),
            int(keep_warm_us * 1e3),
        )
        self.cpu_rows = (x_rows, out_rows)

    def set_cpu_layer(self, row: int, spec) -> None:
        """Make ``row``'s layer with the enabled kernel from ``spec`` (a ``CpuExpertLayerSpec``).

        Once per row, at any time. The host keeps ``spec.keep`` (the slabs the layer reads) alive.
        """
        slabs, params = _layer_tensors(spec)
        self._module.expert_stream_set_cpu_layer(
            self.handle, row, slabs, int(spec.capacity), int(spec.hidden), int(spec.intermediate),
            int(spec.activation), float(spec.act_limit), params,
        )
        self._cpu_layer_keep[row] = spec.keep

    def set_cpu_split(self, split: Sequence[int], group: int = 0) -> None:
        """Set NUMA group ``group``'s new split table (CPU lanes per n eligible lanes, n = 0..``wire.lanes``).

        Allowed at any time; the device reads it.
        """
        if len(split) != self.wire.lanes + 1:
            raise ValueError(
                f"the CPU split table has {self.wire.lanes + 1} entries (n = 0..{self.wire.lanes}), not {len(split)}"
            )
        self._module.expert_stream_set_cpu_split(
            self.handle, int(group), torch.tensor(list(split), dtype=torch.int64)
        )

    def cpu_stats(self, group: int = 0) -> dict[str, int]:
        """Return the jobs and lanes group ``group``'s CPU expert thread computed, and its forward ns."""
        out = torch.zeros(3, dtype=torch.int64)
        self._module.expert_stream_cpu_stats(self.handle, int(group), out)
        jobs, lanes, ns = out.tolist()
        return {"jobs": jobs, "lanes": lanes, "forward_ns": ns}

    def copy_expert_bytes(self, row: int) -> int:
        """Return the bytes the DMA moves per expert of ``row``, for CPU calibration.

        That is the row's copy table less its SM entries.
        """
        return int(self._module.expert_stream_copy_expert_bytes(self.handle, int(row)))

    def calibrate_cpu_split(
        self,
        row: int,
        *,
        device: int,
        reps: int,
        scratch: torch.Tensor,
        timeout_s: float = 1.0,
        group: int = 0,
    ) -> torch.Tensor:
        """Run the CPU split's startup calibration on ``row``, on group ``group``'s engine and RAM slots.

        Returns float64 ``calibration_shape(wire.lanes)`` mean ms: row 0 is ``cpu[k]``, row 1 ``link[m]`` and
        row ``1 + n`` ``both[n][k]`` (k <= n). The caller owns the tier (paused, or no
        thread). ``device`` -1 copies with the test backend; ``scratch`` holds
        one expert per lane (``wire.lanes`` of them) on that device. Raises RuntimeError on failure.
        """
        out = torch.zeros(calibration_shape(self.wire.lanes), dtype=torch.float64)
        self._module.expert_stream_calibrate_cpu_split(
            self.handle,
            int(group),
            int(row),
            int(device),
            int(reps),
            scratch.data_ptr(),
            scratch.numel() * scratch.element_size(),
            int(timeout_s * 1e9),
            out,
        )
        return out

    def test_kernel_address(self, ns_per_expert: int = 0, *, fail: int = 0, zero: bool = False) -> int:
        """Test only: the instr build's fake ``CpuExpertKernel`` (``FakeKernel``, ffi_test_exports.h), reset.

        A forward spins ``ns_per_expert`` per expert and writes ``out[j] = (out[j] if accumulate else j) + sum(w * (s +
        1))`` (``zero``: a zero partial); ``fail`` nonzero makes every forward throw. Instrumented build only.
        """
        _refuse_test_only("test_kernel_address", self.variant)
        return int(self._module.expert_stream_test_kernel_address(int(ns_per_expert), int(fail), int(bool(zero))))

    def test_kernel_calls(self) -> list[dict]:
        """Test only: the fake kernel's forwards since :meth:`test_kernel_address`, one per row, in order."""
        _refuse_test_only("test_kernel_calls", self.variant)
        lanes = self.wire.lanes
        width = 6 + 2 * lanes
        count = int(self._module.expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
        out = torch.zeros((count, width), dtype=torch.float64)
        self._module.expert_stream_test_kernel_calls(out)
        calls = []
        for row in out.tolist()[:count]:
            k = int(row[4])
            calls.append({
                "core": int(row[0]), "affinity": int(row[1]), "threads": int(row[2]), "accumulate": bool(row[3]),
                "slots": [int(s) for s in row[5 : 5 + k]], "weights": row[5 + lanes : 5 + lanes + k],
                "capacity": int(row[5 + 2 * lanes]),
            })
        return calls

    def test_kernel_hold(self, core: int, on: bool = True) -> None:
        """Test only: hold (or release) the fake forwards whose worker-0 core is ``core``."""
        _refuse_test_only("test_kernel_hold", self.variant)
        self._module.expert_stream_test_kernel_hold(int(core), int(bool(on)))

    def test_keep_warm_calls(self) -> int:
        """Test only: calls of the fake kernel's keep-warm since :meth:`test_kernel_address`."""
        _refuse_test_only("test_keep_warm_calls", self.variant)
        return int(self._module.expert_stream_test_keep_warm_calls())

    def test_keep_warm_core(self) -> int:
        """Test only: the first core the fake keep-warm's last call took (-1 before any call)."""
        _refuse_test_only("test_keep_warm_core", self.variant)
        return int(self._module.expert_stream_test_keep_warm_core())

    def copy_engine_idle(self, timeout_s: float) -> bool:
        """Return whether every job given to the copy thread finished in ``timeout_s``.

        A job that failed counts as finished.
        """
        return bool(
            self._module.expert_stream_copy_engine_idle(
                self.handle, int(timeout_s * 1e9)
            )
        )

    def copy_engine_release(self, marks: int = -1) -> None:
        """Test only (CPU backend): let ``marks`` more copy marks complete.

        A negative value lets every mark complete.
        """
        self._module.expert_stream_copy_engine_release(self.handle, int(marks))

    def copy_engine_fail(self, *, issue: bool = False, query: bool = False) -> None:
        """Test only (CPU backend): make issuing a copy, or querying a mark, fail.

        Instrumented build only.
        """
        _refuse_test_only("copy_engine_fail", self.variant)
        self._module.expert_stream_copy_engine_fail(self.handle, int(issue), int(query))

    def copy_engine_ballast(
        self, dst: Optional[torch.Tensor], src: Optional[torch.Tensor]
    ) -> None:
        """Test only: copy ``src`` into ``dst`` (same size) ahead of every copy job.

        This delays each job's completion; ``None`` turns it off. The caller keeps both
        tensors alive while it is on. Instrumented build only.
        """
        _refuse_test_only("copy_engine_ballast", self.variant)
        if dst is None or src is None:
            self._module.expert_stream_copy_engine_ballast(self.handle, 0, 0, 0)
            return
        nbytes = dst.numel() * dst.element_size()
        if nbytes != src.numel() * src.element_size() or not (
            dst.is_contiguous() and src.is_contiguous()
        ):
            raise ValueError("ballast tensors must be contiguous and of one byte size")
        self._module.expert_stream_copy_engine_ballast(
            self.handle, dst.data_ptr(), src.data_ptr(), nbytes
        )

    def copy_engine_marked(self) -> int:
        """Test only (CPU backend): return the copy marks recorded so far.

        There is one per job issued.
        """
        return int(self._module.expert_stream_copy_engine_marked(self.handle))

    def mapping(self, row: int) -> list[int]:
        """Return each expert's READY slot, else -1.

        This is the published slot map (``slot_map``), read without waiting.
        """
        self._check(row)
        out = torch.empty(self.experts, dtype=torch.int64)
        self._module.expert_stream_mapping(self.handle, row, out)
        return out.tolist()

    def slot_to_expert(self, row: int) -> list[int]:
        """Return the expert held by each slot of ``row``. Paused or pumping only."""
        self._check(row)
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        self._module.expert_stream_slot_to_expert(self.handle, row, out)
        return out.tolist()

    def lru_order(self, row: int) -> list[int]:
        """Return the READY slots' experts, least recently used first.

        Paused or pumping only.
        """
        self._check(row)
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        count = int(self._module.expert_stream_lru_order(self.handle, row, out))
        return out[:count].tolist()

    def set_hot(self, row: int, experts: Iterable[int]) -> None:
        """Set the row's hot set (never evicted). Paused or pumping only."""
        self._check(row)
        self._module.expert_stream_set_hot(
            self.handle, row, _ids(e for e in experts if e >= 0)
        )

    def version(self) -> int:
        """Return the functional version counter."""
        return self.counters()["version"]

    def inject(
        self,
        delay_s: float = 0.0,
        fail_reads: bool = False,
        delay_after_demands: int = 0,
    ) -> None:
        """Test only: inject faults (see ``RamTier::inject``).

        A failed read aborts the process. Instrumented build only.
        """
        _refuse_test_only("inject", self.variant)
        self._module.expert_stream_inject(
            self.handle, int(delay_s * 1e9), int(fail_reads), delay_after_demands
        )

    def inject_group_stall(self, group: int, seconds: float) -> None:
        """Test only: NUMA group ``group``'s service thread sleeps ``seconds`` before it reads its next record."""
        _refuse_test_only("inject_group_stall", self.variant)
        self._module.expert_stream_inject_group_stall(self.handle, int(group), int(seconds * 1e9))

    def piece_runs(self) -> torch.Tensor:
        """Return the stream kernel's piece table.

        The table is int32 ``[layers, experts, STAGE_PIECES, segments, 2]``; each run is
        a ``(lo, hi)`` byte range of its segment's name row, cut by the reader's own row
        geometry. Raises RuntimeError if a (row, expert) cannot be cut into pieces.
        """
        tables = self.tables
        runs = torch.zeros(
            (self.layers, self.experts, STAGE_PIECES, int(tables.segments.shape[0]), 2),
            dtype=torch.int32,
        )
        refused = int(
            self._module.expert_stream_piece_runs(*_table_args(tables)[:-1], runs)
        )
        if refused:
            # A refused row's runs are empty: the stream kernel would admit a READY hit
            # of it, copy nothing and commit.
            raise RuntimeError(
                f"exl3 RAM miss: piece streaming cannot cut {refused} (row, expert) rows into pieces"
            )
        return runs

    def inject_fault(self, **faults) -> None:
        """Test only: hand the tier's reader a whole ``ReadFault``.

        The keywords are those of ``_fault_tensor``, the vocabulary
        ``read_rows_with_fault`` documents. Unlike ``inject(fail_reads=True)`` the read
        still runs, so the fault acts on rows that already packed. It is installed
        before the next read and stays until replaced; ``inject_fault()`` with no
        keywords clears it. ``abandon_after`` and ``step`` are not faults and are
        ignored. Call-numbered faults (``submit_call``, ``cqe_call``) count from the
        reader's creation. Instrumented build only: production refuses rather than
        store a fault it has no code to apply.
        """
        _refuse_test_only("inject_fault", self.variant)
        self._module.expert_stream_inject_fault(self.handle, _fault_tensor(**faults))

    def enable_trace(self, capacity: int = 8192) -> None:
        """Record one stage record per served request.

        Up to ``capacity`` records are kept undrained; more are dropped and counted.
        Call before ``start_thread``. With the trace off the service takes no
        timestamps.
        """
        _stage_words(self._layout, self.variant, self.wire.lanes)
        self._module.expert_stream_trace_enable(self.handle, int(capacity))

    def drain_trace(self, limit: int = 4096) -> list[dict]:
        """Return the stage records not yet drained, oldest first, decoded by
        ``stage_records``."""
        out = []
        while True:
            words = torch.empty(
                (limit, len(stage_fields(self.wire.lanes))), dtype=torch.int64
            )
            count = int(self._module.expert_stream_trace_drain(self.handle, words))
            out.extend(stage_records(words[:count], self.wire.lanes))
            if count < limit:
                return out

    def trace_dropped(self) -> int:
        """Return the stage records dropped because the trace ring was full."""
        return int(self._module.expert_stream_trace_dropped(self.handle))

    def trace_clock_reads(self) -> int:
        """Test only: return the clock reads taken for trace records.

        The count is process-wide and cumulative; zero growth while the trace is off
        shows that a disabled trace does no timing work. Instrumented build only.
        """
        _refuse_test_only("trace_clock_reads", self.variant)
        return int(self._module.expert_stream_trace_clock_reads())

    def counters(self) -> dict[str, int]:
        """Return every counter on the instrumented build.

        Production has no metrics and returns only ``CORE_COUNTERS``.
        """
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.expert_stream_counters(self.handle, out)
        values = dict(zip(COUNTERS, out.tolist()))
        # Every instrumented variant ("instr", "instr_tsan") compiles the metrics; only
        # "prod" drops them.
        return (
            values if self.variant != "prod" else {k: values[k] for k in CORE_COUNTERS}
        )

    def group_counters(self, group: int) -> dict[str, int]:
        """NUMA group ``group``'s own counters (its service thread's), by the names ``counters`` uses."""
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.expert_stream_group_counters(self.handle, int(group), out)
        return dict(zip(COUNTERS, out.tolist()))

    def layer_rows(self) -> list[int]:
        """Return the rows read for demands, per streamed layer (the RAM misses)."""
        out = torch.zeros(self.layers, dtype=torch.int64)
        self._module.expert_stream_layer_rows(self.handle, out)
        return out.tolist()

    def stop(self) -> None:
        """Stop the service thread, log the counters and close the service.

        Idempotent.
        """
        close = getattr(self, "_close", None)
        if close is not None and close.alive:
            try:
                if self.threaded:
                    self._module.expert_stream_stop_thread(self.handle)
                    self.threaded = False
                # One line per window; benchmark arms grep it, so keep its shape.
                sys.stderr.write(
                    "exl3 RAM miss thread counters "
                    + json.dumps(self.counters())
                    + "\n"
                )
                if self.nodes > 1:
                    for group in range(self.nodes):
                        sys.stderr.write(
                            f"exl3 RAM miss group {group} counters " + json.dumps(self.group_counters(group)) + "\n"
                        )
            finally:
                close()


# The device's own words (``lease_device.cuh``, ``kPosted``..``kDeadlineHi``): they live
# in device memory and are never on the wire.
STATE_WORDS = {
    "posted": 0,
    "pending": 1,
    "epoch": 2,
    "pending_epoch": 3,
    "deadline_lo": 4,
    "deadline_hi": 5,
}


_LEASE_METHODS = {
    "expert_stream_post": "post",
    "expert_stream_map_bulk_apply": "map_bulk_apply",
    "expert_stream_wire_nodes": "wire_nodes",
}
_ROW_COPY_METHODS = {
    "expert_stream_lease_stream": "lease_stream",
    "expert_stream_lease_copy_wait": "lease_copy_wait",
}


def _device_wrappers(layout: str = "exl3") -> list[tuple[str, str]]:
    """Return the (FFI name, C++ symbol) pairs the device module exports."""
    device_layout = LAYOUTS[layout].device_layout
    return [
        (name, f"LeaseProtocolKernel::{method}")
        for name, method in _LEASE_METHODS.items()
    ] + [
        (name, f"RowCopyKernel<{device_layout}>::{method}")
        for name, method in _ROW_COPY_METHODS.items()
    ]


def _device_module(layout: str = "exl3", lanes: int = 8, nodes: int = 1) -> Module:
    """Return the cached device module for ``layout``, ``lanes`` and ``nodes``."""
    wire = expert_lease_block.wire_layout(lanes, nodes)
    return _device_module_cached(layout, wire.lanes, wire.nodes)


@cache_once
def _device_module_cached(layout: str, lanes: int, nodes: int) -> Module:
    return load_jit(
        f"expert_stream_{layout}{_suffix(lanes, nodes)}",
        cuda_files=[LAYOUTS[layout].device_source],
        cuda_wrappers=_device_wrappers(layout),
        extra_cuda_cflags=[
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
    )


def device_module_with_hooks(
    defines: Sequence[str], layout: str = "exl3", lanes: int = 8, nodes: int = 1
) -> Module:
    """Test only: build the device kernels with the ``EXL3_RAM_MISS_TEST_*`` hooks on.

    ``defines`` are ``NAME`` or ``NAME=value`` entries. The result is a module of its
    own; production builds with none, so its kernels carry no test knob.
    """
    if not defines or not all(d.startswith("EXL3_RAM_MISS_TEST_") for d in defines):
        raise ValueError(f"not a set of EXL3_RAM_MISS_TEST_* hooks: {defines}")
    wire = expert_lease_block.wire_layout(lanes, nodes)
    lanes, nodes = wire.lanes, wire.nodes
    return load_jit(
        f"expert_stream_{layout}{_suffix(lanes, nodes)}",
        "test",
        cuda_files=[LAYOUTS[layout].device_source],
        cuda_wrappers=_device_wrappers(layout),
        extra_cuda_cflags=[
            *(f"-D{d}" for d in defines),
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
    )


def stream_segment_map(segments, tables, row: int) -> torch.Tensor:
    """Return the stream kernel's view of a copy table (``ExpertRowSegments``).

    The result is int32 ``[S + n]``. The first S entries give, for each of ``tables``'
    S row segments, its entry in the table (the pair whose source is that segment's
    name slab). Then comes one flag per table entry, 1 for an entry no row segment
    names: such an entry holds no host-read bytes and is copied whole with piece 0.
    """
    table = segments.table.cpu().tolist()
    for source, destination, row_bytes in table:
        # stream_copy_slice would fall back to 1-byte copies off 16-byte alignment;
        # refuse instead.
        if destination % 16 or row_bytes % 16:
            raise ValueError(
                f"the stream kernel copies 16-byte units: destination {destination:#x} with rows of "
                f"{row_bytes} B is not 16-byte aligned"
            )
    slabs = [int(address) for address in tables.slabs[row].tolist()]
    entry_of: dict[int, int] = {}
    for name, address in enumerate(slabs):
        hits = [k for k, entry in enumerate(table) if entry[0] == address]
        if len(hits) > 1:
            raise ValueError(
                f"streamed name {name}'s slab is the source of {len(hits)} copy-table entries"
            )
        if hits:
            if table[hits[0]][2] != int(tables.row_bytes[name]):
                raise ValueError(
                    f"streamed name {name}: the copy table's rows hold {table[hits[0]][2]} B, "
                    f"the slab's {int(tables.row_bytes[name])} B"
                )
            entry_of[name] = hits[0]
    names = [int(name) for name in tables.segments[:, 0].tolist()]
    missing = sorted(set(names) - set(entry_of))
    if missing:
        raise ValueError(
            f"the copy table has no entry for streamed names {missing} of row {row}"
        )
    whole = [0 if k in entry_of.values() else 1 for k in range(len(table))]
    return torch.tensor(
        [entry_of[name] for name in names] + whole,
        dtype=torch.int32,
        device=segments.table.device,
    )


class ExpertStreamDevice:
    """The chain's device kernels, capturable in a CUDA graph.

    One chain runs per layer, in one stream: ``post -> C1 -> S -> CW -> stream wait ->
    CC`` (``analysis/dsv41-drive/LEASE_PROTOCOL.md``, "The chain"):

      * ``post`` applies the row's pending map delta, types each lane (hit, miss, copy
        engine, CPU) from the device map and publishes the request record;
      * C1 (row copy, outside this class) copies the lanes the SMs read from RAM;
      * ``stream`` (S) copies the miss lanes from their staging slots, piece by piece
        as the reader publishes them;
      * ``copy_wait`` runs CW (close the gate), the ``cuStreamWaitValue32`` wait and
        CC (commit), which traps unless ``CopyDone`` names this request.

    ``page`` and ``lease_block`` are the host's pinned tensors (device-readable through
    UVA); ``state`` holds the device words ``STATE_WORDS``. ``piece_runs`` is
    ``host.piece_runs()``; ``row_capacities`` is each row's pinned slot count
    (``tables.capacity``), which the post and S bound every host slot by.
    ``timeout_ms`` is the post's delta wait and S's deadline. ``hit_copy`` ("ce" or
    "sm") and ``cpu_misses`` carry ``SGLANG_DSV41_RAM_HIT_COPY`` and
    ``SGLANG_DSV41_CPU_EXPERTS_MISSES``; ``lease_pdl``
    (``SGLANG_DSV41_ENABLE_LEASE_PDL``) launches every chain kernel but C1 and CC with
    PDL.

    ``map_bank`` is the device's copy of the RAM tier's map: ``ram_slot``
    ``[layers, experts]`` starts at -1 and changes only through the deltas the post
    applies and ``map_bulk_apply``; ``staging`` is ``[layers, nodes * lanes]``, node-major.
    ``map_chain`` starts at 1, because tag 1 is the
    attach delta, so a zero-filled delta record (tag 0) is never taken for a published
    one. Per row, ``ce_ok``/``dst_rows`` say the copy engine may take a hit and
    ``cpu_ok`` that the CPU may; the service sets them as it learns them and the post
    reads them. The tensors the kernels exchange (``lane_kind``, ``lane_slot``, ...)
    are allocated once: a graph captures their addresses like any other argument.
    """

    def __init__(
        self,
        page,
        lease_block,
        *,
        device,
        layers: int,
        experts: int,
        timeout_ms: int,
        piece_runs: torch.Tensor,
        row_capacities: Sequence[int],
        hot_page=None,
        layout: str = "exl3",
        lease_pdl: bool = False,
        hit_copy: str = "ce",
        cpu_misses: bool = False,
        lanes: int = 8,
        nodes: int = 1,
    ) -> None:
        self.wire = expert_lease_block.wire_layout(lanes, nodes)
        if (
            page.numel() != self.wire.page_bytes
            or page.dtype != torch.uint8
            or page.device.type != "cpu"
            or not page.is_contiguous()
        ):
            raise ValueError(
                f"page must be a contiguous CPU uint8 tensor of {self.wire.page_bytes} bytes (new_page of this wire)"
            )
        if timeout_ms <= 0:
            raise ValueError("the RAM-miss wait timeout must be positive")
        cuda = torch.device(device).type == "cuda"
        # Checked before any CUDA call: the kernels read both through UVA, and an
        # unpinned address faults inside the captured graph.
        if cuda and not page.is_pinned():
            raise ValueError("page must be pinned for a CUDA device")
        expert_lease_block.check_lease_block(
            lease_block, layers, need_pinned=cuda, wire=self.wire
        )
        if lease_pdl and cuda and not is_arch_support_pdl():
            raise ValueError("lease-chain PDL needs sm_90 or newer (griddepcontrol)")
        if len(row_capacities) != layers:
            raise ValueError(
                f"{len(row_capacities)} row capacities for {layers} layers"
            )
        if (
            piece_runs.dtype != torch.int32
            or piece_runs.dim() != 5
            or piece_runs.shape[0] != layers
        ):
            raise ValueError(
                "piece_runs must be int32 [layers, experts, pieces, segments, 2] (host.piece_runs())"
            )
        if (
            piece_runs.shape[1] != experts
            or piece_runs.shape[2] != STAGE_PIECES
            or piece_runs.shape[4] != 2
        ):
            raise ValueError(
                f"piece_runs has shape {tuple(piece_runs.shape)} for {experts} experts"
            )
        self.page = page
        self.lease_block = lease_block
        self.layers = layers
        self.experts = experts
        self.timeout_ns = int(timeout_ms * 1_000_000)
        self.lease_pdl = bool(lease_pdl)
        self._row_capacities = tuple(int(c) for c in row_capacities)
        self._lease_address = int(lease_block.data_ptr())
        if hit_copy not in ("ce", "sm"):
            raise ValueError(f"hit_copy is 'ce' or 'sm', not {hit_copy!r}")
        self.hit_copy = hit_copy
        self.cpu_misses = bool(cpu_misses)
        state = torch.zeros(len(STATE_WORDS), dtype=torch.int32)
        # Continue from the page's head: the thread serves demand_head + 1 next, so a
        # device restarting at 1 over a used page would never be served.
        state[STATE_WORDS["posted"]] = page[
            WORDS["demand_head"] : WORDS["demand_head"] + 4
        ].view(torch.int32)[0]
        self.state = state.to(device)
        # Stable sentinels for absent tensors (see the class doc on graph capture).
        self._no_hot_slots = torch.empty(0, dtype=torch.int64, device=device)
        self._no_cpu = torch.empty(0, dtype=torch.int32, device=device)
        # The spill words of a post without spill: never read (the post's spill is 0).
        self._no_spill = (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
        self._module = None
        self._layout = layout
        self.hot_page = hot_page
        self._hot_address = 0
        self._hot_stride = 0
        if hot_page is not None:
            stride = hot_record_bytes(experts)
            if (
                hot_page.dtype != torch.uint8
                or hot_page.device.type != "cpu"
                or not hot_page.is_contiguous()
                or hot_page.numel() != HOT_RECORDS * stride
                or (cuda and not hot_page.is_pinned())
            ):
                raise ValueError(
                    f"hot_page must be a contiguous pinned CPU uint8 tensor of {HOT_RECORDS * stride} bytes"
                )
            self._hot_address = int(hot_page.data_ptr())
            self._hot_stride = stride
        self.map_bank = {
            "ram_slot": torch.full(
                (layers, experts), -1, dtype=torch.int32, device=device
            ),
            "staging": torch.full(
                (layers, self.wire.nodes * self.wire.lanes),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            "map_chain": torch.ones(layers, dtype=torch.int64, device=device),
            "map_applied": torch.zeros(layers, dtype=torch.int64, device=device),
            # Per row: the copy engine may take a hit (its copy table is set), its
            # destination rows, and the CPU may (its layer is registered).
            "ce_ok": torch.zeros(layers, dtype=torch.uint8, device=device),
            "dst_rows": torch.zeros(layers, dtype=torch.int32, device=device),
            "cpu_ok": torch.zeros(layers, dtype=torch.uint8, device=device),
        }
        self._row_capacity_tensor = torch.tensor(
            self._row_capacities, dtype=torch.int32, device=device
        )
        # The post's outputs: each lane's kind, source slot (S and CW read them) and home
        # node (CW reads it), and C1's compacted SM hits, in one order.
        self.lane_kind = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        self.lane_slot = torch.full((self.wire.lanes,), -1, dtype=torch.int32, device=device)
        self.lane_node = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        self.go_1 = torch.zeros(1, dtype=torch.int32, device=device)
        self.host_rows_1 = torch.zeros(self.wire.lanes, dtype=torch.int64, device=device)
        self.dst_slots_1 = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        # CW's words for CC: the lanes it armed the gate for, the CPU lanes, the CPU output parts; 0: none; a wide wire
        # appends the masks' high halves.
        self.ce_mask = torch.zeros(self.wire.ce_mask_words, dtype=torch.int32, device=device)
        # CPU experts, written by CC: the lanes the CPU computed, then the output parts holding their partial sums (0: none);
        # a wide wire appends the lanes' high half.
        self.cpu_lanes = torch.zeros(self.wire.cpu_lane_words, dtype=torch.int32, device=device)
        # CPU experts (``enable_cpu_experts``): the host rows the post stages each
        # layer's input into, and those the CPU expert thread writes each layer's
        # partial sum to; None when off.
        self.cpu_x_rows = None
        self.cpu_out_rows = None
        # Set by the row backend when it captures a post that lets the service copy; the
        # service arms its copy engine on it.
        self.copy_engine_captured = False
        self.piece_runs = piece_runs.to(device).contiguous()

    def _kernels(self):
        """Return the device module, loading it on first use."""
        if self._module is None:
            self._module = _device_module(
                self._layout, self.wire.lanes, self.wire.nodes
            )
        return self._module

    def enable_cpu_experts(self, x_rows: torch.Tensor, out_rows: torch.Tensor) -> None:
        """Take the same pinned rows the host's ``enable_cpu_experts`` took.

        There is one row per layer, readable through UVA.
        """
        for rows, name in ((x_rows, "x_rows"), (out_rows, "out_rows")):
            if (
                rows.shape[0] != self.layers
                or rows.device.type != "cpu"
                or not rows.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous host tensor with one row per layer"
                )
            if torch.device(self.state.device).type == "cuda" and not rows.is_pinned():
                raise ValueError(
                    f"{name} must be pinned: the kernels read and write it through UVA"
                )
        if (
            x_rows.stride(0) % 16
            or out_rows.data_ptr() % 16
            or x_rows.data_ptr() % 16
            or (out_rows.stride(0) * 4) % 16
        ):
            raise ValueError("the CPU expert rows must be 16-byte aligned")
        self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows

    def set_row_copy(self, row: int, dst_rows: int) -> None:
        """Record that the row's copy table is registered with ``dst_rows`` rows.

        The post may then type the row's hits kHitCopy.
        """
        self._check_row(row)
        self.map_bank["dst_rows"][row] = int(dst_rows)
        self.map_bank["ce_ok"][row] = 1

    def set_row_cpu(self, row: int) -> None:
        """Record that the row's CPU layer is registered.

        The post may then type its lanes kHitCpu or kMissCpu.
        """
        self._check_row(row)
        self.map_bank["cpu_ok"][row] = 1

    def map_bulk_apply(self, bulk: torch.Tensor) -> None:
        """Apply the eager paths' map changes on the current stream.

        ``bulk`` is ``host.take_bulk_delta()``, int32 ``[n, 3]``; it applies after every
        row's pending decode delta. The caller has synchronized the stream and paused
        the service.
        """
        if bulk.dtype != torch.int32 or bulk.dim() != 2 or bulk.shape[1] != 3:
            raise ValueError(
                f"the bulk delta is int32 [n, 3], not {bulk.dtype} {tuple(bulk.shape)}"
            )
        bank = self.map_bank
        self._kernels().expert_stream_map_bulk_apply(
            self._lease_address,
            bank["ram_slot"],
            bank["staging"],
            bank["map_chain"],
            bank["map_applied"],
            bulk.to(self.state.device).contiguous(),
            self._row_capacity_tensor,
        )

    def cpu_out_address(self, row: int) -> int:
        """Return the host address of ``row``'s CPU partial sums (part 0).

        The fused MoE's route tables read it.
        """
        self._check_row(row)
        return int(self.cpu_out_rows[row].data_ptr())

    def cpu_out_part_stride(self) -> int:
        """Return the floats from a row's part 0 (CPU hits) to its part 1 (CPU misses).

        0 for one-part rows.
        """
        return int(self.cpu_out_rows.stride(1)) if self.cpu_out_rows.dim() == 3 else 0

    def _check_row(self, row: int) -> None:
        """Raise ValueError for a row outside ``[0, layers)``."""
        if not 0 <= row < self.layers:
            raise ValueError(f"row {row} is outside [0, {self.layers})")

    def _check_buffers(self, **buffers) -> None:
        """Raise ValueError unless each ``(tensor, dtype)`` buffer suits a kernel.

        The kernels cast each buffer's data pointer to one fixed type and read
        ``[0, lanes)``.
        """
        for name, (tensor, dtype) in buffers.items():
            if (
                tensor.dtype != dtype
                or tensor.device != self.state.device
                or not tensor.is_contiguous()
                or tensor.numel() < 1
            ):
                raise ValueError(
                    f"{name} must be a non-empty contiguous {dtype} tensor on {self.state.device}"
                )

    def post(
        self,
        row: int,
        planned,
        count,
        routes,
        dst_slots,
        hot_slots=None,
        hot_capacity: int = 0,
        captured: bool = False,
        cpu_input=None,
        spill=None,
    ) -> None:
        """Post the layer's request.

        Applies the row's pending map delta, types its lanes from the device map into
        ``lane_kind``/``lane_slot`` and C1's compaction, and publishes the record.
        ``dst_slots`` are the plan's int32 destination slots; ``captured`` lets the post
        type copy-engine and CPU lanes.

        ``cpu_input`` is ``(x [1, hidden], route weights aligned with routes)``, for CPU
        experts and only with ``captured``: the post stages x in the row's host row when
        a lane is the CPU's, and the weights in the record.

        ``spill`` is DIRECT's ``(overflow_flag int32 [1], gather_overflow int64 [1])`` for this row, given when lanes
        may have no VRAM victim (SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES): a lane whose ``dst_slots`` entry is -1
        becomes a CPU lane, or, when one cannot, the post serves the live prefix, lowers ``count`` and flags both words.
        """
        self._check_row(row)
        self._check_buffers(
            planned=(planned, torch.int64),
            count=(count, torch.int32),
            routes=(routes, torch.int64),
            dst_slots=(dst_slots, torch.int32),
        )
        if hot_slots is not None:
            self._check_buffers(hot_slots=(hot_slots, torch.int64))
            if self._hot_address == 0 or not 0 < hot_capacity <= hot_slots.numel():
                raise ValueError(
                    "EXL3 DIRECT needs a hot sidecar and a valid slot capacity"
                )
        cpu_x, cpu_weights, cpu_x_dst = self._no_cpu, self._no_cpu, 0
        if cpu_input is not None:
            if self.cpu_x_rows is None:
                raise RuntimeError("CPU experts are not enabled on this device")
            if not captured:
                raise ValueError("only a captured post stages the CPU experts' input")
            cpu_x, cpu_weights = cpu_input
            if cpu_x.shape[-1] * 2 > self.cpu_x_rows.shape[1]:
                raise ValueError(
                    f"a {cpu_x.shape[-1]}-wide input does not fit the {self.cpu_x_rows.shape[1]}-byte row"
                )
            cpu_x, cpu_weights = cpu_x.reshape(1, -1), cpu_weights.reshape(-1)
            cpu_x_dst = int(self.cpu_x_rows[row].data_ptr())
        spill_on, (overflow_flag, gather_overflow) = 0, self._no_spill
        if spill is not None:
            overflow_flag, gather_overflow = spill
            for name, t, dtype in (
                ("overflow_flag", overflow_flag, torch.int32),
                ("gather_overflow", gather_overflow, torch.int64),
            ):
                if t.dtype != dtype or t.numel() != 1 or t.device != self.state.device:
                    raise ValueError(f"{name} must be one {dtype} word on {self.state.device}")
            spill_on = 1
        bank = self.map_bank
        cpu_on = self.cpu_x_rows is not None
        self._kernels().expert_stream_post(
            self.page,
            self.state,
            planned,
            count,
            routes,
            row,
            self.experts,
            self._lease_address,
            self.timeout_ns,
            self._hot_address,
            self._hot_stride,
            hot_slots if hot_slots is not None else self._no_hot_slots,
            hot_capacity,
            dst_slots,
            int(bool(captured)),
            bank["ram_slot"],
            bank["staging"],
            bank["map_chain"],
            bank["map_applied"],
            bank["ce_ok"],
            bank["cpu_ok"],
            bank["dst_rows"],
            self._row_capacities[row],
            int(self.hit_copy == "ce"),
            int(cpu_on),
            int(self.cpu_misses and cpu_on),
            self.lane_kind,
            self.lane_slot,
            self.lane_node,
            self.go_1,
            self.host_rows_1,
            self.dst_slots_1,
            cpu_x,
            cpu_x_dst,
            cpu_weights,
            spill_on,
            overflow_flag,
            gather_overflow,
            int(self.lease_pdl),
        )

    def stream(
        self, row: int, planned, count, dst_slots, segments, segment_map
    ) -> None:
        """Copy the post's kMissGpu lanes from their staging slots (kernel S).

        Each piece goes to ``segments``' destinations as its bit is published, until
        every piece is copied. ``segment_map`` is ``stream_segment_map(segments, ...)``.
        """
        self._check_row(row)
        self._check_buffers(
            planned=(planned, torch.int64),
            count=(count, torch.int32),
            dst_slots=(dst_slots, torch.int32),
            segment_map=(segment_map, torch.int32),
        )
        row_segments = int(self.piece_runs.shape[3])
        if segment_map.numel() != row_segments + segments.table.shape[0]:
            raise ValueError(
                f"segment_map has {segment_map.numel()} entries, the kernel reads "
                f"{row_segments} + {segments.table.shape[0]}"
            )
        self._kernels().expert_stream_lease_stream(
            self.state,
            planned,
            count,
            dst_slots,
            row,
            self.experts,
            self._lease_address,
            self.lane_kind,
            self.lane_slot,
            segments.table,
            segment_map,
            row_segments,
            self.piece_runs,
            self._row_capacities[row],
            int(self.lease_pdl),
        )

    def copy_wait(
        self, count, dst_slots, sm_table: Optional[torch.Tensor] = None
    ) -> None:
        """Run the chain's tail: CW, a ``cuStreamWaitValue32`` on the gate, and CC.

        With ``sm_table`` (int64 ``[n, 3]`` on the device: the copy-table rows the
        service's ``sm_mask`` named), CW reads the small tensors of every kHitCopy lane
        from its RAM slot. It closes the gate when the record has host lanes; the
        stream then waits, with no SM spinning, until CW or the service's copy thread
        opens it. CC traps unless ``CopyDone`` names this request.
        """
        self._check_buffers(
            count=(count, torch.int32), dst_slots=(dst_slots, torch.int32)
        )
        sm_address, sm_count = 0, 0
        if sm_table is not None:
            if (
                sm_table.dtype != torch.int64
                or sm_table.dim() != 2
                or sm_table.shape[1] != 3
                or not sm_table.is_contiguous()
            ):
                raise ValueError(
                    f"the copy wait's SM table is a contiguous int64 [n, 3], not {tuple(sm_table.shape)}"
                )
            if sm_table.device != self.state.device:
                raise ValueError(
                    "the copy wait's SM table must live on the device the kernel reads it from"
                )
            sm_address, sm_count = sm_table.data_ptr(), int(sm_table.shape[0])
        self._kernels().expert_stream_lease_copy_wait(
            self.state,
            count,
            self._lease_address,
            self.lane_kind,
            self.lane_slot,
            self.lane_node,
            dst_slots,
            sm_address,
            sm_count,
            self.ce_mask,
            self.cpu_lanes if self.cpu_x_rows is not None else self._no_cpu,
            int(self.lease_pdl),
        )

    def stats(self) -> dict[str, int]:
        """Return the device words ``STATE_WORDS`` (a device-to-host copy)."""
        values = self.state.cpu().tolist()
        return {name: values[index] for name, index in STATE_WORDS.items()}
