"""Python wrappers for the RAM-miss expert-stream transport of streamed experts.

The transport is built per (format, router) pair (``LAYOUTS``: the engine's template
translation units, ``csrc/moe/expert_stream/engine_{host.cpp,device.cuh}``, instantiated
for ``csrc/moe/formats/<format>`` and the router of ``csrc/moe/expert_stream/router.h``);
every entry point takes its pair as ``layout``.

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
from typing import Iterable, NamedTuple, Optional, Sequence, TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.ops.moe import expert_lease_block
from sglang.kernels.ops.moe.expert_lease_block import CPU_TOKENS_MAX, cpu_row_bytes
from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_engine.config import ExpertStreamConfig, stream_config

# Rows per io_uring batch and per bounce bank. The C++ reader has kBanks = 2 banks of
# kBounceRows = 8 row slots each; a bank is reused only after every row read into it
# has packed.
BOUNCE_ROWS = 8

if TYPE_CHECKING:
    from tvm_ffi.module import Module


class TransportBuild(msgspec.Struct, frozen=True):
    """One instantiation of the transport for a (format, router) pair.

    ``host_sources`` maps each build variant (``VARIANTS``) to the host translation
    unit that binds the format's C++ layout and file reader. ``device_source`` is the
    translation unit ``RowCopyKernel`` is compiled in, ``device_layout`` the layout
    type it is instantiated with and ``spec_score_kernel`` the GPU scorer launcher
    (``spec_score.cuh``) the scorer exports bind. ``defines`` are the ``-D`` flags
    both translation units take; the engine's pick the format and router.
    """

    host_sources: dict[str, str]
    device_source: str
    device_layout: str
    spec_score_kernel: str = "SpecScoreKernel"
    defines: tuple[str, ...] = ()


# A transport build is named by its (format, router) pair, in this order.
BuildKey = tuple[str, str]
EXL3_BUILD: BuildKey = ("exl3", "sqrt_softplus")
NVFP4_BUILD: BuildKey = ("nvfp4", "sqrt_softplus")
MXFP4_BUILD: BuildKey = ("mxfp4", "sigmoid")
# MiMo V2.6 Pro once converted to NVFP4: the NVFP4 rows with MiMo's sigmoid router. The same host and device sources,
# so only the -D macros (and so the JIT keys) differ from NVFP4_BUILD.
NVFP4_MIMO_BUILD: BuildKey = ("nvfp4", "sigmoid")

_ENGINE_HOST = "moe/expert_stream/engine_host.cpp"
_ENGINE_DEVICE = "moe/expert_stream/engine_device.cuh"


def _engine_build(key: BuildKey) -> TransportBuild:
    """The engine's template translation units for ``key``: one host file for both variants (the instrumented one adds
    ``-DSGLANG_EXPERT_STREAM_INSTR``) and one device file, with the format and router chosen by macro."""
    fmt, router = key
    return TransportBuild(
        host_sources={"prod": _ENGINE_HOST, "instr": _ENGINE_HOST},
        device_source=_ENGINE_DEVICE,
        device_layout="sglang::expert_stream::EngineLayout",
        spec_score_kernel="SpecScoreKernelFor<sglang::expert_stream::EngineRouter>",
        defines=(f"-DSGLANG_EXPERT_FORMAT={fmt}", f"-DSGLANG_EXPERT_ROUTER={router}"),
    )


# One entry per (format, router) the transport is built for. A pair the engine does not
# build is absent: ``_build_for`` raises with the JIT key it looked for. Adding a format
# adds a ``csrc/moe/formats/<format>/format.h`` and an entry here.
LAYOUTS: dict[BuildKey, TransportBuild] = {
    key: _engine_build(key) for key in (EXL3_BUILD, NVFP4_BUILD, NVFP4_MIMO_BUILD, MXFP4_BUILD)
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


def host_variant(config: Optional[ExpertStreamConfig] = None) -> str:
    """Return the host build a new service loads.

    The instrumented build when this process writes a stream trace
    (``SGLANG_EXPERT_STREAM_TRACE``'s ``expert_path`` or ``job.prefix``)
    or injects a RAM-miss fault
    (``SGLANG_TEST_DSV41_RAM_MISS_FAULT``); production otherwise. ``config``
    defaults to the process's (``stream_config()``).
    """
    if _DEFAULT_VARIANT is not None:
        return _DEFAULT_VARIANT
    trace = (config or stream_config()).trace
    if trace.expert_path or trace.job.prefix or envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get():
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
    "read_rows_idle",
    "read_rows_two_span",
    "inject",
    "inject_fault",
    "inject_group_stall",
    "spec_place",
    "spec_pump",
    "hold_row",
    "inject_spec",
    "inject_demand_load",
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
    "test_kernel_max_rows",
    "draft_test_post",
    "draft_test_tear",
    "draft_test_finish_close",
    "draft_test_poll_pause",
    "kernel_forward_two_stage",
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


# cache_once keys f(), f(EXL3_BUILD) and f(layout=EXL3_BUILD) apart, so each cached loader
# below is called only positionally, through a wrapper: a (format, router) key and
# variant then have exactly one module whatever the call form.
def _suffix(lanes: int, nodes: int) -> str:
    """A build's module-name suffix: one node keeps the Phase 1 names, so its modules and caches are unchanged."""
    return f"_l{lanes}" if nodes == 1 else f"_l{lanes}_n{nodes}"


def _host_module(
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
    lanes: int = 8,
    nodes: int = 1,
    config: Optional[ExpertStreamConfig] = None,
) -> Module:
    """Return the cached host module for ``layout``, ``variant`` (default build), ``lanes`` and ``nodes``, with the
    options of ``config`` (default ``stream_config()``) installed: the C++ host reads no env, and each object it
    builds copies the options installed when it is built."""
    config = config or stream_config()
    variant = host_variant(config) if variant is None else variant
    wire = expert_lease_block.wire_layout(lanes, nodes)
    lanes, nodes = wire.lanes, wire.nodes
    if variant == "instr_tsan" and _ALLOW_TSAN:
        return _install_host_options(_host_module_tsan(layout, lanes, nodes), config)
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown host build variant {variant!r}; expected one of {VARIANTS}"
        )
    trigger_only = config.trace.draft_trigger.trigger_only
    if variant not in _build_for(layout, host_jit_key(layout, variant, lanes, nodes, trigger_only)).host_sources:
        raise ValueError(f"build {layout!r} has no {variant!r} host build variant")
    if trigger_only and variant != "prod":
        raise ValueError("SGLANG_EXPERT_STREAM_TRACE draft_trigger.trigger_only requires the production host build")
    return _install_host_options(_host_module_cached(layout, variant, lanes, nodes, trigger_only), config)


# The options last installed in each loaded host module, by id(module): a module is cached for the process, so an
# unchanged config costs no FFI call.
_INSTALLED_OPTIONS: dict[int, tuple] = {}


def _install_host_options(module: Module, config: ExpertStreamConfig) -> Module:
    """Install ``config``'s C++ options in ``module`` (trace_options.h, uring_options.h); returns the module."""
    trace, uring = config.trace, config.tiers.uring
    trace_args = (
        trace.job.prefix,
        int(trace.job.capacity),
        bool(trace.job.resources),
        int(trace.draft_trigger.arrival_us),
        int(trace.draft_trigger.pending_us),
        int(trace.draft_trigger.forward_us),
        trace.draft_trigger.report_prefix,
        trace.gate_path,
    )
    uring_args = (
        uring.mode,
        int(uring.queue_depth),
        bool(uring.fixed_files),
        uring.read_mode,
        uring.wait_mode,
        int(uring.sq_thread_idle_ms),
        bool(uring.diagnostics),
        uring.read_cuts,
    )
    if _INSTALLED_OPTIONS.get(id(module)) != (trace_args, uring_args):
        module.expert_stream_set_trace_options(*trace_args)
        module.expert_stream_set_uring_options(*uring_args)
        _INSTALLED_OPTIONS[id(module)] = (trace_args, uring_args)
    return module


def _check_key(layout) -> BuildKey:
    """``layout`` as a (format, router) key, or ValueError: a bare format name is refused so every caller states its router."""
    if not (isinstance(layout, tuple) and len(layout) == 2 and all(isinstance(part, str) for part in layout)):
        raise ValueError(
            f"an expert-stream build is a (format, router) key such as {EXL3_BUILD}, got {layout!r}; "
            f"registered: {sorted(LAYOUTS)}"
        )
    return layout


def host_jit_key(layout: BuildKey, variant: str, lanes: int = 8, nodes: int = 1, trigger_only: bool = False) -> str:
    """The JIT module name of the host build ``layout`` x ``variant`` x ``lanes`` x ``nodes``: it names both the format
    and the router, so two pairs never share a module."""
    fmt, router = _check_key(layout)
    return f"expert_stream_host_{fmt}_{router}_{variant}{'_trigger' if trigger_only else ''}{_suffix(lanes, nodes)}"


def device_jit_key(layout: BuildKey, lanes: int = 8, nodes: int = 1) -> str:
    """The JIT module name of the device build of ``layout`` for ``lanes`` and ``nodes`` (test and stamped builds add a
    marker after it)."""
    fmt, router = _check_key(layout)
    return f"expert_stream_{fmt}_{router}{_suffix(lanes, nodes)}"


def _build_for(layout: BuildKey, jit_key: str) -> TransportBuild:
    """The registered ``TransportBuild`` of ``layout``; ValueError naming ``jit_key``, the module it looked for, when the
    (format, router) pair is not one of ``LAYOUTS``."""
    try:
        return LAYOUTS[_check_key(layout)]
    except KeyError:
        raise ValueError(
            f"no expert-stream transport is registered for (format, router) {layout!r}: JIT key {jit_key!r} has no "
            f"build; registered: {sorted(LAYOUTS)}"
        ) from None


@cache_once
def _host_module_cached(layout: BuildKey, variant: str, lanes: int, nodes: int, trigger_only: bool = False) -> Module:
    # Hidden visibility keeps HostExports' registries and members private to each
    # module's .so; only the TVM_FFI_DLL_EXPORT entry points are exported.
    key = host_jit_key(layout, variant, lanes, nodes, trigger_only)
    build = _build_for(layout, key)
    return load_jit(
        key,
        cpp_files=[build.host_sources[variant]],
        extra_cflags=[
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            *build.defines,
            *(["-DSGLANG_EXPERT_STREAM_INSTR=1"] if variant == "instr" else []),
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
            *(["-DSGLANG_EXPERT_STREAM_DRAFT_TRIGGER_ONLY=1"] if trigger_only else []),
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
def _host_module_tsan(layout: BuildKey, lanes: int, nodes: int) -> Module:
    key = host_jit_key(layout, "instr_tsan", lanes, nodes)
    return load_jit(
        key,
        cpp_files=[_build_for(layout, key).host_sources["instr"]],
        extra_cflags=[
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            *_build_for(layout, key).defines,
            "-DSGLANG_EXPERT_STREAM_INSTR=1",
            "-fsanitize=thread",
            "-O1",
            "-g",
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
        extra_ldflags=["-luring", "-lpthread", "-ldl", "-fsanitize=thread"],
        header_only=False,
    )


def host_layout(layout: BuildKey = EXL3_BUILD, lanes: int = 8) -> tuple[tuple[str, ...], int]:
    """Return the host module's row layout.

    That is its tensor names in copy-table order and a bit mask of the ones the SMs
    can read. ``lanes`` picks the build whose module is read, so a launch reads the
    one it already builds instead of compiling another.
    """
    return _host_layout_cached(layout, expert_lease_block.wire_layout(lanes).lanes)


@cache_once
def _host_layout_cached(layout: BuildKey, lanes: int) -> tuple[tuple[str, ...], int]:
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


def _span_tensor(span_weights) -> torch.Tensor:
    """Per-span weights (w13's, w2's: one per mirror root each) as the host's float64 [2, roots]; None: [0, 0], the
    mirror's shares in both spans."""
    if span_weights is None:
        return torch.zeros((0, 0), dtype=torch.float64)
    w13, w2 = span_weights
    return torch.tensor([list(w13), list(w2)], dtype=torch.float64)


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
    layout: BuildKey = EXL3_BUILD,
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
    root: int = -1,
    mirror_caps: Sequence[int] = (),
    two_span_rows: int = 0,
    suffix_delay_ns: int = 0,
) -> torch.Tensor:
    """Pack fault keywords into the C++ fault tensor (``kFaultWords`` int64 words).

    The word order must stay in step with ``fault_from()`` in
    ``host/read_fault.h``; word 26 is reserved. The keywords are
    documented at ``read_rows_with_fault``; ``root`` narrows the per-extent faults to
    the drive a sub-read reads (``file % parts``), and ``mirror_caps`` (one cap per
    mirror root, at most 4 roots and 255 each; empty: off) turns the dynamic root
    choice of ``SGLANG_EXPERT_STREAM_MIRROR`` dynamic on in the entry points that take it
    (``read_rows_sqes``, ``read_rows_drive_load``).
    """
    caps = [int(c) for c in mirror_caps]
    if len(caps) > DRIVE_SLOTS or any(not 0 <= c <= 255 for c in caps):
        raise ValueError(f"mirror_caps {caps}: at most {DRIVE_SLOTS} caps of 0..255 fit the fault word")
    caps_word = len(caps) | sum(c << (8 * (q + 1)) for q, c in enumerate(caps))
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
            root + 1,  # word 19: 0 any root (was pack_workers; the packed path is gone)
            caps_word,  # word 20: the mirror caps (was pack_split)
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
            two_span_rows,  # word 32
            suffix_delay_ns,  # word 33
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
    layout: BuildKey = EXL3_BUILD,
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


# The tier's drive load (``host/drive_load.h``, ``DriveLoad::snapshot``): a header of
# ``elapsed_ns`` (since the load's origin), ``clock_reads`` and ``roots``, then these
# fields per root for ``DRIVE_SLOTS`` roots. Reads and bytes "in flight" are
# instantaneous; ``*_bytes`` landed and ``*_busy_ns`` accumulate over the load's life.
# ``overlap_ns`` is the time a root had demand and speculative reads in flight at once;
# ``moved_from``/``moved_to`` count the sub-reads the dynamic root choice
# (``SGLANG_EXPERT_STREAM_MIRROR`` dynamic) sent away from and to the root.
# ``DRIVE_TAIL_FIELDS`` follow every root's ``DRIVE_FIELDS``, field-major, and decode
# into the same per-root dict: ``fill_bytes`` are the bytes of the tier's fills (prefill
# fills, hot-cache promotions, full layers), already counted in ``demand_bytes``.
DRIVE_SLOTS = 4  # kMaxDrives
DRIVE_FIELDS = (
    "demand_reads",
    "spec_reads",
    "demand_inflight_bytes",
    "spec_inflight_bytes",
    "demand_bytes",
    "spec_bytes",
    "demand_busy_ns",
    "spec_busy_ns",
    "overlap_ns",
    "moved_from",
    "moved_to",
)
DRIVE_TAIL_FIELDS = ("fill_bytes",)
DRIVE_TAIL_START = 3 + DRIVE_SLOTS * len(DRIVE_FIELDS)  # kDriveTailStart
DRIVE_LOAD_WORDS = DRIVE_TAIL_START + DRIVE_SLOTS * len(DRIVE_TAIL_FIELDS)


def decode_drive_load(words) -> dict:
    """Decode ``DRIVE_LOAD_WORDS`` int64 words into ``{"elapsed_ns", "clock_reads",
    "roots": [one dict of DRIVE_FIELDS then DRIVE_TAIL_FIELDS per root]}``."""
    words = [int(w) for w in words]
    roots = words[2]
    fields = len(DRIVE_FIELDS)
    return {
        "elapsed_ns": words[0],
        "clock_reads": words[1],
        "roots": [
            {
                **dict(zip(DRIVE_FIELDS, words[3 + q * fields : 3 + (q + 1) * fields])),
                **{
                    name: words[DRIVE_TAIL_START + k * DRIVE_SLOTS + q]
                    for k, name in enumerate(DRIVE_TAIL_FIELDS)
                },
            }
            for q in range(roots)
        ],
    }


def read_rows_drive_load(
    tables,
    row: int,
    experts,
    slots,
    then_experts=(),
    then_slots=(),
    *,
    a_spec: bool = False,
    b_spec: bool = False,
    nested: bool = False,
    step: int = BOUNCE_ROWS,
    preload: Sequence[tuple[int, int]] = (),
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
    **faults,
) -> dict:
    """Test only: two C++ readers over ``tables`` count into one drive load, as the
    tier's NUMA groups do.

    Reader A reads ``experts`` into ``slots`` with the fault keywords of
    ``read_rows_with_fault`` (``piece_stream`` applies to both readers); reader B then
    reads ``then_experts`` into ``then_slots`` with none. ``a_spec``/``b_spec`` make
    that reader's read speculative. ``nested`` runs B's read inside A's, at the first
    turn that finds A with sub-reads in flight. ``mirror_caps`` (a fault keyword) turns
    the dynamic root choice on for both readers; ``preload`` (``(reads, bytes)`` per
    root) is counted as another reader's demand reads in flight while A reads, and
    taken back before ``"after_a"``.

    Returns ``{"a", "b"}`` (the results: 1, 0, -1; A's is -2 when it raised, B's -9
    when it did not run), ``"a_raised"``, ``"a_in_flight"`` (A's sub-reads in flight
    when B ran nested) and the decoded drive load ``"nested"`` (after the nested B),
    ``"after_a"`` and ``"after_b"``. Both builds; production refuses a fault.
    """
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    then_ids, then_at = _checked_rows(tables, row, then_experts, then_slots)
    fault = _fault_tensor(**faults)
    if len(preload) > DRIVE_SLOTS:
        raise ValueError(f"preload names {len(preload)} roots, the drive load has {DRIVE_SLOTS}")
    pre = torch.zeros(2 * DRIVE_SLOTS, dtype=torch.int64)
    for q, (reads, nbytes) in enumerate(preload):
        pre[q], pre[DRIVE_SLOTS + q] = int(reads), int(nbytes)
    out = torch.zeros(4 + 3 * DRIVE_LOAD_WORDS, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_rows_drive_load(
        *_table_args(tables),
        row,
        expert_ids,
        slot_ids,
        then_ids,
        then_at,
        int(step),
        fault,
        int(a_spec) | int(b_spec) << 1 | int(nested) << 2,
        pre,
        out,
    )
    words = out.tolist()
    snaps = [
        words[4 + k * DRIVE_LOAD_WORDS : 4 + (k + 1) * DRIVE_LOAD_WORDS]
        for k in range(3)
    ]
    return {
        "a": words[0],
        "b": words[1],
        "a_in_flight": words[2],
        "a_raised": bool(words[3]),
        "nested": decode_drive_load(snaps[0]) if nested else None,
        "after_a": decode_drive_load(snaps[1]),
        "after_b": decode_drive_load(snaps[2]),
    }


def read_rows_idle(
    tables,
    row: int,
    experts,
    slots,
    *,
    preload: Sequence[int] = (),
    follow: bool = False,
    boost: bool = False,
    give_up_after: int = 0,
    deadline_s: float = 1.0,
    spread: bool = False,
    pieces: int = 1,
    reads: int = 1,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
    **faults,
) -> dict:
    """Test only: one speculative C++ read of ``experts`` into ``slots`` under the idle-drive root policy
    (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE, ``IdleRoots``), with the fault keywords of ``read_rows_with_fault`` (piece
    streaming always on).

    ``preload`` (demand reads per root) stands for other readers' demand in flight for the whole read. ``follow``: a
    demand read arrives on the first piece's root once it is in flight, and stays. ``boost`` boosts the read from its
    start. ``give_up_after`` > 0 gives up once that many pieces were issued. ``deadline_s``: the longest wait for an
    idle root. ``spread`` and ``pieces``: SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_SPREAD and
    SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES. ``reads`` repeats the
    read on the one reader, as its speculative reader reads row after row (stopping at the first that does not return
    1); the counts and SQEs add up over them.

    Returns ``{"result"`` (1, 0; -2 when it raised), ``"abandoned"``, ``"boosted"``, ``"deferrals"``, ``"moves"``,
    ``"pieces"``, ``"widest"`` (the most of its sub-reads in flight when one was issued), ``"share_after"`` (its
    sub-reads still counted after it returned), ``"sqes"`` (``(root, offset, length)`` per SQE, in order) and
    ``"load"`` (the decoded drive load after the read, the preload and the follower taken back)``}``. Instrumented
    build only.
    """
    _refuse_test_only("read_rows_idle", variant)
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    fault = _fault_tensor(piece_stream=True, **faults)
    if len(preload) > DRIVE_SLOTS:
        raise ValueError(f"preload names {len(preload)} roots, the drive load has {DRIVE_SLOTS}")
    pre = torch.zeros(2 * DRIVE_SLOTS, dtype=torch.int64)
    for q, reads in enumerate(preload):
        pre[q] = int(reads)
    sqes = torch.zeros((4096, 4), dtype=torch.int64)
    out = torch.zeros(10 + DRIVE_LOAD_WORDS, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_rows_idle(
        *_table_args(tables),
        row,
        expert_ids,
        slot_ids,
        fault,
        pre,
        int(bool(follow)),
        int(bool(boost)),
        int(give_up_after),
        int(deadline_s * 1e9),
        int(bool(spread)),
        int(pieces),
        int(reads),
        sqes,
        out,
    )
    words = out.tolist()
    parts = int(tables.extents.shape[2])
    count = min(words[8], sqes.shape[0])
    return {
        "result": words[0],
        "abandoned": bool(words[1]),
        "boosted": bool(words[2]),
        "deferrals": words[3],
        "moves": words[4],
        "pieces": words[5],
        "widest": words[6],
        "share_after": words[7],
        "sqes": [(int(f) % parts, int(o), int(n)) for f, o, n, _ in sqes[:count].tolist()],
        "load": decode_drive_load(words[10:]),
    }


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
    layout: BuildKey = EXL3_BUILD,
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

    Two spans (``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``, piece streaming only):

    - ``two_span_rows`` (a bit per row ordinal) reads those rows in two spans, the
      layout's first stage over every root before its second (``piece_geometry``'s
      ``two_span``). Not a fault.
    - ``suffix_delay_ns`` withholds every second-span sub-read's completions until
      nothing else is in flight and that long after the first was withheld.

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
    span_weights=None,
    layout: BuildKey = EXL3_BUILD,
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
    are as in ``read_rows_with_fault``; ``span_weights`` divides the two-span rows'
    spans (``piece_geometry``). Instrumented build only
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
        _span_tensor(span_weights),
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
    layout: BuildKey = EXL3_BUILD,
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


def piece_source_slot(word: int, generation: int) -> int:
    """The slot a lane's ``PieceSource`` word names for ``generation``, or -1 (S streams from the staging slot).

    The word is generation40 << 24 | (slot + 1), as ``piece_source_word`` in host/piece_geometry.h writes it and
    ``piece_source_slot`` in row_copy_kernels.cuh reads it.
    """
    slot1 = word & 0xFFFFFF
    return slot1 - 1 if (word >> 24) == (generation & ((1 << 40) - 1)) and slot1 != 0 else -1


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
    layout: BuildKey = EXL3_BUILD,
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
    two_span: bool = False,
    span_weights=None,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
) -> Optional[tuple[list[dict], list[dict]]]:
    """Test only: how the C++ reader cuts one expert under piece streaming.

    Returns ``(sub_reads, pieces)`` for expert ``expert`` of streamed row ``row``, or
    None when the reader refuses the row. Sub-reads, in file order, are
    ``{file, offset, length, dest, part, k}``. Pieces (``STAGE_PIECES``) are
    ``{deps, runs}``: ``deps`` is the bitmask of sub-reads the piece depends on and
    ``runs`` holds one ``(dst_lo, dst_hi)`` per segment, in segment destination
    coordinates. ``two_span`` cuts it as a two-stage CPU miss
    (``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``): the first span's sub-reads are ``k`` 0, the
    second's ``k`` 1, each span divided by ``span_weights`` (w13's, w2's) when given.
    """
    segments = int(tables.segments.shape[1])
    subs = torch.zeros((STAGE_PIECES, 6), dtype=torch.int64)
    pieces = torch.zeros((STAGE_PIECES, 1 + 2 * segments), dtype=torch.int64)
    count = int(
        _host_module(layout, variant).expert_stream_piece_geometry(
            *_table_args(tables)[:-1], row, expert, subs, pieces, int(bool(two_span)), _span_tensor(span_weights)
        )
    )
    if count < 0:
        return None
    count &= 0xFF
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


def read_rows_two_span(
    tables,
    row: int,
    experts,
    slots,
    *,
    reference,
    ref_slots,
    step: int = BOUNCE_ROWS,
    span_weights=None,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
    **faults,
) -> dict:
    """Test only: one piece-streamed read of ``experts`` into ``slots``, the rows of ``two_span_rows`` in two spans.

    When the read first reports a row's first span landed, its first-stage bytes are compared with the ``reference``
    slabs (a pointer table shaped like ``tables.slabs``, row o at ``ref_slots[o]``). Returns ``result``, ``reported``
    (rows reported), ``early`` (reported before the row was whole), ``differed`` (rows whose first-stage bytes did not
    match when reported), ``in_flight`` (the reader's sub-reads still counted in flight after the read), ``split`` (the
    second stage's start) and ``drive_load`` (``decode_drive_load`` of the reader's own load). Faults as in
    ``read_rows_with_fault``. Instrumented build only.
    """
    _refuse_test_only("read_rows_two_span", variant)
    expert_ids, slot_ids = _checked_rows(tables, row, experts, slots)
    fault = _fault_tensor(**faults)
    info = torch.zeros(6, dtype=torch.int64)
    drive = torch.zeros(DRIVE_LOAD_WORDS, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_rows_two_span(
        *_table_args(tables),
        row,
        expert_ids,
        slot_ids,
        int(step),
        fault,
        reference,
        _ids(ref_slots),
        info,
        drive,
        _span_tensor(span_weights),
    )
    result, reported, early, differed, in_flight, split = info.tolist()
    return dict(
        result=result, reported=reported, early=early, differed=differed, in_flight=in_flight, split=split,
        drive_load=decode_drive_load(drive.tolist()),
    )


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


# The split model calibration's cell kinds (split_calibration.h kSplitCell*): a CPU hit job, a CPU miss job, a DMA, an
# NVMe read.
SPLIT_CELL_HIT, SPLIT_CELL_MISS, SPLIT_CELL_COPY, SPLIT_CELL_LANDING = 1, 2, 3, 4
# The CPU engine's fit moments per output part (cpu_experts.h kCpuFitMoments), in order.
CPU_FIT_MOMENTS = ("n", "k", "p", "kk", "kp", "pp", "y", "ky", "py")


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
    layout: BuildKey = EXL3_BUILD, variant: Optional[str] = None, lanes: int = 8
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
            f"expert-stream hot bitmap expert count {experts} is outside [1, 65535]"
        )
    return (
        (HOT_HEADER_BYTES + (experts + 7) // 8 + HOT_ALIGNMENT - 1) // HOT_ALIGNMENT
    ) * HOT_ALIGNMENT


def new_hot_page(experts: int, *, pin: bool = True) -> torch.Tensor:
    """Allocate a zeroed hot page: ``HOT_RECORDS`` hot-bitmap records."""
    return torch.zeros(
        HOT_RECORDS * hot_record_bytes(experts), dtype=torch.uint8, pin_memory=pin
    )


# The most of a row's pieces an idle-drive read may have in flight (RamPrefetchConfig::kMaxIdlePieces).
MAX_IDLE_PIECES = 4


# The GPU scorer's candidate page (csrc/moe/expert_stream/spec_candidates.h): CAND_RECORDS slots of CAND_STRIDE bytes,
# slot (seq - 1) % CAND_RECORDS; u32 seq, u16 count, u16 flags, then CAND_MAX entries (u16 expert, u8 rank, pad, f32).
CAND_MAX = 8
CAND_RECORDS = 16
CAND_STRIDE = 128
CAND_ENTRIES = 8
CAND_ENTRY_BYTES = 8
CAND_PAGE_BYTES = CAND_RECORDS * CAND_STRIDE
CAND_FLAG_OVERSIZE = 1
# The GPU drip's picks after the candidates, under their own seq word: u32 seq at CAND_DRIP_SEQ, u16 count at
# CAND_DRIP_COUNT, CAND_MAX_DRIPS u16 experts at CAND_DRIP_EXPERTS (SpecCandidates::kCandDrip*, kMaxDrips).
CAND_MAX_DRIPS = 8
CAND_DRIP_SEQ = 72
CAND_DRIP_COUNT = 76
CAND_DRIP_EXPERTS = 80


def new_candidate_page(*, pin: bool) -> torch.Tensor:
    """A zeroed candidate page; pinned (the select kernel writes it through UVA) for a real device."""
    return torch.zeros(CAND_PAGE_BYTES, dtype=torch.uint8, pin_memory=pin)


def candidate_offset(seq: int) -> int:
    """The byte offset of record ``seq``'s slot."""
    return ((int(seq) - 1) % CAND_RECORDS) * CAND_STRIDE


class CandidateRead(NamedTuple):
    status: str  # "ready", "not_yet" (an older record's, or open) or "lapped" (a later record's): ram_tier.h CandRead
    count: int = 0
    flags: int = 0
    picks: tuple = ()  # (expert, rank, margin) per candidate, best margin first


def read_candidates(page: torch.Tensor, seq: int) -> CandidateRead:
    """Record ``seq``'s slot as the host reads it, without the torn-copy check: read it while no kernel writes."""
    off = candidate_offset(seq)
    raw = bytes(page[off : off + CAND_ENTRIES + CAND_MAX * CAND_ENTRY_BYTES].tolist())
    word, count, flags = struct.unpack_from("<IHH", raw)
    seq &= 0xFFFFFFFF
    if word != seq:
        following = (seq + 1) & 0xFFFFFFFF or 1
        lapped = word != 0 and ((word - following) & 0xFFFFFFFF) < 0x80000000
        return CandidateRead("lapped" if lapped else "not_yet")
    picks = tuple(
        struct.unpack_from("<HBxf", raw, CAND_ENTRIES + i * CAND_ENTRY_BYTES) for i in range(min(count, CAND_MAX))
    )
    return CandidateRead("ready", count, flags, picks)


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
    # RAM prefetch.
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
    "spec_late",
    "spec_deferred",
    "spec_abandoned",
    "spec_boosted",
    "spec_moved_root",
    "spec_promoted_early",
    "spec_promote_no_delta",
    "spec_promote_no_room",
    "spec_promote_no_victim",
    "spec_promote_victim_routed",
    "spec_promote_used",
    "spec_promote_reading",
    "spec_publish_backstop",
    "spec_promote_row_busy",
    "spec_pool_served",
    "spec_skip_landed_gpu",
    "spec_skip_landed_cpu",
    "spec_skip_reading_gpu",
    "spec_skip_reading_cpu",
    "spec_scored",
    "spec_score_ns",
    # GPU drip.
    "drip_issued",
    "drip_landed",
    "drip_used",
    "drip_abandoned",
    "drip_no_slot",
    "drip_paused",
    "drip_skipped_bytes",
    "drip_late",
    # Eager admissions' evictions, also counted in "evictions".
    "eager_evictions",
)

# The counters a production host keeps: the shutdown line's served, rows and errors,
# the admission policy's outcomes, the RAM prefetch's outcomes, and the functional
# version, in ``COUNTERS`` order.
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
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
    "spec_late",
    "spec_deferred",
    "spec_abandoned",
    "spec_boosted",
    "spec_moved_root",
    "spec_promoted_early",
    "spec_promote_no_delta",
    "spec_promote_no_room",
    "spec_promote_no_victim",
    "spec_promote_victim_routed",
    "spec_promote_used",
    "spec_promote_reading",
    "spec_publish_backstop",
    "spec_promote_row_busy",
    "spec_pool_served",
    "drip_issued",
    "drip_landed",
    "drip_used",
    "drip_abandoned",
    "drip_no_slot",
    "drip_paused",
    "drip_skipped_bytes",
    "drip_late",
    "eager_evictions",
)
assert CORE_COUNTERS == tuple(sorted(CORE_COUNTERS, key=COUNTERS.index))

# A speculative pool entry's states (host/ram_prefetch.h kPoolEmpty, kPoolReading, kPoolLanded, kPoolSwapped).
SPEC_POOL_STATES = ("empty", "reading", "landed", "swapped")
# The GPU drip's scratch table: a u64 per slot (lease_layout.h kScratchSlots).
DRIP_TABLE_SLOTS = 32
# A GPU drip scratch slot's states (host/gpu_drip.h kFree, kFilling, kPublished, kRetiring, kPublishing, kClearing).
DRIP_STATES = ("free", "filling", "published", "retiring", "publishing", "clearing")


def seqlock_stress(
    seconds: float,
    *,
    layout: BuildKey = EXL3_BUILD,
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
    layout: BuildKey = EXL3_BUILD,
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


def kernel_layer(kernel: int, spec, *, layout: BuildKey = EXL3_BUILD, variant: Optional[str] = None) -> int:
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
    store_first: bool = False,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
) -> tuple[int, str]:
    """One forward of :func:`kernel_layer`'s ``layer`` over ``rows`` token rows (x ``[rows, H]``, slots and weights
    ``[rows, k]``). ``store_first`` is ForwardCall::store_first: an overwrite stores each token's first sum.

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
            int(bool(store_first)),
        )
    )
    # The message is this thread's (a thread-local in the module), read on the same thread.
    return status, (str(module.expert_stream_kernel_error()) if status else "")


def kernel_forward_two_stage(
    layer: int,
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    *,
    threads: int,
    restore: Sequence[tuple[int, int, int]],
    delay_s: float,
    cores: Sequence[int] = (),
    accumulate: bool = False,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
) -> tuple[int, str, dict]:
    """Test only: :func:`kernel_forward` as a two-stage CPU miss runs it (``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``).

    The forward's stage-two wait holds until a helper thread, ``delay_s`` after the call, has copied each ``restore``
    row ``(dst, src, bytes)`` (the second-stage bytes the caller held back) and opened the gate. Returns the status, the
    kernel's refusal and ``{"calls", "waited_ns"}`` of the wait. Instrumented build only.
    """
    _refuse_test_only("kernel_forward_two_stage", variant)
    module = _host_module(layout, variant)
    info = torch.zeros(2, dtype=torch.int64)
    status = int(
        module.expert_stream_kernel_forward_two_stage(
            int(layer),
            x.to(torch.float16).contiguous(),
            slots.to(torch.int32).contiguous(),
            weights.to(torch.float32).contiguous(),
            out,
            int(threads),
            torch.tensor(list(cores), dtype=torch.int64),
            int(bool(accumulate)),
            torch.tensor([list(r) for r in restore], dtype=torch.int64).reshape(-1, 3),
            int(delay_s * 1e9),
            info,
        )
    )
    calls, waited = info.tolist()
    return status, (str(module.expert_stream_kernel_error()) if status else ""), {"calls": calls, "waited_ns": waited}


def kernel_drop(layer: int, *, layout: BuildKey = EXL3_BUILD, variant: Optional[str] = None) -> None:
    """Release :func:`kernel_layer`'s ``layer``. Both builds."""
    _host_module(layout, variant).expert_stream_kernel_drop(int(layer))
    _kernel_layer_keep.pop(int(layer), None)


def pause_ns(*, layout: BuildKey = EXL3_BUILD, variant: Optional[str] = None) -> float:
    """Test only: ns per ``_mm_pause`` on this core.

    That is the RAM-miss service's idle-poll quantum. Instrumented build only.
    """
    _refuse_test_only("pause_ns", variant)
    return float(_host_module(layout, variant).expert_stream_pause_ns())


def score_gate(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    skip: Sequence[bool],
    *,
    top_k: int,
    per_token: int,
    per_layer: int,
    top_k_only: bool = False,
    return_meta: bool = False,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
) -> list:
    """The RAM prefetch's gate scorer (``host/gate_scorer.h``): ``x`` fp16 ``[tokens, >= hidden]`` (columns past
    ``hidden`` are row padding), ``w`` bf16 ``[experts, hidden]``, ``bias`` fp32 ``[experts]``, skipping ``skip``;
    returns the chosen experts, best first, or with ``return_meta`` ``(expert, rank, margin)`` triples. ``top_k_only``
    walks each token's predicted ``top_k`` alone. Both builds."""
    if x.dtype != torch.float16 or w.dtype != torch.bfloat16 or bias.dtype != torch.float32:
        raise ValueError("score_gate takes an fp16 x, a bf16 w and an fp32 bias")
    out = torch.full((int(per_layer),), -1, dtype=torch.int64)
    meta = torch.zeros((int(per_layer), 2), dtype=torch.float32)
    n = int(
        _host_module(layout, variant).expert_stream_score_gate(
            x.contiguous().view(torch.uint8),
            w.contiguous().view(torch.uint8),
            bias.contiguous(),
            torch.tensor([bool(s) for s in skip], dtype=torch.uint8),
            int(top_k),
            int(per_token),
            int(per_layer),
            int(bool(top_k_only)),
            out,
            meta,
        )
    )
    if return_meta:
        return [(int(e), int(r), float(m)) for e, (r, m) in zip(out[:n].tolist(), meta[:n].tolist())]
    return out[:n].tolist()


ROUTERS = {"sqrt_softplus": 0, "sigmoid": 1}  # router.h kName -> router_score's router index


def router_score(
    router: str,
    dots: torch.Tensor,
    biases: torch.Tensor,
    *,
    layout: BuildKey = EXL3_BUILD,
    variant: Optional[str] = None,
) -> torch.Tensor:
    """The host score of router ``router`` (``ROUTERS``, ``csrc/moe/expert_stream/router.h``) for fp32 ``dots`` and
    ``biases`` of one shape ``[n]``. Both builds."""
    if router not in ROUTERS:
        raise ValueError(f"router_score: router is one of {sorted(ROUTERS)}")
    if dots.dtype != torch.float32 or biases.dtype != torch.float32 or dots.shape != biases.shape or dots.dim() != 1:
        raise ValueError("router_score takes fp32 dots and biases of one 1-D shape")
    out = torch.zeros_like(dots)
    _host_module(layout, variant).expert_stream_router_score(
        ROUTERS[router], dots.contiguous(), biases.contiguous(), out
    )
    return out


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
            sys.stderr.write(f"{host.format_name} RAM miss: stopping a host failed: {error!r}\n")


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
        layout: BuildKey = EXL3_BUILD,
        variant: Optional[str] = None,
        lanes: int = 8,
        node_ranges: Optional[Sequence[Sequence[tuple[int, int]]]] = None,
        sq_thread_cpus: Optional[Sequence[int]] = None,
        config: Optional[ExpertStreamConfig] = None,
    ) -> None:
        # NUMA groups: node_ranges[g][row] = (lo, hi), group g's slots of the row; one group owning every slot by
        # default. sq_thread_cpus: each group's SQPOLL core (-1 unpinned); None keeps the uring options' (unpinned).
        # config: the engine config whose C++ options (uring, trace) the module takes; None is stream_config().
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
        self.config = config or stream_config()
        # The host build is chosen once here: every later call goes to this module.
        self.variant = host_variant(self.config) if variant is None else variant
        self._module = _host_module(self._layout, self.variant, self.wire.lanes, self.nodes, self.config)
        self.threaded = False
        # set_cpu_layer's specs' tensors, kept alive while the host's layers read them.
        self._cpu_layer_keep: dict[int, tuple] = {}
        # The GPU drip's candidate page and scratch tensors (enable_gpu_drip, set_drip_scratch), read by address.
        self.drip_candidates: Optional[torch.Tensor] = None
        self.drip_scratch: tuple = ()
        self.drip_table: Optional[torch.Tensor] = None
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
                f"{self.format_name} RAM miss service failed to open (files, io_uring or bounce)"
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
        affinity. Cores 52-55,108-111 are reserved for NVMe completion interrupts. An idle service thread spins with PAUSE,
        or with ``busy_poll`` with no PAUSE (which the C++ side refuses unless each physical core is its service's
        alone), for ``spin_us``, then sleeps 50 us between polls; parked, it sleeps 20 us between checks. -1 never
        sleeps (``SGLANG_EXPERT_STREAM_RAM_MISS_IDLE_SPIN_US`` in a server).
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
                f"{self.format_name} RAM miss: not paused, the copy thread still has a job (unsynchronized stream)"
            )
        if outcome != 1:
            raise RuntimeError(
                f"{self.format_name} RAM miss thread did not pause within {timeout_s} s"
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
                f"{self.format_name} RAM miss: a prefill fill failed before its first {rows} rows landed"
            )
        if outcome != 1:
            raise RuntimeError(
                f"{self.format_name} RAM miss: a prefill fill did not land {rows} rows within {timeout_s} s"
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
        """Return ``(state, expert, stamp)`` per slot; state 0 FREE, 2 READY, 3 STAGING, 4 SPEC (the prefetch pool).

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

    def reserve_spec_pool(self, share: int) -> None:
        """Reserve every row's speculative pool (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH): ``share`` slots per row and NUMA group
        after its staging slots, never mapped, never a victim, never released. Once, after ``reserve_staging``, before
        any slot fills and before the thread starts."""
        self._module.expert_stream_reserve_spec_pool(self.handle, int(share))

    def spec_pool(self, row: int) -> list[dict]:
        """Every pool entry of ``row``, group-major: ``group``, ``slot``, ``state`` (one of ``SPEC_POOL_STATES``) and
        ``expert`` (-1 when empty or swapped). Any time."""
        self._check(row)
        share = int(self._module.expert_stream_spec_share(self.handle))
        out = torch.empty((self.nodes * share, 4), dtype=torch.int64)
        self._module.expert_stream_spec_pool(self.handle, row, out)
        return [
            {"group": g, "slot": slot, "state": SPEC_POOL_STATES[state], "expert": expert}
            for g, slot, state, expert in out.tolist()
        ]

    def enable_ram_prefetch(
        self,
        targets: torch.Tensor,
        gates: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        top_k: int,
        per_token: int,
        per_layer: int,
        cores: Sequence[Sequence[int]],
        top_k_only: bool = False,
        candidates: Optional[torch.Tensor] = None,
        min_margin: Optional[torch.Tensor] = None,
        idle_drive: bool = False,
        idle_deadline_s: float = 0.004,
        idle_spread: bool = False,
        idle_pieces: int = 1,
        promote: bool = False,
        pool_serve: bool = False,
    ) -> None:
        """Enable the RAM prefetch over the pool ``reserve_spec_pool`` took, after ``enable_cpu_experts`` and before
        the thread. ``targets`` int64 ``[layers, 2]``: per source row its target row and gate index, or (-1, -1);
        ``cores`` is each NUMA group's speculative-thread core list (empty: the caller's affinity). CPU scorer:
        ``gates`` bf16 ``[n, experts, hidden]`` and ``bias`` fp32 ``[n, experts]``, host tensors this host keeps alive.
        GPU scorer: ``candidates`` (``new_candidate_page``), which the select kernel writes, and no gates; optional
        ``min_margin`` fp32 ``[layers]``, per target row the least margin a candidate needs to be read.

        ``idle_drive`` (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE): each group reads its pool rows through a reader of its
        own, one piece at a time from a drive no demand reads, abandoning a read that waited ``idle_deadline_s`` for
        one; off, the speculative reads take turns with the demand reads on the group's reader. With it,
        ``idle_spread`` (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_SPREAD) spreads each row's pieces over the idle drives, ties
        rotating from row to row, and ``idle_pieces`` (SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES, 1 to 4) is how many of a
        row's pieces may be in flight at once.

        ``promote`` (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_PROMOTE) maps rows that landed before their target layer's record into
        that layer's map delta, so the device types them as RAM hits.

        ``pool_serve`` (SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_POOL_SERVE) serves a GPU miss whose row landed in the pool by naming
        the pool slot in the lane's PieceSource word, so the stream kernel copies from it, instead of reading it from
        NVMe."""
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        if candidates is not None:
            if gates is not None or bias is not None:
                raise ValueError("the GPU scorer reads the device's gates: pass gates=None and bias=None")
            if (
                candidates.dtype != torch.uint8
                or candidates.device.type != "cpu"
                or not candidates.is_contiguous()
                or candidates.numel() != CAND_PAGE_BYTES
            ):
                raise ValueError(
                    f"candidates must be a contiguous host uint8 tensor of {CAND_PAGE_BYTES} bytes (new_candidate_page)"
                )
            # [0, 1], not [0, 0]: the FFI's contiguity check rejects the strides torch gives a [0, 0] tensor.
            gate_bytes = torch.empty((0, 1), dtype=torch.uint8)
            bias = torch.empty((0, 1), dtype=torch.float32)
            hidden = 0
        else:
            if (
                gates is None
                or gates.dtype != torch.bfloat16
                or gates.dim() != 3
                or gates.device.type != "cpu"
                or not gates.is_contiguous()
            ):
                raise ValueError("gates must be a contiguous host bf16 [n, experts, hidden] tensor")
            if (
                bias is None
                or bias.dtype != torch.float32
                or tuple(bias.shape) != tuple(gates.shape[:2])
                or bias.device.type != "cpu"
            ):
                raise ValueError("bias must be a host fp32 [n, experts] tensor")
            gate_bytes = gates.view(gates.shape[0], -1).view(torch.uint8)
            bias = bias.contiguous()
            hidden = int(gates.shape[2])
        if min_margin is not None:
            if candidates is None:
                raise ValueError("min_margin needs the GPU scorer (candidates)")
            if (
                min_margin.dtype != torch.float32
                or tuple(min_margin.shape) != (targets.shape[0],)
                or min_margin.device.type != "cpu"
                or bool(min_margin.isnan().any())
            ):
                raise ValueError(f"min_margin must be a host fp32 [{targets.shape[0]}] tensor without NaN")
            min_margin = min_margin.contiguous()
        if len(cores) != self.nodes:
            raise ValueError(f"one core list per NUMA group ({self.nodes}), got {len(cores)}")
        idle_deadline_ns = int(idle_deadline_s * 1e9) if idle_drive else 0
        if idle_drive and idle_deadline_ns < 1:
            raise ValueError(f"idle_deadline_s must be positive, got {idle_deadline_s}")
        if not 1 <= idle_pieces <= MAX_IDLE_PIECES:
            raise ValueError(f"idle_pieces must be in [1, {MAX_IDLE_PIECES}], got {idle_pieces}")
        table = torch.full((self.nodes, max(1, max(len(own) for own in cores))), -1, dtype=torch.int64)
        for g, own in enumerate(cores):
            for j, core in enumerate(own):
                check_not_reserved(int(core))
                table[g, j] = int(core)
        self._module.expert_stream_enable_ram_prefetch(
            self.handle,
            targets.to(torch.int64).contiguous(),
            gate_bytes,
            bias,
            table,
            hidden,
            int(top_k),
            int(per_token),
            int(per_layer),
            int(bool(top_k_only)),
            candidates if candidates is not None else torch.empty(0, dtype=torch.uint8),
            min_margin if min_margin is not None else torch.empty(0, dtype=torch.float32),
            idle_deadline_ns,
            int(bool(idle_spread)),
            int(idle_pieces),
            int(bool(promote)),
            int(bool(pool_serve)),
        )
        self.ram_prefetch_tensors = (gates, bias, candidates, min_margin)

    def spec_pump(self, group: int = 0) -> bool:
        """Test only: serve NUMA group ``group``'s next speculative job on the calling thread (no service thread)."""
        _refuse_test_only("spec_pump", self.variant)
        return bool(self._module.expert_stream_spec_pump(self.handle, int(group)))

    def inject_spec(self, delay_s: float = 0.0, fail: bool = False) -> None:
        """Test only: each speculative read sleeps ``delay_s``: first, holding the reader's turn, on the shared reader;
        once its first piece is in flight on an idle-drive reader. With ``fail`` it is reported failed without
        reading."""
        _refuse_test_only("inject_spec", self.variant)
        self._module.expert_stream_inject_spec(self.handle, int(delay_s * 1e9), int(fail))

    def inject_demand_load(self, reads: Sequence[int]) -> None:
        """Test only: adds ``reads[q]`` demand sub-reads to mirror root q's drive load (negative takes them back), as
        another reader's demand in flight an idle-drive read must avoid."""
        _refuse_test_only("inject_demand_load", self.variant)
        if len(reads) > DRIVE_SLOTS:
            raise ValueError(f"reads names {len(reads)} roots, the drive load has {DRIVE_SLOTS}")
        words = torch.zeros(DRIVE_SLOTS, dtype=torch.int64)
        for q, n in enumerate(reads):
            words[q] = int(n)
        self._module.expert_stream_inject_demand_load(self.handle, words)

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
        (``SGLANG_EXPERT_STREAM_RAM_MISS_TIMEOUT_MS`` in a server). The idle copy thread polls for
        ``spin_us``, then sleeps on its doorbell until the next submit; -1 never sleeps
        (``SGLANG_EXPERT_STREAM_COPY_IDLE_SPIN_US`` in a server). ``cpus`` is its core
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

    def enable_gpu_drip(
        self,
        slots: int,
        *,
        picks: int,
        chunk_bytes: int,
        targets: Optional[Sequence[int]] = None,
        candidates: Optional[torch.Tensor] = None,
    ) -> None:
        """The GPU drip (host/gpu_drip.h): the copy thread copies up to ``picks`` RAM-resident experts per scored record
        into ``slots`` VRAM scratch slots, ``chunk_bytes`` per copy-engine mark, while no demand copy is pending.
        ``targets`` (per source row its target row, -1 none) and ``candidates`` (the GPU scorer's page) feed it the
        scorer's drip picks; without them only ``drip_offer`` does. After the copy engine, before the service thread
        starts; ``set_drip_scratch`` before any copy table."""
        if candidates is not None and targets is None:
            raise ValueError("the candidate page needs the targets")
        if candidates is not None and (
            candidates.dtype != torch.uint8 or candidates.device.type != "cpu" or not candidates.is_contiguous()
        ):
            raise ValueError("candidates must be a contiguous host uint8 tensor (new_candidate_page)")
        self._module.expert_stream_enable_gpu_drip(
            self.handle,
            int(slots),
            int(picks),
            int(chunk_bytes),
            torch.tensor([int(t) for t in targets] if targets is not None else [], dtype=torch.int64),
            candidates if candidates is not None else torch.empty(0, dtype=torch.uint8),
        )
        # The page is read through its address: keep it alive with the host.
        self.drip_candidates = candidates

    def set_drip_scratch(self, scratch: Sequence[torch.Tensor], table: torch.Tensor) -> None:
        """The drip's scratch tensors, one per copy-table entry in its order, each ``[slots, ...]`` rows of that
        entry's row bytes, and its scratch table: int64 [DRIP_TABLE_SLOTS], zeroed, which the copy stream writes and
        the post reads (device memory on a GPU, ``ExpertStreamDevice.enable_gpu_drip``'s). Kept alive here."""
        if table.dtype != torch.int64 or table.numel() != DRIP_TABLE_SLOTS or not table.is_contiguous():
            raise ValueError(f"the drip's scratch table is a contiguous int64 [{DRIP_TABLE_SLOTS}] tensor")
        entries = torch.tensor(
            [[int(t.data_ptr()), t.numel() // t.shape[0] * t.element_size()] for t in scratch], dtype=torch.int64
        ).reshape(-1, 2)
        self._module.expert_stream_set_drip_scratch(self.handle, entries, int(table.data_ptr()))
        self.drip_scratch = tuple(scratch)
        self.drip_table = table

    def drip_offer(self, row: int, experts: Sequence[int]) -> None:
        """Test only: offer ``experts`` to the drip for ``row`` as a scored record's picks. Instrumented build only."""
        _refuse_test_only("drip_offer", self.variant)
        self._check(row)
        self._module.expert_stream_drip_offer(self.handle, int(row), torch.tensor(list(experts), dtype=torch.int64))

    def drip_hold(self, on: bool = True) -> None:
        """Test only: while held the drip issues no copy, so admitted drips stay unlanded. Instrumented build only."""
        _refuse_test_only("drip_hold", self.variant)
        self._module.expert_stream_drip_hold(self.handle, int(bool(on)))

    def drip_slots(self) -> list[dict]:
        """Test only: each scratch slot's {state, row, expert, ram_slot}, read racily (state: DRIP_STATES)."""
        out = torch.zeros((expert_lease_block.WireLayout.scratch_slots, 4), dtype=torch.int64)
        self._module.expert_stream_drip_view(self.handle, out)
        return [
            {"state": DRIP_STATES[s], "row": r, "expert": e, "ram_slot": m} for s, r, e, m in out.tolist()
        ]

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
        keep_warm_us: int = 0,
        two_stage: bool = False,
        two_stage_weights=None,
        store_out: bool = False,
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
        references to both. The thread's team never sleeps: its workers run the kernel's
        register work for ``keep_warm_us`` after each job and PAUSE after that, until the
        next submit. With ``[rows, 2 * nodes, tokens, hidden]`` rows a record's CPU job runs
        one forward of its tokens from the row's token table. ``two_stage``
        (``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``) reads each CPU miss's w13 before its w2 and starts
        its job once w13 has landed: the gate/up GEMVs run while w2 is still being read.
        ``two_stage_weights`` (w13's, w2's, each one weight per mirror root or None for the
        mirror's shares) divides those two spans over the roots. ``store_out``
        (``SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT``) leaves a job's overwrite of its token rows to
        the kernel, which stores each token's first sum, instead of zeroing the rows first.
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
                or (out_rows.dim() in (3, 4) and out_rows.shape[1] == 2 * self.nodes)
            )
        ):
            raise ValueError(
                "out_rows must be a contiguous host float32 [rows, hidden] at one node, [rows, 2 * nodes, hidden], or "
                "[rows, 2 * nodes, tokens, hidden] for multi-token rows"
            )
        parts = 1 if out_rows.dim() == 2 else 2
        tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
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
            int(tokens),
            int(threads),
            int(keep_warm_us * 1e3),
            int(bool(two_stage)),
            int(bool(store_out)),
            *(
                torch.tensor(list(w) if w is not None else [], dtype=torch.float64)
                for w in (two_stage_weights or (None, None))
            ),
        )
        self.cpu_rows = (x_rows, out_rows)

    def draft_source(self, areas, *, fatal_wait_s: float, group: int = 0):
        """The DSpark draft channel over ``areas`` (a ``DraftCpuAreas``) as group ``group``'s CPU expert engine's second
        job source: a ``SharedDraftHost``; set its stages' layers, then ``start``. After enable_cpu_experts."""
        from sglang.kernels.ops.moe.dspark_draft_cpu import SharedDraftHost

        return SharedDraftHost(self, areas, group=group, fatal_wait_s=fatal_wait_s)

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
        """Return the jobs and lanes group ``group``'s CPU expert thread computed, and its forward ns.

        With two-stage CPU misses, also its staged jobs, those whose w2 had not landed when their gate/up GEMVs were
        done, and the ns they waited for it.
        """
        out = torch.zeros(6, dtype=torch.int64)
        self._module.expert_stream_cpu_stats(self.handle, int(group), out)
        jobs, lanes, ns, staged, waits, wait_ns = out.tolist()
        return {
            "jobs": jobs, "lanes": lanes, "forward_ns": ns, "staged_jobs": staged, "stage_two_waits": waits,
            "stage_two_wait_ns": wait_ns,
        }

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
        lanes: Optional[int] = None,
        tokens: int = 1,
    ) -> torch.Tensor:
        """Run the CPU split's startup calibration on ``row``, on group ``group``'s engine and RAM slots.

        Returns float64 ``calibration_shape(wire.lanes)`` mean ms: row 0 is ``cpu[k]``, row 1 ``link[m]`` and
        row ``1 + n`` ``both[n][k]`` (k <= n). The caller owns the tier (paused, or no
        thread). ``device`` -1 copies with the test backend; ``scratch`` holds
        one expert per lane (``wire.lanes`` of them) on that device. ``lanes`` caps the lanes measured (default the
        wire's); cells past it stay 0, and ``scratch`` needs that many experts. ``tokens`` above 1 times per-token CPU jobs
        of that many rows with every token routing every lane, the job a verify's record runs (at most the rows' tokens).
        Raises RuntimeError on failure.
        """
        out = torch.zeros(calibration_shape(self.wire.lanes), dtype=torch.float64)
        self._module.expert_stream_calibrate_cpu_split(
            self.handle,
            int(group),
            int(row),
            int(device),
            int(reps),
            int(lanes or self.wire.lanes),
            int(tokens),
            scratch.data_ptr(),
            scratch.numel() * scratch.element_size(),
            int(timeout_s * 1e9),
            out,
        )
        return out

    def cpu_fit_moments(self, group: int = 0) -> list[list[int]]:
        """Group ``group``'s CPU fit moments since start (cpu_experts.h fit_moments), hit jobs then miss jobs, each
        ``CPU_FIT_MOMENTS`` sums over its unstaged target jobs: jobs, k, pairs, k*k, k*pairs, pairs*pairs, ns, k*ns,
        pairs*ns."""
        out = torch.zeros((2, len(CPU_FIT_MOMENTS)), dtype=torch.int64)
        self._module.expert_stream_cpu_fit_moments(self.handle, int(group), out)
        return out.tolist()

    def calibrate_split_model(
        self,
        rows: Sequence[int],
        cells: Sequence[Sequence[int]],
        *,
        device: int,
        reps: int,
        scratch: Optional[torch.Tensor],
        timeout_s: float = 1.0,
        group: int = 0,
    ) -> list[float]:
        """Run the split model's startup cells on group ``group`` (split_calibration.h): each cell ``(kind, a, b)`` of
        ``SPLIT_CELL_*``, mean ms over ``reps`` runs after a warm-up, rotating over ``rows`` (registered CPU rows) and
        the group's slots. ``scratch`` holds the widest copy cell's experts on ``device`` (-1: the test backend), or is
        None without copy cells. The caller owns the tier and has not armed the copy engine. Raises RuntimeError on
        failure."""
        out = torch.zeros(len(cells), dtype=torch.float64)
        self._module.expert_stream_calibrate_split_model(
            self.handle,
            int(group),
            torch.tensor(list(rows), dtype=torch.int64),
            torch.tensor([list(c) for c in cells], dtype=torch.int64).reshape(-1, 3),
            int(device),
            int(reps),
            0 if scratch is None else scratch.data_ptr(),
            0 if scratch is None else scratch.numel() * scratch.element_size(),
            int(timeout_s * 1e9),
            out,
        )
        return out.tolist()

    def test_kernel_address(self, ns_per_expert: int = 0, *, fail: int = 0, zero: bool = False) -> int:
        """Test only: the instr build's fake ``CpuExpertKernel`` (``FakeKernel``, ffi_test_exports.h), reset.

        A forward spins ``ns_per_expert`` per expert and writes ``out[j] = (out[j] if accumulate else j) + sum(w * (s +
        1))`` (``zero``: a zero partial); ``fail`` nonzero makes every forward throw. Instrumented build only.
        """
        _refuse_test_only("test_kernel_address", self.variant)
        return int(self._module.expert_stream_test_kernel_address(int(ns_per_expert), int(fail), int(bool(zero))))

    def test_kernel_calls(self) -> list[dict]:
        """Test only: the fake kernel's forwards since :meth:`test_kernel_address`, one per row, in order.

        ``staged`` says the forward was a two-stage one; over a six-slab layer ``w13`` and ``w2`` are the byte sums of
        its live slots' w13 slabs at its start and w2 slabs after its stage-two wait (else 0).
        """
        _refuse_test_only("test_kernel_calls", self.variant)
        lanes = self.wire.lanes
        width = 9 + 2 * lanes
        count = int(self._module.expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
        out = torch.zeros((count, width), dtype=torch.float64)
        self._module.expert_stream_test_kernel_calls(out)
        calls = []
        for row in out.tolist()[:count]:
            k = int(row[4])
            calls.append({
                "core": int(row[0]), "affinity": int(row[1]), "threads": int(row[2]), "accumulate": bool(row[3]),
                "slots": [int(s) for s in row[5 : 5 + k]], "weights": row[5 + lanes : 5 + lanes + k],
                "capacity": int(row[5 + 2 * lanes]), "staged": bool(row[6 + 2 * lanes]),
                "w13": row[7 + 2 * lanes], "w2": row[8 + 2 * lanes],
            })
        return calls

    def test_kernel_hold(self, core: int, on: bool = True) -> None:
        """Test only: hold (or release) the fake forwards whose worker-0 core is ``core``."""
        _refuse_test_only("test_kernel_hold", self.variant)
        self._module.expert_stream_test_kernel_hold(int(core), int(bool(on)))

    def test_kernel_max_rows(self, rows: int) -> None:
        """Test only: the fake kernel's max_rows (restored by :meth:`test_kernel_address`); call it before the engine
        is enabled, whose validation reads it."""
        _refuse_test_only("test_kernel_max_rows", self.variant)
        self._module.expert_stream_test_kernel_max_rows(int(rows))

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

    def spec_place(self, row: int, expert: int) -> int:
        """Test only: land ``expert``'s row of ``row`` in an empty pool entry of its home group, as a speculative read
        does, and return the slot. Paused or pumping only; nothing is mapped."""
        _refuse_test_only("spec_place", self.variant)
        self._check(row, expert)
        return int(self._module.expert_stream_spec_place(self.handle, row, expert))

    def hold_row(self, row: int, group: int, hold: bool = True) -> None:
        """Test only: a helper thread holds ``group``'s row lock of ``row`` until a call with ``hold=False``, so a
        promotion claim finds the row busy. Never serve ``row`` while it is held."""
        _refuse_test_only("hold_row", self.variant)
        self._module.expert_stream_hold_row(self.handle, int(row), int(group), int(hold))

    @property
    def format_name(self) -> str:
        """The row format's name, the prefix of this host's messages (``<format> RAM miss: ``)."""
        return self._layout[0]

    def piece_runs(self) -> torch.Tensor:
        """Return the stream kernel's piece table.

        The table is int32 ``[layers, experts, STAGE_PIECES, segments, 2]``; each run is
        a ``(lo, hi)`` byte range of its segment's name row, cut by the reader's own row
        geometry. Raises RuntimeError if a (row, expert) cannot be cut into pieces.
        """
        tables = self.tables
        runs = torch.zeros(
            (self.layers, self.experts, STAGE_PIECES, int(tables.segments.shape[1]), 2),
            dtype=torch.int32,
        )
        refused = int(
            self._module.expert_stream_piece_runs(*_table_args(tables)[:-1], runs)
        )
        if refused:
            # A refused row's runs are empty: the stream kernel would admit a READY hit
            # of it, copy nothing and commit.
            raise RuntimeError(
                f"{self.format_name} RAM miss: piece streaming cannot cut {refused} (row, expert) rows into pieces"
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

    def set_mirror_caps(self, caps: Sequence[int]) -> None:
        """``SGLANG_EXPERT_STREAM_MIRROR`` dynamic: every NUMA group's reader picks each sub-read's mirror root, skipping
        roots with ``caps[q]`` sub-reads in flight (demand and speculative, every group) and preferring the fewest bytes
        in flight; with every root at its cap a sub-read keeps its own root. One positive cap per mirror root; empty
        turns it off. Refused when the tables' files are not a full mirror set. Paused, or before the thread starts."""
        self._module.expert_stream_set_mirror_caps(self.handle, torch.tensor([int(c) for c in caps], dtype=torch.int64))

    def drive_load(self) -> dict:
        """The tier's per-root drive load (``decode_drive_load``), shared by every NUMA
        group's reader. Callable while the service runs: relaxed reads, exact once every
        reader is idle (as at ``stop``)."""
        out = torch.zeros(DRIVE_LOAD_WORDS, dtype=torch.int64)
        self._module.expert_stream_drive_load(self.handle, out)
        return decode_drive_load(out.tolist())

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
                    f"{self.format_name} RAM miss thread counters "
                    + json.dumps(self.counters())
                    + "\n"
                )
                if self.nodes > 1:
                    for group in range(self.nodes):
                        sys.stderr.write(
                            f"{self.format_name} RAM miss group {group} counters "
                            + json.dumps(self.group_counters(group))
                            + "\n"
                        )
                # Per root: reads in flight (0 here), bytes landed and busy/overlap time by kind
                # (analysis/dsv41-drive/dspark/drive_busy.py drive-load reads it).
                sys.stderr.write(f"{self.format_name} RAM miss drive load " + json.dumps(self.drive_load()) + "\n")
                if getattr(self, "cpu_rows", None) is not None:
                    # Per group, the CPU expert thread's jobs, with the two-stage misses' waits for w2.
                    stats = [self.cpu_stats(group) for group in range(self.nodes)]
                    sys.stderr.write(f"{self.format_name} RAM miss cpu stats " + json.dumps(stats) + "\n")
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
    "expert_stream_calibration_pull": "calibration_pull",
}
_SPEC_METHODS = {
    "expert_stream_spec_score": "score",
    "expert_stream_spec_select": "select",
    "expert_stream_spec_drip": "drip",
}
_RESIDENCY_METHODS = {
    "expert_stream_residency_snapshot": "snapshot",
    "expert_stream_route_labels": "labels",
}
# The select kernel's bounds (spec_score.cuh kSpecDepth, kSelectMaxExperts).
SPEC_DEPTH = 12
SPEC_SELECT_MAX_EXPERTS = 1024


def _device_wrappers(layout: BuildKey = EXL3_BUILD) -> list[tuple[str, str]]:
    """Return the (FFI name, C++ symbol) pairs the device module exports."""
    build = _build_for(layout, device_jit_key(layout))
    device_layout = build.device_layout
    return [
        (name, f"LeaseProtocolKernel::{method}")
        for name, method in _LEASE_METHODS.items()
    ] + [
        (name, f"RowCopyKernel<{device_layout}>::{method}")
        for name, method in _ROW_COPY_METHODS.items()
    ] + [
        (name, f"{build.spec_score_kernel}::{method}")
        for name, method in _SPEC_METHODS.items()
    ] + [
        (name, f"ResidencySnapshotKernel::{method}")
        for name, method in _RESIDENCY_METHODS.items()
    ]


def device_stamps_enabled(config: Optional[ExpertStreamConfig] = None) -> bool:
    """Whether this process loads the stamped device module (device_stamps.cuh): with a job trace
    (``SGLANG_EXPERT_STREAM_TRACE``'s ``job.prefix``), as host_variant picks the instrumented host build."""
    return bool((config or stream_config()).trace.job.prefix)


# device_stamps.cuh's ring: kStampRing rows of kStampWords int64, the seq then one %globaltimer ns per point.
DEVICE_STAMP_RING = 65536
DEVICE_STAMP_WORDS = 8
DEVICE_STAMP_POINTS = ("post_start", "post_end", "stream_start", "stream_end", "wait_start", "wait_end")


def _device_module(
    layout: BuildKey = EXL3_BUILD, lanes: int = 8, nodes: int = 1, config: Optional[ExpertStreamConfig] = None
) -> Module:
    """Return the cached device module for ``layout``, ``lanes`` and ``nodes``; ``config`` (default
    ``stream_config()``) says whether it is the stamped build."""
    wire = expert_lease_block.wire_layout(lanes, nodes)
    return _device_module_cached(layout, wire.lanes, wire.nodes, device_stamps_enabled(config))


@cache_once
def _device_module_cached(layout: BuildKey, lanes: int, nodes: int, stamps: bool) -> Module:
    key = device_jit_key(layout, lanes, nodes)
    build = _build_for(layout, key)
    wrappers = _device_wrappers(layout)
    if stamps:
        wrappers = wrappers + [
            ("expert_stream_set_device_stamps", f"RowCopyKernel<{build.device_layout}>::set_device_stamps")
        ]
    return load_jit(
        key,
        *(["stamps"] if stamps else []),
        cuda_files=[build.device_source],
        cuda_wrappers=wrappers,
        extra_cuda_cflags=[
            *build.defines,
            *(["-DEXPERT_STREAM_DEVICE_STAMPS"] if stamps else []),
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
    )


def device_module_with_hooks(
    defines: Sequence[str], layout: BuildKey = EXL3_BUILD, lanes: int = 8, nodes: int = 1
) -> Module:
    """Test only: build the device kernels with the ``EXPERT_STREAM_TEST_*`` hooks on.

    ``defines`` are ``NAME`` or ``NAME=value`` entries. The result is a module of its
    own; production builds with none, so its kernels carry no test knob.
    """
    if not defines or not all(d.startswith("EXPERT_STREAM_TEST_") for d in defines):
        raise ValueError(f"not a set of EXPERT_STREAM_TEST_* hooks: {defines}")
    wire = expert_lease_block.wire_layout(lanes, nodes)
    lanes, nodes = wire.lanes, wire.nodes
    key = device_jit_key(layout, lanes, nodes)
    build = _build_for(layout, key)
    return load_jit(
        key,
        "test",
        cuda_files=[build.device_source],
        cuda_wrappers=_device_wrappers(layout),
        extra_cuda_cflags=[
            *build.defines,
            *(f"-D{d}" for d in defines),
            f"-DSGLANG_EXPERT_STREAM_LANES={lanes}",
            f"-DSGLANG_EXPERT_STREAM_NODES={nodes}",
        ],
    )


def run_spec_score(x, w, bias, scores, *, module=None) -> None:
    """The GPU scorer's score kernel (spec_score.cuh): sqrt(softplus(w @ x_t)) + bias for each row of ``x`` into the
    same row of ``scores``. ``x`` bf16 [tokens, hidden], ``w`` bf16 or fp32 [experts, hidden], ``bias`` fp32
    [experts], ``scores`` fp32 [tokens_max, experts], one CUDA device. ``module``: the device module (default: the
    8-lane, one-node build)."""
    (module or _device_module()).expert_stream_spec_score(x, w, bias, scores)


def run_spec_select(
    scores, tokens, *, top_k, per_token, top_k_only, hot_slots, hot_capacity, ram_slot, target, state, candidates,
    routes=None, route_top_k=0, module=None,
) -> None:
    """The GPU scorer's select kernel: ranks ``scores``' first ``tokens`` rows past ``hot_slots[:hot_capacity]`` and
    the experts ``ram_slot[target]`` maps, and publishes up to CAND_MAX candidates to the slot of ``state``'s posted
    seq in the pinned ``candidates`` page; count 0 when ``tokens`` is outside 1..scores' rows. With ``routes`` (int64
    flat, ``route_top_k`` per token) it ranks only the rows through the last with a route >= 0: a verify's dead rows
    (layers/moe/verify_dead_rows.py) take no picks."""
    (module or _device_module()).expert_stream_spec_select(
        scores,
        int(tokens),
        int(top_k),
        int(per_token),
        int(bool(top_k_only)),
        hot_slots,
        int(hot_capacity),
        ram_slot,
        int(target),
        state,
        int(candidates.data_ptr()),
        routes,
        int(route_top_k),
    )


def run_spec_drip(
    scores, rows, *, hot_slots, hot_capacity, ram_slot, target, picks, state, candidates, module=None
) -> None:
    """The GPU drip's select kernel (spec_score.cuh): the ``picks`` experts of ``ram_slot[target]`` mapped in RAM
    and not in ``hot_slots[:hot_capacity]``, by their best score over ``scores``' first ``rows`` rows (fp32
    [rows_max, experts]), published under their own seq after the candidates of ``state``'s posted record in the
    pinned ``candidates`` page; count 0 when ``rows`` is 0."""
    (module or _device_module()).expert_stream_spec_drip(
        scores,
        int(rows),
        hot_slots,
        int(hot_capacity),
        ram_slot,
        int(target),
        int(picks),
        state,
        int(candidates.data_ptr()),
    )


def read_drip_picks(page: torch.Tensor, seq: int) -> Optional[list[int]]:
    """Record ``seq``'s drip picks as the host reads them, or None while their seq word names another record; read
    while no kernel writes."""
    off = candidate_offset(seq)
    raw = bytes(page[off + CAND_DRIP_SEQ : off + CAND_DRIP_EXPERTS + 2 * CAND_MAX_DRIPS].tolist())
    word, count = struct.unpack_from("<IH", raw)
    if word != seq & 0xFFFFFFFF:
        return None
    return list(struct.unpack_from(f"<{min(count, CAND_MAX_DRIPS)}H", raw, CAND_DRIP_EXPERTS - CAND_DRIP_SEQ))


def run_residency_snapshot(ram_slot, targets, valid, out_masks, out_valid, out_stamp, *, module=None) -> None:
    """The residency capture's snapshot kernel (residency_snapshot.cuh; record layout in residency_capture_ref):
    ``ram_slot`` int32 [layers, experts <= 384], the horizons' precomputed ``targets`` int32 [4] and ``valid`` uint8
    [4], into ``out_masks`` int64 [4, 6], ``out_valid`` uint8 [4] and a %globaltimer ``out_stamp`` int64 [1], one CUDA
    device. ``module``: the device module (default: the 8-lane, one-node build)."""
    device = ram_slot.device
    for name, (tensor, dtype, shape) in {
        "ram_slot": (ram_slot, torch.int32, None),
        "targets": (targets, torch.int32, (4,)),
        "valid": (valid, torch.uint8, (4,)),
        "out_masks": (out_masks, torch.int64, (4, 6)),
        "out_valid": (out_valid, torch.uint8, (4,)),
        "out_stamp": (out_stamp, torch.int64, (1,)),
    }.items():
        if (
            tensor.dtype != dtype
            or tensor.device != device
            or device.type != "cuda"
            or not tensor.is_contiguous()
            or (shape is not None and tuple(tensor.shape) != shape)
        ):
            raise ValueError(
                f"{name} must be a contiguous {dtype} tensor of shape {shape or '[layers, experts]'} on {device}"
            )
    if ram_slot.dim() != 2 or not 1 <= ram_slot.shape[1] <= 384:
        raise ValueError(f"ram_slot must be [layers, 1..384 experts], got {tuple(ram_slot.shape)}")
    (module or _device_module()).expert_stream_residency_snapshot(
        ram_slot, targets, valid, out_masks, out_valid, out_stamp
    )


def run_route_labels(routes, planned, count, lane_kind, out_kinds, out_stamp, *, module=None) -> None:
    """The residency capture's label kernel (residency_snapshot.cuh; record layout in residency_capture_ref): each of
    ``routes`` int64 [width] gets the ``lane_kind`` int32 [lanes] of its lane among the first ``count`` int32 [1] of
    ``planned`` int64 [lanes], else KIND_VRAM_HOT, into ``out_kinds`` uint8 [width], with a %globaltimer ``out_stamp``
    int64 [1], one CUDA device. ``module``: the device module (default: the 8-lane, one-node build)."""
    device = routes.device
    width, lanes = routes.numel(), planned.numel()
    for name, (tensor, dtype, numel) in {
        "routes": (routes, torch.int64, width),
        "planned": (planned, torch.int64, lanes),
        "count": (count, torch.int32, 1),
        "lane_kind": (lane_kind, torch.int32, lanes),
        "out_kinds": (out_kinds, torch.uint8, width),
        "out_stamp": (out_stamp, torch.int64, 1),
    }.items():
        if (
            tensor.dtype != dtype
            or tensor.device != device
            or device.type != "cuda"
            or not tensor.is_contiguous()
            or tensor.dim() != 1
            or tensor.numel() != numel
            or numel < 1
        ):
            raise ValueError(f"{name} must be a non-empty contiguous 1-D {dtype} tensor of {numel} on {device}")
    (module or _device_module()).expert_stream_route_labels(
        routes, planned, count, lane_kind, out_kinds, out_stamp
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
            if table[hits[0]][2] != int(tables.row_bytes[row, name]):
                raise ValueError(
                    f"streamed name {name}: the copy table's rows hold {table[hits[0]][2]} B, "
                    f"the slab's {int(tables.row_bytes[row, name])} B"
                )
            entry_of[name] = hits[0]
    names = [int(name) for name in tables.segments[row, :, 0].tolist()]
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
    "sm") and ``cpu_misses`` carry ``SGLANG_EXPERT_STREAM_RAM_HIT_COPY`` and
    ``SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_MISSES``; ``miss_cut``/``miss_cut_max`` carry the experiment
    ``SGLANG_EXPERT_STREAM_CPU_SPLIT``'s miss_cut/miss_cut_max (ram_slot_map.type_lanes); ``split_model`` carries the
    experiment ``SGLANG_EXPERT_STREAM_CPU_SPLIT`` model's cost words (ram_slot_map.split_model_words); ``lease_pdl``
    (``SGLANG_EXPERT_STREAM_ENABLE_LEASE_PDL``) launches every chain kernel but C1 and CC with
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
        layout: BuildKey = EXL3_BUILD,
        lease_pdl: bool = False,
        hit_copy: str = "ce",
        cpu_misses: bool = False,
        miss_cut: int = 0,
        miss_cut_max: int = 3,
        lanes: int = 8,
        nodes: int = 1,
        split_model: Optional[Sequence[int]] = None,
        config: Optional[ExpertStreamConfig] = None,
    ) -> None:
        self.wire = expert_lease_block.wire_layout(lanes, nodes)
        # Whether the device module is the stamped build (device_stamps_enabled); None is stream_config().
        self._config = config
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
        for name, value in (("miss_cut", miss_cut), ("miss_cut_max", miss_cut_max)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self.miss_cut, self.miss_cut_max = int(miss_cut), int(miss_cut_max)
        # The split model's cost words (ram_slot_map.split_model_words) and its counters, or empty: the split and the
        # miss cut.
        costs = 5 * self.wire.nodes + 6
        if split_model is not None and len(split_model) != costs:
            raise ValueError(f"the split model has {costs} cost words for {self.wire.nodes} nodes, got {len(split_model)}")
        # The counters (ram_slot_map.split_model_stats_words): records, then per node to_cpu and to_copy.
        words = [] if split_model is None else list(split_model) + [0] * (1 + 2 * self.wire.nodes)
        self.split_model = torch.tensor(words, dtype=torch.int64, device=device)
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
        # Per row, node-major: its own split (set_row_split), or -1 first: the lease's.
        self.row_split = torch.full(
            (layers, self.wire.nodes * (self.wire.lanes + 1)), -1, dtype=torch.int32, device=device
        )
        # The post's outputs: each lane's kind, source slot (S and CW read them) and home
        # node (CW reads it), and C1's compacted SM hits, in one order.
        self.lane_kind = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        self.lane_slot = torch.full((self.wire.lanes,), -1, dtype=torch.int32, device=device)
        self.lane_node = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        self.go_1 = torch.zeros(1, dtype=torch.int32, device=device)
        self.host_rows_1 = torch.zeros(self.wire.lanes, dtype=torch.int64, device=device)
        self.dst_slots_1 = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
        # The GPU drip (enable_gpu_drip): the scratch slots the post reads, the picks spec_score publishes, and C1s's
        # compacted HIT_SCRATCH lanes.
        self.drip_slots = 0
        self.drip_picks = 0
        self.scratch_table = torch.zeros(DRIP_TABLE_SLOTS, dtype=torch.int64, device=device)  # unread while off
        self.scratch_go_1 = torch.zeros(1, dtype=torch.int32, device=device)
        self.scratch_rows_1 = torch.zeros(self.wire.lanes, dtype=torch.int64, device=device)
        self.scratch_dst_1 = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)
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
        # The rows' token capacity and the bytes between two tokens' staged inputs (enable_cpu_experts); 1 and 0
        # until then.
        self.cpu_tokens_max, self.cpu_x_token_bytes = 1, 0
        # Set by the row backend when it captures a post that lets the service copy; the
        # service arms its copy engine on it.
        self.copy_engine_captured = False
        self.piece_runs = piece_runs.to(device).contiguous()
        # The GPU scorer (enable_spec_scorer): its (top_k, per_token, top_k_only), the candidate page and the score
        # scratch; None when off.
        self._spec = None
        self.spec_candidates = None
        self.spec_scores = None
        # The stream the scoring forks onto (spec_score) until spec_join; None on a CPU device.
        self.spec_stream = None
        self._spec_forked = False
        # The learned head (enable_spec_head, a ServingHead) that replaces spec_select; None when off.
        self.spec_head = None
        self._stamp_ring: Optional[torch.Tensor] = None  # enable_device_stamps

    def _kernels(self):
        """Return the device module, loading it on first use."""
        if self._module is None:
            self._module = _device_module(
                self._layout, self.wire.lanes, self.wire.nodes, self._config
            )
        return self._module

    def enable_device_stamps(self) -> None:
        """Point the stamped module's chain kernels at a pinned ring (device_stamps.cuh). Needs
        ``device_stamps_enabled()``; the ring lives as long as this object."""
        if not device_stamps_enabled(self._config):
            raise RuntimeError(
                "device stamps need SGLANG_EXPERT_STREAM_TRACE's job.prefix (the stamped device module)"
            )
        ring = torch.zeros(DEVICE_STAMP_RING * DEVICE_STAMP_WORDS, dtype=torch.int64).pin_memory()
        self._kernels().expert_stream_set_device_stamps(ring)
        self._stamp_ring = ring

    def dump_device_stamps(self, path: str) -> int:
        """Write each stamped record as a JSON line (seq, then each DEVICE_STAMP_POINTS in %globaltimer ns, 0 when
        not reached), by post_start. Call once the device is idle. Returns the record count; 0 without stamps."""
        ring = self._stamp_ring
        if ring is None:
            return 0
        rows = ring.view(DEVICE_STAMP_RING, DEVICE_STAMP_WORDS).tolist()
        records = sorted((r for r in rows if r[0] != 0), key=lambda r: r[1])
        with open(path, "w") as f:
            f.write(json.dumps({"schema": 1, "clock": "globaltimer", "points": list(DEVICE_STAMP_POINTS)}) + "\n")
            for r in records:
                f.write(json.dumps({"seq": r[0], **dict(zip(DEVICE_STAMP_POINTS, r[1:]))}) + "\n")
        return len(records)

    def enable_cpu_experts(self, x_rows: torch.Tensor, out_rows: torch.Tensor) -> None:
        """Take the same pinned rows the host's ``enable_cpu_experts`` took.

        There is one row per layer, readable through UVA. A multi-token row's ``out_rows`` is
        ``[layers, 2 * nodes, tokens, hidden]``; ``x_rows`` then holds
        ``expert_lease_block.cpu_row_bytes`` a row (the token table follows the inputs).
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
        tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
        hidden = int(out_rows.shape[-1])
        if not 1 <= tokens <= CPU_TOKENS_MAX:
            raise ValueError(f"CPU expert rows hold 1-{CPU_TOKENS_MAX} tokens, not {tokens}")
        if x_rows.shape[1] < cpu_row_bytes(hidden, tokens, self.wire.lanes):
            raise ValueError(
                f"x_rows holds {x_rows.shape[1]} bytes a row; {tokens} tokens of {hidden} need "
                f"{cpu_row_bytes(hidden, tokens, self.wire.lanes)}"
            )
        self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows
        self.cpu_tokens_max, self.cpu_x_token_bytes = tokens, -(-2 * hidden // 16) * 16

    def set_row_copy(self, row: int, dst_rows: int) -> None:
        """Record that the row's copy table is registered with ``dst_rows`` rows.

        The post may then type the row's hits kHitCopy.
        """
        self._check_row(row)
        self.map_bank["dst_rows"][row] = int(dst_rows)
        self.map_bank["ce_ok"][row] = 1

    def enable_gpu_drip(self, slots: int, *, table: torch.Tensor, picks: int = 0) -> None:
        """The GPU drip (host/gpu_drip.h, the host's ``enable_gpu_drip``): every captured armed post from now on reads
        the first ``slots`` words of ``table`` (int64 [DRIP_TABLE_SLOTS] on this device, zeroed, the host's
        ``set_drip_scratch`` table) and types a lane whose (row, expert) one names HIT_SCRATCH, for C1s
        (``scratch_rows_1``, ``scratch_dst_1``, ``scratch_go_1``). ``picks`` > 0 (needs ``enable_spec_scorer``): each
        ``spec_score`` also publishes that many drip picks of its target layer. Before any graph captures a post."""
        if not 1 <= slots <= DRIP_TABLE_SLOTS:
            raise ValueError(f"the scratch table holds 1..{DRIP_TABLE_SLOTS} slots, not {slots}")
        if table.dtype != torch.int64 or table.numel() != DRIP_TABLE_SLOTS or table.device != self.state.device:
            raise ValueError(f"the scratch table is int64 [{DRIP_TABLE_SLOTS}] on {self.state.device}")
        self.scratch_table = table
        if not 0 <= picks <= CAND_MAX_DRIPS:
            raise ValueError(f"the drip publishes 0..{CAND_MAX_DRIPS} picks a record, not {picks}")
        if picks and self._spec is None:
            raise RuntimeError("drip picks come from the GPU scorer: enable_spec_scorer first")
        self.drip_slots = int(slots)
        self.drip_picks = int(picks)

    def set_row_split(self, row: int, split: Sequence[int]) -> None:
        """Give the row its own split, ``split[n]`` CPU lanes of ``n`` eligible on every node, in place of the lease's
        tables (which calibration and the retune write). Before any graph captures a post of the row."""
        self._check_row(row)
        if len(split) != self.wire.lanes + 1 or any(not 0 <= k <= n for n, k in enumerate(split)):
            raise ValueError(f"a row split lists {self.wire.lanes + 1} counts with 0 <= split[n] <= n, got {list(split)}")
        self.row_split[row] = torch.tensor(list(split) * self.wire.nodes, dtype=torch.int32)

    def set_row_cpu(self, row: int) -> None:
        """Record that the row's CPU layer is registered.

        The post may then type its lanes kHitCpu or kMissCpu.
        """
        self._check_row(row)
        self.map_bank["cpu_ok"][row] = 1

    def enable_spec_scorer(self, candidates: torch.Tensor, *, top_k: int, per_token: int, top_k_only: bool) -> None:
        """The GPU scorer (SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu): ``candidates`` is the host's candidate page
        (``new_candidate_page``); scores go to an fp32 [cpu_tokens_max, experts] scratch allocated here, so this runs
        after ``enable_cpu_experts`` and before any graph captures ``spec_score``."""
        if self._spec is not None:
            raise RuntimeError("the GPU scorer is already enabled on this device")
        if (
            candidates.dtype != torch.uint8
            or candidates.device.type != "cpu"
            or not candidates.is_contiguous()
            or candidates.numel() != CAND_PAGE_BYTES
        ):
            raise ValueError(
                f"candidates must be a contiguous CPU uint8 tensor of {CAND_PAGE_BYTES} bytes (new_candidate_page)"
            )
        if torch.device(self.state.device).type == "cuda" and not candidates.is_pinned():
            raise ValueError("candidates must be pinned: the select kernel writes it through UVA")
        if self.experts > SPEC_SELECT_MAX_EXPERTS:
            raise ValueError(
                f"the GPU scorer's select kernel holds at most {SPEC_SELECT_MAX_EXPERTS} experts, not {self.experts}"
            )
        if not 1 <= top_k <= min(SPEC_DEPTH, self.experts):
            raise ValueError(f"top_k must be in 1..{min(SPEC_DEPTH, self.experts)}, got {top_k}")
        if not 1 <= per_token <= SPEC_DEPTH:
            raise ValueError(f"per_token must be in 1..{SPEC_DEPTH}, got {per_token}")
        self._spec = (int(top_k), int(per_token), bool(top_k_only))
        self.spec_candidates = candidates
        self.spec_scores = torch.empty(
            (self.cpu_tokens_max, self.experts), dtype=torch.float32, device=self.state.device
        )
        if torch.device(self.state.device).type == "cuda":
            self.spec_stream = torch.cuda.Stream(device=self.state.device)

    def enable_spec_head(self, head) -> None:
        """Serve the learned head (a ``ServingHead``) after ``spec_score``'s native scoring instead of
        ``spec_select``. Runs after ``enable_spec_scorer`` and before any graph captures ``spec_score``."""
        if self._spec is None:
            raise RuntimeError("enable_spec_scorer must run before enable_spec_head")
        if self.spec_head is not None:
            raise RuntimeError("a learned head is already enabled on this device")
        experts, hidden = int(head.weights.w_out.shape[0]), int(head.weights.w_pca.shape[1])
        if experts != self.experts or hidden != int(self.cpu_out_rows.shape[-1]):
            raise ValueError(
                f"the head has {experts} experts and hidden size {hidden}; this device has {self.experts} "
                f"and {int(self.cpu_out_rows.shape[-1])}"
            )
        self.spec_head = head

    def spec_score(
        self,
        entry,
        x: Optional[torch.Tensor],
        *,
        routes: Optional[torch.Tensor] = None,
        route_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Score the record this device just posted: layer T's MoE input ``x`` (bf16 [tokens, hidden] or [hidden], or
        None) against ``entry`` (a ``SpecScoreRow``), then publish its candidates to the record's slot. A record of no
        input or more than cpu_tokens_max tokens publishes count 0. Right after ``post``: forks onto ``spec_stream``
        until ``spec_join``, so the post's stream goes on to C1 without waiting for it. ``routes`` (int64, flat
        [>= tokens * top_k]) and ``route_weights`` ([tokens, top_k]) feed the learned head and are ignored without one."""
        if self._spec is None:
            raise RuntimeError("the GPU scorer is not enabled on this device (enable_spec_scorer)")
        self._check_row(entry.target)
        tokens = 0 if x is None else (int(x.shape[0]) if x.dim() == 2 else 1)
        module = self._kernels()
        if self.spec_stream is None:
            self._spec_launch(entry, x, tokens, module, routes=routes, route_weights=route_weights)
            return
        # A wait_stream fork, which a breakable capture tracks and joins at a segment end if spec_join has not.
        self.spec_stream.wait_stream(torch.cuda.current_stream(self.spec_stream.device))
        with torch.cuda.stream(self.spec_stream):
            self._spec_launch(entry, x, tokens, module, routes=routes, route_weights=route_weights)
        self._spec_forked = True

    def spec_join(self) -> None:
        """Make the current stream wait for the scoring spec_score forked: before anything that may post again (the
        record's seq), free ``x`` or rewrite the score scratch."""
        if self._spec_forked:
            torch.cuda.current_stream(self.spec_stream.device).wait_stream(self.spec_stream)
            self._spec_forked = False

    def _spec_head_launch(self, entry, x, tokens: int, module, routes, route_weights) -> bool:
        """Launches the learned head's serving chain; returns whether its block scores are this record's."""
        # Lazy: spec_head imports this module, and the production path must not load it.
        from sglang.kernels.ops.moe import spec_head

        head = self.spec_head
        if 1 <= tokens <= self.cpu_tokens_max:
            run_spec_score(x.reshape(tokens, -1), entry.weight, entry.bias, self.spec_scores, module=module)
        top_k = int(head.src_w.shape[1])
        in_range = (
            x is not None and 1 <= tokens <= min(spec_head.HEAD_MAX_TOKENS, self.cpu_tokens_max)
        )
        if in_range:
            head.src_w[:tokens].copy_(route_weights.to(torch.float32))
            src_ids = routes[: tokens * top_k].view(tokens, top_k)
        else:
            src_ids = None
        spec_head.run_spec_head_serving(
            weights=head.weights,
            workspace=head.workspace,
            x=x.reshape(tokens, -1) if in_range else x,
            tokens=tokens,
            scores=self.spec_scores,
            src_ids=src_ids,
            src_w=head.src_w,
            positions=head.positions,
            row=head.row_of_target[entry.target],
            hot_slots=entry.hot_slots,
            hot_capacity=entry.hot_capacity,
            ram_slot=self.map_bank["ram_slot"],
            target=entry.target,
            threshold=head.threshold,
            state=self.state,
            candidates=self.spec_candidates,
            picks=head.picks,
        )
        return in_range

    def _spec_launch(self, entry, x, tokens: int, module, *, routes=None, route_weights=None) -> None:
        if self.spec_head is not None:
            in_range = self._spec_head_launch(entry, x, tokens, module, routes, route_weights)
            if self.drip_picks:
                # The head's block scores, one row; stale when the head published its empty record.
                scores = self.spec_head.workspace.block_scores.view(1, -1)
                self._spec_drip_launch(entry, scores, 1 if in_range else 0, module)
            return
        self._spec_select_launch(entry, x, tokens, module, routes=routes, route_weights=route_weights)
        if self.drip_picks:
            # A dead row's scores are the anchor's (its input is), so the rows' max is the live rows'.
            rows = tokens if 1 <= tokens <= self.cpu_tokens_max else 0
            self._spec_drip_launch(entry, self.spec_scores, rows, module)

    def _spec_drip_launch(self, entry, scores, rows: int, module) -> None:
        run_spec_drip(
            scores,
            rows,
            hot_slots=entry.hot_slots,
            hot_capacity=entry.hot_capacity,
            ram_slot=self.map_bank["ram_slot"],
            target=entry.target,
            picks=self.drip_picks,
            state=self.state,
            candidates=self.spec_candidates,
            module=module,
        )

    def _spec_select_launch(self, entry, x, tokens: int, module, *, routes=None, route_weights=None) -> None:
        top_k, per_token, top_k_only = self._spec
        if 1 <= tokens <= self.cpu_tokens_max:
            run_spec_score(x.reshape(tokens, -1), entry.weight, entry.bias, self.spec_scores, module=module)
        # The post's routes per token, as the post kernel derives them (cpu_weights_count / cpu_tokens).
        route_top_k = (
            route_weights.numel() // tokens if routes is not None and route_weights is not None and tokens else 0
        )
        run_spec_select(
            self.spec_scores,
            tokens,
            top_k=top_k,
            per_token=per_token,
            top_k_only=top_k_only,
            hot_slots=entry.hot_slots,
            hot_capacity=entry.hot_capacity,
            ram_slot=self.map_bank["ram_slot"],
            target=entry.target,
            state=self.state,
            candidates=self.spec_candidates,
            routes=routes if route_top_k else None,
            route_top_k=route_top_k,
            module=module,
        )

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

        0 for one-part rows. (``tokens * hidden`` for a multi-token row.)
        """
        return int(self.cpu_out_rows.stride(1)) if self.cpu_out_rows.dim() >= 3 else 0

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
        may have no VRAM victim (SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES): a lane whose ``dst_slots`` entry is -1
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
                    "expert-stream DIRECT needs a hot sidecar and a valid slot capacity"
                )
        cpu_x, cpu_weights, cpu_x_dst = self._no_cpu, self._no_cpu, 0
        if cpu_input is not None:
            if self.cpu_x_rows is None:
                raise RuntimeError("CPU experts are not enabled on this device")
            if not captured:
                raise ValueError("only a captured post stages the CPU experts' input")
            cpu_x, cpu_weights = cpu_input
            tokens = cpu_x.shape[0] if cpu_x.dim() == 2 else 1
            if tokens > self.cpu_tokens_max:
                raise ValueError(f"a {tokens}-token input does not fit rows of {self.cpu_tokens_max} tokens")
            if cpu_x.shape[-1] * 2 > self.cpu_x_rows.shape[1]:
                raise ValueError(
                    f"a {cpu_x.shape[-1]}-wide input does not fit the {self.cpu_x_rows.shape[1]}-byte row"
                )
            cpu_x, cpu_weights = cpu_x.reshape(tokens, -1), cpu_weights.reshape(-1)
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
        token_bytes = self.cpu_x_token_bytes or -(-2 * int(cpu_x.shape[-1]) // 16) * 16
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
            self.miss_cut,
            self.miss_cut_max,
            self.lane_kind,
            self.lane_slot,
            self.lane_node,
            self.go_1,
            self.host_rows_1,
            self.dst_slots_1,
            cpu_x,
            cpu_x_dst,
            cpu_weights,
            self.cpu_tokens_max,
            token_bytes,
            spill_on,
            overflow_flag,
            gather_overflow,
            self.drip_slots,
            self.scratch_table,
            self.scratch_rows_1,
            self.scratch_dst_1,
            self.scratch_go_1,
            self.row_split,
            self.split_model,
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

    def calibration_pull(self, table: torch.Tensor) -> None:
        """The split model calibration's stand-in for S's host pulls (expert_stream_calibration_pull_kernel), on the
        current stream: ``table`` is int64 ``[n, 3]`` {source, destination, bytes} on this device, each source a pinned
        host address."""
        self._kernels().expert_stream_calibration_pull(table)

    def set_split_model_costs(self, words: Sequence[int]) -> None:
        """Replace the split model's cost words (ram_slot_map.split_model_words, or split_model_off_words) in place,
        leaving its counters. The copy is queued on the current stream from pinned memory, so a captured post reads
        the new words from its next replay on; the host does not wait for it."""
        costs = 5 * self.wire.nodes + 6  # the costs and the flags word
        if self.split_model.numel() == 0:
            raise ValueError("the split model is off: the post has no cost words to replace")
        if len(words) != costs:
            raise ValueError(f"the split model has {costs} cost words for {self.wire.nodes} nodes, got {len(words)}")
        src = torch.tensor(list(words), dtype=torch.int64)
        if self.split_model.is_cuda:
            # The caching host allocator keeps the pinned block until the copy that reads it has run.
            src = src.pin_memory()
        self.split_model[:costs].copy_(src, non_blocking=True)

    def split_model_costs(self) -> Optional[list[int]]:
        """The split model's cost words as the device holds them (a device-to-host copy), or None when it is off."""
        if self.split_model.numel() == 0:
            return None
        return self.split_model[: 5 * self.wire.nodes + 6].cpu().tolist()

    def split_model_stats(self) -> Optional[dict]:
        """The split model's counters (a device-to-host copy), or None when it is off: records it typed, and per node
        the lanes it put on the CPU that the table would not (``to_cpu``) and took off the CPU (``to_copy``)."""
        if self.split_model.numel() == 0:
            return None
        words = self.split_model[5 * self.wire.nodes + 6 :].cpu().tolist()
        return {
            "records": words[0],
            "to_cpu": words[1::2],
            "to_copy": words[2::2],
        }

    def stats(self) -> dict[str, int]:
        """Return the device words ``STATE_WORDS`` (a device-to-host copy)."""
        values = self.state.cpu().tolist()
        return {name: values[index] for name, index in STATE_WORDS.items()}
