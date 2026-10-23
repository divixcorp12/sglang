"""The expert engine service: RAM-miss service for streamed experts.

A native service thread fills the pinned RAM tier's slots: it reads the missing experts'
row images with O_DIRECT straight into the layers' pinned slabs, with no Python on the
path. Device kernels in the decode graph exchange records with that thread over the
slot-map protocol (``analysis/dsv41-drive/LEASE_PROTOCOL.md``).

``ExpertEngineService`` is built from an expert format and a model profile, and loads the
transport build keyed by ``(format.transport_build, profile.router)``. A key the transport
does not build, a model beyond the profile's caps, or CPU experts or a draft requested of
a format or profile that only has a stub, refuse at startup.

The pieces around it: ``startup`` (the checks), ``row_backend`` (a layer's slot table and
gather), ``formats/exl3_tables`` (the EXL3 flat tables the native reader runs from).
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import threading
import time
import weakref
from contextlib import nullcontext
from typing import Callable, Mapping, Optional

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_lease_block import WireLayout, wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    DRIP_TABLE_SLOTS,
    ExpertStreamDevice,
    ExpertStreamHost,
    device_stamps_enabled,
    host_layout,
    new_candidate_page,
    new_hot_page,
    new_page,
    stream_segment_map,
)
from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.layers.moe.copy_engine_quiet import (
    QuietReport,
    block_device_inflight_paths,
    cuda_module_loading_env,
    inflight_busy,
    wait_for_quiet,
)
from sglang.srt.layers.moe.cpu_experts.threading_config import ThreadingConfig
from sglang.srt.layers.moe.exl3_read_split import mirror_caps
from sglang.srt.layers.moe.exl3_stream_trace import GraphRouteLog
from sglang.srt.layers.moe.expert_engine.config import (
    CopyMode,
    CpuSplitConfig,
    ExpertStreamConfig,
    GpuDripConfig,
    PullMode,
    ResolvedConfig,
    resolved_config,
)
from sglang.srt.layers.moe.expert_engine.formats import ExpertFormat
from sglang.srt.layers.moe.expert_engine.profiles import ModelProfile
from sglang.srt.layers.moe.expert_engine.row_backend import (
    PREFILL_SHARE_FRACTION,
    ExpertEngineRowBackend,
    NativePinnedSlotTable,
    check_sm_small_copies,
    full_layer_shortfall,
    padded_plan_width,
    pool_room_shortfall,
    prefill_share_rows,
    sm_copy_mask,
    sm_copy_table,
    spill_room_shortfall,
    usable_rows,
)
from sglang.srt.layers.moe.expert_engine.startup import (
    check_requested_capabilities,
    resolve_transport_build,
)
from sglang.srt.layers.moe.expert_host_tier import quarantine_host_slabs
from sglang.srt.layers.moe.expert_stream import (
    pinned_full_layers,
    ram_prefetch_redundant,
    ram_prefetch_spec_share,
)
from sglang.srt.layers.moe.host_numa import group_ranges, slot_nodes
from sglang.srt.layers.moe.ram_slot_map import (
    split_miss_cut,
    split_model_off_words,
    split_model_overrides_complete,
    split_model_words,
)

logger = logging.getLogger(__name__)


def _quarantine_service_at_exit(service: weakref.ref[ExpertEngineService]) -> None:
    """Exit hook: quarantine the tiers, never free them.

    No device barrier is attempted in an exit handler, so nothing can prove a GPU
    reader has finished.
    """
    live = service()
    if live is not None:
        live.shutdown(at_exit=True)


# Decode forwards, counted after the first copy-engine capture, that run unarmed. The
# kernels a decode step launches for the first time (lazily loaded under
# CUDA_MODULE_LOADING=LAZY, outside any hook) then load while no copy wait can spin
# (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine"). Counting batches instead
# armed inside the server's 8-token warm-up request, whose first decode steps then
# deadlocked (at thresholds of 4 batches and of 1). 16 covers the warm-up; the value
# beyond that is arbitrary.
COPY_ENGINE_ARM_DECODES = 16
# CPU experts log their cumulative counters every this many batches, about twice per
# 128-token turn at batch 1.
CPU_STATS_LOG_BATCHES = 64

# The copy engine needs every kernel of every library loaded before it arms. Under LAZY
# (torch's default) a kernel loads at its first launch, and a first launch after arming
# can stop the copy thread's copies until the copy-wait timeout fail-stops the server. A
# seeded stress run of two min_p requests (a 525-token prompt, then a 3,314-token one)
# did so on the second's 14th decode step every time, and never under EAGER. EAGER loads
# a library's kernels when it is loaded, so what remains is a library loaded after
# arming, which the module-load guard drains for.
COPY_ENGINE_MODULE_LOADING = "EAGER"


def check_copy_engine_module_loading(environ: Mapping[str, str]) -> None:
    """Refuse the copy engine unless ``CUDA_MODULE_LOADING`` is EAGER.

    Torch sets LAZY when the variable is unset, so an unset variable is refused too.
    """
    value = environ.get("CUDA_MODULE_LOADING", "")
    if value != COPY_ENGINE_MODULE_LOADING:
        raise RuntimeError(
            "exl3 RAM miss: SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine needs CUDA_MODULE_LOADING="
            f"{COPY_ENGINE_MODULE_LOADING} in the server's environment (it is {value or 'unset'}): a kernel loaded "
            "lazily after the copy engine arms can hold its copies back until the RAM-miss deadline stops the server. "
            "EAGER costs ~1 GiB of device memory; raise --mem-fraction-static by ~0.03."
        )


def watchdog_wait_s(timeout_ms: int) -> float:
    """The C++ watchdog's abort limit for a wait timeout of ``timeout_ms``, in seconds.

    The limit is ``max(30 s, 3 x timeout)``. It must outlast the miss stream's deadline
    (a slow demand traps the device at the timeout) and the eager pause bound of
    ``2 x timeout + 1 s`` (``before_host_use``). Three times the timeout exceeds both
    for any timeout over 1 s; 30 s covers the rest.
    """
    return max(30.0, 3.0 * timeout_ms / 1000)


def parse_fault(spec: str) -> Optional[tuple[int, float]]:
    """Parse ``Dsv41Config.ram_miss_fault``, ``"<demands>:<seconds>"``.

    The fault delays each demand read by ``seconds`` once ``demands`` demands have read
    rows. An empty spec means no fault (``None``).
    """
    if not spec:
        return None
    demands, seconds = spec.split(":")
    return int(demands), float(seconds)


class ExpertEngineService:
    """The process-wide expert engine service: native thread, device kernels and hooks.

    Built from an expert format and a model profile, and loads the transport build keyed by
    ``(format.transport_build, profile.router)``. A singleton: ``bind`` creates it for a pairing
    (a second, different pairing in one process raises) and ``get`` returns it.

    Pinned tiers register their ``NativePinnedSlotTable`` before the service starts; it
    starts lazily on first use (``ensure_started``), once every layer's slabs exist, and
    refuses a tier built afterwards. ``attach`` then runs once per streamed layer to
    install its row backend.

    Eager use of a pinned tier brackets with ``before_host_use`` / ``after_host_use``,
    which drain queued device work and pause the thread (nesting counted); the eager
    paths' map changes reach the device before the thread runs again. The scheduler's
    fail-stop hook (``fail_stop_check``) does the per-batch upkeep.

    The copy engine is enabled at start but armed only by the scheduler
    (``arm_copy_engine_when_quiet``): after the server's startup warm-up, with the
    scheduler idle and the box quiet, once ``COPY_ENGINE_ARM_DECODES`` decode forwards
    have run after a copy-engine graph was captured, so the first decode steps load
    their kernels unarmed. Once armed, an eager forward and a module load first drain
    the device.

    ``shutdown`` frees the tiers only when it can establish that no GPU reader and no
    service thread still runs; otherwise it quarantines every buffer until process
    exit (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Shutdown").
    """

    _instance: Optional[ExpertEngineService] = None

    @classmethod
    def bind(
        cls,
        *,
        expert_format: type[ExpertFormat],
        profile: ModelProfile,
        config: Optional[ResolvedConfig] = None,
    ) -> ExpertEngineService:
        """The process's service for this (format, profile) pairing, created on first use.

        ``config`` is the process's resolved config (``resolved_config()`` when None); a later bind keeps the first's.
        Raises when a service is bound to another pairing, or when the pairing has no transport build.
        """
        if cls._instance is None:
            cls._instance = cls(expert_format=expert_format, profile=profile, config=config)
            return cls._instance
        bound = cls._instance
        if bound.expert_format.key != expert_format.key or bound.profile.name != profile.name:
            raise RuntimeError(
                f"the expert engine is bound to format {bound.expert_format.key!r} with profile "
                f"{bound.profile.name!r}; it cannot also serve format {expert_format.key!r} with profile "
                f"{profile.name!r} in one process"
            )
        return bound

    @classmethod
    def get(cls) -> ExpertEngineService:
        if cls._instance is None:
            raise RuntimeError(
                "no expert engine service is bound: quant_hook.attach binds it when the first streamed layer loads"
            )
        return cls._instance

    def __init__(
        self,
        *,
        expert_format: type[ExpertFormat],
        profile: ModelProfile,
        config: Optional[ResolvedConfig] = None,
    ) -> None:
        # Resolved once: every engine knob below is read from it, never from the environment.
        self.config = resolved_config() if config is None else config
        if self.config.stream.copy.mode is CopyMode.DMA:
            raise ValueError(
                "SGLANG_EXPERT_STREAM_COPY_MODE=dma: dma is the legacy Qwen streamer's copy mode; the native engine "
                "takes sm_copy or copy_engine"
            )
        # NVTX ranges around the eager host-use pause's stream sync and thread pause.
        self._sync_wait_nvtx = self.config.stream.trace.sync_wait_nvtx
        self.expert_format = expert_format
        self.profile = profile
        # (format, router): which transport build this service loads; refused here when the pair is not built.
        self.build_key = resolve_transport_build(expert_format=expert_format, profile=profile)
        self.tables: dict[int, NativePinnedSlotTable] = {}
        self.host: Optional[ExpertStreamHost] = None
        # The wire group of the GPU's node, whose CPU expert engine serves the DSpark draft (set by ensure_started).
        self.gpu_group: Optional[int] = None
        self._gpu_node_placement: Optional[tuple[int, list[int]]] = None
        self.device_side: Optional[ExpertStreamDevice] = None
        self.page = None
        self.slot_map = None
        self._node_ranges = None
        # The RAM prefetch's pool slots per row and group, set at start (0: no pool).
        self._spec_share = 0
        # SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE: the rows per layer a prefill may own, set at start (0: off).
        self._prefill_share_rows = 0
        # The GPU scorer (SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu), set at start; its per-row table is built as
        # rows attach.
        self._spec_gpu = None
        self._spec_hot: dict[int, tuple] = {}
        # SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS: (slots, picks) once enabled at start, and the scratch tensors the first attach
        # allocates; None when off.
        self._drip: Optional[tuple[int, int]] = None
        self._drip_scratch: Optional[list[torch.Tensor]] = None
        self._drip_table: Optional[torch.Tensor] = None  # the scratch table, device memory (attach)
        self._row_backends: dict[int, object] = {}
        self._rows: dict[int, int] = {}
        self._manager = None
        # The only caller of host.pause()/resume(), which are not reentrant.
        self._pause_depth = 0
        self._trace_rows: Optional[list[int]] = None
        self._trace_graph: Optional[list[int]] = None
        # One reusable pinned snapshot. A busy slot coalesces later cumulative register
        # values; neither the scheduler nor the CUDA graph waits for it.
        self._trace_snapshot: Optional[torch.Tensor] = None
        self._trace_event: Optional[torch.cuda.Event] = None
        self._trace_pending_rows: Optional[list[int]] = None
        # The host records a stage line per request (trace runs only).
        self._stages_traced = False
        self._stages_dropped = 0
        # Trace runs only.
        self.route_log: Optional[GraphRouteLog] = None
        # Routed rows of one batch-1 decode step over all in-graph layers (attach sums).
        self.routed_rows_per_step = 0
        self._shut_down = False
        self._quarantined = False
        self._completed = False
        # Enabled at start; armed later (see the class doc).
        self.copy_engine = False
        self.sm_small_copies = False
        # SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS: the CPU expert service; None when off.
        self.cpu_experts = None
        # SGLANG_EXPERT_STREAM_CPU_SPLIT's layer_tables: {layer id: its own split table}, set at start.
        self._layer_splits: dict[int, list[int]] = {}
        self._cpu_retune_batches = 0
        self._cpu_batches = 0
        self._cpu_log_batches = 0
        # SGLANG_EXPERT_STREAM_CPU_SPLIT's model: the live refinement of its calibrated costs, None until the
        # calibration published them (or when the model is off), and its model_refine_batches count.
        self._split_model_refiner = None
        self._split_model_refine_batches = 0
        self._split_model_batches = 0
        # SGLANG_EXPERT_STREAM_ENABLE_LEASE_PDL: the lease chain's kernels launch with PDL.
        self.lease_pdl = False
        # The widest gather any layer planned (plan_gather_width); None until one does.
        self._gather_planned: Optional[int] = None
        # The most routes one layer's graph gather takes (a verify's tokens x top_k); the CPU expert rows size from it.
        self._gather_routes_planned = 0
        # The victim lanes a spill gather plans (plan_staging_width); None without spill.
        self._staging_planned: Optional[int] = None
        # The build's wire, fixed at start from the planned width.
        self._wire: Optional[WireLayout] = None
        self._copy_armed = False
        self._copy_decodes = 0
        # Module loads that drained the device first.
        self.copy_engine_module_loads = 0
        # A prefill fill's wait gives up after the watchdog's limit, so a hung fill ends
        # in the watchdog's abort first.
        self.fill_timeout_s = watchdog_wait_s(2000) + 5.0
        self.hot_page = None
        self.gpu_hot_enabled = False
        self._gpu_hot_updater = None
        self._hot_snapshot = None
        self._hot_lists: dict[int, list[int]] = {}

    def register(self, layer_id: int, table: NativePinnedSlotTable) -> None:
        """Register a layer's slot table; only valid before the service starts."""
        if self.host is not None:
            raise RuntimeError(
                "exl3 RAM miss: a pinned tier was built after the service started"
            )
        self.tables[layer_id] = table

    def plan_gather_width(self, rows: int, routes: int = 0) -> None:
        """Plan a layer whose graph gather misses ``rows`` ids; the widest layer sets the build's lane count.

        ``routes`` is the layer's gather routes when they exceed its miss lanes: the service starts before the streamers
        enable their graph gather, so the CPU expert rows learn the verify's token count here.
        Only valid before the service starts. Raises ValueError for a width outside 1..64.
        """
        if self.host is not None:
            raise RuntimeError(
                "exl3 RAM miss: the graph gather width was planned after the service started"
            )
        self._gather_routes_planned = max(self._gather_routes_planned, int(routes))
        planned = max(1, int(rows))
        wire_layout(planned)
        self._gather_planned = (
            planned
            if self._gather_planned is None
            else max(self._gather_planned, planned)
        )

    def plan_staging_width(self, rows: int) -> None:
        """Plan the staging slots a row reserves under spill: the victim lanes, since only live misses stage (a forced
        miss is read into a RAM victim). Only valid before the service starts; the widest plan wins."""
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: the staging width was planned after the service started")
        planned = max(1, int(rows))
        self._staging_planned = planned if self._staging_planned is None else max(self._staging_planned, planned)

    @property
    def wire(self) -> WireLayout:
        """The started build's wire; raises before ``ensure_started`` fixes the lane count."""
        if self._wire is None:
            raise RuntimeError(
                "exl3 RAM miss: the build's wire is read before the service started"
            )
        return self._wire

    @property
    def lanes(self) -> int:
        """The started build's lane count; raises before ``ensure_started``."""
        return self.wire.lanes

    @property
    def copy_engine_armed(self) -> bool:
        """Whether the copy engine armed; with CPU experts, their split and split model calibrated first."""
        return self._copy_armed

    def resolved_lanes(self) -> int:
        """The lane count the service builds for: the widest planned gather, rounded up to 8."""
        return wire_layout(self._gather_planned or 1).lanes

    def staging_width(self) -> int:
        """The staging slots every row asks for: the planned victim lanes under spill, the planned gather width, else
        the build's lanes."""
        return self._staging_planned or self._gather_planned or self.resolved_lanes()

    def _check_spill_room(self, row: int, streamer, width: int) -> None:
        """Refuse a layer whose node ranges cannot place every forced CPU miss (spill_room_shortfall)."""
        capacity = int(self.host.tables.capacity[row])
        # group_ranges is per group, then per row.
        ranges = [group[row] for group in self._node_ranges] if self._node_ranges is not None else [(0, capacity)]
        short = spill_room_shortfall(
            ranges,
            staging=self.staging_for(capacity),
            lanes=width,
            hot=int(streamer.hot_cache.capacity),
            pool=self._spec_share,
        )
        if short:
            raise ValueError(
                f"exl3 RAM miss: spill reads every forced CPU miss into a RAM victim; layer {streamer.layer_id}'s "
                f"node ranges (group, slots, needed) {short} are too small: raise SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB "
                "or lower SGLANG_EXPERT_STREAM_HOT_GPU_MB"
            )

    def pool_slots(self) -> int:
        """The RAM prefetch pool's slots in every row: its share per NUMA group."""
        return self._spec_share * self.host.nodes if self._spec_share else 0

    def _check_pool_room(self, tables, share: int) -> None:
        """Refuse a pool share the host would refuse in any group's range (pool_room_shortfall), naming the option."""
        for row, layer_id in enumerate(tables.layer_ids):
            capacity = int(tables.capacity[row])
            ranges = [group[row] for group in self._node_ranges] if self._node_ranges is not None else [(0, capacity)]
            short = pool_room_shortfall(ranges, staging_width=self.staging_width(), share=share)
            if short:
                g, lo, hi, staging = short[0]
                raise RuntimeError(
                    f"exl3 RAM miss: SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE={share} leaves layer {layer_id}'s group {g} "
                    f"slots [{lo}, {hi}) fewer than 2 assignable slots after {staging} staging and {share} pool "
                    "slots: lower the share or raise the pinned tier"
                )

    def _check_full_layers(self, tables, experts: int) -> None:
        """Refuse a SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS layer whose groups cannot map every expert past their staging
        and pool slots (full_layer_shortfall)."""
        full = pinned_full_layers(config=self.config.stream)
        share = ram_prefetch_spec_share(config=self.config.stream)
        for row, layer_id in enumerate(tables.layer_ids):
            if layer_id not in full:
                continue
            capacity = int(tables.capacity[row])
            ranges = [group[row] for group in self._node_ranges] if self._node_ranges is not None else [(0, capacity)]
            short = full_layer_shortfall(ranges, staging_width=self.staging_width(), share=share, experts=experts)
            if short:
                raise ValueError(
                    f"exl3 RAM miss: layer {layer_id} is in SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS, but its {capacity} "
                    f"slots leave groups (group, assignable slots, homed experts) {short} short of every expert"
                )

    @staticmethod
    def _configured_layer_splits(
        tables, lanes: int, *, cpu_experts: bool, split: Optional[CpuSplitConfig] = None
    ) -> dict[int, list[int]]:
        """SGLANG_EXPERT_STREAM_CPU_SPLIT's layer_tables (``split``; the process's when None) for a ``lanes``-lane
        build; refused without CPU experts (``cpu_experts``) or for a layer the service does not stream."""
        from sglang.srt.layers.moe.cpu_experts.service import configured_layer_splits

        splits = configured_layer_splits(lanes, config=split)
        if not splits:
            return splits
        if not cpu_experts:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_CPU_SPLIT's layer_tables needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS"
            )
        unknown = sorted(set(splits) - {int(layer_id) for layer_id in tables.layer_ids})
        if unknown:
            raise ValueError(
                f"exl3 RAM miss: SGLANG_EXPERT_STREAM_CPU_SPLIT's layer_tables names layers {unknown}, which are not "
                "streamed"
            )
        for layer_id, split in sorted(splits.items()):
            logger.info("exl3 RAM miss: layer %d uses its own CPU split %s on every node", layer_id, split)
        return splits

    def _fill_full_layer(self, streamer) -> None:
        """Read every expert of a SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS layer into its tier, so no verify reads one."""
        began = time.perf_counter()
        streamer.pinned_host_cache.ensure_rows(torch.arange(streamer.num_experts))
        mapping = self.host.mapping(self.row_of(streamer.layer_id))
        missing = [expert for expert, slot in enumerate(mapping) if slot < 0]
        if missing:
            raise RuntimeError(f"exl3 RAM miss: full layer {streamer.layer_id} could not map experts {missing}")
        logger.info(
            "exl3 RAM miss: full layer %d maps all %d experts (filled in %.2f s)",
            streamer.layer_id,
            streamer.num_experts,
            time.perf_counter() - began,
        )

    def staging_for(self, capacity: int) -> int:
        """The staging slots a row of ``capacity`` slots keeps: the planned width, and never its last slot."""
        return min(self.staging_width(), capacity - 1)

    def planned_padding(self, capacity: int) -> int:
        """The length of a row backend's ``planned`` tensor: its plan width, and at least the build's lanes."""
        return padded_plan_width(capacity, self.resolved_lanes())

    def row_of(self, layer_id: int) -> int:
        """The service row (position in the tables) of a streamed layer."""
        return self._rows[layer_id]

    def _refuse_if_shut_down(self) -> None:
        # shutdown() closes the C++ host but keeps self.host, whose calls would then
        # fail with an opaque "unknown handle".
        if self._shut_down:
            raise RuntimeError(
                "exl3 RAM miss: the option C service was shut down; its pinned tiers are closed"
            )

    def draft_host(self, areas, *, fatal_wait_s: float):
        """The DSpark draft channel on the GPU node's CPU expert engine (one team per node): a SharedDraftHost. Starts
        the service first when the draft is prepared before the target's first gather."""
        check_requested_capabilities(
            expert_format=self.expert_format, profile=self.profile, cpu_experts=True, draft=True, cpu_draft=True
        )
        self.ensure_started()
        if self.cpu_experts is None:
            raise RuntimeError(
                "exl3 RAM miss: the draft shares the CPU expert team, but SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS is off"
            )
        if self.gpu_group is None:
            gpu_node, nodes = self._gpu_node_placement
            raise RuntimeError(
                f"exl3 RAM miss: the draft runs on the GPU's node {gpu_node} CPU expert team, but the tier's placement "
                f"(SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB) covers only nodes {nodes}; place the tier on node {gpu_node} too"
            )
        return self.host.draft_source(areas, fatal_wait_s=fatal_wait_s, group=self.gpu_group)

    def ensure_started(self) -> None:
        """Start the service on first use; a no-op once started.

        Validates the launch configuration, builds the tables over every layer's slabs
        and row images, creates the native host, enables the copy engine, CPU experts
        and trace as configured, starts the thread and installs the exit quarantine
        hook. Raises if the service was shut down or the configuration is refused.
        """
        self._refuse_if_shut_down()
        if self.host is not None:
            return
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        stream = self.config.stream
        if stream.trace.router_capture_path and not get_exl3_stream_trace().enabled:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_TRACE's router_capture_path needs its expert_path: the "
                "router capture is written by the stage trace's graph route log"
            )
        if stream.trace.residency_capture and not stream.trace.router_capture_path:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_TRACE's residency_capture needs its router_capture_path: the "
                "residency capture is written beside the router capture"
            )
        # The DSV4.1 knobs outside the engine config: layer fusion and the test fault.
        cfg = Dsv41Config.from_envs()
        streamers = {
            layer_id: table.streamer_of()
            for layer_id, table in sorted(self.tables.items())
        }
        missing = [
            layer_id
            for layer_id, s in streamers.items()
            if s is None or s.pinned_host_cache is None
        ]
        if missing:
            raise RuntimeError(
                f"exl3 RAM miss: layers {missing} have no pinned tier yet"
            )
        fmt = next(iter(streamers.values())).format
        check_requested_capabilities(
            expert_format=self.expert_format,
            profile=self.profile,
            cpu_experts=stream.cpu_experts.enabled,
            draft=False,
        )
        tables = fmt.engine_tables(streamers)
        lanes = self.resolved_lanes()
        layout_names, _ = host_layout(layout=self.build_key, lanes=lanes)
        if layout_names != fmt.names:
            raise RuntimeError(
                f"exl3 RAM miss: the host module's layout {layout_names} is not the format's names {fmt.names}"
            )
        numa = ThreadingConfig.from_env(
            cpu_experts=stream.cpu_experts.enabled,
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            config=stream,
        )
        # None when the tier's placement leaves out the GPU's node: only the DSpark draft needs the group (draft_host).
        self.gpu_group = next((p.group for p in numa.plans if p.node == numa.gpu_node), None)
        self._gpu_node_placement = (numa.gpu_node, [p.node for p in numa.plans])
        for line in numa.log_lines():
            logger.info("exl3 RAM miss %s", line)
        node_ranges = None
        if numa.nodes > 1:
            # Rows in layer order, as exl3_ram_miss_tables numbers them; a slot across a seam is in no range.
            node_ranges = group_ranges(
                [
                    slot_nodes(s.pinned_host_cache.tensors, s.pinned_host_cache.capacity)
                    for _, s in sorted(streamers.items())
                ],
                [plan.node for plan in numa.plans],
            )
        self._node_ranges = node_ranges
        self._check_full_layers(tables, fmt.num_experts(next(iter(streamers.values())).layer))
        layer_splits = self._configured_layer_splits(
            tables, wire_layout(lanes).lanes, cpu_experts=stream.cpu_experts.enabled, split=stream.cpu_experts.split
        )
        pin = torch.cuda.is_available()
        self._wire = wire_layout(lanes, numa.nodes)
        page = new_page(pin=pin, wire=self._wire)
        hot_page = new_hot_page(tables.starts.shape[1], pin=pin)
        slot_map = torch.full(tuple(tables.starts.shape), -1, dtype=torch.int32)
        slot_map = slot_map.pin_memory() if pin else slot_map
        # The native build is chosen once, here: the instrumented one whenever this
        # service will enable the stage trace or inject the test fault, however they
        # were switched on (the cached trace object or the config, not only their env
        # vars). That way it never loads the production build and then calls an entry
        # point production refuses.
        fault = parse_fault(cfg.ram_miss_fault)
        instrumented = get_exl3_stream_trace().enabled or fault is not None
        host = ExpertStreamHost(
            tables,
            page=page,
            slot_map=slot_map,
            hot_page=hot_page,
            layout=self.build_key,
            variant="instr" if instrumented else None,
            lanes=self.lanes,
            node_ranges=node_ranges,
            sq_thread_cpus=[-1 if p.sq is None else p.sq for p in numa.plans],
            config=stream,
        )
        try:
            # SGLANG_EXPERT_STREAM_MIRROR's dynamic (validated at launch): before the thread
            # starts, every group's reader picks each sub-read's root under these caps.
            mirror = stream.tiers.mirror
            caps = mirror_caps(mirror.dynamic, mirror.caps, mirror.dirs)
            if caps is not None:
                host.set_mirror_caps(caps)
            logger.info(
                "exl3 RAM miss mirror roots: %s",
                "static split" if caps is None
                else "dynamic, in-flight caps " + ",".join(str(c) for c in caps),
            )
            # Before any slot is filled: the hot cache fills the tiers first, after
            # plan_gather_width.
            host.reserve_staging(self.staging_width())
            spec_share = ram_prefetch_spec_share(config=stream)
            if stream.prefetch.ram.enabled and ram_prefetch_redundant(config=stream):
                logger.info(
                    "exl3 RAM miss prefetch: off although SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH=1: every streamed "
                    "layer is in SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS, so no expert is left on NVMe to "
                    "prefetch (no pool, scorer or speculative cores)"
                )
            if spec_share:
                if not stream.cpu_experts.enabled:
                    raise RuntimeError(
                        "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH needs "
                        "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS"
                    )
                self._check_pool_room(tables, spec_share)
                # Right after the staging slots, before the hot cache fills any slot.
                host.reserve_spec_pool(spec_share)
            copy_engine = stream.copy.mode is CopyMode.COPY_ENGINE
            check_sm_small_copies(stream.copy)
            if copy_engine:
                check_copy_engine_module_loading(cuda_module_loading_env())
                # Before the thread starts, and unarmed. The watchdog bounds each copy
                # wait by the RAM-miss timeout.
                host.enable_copy_engine(
                    torch.cuda.current_device(),
                    spin_us=stream.copy.copy_idle_spin_us,
                    wait_timeout_ms=stream.copy.timeout_ms,
                    cpus=numa.copy_cpus,
                )
            cpu_experts = None
            if stream.cpu_experts.enabled:
                cpu_experts = self._start_cpu_experts(
                    cfg, host, fmt, streamers, pin, numa, self._gather_routes_planned, stream=stream
                )
            spec_gpu = (
                self._enable_ram_prefetch(
                    host, list(tables.layer_ids), numa, cpu_experts, pin=pin, config=self.config
                )
                if spec_share
                else None
            )
            if stream.copy.gpu_drip.slots:
                self._enable_gpu_drip(stream.copy.gpu_drip, host, spec_gpu, copy_engine=copy_engine)
            if get_exl3_stream_trace().enabled:
                # Before the thread starts: without a trace file it takes no timestamps.
                host.enable_trace()
                self._stages_traced = True
            share_recorder = None
            if stream.copy.prefill.share:
                from sglang.srt.eplb.expert_distribution import (
                    _ExpertDistributionRecorderNoop,
                    get_global_expert_distribution_recorder,
                )

                share_recorder = get_global_expert_distribution_recorder()
                if isinstance(share_recorder, _ExpertDistributionRecorderNoop):
                    # The no-op recorder drops pre-forward observers, so the share would
                    # never be set.
                    raise RuntimeError(
                        "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE needs the expert distribution recorder "
                        "(--expert-distribution-recorder-mode), which calls the pre-forward observer that sets it"
                    )
                self._prefill_share_rows = self._size_prefill_share(tables, spec_share)
            cores =[-1 if plan.ram is None else plan.ram for plan in numa.plans]
            busy_poll = all(plan.busy_poll for plan in numa.plans)
            spin_us = stream.copy.idle_spin_us
            if numa.nodes == 1 and cores[0] == -1 and not busy_poll:
                # The server's affinity: the thread spins there with PAUSE, then sleeps.
                host.start_thread(fatal_wait_s=stream.copy.watchdog_wait_s(), spin_us=spin_us)
            else:
                host.start_thread(
                    cpu_core=cores if numa.nodes > 1 else cores[0],
                    busy_poll=busy_poll,
                    fatal_wait_s=stream.copy.watchdog_wait_s(),
                    spin_us=spin_us,
                )
            if fault is not None:
                demands, seconds = fault
                host.inject(delay_s=seconds, delay_after_demands=demands)
                logger.warning(
                    "exl3 RAM miss TEST FAULT: demand reads sleep %.1f s after %d demands",
                    seconds,
                    demands,
                )
        except BaseException:
            # The thread writes into the tiers' slabs through raw addresses: join it
            # before anything can release them.
            host.stop()
            raise
        self.page, self.slot_map, self.host = page, slot_map, host
        self._spec_share = spec_share
        self._spec_gpu = spec_gpu
        self.numa = numa
        self.hot_page = hot_page
        self.copy_engine = copy_engine
        self.cpu_experts = cpu_experts
        self._layer_splits = layer_splits
        self._cpu_retune_batches = stream.cpu_experts.split.retune_batches
        self._split_model_refine_batches = stream.cpu_experts.split.model_refine_batches
        self.sm_small_copies = stream.copy.sm_small_copies
        self.lease_pdl = stream.copy.lease_pdl
        self.fill_timeout_s = stream.copy.fill_timeout_s()
        # Order matters. atexit runs last-registered first, and weakref.finalize
        # installs its single exit hook when the first finalizer (any tier's slab
        # unregister) is created. Registering here, after every tier exists (register()
        # refuses a later one), makes this hook run before that one, so the slabs are
        # quarantined and their finalizers detached before they could unregister.
        # Registered earlier, the quarantine would be silently undone.
        atexit.register(_quarantine_service_at_exit, weakref.ref(self))
        self._rows = {layer_id: row for row, layer_id in enumerate(tables.layer_ids)}
        if share_recorder is not None:
            share_recorder.register_pre_forward_observer(self._set_prefill_share)
        logger.info(
            "exl3 RAM miss thread started: %d layers, %d files, slot bytes %d, wait timeout %d ms, "
            "copy engine %s, CPU experts %s, RAM prefetch %s, build %s",
            len(tables.layer_ids),
            len(tables.paths),
            tables.slot_bytes,
            stream.copy.timeout_ms,
            "on" if copy_engine else "off",
            "on" if cpu_experts is not None else "off",
            f"share {spec_share}, {'GPU' if spec_gpu else 'CPU'} scorer" if spec_share else "off",
            self.host.variant,
        )

    @staticmethod
    def _start_cpu_experts(
        cfg: Dsv41Config,
        host,
        fmt,
        streamers,
        pin: bool,
        numa: ThreadingConfig,
        planned_routes: int = 0,
        *,
        stream: Optional[ExpertStreamConfig] = None,
    ):
        """Build the CPU expert service for ``SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS``; ``stream`` is the service's
        config (``resolved_config().stream`` when None).

        Runs after the copy engine is enabled and before the service thread starts. CPU
        lanes complete in the copy wait and leave the fused MoE through layer fusion's
        route tables, so the copy engine and layer fusion are both required; so is the
        fused plan, and the prefetch pull join must be off.
        """
        from sglang.srt.layers.moe.cpu_experts.service import (
            CpuExpertGroups,
            configured_split,
            cpu_trait_for,
        )

        stream = resolved_config().stream if stream is None else stream
        if stream.copy.mode is not CopyMode.COPY_ENGINE:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine"
            )
        if not cfg.enable_layer_fusion:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs SGLANG_DSV41_ENABLE_LAYER_FUSION"
            )
        if stream.prefetch.shadow.pull_mode is not PullMode.OFF:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS cannot run with the prefetch pull join"
            )
        if not stream.copy.fused_plan:
            # Only the fused plan sorts the miss lanes, which makes the tail lanes the
            # lowest-scored.
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs SGLANG_EXPERT_STREAM_ENABLE_FUSED_PLAN"
            )
        # Rows in layer order, as exl3_ram_miss_tables numbers them.
        caches = {
            row: s.pinned_host_cache.tensors
            for row, (_, s) in enumerate(sorted(streamers.items()))
        }
        trait = cpu_trait_for(fmt.key, source="target")
        hidden = {trait.hidden_size(slabs) for slabs in caches.values()}
        if len(hidden) != 1:
            raise RuntimeError(
                f"exl3 RAM miss: CPU experts need one hidden size, the layers have {hidden}"
            )
        # A DSpark verify gathers tokens x top_k routes a layer; the CPU rows hold that many tokens.
        # The service starts before the streamers enable their graph gather, so the planned routes stand in for it.
        for layer_id, s in streamers.items():
            if (s.graph_gather_rows or planned_routes) and getattr(s.layer, "top_k", None) is None:
                raise RuntimeError(
                    f"exl3 RAM miss: layer {layer_id}'s experts module has no top_k, which "
                    "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS sizes its rows from"
                )
        tokens = max(
            (
                (s.graph_gather_rows or planned_routes) // s.layer.top_k
                for s in streamers.values()
                if s.graph_gather_rows or planned_routes
            ),
            default=1,
        )
        # Spill: a split entry is only read for the live lanes, at most the victim lanes.
        victims = stream.copy.graph_gather_victim_lanes or None
        return CpuExpertGroups(
            host,
            trait,
            caches,
            hidden=hidden.pop(),
            plans=numa.plans,
            split=configured_split(host.wire.lanes, config=stream.cpu_experts.split),
            pin=pin,
            tokens=tokens,
            calibration_lanes=victims,
            config=stream.cpu_experts,
        )

    @staticmethod
    def _enable_ram_prefetch(
        host, layer_ids, numa, cpu_experts, *, pin: bool = False, config: Optional[ResolvedConfig] = None
    ):
        """Start the RAM prefetch over the pool reserved at start, each group's speculative thread on its plan's spare
        cores. CPU scorer: the router gates the model registered at load, copied to host memory once; returns None.
        GPU scorer: the targets and a candidate page (pinned when ``pin``) the host reads; returns the GpuScorer whose
        per-row table attach builds. ``config`` is the service's (``resolved_config()`` when None)."""
        from sglang.srt.layers.moe.ram_prefetch import (
            SCORERS,
            GpuScorer,
            load_margin_floors,
            margin_floors_path,
            prefetch_tables,
            prefetch_targets,
            registered_gates,
        )

        config = resolved_config() if config is None else config
        if cpu_experts is None:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS"
            )
        ram = config.stream.prefetch.ram
        scorer = ram.scorer
        if scorer not in SCORERS:
            raise ValueError(f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER must be one of {SCORERS}, got {scorer!r}")
        per_token = ram.per_token
        per_layer = ram.per_layer
        top_k_only = ram.top_k_only
        hidden = cpu_experts.services[0].hidden
        cores = [list(plan.spec) for plan in numa.plans]
        floors = margin_floors_path(scorer, dsv41=config.dsv41)
        horizon = ram.horizon
        idle = dict(
            idle_drive=ram.idle_drive,
            idle_deadline_s=ram.idle_deadline_us / 1e6,
            idle_spread=ram.idle_spread,
            idle_pieces=ram.idle_pieces,
        )
        promote = ram.promote
        pool_serve = ram.pool_serve
        if scorer == "gpu":
            picked = prefetch_targets(layer_ids, registered_gates(), hidden=hidden, horizon=horizon)
            candidates = new_candidate_page(pin=pin)
            min_margin = load_margin_floors(floors, len(layer_ids)) if floors else None
            learned = None
            head_path = config.dsv41.ram_prefetch.learned_head_path
            if head_path:
                from sglang.srt.layers.moe.ram_prefetch_learned_head import load_learned_head

                learned = load_learned_head(head_path, layer_ids=layer_ids, targets=picked.targets)
            host.enable_ram_prefetch(
                picked.targets,
                None,
                None,
                top_k=picked.top_k,
                per_token=per_token,
                per_layer=per_layer,
                cores=cores,
                top_k_only=top_k_only,
                candidates=candidates,
                min_margin=min_margin,
                promote=promote,
                pool_serve=pool_serve,
                **idle,
            )
            targets = picked.targets
            result = GpuScorer(picked, candidates, per_token, top_k_only, learned_head=learned, per_layer=per_layer)
            if learned is not None:
                from sglang.srt.layers.moe.ram_prefetch_learned_head import describe

                logger.info("exl3 RAM miss prefetch: %s", describe(learned))
        else:
            # Off node 0, which the tier keeps near full (DSV41_REFERENCE.md section 33.12).
            node = numa.plans[-1].node if numa.nodes > 1 else None
            tables = prefetch_tables(layer_ids, registered_gates(), hidden=hidden, node=node, horizon=horizon)
            host.enable_ram_prefetch(
                tables.targets,
                tables.gates,
                tables.bias,
                top_k=tables.top_k,
                per_token=per_token,
                per_layer=per_layer,
                cores=cores,
                top_k_only=top_k_only,
                promote=promote,
                pool_serve=pool_serve,
                **idle,
            )
            targets, result = tables.targets, None
        logger.info(
            "exl3 RAM miss prefetch: %d of %d rows target %s, %s scorer, %d per token, %d per layer%s, "
            "cores %s, %s, promote landed rows %s",
            int((targets[:, 0] >= 0).sum()),
            len(layer_ids),
            "the next layer" if horizon == 1 else f"layer +{horizon}",
            scorer.upper(),
            per_token,
            per_layer,
            (", predicted top-k only" if top_k_only else "") + f", margin floors {floors or 'none'}",
            [plan.spec for plan in numa.plans],
            (
                f"idle-drive reads (own reader per group, deadline {idle['idle_deadline_s'] * 1e3:g} ms, "
                f"{'spread ties' if idle['idle_spread'] else 'row on one drive'}, "
                f"{idle['idle_pieces']} piece{'s' if idle['idle_pieces'] > 1 else ''} in flight)"
                if idle["idle_drive"]
                else "shared reader (turns with demand)"
            ),
            "on" if promote else "off",
        )
        return result

    def _enable_gpu_drip(self, drip: GpuDripConfig, host, spec_gpu, *, copy_engine: bool) -> None:
        """SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS (host/gpu_drip.h): the copy thread drips the GPU scorer's picks of
        RAM-resident experts into VRAM scratch slots, allocated at the first attach. Before the service thread
        starts."""
        if not copy_engine:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS needs SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine, "
                "which copies the drips"
            )
        if spec_gpu is None:
            raise RuntimeError(
                "exl3 RAM miss: SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS needs SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH with "
                "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu, whose scoring picks the drips"
            )
        host.enable_gpu_drip(
            drip.slots,
            picks=drip.picks,
            chunk_bytes=drip.chunk_kb * 1024,
            targets=spec_gpu.picked.targets[:, 0].tolist(),
            candidates=spec_gpu.candidates,
        )
        self._drip = (drip.slots, drip.picks)
        logger.info(
            "exl3 RAM miss GPU drip: %d scratch slots, %d picks per scored record, %d KiB chunks",
            drip.slots,
            drip.picks,
            drip.chunk_kb,
        )

    def _drip_segments(self, row: int, segments):
        """C1s's segments for ``row``: from the drip's scratch tensors to the row's destinations. The first row
        allocates the scratch, ``slots`` rows of each destination's row shape, and registers it with the host; every
        later row must have the same shapes, since the slots serve every row."""
        slots, _ = self._drip
        destinations = [destination for _, destination in segments.pairs]
        if self._drip_scratch is None:
            self._drip_scratch = [
                torch.empty((slots,) + tuple(d.shape[1:]), dtype=d.dtype, device=d.device) for d in destinations
            ]
            self.host.set_drip_scratch(self._drip_scratch, self._drip_table)
        shapes = [(tuple(d.shape[1:]), d.dtype) for d in destinations]
        if shapes != [(tuple(s.shape[1:]), s.dtype) for s in self._drip_scratch]:
            raise ValueError(f"exl3 RAM miss GPU drip: row {row}'s destinations {shapes} differ from the scratch's")
        return expert_row_segments(list(zip(self._drip_scratch, destinations)))

    def before_host_use(self) -> None:
        """Begin eager pinned-tier use: drain queued device work, then pause the thread.

        Nesting is counted; only the outermost call pauses. The pause is bounded by
        ``2 x timeout + 1 s``, which the watchdog's limit exceeds (``watchdog_wait_s``).
        """
        self.ensure_started()
        if self._pause_depth == 0:
            if self.gpu_hot_enabled:
                self._hot_snapshot.copy_(
                    self._gpu_hot_updater.slot_to_expert, non_blocking=True
                )
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                with (
                    torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_stream_sync")
                    if self._sync_wait_nvtx
                    else nullcontext()
                ):
                    torch.cuda.current_stream().synchronize()
            with (
                torch.cuda.nvtx.range("dsv41.ram_miss.before_host_use_thread_pause")
                if self._sync_wait_nvtx
                else nullcontext()
            ):
                self.host.pause(self.config.stream.copy.eager_pause_bound_s())
            if self.gpu_hot_enabled:
                self._refresh_hot_lists()
        self._pause_depth += 1

    def after_host_use(self) -> None:
        """End eager pinned-tier use; the outermost call applies map changes."""
        self._pause_depth -= 1
        if self._pause_depth == 0:
            # The eager paths' map changes reach the device before the service runs
            # again, on the stream every later post follows
            # (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Deltas and the bulk delta").
            bulk = self.host.take_bulk_delta()
            # With no device map yet the list is dropped: a new device side starts from
            # a snapshot (attach).
            if self.device_side is not None and bulk.numel():
                self.device_side.map_bulk_apply(bulk)
            self.host.resume()

    def attach(self, manager, streamer) -> None:
        """The format's ``attach_hot_cache_manager``: hooks once, a backend per layer.

        The first call also creates the device side and seeds its slot map from the
        host's. Requires DIRECT residency on the manager. A layer without a graph pinned
        tier is left alone.
        """
        self.ensure_started()
        if self._manager is None:
            updater = getattr(manager, "gpu_residency", None)
            if updater is None or not updater.insert_direct:
                raise RuntimeError(
                    "exl3 RAM miss: the service needs DIRECT residency (SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE=1, "
                    "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE=2): every record carries its hot set, and the CPU experts' "
                    "miss lanes are ordered by its ranking"
                )
            self._manager = manager
            manager.register_fail_stop_check(self.fail_stop_check)
            self._enable_gpu_hot(updater)
            if self.cpu_experts is not None:
                updater.enable_miss_order()
        if not getattr(streamer, "_graph_pinned_tier", False):
            return
        if streamer.layer_id in pinned_full_layers(config=self.config.stream):
            self._fill_full_layer(streamer)
        width = streamer.graph_miss_width
        if width > self.lanes:
            # The post kernel requests min(count, lanes) lanes and traps on a wider
            # plan. A verify's misses take lanes, not its routes.
            raise ValueError(
                f"exl3 RAM miss: layer {streamer.layer_id} gathers up to {streamer.graph_gather_rows} rows "
                f"({width} miss lanes) per call but the service requests at most {self.lanes} lanes"
            )
        cache = streamer.hot_cache
        if self.cpu_experts is None and getattr(manager, "lane_head", None) is not None:
            raise ValueError(
                "SGLANG_OPT_LANE_HEAD_PATH orders the CPU experts' miss lanes; this launch has no CPU experts "
                "(SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS)"
            )
        if self.cpu_experts is not None:
            # The fused plan sorts this layer's miss lanes by residency key, highest
            # first, and the CPU lanes are the tail, so the lanes worth inserting are
            # the ones copied.
            direct = manager.gpu_residency
            lane_head = getattr(manager, "lane_head", None)
            if lane_head is not None:
                # SGLANG_OPT_LANE_HEAD_PATH: the head writes this layer's keys at each gather, before the plan.
                from sglang.srt.layers.moe.lane_head import LaneHeadGather

                streamer._plan_miss_keys = lane_head.keys[streamer.residency_row]
                streamer.lane_head_hook = LaneHeadGather(lane_head, direct, streamer.residency_row, streamer)
            else:
                streamer._plan_miss_keys = direct.miss_keys[streamer.residency_row]
        if self.device_side is None:
            stream = self.config.stream
            split = stream.cpu_experts.split
            miss_cut, miss_cut_max = split_miss_cut(
                split.miss_cut, split.miss_cut_max, cpu_experts=stream.cpu_experts.enabled
            )
            self.device_side = ExpertStreamDevice(
                self.page,
                # The host allocates the lease block; the device reads the same one.
                self.host.lease_block,
                layout=self.build_key,
                device=cache.device,
                layers=len(self._rows),
                experts=self.host.experts,
                timeout_ms=stream.copy.timeout_ms,
                piece_runs=self.host.piece_runs(),
                row_capacities=[int(c) for c in self.host.tables.capacity],
                hot_page=self.hot_page,
                lease_pdl=self.lease_pdl,
                hit_copy=stream.copy.ram_hit_copy,
                cpu_misses=stream.cpu_experts.misses,
                miss_cut=miss_cut,
                miss_cut_max=miss_cut_max,
                lanes=self.lanes,
                nodes=self.wire.nodes,
                split_model=self._split_model_words(),
                config=stream,
            )
            if device_stamps_enabled(stream):
                self.device_side.enable_device_stamps()
            if self.cpu_experts is not None:
                self.device_side.enable_cpu_experts(
                    self.cpu_experts.x_rows, self.cpu_experts.out_rows
                )
                self.cpu_experts.attach_device(self.device_side)
            if self._spec_gpu is not None:
                spec = self._spec_gpu
                self.device_side.enable_spec_scorer(
                    spec.candidates, top_k=spec.picked.top_k, per_token=spec.per_token, top_k_only=spec.top_k_only
                )
                if spec.learned_head is not None:
                    from sglang.srt.layers.moe.ram_prefetch_learned_head import serving_head

                    head = serving_head(
                        spec.learned_head, device=cache.device, top_k=spec.picked.top_k, picks=spec.per_layer
                    )
                    self.device_side.enable_spec_head(head)
            if self._drip is not None:
                slots, picks = self._drip
                self._drip_table = torch.zeros(DRIP_TABLE_SLOTS, dtype=torch.int64, device=cache.device)
                self.device_side.enable_gpu_drip(slots, table=self._drip_table, picks=picks)
            # The device's map starts empty: every row the tier maps now reaches it as
            # one bulk of entries, taken paused. The bulk list before this point was
            # dropped (after_host_use), so the snapshot is the source.
            self.before_host_use()
            try:
                self.host.take_bulk_delta()
                snapshot = [
                    (row, expert, slot)
                    for row in range(len(self._rows))
                    for expert, slot in enumerate(self.host.mapping(row))
                    if slot >= 0
                ]
                if snapshot:
                    self.device_side.map_bulk_apply(
                        torch.tensor(snapshot, dtype=torch.int32)
                    )
            finally:
                self.after_host_use()
            if self._stages_traced:
                self._start_route_log(cache.device)
            if self.copy_engine:
                from sglang.srt.eplb.expert_distribution import (
                    get_global_expert_distribution_recorder,
                )

                get_global_expert_distribution_recorder().register_pre_forward_observer(
                    self._copy_engine_barrier
                )
                self._install_copy_engine_module_load_guard()
        self.routed_rows_per_step += streamer.graph_gather_rows
        row = self.row_of(streamer.layer_id)
        # The host staged staging_for(capacity) slots for the row at start. A post requests
        # at most the miss width (a verify's clamp keeps its count there), not the routes; under spill, the victim
        # lanes (only live misses stage).
        victims = getattr(manager.gpu_residency, "victim_lanes", None)
        want = max(1, min(streamer.graph_miss_width, victims) if victims else streamer.graph_miss_width)
        staged = self.staging_for(int(self.host.tables.capacity[row]))
        if staged < want:
            logger.warning(
                "exl3 RAM miss: row %d stages %d slots, not %d: its tier has %d",
                row,
                staged,
                want,
                int(self.host.tables.capacity[row]),
            )
        if self.route_log is not None:
            self.route_log.bind(row, streamer.layer_id, cache.capacity)
        previous = streamer.row_backend
        sm_mask, copy_sm = 0, None
        if self.copy_engine:
            if list(previous.segments) != [streamer.row_tag]:
                raise ValueError(
                    f"exl3 RAM miss copy engine: layer {streamer.layer_id} has copy tables {list(previous.segments)}"
                )
            segments = previous.segments[streamer.row_tag]
            if self.sm_small_copies:
                # The pinned tier's copy table has one row per host source, in the
                # sources' order.
                names = list(streamer._graph_sources)
                if len(names) != segments.table.shape[0]:
                    raise ValueError(
                        f"exl3 RAM miss: layer {streamer.layer_id}'s copy table does not match its sources {names}"
                    )
                sm_mask = sm_copy_mask(names, self.lanes, build=self.build_key)
                copy_sm = sm_copy_table(segments, sm_mask) if sm_mask else None
        streamer.row_backend = ExpertEngineRowBackend(
            previous.segments,
            previous.host_row_map,
            self.device_side,
            row,
            streamer.graph_gather_rows,
            {
                tag: stream_segment_map(segments, self.host.tables, row)
                for tag, segments in previous.segments.items()
            },
            hot_slots=manager.gpu_residency.slot_to_expert[
                manager.gpu_residency.layer_ids.index(streamer.layer_id)
            ],
            hot_capacity=cache.capacity,
            route_log=self.route_log,
            copy_engine=self.copy_engine,
            copy_sm_table=copy_sm,
            cpu_experts=self.cpu_experts is not None,
            streamer_of=self.tables[streamer.layer_id].streamer_of,
        )
        if self.route_log is not None and self.route_log.residency is not None:
            from sglang.srt.layers.moe.residency_hook import residency_hook_for

            streamer.row_backend.residency_hook = residency_hook_for(
                row,
                layer_ids=sorted(self._rows, key=self._rows.__getitem__),
                width=streamer.graph_gather_rows,
                topk=streamer.layer.top_k,
                device=cache.device,
            )
        if self._spec_gpu is not None:
            self._note_spec_row(row, streamer.row_backend)
        if streamer.layer_id in self._layer_splits:
            self.device_side.set_row_split(row, self._layer_splits[streamer.layer_id])
        updater = manager.gpu_residency
        if self.cpu_experts is not None and getattr(updater, "victim_lanes", width) < width:
            # Spill: forced CPU misses land in RAM victims (RamTier::reserve_victims_locked), so the layer must have
            # room for them; the post flags this layer itself only before the copy engine arms.
            self._check_spill_room(row, streamer, width)
            residency_row = streamer.residency_row
            streamer.row_backend.spill = (
                updater.overflow_flag,
                updater.gather_overflow[residency_row : residency_row + 1],
            )
        if self.copy_engine:
            dst_rows = min(
                int(destination.shape[0]) for _, destination in segments.pairs
            )
            if self._drip is not None:
                # The scratch is registered before the row's copy table, which names the drip's sources.
                streamer.row_backend.drip_segments = self._drip_segments(row, segments)
            self.host.set_copy_table(row, segments.table, dst_rows, sm_mask=sm_mask)
            self.device_side.set_row_copy(row, dst_rows)

    def _note_spec_row(self, row: int, backend) -> None:
        """attach's half of the GPU scorer's table: records the row's VRAM hot slots and rebinds every attached row."""
        self._spec_hot[row] = (backend.hot_slots, backend.hot_capacity)
        self._row_backends[row] = backend
        self._bind_spec_scores()

    def _bind_spec_scores(self) -> None:
        """Gives each attached row with a target its SpecScoreRow once the target row attached. ModelRunner attaches
        every row (maybe_init_expert_hot_cache) before it captures the decode graph, which freezes the entries."""
        from sglang.srt.layers.moe.ram_prefetch import spec_score_rows

        picked = self._spec_gpu.picked
        rows = spec_score_rows(picked, self._spec_hot)
        for row, backend in self._row_backends.items():
            backend.spec_expected = int(picked.targets[row, 0]) >= 0
            backend.spec_score = rows.get(row)

    def _start_route_log(self, device) -> None:
        """Trace runs only: log every graph forward's routes in a ``GraphRouteLog``.

        Entries are stamped with the scheduler's forward identity and, when the updater
        owns them, the GPU residency bank's hot sets.
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        router_prefix = self.config.stream.trace.router_capture_path
        # The router rings hold ~400 KB per forward, so they get a shallower ring: 32
        # deep is ~13 MB of VRAM.
        log = GraphRouteLog(
            len(self._rows), self.lanes, device, depth=32 if router_prefix else 64
        )
        if router_prefix:
            log.enable_router(router_prefix)
        if self.config.stream.trace.residency_capture:
            from sglang.srt.layers.moe.ram_prefetch import registered_gates
            from sglang.srt.layers.moe.residency_capture import residency_rows

            # ensure_started refused the switch without a router prefix.
            layer_ids = sorted(self._rows, key=self._rows.__getitem__)
            log.enable_residency(
                router_prefix + ".res",
                rows=residency_rows(layer_ids, gates=registered_gates(), strata=self.profile.strata),
            )
        if self._gpu_hot_updater is not None:
            log.bind_hot(
                self._gpu_hot_updater.slot_to_expert, self._gpu_hot_updater.layer_ids
            )
        trace = get_exl3_stream_trace()
        trace.graph_seq_source = log.read_seq
        trace.forward_meta_source = log.current_meta
        get_global_expert_distribution_recorder().register_pre_forward_observer(
            log.on_pre_forward
        )
        self.route_log = log

    def _refresh_hot_lists(self) -> None:
        """Rebuild each layer's hot-expert list from the host copy of the GPU map."""
        updater = self._gpu_hot_updater
        for row, layer_id in enumerate(updater.layer_ids):
            capacity = updater.caches[row].capacity
            self._hot_lists[layer_id] = [
                int(e) for e in self._hot_snapshot[row, :capacity].tolist() if e >= 0
            ]

    def _enable_gpu_hot(self, updater) -> None:
        """Adopt the updater's GPU residency as the hot set and push it to every row."""
        self._gpu_hot_updater = updater
        self._hot_snapshot = torch.empty_like(
            updater.slot_to_expert,
            device="cpu",
            pin_memory=updater.device.type == "cuda",
        )
        self._hot_snapshot.copy_(updater.slot_to_expert, non_blocking=True)
        if updater.device.type == "cuda":
            torch.cuda.current_stream(updater.device).synchronize()
        self._refresh_hot_lists()
        self.before_host_use()
        try:
            for layer_id, experts in self._hot_lists.items():
                self.host.set_hot(self.row_of(layer_id), experts)
        finally:
            self.after_host_use()
        self.gpu_hot_enabled = True

    def hot_experts(self, layer_id: int) -> list[int]:
        """The layer's GPU-resident experts as of the last pause."""
        return self._hot_lists.get(layer_id, [])

    def fail_stop_check(self) -> None:
        """The service's per-batch upkeep, run from the scheduler's fail-stop hook.

        Retunes and logs CPU experts, and drains the trace. It does not arm the copy engine: that waits for the
        scheduler to be idle after its startup warm-up (``arm_copy_engine_when_quiet``). A failed request never
        reaches here: the host aborts and the device traps (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Fail-stop").
        """
        if self.host is None or self._shut_down:
            return
        if self.cpu_experts is not None and self._cpu_retune_batches > 0:
            self._cpu_batches += 1
            if self._cpu_batches >= self._cpu_retune_batches:
                self._cpu_batches = 0
                self.cpu_experts.retune()
        if self._split_model_refiner is not None and self._split_model_refine_batches > 0:
            self._split_model_batches += 1
            if self._split_model_batches >= self._split_model_refine_batches:
                self._split_model_batches = 0
                self._refine_split_model()
        if self.cpu_experts is not None:
            self._cpu_log_batches += 1
            if self._cpu_log_batches >= CPU_STATS_LOG_BATCHES:
                self._cpu_log_batches = 0
                self.cpu_experts.log_stats()
        self._trace_step()
        self._trace_stages()

    def _size_prefill_share(self, tables, spec_share: int) -> int:
        """The prefill share's rows per layer: PREFILL_SHARE_FRACTION of the smallest admitting row's usable rows
        (full layers map every expert and admit nothing), or SGLANG_EXPERT_STREAM_FORCE_PREFILL_SHARE_ROWS when set.
        Logs its inputs."""
        full = pinned_full_layers(config=self.config.stream)
        usable = {}
        for row, layer_id in enumerate(tables.layer_ids):
            if layer_id in full:
                continue
            capacity = int(tables.capacity[row])
            ranges = [group[row] for group in self._node_ranges] if self._node_ranges is not None else [(0, capacity)]
            usable[int(layer_id)] = usable_rows(ranges, staging_width=self.staging_width(), share=spec_share)
        forced = self.config.stream.copy.prefill.share_rows
        if forced is not None and forced < 1:
            raise ValueError(
                f"exl3 RAM miss: SGLANG_EXPERT_STREAM_FORCE_PREFILL_SHARE_ROWS={forced} must be at least 1"
            )
        if not usable:
            logger.info("exl3 RAM miss prefill share: every layer is a full layer and admits nothing, so no share")
            return 0
        computed = prefill_share_rows(list(usable.values()))
        rows = computed if forced is None else forced
        logger.info(
            "exl3 RAM miss prefill share: %d rows per layer (%s; computed %d = %.2f x the smallest layer's %d usable "
            "rows; usable rows per layer %d-%d over %d admitting layers, staging %d and pool %d slots per group)",
            rows,
            "computed" if forced is None else "SGLANG_EXPERT_STREAM_FORCE_PREFILL_SHARE_ROWS",
            computed,
            PREFILL_SHARE_FRACTION,
            min(usable.values()),
            min(usable.values()),
            max(usable.values()),
            len(usable),
            self.staging_width(),
            spec_share,
        )
        return rows

    def _set_prefill_share(self, forward_pass_id: int, forward_batch) -> None:
        """Pre-forward observer for ``SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE``.

        A prefill's pinned-tier admissions own at most the share sized at start rows per
        layer (then they evict their own); any other forward's own none. The value holds
        until the next forward's observer runs.
        """
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        self.host.set_prefill_share(self._prefill_share_rows if prefill else 0)

    def _copy_engine_barrier(self, forward_pass_id: int, forward_batch) -> None:
        """Pre-forward observer: drain the device before an eager forward once armed.

        An eager forward may load a kernel module, and a load blocks the copy thread's
        ``cuMemcpyAsync`` until the deadline while a decode graph in flight is held in
        its copy wait, so that step must end first. Captured decode forwards are
        counted toward arming. A DSpark verify counts as a captured forward unless the
        manager's graph gather is suspended (the eager re-run of an overflowed verify).
        """
        mode = forward_batch.forward_mode
        # A DSpark verify replays the decode graph unless its eager re-run suspended the graph gather.
        graphed = mode.is_decode() or (
            mode.is_target_verify()
            and not (self._manager is not None and self._manager.graph_gather_suspended)
        )
        if not graphed:
            if self._copy_armed:
                torch.cuda.synchronize()
        elif self.device_side is not None and self.device_side.copy_engine_captured:
            self._copy_decodes += 1

    @property
    def copy_engine_arm_pending(self) -> bool:
        """Whether a started service still has its copy engine to arm."""
        return (
            self.copy_engine and not self._copy_armed and self.device_side is not None and not self._shut_down
        )

    @property
    def copy_decodes(self) -> int:
        """The decode forwards counted toward arming since capture."""
        return self._copy_decodes

    def quiet_probes(self) -> dict[str, Callable[[], bool]]:
        """Host-side busy probes (no device read) for the box going quiet before calibration: the RAM-miss thread,
        the copy engine's jobs, the tier's demand and speculative (RAM prefetch) drive reads, and every disk under the
        expert mirror roots and the Engram table."""
        host = self.host

        def drive_reads() -> bool:
            return any(
                root["demand_inflight_bytes"] or root["spec_inflight_bytes"] for root in host.drive_load()["roots"]
            )

        probes = {
            "ram_miss_thread": lambda: host.busy_episode() != 0,
            "copy_engine_jobs": lambda: not host.copy_engine_idle(0.0),
            "tier_drive_reads": drive_reads,
        }
        dirs = [d for d in self.config.stream.tiers.mirror.dirs.split(os.pathsep) if d]
        dirs += [d for d in (Dsv41Config.from_envs().engram_table_dir,) if d]
        for disk, path in block_device_inflight_paths(dirs).items():
            probes[f"{disk}_inflight"] = functools.partial(inflight_busy, path)
        return probes

    def arm_copy_engine_when_quiet(
        self, *, deadline: float, busy_s: dict[str, float], warmup_ended_s_ago: Optional[float]
    ) -> QuietReport:
        """Arm the copy engine (its CPU split and split model calibrated first) once the box is quiet, or at
        ``deadline`` (time.monotonic()) anyway. The scheduler calls it idle, after its startup warm-up, with at least
        COPY_ENGINE_ARM_DECODES decode forwards counted."""
        torch.cuda.synchronize()
        report = wait_for_quiet(self.quiet_probes(), deadline=deadline, busy_s=busy_s)
        since = (
            "with no startup warm-up" if warmup_ended_s_ago is None else f"{warmup_ended_s_ago:.2f} s after warm-up ended"
        )
        log = logger.info if report.quiet else logger.warning
        log(
            "exl3 RAM miss copy engine calibration starts %s, %d decode forwards since capture, device drained; %s",
            since, self._copy_decodes, report.line(),
        )
        self._arm_copy_engine()
        return report

    def _arm_copy_engine(self) -> None:
        """Arm the copy engine once enough decode forwards have run since capture."""
        if not self.copy_engine or self._copy_armed or self.device_side is None:
            return
        if self._copy_decodes >= COPY_ENGINE_ARM_DECODES:
            if self.cpu_experts is not None:
                self._calibrate_cpu_split()
            self.host.arm_copy_engine()
            self._copy_armed = True
            logger.info(
                "exl3 RAM miss copy engine armed after %d decode forwards since capture",
                self._copy_decodes,
            )

    def _calibrate_cpu_split(self) -> None:
        """Calibrate the CPU split before arming, while no CPU or copy lane is typed.

        The device drains first. Under the overlap scheduler this runs while the next
        decode still replays on the forward stream, which ``before_host_use``'s
        current-stream sync does not cover; a paused RAM thread under it would leave a
        miss lane unserved until its device deadline traps.
        """
        torch.cuda.synchronize()
        self.before_host_use()
        try:
            self.cpu_experts.calibrate(torch.cuda.current_device())
            self._calibrate_split_model()
        finally:
            self.after_host_use()

    def _calibrate_split_model(self) -> None:
        """SGLANG_EXPERT_STREAM_CPU_SPLIT's model: measure its costs on the loaded model and publish them
        (split_model_calibration.py), in _calibrate_cpu_split's paused window. Runs whenever the model is on, whatever
        the split's calibration and table say, and not when its model_costs names every cost. A failure warns and
        leaves the model off (its words' SPLIT_MODEL_OFF): the split table types the CPU's lanes."""
        from sglang.srt.layers.moe import split_model_calibration as calib

        if self.device_side is None or self.device_side.split_model.numel() == 0:
            return
        split = self.config.stream.cpu_experts.split
        overrides = split.model_costs
        if split_model_overrides_complete(overrides, self.wire.nodes):
            costs = calib.costs_from_words(split_model_words(self.wire.nodes, overrides), self.wire.nodes)
            logger.info("exl3 RAM miss split model: every cost from SGLANG_EXPERT_STREAM_CPU_SPLIT's model_costs, not "
                        "calibrated: %s", costs.line())
        else:
            services = self.cpu_experts.services
            before = [self.host.cpu_stats(s.group) for s in services]
            try:
                costs, clamped, seconds = calib.calibrate(
                    self.host, services, self.device_side,
                    device=torch.cuda.current_device() if torch.cuda.is_available() else -1,
                    reps=split.calibration_reps, staging_width=self.staging_width(),
                )
            except (calib.SplitModelCalibrationError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
                logger.warning("exl3 RAM miss split model calibration failed (%s): the model stays off and the split "
                               "table types the CPU's lanes", error)
                return
            finally:
                for service, stats in zip(services, before):
                    service.exclude_calibration_stats(stats)
            costs = calib.apply_overrides(costs, overrides)
            logger.info("exl3 RAM miss split model calibrated in %.2f s%s%s: %s", seconds,
                        f", clamped {', '.join(clamped)}" if clamped else "",
                        ", with SGLANG_EXPERT_STREAM_CPU_SPLIT's model_costs" if overrides.strip() else "",
                        costs.line())
        self._publish_split_model(costs)
        self._split_model_refiner = calib.SplitModelRefiner(costs, self._split_model_moments(), overrides)

    def _publish_split_model(self, costs) -> None:
        """Write ``costs`` (SplitModelCosts) and the flags word (SGLANG_EXPERT_STREAM_CPU_SPLIT's
        model_residency_tiebreak) to the device."""
        tiebreak = self.config.stream.cpu_experts.split.model_residency_tiebreak
        self.device_side.set_split_model_costs(costs.words(residency_tiebreak=tiebreak))

    def _split_model_moments(self) -> list:
        """Every node's CPU fit moments (cpu_experts.h fit_moments), in node order."""
        return [self.host.cpu_fit_moments(s.group) for s in self.cpu_experts.services]

    def _refine_split_model(self) -> None:
        """SGLANG_EXPERT_STREAM_CPU_SPLIT's model_refine_batches: refit the CPU costs from the jobs served since the last refit and
        publish them (SplitModelRefiner). A failure is logged and stops the refinement; the published costs stay."""
        refiner = self._split_model_refiner
        try:
            costs = refiner.refine(self._split_model_moments())
            if costs is not None:
                self._publish_split_model(costs)
                logger.info("exl3 RAM miss split model refined (%d): %s", refiner.refines, costs.line())
        except Exception as error:  # noqa: BLE001 - a refit must not stop serving
            logger.warning("exl3 RAM miss split model refinement failed (%r); keeping the published costs", error)
            self._split_model_refiner = None

    def _copy_engine_module_load_guard(self, load):
        """Wrap a module loader (Triton's ``load_binary``, tvm-ffi's ``load_module``).

        Once armed, the device drains before a module loads. ``cuModuleLoadData`` takes
        the driver lock that the copy thread's ``cuMemcpyAsync`` needs, then waits for
        the device, whose stream may be held in a copy wait for exactly that copy. That
        deadlock ends in a fail-stop at the copy-wait timeout (observed: the scheduler
        in ``loadBinary`` -> ``cuModuleLoadData`` and the copy thread in
        ``cuMemcpyAsync`` on the lock, for 20 s). Draining first is safe because no load
        holds the lock yet.
        """

        def guarded(*args, **kwargs):
            if self._copy_armed and not torch.cuda.is_current_stream_capturing():
                self.copy_engine_module_loads += 1
                torch.cuda.synchronize()
            return load(*args, **kwargs)

        return guarded

    def _install_copy_engine_module_load_guard(self) -> None:
        """Guard tvm-ffi's ``load_module`` and Triton's ``load_binary``.

        tvm-ffi loads every JIT library (sglang's ``load_jit`` and flashinfer's), and
        under EAGER its ``dlopen`` loads the library's kernels there and then: the same
        hazard as a Triton load. Both callers look ``load_module`` up on the module at
        call time, so wrapping the attribute covers them.
        """
        try:
            import tvm_ffi
        except ImportError:
            tvm_ffi = None
        if tvm_ffi is not None:
            tvm_ffi.load_module = self._copy_engine_module_load_guard(
                tvm_ffi.load_module
            )
        try:
            from triton.runtime import driver
        except ImportError:
            return
        utils = driver.active.utils
        utils.load_binary = self._copy_engine_module_load_guard(utils.load_binary)

    def _trace_stages(self) -> None:
        """Move the stage records produced since the last check into the trace."""
        if not self._stages_traced:
            return
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        get_exl3_stream_trace().record_ram_miss_requests(
            self.host.drain_trace(), sorted(self._rows, key=self._rows.__getitem__)
        )
        dropped = self.host.trace_dropped()
        if dropped > self._stages_dropped:
            logger.warning(
                "exl3 RAM miss: %d stage records dropped (ring full)",
                dropped - self._stages_dropped,
            )
            self._stages_dropped = dropped

    def _graph_rows(
        self, rows: list[int], *, final: bool = False
    ) -> Optional[tuple[list[int], list[int]]]:
        """Poll a pinned graph-counter readback and queue the next one without waiting.

        The manager's registers survive the forward observer's zeroing of each
        streamer's graph counters. Cumulative snapshots may coalesce while the sole
        pinned slot is in flight. ``final`` runs only after shutdown's GPU barrier; it
        drains the slot and takes the last register value.
        """
        registers = getattr(self._manager, "_registers", None) or {}
        decode = registers.get("decode")
        if decode is None:
            return None
        source = decode["graph_rows"]
        if not source.is_cuda:
            return [int(value) for value in source.sum(dim=0).tolist()], rows

        if not final and torch.cuda.is_current_stream_capturing():
            return None
        ready = None
        if self._trace_pending_rows is not None:
            if final:
                self._trace_event.synchronize()
            elif not self._trace_event.query():
                return None
            ready = (
                [int(value) for value in self._trace_snapshot.tolist()],
                self._trace_pending_rows,
            )
            self._trace_pending_rows = None
        if not final:
            if self._trace_snapshot is None:
                self._trace_snapshot = torch.empty(
                    2, dtype=torch.int64, pin_memory=True
                )
                self._trace_event = torch.cuda.Event(enable_timing=False)
            self._trace_snapshot.copy_(source.sum(dim=0), non_blocking=True)
            self._trace_event.record(torch.cuda.current_stream(source.device))
            self._trace_pending_rows = rows
        return ready

    def _record_graph_snapshot(self, trace, graph: list[int], rows: list[int]) -> None:
        """Record the step delta against the previous snapshot, then keep this one."""
        # A register reset (discard_graph_capture_routes) moves the totals back:
        # re-baseline.
        if self._trace_graph is not None and graph[0] > self._trace_graph[0]:
            routed = graph[0] - self._trace_graph[0]
            # A lagged read can fold multiple bs-1 decode graph steps into one line.
            steps = (
                max(1, round(routed / self.routed_rows_per_step))
                if self.routed_rows_per_step
                else 1
            )
            trace.record_graph_step(
                layer_rows_delta=[a - b for a, b in zip(rows, self._trace_rows)],
                routed_rows=routed,
                routed_misses=graph[1] - self._trace_graph[1],
                thread=self.host.counters(),
                steps=steps,
            )
        self._trace_rows, self._trace_graph = rows, graph

    def _trace_step(self, *, final: bool = False) -> None:
        """Record one graph decode step's gathers and RAM misses (trace runs only)."""
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        trace = get_exl3_stream_trace()
        if not trace.enabled:
            return
        if self.route_log is not None:
            self.route_log.poll(trace, final=final)
        rows = self.host.layer_rows()
        sample = self._graph_rows(rows, final=final)
        if sample is not None:
            self._record_graph_snapshot(trace, *sample)
        if final and self._trace_snapshot is not None:
            # The orderly shutdown barrier already finished all graph work. This last
            # copy captures registers updated after the prior snapshot.
            registers = self._manager._registers["decode"]
            final_graph = [
                int(value) for value in registers["graph_rows"].sum(dim=0).tolist()
            ]
            self._record_graph_snapshot(trace, final_graph, rows)

    def _cuda_active(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_initialized()

    def _synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def _barrier_devices(self) -> list[torch.device]:
        """The CUDA devices whose kernels may read the slabs, each with an index.

        Chosen on the calling thread: a helper thread starts on device 0, so a
        device-less synchronize there waits for the wrong device on any rank that serves
        another GPU (and would resolve an index-less ``cuda`` to device 0 the same way).
        """
        candidates = []
        if self.device_side is not None:
            candidates.append(self.device_side.state.device)
        for layer_id in sorted(self.tables):
            tier = getattr(
                self.tables[layer_id].streamer_of(), "pinned_host_cache", None
            )
            if tier is not None:
                candidates.append(tier.device)
        devices: list[torch.device] = []
        for device in candidates:
            if device.type != "cuda":
                continue
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device not in devices:
                devices.append(device)
        return devices or [torch.device("cuda", torch.cuda.current_device())]

    def _completion_deadline_s(self) -> float:
        # The miss stream is bounded by its deadline (a trap), the copy wait by the
        # watchdog's copy-wait timeout (an abort).
        return self.config.stream.copy.completion_deadline_s()

    def _establish_gpu_completion(self) -> Optional[str]:
        """None when every GPU reader is known to have finished, else the reason not.

        The device-wide synchronize runs on a helper thread with a deadline: it is not
        interruptible, so a hung kernel would otherwise hang shutdown with no way to say
        "uncertain".
        """
        if not self._cuda_active():
            return None
        outcome: dict = {}
        devices = self._barrier_devices()

        def run() -> None:
            try:
                for device in devices:
                    self._synchronize(device)
                outcome["done"] = True
            except BaseException as error:  # noqa: BLE001
                # Any CUDA error means completion is not established.
                outcome["error"] = error

        deadline = self._completion_deadline_s()
        helper = threading.Thread(target=run, daemon=True, name="exl3-shutdown-sync")
        helper.start()
        helper.join(deadline)
        if helper.is_alive():
            return f"the device synchronize did not return within {deadline:.1f} s"
        if "error" in outcome:
            return f"the device synchronize failed: {outcome['error']!r}"
        return None

    def _log_draft_stats(self) -> None:
        """The shared DSpark draft host's counters, read before ``host.stop()`` closes the handle (its ``stats()`` is
        ``{}`` afterwards): a both-mode server has no other place that reports them."""
        draft = getattr(self.host, "_draft_keep", None)
        if draft is None:
            return
        try:
            logger.info("DSpark CPU experts: %s", draft[1].stats())
        except Exception as error:  # noqa: BLE001 - a diagnostic must not change the shutdown's outcome
            logger.warning("exl3 RAM miss: reading the DSpark draft's counters failed: %r", error)

    def _split_model_words(self) -> Optional[list[int]]:
        """SGLANG_EXPERT_STREAM_CPU_SPLIT's model: its first cost words for the device, or None when it is off or there
        is no CPU split to replace. They hold the model off (split_model_off_words) until _calibrate_split_model
        publishes its costs, as the copy engine arms; before that the device types no CPU lane anyway."""
        if not self.config.stream.cpu_experts.split.model or self.cpu_experts is None:
            return None
        logger.info("exl3 RAM miss split model on: its costs are calibrated when the copy engine arms")
        return split_model_off_words(self.wire.nodes)

    def _log_split_model_stats(self) -> None:
        """The split model's device counters (SGLANG_EXPERT_STREAM_CPU_SPLIT's model), after the GPU barrier; nothing when
        it is off. A failure is logged, not raised."""
        try:
            stats = self.device_side.split_model_stats()
        except Exception as error:  # noqa: BLE001 - a diagnostic must not change the shutdown's outcome
            logger.warning("exl3 RAM miss: reading the split model's counters failed: %r", error)
            return
        if stats is not None:
            logger.info("exl3 RAM miss split model: %s", json.dumps(stats))

    def _dump_device_stamps(self) -> None:
        """Write the chain's device stamps beside the job trace, as ``<prefix>.<pid>.device-stamps.jsonl``. A failure is
        logged, not raised: it must not turn an orderly shutdown into a quarantine."""
        prefix = self.config.stream.trace.job.prefix
        if not prefix:
            return
        path = f"{prefix}.{os.getpid()}.device-stamps.jsonl"
        try:
            count = self.device_side.dump_device_stamps(path)
        except Exception:  # noqa: BLE001
            logger.exception("exl3 RAM miss: writing the device stamps to %s failed", path)
            return
        logger.info("exl3 RAM miss: %d device-stamped records written to %s", count, path)

    def shutdown(self, *, at_exit: bool = False) -> None:
        """Shut the service down: free the tiers if safe, else quarantine them.

        Idempotent; the tiers must not be used afterwards. The sequence is a GPU barrier
        proving no reader runs while the service still serves, then close admission,
        stop the thread and free. Anything that leaves it unknown whether the service
        thread or a GPU reader still runs (the barrier, or stopping the thread) makes
        the shutdown quarantine instead (analysis/dsv41-drive/LEASE_PROTOCOL.md,
        "Shutdown").

        The barrier runs before admission closes: an in-flight chain waits on the
        service, so closing first would leave it spinning until the miss stream's
        deadline traps. ``_shut_down`` is set first, so no new forward starts.

        ``at_exit`` is the exit hook's path: no device barrier is attempted in an exit
        handler, so the tiers are quarantined unconditionally. The scheduler's graceful
        shutdown takes the orderly path through ``shutdown_expert_engine_service``.
        """
        if self._completed or (self._shut_down and not at_exit):
            return
        # At exit a shutdown that started and did not complete is finished as a
        # quarantine, never left half done.
        uncertain: Optional[str] = (
            "an earlier shutdown did not complete" if self._shut_down else None
        )
        self._shut_down = True
        # Each counted load drained the device first; 0 means every module loaded before the copy engine armed.
        logger.info(
            "exl3 RAM miss: copy_engine_module_loads=%d (module loads after the copy engine armed; armed: %s)",
            self.copy_engine_module_loads,
            self._copy_armed,
        )
        stop_error: Optional[BaseException] = None
        # A KeyboardInterrupt or SystemExit: re-raised once the slabs are safe.
        interrupt: Optional[BaseException] = None
        try:
            if self.host is not None:
                if uncertain is None:
                    uncertain = (
                        "process exit: no device barrier is attempted"
                        if at_exit
                        else self._establish_gpu_completion()
                    )
                if uncertain is None and self.device_side is not None:
                    self._dump_device_stamps()
                    self._log_split_model_stats()
                self.host.close_admission()
                if uncertain is None and self._stages_traced:
                    # host.stop() closes its native handle, so freeze the reader between
                    # requests before taking its final demand counters. The GPU barrier
                    # above has retired every demand and lease.
                    self.host.pause(self.config.stream.copy.eager_pause_bound_s())
                    self._trace_step(final=True)
        except BaseException as error:  # noqa: BLE001
            # A failure even to close admission leaves the state uncertain.
            uncertain = f"closing admission or the barrier failed: {error!r}"
            if not isinstance(error, Exception):
                interrupt = error
        finally:
            try:
                # The service thread writes into the slabs through raw addresses, so it
                # stops before anything is freed. A thread hung in a read ends this in
                # the service watchdog's abort.
                if self.host is not None:
                    self._log_draft_stats()
                    logger.info(
                        "exl3 RAM miss: stopping the service thread; a read that hangs ends in the watchdog's abort "
                        "after max(30 s, 3 x the wait timeout)"
                    )
                    self.host.stop()
            except BaseException as error:  # noqa: BLE001
                # A thread that may still run must not have its slabs freed.
                stop_error = error
                uncertain = (
                    uncertain or f"stopping the service thread failed: {error!r}"
                )
                if not isinstance(error, Exception):
                    interrupt = interrupt or error
            finally:
                if uncertain is None:
                    for layer_id in sorted(self.tables):
                        streamer = self.tables[layer_id].streamer_of()
                        tier = getattr(streamer, "pinned_host_cache", None)
                        if tier is not None:
                            tier.close()
                else:
                    self._quarantine(uncertain)
                self._completed = True
        if interrupt is not None:
            # KeyboardInterrupt and SystemExit go on, after the quarantine.
            raise interrupt
        if stop_error is not None:
            logger.error(
                "exl3 RAM miss: stopping the service thread failed: %r", stop_error
            )

    def _quarantine(self, reason: str) -> None:
        """Keep everything a GPU kernel may still read or write alive until exit.

        Frees nothing: the tiers' slabs, the request page, the slot map, the lease block
        and every device buffer of the chain are handed to ``quarantine_host_slabs``.
        """
        logger.error(
            "exl3 RAM miss: shutdown could not establish that no GPU reader is running (%s); the pinned slabs, the "
            "request page, the slot map, the lease block and the device buffers are quarantined until process exit",
            reason,
        )
        owned: list[torch.Tensor] = []
        if self.host is not None:
            owned += [self.host.page, self.host.slot_map, self.host.lease_block]
            if self.host.hot_page is not None:
                owned.append(self.host.hot_page)
        if self._spec_gpu is not None:
            owned.append(self._spec_gpu.candidates)  # the select kernel writes it through UVA
        if self.device_side is not None:
            side = self.device_side
            owned += [
                side.state,
                side.go_1,
                side.host_rows_1,
                side.dst_slots_1,
                side.lane_kind,
                side.lane_slot,
                side.lane_node,
                side.ce_mask,
                side.cpu_lanes,
                side.piece_runs,
                *side.map_bank.values(),
            ]
            if side.spec_scores is not None:
                owned.append(side.spec_scores)
        if self._trace_snapshot is not None:
            # A copy can still be writing this pinned block when the device barrier
            # failed.
            owned.append(self._trace_snapshot)
        if self._hot_snapshot is not None:
            owned.append(self._hot_snapshot)
        if self.route_log is not None:
            owned += self.route_log.tensors()
        if self._gpu_hot_updater is not None:
            owned.append(self._gpu_hot_updater.slot_to_expert)
        for layer_id in sorted(self.tables):
            streamer = self.tables[layer_id].streamer_of()
            tier = getattr(streamer, "pinned_host_cache", None)
            if tier is not None:
                tier.quarantine()
            backend = getattr(streamer, "row_backend", None)
            for name in (
                "host_rows",
                "ram_miss",
                "keep",
                "routes",
                "planned",
                "hot_slots",
            ):
                tensor = getattr(backend, name, None)
                if isinstance(tensor, torch.Tensor):
                    owned.append(tensor)
        quarantine_host_slabs(owned)
        self._quarantined = True


def shutdown_expert_engine_service() -> None:
    """The scheduler's graceful-shutdown entry: shut the service down if one exists.

    A no-op when no service was ever created, so a run without expert streaming does not construct
    the singleton at shutdown.
    """
    service = ExpertEngineService._instance
    if service is not None:
        service.shutdown()
