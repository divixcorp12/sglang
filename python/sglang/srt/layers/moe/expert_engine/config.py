"""The expert engine's configuration: one frozen ``msgspec.Struct`` tree, resolved once per process from the env.

``ExpertStreamConfig`` holds every quant- and model-agnostic knob, ``Dsv41ProfileConfig`` and ``MimoV26ProfileConfig``
the model profiles' own, and ``Exl3FormatConfig`` the EXL3-only ones. This module is the only engine module that reads
``envs``; everything downstream takes a config object (``docs/hx/F145-env-merge-table.draft.md`` is the binding table).

A struct reads its fields from the env names in its ``_ENV`` map. A struct with a ``_GROUP_ENV`` is one JSON-valued env
instead: the object is decoded into the struct, an unknown key is an error naming the env and the key, and a field the
object leaves out keeps the default below, which is today's ``Envs`` default. Group keys are the field names of the
struct (a nested struct is a nested object), for example::

    SGLANG_EXPERT_STREAM_URING='{"mode": "sqpoll", "queue_depth": 256}'
    SGLANG_EXPERT_STREAM_TRACE='{"job": {"prefix": "/tmp/job"}, "draft_trigger": {"forward_us": 5000}}'

``CopyMode`` and ``PullMode`` are string-valued enums so the env, the JSON and the error messages use the same words.
"""

from __future__ import annotations

import enum
import os
import typing
from typing import Any, ClassVar, Optional

import msgspec

from sglang.srt.environ import InsertOnMissStage, envs


class CopyMode(enum.Enum):
    """What moves a RAM-resident expert row into its VRAM slot.

    ``SM_COPY`` is the in-graph SM copy kernel (the legacy streamer's ``COPY_BACKEND=gpu``, and the native engine
    with its copy engine off). ``DMA`` is the legacy streamer's CUDA copy engine on a copy stream
    (``COPY_BACKEND=dma``). ``COPY_ENGINE`` is the native engine's own copy thread (``cuMemcpyAsync``), whose
    completion the captured decode graph waits for.
    """

    SM_COPY = "sm_copy"
    DMA = "dma"
    COPY_ENGINE = "copy_engine"


class PullMode(enum.Enum):
    """How the shadow prefetch delivers its pull; the legacy boolean maps ``true`` to ALWAYS and ``false`` to OFF."""

    OFF = "off"
    COUNT_ZERO = "count_zero"
    ALWAYS = "always"


def _choice(env_name: str, value: str, allowed: tuple[str, ...], *, key: str = "", words: bool = False) -> None:
    """Refuse a value outside ``allowed``, naming the env the user sets (and the group key inside a group env).

    ``words`` phrases the allowed values as "ce or sm" instead of "one of 'ce', 'sm'".
    """
    if value in allowed:
        return
    label = f"{env_name} {key}" if key else env_name
    if words:
        raise ValueError(f"{label} must be {' or '.join(allowed)}, got {value!r}")
    raise ValueError(f"{label} must be one of {', '.join(map(repr, allowed))}, got {value!r}")


def _coerce(env_name: str, value: Any, annotation: Any) -> Any:
    """An env's parsed value as the field's type: enums by value, a layer list as ints."""
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        if isinstance(value, annotation):
            return value
        try:
            if issubclass(annotation, enum.IntEnum):
                return annotation(int(value))
            return annotation(str(value).strip().lower())
        except ValueError:
            names = ", ".join(str(m.value) for m in annotation)
            raise ValueError(f"{env_name}={value!r}: expected one of {names}") from None
    if typing.get_origin(annotation) is tuple and typing.get_args(annotation) == (int, Ellipsis):
        try:
            return tuple(int(item) for item in value)
        except ValueError:
            raise ValueError(f"{env_name}={','.join(value)!r}: expected comma-separated integers") from None
    return value


class _Config(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Base of every config struct: ``from_envs`` and the registry of the env names it reads."""

    _ENV: ClassVar[dict[str, str]] = {}
    _GROUP_ENV: ClassVar[Optional[str]] = None

    @classmethod
    def from_envs(cls):
        if cls._GROUP_ENV is not None:
            return _decode_group(cls._GROUP_ENV, getattr(envs, cls._GROUP_ENV).get(), cls)
        values = {}
        for info in msgspec.structs.fields(cls):
            if info.name in cls._ENV:
                env_name = cls._ENV[info.name]
                values[info.name] = _coerce(env_name, getattr(envs, env_name).get(), info.type)
            elif isinstance(info.type, type) and issubclass(info.type, _Config):
                values[info.name] = info.type.from_envs()
        return cls(**values)


def _decode_group(env_name: str, raw: str, struct_type: type):
    """A group env's JSON object as ``struct_type``; an empty or unset env is every default."""
    if not raw.strip():
        return struct_type()
    try:
        return msgspec.json.decode(raw, type=struct_type, strict=True)
    except (msgspec.DecodeError, ValueError) as error:
        # A member's own refusal already names the env; a decode error ("unknown field ...") does not.
        message = str(error)
        raise ValueError(message if message.startswith(env_name) else f"{env_name}: {message}") from None


def env_fields(config_type: type = None, prefix: str = "") -> dict[str, str]:
    """Every individually-read env name under a struct, mapped to the dotted config path it fills.

    A group env maps to its struct's path. Used by the tests and the removed-name messages.
    """
    config_type = config_type or ExpertStreamConfig
    out: dict[str, str] = {}
    if config_type._GROUP_ENV is not None:
        out[config_type._GROUP_ENV] = prefix.rstrip(".")
        return out
    for info in msgspec.structs.fields(config_type):
        if info.name in config_type._ENV:
            out[config_type._ENV[info.name]] = prefix + info.name
        elif isinstance(info.type, type) and issubclass(info.type, _Config):
            out.update(env_fields(info.type, f"{prefix}{info.name}."))
    return out


# --- tiers ------------------------------------------------------------------------------------------------------


class UringConfig(_Config):
    """io_uring options of the NVMe readers (``SGLANG_EXPERT_STREAM_URING``); the SQPOLL core is in ``numa_cores``."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_URING"

    mode: str = "default"
    queue_depth: int = 0
    fixed_files: bool = False
    read_mode: str = "normal"
    wait_mode: str = "block"
    sq_thread_idle_ms: int = 10000
    diagnostics: bool = False
    read_cuts: str = "auto"
    slab_arena: bool = False

    def __post_init__(self):
        env = self._GROUP_ENV
        _choice(env, self.mode, ("default", "iopoll", "sqpoll", "sqpoll_iopoll"), key="mode")
        _choice(env, self.read_mode, ("normal", "fixed", "readv_fixed"), key="read_mode")
        _choice(env, self.wait_mode, ("block", "spin"), key="wait_mode")
        _choice(env, self.read_cuts, ("auto", "0", "1"), key="read_cuts")
        if not 0 <= self.queue_depth <= 32768:
            raise ValueError(f"{env} queue_depth must be 0..32768, got {self.queue_depth}")
        if self.sq_thread_idle_ms < 0:
            raise ValueError(f"{env} sq_thread_idle_ms must be >= 0, got {self.sq_thread_idle_ms}")


class MirrorConfig(_Config):
    """Mirror roots of the expert rows (``SGLANG_EXPERT_STREAM_MIRROR``): one byte-identical checkpoint copy per drive."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_MIRROR"

    dirs: str = ""
    weights: str = ""
    dynamic: bool = False
    caps: str = ""


class TiersConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "expert_dir": "SGLANG_EXPERT_STREAM_EXPERT_DIR",
        "file_cache_dir": "SGLANG_EXPERT_STREAM_FILE_CACHE_DIR",
        "pinned_host_mb": "SGLANG_EXPERT_STREAM_PINNED_HOST_MB",
        "pinned_host_numa_mb": "SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB",
        "pinned_host_full_layers": "SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS",
        "host_arena": "SGLANG_EXPERT_STREAM_ENABLE_HOST_ARENA",
        "file_reader": "SGLANG_EXPERT_STREAM_FILE_READER",
        "row_source": "SGLANG_EXPERT_STREAM_ROW_SOURCE",
        "numa_cores": "SGLANG_EXPERT_STREAM_NUMA_CORES",
    }

    # The format's read-only source shard directory, and the engine's writable file-cache directory: two knobs.
    expert_dir: str = ""
    file_cache_dir: str = ""
    pinned_host_mb: int = 0
    # "node:MiB,node:MiB", summing to pinned_host_mb.
    pinned_host_numa_mb: str = ""
    pinned_host_full_layers: tuple[int, ...] = ()
    host_arena: bool = False
    file_reader: str = "mmap"
    row_source: str = "auto"
    # Per-node thread plans, "node:ram=..,cpu=..,sq=..;node:...". Also takes the cores the retired
    # CPU_EXPERTS_CORES, RAM_MISS_SPIN_CORE and URING_SQ_THREAD_CPU named.
    numa_cores: str = ""
    uring: UringConfig = msgspec.field(default_factory=UringConfig)
    mirror: MirrorConfig = msgspec.field(default_factory=MirrorConfig)


# --- copy path --------------------------------------------------------------------------------------------------


class PrefillConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "fills": "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS",
        "share": "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE",
        "share_rows": "SGLANG_EXPERT_STREAM_FORCE_PREFILL_SHARE_ROWS",
        "route_plan": "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_ROUTE_PLAN",
        "split_gather": "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SPLIT_GATHER",
    }

    fills: bool = False
    share: bool = False
    # An A/B override of the share's computed size, in rows per layer; None uses the computed size.
    share_rows: Optional[int] = None
    route_plan: bool = False
    split_gather: bool = False


class GpuDripConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "slots": "SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS",
        "picks": "SGLANG_EXPERT_STREAM_GPU_DRIP_PICKS",
        "chunk_kb": "SGLANG_EXPERT_STREAM_GPU_DRIP_CHUNK_KB",
    }

    slots: int = 0
    picks: int = 4
    chunk_kb: int = 1024


class CopyConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "mode": "SGLANG_EXPERT_STREAM_COPY_MODE",
        "ram_hit_copy": "SGLANG_EXPERT_STREAM_RAM_HIT_COPY",
        "sm_small_copies": "SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES",
        "lease_pdl": "SGLANG_EXPERT_STREAM_ENABLE_LEASE_PDL",
        "timeout_ms": "SGLANG_EXPERT_STREAM_RAM_MISS_TIMEOUT_MS",
        "idle_spin_us": "SGLANG_EXPERT_STREAM_RAM_MISS_IDLE_SPIN_US",
        "copy_idle_spin_us": "SGLANG_EXPERT_STREAM_COPY_IDLE_SPIN_US",
        "graph_gather": "SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER",
        "fused_plan": "SGLANG_EXPERT_STREAM_ENABLE_FUSED_PLAN",
        "graph_gather_scratch_rows": "SGLANG_EXPERT_STREAM_GRAPH_GATHER_SCRATCH_ROWS",
        "graph_gather_miss_lanes": "SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES",
        "graph_gather_victim_lanes": "SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES",
    }

    mode: CopyMode = CopyMode.SM_COPY
    # What moves a captured decode post's RAM hits once the copy engine is armed: "ce" (its DMA) or "sm" (the SM copy).
    ram_hit_copy: str = "ce"
    sm_small_copies: bool = False
    lease_pdl: bool = False
    # How long the in-graph wait for the RAM-miss thread may take per MoE layer before the process fails stop.
    timeout_ms: int = 2000
    idle_spin_us: int = 5000
    copy_idle_spin_us: int = 5000
    graph_gather: bool = False
    fused_plan: bool = False
    graph_gather_scratch_rows: int = 0
    graph_gather_miss_lanes: int = 0
    graph_gather_victim_lanes: int = 0
    prefill: PrefillConfig = msgspec.field(default_factory=PrefillConfig)
    gpu_drip: GpuDripConfig = msgspec.field(default_factory=GpuDripConfig)

    def __post_init__(self):
        _choice(self._ENV["ram_hit_copy"], self.ram_hit_copy, ("ce", "sm"), words=True)

    # The RAM-miss timeout T feeds four limits that were written out inline in several places; they are defined once.

    def watchdog_wait_s(self) -> float:
        """The C++ watchdog's abort limit, ``max(30 s, 3 x T)``.

        It must outlast the miss stream's deadline (a slow demand traps the device at T) and the eager pause bound
        of ``2 x T + 1 s``. Three times T exceeds both for any T over 1 s; 30 s covers the rest.
        """
        return max(30.0, 3.0 * self.timeout_ms / 1000)

    def eager_pause_bound_s(self) -> float:
        """How long the host thread may be paused before an eager host use: ``2 x T + 1 s``."""
        return 2 * self.timeout_ms / 1000 + 1.0

    def completion_deadline_s(self) -> float:
        """The miss stream's completion deadline: ``T + 5 s``."""
        return self.timeout_ms / 1000 + 5.0

    def fill_timeout_s(self) -> float:
        """How long a prefill fill waits for its rows: the watchdog limit plus 5 s."""
        return self.watchdog_wait_s() + 5.0


# --- hot cache and residency ------------------------------------------------------------------------------------


class HotCacheConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "gpu_mb": "SGLANG_EXPERT_STREAM_HOT_GPU_MB",
        "seed_path": "SGLANG_EXPERT_STREAM_HOT_SEED",
        "log_interval": "SGLANG_EXPERT_STREAM_HOT_LOG_INTERVAL",
    }

    gpu_mb: int = 0
    # A startup residency JSON, {"count": [[...]]}.
    seed_path: str = ""
    log_interval: int = 100


class ResidencyConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "dynamic": "SGLANG_EXPERT_STREAM_ENABLE_HOT_DYNAMIC",
        "update_prefill_tokens": "SGLANG_EXPERT_STREAM_HOT_UPDATE_PREFILL_TOKENS",
        "update_decode_forwards": "SGLANG_EXPERT_STREAM_HOT_UPDATE_DECODE_FORWARDS",
        "min_residence_forwards": "SGLANG_EXPERT_STREAM_HOT_MIN_RESIDENCE_FORWARDS",
        "benefit_ratio": "SGLANG_EXPERT_STREAM_HOT_BENEFIT_RATIO",
        "decay_tokens": "SGLANG_EXPERT_STREAM_HOT_DECAY_TOKENS",
        "promotion_sigmas": "SGLANG_EXPERT_STREAM_HOT_PROMOTION_SIGMAS",
        "async_promotions": "SGLANG_EXPERT_STREAM_ENABLE_HOT_ASYNC_PROMOTIONS",
        "async_scores": "SGLANG_EXPERT_STREAM_ENABLE_ASYNC_RESIDENCY_SCORES",
        "gpu_update": "SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE",
        "gpu_max_promotions": "SGLANG_EXPERT_STREAM_GPU_RESIDENCY_MAX_PROMOTIONS",
        "insert_on_miss_stage": "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE",
        "insert_on_miss_decay": "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_DECAY",
        "fused_insert": "SGLANG_EXPERT_STREAM_ENABLE_HOT_FUSED_INSERT",
    }

    dynamic: bool = False
    update_prefill_tokens: int = 1024
    update_decode_forwards: int = 0
    min_residence_forwards: int = 8
    benefit_ratio: float = 1.0
    decay_tokens: int = 0
    promotion_sigmas: float = 0.0
    async_promotions: bool = False
    async_scores: bool = False
    gpu_update: bool = False
    gpu_max_promotions: int = 64
    insert_on_miss_stage: InsertOnMissStage = InsertOnMissStage.OFF
    insert_on_miss_decay: float = 0.98
    fused_insert: bool = False


# --- prefetch ---------------------------------------------------------------------------------------------------


class RamPrefetchConfig(_Config):
    """The NVMe-to-RAM prefetch."""

    _ENV: ClassVar[dict[str, str]] = {
        "enabled": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH",
        "per_token": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_TOKEN",
        "per_layer": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_LAYER",
        "spec_share": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE",
        "top_k_only": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_TOP_K_ONLY",
        "scorer": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER",
        "idle_drive": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE",
        "idle_deadline_us": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_DEADLINE_US",
        "idle_spread": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_SPREAD",
        "idle_pieces": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES",
        "promote": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_PROMOTE",
        "pool_serve": "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_POOL_SERVE",
        "horizon": "SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON",
    }

    enabled: bool = False
    per_token: int = 1
    per_layer: int = 1
    spec_share: int = 2
    top_k_only: bool = False
    # Where the target layer's gate is scored: "cpu" (the group's speculative thread) or "gpu" (in the decode graph).
    scorer: str = "gpu"
    idle_drive: bool = False
    idle_deadline_us: int = 4000
    idle_spread: bool = False
    idle_pieces: int = 1
    promote: bool = False
    pool_serve: bool = False
    horizon: int = 1

    def __post_init__(self):
        _choice(self._ENV["scorer"], self.scorer, ("cpu", "gpu"))


class ShadowPrefetchConfig(_Config):
    """In-graph prefetch candidate scoring and the shadow predictors (``SGLANG_EXPERT_STREAM_SHADOW_PREFETCH``)."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_SHADOW_PREFETCH"

    # "", "llapor" or "apex".
    predictor: str = ""
    model_dir: str = ""
    candidates: int = 16
    budget: int = 3
    apex_tau: float = 0.95
    pull_mode: PullMode = PullMode.OFF
    shadow_recall: bool = True
    fused_top1: bool = True
    calibration: bool = False
    calibration_file: str = ""
    calibration_provenance: str = ""
    # Registered shadow predictors, scored against the native routes; empty installs nothing.
    predictors: tuple[str, ...] = ()
    predictor_candidates: int = 16
    score_interval: int = 1
    max_rows: int = 0

    def __post_init__(self):
        if not self.predictor and self.pull_mode is not PullMode.OFF:
            raise ValueError(
                f"{self._GROUP_ENV} pull_mode={self.pull_mode.value!r} needs a predictor ('llapor' or 'apex')"
            )


class PrefetchConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "legacy_max_candidates": "SGLANG_EXPERT_STREAM_LEGACY_PREFETCH_MAX_CANDIDATES",
    }

    ram: RamPrefetchConfig = msgspec.field(default_factory=RamPrefetchConfig)
    # The older GPU-side next-layer prefetch; mutually exclusive with the shadow predictor.
    legacy_max_candidates: int = 0
    shadow: ShadowPrefetchConfig = msgspec.field(default_factory=ShadowPrefetchConfig)


# --- CPU experts and the draft ----------------------------------------------------------------------------------


class CpuSplitConfig(_Config):
    """The CPU/copy split of a layer's lanes (``SGLANG_EXPERT_STREAM_CPU_SPLIT``)."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_CPU_SPLIT"

    # CPU lanes per n resident lanes, n = 0..8, as 9 comma-separated counts; empty computes them from the costs.
    table: str = ""
    # "LAYERS:TABLE;LAYERS:TABLE", replacing the group's table on the listed layers.
    layer_tables: str = ""
    # ms per expert on the CPU pool, per resident row over the link, and per layer handoff; used with calibration off.
    cpu_ms: float = 0.52
    link_ms: float = 1.0
    handoff_ms: float = 0.02
    retune_batches: int = 0
    calibration: bool = True
    calibration_reps: int = 10
    miss_cut: int = 1
    miss_cut_max: int = 3
    model: bool = False
    # name[.node]=ms pairs replacing the calibrated costs of the split model.
    model_costs: str = ""
    model_refine_batches: int = 0
    model_residency_tiebreak: bool = False


class CpuTwoStageConfig(_Config):
    """A two-stage CPU miss (``SGLANG_EXPERT_STREAM_CPU_TWO_STAGE``): the layout's first-stage span is read first."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_CPU_TWO_STAGE"

    enabled: bool = False
    # The row's two spans divided over the mirror roots by these weights, in MIRROR's weight format.
    w13_weights: str = ""
    w2_weights: str = ""


class CpuExpertsConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "enabled": "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS",
        "threads": "SGLANG_EXPERT_STREAM_CPU_EXPERTS_THREADS",
        "smt": "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_SMT",
        "misses": "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_MISSES",
        "keep_warm_us": "SGLANG_EXPERT_STREAM_CPU_EXPERTS_KEEP_WARM_US",
        "store_out": "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT",
        "max_isa": "SGLANG_EXPERT_STREAM_CPU_MAX_ISA",
        "report_isa": "SGLANG_EXPERT_STREAM_CPU_REPORT_ISA",
    }

    enabled: bool = False
    # Worker threads per NUMA node; 0 takes one per core.
    threads: int = 0
    smt: bool = False
    misses: bool = False
    keep_warm_us: int = 2000
    store_out: bool = False
    # Caps the kernel's ISA tier (scalar, avx2, bw, vnni, vbmi); empty keeps the host's.
    max_isa: str = ""
    report_isa: bool = False
    split: CpuSplitConfig = msgspec.field(default_factory=CpuSplitConfig)
    two_stage: CpuTwoStageConfig = msgspec.field(default_factory=CpuTwoStageConfig)


class DraftConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "cpu_experts": "SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS",
        "cores": "SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_CORES",
        "threads": "SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_THREADS",
        "resident_path": "SGLANG_EXPERT_STREAM_DRAFT_RESIDENT_PATH",
    }

    cpu_experts: bool = False
    cores: str = ""
    threads: int = 0
    resident_path: str = ""


# --- trace (one JSON env) ---------------------------------------------------------------------------------------


class PredictorCaptureConfig(_Config):
    dir: str = ""
    max_gb: int = 500
    shard_rows: int = 4096
    frames: int = 2


class JobTraceConfig(_Config):
    # A path prefix; empty is off.
    prefix: str = ""
    capacity: int = 131072
    resources: bool = False


class DraftTriggerConfig(_Config):
    # Thresholds in us, 0 off: reasons 2 (arrival), 0 (pending) and 1 (forward).
    arrival_us: int = 0
    pending_us: int = 0
    forward_us: int = 0
    report_prefix: str = ""
    trigger_only: bool = False


class CpuWorkerTraceConfig(_Config):
    prefix: str = ""
    capacity: int = 65536
    min_us: int = 6000
    scratch_prefix: str = ""


class PreforwardReadinessConfig(_Config):
    enabled: bool = False
    path: str = ""


class RouteTraceConfig(_Config):
    dir: str = ""
    max_tokens: int = 2048
    speculative: bool = False


class TraceConfig(_Config):
    """Diagnostics and captures (``SGLANG_EXPERT_STREAM_TRACE``): all off by default; a trace changes the JIT build key."""

    _GROUP_ENV: ClassVar[Optional[str]] = "SGLANG_EXPERT_STREAM_TRACE"

    hot_metrics_file: str = ""
    predictor_log_interval: int = 100
    predictor_metrics_file: str = ""
    predictor_capture: PredictorCaptureConfig = msgspec.field(default_factory=PredictorCaptureConfig)
    draft_routes_path: str = ""
    expert_path: str = ""
    router_capture_path: str = ""
    residency_capture: bool = False
    verify_accept_path: str = ""
    job: JobTraceConfig = msgspec.field(default_factory=JobTraceConfig)
    draft_trigger: DraftTriggerConfig = msgspec.field(default_factory=DraftTriggerConfig)
    # A four-byte mmap file admitting the trace.
    gate_path: str = ""
    cpu_worker: CpuWorkerTraceConfig = msgspec.field(default_factory=CpuWorkerTraceConfig)
    sync_wait_nvtx: bool = False
    preforward_readiness: PreforwardReadinessConfig = msgspec.field(default_factory=PreforwardReadinessConfig)
    route: RouteTraceConfig = msgspec.field(default_factory=RouteTraceConfig)


# --- the tree ---------------------------------------------------------------------------------------------------


class ExpertStreamConfig(_Config):
    """Every quant- and model-agnostic engine knob, resolved once per process with ``from_envs``."""

    _ENV: ClassVar[dict[str, str]] = {"enabled": "SGLANG_EXPERT_STREAM_ENABLE"}

    enabled: bool = False
    tiers: TiersConfig = msgspec.field(default_factory=TiersConfig)
    copy: CopyConfig = msgspec.field(default_factory=CopyConfig)
    hot_cache: HotCacheConfig = msgspec.field(default_factory=HotCacheConfig)
    residency: ResidencyConfig = msgspec.field(default_factory=ResidencyConfig)
    prefetch: PrefetchConfig = msgspec.field(default_factory=PrefetchConfig)
    cpu_experts: CpuExpertsConfig = msgspec.field(default_factory=CpuExpertsConfig)
    draft: DraftConfig = msgspec.field(default_factory=DraftConfig)
    trace: TraceConfig = msgspec.field(default_factory=TraceConfig)


# --- model profiles and formats ---------------------------------------------------------------------------------


class Dsv41RamPrefetchConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "margin_floors_path": "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS",
        "learned_head_path": "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD",
    }

    # None is the packaged table (fit for horizon 1), "" no floors, a path a floors file.
    margin_floors_path: Optional[str] = None
    # A folded learned L+2 head replacing the native candidate select; None or "" is the native select.
    learned_head_path: Optional[str] = None


class Dsv41TraceConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {"draft_top_prob_path": "SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH"}

    draft_top_prob_path: str = ""


class Dsv41ProfileConfig(_Config):
    """DSV4.1-only knobs (``Dsv41Profile.Config``)."""

    _ENV: ClassVar[dict[str, str]] = {"moe_side_stream": "SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM"}

    moe_side_stream: bool = False
    ram_prefetch: Dsv41RamPrefetchConfig = msgspec.field(default_factory=Dsv41RamPrefetchConfig)
    # False keeps the EXL3 DSpark draft eager (the env is SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH; A/B only).
    dspark_draft_graph: bool = True
    trace: Dsv41TraceConfig = msgspec.field(default_factory=Dsv41TraceConfig)

    @classmethod
    def from_envs(cls) -> "Dsv41ProfileConfig":
        return cls(
            moe_side_stream=envs.SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM.get(),
            ram_prefetch=Dsv41RamPrefetchConfig.from_envs(),
            dspark_draft_graph=not envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.get(),
            trace=Dsv41TraceConfig.from_envs(),
        )


class MimoV26ProfileConfig(_Config):
    """MiMo V2.6 Pro's own knobs: none yet."""


class Exl3BuildConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "src": "SGLANG_EXL3_SRC",
        "dir": "SGLANG_EXL3_BUILD_DIR",
        "cxx": "SGLANG_EXL3_CPU_CXX",
    }

    # The exllamav3 checkout the extension is JIT-built from; required to use EXL3.
    src: str = ""
    dir: str = "~/.cache/sglang/exl3_ext"
    # The optimized CPU kernel's compiler, which must be GCC 15; empty uses CXX.
    cxx: str = ""


class Exl3CpuConfig(_Config):
    _ENV: ClassVar[dict[str, str]] = {
        "act_residual": "SGLANG_EXL3_CPU_ACT_RESIDUAL",
        "act_block": "SGLANG_EXL3_CPU_ACT_BLOCK",
        "max_m": "SGLANG_EXL3_CPU_MAX_M",
        "row_weighted": "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT",
        "pin": "SGLANG_EXL3_MOE_CPU_PIN",
        "small_workers": "SGLANG_EXL3_MOE_CPU_SMALL_WORKERS",
    }

    act_residual: bool = False
    act_block: int = 0
    max_m: int = 16
    # "", "draft", "target" or "both".
    row_weighted: str = ""
    # None leaves the vendored kernel's own default, which pins workers to the first cores; the engine needs "0".
    pin: Optional[str] = None
    small_workers: int = 0


class Exl3FormatConfig(_Config):
    """EXL3-only knobs (``Exl3Format.Config``): the extension build and the CPU kernel."""

    build: Exl3BuildConfig = msgspec.field(default_factory=Exl3BuildConfig)
    cpu: Exl3CpuConfig = msgspec.field(default_factory=Exl3CpuConfig)


# --- the process's resolved config ------------------------------------------------------------------------------


class ResolvedConfig(msgspec.Struct, frozen=True):
    """Everything one process resolves from the env: the generic engine, the DSV4.1 profile and the EXL3 format."""

    stream: ExpertStreamConfig
    dsv41: Dsv41ProfileConfig
    exl3: Exl3FormatConfig

    @classmethod
    def from_envs(cls) -> "ResolvedConfig":
        return cls(
            stream=ExpertStreamConfig.from_envs(),
            dsv41=Dsv41ProfileConfig.from_envs(),
            exl3=Exl3FormatConfig.from_envs(),
        )


_resolved: Optional[tuple[tuple, ResolvedConfig]] = None


def _env_names() -> tuple[str, ...]:
    names: dict[str, None] = {}
    for config_type in (ExpertStreamConfig, Dsv41ProfileConfig, Exl3FormatConfig):
        names.update(dict.fromkeys(env_fields(config_type)))
    return tuple(names)


def resolved_config() -> ResolvedConfig:
    """The config of this process, resolved from the env on first use and handed to every component.

    The env names are read once and the result is reused for as long as none of them changes, so a launcher's late
    edit or a test's ``envs.X.override`` is observed on the next call and a steady process never resolves twice.
    """
    global _resolved
    snapshot = tuple(os.environ.get(name) for name in _env_names())
    if _resolved is None or _resolved[0] != snapshot:
        _resolved = (snapshot, ResolvedConfig.from_envs())
    return _resolved[1]


def stream_config() -> ExpertStreamConfig:
    """The generic engine config of this process (``resolved_config().stream``)."""
    return resolved_config().stream
