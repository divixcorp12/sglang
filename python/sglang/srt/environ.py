import base64
import functools
import json
import os
import warnings
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Dict, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


_NON_UTF8_PREFIX = "base64:"


def _default_cache_subdir(name: str) -> str:
    """A directory under SGLANG_CACHE_DIR, for env defaults that track it.

    Pass as a callable default: SGLANG_CACHE_DIR is declared further down the
    Envs body, and resolving late also lets tests override it.
    """
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), name)


def _default_tree_cache_sanity_check() -> bool:
    """Enable the expensive tree-cache sanity check by default in CI."""
    return envs.SGLANG_IS_IN_CI.get()


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any, secret: bool = False):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False
        self.secret = secret

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvJSON(EnvField):
    def parse(self, value: str | None) -> list | dict | None:
        if not value:
            return None
        if os.path.exists(value):
            with open(value) as f:
                return json.load(f)
        return json.loads(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvPrefetchPullMode(EnvField):
    """Resolve the prefetch-pull delivery mode and its temporary boolean alias."""

    _MODES = frozenset({"off", "count_zero", "always"})

    def __init__(self, legacy_name: str):
        super().__init__("off")
        self.legacy_name = legacy_name

    def parse(self, value: str) -> str:
        mode = value.lower()
        if mode not in self._MODES:
            raise ValueError(
                f'"{value}" is not a valid pull mode; expected one of "off", "count_zero", or "always"'
            )
        return mode

    def get(self) -> str:
        mode_value = os.getenv(self.name)
        legacy_value = os.getenv(self.legacy_name)
        if mode_value is None:
            if legacy_value is None:
                return "off"
            try:
                return "always" if EnvBool(False).parse(legacy_value) else "off"
            except ValueError as error:
                raise ValueError(f"{self.legacy_name}: {error}") from error
        try:
            mode = self.parse(mode_value)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from error
        if legacy_value is None:
            return mode
        try:
            legacy_mode = "always" if EnvBool(False).parse(legacy_value) else "off"
        except ValueError as error:
            raise ValueError(f"{self.legacy_name}: {error}") from error
        if mode != legacy_mode:
            raise ValueError(
                f"{self.name}={mode!r} contradicts {self.legacy_name}={legacy_value!r}; "
                "set only one, or use matching off/false or always/true values"
            )
        return mode


class EnvPrefetchPredictor(EnvStr):
    """Reject a configured pull mode before ModelRunner skips prediction setup."""

    def get(self) -> str:
        predictor = super().get()
        if not predictor and envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off":
            raise ValueError(
                "SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE needs SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR"
            )
        return predictor


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str, secret: bool = False):
        super().__init__(default, secret=secret)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class GateGemvMode(IntEnum):
    """Small-batch Inkling gate linear implementation.

    OFF: always the cublas GEMM
    PAIR: PDL-chained GEMV and gate JIT kernels
    FUSED: single-launch GEMV + gate epilogue (last-block ticket)
    """

    OFF = 0
    PAIR = 1
    FUSED = 2


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class InvariantCheckLevel(IntEnum):
    """Signal level for value/index validity checks (see invariants.py).

    OFF: data layer only (sanitize/containment); no detection, no signal.
    WARN: detect + throttled log/count; degrade, never crash (prod on-demand).
    STRICT: detect + crash on GUARD/FATAL violations (CI default).

    The data layer is unconditional and independent of this level; only the
    detection + signal layer is gated here.
    """

    OFF = 0
    WARN = 1
    STRICT = 2


class DsparkFoldedSampling(IntEnum):
    """Sampling support in the graph-folded DSpark draft proposal: OFF =
    greedy-only folding, AUTO = on when its buffers fit in free GPU memory,
    FORCE = always."""

    OFF = 0
    AUTO = 1
    FORCE = 2


class InsertOnMissStage(IntEnum):
    """How a decode boundary gives a missed expert residency.

    OFF promotes host rows at the boundary, as before. SCRATCH copies the
    previous forward's misses device-to-device out of the graph gather's
    scratch rows into their slots. DIRECT lands each miss copy straight in a
    victim slot chosen from a shortlist the previous boundary ranked, so the
    device-to-device hop disappears and the scratch rows return to the cache.

    The values are ordered so that the retired boolean
    ``SGLANG_MOE_HOT_INSERT_ON_MISS`` keeps its meaning: ``0`` is OFF and ``1``
    is SCRATCH, the behaviour that boolean selected.
    """

    OFF = 0
    SCRATCH = 1
    DIRECT = 2


class Envs:
    # Organization principles for this registry:
    # - Put every field in exactly one topical section. Prefer an existing
    #   section; add a new one only when no current section is a clear fit.
    # - Group by the behavior and owning call sites, not by name similarity
    #   alone. Keep closely related lifecycle or feature knobs adjacent.
    # - Keep each section focused and below 30 fields. Split growing sections
    #   by subsystem or lifecycle instead of creating catch-all groups.
    # - Order broad runtime subsystems before shared storage and backends; keep
    #   platform- and model-specific integrations in dedicated later sections.
    # - Use the same three-line section header everywhere; do not add ad hoc
    #   one-line headings or append unrelated fields at the end of a section.
    # - Keep vendor-specific aliases with their owning integration, and keep
    #   test/debug knobs with the feature or test workflow they exercise.
    # - Keep explanatory comments attached to their field when moving it.
    # - For organization-only changes, AST-check that field names, descriptor
    #   types, and defaults are unchanged and that only field order moved.

    # ===================================================================
    # Runtime configuration and process identity
    # ===================================================================
    # Per-role config-namespace bookkeeping: off / record / enforce (value is
    # validated fail-loud in runtime_context, which resolves it once at import
    # so the read stays dynamo-prunable).
    SGLANG_ROLE_NAMESPACES = EnvStr("off")
    # Record mode: append each newly observed (role, namespace) pair to this
    # file so the audit survives signal-killed workers.
    SGLANG_ROLE_NAMESPACES_OUT = EnvStr(None)
    IS_H200 = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)

    # ===================================================================
    # Model configuration, discovery, and weight loading
    # ===================================================================
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    # Shard the Qwen4-Exp PLE n-gram embedding within each attention-TP group
    # instead of gathering DP tokens for a global-TP lookup.
    SGLANG_USE_ATTN_TP_NGRAM = EnvBool(False)
    # Stream ModelOpt NVFP4 routed experts from host memory; required for hot caching.
    SGLANG_MOE_EXPERT_STREAM = EnvBool(False)
    SGLANG_MOE_EXPERT_FILE_DIR = EnvStr("")
    SGLANG_MOE_PINNED_HOST_MB = EnvInt(0)
    # Where the pinned host tier lives: "node:MiB,node:MiB" (e.g. "0:65536,1:30720"),
    # summing to SGLANG_MOE_PINNED_HOST_MB. Every layer's rows are bound to the nodes
    # in that proportion, and startup refuses a node without the memory. Empty keeps
    # first-touch placement.
    SGLANG_MOE_PINNED_HOST_NUMA_MB = EnvStr("")
    SGLANG_MOE_EXPERT_COPY_BACKEND = EnvStr("gpu")
    SGLANG_MOE_HOT_GPU_MB = EnvInt(0)
    SGLANG_MOE_HOT_SEED = EnvStr("")
    SGLANG_MOE_HOT_DYNAMIC = EnvBool(False)
    SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS = EnvInt(1024)
    # Also update dynamic residency every this many decode forwards; 0 leaves
    # updates to qualifying prefills only.
    SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS = EnvInt(0)
    SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS = EnvInt(8)
    SGLANG_MOE_HOT_BENEFIT_RATIO = EnvFloat(1.0)
    # Decay residency scores by 0.95 once per this many routed tokens instead
    # of once per boundary; 0 keeps per-boundary decay.
    SGLANG_MOE_HOT_DECAY_TOKENS = EnvInt(0)
    # Standard deviations of routing-count noise a candidate expert must lead
    # the expert it replaces by, on top of SGLANG_MOE_HOT_BENEFIT_RATIO.
    SGLANG_MOE_HOT_PROMOTION_SIGMAS = EnvFloat(0.0)
    # Return from a residency boundary once promotion copies are submitted and
    # publish the promoted slots on a later forward after the copies complete.
    SGLANG_MOE_HOT_ASYNC_PROMOTIONS = EnvBool(False)
    # Copy CPU-policy scores to a pinned host buffer at residency boundaries
    # and apply decisions only after a nonblocking CUDA event query succeeds.
    SGLANG_MOE_ASYNC_RESIDENCY_SCORES = EnvBool(False)
    # Run the decode residency update as device ops inside the captured decode
    # graph; requires SGLANG_MOE_HOT_DYNAMIC and SGLANG_MOE_EXPERT_GRAPH_GATHER.
    SGLANG_MOE_GPU_RESIDENCY_UPDATE = EnvBool(False)
    # Most experts one layer promotes at a decode boundary on the GPU path.
    SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS = EnvInt(64)
    # Decode boundaries serve misses out of residency instead of promoting host
    # rows; see InsertOnMissStage. Requires SGLANG_MOE_GPU_RESIDENCY_UPDATE and
    # SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1. The deprecated boolean
    # SGLANG_MOE_HOT_INSERT_ON_MISS still selects stage 1 (SCRATCH), because the
    # enum keeps that boolean's 0/1 values.
    SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE = EnvIntWithAlias(
        InsertOnMissStage.OFF, deprecated_name="SGLANG_MOE_HOT_INSERT_ON_MISS"
    )
    # Per-token decay of the scores that pick insert-on-miss victims.
    SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY = EnvFloat(0.98)
    # Run stage 1's boundary insert copies through the fused masked Triton
    # kernel instead of index_select + index_copy_. Byte-identical output; one
    # pass instead of two, and idle lanes move no bytes. Off by default until a
    # serving arm accepts it, so the measured stage-1 config stays reproducible.
    SGLANG_MOE_HOT_FUSED_INSERT = EnvBool(False)
    # Serve graph-gather miss copies through the doorbell copier thread: the
    # gather posts its miss plan and waits in-graph, falling back to the
    # in-graph copy on timeout. Requires SGLANG_MOE_EXPERT_GRAPH_GATHER.
    SGLANG_MOE_EXPERT_DOORBELL = EnvBool(False)
    SGLANG_MOE_EXPERT_DOORBELL_CPU = EnvInt(71)
    # Wait budgets in polls (about 250 ns each); 0 sizes them from the largest
    # per-layer miss copy.
    SGLANG_MOE_EXPERT_DOORBELL_TIMEOUT_POLLS = EnvInt(0)
    SGLANG_MOE_EXPERT_DOORBELL_DEGRADED_POLLS = EnvInt(0)
    # Polls a timed-out resolve of a request the thread committed to drains
    # before the doorbell is disabled for good (the resolve then keeps waiting
    # for that request's copies); 0 uses about 2 s.
    SGLANG_MOE_EXPERT_DOORBELL_DRAIN_POLLS = EnvInt(0)
    # Which layer a doorbell plan targets: "current" posts and resolves layer
    # L's misses inside layer L; "next_layer" is reserved for prediction.
    SGLANG_MOE_EXPERT_DOORBELL_MODE = EnvStr("current")
    # Seconds a disabled drain may wait for a committed copy before the
    # doorbell watchdog aborts the process (a crash instead of a hang).
    SGLANG_MOE_EXPERT_DOORBELL_FATAL_WAIT_S = EnvFloat(30.0)
    # Static plan capacity per target layer; 0 uses the layer's scratch rows.
    SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY = EnvInt(0)
    SGLANG_MOE_HOT_LOG_INTERVAL = EnvInt(100)
    SGLANG_MOE_HOT_METRICS_FILE = EnvStr("")
    SGLANG_MOE_PREFETCH_MAX_CANDIDATES = EnvInt(0)
    # Shadow-score these registered MoE expert predictors against native routes
    # (comma list, e.g. "affinity,popularity"); empty installs no hooks or buffers.
    SGLANG_MOE_EXPERT_PREDICTOR = EnvTuple(tuple())
    SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES = EnvInt(16)
    # Score and update predictors on every Nth eligible forward; raise it to
    # cut the per-step host overhead of shadow scoring.
    SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL = EnvInt(1)
    # Rows per tap buffer; 0 uses decode CUDA-graph max_bs x tokens per request.
    SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS = EnvInt(0)
    SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL = EnvInt(100)
    SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE = EnvStr("")
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR = EnvStr("")
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB = EnvInt(500)
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS = EnvInt(4096)
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES = EnvInt(2)
    # In-graph expert prefetch candidate scoring (shadow until the shared copy layer consumes it): "", "llapor" or "apex".
    SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR = EnvPrefetchPredictor("")
    SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR = EnvStr("")
    # Candidates per target layer kept in the device bank.
    SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES = EnvInt(16)
    # Rows per layer the shadow budget-recall metric credits.
    SGLANG_MOE_EXPERT_PREFETCH_BUDGET = EnvInt(3)
    SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU = EnvFloat(0.95)
    # Post the candidate bank's best non-resident target-layer expert onto the
    # captured side stream (expert_gpu_pull.py) and join it into the covered/
    # residual remap; requires SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR and a hot
    # cache allocated with the trailing DedicatedPrefetchSlot row. Default off:
    # the plan forbids enabling this by default on synthetic evidence.
    SGLANG_MOE_EXPERT_PREFETCH_PULL = EnvBool(False)
    # Delivery-mode replacement for the legacy boolean above. ``count_zero``
    # preserves the captured post/join graph while publishing a no-payload plan.
    SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE = EnvPrefetchPullMode(
        "SGLANG_MOE_EXPERT_PREFETCH_PULL"
    )
    # Keep BudgetRecall enabled by default for compatibility, but let matched
    # performance arms exclude its device work independently of pull telemetry.
    SGLANG_MOE_EXPERT_PREFETCH_SHADOW_RECALL = EnvBool(True)
    # Serving-only BS1 fp32 candidate selection. When disabled (or when any
    # recall/calibration/analysis consumer is active), use the reference
    # stable top-W candidate bank instead.
    SGLANG_MOE_EXPERT_PREFETCH_FUSED_TOP1 = EnvBool(True)
    # Profiling-only graph-captured score/margin histograms. Timed throughput
    # arms leave this false so the observer has no allocation or device work.
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION = EnvBool(False)
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_FILE = EnvStr("")
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_PROVENANCE = EnvStr("")
    # mmap | uring | uring_direct; see sglang.srt.model_loader.file_row_reader.
    SGLANG_QWEN4_PLE_FILE_READER = EnvStr("mmap")
    # Stage file-backed PLE rows before decode CUDA-graph replay instead of
    # breaking the graph for the host read.
    SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY = EnvBool(False)
    # Keep the Qwen4-Exp token embedding in pinned host memory and gather its
    # rows over PCIe; a speculative draft binds the same host table.
    SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING = EnvBool(False)
    SGLANG_MOE_EXPERT_FILE_READER = EnvStr("mmap")
    # Where host expert rows are read from: auto | files | tensor, or a kind the
    # expert format defines. auto keeps each format's default (dense NVFP4 layers:
    # their expert files through io_uring, unless SGLANG_MOE_EXPERT_FILE_READER=mmap).
    SGLANG_MOE_EXPERT_ROW_SOURCE = EnvStr("auto")
    # Mirror roots for EXL3 expert rows, os.pathsep-separated: each holds a
    # byte-identical copy of the checkpoint (one per drive). Non-empty selects the
    # mirror row source, which reads every row from all roots at once; one entry
    # reads everything from that root. Rows are read from the roots, while the
    # layout is still built from SGLANG_DSV41_EXPERT_DIR. Check the copies with
    # scripts/dsv41/verify_expert_mirror.py before trusting them.
    SGLANG_MOE_EXPERT_MIRROR_DIRS = EnvStr("")
    # Colon-separated relative read shares, one per root of
    # SGLANG_MOE_EXPERT_MIRROR_DIRS (a 0 drops a root); empty means equal shares.
    SGLANG_MOE_EXPERT_MIRROR_WEIGHTS = EnvStr("")
    SGLANG_URING_FILE_READER_QUEUE_DEPTH = EnvInt(128)
    # Copy all host expert rows into registered memory (replaces the pinned LRU).
    SGLANG_MOE_EXPERT_HOST_ARENA = EnvBool(False)
    # Serve decode-sized expert gathers without host syncs so CUDA graphs capture
    # them; requires SGLANG_MOE_EXPERT_HOST_ARENA and SGLANG_MOE_HOT_GPU_MB.
    SGLANG_MOE_EXPERT_GRAPH_GATHER = EnvBool(False)
    # Plan a graph-gather's BS1, unique-ID, top_k<=32 routes with one fused
    # CUDA kernel launch instead of the general tensor-op sequence; requires
    # SGLANG_MOE_EXPERT_GRAPH_GATHER. Shapes/dtypes outside that specialization
    # fall back to the general path unchanged.
    SGLANG_MOE_EXPERT_FUSED_PLAN = EnvBool(False)
    # Speculative decoding only: cap each layer's graph-gather scratch rows below
    # one per verify route (decode max_bs x draft tokens x top_k); 0 keeps that bound.
    # Startup rejects a cap below one request's routes: that needs phase 3's overflow path.
    SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS = EnvInt(0)
    # Debug only: write Qwen4-Exp MoE routing tensors of eager decode forwards to
    # this directory (see sglang.srt.models.qwen4_exp_route_trace); empty disables.
    SGLANG_MOE_ROUTE_TRACE_DIR = EnvStr("")
    SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS = EnvInt(2048)
    # Ignored unless SGLANG_MOE_ROUTE_TRACE_DIR is set. Traces target verify forwards
    # and MTP draft hidden states (<dir>/mtp) under eager NEXTN; startup requires
    # --speculative-eagle-topk 1 passed explicitly, --cuda-graph-backend-prefill
    # disabled and --disable-flashinfer-autotune.
    SGLANG_MOE_ROUTE_TRACE_SPECULATIVE = EnvBool(False)
    # Debug only: append per-request speculative verify rows (draft, argmax, accept
    # length, committed tokens, top-2 logit gap, sampling flags) to this JSONL file,
    # suffixed .rank<N> when several ranks run; empty disables and adds no work.
    SGLANG_SPEC_VERIFY_TRACE = EnvStr("")
    # Model path recorded in expert/PLE file-cache identities instead of the
    # resolved --model-path; after moving a checkpoint, set it to the resolved
    # path the caches were built from.
    SGLANG_FILE_CACHE_MODEL_PATH = EnvStr("")
    # Bitwise-exact, shape-guarded Qwen4 PLE decode fusion. Unsupported inputs
    # and phases fall back to the original implementation.
    SGLANG_ENABLE_QWEN4_PLE_FUSION = EnvBool(True)
    # --ple-offload-backend file: where the sparse, file-backed PLE table lives
    # (deterministic name, reused across restarts), whether prefill-sized
    # gathers hint the page cache first, and an escape hatch for the device
    # attribute check (pageable host memory reachable through host page tables).
    SGLANG_QWEN4_PLE_FILE_DIR = EnvStr(lambda: _default_cache_subdir("ple"))
    SGLANG_QWEN4_PLE_FILE_PREFETCH = EnvBool(True)
    SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK = EnvBool(False)
    # Faulting rows in maps whole page-cache folios, so the mapping creeps
    # towards full residency (~45 KB/token) and eats the free memory that
    # sizes the KV pool. Cap its resident set; 0 disables the trim.
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB = EnvFloat(8.0)
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S = EnvFloat(30.0)
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
    SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
    # Copy rank-local MoE slices into independent CPU storage before H2D when
    # they reference a larger mmap-backed checkpoint storage.
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D = EnvBool(False)
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    # In seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging.
    # Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1)
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # ===================================================================
    # HTTP server and health
    # ===================================================================
    # Decompress request bodies tagged with `x-body-compressed`.
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    # Override parsed request fields from headers.
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)
    SGLANG_EXPOSE_OWN_ENV_VARS = EnvBool(False)
    SGLANG_DIAG_BYPASS_HEALTH_GENERATE = EnvBool(False)

    # ===================================================================
    # Logging
    # ===================================================================
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)
    SGLANG_ENABLE_RANK_CONSENSUS_CHECKER = EnvBool(False)

    # ===================================================================
    # IPC, broadcasters, and ports
    # ===================================================================
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    # Log top-level PickleWrapper frames unwrapped on msgpack IPC decode.
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)
    # Base port hint for ephemeral sockets (ZMQ, SHM broadcaster, etc.).
    # When set, get_open_port() and shm_broadcast search upwards from this
    # value instead of asking the OS for a random port.  Useful to keep all
    # SGLang ports in a predictable range behind a firewall.
    SGLANG_PORT = EnvInt(None)
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # ===================================================================
    # CI and test execution
    # ===================================================================
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    # Set to true by the check-changes CI job when a PR touches nothing under
    # rust/; default false so local and scheduled runs never skip the cargo tests.
    SGLANG_SKIP_RUST_TESTS = EnvBool(False)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)
    # Expand jit_kernel test grids to their full parameter ranges (nightly).
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = EnvBool(False)
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # ===================================================================
    # Crash diagnostics and shutdown
    # ===================================================================
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # ===================================================================
    # Constrained decoding and grammar
    # ===================================================================
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # ===================================================================
    # Fault injection and regression tests
    # ===================================================================
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)
    # Scheduler: force lazy extra_buffer prealloc to fail at decode boundaries
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    # KL tests: skip the cache-hit count assertion (e.g. when alloc failure reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)

    # ===================================================================
    # CI reporting: per-model metrics jsonl for nightly XPU dashboard
    # ===================================================================
    # When set, XPU nightly tests append one JSON record per model to this file
    # so xpu-ci-job-monitor.yml can render per-model ref/actual/status/duration
    # tables. Unset (the default) is a full no-op — pre-existing CI unaffected.
    SGLANG_TEST_METRICS_FILE = EnvStr(None)

    # ===================================================================
    # PD and scripted-runtime tests
    # ===================================================================
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)
    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # ===================================================================
    # Profiling, tracing, and metrics
    # ===================================================================
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    # profile_by_stage: do not start the decode-stage capture until a decode batch
    # reaches this many requests (0 = first decode batch). Lets a batch-size bench
    # capture steady-state full-admission decode steps instead of the ramp-up.
    SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS = EnvInt(0)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_NVTX"
    )
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPERATIONS_ENABLE_PROFILE"
    )
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    # Opt-in: emit one CUDA-graph capture trace per captured batch size (per-bs).
    # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE (single combined trace) takes
    # precedence when both are set.
    SGLANG_GRAPH_BATCH_CAPTURE = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    # Allocator-history buffer for /start_profile activities=["MEM"]; the
    # default truncates long windows (each entry is one alloc/free event).
    SGLANG_MEM_PROFILE_MAX_ENTRIES = EnvInt(100000)
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_TRACE_ASYNC = EnvBool(False)
    SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD = EnvInt(100)
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E_SYNC = EnvBool(False)
    SGLANG_TRACE_SAMPLER_E2E = EnvBool(False)
    SGLANG_TRACE_QWEN_MOE_DEEPEP_E2E = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_CONTIG = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_MASKED = EnvBool(False)

    # ===================================================================
    # Debugging and invariant checks
    # ===================================================================
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS = EnvBool(False)
    # NaN-fill the unified memory pool at boot (debug repro switch).
    SGLANG_DEBUG_POISON_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # The explicit environment variable still takes precedence over this CI
    # default, so production remains opt-in and CI remains opt-out if needed.
    SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK = EnvBool(_default_tree_cache_sanity_check)
    # Physical KV-page checks: committed<=allocated + no page alias.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)
    SGLANG_TBO_DEBUG = EnvBool(False)
    # Timing probe: run the swap-in fully but skip the host->device KV bytes,
    # measuring the "IO is free" floor. GARBAGE OUTPUT -- benchmarking only.
    SGLANG_DEBUG_HISPARSE_SKIP_IO = EnvBool(False)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    # Signal level for value/index validity checks (nan/inf/oob/...); see
    # invariants.py. OFF (prod default) runs only the free data layer, WARN
    # adds throttled logging, STRICT (CI default) crashes on violations.
    # Supersedes SGLANG_ENABLE_ASYNC_ASSERT, which is bridged as STRICT until
    # every callsite migrates.
    SGLANG_INVARIANT_CHECK = EnvInt(InvariantCheckLevel.OFF)

    # ===================================================================
    # Runtime simulations
    # ===================================================================
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)

    # ===================================================================
    # DSpark speculative decoding
    # ===================================================================
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FAST_KERNEL = EnvBool(True)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    SGLANG_DSPARK_FAST_SAMPLING = EnvBool(True)
    SGLANG_DSPARK_FOLDED_SAMPLING = EnvInt(DsparkFoldedSampling.AUTO)
    SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)
    SGLANG_DSPARK_STACKED_CTX_KV = EnvBool(True)
    SGLANG_DSPARK_EMBED_IN_GRAPH = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV = EnvBool(False)
    # With the TP-sharded markov_w2, gather each step's vocab-parallel logits over
    # the NVLink push collective (CustomAllReduceV2's multicast plane) instead of
    # the NCCL ring. Only taken when the group's communicator has a multicast
    # plane; off, or no such plane, keeps the NCCL all-gather.
    SGLANG_DSPARK_NVLINK_VOCAB_GATHER = EnvBool(True)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)

    # ===================================================================
    # Memory pools and KV-cache sizing
    # ===================================================================
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    # Disable lazy compaction in the unified memory pool allocator and
    # fall back to the per-free eager compaction. Used for production
    # A/B and quick rollback. Default False (lazy compaction on).
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    # Periodically log lazy-compaction stats per sub-pool (observability only).
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    # Per-call move cap on a non-urgent lazy-compaction flush, so a large
    # backlog cannot stall the scheduler loop; urgent flushes are uncapped.
    SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL = EnvInt(4096)
    # HND KV layout folds (page, head) into one paged index for per-kv-head sparse
    # page tables (DP attn); paged backends like trtllm_mha consume it directly.
    SGLANG_USE_HND_KVCACHE = EnvBool(False)

    # Attention (aiter, ROCm): route NEXTN spec draft_extend (EAGLE-v2 KV
    # catch-up) through aiter unified_attention (GQA-packed + split-KV) instead
    # of the occupancy-starved mha_batch_prefill FMHA. Independent kill-switch
    # for the new path; pairs with SGLANG_AITER_UNIFIED_VERIFY. Default on.
    SGLANG_AITER_UNIFIED_DRAFT_EXTEND = EnvBool(True)
    # size the KV pool after CUDA-graph capture
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)

    # ===================================================================
    # Scheduler token budgeting and admission
    # ===================================================================
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    # Window for the token-weighted recent cache-hit rate used to estimate
    # waiting-queue prefill load.
    SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS = EnvFloat(15.0)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    # Compact extend-attention scheduler tile-budget admission (AMD/HIP-only).
    # Budget <= 0 disables; >0 sets the max prefix-extend tiles per batch.
    SGLANG_PREFILL_TILE_BUDGET = EnvInt(0)
    # Tile-budget mode: "compact" (default, counts actual per-request tiles) or
    # "legacy" (rectangular grid, max_extend_len-shaped).
    # Internal/testing only - users should not need to change this.
    SGLANG_PREFILL_TILE_BUDGET_MODE = EnvStr("compact")
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)
    # Charge the chunked-prefill compute budget in tokens, not page-ceiled
    # tokens, so a prefill batch runs exactly chunked_prefill_size and the dense
    # GEMMs get an aligned M. gfx95 only; see PrefillAdder.exact_chunk_fill.
    SGLANG_EXACT_CHUNK_FILL = EnvBool(True)

    # ===================================================================
    # Scheduler polling, timeouts, and output
    # ===================================================================
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)
    # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # ===================================================================
    # Overlap scheduler and pipeline parallelism
    # ===================================================================
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # Force delay_sample_func for all overlap decode (not just grammar mode),
    # allowing CPU result processing to overlap with subsequent forward computation
    # and reducing the impact of sampling overhead on the critical path.
    SGLANG_ENABLE_DELAY_SAMPLE = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # Force the WAR barrier to wait for the whole forward instead of the
    # read-done fastpath event.
    SGLANG_FORCE_COARSE_WAR_BARRIER = EnvBool(False)
    # Enable prefill read-done publication after compliant metadata initialization.
    SGLANG_ENABLE_PREFILL_WAR_READ_DONE = EnvBool(False)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    # Run PP tensor communication on a dedicated stream so asynchronous sends
    # do not fence the next forward through the scheduler stream.
    SGLANG_PP_COMM_OVERLAP = EnvBool(False)
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)

    # ===================================================================
    # Radix and sparse KV caches
    # ===================================================================
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)
    # Kill-switch for the shared-index (IndexShare) swap-in prefetch
    # (auto-enabled for GLM-5.2-style DSA); set True to A/B synchronous swap-in.
    SGLANG_DISABLE_HISPARSE_PREFETCH = EnvBool(False)
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(True)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    # Deprecated: the unified radix tree is the default tree cache now, so the
    # registry no longer reads this. Kept because a few model/arch call sites
    # still assert on it; do not use in new code.
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)
    # Registered TreeCore backend serving the unified radix cache.
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("python")
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)

    # ===================================================================
    # PD disaggregation runtime
    # ===================================================================
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    # Enable on both P and D with the same --sampling-mask-max-tokens value.
    SGLANG_ENABLE_DISAGG_SAMPLING_MASK = EnvBool(False)
    # Retained only to reject the removed setting during startup.
    SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS = EnvInt(None)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT = EnvInt(1)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    # A wedged RDMA stack fails startup here instead of at the scheduler watchdog.
    SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT = EnvInt(60)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS = EnvInt(16384)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    # Deferred decode-side KV release: on abort, hold an in-flight request's KV
    # pages/slot until the prefill acks the transfer drained, or the timeout
    # below fires. Off by default (no behavior/perf impact when disabled).
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE = EnvBool(False)
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT = EnvFloat(30.0)

    # ===================================================================
    # Distributed and model-parallel runtime
    # ===================================================================
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)

    # ===================================================================
    # Tool calling and native web search
    # ===================================================================
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)
    # Native web search (Exa). EXA_API_KEY is the vendor BYOK credential
    # (kept as-is, not renamed to SGLANG_*); the SGLANG_EXA_* knobs tune the
    # request defaults for the built-in GPT-OSS web_search tool.
    EXA_API_KEY = EnvStr(None, secret=True)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # ===================================================================
    # HiCache storage backends and mmap allocation
    # ===================================================================
    # Per-call cudaHostRegister limit in GB.
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)
    # Base token count for each MLA/DSA dedup broadcast chunk.
    SGLANG_MLA_DEDUP_CHUNK_TOKENS = EnvInt(2048)
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM = EnvBool(False)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    # Enable client-side metadata caching to optimize filesystem checks (e.g. for Lustre/NFS/FUSE)
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    # Positive cache TTL for filesystem metadata lookups (-1 disables positive expiration)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    # Buffer mode: staged prefetches pin their device anchor from IO commit
    # to consumption so eviction cannot waste the fetch. Cap = fraction of
    # the pool the pins may hold; 0 disables pinning.
    SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP = EnvFloat(0.5)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")

    # ===================================================================
    # KV-transfer staging and Mooncake transport
    # ===================================================================
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    # Opt-in limit for the number of KV cache indices represented by one
    # synchronous all-layer Mooncake batch. Set to a positive value to split
    # larger transfers; 0 preserves the legacy single-batch behavior.
    SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES = EnvInt(0)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # ===================================================================
    # Mooncake store
    # ===================================================================
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)
    MOONCAKE_TENANT_ID = EnvStr("default")

    # ===================================================================
    # Ascend MemCache (HiCache L3); see https://gitcode.com/Ascend/memcache
    # ===================================================================
    SGLANG_HICACHE_MEMCACHE_CONFIG_PATH = EnvStr(None)
    SGLANG_NPU_MEMCACHE_ENABLE_WARMUP = EnvBool(False)

    # ===================================================================
    # MoRI transport and expert dispatch
    # ===================================================================
    SGLANG_DEEPEP_V2_FORCE_MAX_LEN = EnvBool(False)
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)

    # ===================================================================
    # AMD, ROCm, and AITER
    # ===================================================================
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    # Quantize the variable-length DP-MoE gather payload (SGLANG_DP_USE_GATHERV
    # path, prefill/extend only) to fp8-e4m3 with per-token-group-128 scales:
    # halves the gathered hidden-state bytes over NCCL; the combine
    # (reduce_scatterv) leg stays bf16 (NCCL SUM cannot run on fp8).  Lossy on
    # the wire — same group quantization the MoE expert GEMMs apply to their
    # input anyway, but router/shared-expert reads see rounded values, so this
    # stays accuracy-gated and default OFF.
    SGLANG_ENABLE_DP_GATHER_FP8 = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # Fold `silu(gate) * up` into the triton MoE up-GEMM epilogue. W13 rows are
    # permuted in place at load so gate/up land in adjacent columns of the same
    # output tile, which removes intermediate_cache1 and the standalone
    # activation launch per MoE layer. Opt-in because the in-place permute is
    # not compatible with runtime weight updates or EPLB expert rearrangement,
    # both of which assume the checkpoint's halves layout.
    SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED = EnvBool(False)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    USE_ROCM_AITER_ROPE_BACKEND = EnvStr("0")
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    # Fold the KDA [f_a|b] tail into the wide [q,k,v,g] projection so the whole
    # in-proj is one GEMM. Decode is bandwidth bound there, so the 144 extra
    # output columns ride along nearly free.
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ = EnvBool(True)
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS = EnvInt(256)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("tilelang")
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # Above 8192 tokens of context, aiter's non-static workspace is large enough
    # that mem_fraction_static is scaled by 0.85 to leave room for it. Set this to
    # honor an explicitly passed --mem-fraction-static instead. Off by default:
    # the reserve is load-bearing, and skipping it OOMs long-context aiter serving
    # that fits comfortably with it (67.32 GiB request against 47.40 GiB free on a
    # 288 GB MI355 in nightly-4-gpu-mi35x-minimax-m3). Worth setting only when the
    # scaled fraction is itself too small to hold the model weights.
    SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION = EnvBool(False)
    # Route Kimi-K3-style h12 + fp8 MLA decode through aiter Triton Gluon when
    # import and Triton cga_layout prerequisites hold. Set to 0 to force the
    # zero-pad mla_decode_fwd fallback (benchmarking / emergency disable).
    SGLANG_AITER_MLA_GLUON = EnvBool(True)

    # DSV4 Aiter flags
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    # Unified KV wired the fused qk-norm-rope kernel to decode only, so MTP
    # target-verify kept running the norm+RoPE as separate kernels. Set to 0 to
    # go back to the unfused chain on the verify path.
    SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY = EnvBool(True)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)

    # ===================================================================
    # Apple Silicon and MLX
    # ===================================================================
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)
    # MLX buffer-cache cap in GB.
    SGLANG_MLX_CACHE_LIMIT_GB = EnvFloat(None)

    # ===================================================================
    # Ascend NPU
    # ===================================================================
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    # Kimi-K3 attention-TP shared experts: overlap AG / MLP / RS with the
    # routed front / DeepEP dispatch / routed GEMMs, respectively.
    SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Fuse grouped Kimi-K3 SiTU with valid-row MXFP8 quantization before GMM2.
    # Set to 0 to restore the separate SiTU + npu_dynamic_mx_quant path.
    SGLANG_NPU_MOE_SITU_MXFP8_FUSED = EnvBool(True)
    SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD = EnvBool(False)
    # Use FIAS V2 for DSpark MLA target verify and MHA draft paths. Graph
    # replay requires torch_npu's V2 handler to update actual_seq_kvlen.
    SGLANG_NPU_USE_FIAS_V2_BSND = EnvBool(False)
    # BF16 wo_a: use F.linear for single-local-group decode (Flash TP8),
    # retaining the original weight layout. Opt-in for A/B.
    SGLANG_OPT_NPU_BF16_WO_A_GEMM = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Enable int4x2 weights loading
    SGLANG_NPU_W4A4_NEW_PACKING = EnvBool(False)
    # Use the graph-safe Triton-Ascend kernel for masked speculative KV commits.
    SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NPU_USE_TRITON_KV_CACHE_STORE"
    )
    # Quantize x to int8 in the dispatch operator (vendor alias consumed by the
    # Ascend DeepEP library; the MTP draft-build scopes override it to False).
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False)
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    # ===================================================================
    # MUSA
    # ===================================================================
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # ===================================================================
    # Quantization
    # ===================================================================
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    # GLM NextN (MTP): cast the draft layer's bf16 fused MoE to per-channel FP8
    # on load. Unrelated to the NVFP4 block-FP8 NextN path above.
    SGLANG_GLM_NEXTN_MOE_PTPC = EnvBool(False)
    # Load a ModelOpt MIXED_PRECISION draft's FP8_BLOCK_SCALES experts as NVFP4
    # (per-tensor activation scale 1.0); target layers are unaffected.
    SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")
    # On by default; set SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 as a kill switch.
    # Consults the tuned per-(N, K, M) Triton tile config table in
    # apply_fp8_linear. When a tuned config exists for this GPU / weight shape /
    # token count, run the Triton w8a8 FP8 GEMM with it; otherwise keep the
    # default CUTLASS path. Only takes effect on a GPU with a matching
    # dtype=fp8_w8a8_channelwise config JSON under
    # kernels/ops/quantization/configs/ (currently L40S), so it is a no-op on
    # any other GPU / untuned shape even when enabled.
    SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE = EnvBool(True)
    # Path to a pinned exllamav3 checkout (see exl3_ext.EXLLAMAV3_COMMIT) whose
    # exllamav3_ext/ sources are JIT-built into the EXL3 quant method's CUDA
    # extension. Must be set to use EXL3; empty raises when the extension is built.
    SGLANG_EXL3_SRC = EnvStr("")
    # Build directory for the JIT-built EXL3 extension (torch.utils.cpp_extension
    # cache); expanded with os.path.expanduser at use.
    SGLANG_EXL3_BUILD_DIR = EnvStr("~/.cache/sglang/exl3_ext")

    # ===================================================================
    # Humming quantization
    # ===================================================================
    SGLANG_HUMMING_ONLINE_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_INPUT_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_USE_F16_ACCUM = EnvBool(False)
    SGLANG_HUMMING_MOE_GEMM_TYPE = EnvStr("")

    # ===================================================================
    # FlashInfer, FlashMLA, and TRT-LLM
    # ===================================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Per-rank dispatch capacity of the FlashInfer MoE A2A dispatcher. Unset
    # means each call site keeps its own default.
    SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(None)
    # FlashInfer MegaMOE (generic moe_ep.MoEEpMegaLayer backend). Sizes the
    # per-rank symmetric workspace; must be >= the largest padded per-rank batch
    # (derived from cuda_graph_max_bs / chunked_prefill_size when unset).
    SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK = EnvInt(0)
    # Opt-in in-kernel FC2 top-k reduce (cross-rank REDG atomic-add) for the
    # cutedsl mega kernels (NVFP4 / MXFP8). Deletes the multi-GB combine staging
    # region and can win at large batch, but makes the output accumulation order
    # nondeterministic (bf16 unordered sum) -- keep off for bit-reproducibility.
    # No effect on the DeepGEMM (block-FP8) mega path, which lacks the knob.
    SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE = EnvBool(False)
    # Cross-rank combine wire format for the FlashInfer NVFP4 cutedsl MegaMOE
    # kernel. "bf16" is exact/default; "mxfp8" and "nvfp4" reduce combine
    # traffic with a small accuracy tradeoff and require FC2 reduce outside the
    # kernel.
    SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE = EnvStr("bf16")
    # Enable per-token FP32 activation scaling for serialized ModelOpt FP4 with
    # FlashInfer TRT-LLM or CuTe DSL v2 MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # Use BF16 activations with FlashInfer CuTe DSL NVFP4 dense and MoE weights.
    SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16 = EnvBool(False)
    # Launch the TRT-LLM MoE grouped GEMMs with PDL only at or below this
    # token count.
    SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS = EnvInt(8192)
    # Use FlashInfer's fused atomic CUTLASS/CuTe DSL MoE finalize.
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(False)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # Split TRTLLM-GEN decode attention into sorted, equal-size request groups.
    # One preserves the default single-call path; values above one are useful
    # for batches whose KV sequence lengths have a large spread.
    SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS = EnvInt(1)
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
    # Store DeepSeek-V4 SWA KV directly in FlashInfer's 64-token SM120 page
    # layout. The scheduler continues to allocate 256-token logical pages.
    SGLANG_OPT_SM120_DIRECT_SWA_KV = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    # Also autotune one EXTEND-shaped dummy at max_prefill_tokens during
    # warmup. Opt-in: the extra forward needs transient activation headroom
    # that small-VRAM or tightly-packed configs may not have.
    SGLANG_FLASHINFER_AUTOTUNE_EXTEND = EnvBool(False)

    # ===================================================================
    # Triton and Torch compilation
    # ===================================================================
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)
    # A-B kill-switch for Work-Centric (Lean) Attention. When True, forces the
    # standard Triton decode kernel even if --enable-lean-attention or the auto-gate
    # would select Lean. Used to isolate the Lean kernel in benchmarks.
    SGLANG_DISABLE_LEAN_ATTENTION = EnvBool(False)
    # Persistent-grid size multiplier for the Lean decode kernel:
    # total_programs = round(device_CU_count * this). Default 1.0 (one CTA per CU), which
    # maximizes KV work-tiles per CTA and minimizes the cross-CTA combine/atomic reduction.
    # Kernel + E2E A/B sweeps found 1.0 beats 2.0 across uniform and ragged configs on both
    # MI300X (gfx942) and MI355X (gfx950) — 2.0 oversubscribed the CUs and regressed high-batch
    # decode. Exposed as a knob (e.g. set 2.0) for grid A/B tuning without a rebuild.
    SGLANG_FORCE_LEAN_GRID_CU_MULT = EnvFloat(1.0)
    # Most CTAs one fused HC mix launch (hc_mix_triton.py) uses; 0 launches one
    # CTA per SM. Its device-wide barrier waits for every CTA, so a launch that
    # claims every SM stalls behind SMs held by concurrent expert-row copies.
    # On a 170-SM RTX 5090, 168 CTAs still stall for the whole copy, 160 do
    # not, and 128 matches the all-SM launch's uncontended latency.
    SGLANG_OPT_HC_MIX_MAX_CTAS = EnvInt(128)

    # Torch Compile
    # Compact extend-attention query-tile grid: AMD/HIP-only optimization
    # (parity with flash-attn's ragged-aware launch). The feature checks _is_hip
    # explicitly in code; this env var allows override (0=force off, 1=force on).
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION = EnvBool(True)
    # Raise if Triton loads a kernel after the engine starts serving. This
    # verifies that startup warmup covers every kernel specialization used at
    # serving time.
    SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY = EnvBool(False)
    SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS = EnvFloat(1.0)
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB = EnvFloat(1.0)
    # gfx950 MLA decode stage-1: pick the launch geometry and split count per batch.
    # Reorders the fp32 accumulation, so off by default.
    SGLANG_MLA_DECODE_TUNE = EnvBool(False)
    # Native FP8 prefill for exact gfx950 Kimi-K3 zero-prefix and absorbed
    # cached-prefix shapes. Validated at 98% GSM8K accuracy.
    SGLANG_TRITON_FP8_PREFILL_ATTN = EnvBool(True)
    # Route Triton MLA prefill that carries a cached prefix through dense
    # (non-absorbed) one-shot MHA: up-project the prefix out of the latent KV
    # cache and run a single dense FP8 kernel instead of the absorbed 576/512
    # prefill. Materializes K/V for the whole batch, so it only engages when
    # the batch fits the chunk budget.
    SGLANG_TRITON_DENSE_PREFILL_ATTN = EnvBool(True)
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # ===================================================================
    # Expert parallel load balancing
    # ===================================================================
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    # Chunk size for the rebalance expert-weight P2P exchange; set
    # >= num_physical_experts to submit a single batch_isend_irecv.
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )

    # ===================================================================
    # DeepGEMM
    # ===================================================================
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    # Enable the allowlisted low-M BF16 Split-K GEMM path on Blackwell. Shapes
    # outside the measured allowlist continue to use CuTe DSL/cuBLAS.
    SGLANG_ENABLE_BF16_SPLITK_GEMM = EnvBool(True)
    SGLANG_DEEPGEMM_STANDARD_LAYOUT = EnvStr("auto")
    SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION = EnvFloat(0.25)
    # Cap the DeepGEMM masked grouped-GEMM per-expert padded capacity at
    # round_up(max(masked_m), 256) instead of round_up(rank_tokens, 256):
    # shrinks the [num_local_experts, m, *] MoE intermediates ~4x under
    # load imbalance (they otherwise OOM saturated --moe-runner-backend
    # deep_gemm serving).  Costs one D2H sync per MoE layer.
    SGLANG_OPT_DG_MASKED_M_CAP = EnvBool(False)
    # Wide-DP eager prefill uses compact routing storage; masked storage scales
    # with num_local_experts and can OOM on skewed batches.
    SGLANG_OPT_DG_COMPACT_EAGER = EnvBool(False)
    # Drop dp-attention MAX_LEN pad rows from MoE dispatch (StandardDispatcher
    # post-translation topk_ids -> -1): pad rows otherwise run the router on
    # stale hidden values and burn expert compute whose outputs are discarded;
    # colliding pad top-ks also inflate the DeepGEMM masked-GEMM workspace to
    # OOM at saturation.  Capture-safe (reads only global_num_tokens_gpu).
    SGLANG_OPT_MASK_DP_PAD_MOE = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    # Resolved lazily so it tracks SGLANG_CACHE_DIR, which is defined below.
    SGLANG_DG_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # ===================================================================
    # Cache directories
    # ===================================================================
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    # Persistent CuTe DSL AOT objects. Resolved lazily so it tracks
    # SGLANG_CACHE_DIR; set to an empty string to keep compilation
    # process-local. Must be trusted: cached objects are loaded into the process.
    SGLANG_CUTE_AOT_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("cute_aot"))

    # ===================================================================
    # Kernel development: JIT build cache, diagnostics and benchmarks
    # ===================================================================
    # Everything here is a developer knob for working ON kernels -- building
    # them, inspecting what the compiler produced, and benchmarking them. Flags
    # that select a kernel in production live with their own feature instead.
    # JIT kernel build cache. None = unset, resolving to ~/.cache/sglang/jit;
    # point it at a persistent mount to share builds across CI jobs.
    SGLANG_JIT_CACHE_DIR = EnvStr(None)
    # Log, at INFO, which dependency changed whenever a module is rebuilt.
    SGLANG_JIT_CACHE_DEBUG = EnvBool(False)
    # How many builds to keep per module variant. None = unset = keep all, which
    # is what makes reverting an edit an instant hit instead of a rebuild; set
    # it to trade that away for disk (1 keeps only the most recent build).
    SGLANG_JIT_CACHE_KEEP = EnvInt(None)
    # Skip the cache lookup and run the compiler for every module this process
    # loads. The result is still published, so the cost is one rebuild per
    # module, not one per load.
    SGLANG_JIT_FORCE_RECOMPILE = EnvBool(False)
    # Raise instead of compiling when a module misses the cache, so a
    # deployment that expects a pre-seeded cache fails loudly at startup
    # rather than silently eating a cold compile.
    SGLANG_CRASH_ON_JIT_COMPILE = EnvBool(False)
    # Ask the device compiler for per-kernel resource usage (registers, spills,
    # shared memory) and log it at INFO. Changes the build flags, so it compiles
    # into its own cache entry and leaves the normal one alone -- but that entry
    # is a hit on the second run, and a cache hit has nothing to report, so pair
    # this with SGLANG_JIT_FORCE_RECOMPILE to see the report every time.
    SGLANG_JIT_LOG_RESOURCE_USAGE = EnvBool(False)
    # Drop the GB/s and TFLOPS columns from the benchmark marker's table.
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_BANDWIDTH = EnvBool(False)
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_FLOPS = EnvBool(False)

    # ===================================================================
    # Expert-parallel dispatch and MoE execution
    # ===================================================================
    # Deprecated in favor of '--deepep-dispatcher-output-dtype bf16' but still
    # read by several call sites; do not use in new code.
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # Per-rank buffer capacity, not a model token limit.
    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # 0 lets ElasticBuffer select its theoretical communication SM/QP counts.
    SGLANG_DEEPEP_V2_NUM_SMS = EnvInt(0)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    # A5 DSV4 FP4 + DeepEP low-latency dispatch wire format. This is read only
    # by the model-specific dispatcher configuration; all other paths retain
    # their existing behavior.
    SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE = EnvStr("mxfp8")
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP = EnvBool(True)
    # Force dynamic Waterfill with runtime EP all-reduce instead of the default
    # static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(True)
    # DeepSeek/GLM MoE (deepseek_v2.py): quantize the (dp-gathered) MoE input
    # to per-token-group-128 fp8 ONCE and feed both the fused shared-expert
    # GEMM (cutlass w8a8 linear) and the routed experts' triton fused runner,
    # instead of quantizing the same [T, hidden] tensor twice with different
    # scale layouts. Only engages on CUDA with fp8 block-128 weights, the
    # standard dispatcher, and the triton MoE runner; falls back silently
    # otherwise.
    SGLANG_OPT_MOE_QUANT_ONCE = EnvBool(False)

    # ===================================================================
    # DeepGEMM Mega MoE
    # ===================================================================
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(8192)
    # Blackwell MegaMoE uses a whole-grid software barrier. Keep a small
    # residency margin so every cluster can launch beside other streams.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS = EnvInt(2)

    # ===================================================================
    # Top-k kernels
    # ===================================================================
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    # Opt-in: route DeepSeek-V3 grouped topk through the unified Triton router
    # instead of the flashinfer/AOT grouped kernels. Off by default (flashinfer is
    # the tuned production path); the Triton path is bit-exact on DeepSeek-V3.2 e2e
    # and benchmarks at parity, so this is a consolidation escape hatch, not a perf flip.
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # ===================================================================
    # Kernel selection and fused backends
    # ===================================================================
    # MiniCPM sparse attention developer switches
    SGLANG_MINICPM_FUSE_TOPK = EnvBool(False)
    SGLANG_MINICPM_DENSE_AS_SPARSE = EnvBool(False)
    SGLANG_MINICPM_FORCE_DENSE = EnvBool(False)

    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)
    # Force every sglang.kernels BaseFusedOp onto one backend (a KernelBackend
    # value, e.g. "torch" / "torch_compile" / "triton" / "aot"); unset =
    # auto-select by priority. "torch" flips all fused ops to their pure-torch
    # reference implementations for numerical-bug bisection.
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)

    # ===================================================================
    # Logits and log-probability processing
    # ===================================================================
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
    SGLANG_ENABLE_LOGPROB_CHUNK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK"
    )
    SGLANG_LOGPROB_CHUNK_SIZE = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"
    )
    # Compute input logprobs from logits via per-row logsumexp instead of
    # materializing the full-vocab log-softmax. Escape hatch only; the two
    # paths are mathematically identical.
    SGLANG_ENABLE_FAST_INPUT_LOGPROBS = EnvBool(True)

    # ===================================================================
    # Deterministic inference and all-reduce
    # ===================================================================
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    # NCCL channel count pinned on CUDA so the all-reduce reduces a token the
    # same way whatever else shares its batch. Raise it to buy back bandwidth
    # on links that can drive more channels.
    SGLANG_DETERMINISTIC_NCCL_NCHANNELS = EnvInt(8)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    # Default per-direction workspace cap for CustomAllReduceV2; explicit
    # constructor sizes take precedence over this.
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(16 * 1024)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)

    # ===================================================================
    # RoPE cache
    # ===================================================================
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # ===================================================================
    # Speculative decoding
    # ===================================================================
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
    # Experimental: allow pipeline parallelism x speculative decoding
    # (EAGLE/MTP). Off by default; see the PP+spec RFC for constraints
    # (non-overlap schedule, no DP attention).
    SGLANG_ENABLE_PP_SPEC = EnvBool(False)
    # Capture the per-replay attention-metadata prep (init_forward_metadata_out_graph)
    # into a small CUDA graph, collapsing its host dispatch cost to one launch.
    # Experimental; auto-falls back to eager if the backend's prep is not capturable.
    SGLANG_ENABLE_METADATA_GLUE_GRAPH = EnvBool(False)
    SGLANG_OPT_FUSED_KDA_VERIFY = EnvBool(False)
    # A/B: keep the DFLASH draft greedy head eager (not folded in-graph).
    SGLANG_DFLASH_EAGER_DRAFT_SAMPLER = EnvBool(False)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Which speculative decisions rank 0 broadcasts to its TP group; narrowing
    # it under live traffic isolates where ranks actually diverge. Comma
    # separated presets ("all", "rng", "init", "off"), or SpecTpSyncSite slugs
    # and numbers, each negatable with a leading "-": "all,-dspark-plan,-6".
    SGLANG_SPEC_TP_SYNC = EnvStr("all")
    # Kill-switch for the draft-extend cuda graph. Draft extend then always runs
    # eager. Escape hatch for setups where the capture's memory pool costs more
    # than the graph saves (e.g. DeepEP MoE workspace captured at full dispatch
    # capacity).
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)
    # Use the split-KV (flash-decode) kernel for EAGLE target-verify on the
    # Triton backend (ROCm). Only active at speculative topk == 1; falls back to
    # extend_attention_fwd for unsupported cases or when set false (e.g. for
    # debugging). Correctness is unaffected; this only changes performance.
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # ===================================================================
    # Multimodal processing
    # ===================================================================
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # Currently supported by the Kimi-K2.5 image processor only.
    SGLANG_FORCE_CPU_IMAGE_PREPROCESSING = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)

    # ===================================================================
    # Multimodal CUDA IPC transport
    # ===================================================================
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    # Reuse the mapping for the already-allocated bounded CUDA IPC pool. This
    # has no effect unless CUDA IPC feature transport is explicitly selected.
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(True)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # ===================================================================
    # Mamba state and cache
    # ===================================================================
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)
    # Kill-switch for the fused per-slot conv clear/copy kernel (MambaPool);
    # falls back to the per-conv-type Python loop.
    SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS = EnvBool(False)
    # Opt-in: on the unified radix tree, leave the matched-prefix mamba evictable
    # during decode (it is already COW'd to the request's own slot) and shrink the
    # mamba pool ratio accordingly. Frees one resident slot per running request,
    # raising max_running_requests. Off = original locking + ratio (escape hatch).
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = EnvBool(False)

    # ===================================================================
    # CUDA graphs and execution buffers
    # ===================================================================
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
    # Guards CUDA graph executable dedup via cudaGraphExecUpdate.
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)
    # Reuse wholly-free graph-pool segments for step-local eager allocations.
    SGLANG_ENABLE_GRAPH_POOL_BORROW = EnvBool(False)
    # Mint capture's measured footprint as one span so the graph pool is carved
    # out of a single contiguous region instead of grown segment by segment.
    SGLANG_ENABLE_GRAPH_POOL_PRECARVE = EnvBool(False)
    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)

    # ===================================================================
    # Tokenizer, request state, embeddings, and reasoning controls
    # ===================================================================
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)
    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)
    SGLANG_PATCH_TOKENIZER = EnvBool(True)
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)
    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ===================================================================
    # Encoder pipeline and disaggregation
    # ===================================================================
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    SGLANG_ENCODER_MM_LOAD_WORKERS = EnvInt(4)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Seconds before permanently dropping an unhealthy encoder (0 = keep probing).
    SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL = EnvFloat(600.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # ===================================================================
    # Native gRPC server
    # ===================================================================
    # Native gRPC server. SGLANG_GRPC_PORT is the env fallback for the
    # --grpc-port CLI flag; setting either enables the native server alongside
    # HTTP. The worker-threads knob stays env-only (internal tuning, no CLI
    # surface).
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)

    # ===================================================================
    # NUMA and CPU affinity
    # ===================================================================
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(True)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # ===================================================================
    # DeepSeek V4
    # ===================================================================

    # Model and Quantization
    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Set True to dequantize the FP4 experts to FP8 at runtime
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    # Flash-0731 also accepts "low"; the active profile is checkpoint-resolved.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
    # DeepSeek-V4.1 default when a request carries no reasoning_effort: one of
    # low/high/xhigh/max or an integer budget in [1, 100]; unset -> the encoder default.
    SGLANG_DSV41_REASONING_EFFORT = EnvStr(None)
    # Quantize the SWA fp8 KV cache from bf16-rounded values (matches
    # trainer-side QAT and the DSA-CP path) instead of fp32 registers.
    SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE = EnvBool(False)
    # Paged KV layout of the DeepSeek-V4 family pools: "v4" (584 B/token, every
    # GPU), "v41" (the SM100 FlashMLA V4.1 formats: 528 B fp8 SWA cache, fp8 or
    # fp4 compressed caches) or "auto" (v41 on SM100 when FlashMLA supports it).
    SGLANG_DSV4_KV_LAYOUT = EnvStr("v4")
    # Compressed-cache layout under "v41": "auto" (fp4 for the fp4-rounded
    # ratio-1 / ratio-2 latents, fp8 for ratios 4 / 128), "fp8" or "fp4" for all.
    SGLANG_DSV4_COMPRESSED_KV_LAYOUT = EnvStr("auto")
    # unified_kv only: split the pool into an fp8 nope pool plus a parallel
    # bf16 rope pool, 640 B/token instead of 1024. The unified pool takes no
    # dtype, so --kv-cache-dtype has no effect there and this switch is the
    # only way to ask; on separate-KV it is the reverse -- --kv-cache-dtype
    # picks the buffer dtype and this switch is inert.
    SGLANG_DSV4_UNIFIED_KV_FP8 = EnvBool(False)

    # DeepSeek-V4.1 engram host table: keep the tables in host memory (layout
    # below) and gather rows from the GPU instead of sharding them over HBM.
    SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE = EnvBool(False)
    # "shared" is one buffer for the whole TP group, mapped by every rank, with no
    # lookup all-reduce (the ranks must share a PID namespace); "per_rank" is one
    # anonymous mapping per rank holding only its rows, gathered with the
    # all-reduce, and the only layout that gets huge pages without shmem THP.
    SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT = EnvStr("shared")

    # DeepSeek-V4.1 fork knobs (engram file table, EXL3 expert streaming), resolved
    # together by sglang.srt.dsv41_config.Dsv41Config. The declarations above and
    # SGLANG_DSV41_TORCH_PREFILL_INDEXER below are upstream's; leave them in place.

    # When set, every EngramEmbedding serves rows from this directory's safetensors
    # shards via np.memmap instead of loading the table (device or host memory).
    # The layer-1 table is 101.5 GB, larger than the host RAM budget on some rigs.
    SGLANG_DSV41_ENGRAM_TABLE_DIR = EnvStr("")
    # Engram RAM row cache in GiB in front of SGLANG_DSV41_ENGRAM_TABLE_DIR,
    # shared by every Engram layer; misses read with O_DIRECT. 0 keeps the
    # plain np.memmap path.
    SGLANG_DSV41_ENGRAM_RAM_GIB = EnvFloat(0.0)
    # Capture layer-1 Engram lookup as a CUDA host node backed by a shared native
    # row cache and an io_uring worker. Disabled unless explicitly requested.
    SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING = EnvBool(False)

    # DeepSeek-V4.1 EXL3 routed experts streamed from disk (eager only): the
    # checkpoint loader skips them and each MoE layer gathers the experts it
    # routes to through the MoE expert streaming framework. Its knobs size the
    # tiers: SGLANG_MOE_PINNED_HOST_MB (host RAM), SGLANG_MOE_HOT_GPU_MB (VRAM),
    # SGLANG_MOE_HOT_SEED (startup residency, a {"count": [[...]]} JSON).
    SGLANG_DSV41_EXPERT_STREAM = EnvBool(False)
    # Directory whose EXL3 shards the streamed experts are read from (required
    # when streaming; the experts may live on a different drive than the model).
    SGLANG_DSV41_EXPERT_DIR = EnvStr("")
    # When set, append one JSON line per streamed MoE layer call (forward, layer,
    # tokens, experts and route counts, VRAM and RAM misses, read and split time)
    # for offline tier simulation (scripts/dsv41/tier_sim.py).
    SGLANG_DSV41_EXPERT_TRACE_PATH = EnvStr("")
    # Option C (EXL3 graph decode): how long the in-graph wait for the RAM-miss
    # thread may take per MoE layer, in ms, before the process fails stop. The
    # thread's watchdog aborts after max(30 s, 3x this) (exl3_ram_miss.watchdog_wait_s),
    # so it always outlasts this wait and the eager pause bound (2x this + 1 s).
    SGLANG_DSV41_RAM_MISS_TIMEOUT_MS = EnvInt(2000)
    # Option C: copy each RAM-miss row from the bounce bank into the pinned slabs on this many worker
    # threads (each row split into this many byte ranges) instead of on the service thread. 0 keeps the
    # copy on the service thread (there is no auto). Workers never run on cores 64-71. Do not enable in a run
    # that produces stage traces until the trace record carries the mode: overlap_timeline.py misreads them.
    SGLANG_DSV41_RAM_MISS_PACK_WORKERS = EnvInt(0)
    # Test only: "<demands>:<seconds>" makes the RAM-miss thread sleep before every
    # demand read once that many demands have read rows (forces an Engine-level
    # timeout after capture). Empty: off.
    SGLANG_TEST_DSV41_RAM_MISS_FAULT = EnvStr("")
    # Option F on top of option C: during MoE layer L of a graph decode step, post
    # the previous token's routes for layer L+1 that are not in RAM as advisory
    # reads for the RAM-miss thread (demands always go first). Off by default.
    SGLANG_DSV41_ENABLE_EXPERT_PREFETCH = EnvBool(False)
    # Option C lease mode (analysis/dsv41-drive/LEASE_PROTOCOL.md): the RAM-miss thread leases every pinned slot the
    # in-graph copy reads, the device copies only what it holds a lease on and acknowledges after the copy, and the
    # thread never evicts a leased slot. It arms every record that plans rows, so each MoE layer pays one service round
    # trip even when every row is in RAM (LEASE_PROTOCOL.md 15). Read once when the service starts; off is today's
    # protocol bit for bit. Off by default.
    SGLANG_DSV41_ENABLE_RAM_MISS_LEASES = EnvBool(False)
    # Option C lease mode, two-phase (Task 6 V1): the service grants the lanes already resident inside its
    # reservation critical section, before it reads the missing rows, so the device copies those rows while the
    # read is still running instead of after it. Needs SGLANG_DSV41_ENABLE_RAM_MISS_LEASES. Read once when the
    # service starts; off leaves lease mode exactly as it shipped, which is the arm V1 is measured against.
    SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE = EnvBool(False)
    # Two-phase stage 1: how long, in microseconds of %globaltimer from the kernel's start, it waits for the service
    # to publish the resident lanes before copying whatever has arrived. It never waits for the request to be
    # served, so this trades a fuller first stage against a later one. Time, not polls: a poll pass reads each lane
    # across PCIe, so its cost depends on the lane count. At 8 passes the wait measured p50 13 us, p90 93 us.
    SGLANG_DSV41_RAM_MISS_HIT_WAIT_US = EnvInt(100)
    # Per-piece streaming of NVMe-read expert rows on top of two-phase (piece-streaming plan):
    # the read splits into pieces and the device streams each as it lands, instead of the two
    # fixed W1/W2 stages. Needs SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE, SGLANG_DSV41_ENABLE_RAM_MISS_LEASES
    # and SGLANG_DSV41_RAM_MISS_PACK_WORKERS > 0 (the inline pack path has no publisher). Off by default.
    SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM = EnvBool(False)
    # Row images (plan 2026-09-24-dsv41-row-images): the native RAM-miss reader reads <root>/exl3_row_images of every
    # SGLANG_MOE_EXPERT_MIRROR_DIRS root (split by SGLANG_MOE_EXPERT_MIRROR_WEIGHTS, as the mirrors are) with one
    # O_DIRECT readv per read straight into the pinned slab rows: no bounce buffer, no packing workers, no copy.
    # Needs mirror dirs holding images built by scripts/dsv41/build_row_images.py and
    # SGLANG_MOE_EXPERT_FILE_READER=uring_direct. The eager row source keeps reading the mirrored shards. Off by default.
    SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES = EnvBool(False)
    # DSV4 MoE side stream (plan 2026-09-25-dsv41-copy-compute-overlap, 1a): the shared expert and the DIRECT
    # residency commit run on one side stream, joined before the shared-expert add, so they overlap the RAM-miss
    # copies and the routed MoE kernel instead of running in line. Read once per process. Off by default.
    SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM = EnvBool(False)

    # Kernels and indexer
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_FLASHINFER_MHC = EnvBool(False)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    # Per-rank local query rows (after DP-attention sharding when enabled),
    # not request ISL.
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    # Run the DeepSeek-V4.1 ratio-1/2 prefill indexer on the torch path instead
    # of the DeepGEMM dense fp4 logits kernel (test oracle / fallback).
    SGLANG_DSV41_TORCH_PREFILL_INDEXER = EnvBool(False)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)

    # cache, GEMM, and distributed
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    # ROCm gfx950: fuse inverse-RoPE into the wo_a mxfp8 quant (aiter
    # inverse_rope_group_quant) instead of a separate fused_rope_inplace + Triton
    # quant. Off by default; requires SGLANG_OPT_FP8_WO_A_GEMM and the aiter op.
    SGLANG_OPT_FP8_WO_A_FUSED_INVROPE = EnvBool(False)
    # SM100/SM103: collapse the bf16 wo_a verify chain (fused_rope_inplace,
    # _wo_a_partial, _wo_a_reduce_quant) into one cluster-launched megakernel.
    # Emits MXFP8 when wo_b supports it, otherwise BF16.
    SGLANG_DSV41_FUSED_WO_A = EnvBool(True)
    # Route the decode wo_a bf16 batched matmul off rocBLAS/Tensile onto aiter's
    # tuned batched_gemm_bf16 (gfx95). Off by default; see deepseek_v4.py
    # _apply_wo_a_bf16_matmul.
    SGLANG_OPT_USE_AITER_BATCHED_GEMM = EnvBool(False)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # ===================================================================
    # Inkling
    # ===================================================================
    SGLANG_OPT_USE_FUSED_GATE_TOPK = EnvBool(True)
    # Inside the fused gate: use the CUDA JIT top-k+renorm kernel (v2) instead
    # of the triton kernel when the production Inkling shape applies.
    SGLANG_OPT_USE_GATE_TOPK_JIT = EnvBool(True)
    # Inside the fused gate: replace the cublas gate linear with the
    # expert-per-block GEMV JIT kernel at small token counts (GateGemvMode).
    SGLANG_OPT_GATE_GEMV_MODE = EnvInt(GateGemvMode.PAIR)
    # Capture all multi-layer EAGLE draft-extend steps and the in-graph chain
    # rotation into ONE CUDA graph instead of one captured graph per step.
    SGLANG_ENABLE_SINGLE_CG_DRAFT = EnvBool(True)
    # Draft sampler uses the Gumbel-max trick (argmax(probs / Exp(1))) instead of
    # torch.multinomial, whose device-side validity assert breaks draft-graph replay.
    SGLANG_OPT_USE_GUMBEL_SAMPLE = EnvBool(True)
    # Multi-layer chain-MTP boundary-KV fix: widen the draft-extend window to
    # rewrite rejected-draft KV rows before reuse (acc_len repair; on by default).
    SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX = EnvBool(True)
    SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHEARED_BIAS = EnvBool(True)
    # Use feature-stacked GEMMs for the no-LoRA BF16 shared sink. Eligible LoRA
    # serving enables this layout independently of the flag.
    SGLANG_OPT_LINEARIZED_SHARED_SINK = EnvBool(True)
    # Use the autotuned JIT all-reduce, falling back to torch multimem for
    # shapes where it wins.
    SGLANG_OPT_USE_INKLING_CUSTOM_AR = EnvBool(True)
    # Fuse small-batch decode all-reduce, MLP convolution, and attention norm.
    # Requires the custom all-reduce; other shapes use the unfused path.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM = EnvBool(True)
    # Fuse eligible extend all-reduce, convolution, and cache updates.
    # Supports scattered or full-width state and requires the custom all-reduce.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV = EnvBool(True)
    # Fuse eligible convolution, QK norm, window, and KV-store prologue work.
    # Non-BF16 caches retain the backend KV store.
    SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE = EnvBool(True)
    # Override shared-expert selection: true uses grouped GEMM, false uses BMM.
    # When unset, selection follows model, quantization, and LoRA requirements.
    SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE = EnvBool(True)
    # Fold the conditional long-context log-scaling tau into its producers
    # instead of separate output-sized scale kernels: the fused attn
    # prologue's q path (bit-exact, before MXFP8 quantization there) and the
    # rel_logits projection's r OPERAND (the diagonal scale commutes through
    # the einsum, shrinking the pass by rel_extent/d_rel = 64x; rounding moves
    # before the GEMM). Flag-off keeps the standalone apply_log_scaling_tau
    # on the outputs.
    # Fold the MoE shared-expert partials into the custom AR kernels instead
    # of a separate {routed + shared} torch.add per MoE layer; some buckets
    # keep a pre-add during the AR stage-in. torch.add numerics
    # (bit-identical). Requires SGLANG_OPT_USE_INKLING_CUSTOM_AR.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU = EnvBool(True)
    # Dispatch the rel_logits projection around einsum's hidden compaction
    # copy of the strided r operand (a view into the packed qkvr output):
    # zero-copy strided-batched matmul at small t, JIT row-compact + einsum
    # above the band, single-launch tau-folded kernel in the small-t tau
    # band. Bit-identical to the plain einsum; flag-off restores it.
    SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH = EnvBool(True)
    # Quantize and store MXFP8 K/V data and scales in one fused kernel.
    SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE = EnvBool(True)
    # Default reasoning effort in [0.0, 0.99] when omitted by a request.
    # An empty string falls back to the protocol default (0.9); the effort
    # directive is always emitted.
    SGLANG_INKLING_DEFAULT_REASONING_EFFORT = EnvStr("0.9")
    SGLANG_INKLING_RS_MM_PREPROCESS = EnvBool(True)

    # ===================================================================
    # DSA backend (GLM 5 and DeepSeek V3.2)
    # ===================================================================
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_FUSE_TOPK"
    )
    # Enabled for supported CUDA KPool geometry; set to 0 to use ordinary metadata.
    SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION = EnvBool(True)
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: fuse the
    # absorbed q bmm with the nope/rope concat + fp8 cast so q is written
    # directly in fp8 ("born fp8") and the standalone concat-cast kernel
    # disappears.  Not bit-exact vs the default path (same rounding stages,
    # different GEMM accumulation order), hence default OFF until accuracy-
    # gated (oracle + full-set gsm8k).
    SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: pass a
    # per-row valid-topk count (derived from the trailing -1 pad run of the
    # topk indices) so the kernel skips whole pad-only topk blocks instead of
    # computing masked zero contributions.  Bit-exact by construction: skipped
    # blocks contain only -1 pads, and -1 entries inside the consumed range
    # still take the in-kernel clamp+mask path.
    SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH = EnvBool(False)
    # Opt-in: run the born-fp8 q-prep (absorbed bmm + concat + fp8 cast,
    # ~173us/layer-call) on alt_stream underneath the DSA indexer — the two
    # chains fork independently from the q_a_layernorm output.  Requires
    # SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q; eager-prefill-only via the born
    # predicate.  Coarse per-layer join keeps the single-slot born-q buffer
    # WAR-safe.
    SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = EnvBool(False)
    # Opt-in: fuse the Q8KV8 non-prefix KV prep — cast-concat k/k_rope
    # directly into the persistent fp8 kv buffer and zero the pad band in one
    # Triton kernel (replaces bf16 _cat + copy_ cast + zero_ tail).
    SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION = EnvBool(False)
    # Q8KV8 born-fp8 q-prep codegen: "auto" = per-K Triton dispatch (default);
    # "cuda" = the hand-written SM90 WGMMA kernel (bitwise identical to the
    # Triton two_dot variant, 1.16-1.38x faster across GLM/DS shapes).
    SGLANG_OPT_Q8KV8_QPREP_VARIANT = EnvStr("auto")

    # ===================================================================
    # MiniMax M3
    # ===================================================================
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)
    # Kill switch for the derived fp8 attention-GEMM mode (m3_fp8_attn_gemm_enabled):
    # forces the pre-fp8 behavior (bf16 indexer + widening sparse path, bf16 q)
    # even when kv_cache_dtype fp8_e4m3 + trtllm_mha + SM100 would activate it.
    SGLANG_DISABLE_M3_FP8_ATTN_GEMM = EnvBool(False)
    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)
    # Fused JIT store (minimax_store_kv_index) of main+index K/V instead of separate
    # set_*_buffer copies; falls back when main/index dtypes differ or non-CUDA.
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(True)
    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)

    # MiniMax-M3 sparse-attention toggles for ROCm.
    # Share one index top-k across every N sparse layers; 1 disables sharing.
    # Changes which KV blocks the skip layers attend, so it applies on ROCm only
    # (never under two-batch overlap); elsewhere the backend pins 1.
    # 2 is the accuracy-safe default: higher values reuse staler selections
    # in the skip layers.
    SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ = EnvInt(2)
    # MiniMax M3 NPU prefill MAIN-attention: route the sparse main attention through
    # the native Ascend FA op `torch.ops.npu.npu_fused_infer_attention_score` (FIA)
    # with a per-query CUSTOM block_table
    SGLANG_MINIMAX_NPU_PREFILL_FIA = EnvBool(True)
    # MiniMax-M3 NPU sparse INDEXER (decode + verify topk block selection): route
    # through the native AscendC packed indexer op instead of the Triton indexer.
    SGLANG_MINIMAX_NPU_NATIVE_INDEXER = EnvBool(False)
    # MiniMax-M3 NPU sparse MAIN-attention (decode-main + verify-main): route the
    # sparse main attention through the native AscendC sparse-attention op with the
    # cached block_table override.
    SGLANG_MINIMAX_NPU_NATIVE_ATTN = EnvBool(False)
    # MiniMax-M3 on ROCm force-disables custom all-reduce in its model override
    # (arg_groups/overrides.py) when aiter all-reduce fusion is off. Set this to
    # opt back in and keep custom/quick all-reduce enabled -- e.g. to run the
    # INT4 quick-reduce path via ROCM_QUICK_REDUCE_QUANTIZATION={INT4,INT6,INT8}.
    SGLANG_M3_ALLOW_CUSTOM_AR = EnvBool(False)

    # ===================================================================
    # Kimi K3
    # ===================================================================
    # MNNVL fused all-reduce (bf16, TP8): zero-copy 1shot multicast-push for
    # small messages and in-place NVLS 2shot on symmetric-memory tensors for
    # large ones, with an optional fused residual add. Covers the KDA o_proj
    # output and the latent|shared MoE reduce; everything else falls back to
    # the regular all-reduce path. Auto-enabled on SM100/SM103 when
    # CustomAllReduceV2 with multicast is available; set 0/1 to override in
    # either direction. See srt/layers/k3_ar_fusion.py.
    SGLANG_K3_AR_FUSION = EnvBool(False)
    # K3 SP-MoE fused residual + reduce-scatter and matching all-gather over
    # CustomAllReduceV2's MNNVL push workspace. Auto-probed for the validated
    # TP8 GB300 configuration; set 0/1 to override. See
    # srt/layers/k3_sp_collective.py.
    SGLANG_K3_SP_COLLECTIVE = EnvBool(False)
    # Keep K3's post-MoE residual stream token-sharded between consecutive
    # SP-MoE layers. The next attention-residual aggregation and snapshot
    # bank write run on the local shard, then only the normalized attention
    # input is all-gathered. Requires SGLANG_K3_SP_COLLECTIVE.
    SGLANG_K3_SP_ATTN_RES = EnvBool(False)
    # Fused o_proj GEMM + all-reduce (bf16, TP 2..8, SM100+): one
    # kernel computes the TP-local o_proj partial and the cross-rank sum over
    # a P2P comm region, replacing the GEMM + NCCL AR pair at M <= 512.
    SGLANG_K3_GEMM_AR = EnvBool(False)
    # Merge the router gate and routed_expert_down_proj weights so the K3 MoE
    # front reads hidden_states once, and run the top-k plus the bf16 cast in one
    # epilogue kernel. See kernels/ops/moe/moe_front.py. Default on.
    SGLANG_K3_FUSED_FRONT = EnvBool(True)
    # Use the ROCm radix-4 router for covered K3 top-k workloads.
    SGLANG_K3_RADIX4_TOPK = EnvBool(False)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN = EnvInt(6144)

    # ===================================================================
    # Symmetric memory
    # ===================================================================
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Qwen3.5 and GDN
    SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV = EnvBool(True)
    SGLANG_TRACE_QWEN35_FINAL_NORM = EnvBool(False)
    SGLANG_QWEN35_NATIVE_FINAL_NORM = EnvBool(False)
    # One switch enables deferred MoE finalize and AR + residual + RMSNorm.
    SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION = EnvBool(False)
    # Distinct workspace configurations allowed in one process. Production
    # uses one model/configuration per rank, so fail closed on accidental reuse.
    SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES = EnvInt(1)

    # ===================================================================
    # Plugin system
    # ===================================================================
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")

    # ===================================================================
    # KV-Canary and Token-Oracle (testing only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)

    # ===================================================================
    # Rust server
    # ===================================================================
    SGLANG_RUST_SERVER = EnvBool(False)
    # Build a missing Rust extension from source (auto), require a bundled or
    # cached extension (never), or rebuild the local cache entry (force).
    SGLANG_RUST_BUILD_MODE = EnvStr("auto")
    # Most batched requests one /generate HTTP call may expand into.
    SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ = EnvInt(4096)

    # ===================================================================
    # Weight Cache Daemon
    # ===================================================================
    # Paths the daemon and the engine ranks it serves must agree on. Both are
    # format templates and must keep the {device_uuid} placeholder: each daemon
    # is keyed by the physical GPU it runs on, so a GPU-independent path would
    # let one job's client discover another job's daemon.
    SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.sock"
    )
    SGLANG_WEIGHT_CACHE_READY_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.ready"
    )


envs = Envs()
EnvField._allow_set_name = False


def exportable_env_vars() -> dict[str, str]:
    return {
        field.name: _exportable_value(os.environ[field.name])
        for field in sorted(
            (value for value in vars(Envs).values() if isinstance(value, EnvField)),
            key=lambda field: field.name,
        )
        if not field.secret and field.name in os.environ
    }


def _exportable_value(value: str) -> str:
    try:
        value.encode()
    except UnicodeEncodeError:
        return (
            _NON_UTF8_PREFIX
            + base64.b64encode(value.encode(errors="surrogateescape")).decode()
        )
    return value


class _DeprecatedEnv:
    """One deprecated env var: warn if it is set, and optionally forward its
    (possibly transformed) value to a replacement env var."""

    def __init__(
        self,
        replacement: Optional[str] = None,
        transform: Optional[Callable[[str], str]] = None,
        note: Optional[str] = None,
    ):
        self.replacement = replacement
        self.transform = transform
        self.note = note

    def apply(self, old_name: str):
        if old_name not in os.environ:
            return
        message = f"Environment variable {old_name} is deprecated."
        if self.replacement is not None:
            message += f" Please use {self.replacement} instead."
        if self.note is not None:
            message += f" {self.note}"
        warnings.warn(message)
        if self.replacement is not None:
            value = os.environ[old_name]
            if self.transform is not None:
                value = self.transform(value)
            os.environ[self.replacement] = value


def _ms_to_s(value: str) -> str:
    return str(float(value) / 1000.0)


def _invert_bool(value: str) -> str:
    return "0" if value.lower() in ("true", "1", "yes", "y") else "1"


# The single registry for deprecated environment variables, processed once at
# import by _handle_deprecated_envs(). Add new deprecations here instead of
# ad-hoc warnings. For a rename where the old name must keep working through a
# descriptor, use EnvBoolWithAlias / EnvIntWithAlias instead.
_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {
    # Renamed: the value is forwarded to the replacement.
    "SGLANG_GC_LOG": _DeprecatedEnv(replacement="SGLANG_LOG_GC"),
    "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH": _DeprecatedEnv(
        replacement="SGLANG_MOE_NVFP4_DISPATCH"
    ),
    "SGLANG_ENABLE_THINKING": _DeprecatedEnv(replacement="SGLANG_DEFAULT_THINKING"),
    "SGLANG_REASONING_EFFORT": _DeprecatedEnv(
        replacement="SGLANG_DSV4_REASONING_EFFORT"
    ),
    "SGLANG_USE_JIT_ALL_REDUCE": _DeprecatedEnv(
        replacement="SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    ),
    # The legacy DISABLE flags have the opposite polarity of their replacement.
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": _DeprecatedEnv(
        replacement="SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", transform=_invert_bool
    ),
    # Renamed with a unit change.
    "SGLANG_QUEUED_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_WAITING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    "SGLANG_FORWARD_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_RUNNING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    # Removed without replacement.
    "SGLANG_ENABLE_CP_V2": _DeprecatedEnv(
        note="Strategy-based prefill context parallelism is now the only generic implementation."
    ),
    "SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2": _DeprecatedEnv(),
    # Superseded by the unified JIT per_token_group_quant, the default CUDA path.
    "SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT": _DeprecatedEnv(),
    "SGLANG_MASKED_GEMM_FAST_ACT": _DeprecatedEnv(),
    # The unified free list is kept unsorted between flushes by design; the
    # sort-after-merge A/B knob never left its off default and is gone.
    "SGLANG_SORT_FREE_LIST_AFTER_MERGE": _DeprecatedEnv(),
    "SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN": _DeprecatedEnv(),
    # sconv-family kernels always use the CUDA-JIT ports when supported; no toggle.
    "SGLANG_OPT_USE_CUDA_SCONV": _DeprecatedEnv(),
    # DSV4 compressor V2 is always used.
    "SGLANG_OPT_USE_COMPRESSOR_V2": _DeprecatedEnv(),
    "SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK": _DeprecatedEnv(
        note="Buffer-mode anchor pinning is always on; set "
        "SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP=0 to disable it."
    ),
    # Replaced by CLI flags.
    "SGLANG_ENABLE_GRPC": _DeprecatedEnv(
        note="Please use '--grpc-port' to enable the native gRPC server."
    ),
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": _DeprecatedEnv(
        note="Please use '--enable-prefill-delayer' instead."
    ),
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": _DeprecatedEnv(
        note="Please use '--prefill-delayer-max-delay-passes' instead."
    ),
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": _DeprecatedEnv(
        note="Please use '--prefill-delayer-token-usage-low-watermark' instead."
    ),
    "SGLANG_CUTLASS_MOE": _DeprecatedEnv(
        note="Please use '--moe-runner-backend=cutlass' and/or "
        "'--speculative-moe-runner-backend=cutlass' instead."
    ),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": _DeprecatedEnv(
        note="Please use '--enable-w4a4-mxfp4-megamoe' instead."
    ),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": _DeprecatedEnv(
        note="Please use '--enable-w4a4-mxfp4-megamoe' instead."
    ),
    "SGLANG_DFLASH_PREFILL_REFILL_TARGET": _DeprecatedEnv(
        note="DFlash now auto-enables the min-free-slots delay; unset this env. "
        "To override the threshold, use '--min-free-slots-delay'."
    ),
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": _DeprecatedEnv(
        note="The unified radix tree is the default tree cache now; unset this "
        "env. The field is still defined for legacy call sites."
    ),
}


def _handle_deprecated_envs():
    for old_name, deprecation in _DEPRECATED_ENVS.items():
        deprecation.apply(old_name)

    # Rewrite the legacy SGL_ prefix to SGLANG_ (names not covered above).
    for key, value in list(os.environ.items()):
        if key.startswith("SGL_") and key not in _DEPRECATED_ENVS:
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


def third_party_cache_defaults() -> Dict[str, str]:
    base = os.path.expanduser(envs.SGLANG_CACHE_DIR.get())
    return {
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "CUDA_CACHE_PATH": os.path.join(base, "nv"),
        # TileLang compiles the DeepSeek-V4 MHC prenorm kernels; left at its own
        # default the burst is invisible to anyone warming, mounting or baking
        # SGLANG_CACHE_DIR, and gets paid again on every cold container.
        "TILELANG_CACHE_DIR": os.path.join(base, "tilelang"),
        # FlashInfer appends ".cache/flashinfer" to this base itself, so this
        # is the base dir rather than the final cache dir.
        "FLASHINFER_WORKSPACE_BASE": base,
    }


def redirect_third_party_caches():
    """Point third-party JIT caches at SGLANG_CACHE_DIR, so a run's compiled
    kernels can be cleaned, warmed or volume-mounted as one directory.

    Must be called early. The redirect silently does nothing if either of
    these has already happened:

    - FlashInfer was imported. It resolves its workspace at import time.
    - Inductor made its first ``cache_dir()`` call. That call setdefaults
      TORCHINDUCTOR_CACHE_DIR itself.
    """
    for key, value in third_party_cache_defaults().items():
        os.environ.setdefault(key, value)


_handle_deprecated_envs()

# Trigger auto-injection of CUDA coredump env vars when SGLANG_CUDA_COREDUMP=1.
# Best-effort; for strict guarantees, set CUDA_* env vars in the shell before
# launching Python. Imported conditionally to keep the default import of this
# module free of non-stdlib side effects.
if envs.SGLANG_CUDA_COREDUMP.get():
    import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip
