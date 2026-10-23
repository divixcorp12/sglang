"""Env names the expert-engine rename removed, and the startup check that refuses a stale script.

The names are not aliased (``docs/hx/F145-env-merge-table.draft.md``, "Removed-name table"): setting one fails startup
with the name that replaces it, so a launch script cannot quietly lose its settings. ``check_removed_env`` must run
once, at process start, before the config is resolved; it reads only the environment passed in, so it works on a
launcher's merged env dict as well as on ``os.environ``.

Names retired earlier with no replacement (the lease-chain modes, the doorbell, ``SGLANG_DSV41_RAM_MISS_HIT_WAIT_US``)
are not here: ``environ._DEPRECATED_ENVS`` still warns and ignores them.

Stdlib and msgspec only, so a launcher can load this file by path without importing sglang.
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import msgspec


class Replacement(msgspec.Struct, frozen=True):
    """What took over a removed env: the env name now carrying it, the config field, and how the value changes."""

    env: str
    field: str
    note: str = ""


def _R(env: str, field: str, note: str = "") -> Replacement:
    return Replacement(env, field, note)


# The env that lists the vendored-kernel names (``EXL3_MOE_CPU_PIN``, ...) the engine itself set from its config, as a
# comma-separated list: the vendored kernel reads those names with getenv, so they stay in the process and its
# children, and ``check_removed_env`` must not mistake them for a stale launch script.
VENDORED_ENV_MARKER = "SGLANG_EXL3_VENDORED_ENV"


class RemovedEnvError(ValueError):
    """A removed env name is set; the message names each one and its replacement."""


# A group env is one JSON object decoded into the config struct at this path; its members are keyed by their field
# path below it.
GROUP_PATHS = {
    "SGLANG_EXPERT_STREAM_TRACE": "trace",
    "SGLANG_EXPERT_STREAM_URING": "tiers.uring",
    "SGLANG_EXPERT_STREAM_MIRROR": "tiers.mirror",
    "SGLANG_EXPERT_STREAM_CPU_SPLIT": "cpu_experts.split",
    "SGLANG_EXPERT_STREAM_CPU_TWO_STAGE": "cpu_experts.two_stage",
    "SGLANG_EXPERT_STREAM_SHADOW_PREFETCH": "prefetch.shadow",
}

# Old name -> replacement. ``field`` is the dotted path under ExpertStreamConfig, or ``exl3.`` for the EXL3 format
# config (Exl3FormatConfig). Merged names appear once per old name, so several olds can name the same replacement.
REMOVED_ENV: dict[str, Replacement] = {
    "SGLANG_MOE_EXPERT_STREAM": _R("SGLANG_EXPERT_STREAM_ENABLE", "enabled", 'set it to the same 1/0'),
    "SGLANG_DSV41_EXPERT_STREAM": _R("SGLANG_EXPERT_STREAM_ENABLE", "enabled", 'set it to the same 1/0'),
    "SGLANG_DSV41_EXPERT_DIR": _R("SGLANG_EXPERT_STREAM_EXPERT_DIR", "tiers.expert_dir"),
    "SGLANG_MOE_EXPERT_FILE_DIR": _R("SGLANG_EXPERT_STREAM_FILE_CACHE_DIR", "tiers.file_cache_dir"),
    "SGLANG_MOE_PINNED_HOST_MB": _R("SGLANG_EXPERT_STREAM_PINNED_HOST_MB", "tiers.pinned_host_mb"),
    "SGLANG_MOE_PINNED_HOST_NUMA_MB": _R("SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB", "tiers.pinned_host_numa_mb"),
    "SGLANG_MOE_PINNED_HOST_FULL_LAYERS": _R("SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS", "tiers.pinned_host_full_layers"),
    "SGLANG_MOE_EXPERT_HOST_ARENA": _R("SGLANG_EXPERT_STREAM_ENABLE_HOST_ARENA", "tiers.host_arena"),
    "SGLANG_MOE_EXPERT_FILE_READER": _R("SGLANG_EXPERT_STREAM_FILE_READER", "tiers.file_reader"),
    "SGLANG_MOE_EXPERT_ROW_SOURCE": _R("SGLANG_EXPERT_STREAM_ROW_SOURCE", "tiers.row_source"),
    "SGLANG_MOE_EXPERT_MIRROR_DIRS": _R("SGLANG_EXPERT_STREAM_MIRROR", "tiers.mirror.dirs"),
    "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS": _R("SGLANG_EXPERT_STREAM_MIRROR", "tiers.mirror.weights"),
    "SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": _R("SGLANG_EXPERT_STREAM_MIRROR", "tiers.mirror.dynamic"),
    "SGLANG_MOE_EXPERT_MIRROR_CAPS": _R("SGLANG_EXPERT_STREAM_MIRROR", "tiers.mirror.caps"),
    "SGLANG_EXPERT_NUMA_CORES": _R("SGLANG_EXPERT_STREAM_NUMA_CORES", "tiers.numa_cores"),
    "SGLANG_DSV41_CPU_EXPERTS_CORES": _R("SGLANG_EXPERT_STREAM_NUMA_CORES", "tiers.numa_cores", '"<node of the cores>:cpu=<list>"'),
    "SGLANG_DSV41_RAM_MISS_SPIN_CORE": _R("SGLANG_EXPERT_STREAM_NUMA_CORES", "tiers.numa_cores", '"<node of the core>:ram=<core>"'),
    "SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU": _R("SGLANG_EXPERT_STREAM_NUMA_CORES", "tiers.numa_cores", '"<the tier\'s node>:sq=<core>"'),
    "SGLANG_EXPERT_STREAM_URING_MODE": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.mode"),
    "SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.queue_depth"),
    "SGLANG_EXPERT_STREAM_URING_FIXED_FILES": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.fixed_files"),
    "SGLANG_EXPERT_STREAM_URING_READ_MODE": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.read_mode"),
    "SGLANG_EXPERT_STREAM_URING_WAIT_MODE": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.wait_mode"),
    "SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.sq_thread_idle_ms"),
    "SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.diagnostics"),
    "SGLANG_EXPERT_STREAM_URING_READ_CUTS": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.read_cuts"),
    "SGLANG_EXPERT_STREAM_URING_SLAB_ARENA": _R("SGLANG_EXPERT_STREAM_URING", "tiers.uring.slab_arena"),
    "SGLANG_MOE_EXPERT_COPY_BACKEND": _R("SGLANG_EXPERT_STREAM_COPY_MODE", "copy.mode", 'gpu -> sm_copy, dma -> dma'),
    "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": _R("SGLANG_EXPERT_STREAM_COPY_MODE", "copy.mode", '1 -> copy_engine, 0 -> sm_copy'),
    "SGLANG_DSV41_RAM_HIT_COPY": _R("SGLANG_EXPERT_STREAM_RAM_HIT_COPY", "copy.ram_hit_copy"),
    "SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES": _R("SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES", "copy.sm_small_copies"),
    "SGLANG_DSV41_ENABLE_LEASE_PDL": _R("SGLANG_EXPERT_STREAM_ENABLE_LEASE_PDL", "copy.lease_pdl"),
    "SGLANG_DSV41_RAM_MISS_TIMEOUT_MS": _R("SGLANG_EXPERT_STREAM_RAM_MISS_TIMEOUT_MS", "copy.timeout_ms"),
    "SGLANG_DSV41_RAM_MISS_IDLE_SPIN_US": _R("SGLANG_EXPERT_STREAM_RAM_MISS_IDLE_SPIN_US", "copy.idle_spin_us"),
    "SGLANG_DSV41_RAM_MISS_COPY_IDLE_SPIN_US": _R("SGLANG_EXPERT_STREAM_COPY_IDLE_SPIN_US", "copy.copy_idle_spin_us"),
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": _R("SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER", "copy.graph_gather"),
    "SGLANG_MOE_EXPERT_FUSED_PLAN": _R("SGLANG_EXPERT_STREAM_ENABLE_FUSED_PLAN", "copy.fused_plan"),
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS": _R("SGLANG_EXPERT_STREAM_GRAPH_GATHER_SCRATCH_ROWS", "copy.graph_gather_scratch_rows"),
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": _R("SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES", "copy.graph_gather_miss_lanes"),
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": _R("SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES", "copy.graph_gather_victim_lanes"),
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS": _R("SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS", "copy.prefill.fills"),
    "SGLANG_DSV41_ENABLE_PREFILL_SHARE": _R("SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE", "copy.prefill.share"),
    "SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS": _R("SGLANG_EXPERT_STREAM_FORCE_PREFILL_SHARE_ROWS", "copy.prefill.share_rows"),
    "SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN": _R("SGLANG_EXPERT_STREAM_ENABLE_PREFILL_ROUTE_PLAN", "copy.prefill.route_plan"),
    "SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER": _R("SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SPLIT_GATHER", "copy.prefill.split_gather"),
    "SGLANG_DSV41_GPU_DRIP_SLOTS": _R("SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS", "copy.gpu_drip.slots"),
    "SGLANG_DSV41_GPU_DRIP_PICKS": _R("SGLANG_EXPERT_STREAM_GPU_DRIP_PICKS", "copy.gpu_drip.picks"),
    "SGLANG_DSV41_GPU_DRIP_CHUNK_KB": _R("SGLANG_EXPERT_STREAM_GPU_DRIP_CHUNK_KB", "copy.gpu_drip.chunk_kb"),
    "SGLANG_MOE_HOT_GPU_MB": _R("SGLANG_EXPERT_STREAM_HOT_GPU_MB", "hot_cache.gpu_mb"),
    "SGLANG_MOE_HOT_SEED": _R("SGLANG_EXPERT_STREAM_HOT_SEED", "hot_cache.seed_path"),
    "SGLANG_MOE_HOT_LOG_INTERVAL": _R("SGLANG_EXPERT_STREAM_HOT_LOG_INTERVAL", "hot_cache.log_interval"),
    "SGLANG_MOE_HOT_METRICS_FILE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.hot_metrics_file"),
    "SGLANG_MOE_HOT_DYNAMIC": _R("SGLANG_EXPERT_STREAM_ENABLE_HOT_DYNAMIC", "residency.dynamic"),
    "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": _R("SGLANG_EXPERT_STREAM_HOT_UPDATE_PREFILL_TOKENS", "residency.update_prefill_tokens"),
    "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": _R("SGLANG_EXPERT_STREAM_HOT_UPDATE_DECODE_FORWARDS", "residency.update_decode_forwards"),
    "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": _R("SGLANG_EXPERT_STREAM_HOT_MIN_RESIDENCE_FORWARDS", "residency.min_residence_forwards"),
    "SGLANG_MOE_HOT_BENEFIT_RATIO": _R("SGLANG_EXPERT_STREAM_HOT_BENEFIT_RATIO", "residency.benefit_ratio"),
    "SGLANG_MOE_HOT_DECAY_TOKENS": _R("SGLANG_EXPERT_STREAM_HOT_DECAY_TOKENS", "residency.decay_tokens"),
    "SGLANG_MOE_HOT_PROMOTION_SIGMAS": _R("SGLANG_EXPERT_STREAM_HOT_PROMOTION_SIGMAS", "residency.promotion_sigmas"),
    "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": _R("SGLANG_EXPERT_STREAM_ENABLE_HOT_ASYNC_PROMOTIONS", "residency.async_promotions"),
    "SGLANG_MOE_ASYNC_RESIDENCY_SCORES": _R("SGLANG_EXPERT_STREAM_ENABLE_ASYNC_RESIDENCY_SCORES", "residency.async_scores"),
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": _R("SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE", "residency.gpu_update"),
    "SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS": _R("SGLANG_EXPERT_STREAM_GPU_RESIDENCY_MAX_PROMOTIONS", "residency.gpu_max_promotions"),
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": _R("SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE", "residency.insert_on_miss_stage"),
    "SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY": _R("SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_DECAY", "residency.insert_on_miss_decay"),
    "SGLANG_MOE_HOT_FUSED_INSERT": _R("SGLANG_EXPERT_STREAM_ENABLE_HOT_FUSED_INSERT", "residency.fused_insert"),
    "SGLANG_DSV41_RAM_PREFETCH": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH", "prefetch.ram.enabled"),
    "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_TOKEN", "prefetch.ram.per_token"),
    "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_LAYER", "prefetch.ram.per_layer"),
    "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE", "prefetch.ram.spec_share"),
    "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_TOP_K_ONLY", "prefetch.ram.top_k_only"),
    "SGLANG_DSV41_RAM_PREFETCH_SCORER": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER", "prefetch.ram.scorer"),
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE", "prefetch.ram.idle_drive"),
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_DEADLINE_US", "prefetch.ram.idle_deadline_us"),
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_SPREAD", "prefetch.ram.idle_spread"),
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES", "prefetch.ram.idle_pieces"),
    "SGLANG_DSV41_RAM_PREFETCH_PROMOTE": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_PROMOTE", "prefetch.ram.promote"),
    "SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE": _R("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_POOL_SERVE", "prefetch.ram.pool_serve"),
    "SGLANG_DSV41_RAM_PREFETCH_HORIZON": _R("SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON", "prefetch.ram.horizon"),
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": _R("SGLANG_EXPERT_STREAM_LEGACY_PREFETCH_MAX_CANDIDATES", "prefetch.legacy_max_candidates"),
    "SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.predictor"),
    "SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.model_dir"),
    "SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.candidates"),
    "SGLANG_MOE_EXPERT_PREFETCH_BUDGET": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.budget"),
    "SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.apex_tau"),
    "SGLANG_MOE_EXPERT_PREFETCH_PULL": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.pull_mode", '1 -> {"pull_mode": "always"}, 0 -> {"pull_mode": "off"}'),
    "SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.pull_mode"),
    "SGLANG_MOE_EXPERT_PREFETCH_SHADOW_RECALL": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.shadow_recall"),
    "SGLANG_MOE_EXPERT_PREFETCH_FUSED_TOP1": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.fused_top1"),
    "SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.calibration"),
    "SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_FILE": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.calibration_file"),
    "SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_PROVENANCE": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.calibration_provenance"),
    "SGLANG_MOE_EXPERT_PREDICTOR": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.predictors"),
    "SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.predictor_candidates"),
    "SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.score_interval"),
    "SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS": _R("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", "prefetch.shadow.max_rows"),
    "SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_log_interval"),
    "SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_metrics_file"),
    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_capture.dir"),
    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_capture.max_gb"),
    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_capture.shard_rows"),
    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.predictor_capture.frames"),
    "SGLANG_DSV41_CPU_EXPERTS": _R("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS", "cpu_experts.enabled"),
    "SGLANG_DSV41_CPU_EXPERTS_THREADS": _R("SGLANG_EXPERT_STREAM_CPU_EXPERTS_THREADS", "cpu_experts.threads"),
    "SGLANG_DSV41_ENABLE_CPU_EXPERTS_SMT": _R("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_SMT", "cpu_experts.smt"),
    "SGLANG_DSV41_CPU_EXPERTS_SPLIT": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.table"),
    "SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.layer_tables"),
    "SGLANG_DSV41_CPU_EXPERTS_CPU_MS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.cpu_ms"),
    "SGLANG_DSV41_CPU_EXPERTS_LINK_MS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.link_ms"),
    "SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.handoff_ms"),
    "SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.retune_batches"),
    "SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.calibration"),
    "SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.calibration_reps"),
    "SGLANG_DSV41_CPU_EXPERTS_MISSES": _R("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_MISSES", "cpu_experts.misses"),
    "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.miss_cut"),
    "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.miss_cut_max"),
    "SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.model"),
    "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.model_costs"),
    "SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.model_refine_batches"),
    "SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK": _R("SGLANG_EXPERT_STREAM_CPU_SPLIT", "cpu_experts.split.model_residency_tiebreak"),
    "SGLANG_DSV41_CPU_TWO_STAGE": _R("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE", "cpu_experts.two_stage.enabled"),
    "SGLANG_DSV41_CPU_TWO_STAGE_W13_WEIGHTS": _R("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE", "cpu_experts.two_stage.w13_weights"),
    "SGLANG_DSV41_CPU_TWO_STAGE_W2_WEIGHTS": _R("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE", "cpu_experts.two_stage.w2_weights"),
    "SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US": _R("SGLANG_EXPERT_STREAM_CPU_EXPERTS_KEEP_WARM_US", "cpu_experts.keep_warm_us"),
    "SGLANG_DSV41_CPU_EXPERTS_STORE_OUT": _R("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT", "cpu_experts.store_out"),
    "EXL3_MOE_CPU_MAX_ISA": _R("SGLANG_EXPERT_STREAM_CPU_MAX_ISA", "cpu_experts.max_isa"),
    "NVFP4_CPU_MAX_ISA": _R("SGLANG_EXPERT_STREAM_CPU_MAX_ISA", "cpu_experts.max_isa"),
    "EXL3_MOE_CPU_REPORT_ISA": _R("SGLANG_EXPERT_STREAM_CPU_REPORT_ISA", "cpu_experts.report_isa"),
    "NVFP4_CPU_REPORT_ISA": _R("SGLANG_EXPERT_STREAM_CPU_REPORT_ISA", "cpu_experts.report_isa"),
    "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": _R("SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS", "draft.cpu_experts"),
    "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": _R("SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_CORES", "draft.cores"),
    "SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS": _R("SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_THREADS", "draft.threads"),
    "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": _R("SGLANG_EXPERT_STREAM_DRAFT_RESIDENT_PATH", "draft.resident_path"),
    "SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_routes_path"),
    "SGLANG_DSV41_EXPERT_TRACE_PATH": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.expert_path"),
    "SGLANG_DSV41_ROUTER_CAPTURE_PATH": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.router_capture_path"),
    "SGLANG_DSV41_RESIDENCY_CAPTURE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.residency_capture"),
    "SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.verify_accept_path"),
    "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.job.prefix"),
    "SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.job.capacity"),
    "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.job.resources"),
    "SGLANG_DRAFT_ARRIVAL_TRIGGER_US": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_trigger.arrival_us"),
    "SGLANG_DRAFT_PENDING_TRIGGER_US": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_trigger.pending_us"),
    "SGLANG_DRAFT_FORWARD_TRIGGER_US": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_trigger.forward_us"),
    "SGLANG_DRAFT_DELAY_TRIGGER_REPORT_PREFIX": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_trigger.report_prefix"),
    "SGLANG_DRAFT_DELAY_TRIGGER_ONLY": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.draft_trigger.trigger_only"),
    "SGLANG_CPU_EXPERT_TRACE_GATE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.gate_path"),
    "SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.cpu_worker.prefix"),
    "SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.cpu_worker.capacity"),
    "SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.cpu_worker.min_us"),
    "SGLANG_EXL3_CPU_SCRATCH_TRACE_PREFIX": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.cpu_worker.scratch_prefix"),
    "SGLANG_DSV41_SYNC_WAIT_NVTX": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.sync_wait_nvtx"),
    "SGLANG_MOE_PREFORWARD_READINESS_DIAGNOSTIC": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.preforward_readiness.enabled"),
    "SGLANG_MOE_PREFORWARD_READINESS_PATH": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.preforward_readiness.path"),
    "SGLANG_MOE_ROUTE_TRACE_DIR": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.route.dir"),
    "SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.route.max_tokens"),
    "SGLANG_MOE_ROUTE_TRACE_SPECULATIVE": _R("SGLANG_EXPERT_STREAM_TRACE", "trace.route.speculative"),
    "EXL3_MOE_CPU_PIN": _R("SGLANG_EXL3_MOE_CPU_PIN", "exl3.cpu.pin"),
    "EXL3_MOE_CPU_SMALL_WORKERS": _R("SGLANG_EXL3_MOE_CPU_SMALL_WORKERS", "exl3.cpu.small_workers"),
    "SGLANG_MOE_HOT_INSERT_ON_MISS": _R("SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE", "residency.insert_on_miss_stage", 'the old boolean 1 means stage 1 (scratch)'),
    "SGLANG_DSV41_EXPERT_RAM_GIB": _R("SGLANG_EXPERT_STREAM_PINNED_HOST_MB", "tiers.pinned_host_mb", 'the old value was GiB; the new one is MiB'),
}


def _group_snippet(group_env: str, field: str) -> str:
    """The JSON a group env takes for ``field``, with ``<value>`` where the old env's value goes."""
    keys = field[len(GROUP_PATHS[group_env]) + 1 :].split(".")
    snippet = "<value>"
    for key in reversed(keys):
        snippet = "{" + json.dumps(key) + ": " + snippet + "}"
    return f"{group_env}='{snippet}'"


def describe(old_name: str) -> str:
    """One line saying what replaces ``old_name`` (a name in REMOVED_ENV)."""
    replacement = REMOVED_ENV[old_name]
    if replacement.env in GROUP_PATHS:
        use = _group_snippet(replacement.env, replacement.field)
    else:
        use = replacement.env
    line = f"{old_name} -> {use} (config field {replacement.field})"
    return f"{line}: {replacement.note}" if replacement.note else line


def check_removed_env(environ: Optional[Mapping[str, str]] = None) -> None:
    """Raise ``RemovedEnvError`` naming every removed env that is set in ``environ`` and what replaces it.

    Call it once at process start, before ``ExpertStreamConfig.from_envs``. ``environ`` defaults to ``os.environ``.
    A name set to an empty string still counts as set. A name the engine itself set for a vendored kernel
    (``exl3/ext.py``) is listed in ``VENDORED_ENV_MARKER`` and is skipped, so a child process that inherits it
    is not refused; any other set name still is.
    """
    environ = os.environ if environ is None else environ
    engine_set = {name.strip() for name in environ.get(VENDORED_ENV_MARKER, "").split(",")}
    stale = [name for name in REMOVED_ENV if name in environ and name not in engine_set]
    if not stale:
        return
    lines = "\n".join(f"  {describe(name)}" for name in stale)
    raise RemovedEnvError(
        f"{len(stale)} removed expert-streaming env name(s) are set; they are not read any more. Replace each:\n{lines}"
    )


_TRUE = ("true", "1", "yes", "y")
_FALSE = ("false", "0", "no", "n")


def _flag(name: str, value: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name}={value!r}: not a boolean")


def _field_type(config_path: str):
    """The annotation of a dotted field path under ExpertStreamConfig."""
    from sglang.srt.layers.moe.expert_engine import config

    current = config.ExpertStreamConfig
    annotation = None
    for part in config_path.split("."):
        annotation = {info.name: info.type for info in msgspec.structs.fields(current)}[part]
        current = annotation
    return annotation


def _typed(name: str, value: str, annotation) -> object:
    import enum
    import typing

    if annotation is bool:
        return _flag(name, value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation(value.lower()).value if not issubclass(annotation, enum.IntEnum) else int(value)
    if typing.get_origin(annotation) is tuple:
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def translate_env(old_env: Mapping[str, str], node_of_cpu) -> dict[str, str]:
    """A recipe written with the removed names, as the same settings under the new names.

    ``node_of_cpu(cpu) -> node`` says where a core named by the retired single-core spellings lives. Names the rename
    left alone pass through; the retired ``SGLANG_DSV41_RAM_MISS_HIT_WAIT_US`` is dropped (it was already ignored).
    Group members become one JSON object per group, typed by the config field; the two stream switches merge into
    ``SGLANG_EXPERT_STREAM_ENABLE`` and the copy mechanisms into ``SGLANG_EXPERT_STREAM_COPY_MODE``.
    """
    new: dict[str, str] = {}
    groups: dict[str, dict] = {}
    nodes: dict[int, dict[str, str]] = {}
    enable = None
    backend, engine = None, None
    for name, value in old_env.items():
        if name == "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US":
            continue
        if name not in REMOVED_ENV:
            new[name] = value
            continue
        target = REMOVED_ENV[name]
        if target.env == "SGLANG_EXPERT_STREAM_ENABLE":
            enable = bool(enable) or _flag(name, value)
        elif name == "SGLANG_MOE_EXPERT_COPY_BACKEND":
            backend = value
        elif name == "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE":
            engine = _flag(name, value)
        elif name == "SGLANG_MOE_EXPERT_PREFETCH_PULL":
            groups.setdefault(target.env, {})["pull_mode"] = "always" if _flag(name, value) else "off"
        elif target.env == "SGLANG_EXPERT_STREAM_NUMA_CORES" and name != "SGLANG_EXPERT_NUMA_CORES":
            cores = value.strip()
            if not cores or cores == "-1":
                continue
            key = {"SGLANG_DSV41_CPU_EXPERTS_CORES": "cpu", "SGLANG_DSV41_RAM_MISS_SPIN_CORE": "ram"}.get(name, "sq")
            first = int(cores.replace("-", ",").split(",")[0])
            nodes.setdefault(node_of_cpu(first), {})[key] = cores
        elif name == "SGLANG_MOE_HOT_INSERT_ON_MISS":
            new[target.env] = value
        elif target.env in GROUP_PATHS:
            keys = target.field[len(GROUP_PATHS[target.env]) + 1 :].split(".")
            slot = groups.setdefault(target.env, {})
            for key in keys[:-1]:
                slot = slot.setdefault(key, {})
            slot[keys[-1]] = _typed(name, value, _field_type(target.field))
        elif name == "SGLANG_DSV41_EXPERT_RAM_GIB":
            raise ValueError(f"{name}: no automatic translation (GiB to MiB); set SGLANG_EXPERT_STREAM_PINNED_HOST_MB")
        else:
            new[target.env] = value
    if enable is not None:
        new["SGLANG_EXPERT_STREAM_ENABLE"] = "1" if enable else "0"
    if backend is not None or engine is not None:
        new["SGLANG_EXPERT_STREAM_COPY_MODE"] = legacy_copy_mode(backend or "gpu", bool(engine))
    if nodes:
        existing = new.get("SGLANG_EXPERT_STREAM_NUMA_CORES", "")
        if existing:
            raise ValueError("SGLANG_EXPERT_NUMA_CORES and the single-core spellings are both set; merge them by hand")
        new["SGLANG_EXPERT_STREAM_NUMA_CORES"] = ";".join(
            f"{node}:" + ",".join(f"{k}={plan[k]}" for k in ("ram", "cpu", "sq") if k in plan)
            for node, plan in sorted(nodes.items())
        )
    for env, members in groups.items():
        new[env] = json.dumps(members)
    return new


def legacy_copy_mode(copy_backend: str, copy_engine: bool) -> str:
    """The ``SGLANG_EXPERT_STREAM_COPY_MODE`` value a stale (COPY_BACKEND, ENABLE_RAM_MISS_COPY_ENGINE) pair meant.

    The native engine's copy engine ignores the legacy backend, and the legacy streamer never reads the engine switch,
    so a process used one of them: the engine on is ``copy_engine``, else ``gpu`` is ``sm_copy`` and ``dma`` is ``dma``.
    """
    if copy_engine:
        return "copy_engine"
    if copy_backend == "gpu":
        return "sm_copy"
    if copy_backend == "dma":
        return "dma"
    raise ValueError(f"SGLANG_MOE_EXPERT_COPY_BACKEND={copy_backend!r}: expected gpu or dma")
