"""Named MoE expert-offload configurations selected by ``--moe-offload-preset``.

A preset fills the offload variables the user left unset; an explicitly set
variable always wins. ``resolve_offload_env`` then derives the settings other
settings force, and ``check_offload_config`` refuses invalid combinations at
argument resolution, before the weight load. ``#N`` refers to a row of
"Experiment results" in MOE_EXPERT_TRANSFER.md. Everything here is pure; the
``arg_groups`` hook reads and writes the process state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import msgspec

from sglang.srt.environ import envs


class MoeOffloadPreset(msgspec.Struct, frozen=True, kw_only=True):
    """One offload configuration; a ``None`` field leaves that variable to its ``Envs`` default."""

    # Stream ModelOpt NVFP4 routed experts from host memory; required for hot caching.
    expert_stream: bool | None = None
    # io_uring O_DIRECT reads of the expert file cache.
    expert_file_reader: str | None = None
    # io_uring reads of the file-backed PLE table.
    ple_file_reader: str | None = None
    # Stage PLE rows before decode graph replay instead of breaking the graph.
    ple_stage_before_replay: bool | None = None
    # Token embedding in pinned host memory, shared with the draft: frees 1.18 GB (#33).
    host_token_embedding: bool | None = None
    # Tuned for a 32 GB RTX 5090 with Qwen3.8-Flash-Next-NVFP4 at chunk 4096 (#34);
    # 16,384 OOMs and 15,872 leaves 66 MiB (#36). Override on any other GPU or model.
    hot_gpu_mb: int | None = None
    # 0: the host arena replaces the pinned LRU.
    pinned_host_mb: int | None = None
    # Copy every host expert row into registered memory once.
    expert_host_arena: bool | None = None
    # Host-to-device misses through the CUDA copy engine.
    expert_copy_backend: str | None = None
    # Sync-free in-graph gather of misses; both approaches are built on it.
    expert_graph_gather: bool | None = None
    # Scores decide residency at run time instead of the seed alone.
    hot_dynamic: bool | None = None
    # Residency policy tuned in the E16c/E19 arms.
    hot_decay_tokens: int | None = None
    hot_promotion_sigmas: float | None = None
    hot_benefit_ratio: float | None = None
    hot_min_residence_forwards: int | None = None
    # Residency boundaries run on the device, inside the decode graph.
    gpu_residency_update: bool | None = None
    gpu_residency_max_promotions: int | None = None
    # Insert-on-miss needs a boundary every decode forward.
    hot_update_decode_forwards: int | None = None
    # 0: prefetch was rejected, -4.3% (#5).
    prefetch_max_candidates: int | None = None
    # JIT route planner: about 65 fewer bookkeeping kernels per layer, +6.4%.
    expert_fused_plan: bool | None = None
    # 2 (DIRECT) copies misses straight into victim slots, +4.4% over 1; stage 2 refuses the doorbell.
    insert_on_miss_stage: int | None = None
    # MTP draft experts FP8 to NVFP4 at load: draft 2.46 to 1.45 GB (#30). No effect without such a draft.
    draft_moe_nvfp4_requant: bool | None = None
    # Side-thread copier that starts a miss copy before the graph reaches it.
    expert_doorbell: bool | None = None
    expert_doorbell_mode: str | None = None
    # The spin thread's core; must be in the process's allowed CPUs or it runs unpinned.
    expert_doorbell_cpu: int | None = None


ENV_NAMES: dict[str, str] = {
    "expert_stream": "SGLANG_MOE_EXPERT_STREAM",
    "expert_file_reader": "SGLANG_MOE_EXPERT_FILE_READER",
    "ple_file_reader": "SGLANG_QWEN4_PLE_FILE_READER",
    "ple_stage_before_replay": "SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY",
    "host_token_embedding": "SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING",
    "hot_gpu_mb": "SGLANG_MOE_HOT_GPU_MB",
    "pinned_host_mb": "SGLANG_MOE_PINNED_HOST_MB",
    "expert_host_arena": "SGLANG_MOE_EXPERT_HOST_ARENA",
    "expert_copy_backend": "SGLANG_MOE_EXPERT_COPY_BACKEND",
    "expert_graph_gather": "SGLANG_MOE_EXPERT_GRAPH_GATHER",
    "hot_dynamic": "SGLANG_MOE_HOT_DYNAMIC",
    "hot_decay_tokens": "SGLANG_MOE_HOT_DECAY_TOKENS",
    "hot_promotion_sigmas": "SGLANG_MOE_HOT_PROMOTION_SIGMAS",
    "hot_benefit_ratio": "SGLANG_MOE_HOT_BENEFIT_RATIO",
    "hot_min_residence_forwards": "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS",
    "gpu_residency_update": "SGLANG_MOE_GPU_RESIDENCY_UPDATE",
    "gpu_residency_max_promotions": "SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS",
    "hot_update_decode_forwards": "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS",
    "prefetch_max_candidates": "SGLANG_MOE_PREFETCH_MAX_CANDIDATES",
    "expert_fused_plan": "SGLANG_MOE_EXPERT_FUSED_PLAN",
    "insert_on_miss_stage": "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
    "draft_moe_nvfp4_requant": "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT",
    "expert_doorbell": "SGLANG_MOE_EXPERT_DOORBELL",
    "expert_doorbell_mode": "SGLANG_MOE_EXPERT_DOORBELL_MODE",
    "expert_doorbell_cpu": "SGLANG_MOE_EXPERT_DOORBELL_CPU",
}
OFFLOAD_ENV_NAMES: tuple[str, ...] = tuple(ENV_NAMES.values())

_SHARED = dict(
    expert_stream=True,
    expert_file_reader="uring_direct",
    ple_file_reader="uring",
    ple_stage_before_replay=True,
    host_token_embedding=True,
    hot_gpu_mb=15360,
    pinned_host_mb=0,
    expert_host_arena=True,
    expert_copy_backend="dma",
    expert_graph_gather=True,
    hot_dynamic=True,
    hot_decay_tokens=1,
    hot_promotion_sigmas=0.0,
    hot_benefit_ratio=2.0,
    hot_min_residence_forwards=0,
    gpu_residency_update=True,
    gpu_residency_max_promotions=64,
    hot_update_decode_forwards=1,
    prefetch_max_candidates=0,
    expert_fused_plan=True,
)

# The current best, and prod's config since 2026-09-19: in-graph gather, insert-on-miss
# stage 2 and the fused planner, with overlap scheduling on. 29.30 / 29.88 tok/s median at
# NEXTN-3 on divix01 (#34). Needs decode CUDA graphs at batch size 1; pair it with
# --speculative-algorithm NEXTN, which it does not set.
GRAPH_GATHER_PRESET = MoeOffloadPreset(
    **_SHARED,
    insert_on_miss_stage=2,
    draft_moe_nvfp4_requant=True,
)

# Experimental and unmeasured on this stack. The side-thread doorbell copier on the
# graph-gather base, with insert-on-miss stage 1 because stage 2 refuses it. The doorbell
# requires overlap scheduling off (set automatically), no speculative decoding, no TP, PP,
# DP or DP attention, and its spin core (71) in the process's allowed CPUs.
DOORBELL_PRESET = MoeOffloadPreset(
    **_SHARED,
    insert_on_miss_stage=1,
    expert_doorbell=True,
    expert_doorbell_mode="current",
    expert_doorbell_cpu=71,
)

PRESETS: dict[str, MoeOffloadPreset | None] = {
    "off": None,
    "graph-gather": GRAPH_GATHER_PRESET,
    "doorbell": DOORBELL_PRESET,
}


class ResolvedOffloadEnv(msgspec.Struct, frozen=True):
    """The offload variables after the preset and the derivations are applied."""

    # Every offload variable in play, set explicitly or filled here.
    effective: dict[str, str]
    # The variables this resolution sets.
    filled: dict[str, str]
    # Preset values an explicitly set variable replaced.
    overridden: dict[str, str]


def _env_string(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _value(values: Mapping[str, str], name: str) -> Any:
    descriptor = getattr(envs, name)
    return descriptor.parse(values[name]) if name in values else descriptor.default


def preset_env(preset: MoeOffloadPreset) -> dict[str, str]:
    """The variables ``preset`` sets, as environment strings."""
    return {
        ENV_NAMES[field]: _env_string(value)
        for field in preset.__struct_fields__
        if (value := getattr(preset, field)) is not None
    }


def explicit_offload_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The offload variables already set in ``environ``."""
    return {name: environ[name] for name in OFFLOAD_ENV_NAMES if name in environ}


def _insert_on_miss(values: Mapping[str, str]) -> bool:
    return _value(values, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE") >= 1


def _residency_update(values: Mapping[str, str]) -> bool:
    return _value(values, "SGLANG_MOE_GPU_RESIDENCY_UPDATE")


# (condition, variable, value, why). Ordered: a later rule reads what an earlier one derived.
_DERIVATIONS = (
    (_insert_on_miss, "SGLANG_MOE_GPU_RESIDENCY_UPDATE", "1", "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE>=1"),
    (_insert_on_miss, "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS", "1", "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE>=1"),
    (_residency_update, "SGLANG_MOE_HOT_DYNAMIC", "1", "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1"),
)


def resolve_offload_env(
    preset: MoeOffloadPreset | None, explicit: Mapping[str, str]
) -> ResolvedOffloadEnv:
    """Merge explicit variables over ``preset``, then fill what the derivations force."""
    wanted = {} if preset is None else preset_env(preset)
    filled = {name: value for name, value in wanted.items() if name not in explicit}
    overridden = {
        name: value
        for name, value in wanted.items()
        if name in explicit and _value(explicit, name) != _value(wanted, name)
    }
    effective = {**explicit, **filled}
    for condition, name, value, why in _DERIVATIONS:
        if not condition(effective):
            continue
        if name in explicit:
            if _value(explicit, name) != _value({name: value}, name):
                raise ValueError(f"{why} requires {name}={value}, but it is set to {explicit[name]}")
            continue
        effective[name] = value
        filled[name] = value
    return ResolvedOffloadEnv(effective=effective, filled=filled, overridden=overridden)


def needs_overlap_off(values: Mapping[str, str]) -> bool:
    """Whether these variables require ``--disable-overlap-schedule``."""
    if _value(values, "SGLANG_MOE_EXPERT_DOORBELL"):
        return True
    # memory_hook.handle_offload_compatibility enforces the same rule for the hot cache.
    return _value(values, "SGLANG_MOE_HOT_GPU_MB") > 0 and not (
        _value(values, "SGLANG_MOE_EXPERT_GRAPH_GATHER") and _residency_update(values)
    )


def check_offload_config(
    values: Mapping[str, str],
    *,
    speculative: bool,
    decode_graphs_disabled: bool,
    decode_max_bs: int | None,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    dp_attention: bool,
    allowed_cpus: Collection[int],
) -> None:
    """Refuse combinations that would otherwise fail after the weight load, or silently.

    ``allowed_cpus`` is the launcher process's cpuset; the doorbell spin thread
    actually runs in the spawned scheduler, which can narrow its own cpuset
    later (numactl wrap, SGLANG_SET_CPU_AFFINITY, numa_bind_to_node). Passing
    this check is necessary, not sufficient: the scheduler can still end up
    unpinned even when the launcher's cpuset includes the configured core.
    """
    if _value(values, "SGLANG_MOE_HOT_GPU_MB") > 0 and not _value(values, "SGLANG_MOE_EXPERT_STREAM"):
        raise ValueError("SGLANG_MOE_HOT_GPU_MB requires SGLANG_MOE_EXPERT_STREAM=1")
    if _value(values, "SGLANG_MOE_EXPERT_GRAPH_GATHER") and decode_graphs_disabled:
        raise ValueError(
            "SGLANG_MOE_EXPERT_GRAPH_GATHER=1 requires decode CUDA graphs; "
            "drop --cuda-graph-backend-decode disabled"
        )
    if _residency_update(values) and (decode_max_bs or 0) > 1:
        raise ValueError("SGLANG_MOE_GPU_RESIDENCY_UPDATE=1 requires --cuda-graph-max-bs-decode 1")
    if not _value(values, "SGLANG_MOE_EXPERT_DOORBELL"):
        return
    if speculative:
        raise ValueError(
            "SGLANG_MOE_EXPERT_DOORBELL cannot run with speculative decoding; drop "
            "--speculative-algorithm or use --moe-offload-preset graph-gather"
        )
    if tp_size > 1 or pp_size > 1 or dp_size > 1 or dp_attention:
        raise ValueError(
            "SGLANG_MOE_EXPERT_DOORBELL runs one spin thread on one core and cannot run "
            "with tensor, pipeline or data parallelism, or DP attention"
        )
    if _value(values, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE") == 2:
        raise ValueError(
            "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 cannot run with SGLANG_MOE_EXPERT_DOORBELL: "
            "the doorbell thread can write a slot after stage 2 has committed it"
        )
    cpu = _value(values, "SGLANG_MOE_EXPERT_DOORBELL_CPU")
    if cpu not in allowed_cpus:
        raise ValueError(
            f"SGLANG_MOE_EXPERT_DOORBELL_CPU={cpu} is outside the launcher's allowed CPUs "
            f"({min(allowed_cpus)}-{max(allowed_cpus)}); set it to an allowed core. This "
            "does not guarantee the scheduler process keeps that core: it can narrow its "
            "own cpuset further after this check runs."
        )
