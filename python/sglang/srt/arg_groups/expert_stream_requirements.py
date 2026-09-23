"""Server-argument requirements of MoE expert caching, per expert format.

``handle_offload_compatibility`` runs while server arguments are processed,
before any model is loaded. Once ``SGLANG_MOE_HOT_GPU_MB`` or
``SGLANG_MOE_PINNED_HOST_MB`` is nonzero, it looks up the requirements of the
launch's expert format here and runs their check.

The format is identified by its quantization method (``expert_quant_method``).
Each method that can stream experts maps to one ``ExpertStreamRequirements``;
this module registers NVFP4's. Another format registers its own by calling
``register_expert_stream_requirements`` at import time of a module named
``sglang.srt.arg_groups.expert_stream_requirements_<method>``. The gate imports
that module on the method's first use. Such a module runs during server-args
processing, so it must import only this module, ``sglang.srt.environ`` and
the standard library: no quantization, model or torch-heavy code.

A launch whose method cannot be determined yet (no ``--quantization``, no
model configuration built, no local ``config.json`` naming a method) keeps the
NVFP4 requirements every expert-caching launch had before formats existed. A
determined method that no format registers is rejected.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend

PLUGIN_MODULE_PREFIX = "sglang.srt.arg_groups.expert_stream_requirements_"
# Quantization methods whose MoE layers can get ModelOptNvFp4FusedMoEMethod, or its
# subclass ModelOptNvFp4OnlineFusedMoEMethod, and so reach _attach_expert_streamer:
# the ModelOpt configs, nvfp4_online, fp8/mxfp8 (Fp8Config, wrapped by the loader in
# HybridFp8NvFp4Config for hybrid FP8+NVFP4 checkpoints) and Inkling's NVFP4 config.
NVFP4_QUANT_METHODS = (
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
    "nvfp4_online",
    "fp8",
    "mxfp8",
    "inkling_nvfp4",
)


@dataclass(frozen=True)
class ExpertCacheBudgets:
    """The expert cache settings the gate has already read and range-checked."""

    hot_budget_mb: int
    pinned_budget_mb: int
    graph_gather: bool


@dataclass(frozen=True)
class ExpertStreamRequirements:
    """How one expert format validates a launch that sets expert cache budgets.

    ``check(cfg, budgets)`` raises ``ValueError`` for an unsupported launch;
    ``cfg`` is the resolving view of the server arguments.
    """

    label: str
    check: Callable[[Any, ExpertCacheBudgets], None]
    # Where graph gathers read host rows: "arena" (SGLANG_MOE_EXPERT_HOST_ARENA,
    # indexed by expert id) or "pinned_tier" (the format's pinned host tier).
    graph_gather_host_source: Literal["arena", "pinned_tier"] = "arena"

    def __post_init__(self) -> None:
        if self.graph_gather_host_source not in ("arena", "pinned_tier"):
            raise ValueError(
                f"{self.label}: graph_gather_host_source must be 'arena' or "
                f"'pinned_tier', not {self.graph_gather_host_source!r}"
            )


_REGISTRY: dict[str, ExpertStreamRequirements] = {}


def _normalize(method: Any) -> str:
    key = str(method).strip().lower()
    if not key:
        raise ValueError("quantization method must not be empty")
    return key


def register_expert_stream_requirements(
    quant_methods: Iterable[str], requirements: ExpertStreamRequirements
) -> None:
    """Map each quantization method to ``requirements``; re-registering the same object is a no-op."""
    for method in quant_methods:
        key = _normalize(method)
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not requirements:
            raise ValueError(
                f"expert stream requirements for {key!r} are already registered "
                f"({existing.label})"
            )
        _REGISTRY[key] = requirements


def expert_quant_method(server_args: Any, cfg: Any) -> Optional[str]:
    """The launch's quantization method, from what is known before any model loads.

    In order: ``--quantization``; the quantization of the model configuration,
    when one has been built and memoised (the second gate run, after the model
    hooks); ``quantization_config.quant_method`` of ``<model_path>/config.json``
    when the model path is a local directory. None when none of them names one.
    """
    explicit = getattr(cfg, "quantization", None)
    if isinstance(explicit, str):
        explicit = explicit.strip()
    if explicit:
        return _normalize(explicit)
    from sglang.srt.arg_groups.model_override_base import (
        ResolvedView,
        ResolvingConfig,
        record_of,
    )

    record = server_args
    if isinstance(record, (ResolvedView, ResolvingConfig)):
        record = record_of(record)
    model_config = getattr(record, "_model_config", None)
    resolved = getattr(model_config, "quantization", None)
    if resolved:
        return _normalize(resolved)
    model_path = getattr(cfg, "model_path", None)
    if not isinstance(model_path, str) or not os.path.isdir(model_path):
        return None
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError):
        return None
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    if isinstance(method, str) and method.strip():
        return _normalize(method)
    return None


def _import_plugin(method: str) -> None:
    name = PLUGIN_MODULE_PREFIX + re.sub(r"[^0-9a-z_]", "_", method)
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        # Only this module's absence means "no plugin"; a plugin failing to
        # import one of its own dependencies must surface.
        if error.name != name:
            raise


def expert_stream_requirements_for(
    server_args: Any, cfg: Any
) -> ExpertStreamRequirements:
    """The requirements of the launch's expert format; raises for an unsupported method."""
    method = expert_quant_method(server_args, cfg)
    if method is None:
        return NVFP4_EXPERT_STREAM_REQUIREMENTS
    requirements = _REGISTRY.get(method)
    if requirements is None:
        _import_plugin(method)
        requirements = _REGISTRY.get(method)
    if requirements is None:
        raise ValueError(
            "MoE expert caching (SGLANG_MOE_HOT_GPU_MB, SGLANG_MOE_PINNED_HOST_MB) "
            f"does not support quantization method {method!r}; supported methods: "
            f"{sorted(_REGISTRY)}. A format declares its launch requirements in "
            f"{PLUGIN_MODULE_PREFIX}<method>"
        )
    return requirements


def validate_hot_cache_policy() -> None:
    """Check the hot cache's residency-policy knobs; any format with a hot budget runs this."""
    if envs.SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS.get() < 1:
        raise ValueError("SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS must be positive")
    if envs.SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS must be nonnegative")
    if envs.SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS must be nonnegative")
    if envs.SGLANG_MOE_HOT_DECAY_TOKENS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_DECAY_TOKENS must be nonnegative")
    sigmas = envs.SGLANG_MOE_HOT_PROMOTION_SIGMAS.get()
    if not math.isfinite(sigmas) or sigmas < 0:
        raise ValueError(
            "SGLANG_MOE_HOT_PROMOTION_SIGMAS must be finite and nonnegative"
        )
    ratio = envs.SGLANG_MOE_HOT_BENEFIT_RATIO.get()
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("SGLANG_MOE_HOT_BENEFIT_RATIO must be finite and nonnegative")
    if envs.SGLANG_MOE_HOT_LOG_INTERVAL.get() < 1:
        raise ValueError("SGLANG_MOE_HOT_LOG_INTERVAL must be positive")


def _check_nvfp4(cfg: Any, budgets: ExpertCacheBudgets) -> None:
    # Moved verbatim from handle_offload_compatibility, which read these three
    # values the same way; the stream knob has always been a raw environment read.
    hot_budget_mb = budgets.hot_budget_mb
    pinned_budget_mb = budgets.pinned_budget_mb
    graph_gather = budgets.graph_gather
    streaming = os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"
    if not streaming:
        raise ValueError("NVFP4 hot caching requires SGLANG_MOE_EXPERT_STREAM=1")
    if cfg.moe_runner_backend != "flashinfer_cutlass":
        raise ValueError(
            "NVFP4 hot caching requires --moe-runner-backend flashinfer_cutlass"
        )
    if cfg.tp_size != 1:
        raise ValueError("NVFP4 hot caching requires TP size 1")
    if cfg.ep_size != 1:
        raise ValueError("NVFP4 hot caching requires EP size 1")
    if cfg.moe_a2a_backend != "none":
        raise ValueError("NVFP4 hot caching requires --moe-a2a-backend none")
    if cfg.enable_waterfill:
        raise ValueError("NVFP4 hot caching does not support Waterfill")
    # Graph gather with GPU residency keeps route accounting and slot changes on the
    # forward stream, so result processing may trail the next launch.
    if not cfg.disable_overlap_schedule and not (
        graph_gather and envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get()
    ):
        raise ValueError(
            "NVFP4 hot caching requires --disable-overlap-schedule unless "
            "SGLANG_MOE_EXPERT_GRAPH_GATHER=1 and SGLANG_MOE_GPU_RESIDENCY_UPDATE=1"
        )
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        raise ValueError("NVFP4 hot caching requires both batch overlap modes disabled")
    if cfg.max_running_requests != 1:
        raise ValueError("NVFP4 hot caching requires --max-running-requests 1")
    if any(
        getattr(cfg, name, None)
        for name in (
            "elastic_ep_backend",
            "elastic_ep_rejoin",
            "ep_join_mode",
            "enable_elastic_expert_backup",
            "elastic_ep_initial_size",
            "max_ep_size",
            "ep_join_rank_offset",
        )
    ):
        raise ValueError("NVFP4 hot caching does not support elastic EP")
    if cfg.enable_eplb:
        raise ValueError("NVFP4 hot caching does not support EPLB")
    if hot_budget_mb:
        recorder = cfg.expert_distribution_recorder_mode
        if envs.SGLANG_MOE_HOT_DYNAMIC.get() and recorder not in ("stat", "per_pass"):
            raise ValueError(
                "Dynamic NVFP4 hot caching requires --expert-distribution-recorder-mode "
                "stat or per_pass"
            )
        validate_hot_cache_policy()
    graph_config = cfg.cuda_graph_config
    decode_backends = (
        (Backend.BREAKABLE, Backend.FULL)
        if graph_gather
        else (Backend.DISABLED, Backend.BREAKABLE)
    )
    if graph_config is not None and (
        graph_config.decode.backend not in decode_backends
        or graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError(
            "NVFP4 hot caching requires decode CUDA graph capture to be "
            f"{' or '.join(decode_backends)}, and prefill CUDA graph capture to be "
            "disabled"
        )
    if pinned_budget_mb and graph_config is not None and (
        graph_config.decode.backend not in (Backend.DISABLED, Backend.BREAKABLE)
        or graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError(
            "NVFP4 pinned host caching requires decode CUDA graph capture to be disabled "
            "or breakable, and prefill CUDA graph capture to be disabled"
        )


NVFP4_EXPERT_STREAM_REQUIREMENTS = ExpertStreamRequirements("NVFP4", _check_nvfp4)
register_expert_stream_requirements(NVFP4_QUANT_METHODS, NVFP4_EXPERT_STREAM_REQUIREMENTS)


def eager_expert_stream_requirements(
    label: str, *, enabled: Callable[[], bool], enable_hint: str,
    allow_gpu_residency_update: Callable[[], bool] = lambda: False,
) -> ExpertStreamRequirements:
    """Requirements of a format that streams experts eagerly only.

    The launch must enable the format's streaming (``enabled()``, described by
    ``enable_hint``), capture no CUDA graphs, use neither the graph gather nor
    the host arena, and, for dynamic residency, record routes with the
    ``stat`` or ``per_pass`` recorder. Nothing else is required: not
    ``--max-running-requests 1``, not the overlap schedule setting, not a MoE
    runner backend. A format wanting more composes its own ``check`` around
    this one's.
    """

    def check(cfg: Any, budgets: ExpertCacheBudgets) -> None:
        if not enabled():
            raise ValueError(f"{label} expert caching requires {enable_hint}")
        if budgets.graph_gather:
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_EXPERT_GRAPH_GATHER; set it to 0"
            )
        if envs.SGLANG_MOE_EXPERT_HOST_ARENA.get():
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_EXPERT_HOST_ARENA; set it to 0"
            )
        if envs.SGLANG_MOE_PREFETCH_MAX_CANDIDATES.get() > 0:
            raise ValueError(
                f"{label} expert caching does not support expert prefetch; "
                "set SGLANG_MOE_PREFETCH_MAX_CANDIDATES to 0"
            )
        if envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get() and not allow_gpu_residency_update():
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE; set it to 0"
            )
        if envs.SGLANG_MOE_EXPERT_DOORBELL.get():
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_EXPERT_DOORBELL; set it to 0"
            )
        graph_config = cfg.cuda_graph_config
        if graph_config is not None and (
            graph_config.decode.backend != Backend.DISABLED
            or graph_config.prefill.backend != Backend.DISABLED
        ):
            raise ValueError(
                f"{label} expert caching runs eagerly; use --disable-cuda-graph "
                "(or --cuda-graph-backend-decode disabled "
                "--cuda-graph-backend-prefill disabled)"
            )
        if budgets.hot_budget_mb:
            recorder = cfg.expert_distribution_recorder_mode
            if envs.SGLANG_MOE_HOT_DYNAMIC.get() and recorder not in ("stat", "per_pass"):
                raise ValueError(
                    f"Dynamic {label} hot caching requires "
                    "--expert-distribution-recorder-mode stat or per_pass"
                )
            validate_hot_cache_policy()

    return ExpertStreamRequirements(label, check)
