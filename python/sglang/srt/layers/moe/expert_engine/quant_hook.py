"""The integration helper a streaming quant method calls into the expert engine.

``attach`` runs in ``process_weights_after_loading``: it resolves the method's format and profile, checks
them against each other and against the model's shapes, binds the engine service and gives the layer its
expert streamer. ``apply`` runs inside ``apply()`` for a layer whose gather the engine serves: it posts the
gather, then calls the format's GPU kernel over the slot ids the gather returned.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.moe.expert_engine.config import resolved_config
from sglang.srt.layers.moe.expert_engine.formats import (
    SlotApplyArgs,
    format_for_quant_method,
)
from sglang.srt.layers.moe.expert_engine.profiles import get_profile_by_name
from sglang.srt.layers.moe.expert_engine.startup import (
    check_profile_caps,
    check_requested_capabilities,
)
from sglang.srt.layers.moe.expert_format import STREAMER_ATTRIBUTE
from sglang.srt.layers.moe.verify_dead_rows import mask_dead_rows_for_graph


def attach(layer: torch.nn.Module, quant_method) -> None:
    """Attach ``layer``'s expert streamer, bound to the engine service of ``quant_method``'s format and profile.

    Raises, naming the format and profile, when the quant method names no expert format or profile, when the
    pair has no transport build, when the model exceeds the profile's caps, or when CPU experts are requested
    of a format that only has a stub.
    """
    # Imported here: the service pulls in the expert streamer, whose import reaches the quantization methods.
    from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService

    expert_format = format_for_quant_method(quant_method)
    if expert_format is None or quant_method.expert_profile_name is None:
        raise ValueError(
            f"{type(quant_method).__name__} streams experts but names no expert format and model profile "
            f"(expert_format_name={quant_method.expert_format_name!r}, "
            f"expert_profile_name={quant_method.expert_profile_name!r})"
        )
    profile = get_profile_by_name(quant_method.expert_profile_name)
    config = resolved_config()
    check_requested_capabilities(
        expert_format=expert_format,
        profile=profile,
        cpu_experts=config.stream.cpu_experts.enabled,
        draft=False,
    )
    experts, hidden = expert_format.layer_dims(layer)
    check_profile_caps(profile, experts=experts, top_k=_top_k_of(layer), hidden=hidden)
    ExpertEngineService.bind(expert_format=expert_format, profile=profile, config=config)
    setattr(layer, STREAMER_ATTRIBUTE, expert_format.build_streamer(layer))


def _top_k_of(layer: torch.nn.Module) -> Optional[int]:
    """The routes per token of a ``FusedMoE`` layer, or None for a stand-in layer that has none."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    return layer.top_k if isinstance(layer, FusedMoE) else None


def apply(
    layer,
    streamer,
    x,
    topk_weights,
    topk_ids,
    swiglu_limit,
    cast=True,
    dead=None,
):
    """BS1 decode inside a CUDA graph: device-only gather, then the format's fused MoE over slots.

    Hits are read in place from the hot cache, misses land in scratch rows from
    the pinned tier (PinnedTierRowBackend), and routes are recorded on the device
    by the planner. A row missing from RAM sets the backend's ``keep`` to 0,
    which drops this layer's routed output. Only the engine's backend (Task 14)
    serves such a miss and fail-stops on one it cannot serve, so a plain
    ``PinnedTierRowBackend`` is refused unless the layer sets the test-only
    ``_exl3_allow_p3_only``.
    """
    from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService
    from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend

    if type(streamer.row_backend) is PinnedTierRowBackend and not getattr(
        layer, "_exl3_allow_p3_only", False
    ):
        # Without the engine's backend nothing serves or checks a RAM miss inside a replay.
        raise RuntimeError(
            f"exl3 in-graph MoE, layer {getattr(layer, 'layer_id', '?')}: the graph gather "
            "reads the pinned tier, but option C is not installed (the row backend is a "
            "plain PinnedTierRowBackend, not ExpertEngineRowBackend), so a RAM miss inside a "
            "replay would be neither served nor fail-stopped. Option C installs when "
            "SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER is on at load and the hot cache manager "
            "attaches the EXL3 format."
        )
    if swiglu_limit is None and getattr(streamer.format, "requires_swiglu_limit", True):
        raise NotImplementedError(
            f"{streamer.format.key} in-graph MoE: a swiglu_limit is required (DSV4.1 sets 10.0)"
        )
    routes = getattr(streamer.row_backend, "routes", None)
    backend = streamer.row_backend
    cpu_experts = backend.leased and backend.cpu_experts
    cpu_x = x
    if dead is not None:
        # A verify's dead rows take row 0's experts at weight 0, and -1 routes that keep them out of the
        # protect set, the CPU token mask and the prefetch scorer (verify_dead_rows.py).
        topk_ids, topk_weights, row0_x = mask_dead_rows_for_graph(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            dead=dead,
            routes=routes,
            x=x if cpu_experts else None,
        )
        if cpu_experts:
            cpu_x = row0_x
        if routes is not None:
            routes[topk_ids.numel() :].fill_(-1)
    elif routes is not None:
        # The post protects every routed expert of this layer. Captured as a
        # device copy, so every replay refreshes it from the live topk_ids.
        flat = topk_ids.reshape(-1)
        routes[: flat.numel()].copy_(flat)
        routes[flat.numel() :].fill_(-1)
    if cpu_experts:
        # The first graph-path forward of the layer is a warmup, before capture: the only place that knows the
        # layer's activation limit registers its pinned slabs with the CPU kernel.
        # No limit (a format with requires_swiglu_limit False) registers as 0.0, the CPU kernels' "no clamp".
        ExpertEngineService.get().cpu_experts.register(
            backend.row, 0.0 if swiglu_limit is None else float(swiglu_limit)
        )
        # The post stages x and the lanes' weights for the CPU expert thread (ExpertEngineRowBackend.post).
        backend.cpu_input = (cpu_x, topk_weights)
        # The lane head (SGLANG_OPT_LANE_HEAD_PATH) counts live rows only; a dead row repeats row 0's routes.
        backend.verify_dead = dead
    # The fused plan leaves a dead row's routes out of the residency route counts.
    streamer.verify_dead = dead
    try:
        remap, _ = streamer.gather(topk_ids)
    finally:
        streamer.verify_dead = None
        if cpu_experts:
            backend.cpu_input = None
            backend.verify_dead = None
    if backend.leased and backend.route_log is not None:
        if backend.route_log.router is not None:
            # After the gather: its post ran the layer's route_log.record, which row 0 used to take the slot.
            backend.route_log.record_router(backend.row, x, topk_ids, topk_weights)
    out = streamer.format.moe_apply(
        layer,
        x,
        remap,
        topk_weights,
        SlotApplyArgs(
            streamer=streamer,
            keep=streamer.row_backend.keep,
            swiglu_limit=swiglu_limit,
            cpu=(
                (
                    backend.device_side.cpu_lanes,
                    streamer.row_plan.slots,
                    backend.device_side.cpu_out_address(backend.row),
                    backend.device_side.cpu_out_part_stride(),
                )
                if cpu_experts
                else None
            ),
        ),
    )
    # cast=False hands the fp32 output to exl3_scale_to_bf16, which casts and scales in one kernel.
    return out.to(x.dtype) if cast else out
