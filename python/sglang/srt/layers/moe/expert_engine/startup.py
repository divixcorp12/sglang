"""Fail-fast checks the expert engine runs at startup, before weights are placed.

Each raises with the names involved and never falls back to another format's or profile's rules:
an unknown (format, router) transport, a model beyond its profile's caps, and CPU experts or a draft
requested of a format or profile that only has a stub.
"""

from __future__ import annotations

from typing import Any, Optional

# The engine's own ceilings, which a profile's caps must sit inside (router.h and the residency mask).
ENGINE_MAX_EXPERTS = 384
ENGINE_MAX_TOP_K = 8
ENGINE_HIDDEN_ALIGNMENT = 8


def resolve_transport_build(*, expert_format: Any, profile: Any) -> tuple[str, str]:
    """The transport build key ``(format, router)`` for this pair; ValueError naming the JIT key when it is not built.

    The format supplies the first half (``transport_build``), the profile the router. A format without a
    ``transport_build`` is not served by the native engine at all.
    """
    from sglang.kernels.ops.moe.expert_stream_transport import (
        LAYOUTS,
        host_jit_key,
        host_variant,
    )

    format_build: Optional[str] = expert_format.transport_build
    if format_build is None:
        raise ValueError(
            f"expert format {expert_format.key!r} has no expert-engine transport build, so profile "
            f"{profile.name!r} cannot stream it"
        )
    key = (format_build, profile.router)
    if key not in LAYOUTS:
        raise ValueError(
            f"no expert-stream transport is built for (format, router) {key!r} (format {expert_format.key!r}, "
            f"profile {profile.name!r}): JIT key {host_jit_key(key, host_variant())!r} has no build; "
            f"registered: {sorted(LAYOUTS)}"
        )
    return key


def check_profile_caps(
    profile: Any, *, experts: int, top_k: Optional[int], hidden: int
) -> None:
    """Refuse a model whose shapes exceed its profile's caps, or a profile whose caps exceed the engine's.

    ``top_k`` is None when the layers do not say (the check is then skipped for it).
    """
    caps = profile.caps
    if caps.max_experts > ENGINE_MAX_EXPERTS or caps.max_top_k > ENGINE_MAX_TOP_K:
        raise ValueError(
            f"profile {profile.name!r} claims up to {caps.max_experts} experts and top-{caps.max_top_k}, "
            f"beyond the engine's {ENGINE_MAX_EXPERTS} experts and top-{ENGINE_MAX_TOP_K}"
        )
    if experts > caps.max_experts:
        raise ValueError(
            f"profile {profile.name!r}: the model has {experts} experts per layer, over the profile's "
            f"cap of {caps.max_experts}"
        )
    if top_k is not None and top_k > caps.max_top_k:
        raise ValueError(
            f"profile {profile.name!r}: the model routes top-{top_k}, over the profile's cap of {caps.max_top_k}"
        )
    alignment = max(caps.hidden_alignment, ENGINE_HIDDEN_ALIGNMENT)
    if hidden % alignment:
        raise ValueError(
            f"profile {profile.name!r}: hidden size {hidden} is not a multiple of {alignment}"
        )


def check_requested_capabilities(
    *, expert_format: Any, profile: Any, cpu_experts: bool, draft: bool, cpu_draft: bool = False
) -> None:
    """Refuse CPU experts or a draft requested of a format or profile that only has a stub.

    ``cpu_experts``: CPU experts serve the target's routes. ``draft``: a speculative draft runs on the engine.
    ``cpu_draft``: the draft's non-resident experts run on the CPU.
    """
    if cpu_experts and not expert_format.supports_cpu_experts:
        raise ValueError(
            f"CPU experts were requested, but expert format {expert_format.key!r} has no CPU kernel (it is a stub)"
        )
    if not (draft or cpu_draft):
        return
    adapter = profile.draft_adapter()
    if adapter is None:
        raise ValueError(
            f"a speculative draft was requested, but profile {profile.name!r} has no draft adapter"
        )
    if not expert_format.supports_draft:
        raise ValueError(
            f"a speculative draft was requested, but expert format {expert_format.key!r} does not support one"
        )
    if cpu_draft and not (expert_format.supports_cpu_experts and adapter.supports_cpu_experts):
        raise ValueError(
            f"CPU draft experts were requested, but draft adapter {adapter.name!r} of profile {profile.name!r} "
            f"with expert format {expert_format.key!r} has no CPU kernel (it is a stub)"
        )
