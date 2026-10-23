"""Model profile registry and the ``ModelProfile`` protocol.

A profile holds what the expert engine needs to know about one model family: which of its modules
stream their experts, which router scores them, the caps its shapes must stay inside, its layer strata,
how its router gates are registered and which draft adapter (if any) serves its speculative draft.
Engine code asks the profile; nothing outside ``profiles/`` names a model.

Profiles are looked up by the Hugging Face architecture names they declare, and imported on first use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol, Sequence

import msgspec

if TYPE_CHECKING:
    import torch


class ProfileCaps(msgspec.Struct, frozen=True):
    """The shape limits a profile claims for its model: expert count, routes per token and hidden alignment.

    The engine checks these against the model's real shapes at startup (``startup.check_profile_caps``) and
    against its own ceilings, so a model beyond them refuses to start.
    """

    max_experts: int
    max_top_k: int
    hidden_alignment: int


class DraftAdapter(Protocol):
    """A model's speculative draft, as the engine sees it.

    ``is_draft_module`` says which module paths are the draft's routed-expert layers. ``supports_cpu_experts``
    is False for a stub adapter: requesting CPU draft experts of it refuses at startup. ``cpu_experts`` is the
    registry the draft's CPU-served stages register with.
    """

    name: str
    supports_cpu_experts: bool

    def is_draft_module(self, prefix: str) -> bool: ...

    def cpu_experts(self) -> Any: ...


class ModelProfile(Protocol):
    """One model family's facts for the expert engine.

    ``router`` names the routing math (``"sqrt_softplus"`` or ``"sigmoid"``), the second half of the
    transport's build key. ``strata`` maps a layer id to its stratum label. ``config_type`` is the profile's
    config struct, or None until the model has profile-specific knobs.
    """

    name: str
    architectures: tuple[str, ...]
    router: str
    caps: ProfileCaps
    config_type: Optional[type]

    def strata(self, layer: int) -> str: ...

    def is_streamed_module(self, prefix: str) -> bool:
        """Whether the module at ``prefix`` is a target routed-expert layer that streams."""
        ...

    def streamed_modules(self, model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
        """The ``(path, module)`` pairs of ``model``'s streamed routed-expert layers."""
        ...

    def streamed_weight_skip_hook(
        self, *, streaming: bool, expert_format: Any
    ) -> Optional[Callable[[str], bool]]:
        """The loader's skip predicate over checkpoint tensor names, or None when no tensor can be skipped."""
        ...

    def register_gates(self, model: torch.nn.Module) -> int:
        """Register the model's router gates for the RAM prefetch; returns how many layers registered."""
        ...

    def draft_adapter(self) -> Optional[DraftAdapter]:
        """The model's speculative draft adapter, or None when it has none."""
        ...


class _ProfileEntry(msgspec.Struct, frozen=True):
    """A registered profile: where its class lives and which architectures resolve to it."""

    target: str
    architectures: tuple[str, ...]


# Profiles are imported on first use. A model adds its entry here together with its ``profiles/<name>.py``.
_PROFILES: dict[str, _ProfileEntry] = {
    "dsv41": _ProfileEntry(
        target="sglang.srt.layers.moe.expert_engine.profiles.dsv41:Dsv41Profile",
        architectures=(
            "DeepseekV4ForCausalLM",
            "DeepseekV4ForCausalLMNextN",
            "DeepseekV4ForCausalLMDSpark",
        ),
    ),
    "mimo_v26": _ProfileEntry(
        target="sglang.srt.layers.moe.expert_engine.profiles.mimo_v26:MimoV26Profile",
        architectures=("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"),
    ),
}


def registered_profiles() -> tuple[str, ...]:
    """The registered profile names, sorted."""
    return tuple(sorted(_PROFILES))


def get_profile_by_name(name: str) -> ModelProfile:
    """The profile registered as ``name``; ValueError naming the registered profiles otherwise."""
    entry = _PROFILES.get(name)
    if entry is None:
        raise ValueError(f"no model profile {name!r}; registered: {list(registered_profiles())}")
    module_name, _, class_name = entry.target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)()


def profile_for_architectures(architectures: Optional[Sequence[str]]) -> ModelProfile:
    """The profile one of ``architectures`` resolves to; ValueError naming them and the registered profiles otherwise."""
    for name, entry in _PROFILES.items():
        if any(architecture in entry.architectures for architecture in architectures or ()):
            return get_profile_by_name(name)
    raise ValueError(
        f"no expert-engine model profile for architectures {list(architectures or ())}; "
        f"registered: {list(registered_profiles())}"
    )


def get_profile(model_config: Any) -> ModelProfile:
    """The profile of a ``ModelConfig`` (resolved from its Hugging Face config's architectures)."""
    return profile_for_architectures(model_config.hf_config.architectures)


def profile_for_hf_config(hf_config: Any) -> ModelProfile:
    """The profile of a Hugging Face config."""
    return profile_for_architectures(hf_config.architectures)


def profile_for_model(model: torch.nn.Module) -> ModelProfile:
    """The profile of a built model, from the Hugging Face config every sglang model keeps as ``config``."""
    return profile_for_architectures(model.config.architectures)
