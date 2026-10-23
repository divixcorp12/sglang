"""The MiMo V2.6 Pro model profile: its streamed module paths, sigmoid router, caps, gates and (absent) draft.

MiMo V2.6 Pro (``MiMoV2ForCausalLM``): 70 layers, layer 0 dense and layers 1-69 MoE with 384 routed experts
(the pruned checkpoint 288), top-8, ``noaux_tc`` sigmoid routing with ``e_score_correction_bias``, hidden 6144 and
MoE intermediate 2048. Its experts are MXFP4 (``quantization_config.store_dtype == "mxfp4"``). Scope today is streaming
only: every routed expert is computed on the GPU, there is no learned prefetch head and no draft.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Iterator, Optional

from sglang.srt.layers.moe.expert_engine.profiles import ProfileCaps

if TYPE_CHECKING:
    import torch

# Raw checkpoint tensor names of a routed expert: ``model.layers.<L>.mlp.experts.<E>.<proj>.<tensor>``.
ROUTED_EXPERT_WEIGHT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.")

# The sglang module prefix of a MoE layer's routed experts (MiMoV2ForCausalLM: "model" -> "layers.<L>" -> "mlp" ->
# "experts"). The MTP/nextn draft builds its layers under other roots and never matches.
ROUTED_EXPERT_MODULE_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts$")


class MimoDraftAdapterStub:
    """The MiMo draft adapter that is not written yet: asking for it fails at startup."""

    name = "mimo_v26_draft_stub"
    supports_cpu_experts = False

    def __init__(self) -> None:
        raise NotImplementedError("MiMo V2.6 Pro has no draft adapter yet")

    def is_draft_module(self, prefix: str) -> bool:  # pragma: no cover - unreachable, __init__ raises
        raise NotImplementedError("MiMo V2.6 Pro has no draft adapter yet")

    def cpu_experts(self):  # pragma: no cover - unreachable, __init__ raises
        raise NotImplementedError("MiMo V2.6 Pro has no draft adapter yet")


class MimoV26Profile:
    """``ModelProfile`` of MiMo V2.6 Pro (MXFP4 experts, streaming only)."""

    name = "mimo_v26"
    architectures = ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
    router = "sigmoid"
    caps = ProfileCaps(max_experts=384, max_top_k=8, hidden_alignment=8)
    config_type: Optional[type] = None

    def strata(self, layer: int) -> None:
        """MiMo has no layer strata (the DSV4.1 handoff's strata are DSV4.1-only)."""
        return None

    def is_streamed_module(self, prefix: str) -> bool:
        """True for a target MoE layer's routed-expert FusedMoE (layers 1.. of the model; layer 0 is dense)."""
        match = ROUTED_EXPERT_MODULE_RE.match(prefix)
        return match is not None and int(match.group(1)) >= 1

    def streamed_modules(self, model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
        for path, module in model.named_modules():
            if self.is_streamed_module(path):
                yield path, module

    def is_streamed_expert_weight(self, name: str, *, streaming: bool, expert_format: object) -> bool:
        """True for a routed-expert checkpoint tensor that the engine's format reads from disk itself."""
        return (
            streaming
            and expert_format is not None
            and expert_format.engine_served
            and ROUTED_EXPERT_WEIGHT_RE.match(name) is not None
        )

    def streamed_weight_skip_hook(
        self, *, streaming: bool, expert_format: object
    ) -> Optional[Callable[[str], bool]]:
        """The loader's skip predicate, or None when no tensor can be skipped."""
        if not (streaming and expert_format is not None and expert_format.engine_served):
            return None
        return lambda name: self.is_streamed_expert_weight(
            name, streaming=streaming, expert_format=expert_format
        )

    def register_gates(self, model: torch.nn.Module) -> int:
        """Register every MoE layer's router gate (weight and ``e_score_correction_bias``) with the RAM prefetch."""
        from sglang.srt.layers.moe.ram_prefetch import register_moe_gates

        layers = model.model.layers
        return register_moe_gates(
            {i: layers[i] for i in range(model.model.start_layer, model.model.end_layer)},
            model.config.num_experts_per_tok,
        )

    def draft_adapter(self) -> MimoDraftAdapterStub:
        """MiMo has no draft adapter yet: this raises, so a draft requested of MiMo fails at startup."""
        return MimoDraftAdapterStub()
