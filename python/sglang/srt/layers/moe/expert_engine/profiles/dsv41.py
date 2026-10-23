"""The DeepSeek V4.1 model profile: its streamed module paths, router, caps, strata, gates and DSpark draft.

Everything the expert engine knows about DSV4.1 lives here: the checkpoint and module-path regexes that decide
which experts stream and which belong to the DSpark draft, the sqrt-softplus router, the 384-expert caps the
learned prefetch head was trained inside, the layer strata of the DSV4.1 handoff, and the DSpark draft adapter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from sglang.srt.layers.moe.expert_engine.profiles import ProfileCaps
from sglang.srt.layers.moe.expert_engine.profiles.dsv41_strata import stratum

if TYPE_CHECKING:
    import torch

ROUTED_EXPERT_WEIGHT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")

# Unlike ROUTED_EXPERT_WEIGHT_RE (raw checkpoint tensor names), this matches the
# sglang *module* prefix Exl3Config.get_quant_method receives for a FusedMoE
# layer: DeepseekV4ForCausalLM builds it as "model.layers.<L>.mlp.experts"
# (deepseek_v4.py DeepseekV4Model prefix="model" -> DeepseekV4DecoderLayer
# mlp=add_prefix("mlp", ...) -> deepseek_v2.DeepseekV2MoE
# experts=add_prefix("experts", ...)). The DSpark draft's stages are built with
# an empty root prefix and land under "stages.<S>.mlp.experts" instead
# (deepseek_v4_dspark.py DeepseekV4ForCausalLMDSpark.stages
# prefix=add_prefix(f"stages.{stage_id}", prefix)), so they never match and
# stay resident.
ROUTED_EXPERT_MODULE_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts$")

# The DSpark draft's stage MoEs: "stages.<S>.mlp.experts" (empty root prefix) or "model.stages.<S>.mlp.experts".
DSPARK_DRAFT_EXPERT_MODULE_RE = re.compile(r"^(?:model\.)?stages\.\d+\.mlp\.experts$")


class DsparkDraftAdapter:
    """The DSpark draft of DSV4.1: its stage MoEs are resident on the GPU, with the rest on the CPU when asked."""

    name = "dspark"
    supports_cpu_experts = True

    def is_draft_module(self, prefix: str) -> bool:
        """True for a DSpark draft stage's routed-expert FusedMoE."""
        return DSPARK_DRAFT_EXPERT_MODULE_RE.match(prefix) is not None

    def cpu_experts(self) -> Any:
        """The registry the draft's CPU-served stages register with."""
        from sglang.srt.layers.moe.cpu_experts.draft import DRAFT_CPU_EXPERTS

        return DRAFT_CPU_EXPERTS


class Dsv41Profile:
    """``ModelProfile`` of DeepSeek V4.1 (EXL3 today)."""

    name = "dsv41"
    architectures = (
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForCausalLMNextN",
        "DeepseekV4ForCausalLMDSpark",
    )
    router = "sqrt_softplus"
    caps = ProfileCaps(max_experts=384, max_top_k=8, hidden_alignment=8)
    config_type: Optional[type] = None

    def __init__(self) -> None:
        self._draft = DsparkDraftAdapter()

    def strata(self, layer: int) -> str:
        return stratum(layer)

    def is_streamed_module(self, prefix: str) -> bool:
        """True for a target routed-expert FusedMoE (the draft's stages keep theirs resident)."""
        return ROUTED_EXPERT_MODULE_RE.match(prefix) is not None

    def streamed_modules(self, model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
        for path, module in model.named_modules():
            if self.is_streamed_module(path):
                yield path, module

    def is_streamed_expert_weight(self, name: str, *, streaming: bool, expert_format: Any) -> bool:
        """True for a routed-expert tensor that the engine's format reads from disk itself."""
        return (
            streaming
            and expert_format is not None
            and expert_format.engine_served
            and ROUTED_EXPERT_WEIGHT_RE.match(name) is not None
        )

    def streamed_weight_skip_hook(
        self, *, streaming: bool, expert_format: Any
    ) -> Optional[Callable[[str], bool]]:
        """The loader's skip predicate, or None when no tensor can be skipped.

        None keeps the loader on its no-skip path (and fastsafetensors usable).
        """
        if not (streaming and expert_format is not None and expert_format.engine_served):
            return None
        return lambda name: self.is_streamed_expert_weight(
            name, streaming=streaming, expert_format=expert_format
        )

    def register_gates(self, model: torch.nn.Module) -> int:
        """Register every MoE layer's router gate of ``model`` with the RAM prefetch; returns how many."""
        from sglang.srt.layers.moe.ram_prefetch import register_moe_gates

        layers = model.model.layers
        return register_moe_gates(
            {i: layers[i] for i in range(model.model.start_layer, model.model.end_layer)},
            model.config.num_experts_per_tok,
        )

    def draft_adapter(self) -> DsparkDraftAdapter:
        return self._draft
