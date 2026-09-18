"""EXL3 (exllamav3 trellis) quantization for the DeepSeek V4.1 EXL3 export.

Each linear is stored as trellis/suh/svh/mul1. A merged linear (e.g. the shared
expert's gate_up) keeps its parts separate: every part carries its own input
sign vector, so parts cannot be concatenated into one trellis.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_linear, exl3_moe_loop
from sglang.srt.utils import set_weight_attrs

EXL3_PARAMS = ("trellis", "suh", "svh", "mul1")
UNQUANTIZED_PREFIX_SUFFIXES = (".gate", ".weights_proj")


class Exl3Config(QuantizationConfig):
    def __init__(self, bits: float, head_bits: int, codebook: str, version: str):
        super().__init__()
        if codebook != "mul1":
            raise ValueError(f"exl3: only the mul1 codebook is supported, got {codebook}")
        self.bits, self.head_bits, self.codebook, self.version = bits, head_bits, codebook, version

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=config["bits"],
            head_bits=config.get("head_bits", 6),
            codebook=config["codebook"],
            version=config["version"],
        )

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if isinstance(layer, ParallelLMHead):
            return Exl3LinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, LinearBase):
            if prefix.endswith(UNQUANTIZED_PREFIX_SUFFIXES):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return Exl3MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def _materialize(param: nn.Parameter, lead: tuple[int, ...], loaded: torch.Tensor) -> None:
    shape = lead + tuple(loaded.shape)
    if param.numel() == 0:
        param.data = torch.zeros(shape, dtype=loaded.dtype, device=param.device)
    elif tuple(param.shape) != shape or param.dtype != loaded.dtype:
        raise ValueError(f"exl3: expected {shape} {param.dtype}, got {tuple(loaded.shape)} {loaded.dtype}")


def _load_linear_part(layer, name, param, loaded_weight, shard_id=None):
    part = 0 if shard_id is None else int(shard_id)
    _materialize(param, (layer.exl3_parts,), loaded_weight)
    param.data[part].copy_(loaded_weight)
    layer.exl3_loaded.add((name, part))


class Exl3LinearMethod(LinearMethodBase):
    applies_without_weight = True

    def __init__(self, config: Exl3Config):
        self.config = config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        sizes = list(output_partition_sizes)
        if len(set(sizes)) != 1:
            raise ValueError(f"exl3: merged linear parts must be equal, got {sizes}")
        layer.exl3_parts = len(sizes)
        layer.exl3_in = input_size_per_partition
        layer.exl3_out_part = sizes[0]
        layer.exl3_loaded = set()
        for name in EXL3_PARAMS:
            param = nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
            set_weight_attrs(
                param, {"weight_loader": functools.partial(_load_linear_part, layer, name)}
            )
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for part in range(layer.exl3_parts):
            missing = [n for n in EXL3_PARAMS if (n, part) not in layer.exl3_loaded]
            if missing:
                raise RuntimeError(f"exl3: part {part} is missing {missing}")
        layer.exl3_tensors = [
            Exl3Tensors(
                trellis=layer.trellis[p],
                suh=layer.suh[p],
                svh=layer.svh[p],
                mul1=True,
            )
            for p in range(layer.exl3_parts)
        ]
        for t in layer.exl3_tensors:
            if (t.in_features, t.out_features) != (layer.exl3_in, layer.exl3_out_part):
                raise RuntimeError(
                    f"exl3: loaded {t.in_features}x{t.out_features}, "
                    f"module expects {layer.exl3_in}x{layer.exl3_out_part}"
                )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outs = [exl3_linear(x, t) for t in layer.exl3_tensors]
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return y if bias is None else y + bias


_SLOTS = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}


def _load_expert(layer, prefix, name, param, loaded_weight, weight_name=None, *, shard_id, expert_id):
    want_prefix, slot = _SLOTS[shard_id]
    if want_prefix != prefix:
        raise ValueError(f"exl3: {shard_id} routed to {prefix}_{name}")
    local = layer.exl3_local_expert(expert_id)
    if local < 0:
        return
    _materialize(param, (layer.exl3_num_experts, 2 if prefix == "w13" else 1), loaded_weight)
    param.data[local, slot].copy_(loaded_weight)
    layer.exl3_loaded.add((prefix, name, local, slot))


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config: Exl3Config):
        self.config = config
        self.moe_runner_config = None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.exl3_num_experts = num_experts
        layer.exl3_hidden = hidden_size
        layer.exl3_inter = intermediate_size_per_partition
        layer.exl3_loaded = set()
        # CONTRACT: map a global expert id to this rank's local slot (-1 = not ours),
        # using the same helper FusedMoE.weight_loader uses (identity at TP1/EP1).
        # Evidence: fused_moe_triton/layer.py:993-1000 (_map_global_expert_id_to_local_expert_id)
        # and :1031 (weight_loader calling it before _weight_loader_impl).
        layer.exl3_local_expert = getattr(
            layer, "_map_global_expert_id_to_local_expert_id", lambda expert_id: expert_id
        )
        for prefix in ("w13", "w2"):
            for name in EXL3_PARAMS:
                param = nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
                set_weight_attrs(
                    param, {"weight_loader": functools.partial(_load_expert, layer, prefix, name)}
                )
                layer.register_parameter(f"{prefix}_{name}", param)

    def create_moe_runner(self, layer: nn.Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for e in range(layer.exl3_num_experts):
            for prefix, slots in (("w13", (0, 1)), ("w2", (0,))):
                for slot in slots:
                    missing = [n for n in EXL3_PARAMS if (prefix, n, e, slot) not in layer.exl3_loaded]
                    if missing:
                        raise RuntimeError(f"exl3: expert {e} {prefix}[{slot}] is missing {missing}")

        def tensors(prefix, e, slot):
            return Exl3Tensors(
                trellis=getattr(layer, f"{prefix}_trellis")[e, slot],
                suh=getattr(layer, f"{prefix}_suh")[e, slot],
                svh=getattr(layer, f"{prefix}_svh")[e, slot],
                mul1=True,
            )

        layer.exl3_w13 = [(tensors("w13", e, 0), tensors("w13", e, 1)) for e in range(layer.exl3_num_experts)]
        layer.exl3_w2 = [tensors("w2", e, 0) for e in range(layer.exl3_num_experts)]

    def apply(self, layer: nn.Module, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        cfg = self.moe_runner_config
        # CONTRACT: exl3_moe_loop applies the route weight before w2 and does not
        # apply it on the input; reject configs that ask otherwise, matching
        # moe_forward_native (fused_moe_native.py:66) and fused_moe_forward_native
        # (fused_moe_native.py:30), which both raise NotImplementedError() for
        # apply_router_weight_on_input=True.
        if getattr(cfg, "apply_router_weight_on_input", False):
            raise NotImplementedError("exl3 MoE: apply_router_weight_on_input")
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        out = exl3_moe_loop(
            dispatch_output.hidden_states,
            topk_weights,
            topk_ids,
            layer.exl3_w13,
            layer.exl3_w2,
            # CONTRACT: MoeRunnerConfig.swiglu_limit (moe_runner/base.py:60) is the
            # DeepSeek V4 swiglu clamp field; deep_gemm._apply_swiglu_limit
            # (moe_runner/deep_gemm.py:1666-1667) clamps up to +-limit and gate to
            # <= limit, matching exl3_moe_loop exactly. routed_scaling_factor is
            # NOT applied here: DeepseekV2MoE only fuses it into topk_weights when
            # quant_method.fuse_routed_scaling_factor_in_topk is True (layer.py:108-110,
            # unset here so it defaults False), and otherwise multiplies it into
            # final_hidden_states itself after combine (deepseek_v2.py:1083, 1310).
            # UnquantizedFusedMoEMethod.forward_cpu -> moe_forward_native
            # (fused_moe_native.py:61-163) likewise never applies routed_scaling_factor.
            cfg.swiglu_limit,
        )
        return StandardCombineInput(hidden_states=out)
