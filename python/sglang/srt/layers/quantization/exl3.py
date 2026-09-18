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
from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_linear
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


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config: Exl3Config):
        self.config = config

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError("exl3 MoE lands in Task 4")

    def apply(self, layer, dispatch_output):
        raise NotImplementedError("exl3 MoE lands in Task 4")
