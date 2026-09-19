"""Turns the DeepSeek V4.1 EXL3 export into what DeepseekV4ForCausalLM loads.

Linears that go through a quant method stay EXL3 (trellis/suh/svh/mul1 pass
through). The modules V4.1 keeps as plain bf16 and reads with its own GEMMs are
dequantized here: wo_a (stored as 8 slices, loaded as one [G*R, D] weight), the
ratio-1/2 compressor's wkv/wgate, and the indexer's wk.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Iterator

import torch

from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_dense_weight

EXL3_SUFFIXES = ("suh", "svh", "mul1", "trellis")
ROUTED_EXPERT_WEIGHT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
DEQUANT_PREFIX_RE = re.compile(r"^.+\.attn\.(?:compressor\.(?:wkv|wgate)|indexer\.wk)$")
WO_A_SLICE_RE = re.compile(r"^(?P<prefix>.+\.attn\.wo_a)\.slice\.(?P<group>\d+)$")

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


def is_streamed_expert_module(prefix: str) -> bool:
    """True for a target routed-expert FusedMoE (the draft's stages keep theirs resident)."""
    return ROUTED_EXPERT_MODULE_RE.match(prefix) is not None


def is_streamed_expert_weight(name: str, quant_method: str | None, streaming: bool) -> bool:
    """True for a routed-expert tensor that EXL3 expert streaming reads from disk itself."""
    return (
        streaming
        and quant_method == "exl3"
        and ROUTED_EXPERT_WEIGHT_RE.match(name) is not None
    )


def streamed_expert_skip_hook(
    quant_method: str | None, streaming: bool
) -> Callable[[str], bool] | None:
    """The loader's skip predicate, or None when no tensor can be skipped.

    None keeps the loader on its no-skip path (and fastsafetensors usable).
    """
    if not (streaming and quant_method == "exl3"):
        return None
    return lambda name: is_streamed_expert_weight(name, quant_method, streaming)


def dense_on_device(t: Exl3Tensors) -> torch.Tensor:
    cuda = Exl3Tensors(
        trellis=t.trellis.cuda(), suh=t.suh.cuda(), svh=t.svh.cuda(), mul1=t.mul1
    )
    return exl3_dense_weight(cuda)


def adapt_exl3_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    dequant: Callable[[Exl3Tensors], torch.Tensor],
    num_wo_a_groups: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    pending: dict[str, dict[str, torch.Tensor]] = {}
    wo_a: dict[str, dict[int, torch.Tensor]] = {}
    for name, tensor in weights:
        base, _, suffix = name.rpartition(".")
        wo_a_match = WO_A_SLICE_RE.match(base) if suffix in EXL3_SUFFIXES else None
        if suffix not in EXL3_SUFFIXES or not (wo_a_match or DEQUANT_PREFIX_RE.match(base)):
            yield name, tensor
            continue
        parts = pending.setdefault(base, {})
        parts[suffix] = tensor
        if len(parts) < len(EXL3_SUFFIXES):
            continue
        del pending[base]
        dense = dequant(
            Exl3Tensors(trellis=parts["trellis"], suh=parts["suh"], svh=parts["svh"], mul1=True)
        )
        weight = dense.t().contiguous().to(torch.bfloat16)  # [out, in]
        if wo_a_match is None:
            yield base + ".weight", weight
            continue
        groups = wo_a.setdefault(wo_a_match["prefix"], {})
        groups[int(wo_a_match["group"])] = weight
        if len(groups) == num_wo_a_groups:
            del wo_a[wo_a_match["prefix"]]
            yield wo_a_match["prefix"] + ".weight", torch.cat(
                [groups[g] for g in range(num_wo_a_groups)], dim=0
            )
    if pending or wo_a:
        raise ValueError(
            f"incomplete EXL3 groups: {sorted(pending)[:4]} {sorted(wo_a)[:4]}"
        )
