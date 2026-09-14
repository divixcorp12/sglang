"""Load-time FP8 for the BF16 linear layers of a ModelOpt MIXED_PRECISION checkpoint.

The feature is off unless ``SGLANG_ONLINE_FP8_GROUPS`` names at least one group; with it
unset every layer keeps ``UnquantizedLinearMethod`` exactly as before. Environment:

* ``SGLANG_ONLINE_FP8_GROUPS``: comma-separated groups from ``GROUP_NAMES``, or ``all``.
* ``SGLANG_ONLINE_FP8_SCHEME``: ``w8a8`` (default) quantizes weights per output channel and
  activations per token at run time and runs the SGLang CUTLASS channelwise FP8 GEMM;
  ``w8a16`` stores the same FP8 weights but dequantizes them for a BF16 matmul. ``w8a16``
  exists to separate weight error from activation error while bisecting accuracy.
* ``SGLANG_ONLINE_FP8_LAYERS``: optional decoder-layer ids such as ``0-11,40``. When set,
  only layer-scoped modules in those layers convert; the final hyper-connection mixer and
  ``lm_head`` have no layer id and are then left BF16.

Draft (``mtp.``) modules never convert.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

GROUPS_ENV = "SGLANG_ONLINE_FP8_GROUPS"
SCHEME_ENV = "SGLANG_ONLINE_FP8_SCHEME"
LAYERS_ENV = "SGLANG_ONLINE_FP8_LAYERS"

GROUP_NAMES = ("full_attn", "gdn", "shared_expert", "hc_mix", "lm_head")
SCHEMES = ("w8a8", "w8a16")

FP8_DTYPE = torch.float8_e4m3fn
QUANT_ROW_CHUNK = 4096
WEIGHT_ONLY_ROW_CHUNK = 32768

_LINEAR_GROUP_PATTERNS = (
    (
        "gdn",
        re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn\.(?:in_proj_qkvz|out_proj)$"),
    ),
    (
        "full_attn",
        re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.(?:qkv_proj|o_proj)$"),
    ),
    (
        "shared_expert",
        re.compile(
            r"(?:^|\.)layers\.(\d+)\.(?:linear_attn\.|self_attn\.)?"
            r"mlp\.shared_expert\.(?:gate_up_proj|down_proj)$"
        ),
    ),
)
_LM_HEAD_PATTERN = re.compile(r"(?:^|\.)lm_head$")
_DRAFT_PATTERN = re.compile(r"(?:^|\.)mtp(?:\.|$)")
_LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

_announced_groups: set = set()


@dataclass(frozen=True)
class OnlineFp8Selection:
    """Which module groups and decoder layers convert, and with which scheme."""

    groups: FrozenSet[str]
    scheme: str
    layers: Optional[FrozenSet[int]]

    def includes(self, group: str, layer_id: Optional[int]) -> bool:
        if group not in self.groups:
            return False
        if self.layers is None:
            return True
        return layer_id is not None and layer_id in self.layers


def _parse_layers(value: str) -> Optional[FrozenSet[int]]:
    value = value.strip()
    if not value:
        return None
    layers = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise ValueError(f"{LAYERS_ENV}: cannot parse {part!r}; use ids like 0-11,40")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise ValueError(f"{LAYERS_ENV}: range {part!r} is descending")
        layers.update(range(first, last + 1))
    if not layers:
        raise ValueError(f"{LAYERS_ENV} is set but names no layer")
    return frozenset(layers)


def online_fp8_selection(
    environ: Mapping[str, str] = os.environ,
) -> Optional[OnlineFp8Selection]:
    """Parse the environment; ``None`` means the feature is off."""
    raw_groups = environ.get(GROUPS_ENV, "").strip()
    if not raw_groups:
        return None
    names = [name.strip() for name in raw_groups.split(",") if name.strip()]
    if names == ["all"]:
        groups = frozenset(GROUP_NAMES)
    else:
        unknown = sorted(set(names) - set(GROUP_NAMES))
        if unknown:
            raise ValueError(
                f"{GROUPS_ENV}: unknown group(s) {unknown}; choose from "
                f"{list(GROUP_NAMES)} or 'all'"
            )
        groups = frozenset(names)
    scheme = environ.get(SCHEME_ENV, "").strip() or "w8a8"
    if scheme not in SCHEMES:
        raise ValueError(f"{SCHEME_ENV}: {scheme!r} is not one of {list(SCHEMES)}")
    return OnlineFp8Selection(
        groups=groups, scheme=scheme, layers=_parse_layers(environ.get(LAYERS_ENV, ""))
    )


def linear_group_for_prefix(prefix: str) -> Optional[Tuple[str, Optional[int]]]:
    """Return ``(group, layer_id)`` for a convertible linear module prefix, else ``None``."""
    if _DRAFT_PATTERN.search(prefix):
        return None
    for group, pattern in _LINEAR_GROUP_PATTERNS:
        match = pattern.search(prefix)
        if match is not None:
            return group, int(match.group(1))
    if _LM_HEAD_PATTERN.search(prefix):
        return "lm_head", None
    return None


def _announce(group: str, scheme: str) -> None:
    if group not in _announced_groups:
        _announced_groups.add(group)
        logger.info("Online FP8 (%s) enabled for module group %s", scheme, group)


def online_fp8_or_unquantized(
    prefix: str, environ: Mapping[str, str] = os.environ
) -> LinearMethodBase:
    """The linear method for a BF16 layer: online FP8 when selected, else unquantized."""
    selection = online_fp8_selection(environ)
    if selection is None:
        return UnquantizedLinearMethod()
    resolved = linear_group_for_prefix(prefix)
    if resolved is None or not selection.includes(*resolved):
        return UnquantizedLinearMethod()
    _announce(resolved[0], selection.scheme)
    return OnlineFp8LinearMethod(selection.scheme)


def hc_mix_online_fp8_scheme(
    prefix: str, environ: Mapping[str, str] = os.environ
) -> Optional[str]:
    """The FP8 scheme for a hyper-connection mix owned by ``prefix``, or ``None`` for BF16."""
    selection = online_fp8_selection(environ)
    if selection is None or _DRAFT_PATTERN.search(prefix):
        return None
    match = _LAYER_ID_PATTERN.search(prefix)
    layer_id = int(match.group(1)) if match is not None else None
    if not selection.includes("hc_mix", layer_id):
        return None
    _announce("hc_mix", selection.scheme)
    return selection.scheme


def quantize_fp8_per_channel(
    weight: torch.Tensor, row_chunk: int = QUANT_ROW_CHUNK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric E4M3 quantization with one float32 scale per output row.

    Works ``row_chunk`` rows at a time so the float32 transient stays small even for
    ``lm_head``. Returns ``(qweight [rows, cols], scale [rows, 1])``.
    """
    fp8_max = torch.finfo(FP8_DTYPE).max
    tiny = torch.finfo(torch.float32).tiny
    rows = weight.shape[0]
    qweight = torch.empty(weight.shape, dtype=FP8_DTYPE, device=weight.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        chunk = weight[start:stop].to(torch.float32)
        chunk_scale = chunk.abs().amax(dim=1, keepdim=True).div(fp8_max).clamp_min(tiny)
        qweight[start:stop] = chunk.div(chunk_scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
        scale[start:stop] = chunk_scale
    return qweight, scale


def dequantize_fp8_per_channel(
    qweight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return (qweight.to(torch.float32) * scale).to(dtype)


def fp8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_chunk: int = WEIGHT_ONLY_ROW_CHUNK,
) -> torch.Tensor:
    """``F.linear`` with a per-channel FP8 weight dequantized ``row_chunk`` rows at a time."""
    rows = qweight.shape[0]
    if rows <= row_chunk:
        return F.linear(x, dequantize_fp8_per_channel(qweight, scale, x.dtype), bias)
    output = x.new_empty((*x.shape[:-1], rows))
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        output[..., start:stop] = F.linear(
            x, dequantize_fp8_per_channel(qweight[start:stop], scale[start:stop], x.dtype)
        )
    return output if bias is None else output + bias


class OnlineFp8LinearMethod(LinearMethodBase):
    """Loads a BF16 weight unchanged, then replaces it with a per-channel FP8 weight.

    Deliberately not an ``UnquantizedLinearMethod`` subclass: model code uses that
    isinstance check to alias BF16 weights into fused GEMMs (``finalize_fused_in_proj``),
    which would keep the BF16 storage alive and bypass the FP8 path.
    """

    def __init__(self, scheme: str):
        if scheme not in SCHEMES:
            raise ValueError(f"unknown online FP8 scheme {scheme!r}")
        self.scheme = scheme

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        UnquantizedLinearMethod().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype == FP8_DTYPE:
            return
        qweight, scale = quantize_fp8_per_channel(layer.weight.data)
        stored = qweight.t() if self.scheme == "w8a8" else qweight
        layer.weight = Parameter(stored, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.scheme == "w8a16":
            return fp8_weight_only_linear(x, layer.weight, layer.weight_scale, bias)
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
            use_per_token_if_dynamic=True,
        )
