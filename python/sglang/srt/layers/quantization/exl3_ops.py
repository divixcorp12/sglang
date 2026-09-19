"""EXL3 (exllamav3 trellis) linear ops, the kernel oracle, and an eager MoE loop.

Every EXL3 linear in the DeepSeek V4.1 export uses the mul1 codebook, so mcg is
always False. A linear computes y = had(had(x * suh) @ W_inner) * svh, where
had is exllamav3's blockwise 128 Hadamard and W_inner = reconstruct(trellis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.exl3_ext import exl3_ext

AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768


def assert_not_capturing(module_name: str) -> None:
    """Raise if called while a CUDA graph is being captured.

    EXL3 / Engram file-table paths make host syncs (``.tolist()``, ``torch.where``,
    ``.cpu()``) and only work eagerly; under capture they fail with an opaque
    CUDA error. Call this at the entry of every such path so the failure names
    the actual cause instead.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{module_name}: EXL3 / Engram file-table paths cannot run inside a CUDA graph capture; "
            "run them as an eager break (--cuda-graph-backend-decode breakable) "
            "or launch with --disable-cuda-graph"
        )


@dataclass(frozen=True)
class Exl3Tensors:
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    mul1: bool

    def __post_init__(self):
        if self.trellis.dtype != torch.int16 or self.trellis.dim() != 3:
            raise ValueError(
                f"trellis must be int16 [in/16, out/16, 16K], got {self.trellis.dtype} {tuple(self.trellis.shape)}"
            )
        if self.suh.dtype != torch.float16 or self.svh.dtype != torch.float16:
            raise ValueError("suh/svh must be fp16")
        if self.suh.shape != (self.in_features,) or self.svh.shape != (self.out_features,):
            raise ValueError(
                f"suh {tuple(self.suh.shape)} / svh {tuple(self.svh.shape)} do not match trellis {tuple(self.trellis.shape)}"
            )

    @property
    def in_features(self) -> int:
        return self.trellis.shape[0] * 16

    @property
    def out_features(self) -> int:
        return self.trellis.shape[1] * 16

    @property
    def bits(self) -> int:
        return self.trellis.shape[2] // 16


def exl3_dense_weight(t: Exl3Tensors) -> torch.Tensor:
    """The original-basis weight, fp16 [in, out], so that y = x @ W."""
    ext = exl3_ext()
    w = torch.empty((t.in_features, t.out_features), dtype=torch.float16, device=t.trellis.device)
    for start in range(0, t.out_features, MAX_RECONSTRUCT_SLICE_N):
        end = min(start + MAX_RECONSTRUCT_SLICE_N, t.out_features)
        piece = torch.empty((t.in_features, end - start), dtype=torch.float16, device=w.device)
        ext.reconstruct_had_slice(
            piece, t.trellis, t.suh, t.svh[start:], t.bits, False, t.mul1, start
        )
        w[:, start:end] = piece
    return w


def exl3_linear(
    x: torch.Tensor, t: Exl3Tensors, out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    ext = exl3_ext()
    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(-1, t.in_features).to(torch.float16).contiguous()
    rows = x2.shape[0]
    if rows <= AUTO_RECONSTRUCT_THRESHOLD:
        c_dtype = torch.float32 if out_dtype == torch.float32 else torch.float16
        y = torch.empty((rows, t.out_features), dtype=c_dtype, device=x2.device)
        if rows:
            ext.exl3_gemm(x2, t.trellis, y, t.suh, torch.empty_like(x2), t.svh, -1, False, t.mul1, 0)
    else:
        y = torch.matmul(x2, exl3_dense_weight(t))
    return y.to(out_dtype).reshape(*lead, t.out_features)


def exl3_linear_reference(x: torch.Tensor, t: Exl3Tensors) -> torch.Tensor:
    """exllamav3's reconstruct + had_r_128 composition with an fp32 matmul; fp32 [rows, out]."""
    ext = exl3_ext()
    x2 = x.reshape(-1, t.in_features).to(torch.float16).contiguous()
    xh = torch.empty_like(x2)
    ext.had_r_128(x2, xh, t.suh, None, 1.0)
    w = torch.empty((t.in_features, t.out_features), dtype=torch.float16, device=x2.device)
    ext.reconstruct(w, t.trellis, t.bits, False, t.mul1)
    y = (xh.float() @ w.float()).to(torch.float16).contiguous()
    ext.had_r_128(y, y, None, t.svh, 1.0)
    return y.float()


def exl3_moe_loop(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: Sequence[tuple[Exl3Tensors, Exl3Tensors]],
    w2: Sequence[Exl3Tensors],
    swiglu_limit: Optional[float],
    linear: Callable = exl3_linear,
) -> torch.Tensor:
    """Routed experts, one expert at a time (eager; syncs on the expert counts).

    Mirrors the reference Expert.forward: up clamped to +-limit and gate to <= limit
    in fp32, the route weight applied before w2.
    """
    assert_not_capturing("exl3_moe_loop")
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    flat = topk_ids.reshape(-1)
    counts = torch.bincount(flat[flat >= 0], minlength=len(w2)).tolist()
    experts = [expert for expert, count in enumerate(counts) if count]
    exl3_moe_accumulate(out, x, topk_weights, topk_ids, w13, w2, swiglu_limit, experts, linear)
    return out.to(x.dtype)


def exl3_moe_accumulate(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: Union[Sequence, Mapping[int, tuple[Exl3Tensors, Exl3Tensors]]],
    w2: Union[Sequence, Mapping[int, Exl3Tensors]],
    swiglu_limit: Optional[float],
    experts: Iterable[int],
    linear: Callable = exl3_linear,
) -> None:
    """Add ``experts``' routed outputs into the fp32 ``out`` [tokens, hidden].

    ``w13[e]`` / ``w2[e]`` need to exist only for ``experts``, so a streamed
    caller can pass one gathered chunk at a time; ascending ``experts`` keep the
    accumulation order, and so the result, of ``exl3_moe_loop``.
    """
    for expert in experts:
        token, slot = torch.where(topk_ids == expert)
        if token.numel() == 0:
            continue
        xe = x[token]
        gate = linear(xe, w13[expert][0], torch.float32)
        up = linear(xe, w13[expert][1], torch.float32)
        if swiglu_limit is not None and swiglu_limit > 0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        h = F.silu(gate) * up * topk_weights[token, slot].float().unsqueeze(-1)
        out.index_add_(0, token, linear(h.to(x.dtype), w2[expert], torch.float32))


def random_exl3_tensors(
    in_features: int, out_features: int, bits: int, *, device, seed: int
) -> Exl3Tensors:
    """A valid EXL3 linear with random contents: every int16 trellis state decodes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cpu = torch.device("cpu")
    trellis = torch.randint(
        -32768,
        32768,
        (in_features // 16, out_features // 16, 16 * bits),
        generator=g,
        dtype=torch.int32,
        device=cpu,
    ).to(torch.int16)
    sign = lambda n: (torch.randint(0, 2, (n,), generator=g, device=cpu) * 2 - 1).to(torch.float16)
    svh = sign(out_features) * (0.5 + torch.rand(out_features, generator=g, device=cpu)).to(torch.float16)
    return Exl3Tensors(
        trellis=trellis.to(device),
        suh=sign(in_features).to(device),
        svh=svh.to(device),
        mul1=True,
    )
