"""JIT wrappers for the DSV4.1 EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION).

* ``exl3_silu_mul_clamp_half`` -- the shared expert's ``gate_up.to(bf16)``, ``silu_and_mul_clamp`` and the down
  projection's ``.to(fp16)`` as one kernel on exl3_gemm's fp16 output;
* ``exl3_scale_to_bf16`` -- the routed MoE output's ``out.to(bf16) * routed_scaling_factor``.

Both are bit-identical to the torch chains; the flag-off path runs those chains, and the parity tests compare against
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _silu_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsv41_exl3_silu_mul_clamp_half",
        *args,
        cuda_files=["moe/dsv41_cast_fusion.cuh"],
        cuda_wrappers=[("run", f"exl3_silu_mul_clamp_half<{args}>")],
        # The same flags as silu_and_mul_clamp's module, so silu_and_mul compiles to the same instructions.
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _scale_module() -> Module:
    return load_jit(
        "dsv41_exl3_scale_to_bf16",
        cuda_files=["moe/dsv41_cast_fusion.cuh"],
        cuda_wrappers=[("run", "exl3_scale_to_bf16")],
    )


def exl3_silu_mul_clamp_half(gate_up: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """fp16 ``[rows, 2 * inter]`` gate||up -> fp16 ``[rows, inter]``, rounded through bf16 as the unfused chain is."""
    out = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)
    _silu_module().run(gate_up, out, float(swiglu_limit))
    return out


def exl3_scale_to_bf16(routed: torch.Tensor, factor: float) -> torch.Tensor:
    """``routed.to(bf16) * factor`` for a contiguous fp32 tensor, in one kernel."""
    out = torch.empty(routed.shape, dtype=torch.bfloat16, device=routed.device)
    _scale_module().run(routed.reshape(-1), out.view(-1), float(factor))
    return out
