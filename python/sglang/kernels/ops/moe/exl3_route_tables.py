"""JIT wrapper for the EXL3 fused MoE's route tables (SGLANG_DSV41_ENABLE_LAYER_FUSION).

``exl3_moe_route_tables`` launches one kernel in place of ``exl3_fused_moe.route_tables`` and the copies around it in
``Exl3FusedMoE.run`` (22 kernels per layer), with bit-identical results; the flag-off path runs the torch chain, and
the parity test compares against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@cache_once
def _route_tables_module(
    remap: torch.dtype, weight: torch.dtype, x: torch.dtype
) -> Module:
    args = make_cpp_args(remap, weight, x)
    return load_jit(
        "dsv41_exl3_moe_route_tables",
        *args,
        cuda_files=["moe/dsv41_layer_fusion.cuh"],
        cuda_wrappers=[("run", f"exl3_moe_route_tables_gpu<{args}>")],
    )


def exl3_moe_route_tables(
    remap: torch.Tensor,
    weights: torch.Tensor,
    keep: torch.Tensor,
    x: torch.Tensor,
    remap64_out: torch.Tensor,
    x16_out: torch.Tensor,
    out_zero: torch.Tensor,
    expert_count: torch.Tensor,
    inv_order: torch.Tensor,
    weight_sorted: torch.Tensor,
    det: torch.Tensor,
) -> None:
    """The fused MoE's route tables and input staging; see ``exl3_fused_moe.route_tables``.

    Writes ``remap64_out`` (``remap`` as int64), ``x16_out`` (``x`` as fp16), zeroes ``out_zero``, and fills
    ``expert_count`` [slots + 1], ``inv_order``, ``weight_sorted`` (fp16) and ``det`` [3, slots + 1].

    The launcher checks every tensor; this refuses only a dtype that has no instantiation.
    """
    for name, tensor, dtypes in (
        ("remap", remap, _ID_DTYPES),
        ("weights", weights, _FLOAT_DTYPES),
        ("x", x, _FLOAT_DTYPES),
    ):
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} must be {dtypes}, got {tensor.dtype}")
    _route_tables_module(remap.dtype, weights.dtype, x.dtype).run(
        remap,
        weights,
        keep,
        x,
        remap64_out,
        x16_out,
        out_zero,
        expert_count,
        inv_order,
        weight_sorted,
        det,
    )
