"""JIT wrapper for the EXL3 fused MoE's route tables (SGLANG_DSV41_ENABLE_LAYER_FUSION).

``exl3_moe_route_tables`` launches one kernel in place of ``exl3.fused_moe.route_tables`` and the copies around it in
``Exl3FusedMoE.run`` (22 kernels per layer), with bit-identical results; the flag-off path runs the torch chain, and
the parity test compares against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
        "exl3_moe_route_tables",
        *args,
        cuda_files=["moe/exl3/exl3_route_tables.cuh"],
        cuda_wrappers=[("run", f"exl3::exl3_moe_route_tables_gpu<{args}>")],
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
    cpu_lanes: Optional[torch.Tensor] = None,
    dst_slots: Optional[torch.Tensor] = None,
    cpu_out: int = 0,
    cpu_part_stride: int = 0,
) -> None:
    """The fused MoE's route tables and input staging; see ``exl3.fused_moe.route_tables``.

    Writes ``remap64_out`` (``remap`` as int64), ``x16_out`` (``x`` as fp16), zeroes ``out_zero``, and fills
    ``expert_count`` [slots + 1], ``inv_order``, ``weight_sorted`` (fp16) and ``det`` [3, slots + 1].

    CPU experts: ``cpu_lanes`` (int32 ``[2]``: CPU lanes, then the part bits) masks the plan lanes the CPU computed and
    flags the output parts holding their partial sums in bits 0 (part 0, the CPU hits') and 1 (part 1, the CPU misses');
    ``dst_slots`` (int32) are the plan's lane slots, ``cpu_out`` the address of the row's part 0 and
    ``cpu_part_stride`` the floats from part 0 to part 1 (0 for a one-part row). The flagged parts' sum seeds
    ``out_zero``; the CPU routes' slots count 0 and rank last, so the fused kernel and the gather skip them.

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
        cpu_lanes if cpu_lanes is not None else _empty_i32(det.device),
        dst_slots if dst_slots is not None else _empty_i32(det.device),
        int(cpu_out),
        int(cpu_part_stride),
    )


_EMPTY_I32: dict = {}


def _empty_i32(device) -> torch.Tensor:
    """A stable empty int32 tensor per device: the launcher's "off" for the CPU-expert arguments, capture-safe."""
    key = str(device)
    t = _EMPTY_I32.get(key)
    if t is None:
        t = _EMPTY_I32[key] = torch.empty(0, dtype=torch.int32, device=device)
    return t
