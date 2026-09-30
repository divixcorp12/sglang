"""EXL3 CPU expert kernel (exllamav3 cpu/moe_mul1.cpp, vendored) as a ``CpuExpertQuantTrait``."""

import os
from typing import Any, Mapping, Optional

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES


def _one_part(slab: torch.Tensor, rank: int) -> torch.Tensor:
    """``slab`` as ``[slot, ...]`` of ``rank`` dims, from either that or the tier's ``[slot, 1, ...]``."""
    if slab.dim() == rank + 1 and slab.shape[1] == 1:
        return slab[:, 0]
    if slab.dim() != rank:
        raise ValueError(f"an EXL3 w2 slab of shape {tuple(slab.shape)} is neither [slot, ...] nor [slot, 1, ...]")
    return slab


class Exl3CpuQuantTrait:
    name = "exl3"
    slab_names = EXL3_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self, ext: Any, act_limit: Optional[float], swizzled: bool = False):
        self.ext = ext
        self.act_limit = act_limit
        self.swizzled = swizzled

    def check_environment(self) -> None:
        if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
            raise ValueError(
                "EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        return int(slabs["w13_suh"].shape[-1])  # the gate's input sign vector

    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        if self.act_limit is None:
            raise ValueError("the EXL3 CPU kernel needs the layers' activation limit before a layer registers")
        w13_t, w13_u, w13_v = slabs["w13_trellis"], slabs["w13_suh"], slabs["w13_svh"]
        # The pinned tier keeps w2 with its one-part axis ([slot, 1, ...], the format's row shape); the kernel takes
        # one matrix per expert, so that axis is dropped as a view.
        w2_t, w2_u, w2_v = (
            _one_part(slabs[name], rank) for name, rank in (("w2_trellis", 4), ("w2_suh", 2), ("w2_svh", 2))
        )
        rows = range(capacity)
        # Gate is w13 part 0 and up is part 1; each [slot, part] view is contiguous. Activation 0 is
        # silu with the swiglu limit, as the GPU graph path (exl3_fused_moe.py) runs it.
        return self.ext.exl3_moe_cpu_make_layer(
            [w13_t[s, 0] for s in rows],
            [w13_u[s, 0] for s in rows],
            [w13_v[s, 0] for s in rows],
            [w13_t[s, 1] for s in rows],
            [w13_u[s, 1] for s in rows],
            [w13_v[s, 1] for s in rows],
            [w2_t[s] for s in rows],
            [w2_u[s] for s in rows],
            [w2_v[s] for s in rows],
            [],
            [],
            [],
            0,
            self.act_limit,
            1 if self.swizzled else 0,
        )

    def forward(self, handle, x, slots, weights, out, threads) -> None:
        self.ext.exl3_moe_cpu_forward(handle, x, slots, weights, out, threads)

    def free_layer(self, handle) -> None:
        self.ext.exl3_moe_cpu_free_layer(handle)

    def _native(self, name: str):
        # The vendored kernel's C ABI (exl3_cpu/moe_mul1.cpp); upstream's file, the unflavored build, lacks it.
        import ctypes

        try:
            return getattr(ctypes.CDLL(self.ext.__file__), name)
        except AttributeError as error:
            raise RuntimeError(
                f"the EXL3 extension {self.ext.__file__} has no {name}: CPU experts need the vendored CPU kernel, "
                "which a build flavor selects (SGLANG_EXL3_CPU_ACT_RESIDUAL=1 SGLANG_EXL3_CPU_ACT_BLOCK=128)"
            ) from error

    def native_forward(self) -> int:
        import ctypes

        return ctypes.cast(self._native("sglang_exl3_cpu_experts_forward"), ctypes.c_void_p).value

    def native_set_cores(self, cores) -> None:
        import ctypes

        fn = self._native("sglang_exl3_cpu_experts_set_cores")
        fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32], ctypes.c_int
        array = (ctypes.c_int32 * len(cores))(*cores)
        result = fn(array, len(cores))
        if result != 0:
            raise RuntimeError(
                f"the EXL3 CPU kernel refused cores {list(cores)} ({result}): its pool already runs, "
                "so something called the CPU kernel before the CPU expert thread"
            )
