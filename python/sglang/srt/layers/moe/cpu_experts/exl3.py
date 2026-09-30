"""EXL3 CPU expert kernel (exllamav3 cpu/moe_mul1.cpp, vendored) as a ``CpuExpertQuantTrait``."""

import os
from typing import Any, Mapping

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES


class Exl3CpuQuantTrait:
    name = "exl3"
    slab_names = EXL3_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self, ext: Any, act_limit: float, swizzled: bool = False):
        self.ext = ext
        self.act_limit = act_limit
        self.swizzled = swizzled

    def check_environment(self) -> None:
        if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
            raise ValueError(
                "EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )

    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        w13_t, w13_u, w13_v = slabs["w13_trellis"], slabs["w13_suh"], slabs["w13_svh"]
        w2_t, w2_u, w2_v = slabs["w2_trellis"], slabs["w2_suh"], slabs["w2_svh"]
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
