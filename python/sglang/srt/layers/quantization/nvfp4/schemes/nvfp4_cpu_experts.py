"""NVFP4 binding of the CPU expert kernel, as a ``CpuExpertQuantTrait``.

The kernel is ``python/sglang/kernels/jit/csrc/nvfp4/optimized/kernel.cpp``, compiled into the expert-stream host
module. ``Nvfp4CpuQuantTrait`` describes a layer's pinned slabs and parameters for the kernel's ``make_layer``
(``layer_spec``).
"""

import struct
from typing import Mapping, Optional

import torch

from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

# The layer's required slabs, in the kernel's slab order: packed E2M1 weights, 128x4-swizzled E4M3 scales, FP32 alphas
# per slot. Its seventh, up_alpha, is optional (up shares gate_alpha without it), so it is not a required slab name.
NVFP4_CPU_SLAB_NAMES = ("w13", "w2", "sf13", "sf2", "gate_alpha", "down_alpha")
_UP_ALPHA = "up_alpha"


class Nvfp4CpuQuantTrait:
    """The NVFP4 CPU kernel's trait: fp16 activations in, fp32 routed sum out.

    The slabs do not encode the layer's shape, so ``hidden`` and ``intermediate`` are given here; the kernel checks
    every slab's slot stride against them at registration. ``w13_layout`` is 0 for [gate, up], 1 for [up, gate], 2
    for alternating 64-row [up, gate] chunks. ``inv_input_scale13``/``inv_input_scale2`` cancel activation scales
    folded into GPU GEMM alphas (1 for weight-only alphas). ``act_limit`` 0 means no SwiGLU clamp; ``None`` lets the
    RAM-miss service fill it in at the first registration.
    """

    name = "nvfp4"
    slab_names = NVFP4_CPU_SLAB_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float32
    out_dtype = torch.float32

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        act_limit: Optional[float] = None,
        w13_layout: int = 0,
        inv_input_scale13: float = 1.0,
        inv_input_scale2: float = 1.0,
    ):
        self.hidden = hidden
        self.intermediate = intermediate
        self.act_limit = act_limit
        self.w13_layout = w13_layout
        self.inv_input_scale13 = inv_input_scale13
        self.inv_input_scale2 = inv_input_scale2

    def check_environment(self) -> None:
        """Nothing to check: the kernel pins its workers only to the cores a call names."""

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size, as configured (NVFP4 slabs do not encode it)."""
        return self.hidden

    def _checked(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> tuple[str, ...]:
        """The slab names present (with ``up_alpha`` when given), each checked as ``capacity`` contiguous CPU rows;
        raises if the activation limit is not known yet or a slab is misaddressed."""
        if self.act_limit is None:
            raise ValueError("the NVFP4 CPU kernel needs the layers' activation limit before a layer registers")
        names = self.slab_names + ((_UP_ALPHA,) if _UP_ALPHA in slabs else ())
        for name in names:
            slab = slabs.get(name)
            if slab is None or slab.device.type != "cpu" or not slab.is_contiguous() or slab.shape[0] < capacity:
                shape = None if slab is None else tuple(slab.shape)
                raise ValueError(f"NVFP4 slab {name} {shape} is not {capacity} contiguous CPU rows")
        return names

    def attach_host(self, host) -> None:
        """Nothing to take: the kernel's parameters need no host query."""

    @staticmethod
    def _float_bits(value: float) -> int:
        """``value`` as the IEEE binary32 bit pattern the host's params list carries."""
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """The seven slabs (up_alpha (0, 0) when absent) and ``SglangNvfp4CpuParams``' fields as a list: w13_layout,
        then the two inverse scales' binary32 bits."""
        names = self._checked(slabs, capacity)
        views = [slabs[name] for name in names]
        pairs = [(v.data_ptr(), v[0].numel() * v.element_size()) for v in views]
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=self.hidden,
            intermediate=self.intermediate,
            act_limit=float(self.act_limit),
            slabs=tuple(pairs + [(0, 0)] * (len(self.slab_names) + 1 - len(pairs))),
            params=(
                int(self.w13_layout),
                self._float_bits(self.inv_input_scale13),
                self._float_bits(self.inv_input_scale2),
            ),
            keep=tuple(views),
        )
