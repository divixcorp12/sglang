"""EXL3 binding of the CPU expert kernel, as a ``CpuExpertQuantTrait``.

The kernel is the optimized build of exllamav3's CPU MoE kernel, ``csrc/exl3/optimized/kernel.cpp`` (built by
SGLANG_DSV41_CPU_EXPERTS=1). ``Exl3CpuQuantTrait`` describes each streamed layer's pinned slabs for the kernel's
``make_layer`` (``layer_spec``) and hands out the kernel's address (the extension's torch op
``sglang_exl3_cpu::kernel_address``).
"""

import os
from typing import Any, Mapping, Optional

import torch

from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES


class Exl3CpuQuantTrait:
    """The EXL3 CPU kernel's trait: fp16 activations in, fp32 routed sum out.

    ``ext`` is the EXL3 extension module. ``act_limit`` is the SwiGLU clamp the layers
    run with; the RAM-miss service fills it in at the first registration when it is
    ``None``. ``swizzled`` selects the band-contiguous weight layout, which is
    bit-identical to the native one. ``row_weighted`` splits the layers' output tiles among the workers by chunk row
    count (tile_assignment.hpp), bit-identical too.
    """

    name = "exl3"
    slab_names = EXL3_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self, ext: Any, act_limit: Optional[float], swizzled: bool = False, row_weighted: bool = False):
        self.ext = ext
        self.act_limit = act_limit
        self.swizzled = swizzled
        self.row_weighted = row_weighted

    def check_environment(self) -> None:
        """Require ``EXL3_MOE_CPU_PIN=0``; else the kernel pins to the first cores."""
        if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
            raise ValueError(
                "EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size, read off the gate's input sign vector."""
        return int(slabs["w13_suh"].shape[-1])  # the gate's input sign vector

    def _dims(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> tuple[int, int, int]:
        """``(hidden, intermediate, bits)`` of a layer whose first ``capacity`` slab rows the kernel can address;
        raises if the activation limit is not known yet or a slab would be misaddressed."""
        if self.act_limit is None:
            raise ValueError(
                "the EXL3 CPU kernel needs the layers' activation limit before a layer registers"
            )
        # The w13 trellis ([slot, 2, k/16, n/16, 16 * bits]) fixes every dimension, so each other slab is checked
        # against it rather than against itself.
        w13 = slabs["w13_trellis"]
        hidden, intermediate, bits = int(w13.shape[-3]) * 16, int(w13.shape[-2]) * 16, int(w13.shape[-1]) // 16
        trellis = hidden * intermediate * bits // 16  # int16 elements of one [k/16, n/16, 16 * bits] trellis
        row = {  # elements per slot row: w13 rows hold gate then up, w2 rows hold down
            "w13_trellis": 2 * trellis,
            "w13_suh": 2 * hidden,
            "w13_svh": 2 * intermediate,
            "w2_trellis": trellis,
            "w2_suh": intermediate,
            "w2_svh": hidden,
        }
        for name in self.slab_names:
            slab = slabs[name]
            dtype = torch.int16 if name.endswith("_trellis") else torch.float16
            if (
                slab.device.type != "cpu"
                or not slab.is_contiguous()
                or slab.dtype != dtype
                or slab.shape[0] < capacity
                or slab[0].numel() != row[name]
            ):
                raise ValueError(
                    f"EXL3 slab {name} {tuple(slab.shape)} {slab.dtype} is not {capacity} contiguous CPU rows of "
                    f"{row[name]} {dtype} elements"
                )
        return hidden, intermediate, bits

    def _op(self, name: str):
        """The optimized extension's torch op ``sglang_exl3_cpu::<name>``."""
        try:
            return getattr(torch.ops.sglang_exl3_cpu, name)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"the EXL3 extension {self.ext.__file__} has no sglang_exl3_cpu::{name}: CPU experts need the "
                "optimized CPU kernel, which SGLANG_DSV41_CPU_EXPERTS=1 builds (csrc/exl3/optimized)"
            ) from error

    def kernel_address(self) -> int:
        """The address of the extension's EXL3 CpuExpertKernel (its torch op ``sglang_exl3_cpu::kernel_address``)."""
        return int(self._op("kernel_address")())

    def _params(self, bits: int) -> bytes:
        """``SglangExl3CpuParams``' bytes as the extension packs them (its torch op ``sglang_exl3_cpu::params``)."""
        return self._op("params")(bits, int(self.swizzled), int(self.row_weighted)).numpy().tobytes()

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """The layer's six slabs by base pointer and row size, and ``SglangExl3CpuParams``' bytes
        ({bits, swizzled, row_weighted})."""
        hidden, intermediate, bits = self._dims(slabs, capacity)
        views = [slabs[name] for name in self.slab_names]
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=hidden,
            intermediate=intermediate,
            act_limit=float(self.act_limit),
            # One slot's row: the slab is contiguous (checked), so this is its stride(0), which PyTorch does not keep
            # meaningful for a one-slot slab.
            slabs=tuple((v.data_ptr(), v[0].numel() * v.element_size()) for v in views),
            params=self._params(bits),
            keep=tuple(views),
        )
