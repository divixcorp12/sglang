"""EXL3 binding of the CPU expert kernel, as a ``CpuExpertQuantTrait``.

The kernel is the optimized build of exllamav3's CPU MoE kernel,
``python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`` (built by
SGLANG_DSV41_CPU_EXPERTS=1; only it exports the CPU experts C ABI). ``Exl3CpuQuantTrait``
registers each streamed layer's pinned slabs with its ``register_layer`` by base pointer and
slot stride, and exposes the kernel's C entry points (forward, keep-warm and core placement)
that the native CPU expert thread calls without Python.
"""

import ctypes
import os
from typing import Any, Mapping, Optional

import torch

from sglang.srt.layers.moe.cpu_experts.pool import (
    CPU_EXPERTS_LAYER_ABI_VERSION,
    CpuExpertsLayer,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES


class Exl3CpuParams(ctypes.Structure):
    """``SglangExl3CpuParams`` (``exl3_cpu/optimized/cpu_experts_cabi.h``); the field order is the C struct's."""

    _fields_ = [("bits", ctypes.c_int32), ("swizzled", ctypes.c_int32)]


class Exl3CpuQuantTrait:
    """The EXL3 CPU kernel's trait: fp16 activations in, fp32 routed sum out.

    ``ext`` is the EXL3 extension module. ``act_limit`` is the SwiGLU clamp the layers
    run with; the RAM-miss service fills it in at the first registration when it is
    ``None``. ``swizzled`` selects the band-contiguous weight layout, which is
    bit-identical to the native one.
    """

    name = "exl3"
    slab_names = EXL3_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self, ext: Any, act_limit: Optional[float], swizzled: bool = False):
        self.ext = ext
        self.act_limit = act_limit
        self.swizzled = swizzled
        self._slabs: dict[int, list[torch.Tensor]] = {}

    def check_environment(self) -> None:
        """Require ``EXL3_MOE_CPU_PIN=0``; else the kernel pins to the first cores."""
        if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
            raise ValueError(
                "EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size, read off the gate's input sign vector."""
        return int(slabs["w13_suh"].shape[-1])  # the gate's input sign vector

    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        """Register one layer's first ``capacity`` slab rows with the kernel, by base pointer.

        The kernel addresses slot ``s`` of each slab at its base plus ``s`` rows, so each slab must be a contiguous
        CPU tensor of at least ``capacity`` rows of the format's row size; the trait keeps the tensors alive until
        ``free_layer``. Returns the kernel's layer handle. Raises if the activation limit is not known yet, if a slab
        would be misaddressed, or if the kernel refuses the registration.
        """
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
        params = Exl3CpuParams(bits=bits, swizzled=int(self.swizzled))
        layer = CpuExpertsLayer(
            abi_version=CPU_EXPERTS_LAYER_ABI_VERSION,
            capacity=capacity,
            hidden=hidden,
            intermediate=intermediate,
            activation=0,
            act_limit=self.act_limit,
            slab_count=len(self.slab_names),
            params=ctypes.addressof(params),
        )
        for i, name in enumerate(self.slab_names):
            layer.slabs[i] = slabs[name].data_ptr()
            # One slot's row: the slab is contiguous (checked above), so this is its stride(0), which PyTorch does not
            # keep meaningful for a one-slot slab.
            layer.slot_bytes[i] = slabs[name][0].numel() * slabs[name].element_size()
        fn = self._native("sglang_exl3_cpu_experts_register_layer")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(CpuExpertsLayer), ctypes.POINTER(ctypes.c_int64)]
        handle = ctypes.c_int64(-1)
        status = fn(ctypes.byref(layer), ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"the EXL3 CPU kernel refused the layer's slabs: status {status}")
        # The kernel stores views: the slabs (and, per the ABI, params) stay alive until free_layer.
        self._slabs[handle.value] = [slabs[name] for name in self.slab_names] + [params]
        return handle.value

    def forward(self, handle, x, slots, weights, out, threads) -> None:
        """Overwrite ``out`` with the routed sum of ``slots`` over layer ``handle``."""
        self.ext.exl3_moe_cpu_forward(handle, x, slots, weights, out, threads)

    def free_layer(self, handle) -> None:
        """Release the kernel's layer ``handle`` and the slabs it addressed."""
        self.ext.exl3_moe_cpu_free_layer(handle)
        self._slabs.pop(handle, None)

    def _native(self, name: str):
        """The C function ``name`` of the extension's vendored-kernel ABI.

        Upstream exllamav3's unflavored build lacks these symbols, so a missing one
        raises ``RuntimeError`` naming the build flavor that provides them.
        """
        import ctypes

        try:
            return getattr(ctypes.CDLL(self.ext.__file__), name)
        except AttributeError as error:
            raise RuntimeError(
                f"the EXL3 extension {self.ext.__file__} has no {name}: CPU experts need the vendored CPU kernel, "
                "which SGLANG_DSV41_CPU_EXPERTS=1 builds (exl3_cpu/optimized)"
            ) from error

    def native_forward(self) -> int:
        """The address of the kernel's forward function, for the native CPU thread."""
        import ctypes

        return ctypes.cast(
            self._native("sglang_exl3_cpu_experts_forward"), ctypes.c_void_p
        ).value

    def native_keep_warm(self) -> int:
        """The address of the kernel's keep-warm function, for the idle native CPU thread."""
        import ctypes

        return ctypes.cast(
            self._native("sglang_exl3_cpu_experts_keep_warm"), ctypes.c_void_p
        ).value

    def native_set_cores(self, cores) -> None:
        """Place the kernel's workers on ``cores``, before its first forward."""
        import ctypes

        fn = self._native("sglang_exl3_cpu_experts_set_cores")
        fn.argtypes, fn.restype = (
            [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32],
            ctypes.c_int,
        )
        array = (ctypes.c_int32 * len(cores))(*cores)
        result = fn(array, len(cores))
        if result == 2:
            raise RuntimeError(
                f"the EXL3 CPU kernel refused cores {list(cores)} (status 2): either a core repeats or is "
                "out of range, or the kernel's workers already ran, so something called "
                "the CPU kernel before the CPU expert thread"
            )
        if result != 0:
            raise RuntimeError(
                f"the EXL3 CPU kernel could not take cores {list(cores)}: status {result}"
            )
