"""NVFP4 binding of the CPU expert kernel, as a ``CpuExpertQuantTrait``.

The kernel is ``python/sglang/kernels/jit/csrc/nvfp4/optimized/kernel.cpp``, built and loaded by
``nvfp4.ext.nvfp4_cpu_library``. ``Nvfp4CpuQuantTrait`` registers a layer's pinned slabs with the kernel's
``register_layer`` by base pointer and slot stride, runs its C forward, and exposes the C entry points (forward,
keep-warm and core placement) that the native CPU expert thread calls without Python.
"""

import ctypes
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.cpu_experts.pool import (
    CPU_EXPERTS_FORWARD_ABI_VERSION,
    CPU_EXPERTS_LAYER_ABI_VERSION,
    CpuExpertsForwardCall,
    CpuExpertsLayer,
)

# The descriptor's required slabs, in cpu_experts_cabi.h order: packed E2M1 weights, 128x4-swizzled E4M3 scales, FP32
# alphas per slot. Its seventh, up_alpha, is optional (up shares gate_alpha without it), so it is not a pool slab name.
NVFP4_CPU_SLAB_NAMES = ("w13", "w2", "sf13", "sf2", "gate_alpha", "down_alpha")
_UP_ALPHA = "up_alpha"
_STATUS = {1: "internal error, reason on stderr", 2: "invalid arguments", 3: "concurrent use"}


class Nvfp4CpuParams(ctypes.Structure):
    """``SglangNvfp4CpuParams`` (``csrc/nvfp4/optimized/cpu_experts_cabi.h``); the field order is the C struct's."""

    _fields_ = [
        ("w13_layout", ctypes.c_int32),
        ("inv_input_scale13", ctypes.c_float),
        ("inv_input_scale2", ctypes.c_float),
    ]


class Nvfp4CpuQuantTrait:
    """The NVFP4 CPU kernel's trait: fp16 activations in, fp32 routed sum out.

    The slabs do not encode the layer's shape, so ``hidden`` and ``intermediate`` are given here; the kernel checks
    every slab's slot stride against them at registration. ``w13_layout`` is 0 for [gate, up], 1 for [up, gate], 2
    for alternating 64-row [up, gate] chunks. ``inv_input_scale13``/``inv_input_scale2`` cancel activation scales
    folded into GPU GEMM alphas (1 for weight-only alphas). ``act_limit`` 0 means no SwiGLU clamp; ``None`` lets the
    RAM-miss service fill it in at the first registration. ``library`` is
    the loaded kernel, by default ``nvfp4_cpu_library()``.
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
        library: Optional[ctypes.CDLL] = None,
    ):
        self.hidden = hidden
        self.intermediate = intermediate
        self.act_limit = act_limit
        self.w13_layout = w13_layout
        self.inv_input_scale13 = inv_input_scale13
        self.inv_input_scale2 = inv_input_scale2
        if library is None:
            from sglang.srt.layers.quantization.nvfp4.ext import nvfp4_cpu_library

            library = nvfp4_cpu_library()
        self.library = library
        self._slabs: dict[int, list] = {}

    def check_environment(self) -> None:
        """Nothing to check: the kernel pins its workers only to the cores ``native_set_cores`` gives it."""

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size, as configured (NVFP4 slabs do not encode it)."""
        return self.hidden

    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        """Register one layer's first ``capacity`` slab rows with the kernel, by base pointer.

        Each slab must be a contiguous CPU tensor of at least ``capacity`` rows; an ``up_alpha`` slab is optional. The
        trait keeps the tensors alive until ``free_layer``. Returns the kernel's layer handle. Raises if the
        activation limit is not known yet, a slab is misaddressed, or the kernel refuses the registration (which it
        does for a slot stride too small for the layer's shape).
        """
        if self.act_limit is None:
            raise ValueError("the NVFP4 CPU kernel needs the layers' activation limit before a layer registers")
        names = self.slab_names + ((_UP_ALPHA,) if _UP_ALPHA in slabs else ())
        for name in names:
            slab = slabs.get(name)
            if slab is None or slab.device.type != "cpu" or not slab.is_contiguous() or slab.shape[0] < capacity:
                shape = None if slab is None else tuple(slab.shape)
                raise ValueError(f"NVFP4 slab {name} {shape} is not {capacity} contiguous CPU rows")
        params = Nvfp4CpuParams(
            w13_layout=self.w13_layout,
            inv_input_scale13=self.inv_input_scale13,
            inv_input_scale2=self.inv_input_scale2,
        )
        layer = CpuExpertsLayer(
            abi_version=CPU_EXPERTS_LAYER_ABI_VERSION,
            capacity=capacity,
            hidden=self.hidden,
            intermediate=self.intermediate,
            activation=0,
            act_limit=self.act_limit,
            slab_count=len(self.slab_names) + 1,
            params=ctypes.addressof(params),
        )
        kept = []
        for i, name in enumerate(names):
            slab = slabs[name]
            layer.slabs[i] = slab.data_ptr()
            # One slot's row: the slab is contiguous (checked above), so this is its stride(0), which PyTorch does not
            # keep meaningful for a one-slot slab.
            layer.slot_bytes[i] = slab[0].numel() * slab.element_size()
            kept.append(slab)
        fn = self.library.sglang_nvfp4_cpu_experts_register_layer
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(CpuExpertsLayer), ctypes.POINTER(ctypes.c_int64)]
        handle = ctypes.c_int64(-1)
        status = fn(ctypes.byref(layer), ctypes.byref(handle))
        if status != 0:
            reason = _STATUS.get(status, "unexpected status")
            raise RuntimeError(f"the NVFP4 CPU kernel failed to register the layer: status {status} ({reason})")
        # The kernel stores views: the slabs and params stay alive until free_layer.
        self._slabs[handle.value] = kept + [params]
        return handle.value

    def forward(self, handle, x, slots, weights, out, threads) -> None:
        """Overwrite ``out`` with the routed sum of ``slots`` over layer ``handle``.

        ``x`` is fp16 ``[rows, hidden]``, ``slots`` ``[rows, k]`` (-1 skips), ``weights`` ``[rows, k]``, ``out`` fp32
        ``[rows, hidden]``, all on the CPU. A failed call raises and leaves ``out`` untouched.
        """
        rows, k = slots.shape
        tensors = (x, slots, weights, out)
        if (
            any(t.device.type != "cpu" for t in tensors)
            or x.dtype != torch.float16
            or out.dtype != torch.float32
            or not out.is_contiguous()
            or tuple(x.shape) != (rows, self.hidden)
            or tuple(out.shape) != (rows, self.hidden)
            or tuple(weights.shape) != (rows, k)
        ):
            raise ValueError(
                f"the NVFP4 CPU forward takes CPU fp16 x and contiguous fp32 out of [rows, {self.hidden}] and "
                f"[rows, k] slots and weights; got x {tuple(x.shape)} {x.dtype}, out {tuple(out.shape)} {out.dtype}, "
                f"slots {tuple(slots.shape)}, weights {tuple(weights.shape)}"
            )
        x = x.contiguous()
        slots32 = slots.to(torch.int32).contiguous()
        weights32 = weights.to(torch.float32).contiguous()
        call = CpuExpertsForwardCall(
            abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION,
            rows=rows,
            layer=handle,
            x=x.data_ptr(),
            slots=ctypes.cast(slots32.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            weights=ctypes.cast(weights32.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            k=k,
            threads=threads,
            accumulate=0,
        )
        fn = self.library.sglang_nvfp4_cpu_experts_forward
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(CpuExpertsForwardCall)]
        status = fn(ctypes.byref(call))
        if status != 0:
            raise RuntimeError(
                f"the NVFP4 CPU forward failed: status {status} ({_STATUS.get(status, 'unexpected status')})"
            )

    def free_layer(self, handle) -> None:
        """Release the kernel's layer ``handle``, then the slabs it addressed.

        On a refusal (a forward in flight, or an unknown handle) the kernel may still read the slabs, so they stay
        alive and this raises.
        """
        fn = self.library.sglang_nvfp4_cpu_experts_free_layer
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int64]
        status = fn(handle)
        if status != 0:
            raise RuntimeError(
                f"the NVFP4 CPU kernel refused to free layer {handle}: status {status} "
                f"({_STATUS.get(status, 'unexpected status')})"
            )
        self._slabs.pop(handle, None)

    def native_forward(self) -> int:
        """The address of the kernel's forward function, for the native CPU thread."""
        return ctypes.cast(self.library.sglang_nvfp4_cpu_experts_forward, ctypes.c_void_p).value

    def native_keep_warm(self) -> int:
        """The address of the kernel's keep-warm function, for the idle native CPU thread."""
        return ctypes.cast(self.library.sglang_nvfp4_cpu_experts_keep_warm, ctypes.c_void_p).value

    def native_set_cores(self, cores: Sequence[int]) -> None:
        """Place the kernel's workers on ``cores``, before its first forward."""
        fn = self.library.sglang_nvfp4_cpu_experts_set_cores
        fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32], ctypes.c_int
        array = (ctypes.c_int32 * max(len(cores), 1))(*cores)
        result = fn(array, len(cores))
        if result != 0:
            raise RuntimeError(
                f"the NVFP4 CPU kernel refused cores {list(cores)}: status {result} "
                f"({_STATUS.get(result, 'unexpected status')}; 2 means a core repeats or is out of range, or the "
                "kernel's workers already ran)"
            )
