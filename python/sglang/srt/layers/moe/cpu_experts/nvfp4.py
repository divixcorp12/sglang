"""GPU-layout NVFP4 CpuExpertQuantTrait; explicit layout and scale metadata."""

import ctypes as C
import math
from typing import Optional

import torch

from sglang.srt.layers.quantization.nvfp4_cpu.optimized.native import (
    LayerDescriptor,
    NativeApi,
)


class Nvfp4CpuQuantTrait:
    name = "nvfp4"
    x_dtype = torch.float16
    weights_dtype = torch.float32
    out_dtype = torch.float32
    slab_names = (
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
        "w2_blockscale_swizzled",
        "g1_alphas",
        "g2_alphas",
    )
    layouts = {"gate_up": 0, "up_gate": 1, "up_gate_interleaved64": 2}

    def __init__(
        self,
        library=None,
        *,
        w13_layout: str,
        inv_input_scale13: float,
        inv_input_scale2: float,
        act_limit: Optional[float] = None,
        separate_up_alpha: bool = False,
    ):
        """Scales are reciprocals of the activation factors folded into alphas.

        Pass 1.0 for weight-only alphas. Only layer-wide scalar activation
        factors are supported; per-expert factors must travel with host slots
        in a future descriptor extension. Do not pass a TRTLLM/Marlin/MMA
        layout: this adapter reads packed row-major weights and 128x4 scales.
        ``act_limit`` is the pre-SiLU gate/up clamp; 0 disables it. Bailing,
        GPT-OSS and non-SiLU activation conventions are not supported.
        """
        if w13_layout not in self.layouts:
            raise ValueError(f"unsupported NVFP4 CPU W13 layout: {w13_layout}")
        for scale in (inv_input_scale13, inv_input_scale2):
            if (
                not isinstance(scale, (float, int))
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError(
                    "NVFP4 CPU input scale reciprocals must be positive finite scalars"
                )
        self.layout = self.layouts[w13_layout]
        self.inv13, self.inv2 = float(inv_input_scale13), float(inv_input_scale2)
        self.act_limit = act_limit
        self.api = NativeApi(library)
        self._slabs = {}
        if separate_up_alpha:
            self.slab_names = (*self.slab_names, "g1_alphas_up")

    def check_environment(self):
        # Compiler/runtime independent; native_set_cores validates affinity.
        pass

    def hidden_size(self, slabs):
        return int(slabs["w13_weight"].shape[-1]) * 2

    def register_layer(self, slabs, capacity):
        if (
            self.act_limit is None
            or not math.isfinite(self.act_limit)
            or self.act_limit < 0
        ):
            raise ValueError(
                "set the NVFP4 pre-SiLU act_limit (0 for no clamp) before registration"
            )
        w13, w2 = slabs["w13_weight"], slabs["w2_weight"]
        if w13.ndim != 3 or w2.ndim != 3 or w13.shape[1] % 2:
            raise ValueError(
                "NVFP4 CPU expects W13 [slots,2*N,H/2] and W2 [slots,H,N/2]"
            )
        h, n = self.hidden_size(slabs), int(w13.shape[1]) // 2
        if h % 16 or n % 16 or tuple(w2.shape[1:]) != (h, n // 2):
            raise ValueError(
                "NVFP4 CPU dimensions must be multiples of 16 and W13/W2 must agree"
            )
        if self.layout == 2 and n % 64:
            raise ValueError("interleaved64 W13 needs intermediate divisible by 64")

        def pad(v, a):
            return (v + a - 1) // a * a

        expected = [
            (2 * n, h // 2),
            (h, n // 2),
            (pad(2 * n, 128), pad(h // 16, 4)),
            (pad(h, 128), pad(n // 16, 4)),
        ]
        dtypes = (torch.uint8, torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fn)
        tensors = [slabs[name] for name in self.slab_names]
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("NVFP4 CPU capacity must be a positive integer")
        for i, t in enumerate(tensors):
            if (
                t.device.type != "cpu"
                or not t.is_contiguous()
                or t.ndim < 1
                or t.shape[0] != capacity
            ):
                raise ValueError(
                    f"{self.slab_names[i]} must be contiguous CPU storage with {capacity} slots"
                )
            if i < 4:
                if t.dtype != dtypes[i] or tuple(t.shape[1:]) != expected[i]:
                    raise ValueError(
                        f"{self.slab_names[i]} needs {expected[i]} {dtypes[i]} per slot"
                    )
            elif t.dtype != torch.float32 or t[0].numel() != 1:
                raise ValueError(f"{self.slab_names[i]} needs one FP32 alpha per slot")
        desc = LayerDescriptor(
            abi_version=1,
            capacity=capacity,
            hidden=h,
            intermediate=n,
            w13_layout=self.layout,
            activation=0,
            act_limit=self.act_limit,
            inv_input_scale13=self.inv13,
            inv_input_scale2=self.inv2,
        )
        for i, t in enumerate(tensors):
            desc.slabs[i] = t.data_ptr()
            desc.slot_bytes[i] = t[0].numel() * t.element_size()
        handle = C.c_int64(-1)
        self.api.check(self.api.register(C.byref(desc), C.byref(handle)), "register")
        self._slabs[handle.value] = tensors  # Retain views; slot contents may change.
        return handle.value

    def forward(self, handle, x, slots, weights, out, threads):
        # This convenience path runs under Python; CpuExpertEngine uses the C
        # address directly, without this adapter or any Python work per job.
        h = self.hidden_size(dict(zip(self.slab_names, self._slabs[handle])))
        for t, dtype in (
            (x, self.x_dtype),
            (slots, torch.int64),
            (weights, self.weights_dtype),
            (out, self.out_dtype),
        ):
            if t.device.type != "cpu" or t.dtype != dtype or not t.is_contiguous():
                raise ValueError(
                    "NVFP4 forward needs contiguous CPU tensors with the trait's dtypes"
                )
        if (
            x.shape != (1, h)
            or out.shape != (1, h)
            or slots.ndim != 2
            or slots.shape[0] != 1
            or weights.shape != slots.shape
        ):
            raise ValueError(
                "NVFP4 forward expects x/out [1,H] and slots/weights [1,k]"
            )
        if (
            slots.numel() > 8
            or bool((slots < -1).any())
            or bool((slots >= len(self._slabs[handle][0])).any())
        ):
            raise ValueError(
                "NVFP4 slots must be -1 or inside the host tier, with k<=8"
            )
        ids = (C.c_int32 * slots.numel())(*slots.flatten().tolist())
        self.api.check(
            self.api.forward(
                handle,
                x.data_ptr(),
                ids,
                C.cast(weights.data_ptr(), C.POINTER(C.c_float)),
                len(ids),
                C.cast(out.data_ptr(), C.POINTER(C.c_float)),
                threads,
                0,
            ),
            "forward",
        )

    def native_forward(self):
        return C.cast(self.api.forward, C.c_void_p).value

    def native_set_cores(self, cores):
        values = (C.c_int32 * len(cores))(*cores)
        self.api.check(self.api.set_cores(values, len(values)), "set_cores")

    def free_layer(self, handle):
        # Stop/join CpuExpertEngine before releasing registered slabs.
        self.api.check(self.api.free(handle), "free")
        self._slabs.pop(handle, None)
