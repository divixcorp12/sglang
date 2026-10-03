"""ctypes declarations for ABI v1, also usable without importing PyTorch."""

import ctypes as C
import os


class LayerDescriptor(C.Structure):
    _fields_ = [
        ("abi_version", C.c_uint32),
        ("capacity", C.c_int32),
        ("hidden", C.c_int32),
        ("intermediate", C.c_int32),
        ("w13_layout", C.c_int32),
        ("activation", C.c_int32),
        ("act_limit", C.c_float),
        ("inv_input_scale13", C.c_float),
        ("inv_input_scale2", C.c_float),
        ("slabs", C.c_void_p * 7),
        ("slot_bytes", C.c_uint64 * 7),
    ]


class NativeApi:
    def __init__(self, library=None):
        path = library or os.environ.get("SGLANG_NVFP4_CPU_LIBRARY")
        if not path:
            raise RuntimeError(
                "Build nvfp4_cpu/optimized/build.py and set SGLANG_NVFP4_CPU_LIBRARY"
            )
        self.library = C.CDLL(os.fspath(path))
        self.register = self.library.sglang_nvfp4_cpu_experts_register_slabs
        self.register.argtypes = [C.POINTER(LayerDescriptor), C.POINTER(C.c_int64)]
        self.free = self.library.sglang_nvfp4_cpu_experts_free_layer
        self.free.argtypes = [C.c_int64]
        self.forward = self.library.sglang_nvfp4_cpu_experts_forward
        self.forward.argtypes = [
            C.c_int64,
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_float),
            C.c_int32,
            C.POINTER(C.c_float),
            C.c_int32,
            C.c_int32,
        ]
        self.set_cores = self.library.sglang_nvfp4_cpu_experts_set_cores
        self.set_cores.argtypes = [C.POINTER(C.c_int32), C.c_int32]
        for fn in (self.register, self.free, self.forward, self.set_cores):
            fn.restype = C.c_int

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"NVFP4 CPU {operation} failed: status {status}")
