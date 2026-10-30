"""Synchronous GPTQ INT3 CPU experts (``moe/gptq_int3/moe_ffi.cpp``) for a layer that holds no expert-stream engine."""

from __future__ import annotations

from sglang.kernels.jit.utils import cache_once, load_jit

_FFI, _KERNEL = "moe/gptq_int3/moe_ffi.cpp", "moe/gptq_int3/kernel.cpp"
# The kernel's own flags (cpu_kernel_build._gptq_int3): explicit FP16 boundaries, no contraction outside explicit FMAs.
_FLAGS = ("-std=c++20", "-O3", "-ffp-contract=off", "-pthread")
# The FFI's integer plane repacking runs on OpenMP threads; no floating point there.
_FFI_FLAGS = (*_FLAGS, "-fopenmp")


@cache_once
def gptq_int3_moe_type():
    """The FFI class ``GptqInt3Moe(threads)``: ``forward(hidden, intermediate, capacity, act_limit, slab_ptrs,
    slab_bytes, x, slots, weights, out)`` runs one job on its Team; ``max_rows()`` / ``max_routes()`` bound a call."""
    import tvm_ffi

    module = load_jit(
        "gptq_int3_moe",
        cpp_files=[_FFI, _KERNEL],
        header_only=False,
        source_cflags={_FFI: list(_FFI_FLAGS), _KERNEL: list(_FLAGS)},
        extra_ldflags=["-fopenmp"],
    )
    module.register_once()

    @tvm_ffi.register_object("sgl.GptqInt3Moe")
    class GptqInt3MoeFFI(tvm_ffi.Object):
        __slots__ = ("__dict__",)

        def __init__(self, threads: int) -> None:
            self.__ffi_init__(threads)

    return GptqInt3MoeFFI
