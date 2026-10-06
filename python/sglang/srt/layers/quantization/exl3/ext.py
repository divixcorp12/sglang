"""JIT build of exllamav3's CUDA extension for the EXL3 quant method.

Built from a pinned exllamav3 checkout rather than a vendored subset: the link
closure of its template instantiations cannot be read off its #includes, and
the first question is whether the kernels work on sm_120 at all.
"""

from __future__ import annotations

import functools
import os
import subprocess

from sglang.srt.environ import envs

EXLLAMAV3_COMMIT = "02aef45cd681b960a00afcd0749a4ab99e6c1bfe"

# exllamav3 setup.py, Linux branch.
_EXTRA_CFLAGS = ["-Ofast"]
_EXTRA_CUDA_CFLAGS = [
    "-lineinfo",
    "-O3",
    "--use_fast_math",
    "-Xcudafe",
    "--diag_suppress=177",
    "-Xcudafe",
    "--diag_suppress=20012",
]


# exllamav3's CPU MoE kernel with this fork's accuracy options (see its header).
_CSRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../kernels/jit/csrc/exl3"))
VENDORED_CPU_KERNEL = os.path.join(_CSRC, "moe_mul1.cpp")
OPTIMIZED_CPU_KERNEL = os.path.join(_CSRC, "optimized", "kernel.cpp")
# The torch op sglang_exl3_cpu::kernel_address: the optimized extension hands its CpuExpertKernel to Python.
OPTIMIZED_TORCH_OPS = os.path.join(_CSRC, "optimized", "torch_ops.cpp")
_UPSTREAM_CPU_KERNEL = os.path.join("cpu", "moe_mul1.cpp")


def optimized_cpu(defines: list[str]) -> bool:
    return "-DEXL3_MOE_CPU_ACT_RESIDUAL=1" in defines and "-DEXL3_MOE_CPU_ACT_BLOCK=128" in defines


def cpu_compiler() -> str:
    """The optimized CPU kernel's compiler: SGLANG_EXL3_CPU_CXX, else CXX, else c++."""
    return envs.SGLANG_EXL3_CPU_CXX.get() or os.environ.get("CXX", "c++")


def check_cpu_compiler() -> None:
    """The recorded winning arithmetic and OpenMP unrolling were validated with GCC 15."""
    compiler = cpu_compiler()
    version = subprocess.check_output([compiler, "-dumpfullversion", "-dumpversion"], text=True).strip()
    if version.split(".", 1)[0] != "15":
        raise RuntimeError(
            f"optimized EXL3 CPU experts require GCC 15; {compiler} reports {version}. "
            "Set SGLANG_EXL3_CPU_CXX (or CXX) to your GCC 15 g++ before building the extension."
        )
    identity = subprocess.check_output([compiler, "--version"], text=True)
    if "clang" in identity.lower():
        raise RuntimeError("optimized EXL3 CPU experts require GCC 15, rather than Clang")


OPTIMIZED_CPU_DEFINES = ["-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128"]  # as exl3/build.py


def cpu_act_defines() -> list[str]:
    """Compiler defines for the CPU kernel's activation quantization options; empty when both are off.

    With SGLANG_DSV41_CPU_EXPERTS the CPU experts run only the optimized kernel (csrc/exl3/optimized), so the
    defines are always its residual/128 pair; a flag explicitly set to anything else is refused."""
    if envs.SGLANG_DSV41_CPU_EXPERTS.get():
        residual, block = envs.SGLANG_EXL3_CPU_ACT_RESIDUAL, envs.SGLANG_EXL3_CPU_ACT_BLOCK
        if (residual.is_set() and not residual.get()) or (block.is_set() and block.get() != 128):
            raise ValueError(
                "SGLANG_DSV41_CPU_EXPERTS builds the optimized EXL3 CPU kernel (SGLANG_EXL3_CPU_ACT_RESIDUAL=1, "
                f"SGLANG_EXL3_CPU_ACT_BLOCK=128), but residual={residual.get()} block={block.get()} are set: "
                "unset them or set those values"
            )
        return list(OPTIMIZED_CPU_DEFINES)
    defines = []
    if envs.SGLANG_EXL3_CPU_ACT_RESIDUAL.get():
        defines.append("-DEXL3_MOE_CPU_ACT_RESIDUAL=1")
    block = envs.SGLANG_EXL3_CPU_ACT_BLOCK.get()
    if block:
        if block < 0 or block % 16:
            raise ValueError(f"SGLANG_EXL3_CPU_ACT_BLOCK must be 0 or a positive multiple of 16, got {block}")
        defines.append(f"-DEXL3_MOE_CPU_ACT_BLOCK={block}")
    return defines


def build_flavor(defines: list[str]) -> str:
    """The empty string for upstream's kernel, else a suffix naming the options, e.g. "_resid_b128"."""
    flavor = ""
    if "-DEXL3_MOE_CPU_ACT_RESIDUAL=1" in defines:
        flavor += "_resid"
    for d in defines:
        if d.startswith("-DEXL3_MOE_CPU_ACT_BLOCK="):
            flavor += "_b" + d.split("=", 1)[1]
    if optimized_cpu(defines):
        flavor += "_cpu_v1"
    return flavor


def extension_sources(ext_dir: str, cpu_kernel: str | None = None) -> list[str]:
    """Every C/C++/CUDA source under ``ext_dir``; ``cpu_kernel`` replaces upstream's cpu/moe_mul1.cpp."""
    upstream = os.path.join(ext_dir, _UPSTREAM_CPU_KERNEL)
    sources = []
    for root, _, files in os.walk(ext_dir):
        for name in files:
            if name.endswith((".c", ".cpp", ".cu")):
                path = os.path.join(root, name)
                sources.append(cpu_kernel if cpu_kernel and path == upstream else path)
    return sorted(sources)


def _checkout_commit(src: str) -> str:
    return subprocess.check_output(
        ["git", "-C", src, "rev-parse", "HEAD"], text=True
    ).strip()


def _checked_ext_dir(src: str) -> str:
    commit = _checkout_commit(src)
    if commit != EXLLAMAV3_COMMIT:
        raise RuntimeError(
            f"expected exllamav3 {EXLLAMAV3_COMMIT} at {src}, found {commit}"
        )
    return os.path.join(src, "exllamav3", "exllamav3_ext")


@functools.cache
def exl3_ext():
    src = envs.SGLANG_EXL3_SRC.get()
    if not src:
        raise RuntimeError("SGLANG_EXL3_SRC must point at an exllamav3 checkout")
    ext_dir = _checked_ext_dir(src)
    defines = cpu_act_defines()
    flavor = build_flavor(defines)
    optimized = optimized_cpu(defines)
    if optimized:
        check_cpu_compiler()
    # One build directory per flavor: extensions sharing a directory overwrite each other's build.ninja.
    build_dir = os.path.expanduser(envs.SGLANG_EXL3_BUILD_DIR.get())
    if flavor:
        build_dir = os.path.join(build_dir, flavor.lstrip("_"))
    os.makedirs(build_dir, exist_ok=True)
    # sm_120 only: the RTX 5090 is the one target, and an unset list makes torch
    # probe the GPU, which a CPU-only build must not touch.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    # load() reads CXX from the environment; the GCC 15 compiler is set for this build only and restored after.
    saved_cxx = os.environ.get("CXX")
    if optimized:
        os.environ["CXX"] = cpu_compiler()
    try:
        return load(
            name="sglang_exl3_ext" + flavor,
            sources=extension_sources(
                ext_dir, OPTIMIZED_CPU_KERNEL if optimized else VENDORED_CPU_KERNEL if flavor else None
            )
            + ([OPTIMIZED_TORCH_OPS] if optimized else []),
            # The generic vendored kernel uses the upstream CPU header; optimized has its own.
            extra_include_paths=[ext_dir] + ([os.path.join(ext_dir, "cpu")] if flavor else []),
            extra_cflags=_EXTRA_CFLAGS
            + defines
            + (["-march=native", "-std=c++20", "-fopenmp", "-pthread"] if optimized else []),
            extra_ldflags=["-fopenmp"] if optimized else [],
            extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if saved_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = saved_cxx
