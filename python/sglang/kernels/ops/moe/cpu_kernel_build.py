"""How a format's CPU expert kernel is compiled into its expert-stream host module.

The kernel's sources join the host module's ``cpp_files`` (``expert_stream_transport._host_module_cached``) and take
their own flags (``load_jit``'s ``source_cflags``) in place of the module's: ``-Ofast`` and the resolved ``-march``
(the toolchain's base flags carry it, so ``SGLANG_JIT_HOST_MARCH`` reaches the kernel), never ``-fno-finite-math-only``
(it changes EXL3's bits). The host's own files keep the module flags. Each kernel is one translation unit, and so is
each of its C files; a format with no CPU kernel compiles none.
"""

from __future__ import annotations

import subprocess
from typing import Optional

import msgspec

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.jit.utils.compile import toolchain
from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_engine.config import ExpertStreamConfig

_HIDDEN = ("-fvisibility=hidden", "-fvisibility-inlines-hidden")
_KERNEL_CXX = ("-std=c++20", "-Ofast", "-pthread", "-fopenmp", *_HIDDEN)
# The vendored GGML subset relies on C's implicit void* conversions.
_KERNEL_C = ("-x", "c", "-std=c11", "-Ofast", "-fvisibility=hidden")


class CpuKernelBuild(msgspec.Struct, frozen=True):
    """One format's kernel as the host module builds it: its sources, the flags each takes in place of the module's,
    and the link flags."""

    cpp_files: tuple[str, ...]
    source_cflags: tuple[tuple[str, tuple[str, ...]], ...]
    ldflags: tuple[str, ...] = ()


def max_m_defines() -> list[str]:
    """-DEXL3_MOE_CPU_MAX_M from SGLANG_EXL3_CPU_MAX_M; none at 0, the kernel's default."""
    max_m = envs.SGLANG_EXL3_CPU_MAX_M.get()
    if not max_m:
        return []
    if max_m < 2 or max_m > 16 or max_m % 2:
        raise ValueError(f"SGLANG_EXL3_CPU_MAX_M must be 0 or even in [2, 16], got {max_m}")
    return [f"-DEXL3_MOE_CPU_MAX_M={max_m}"]


def check_optimized_act_options() -> None:
    """The EXL3 CPU kernel is the residual/128 flavor only: a flag explicitly set to anything else is refused."""
    residual, block = envs.SGLANG_EXL3_CPU_ACT_RESIDUAL, envs.SGLANG_EXL3_CPU_ACT_BLOCK
    if (residual.is_set() and not residual.get()) or (block.is_set() and block.get() != 128):
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS builds the optimized EXL3 CPU kernel "
            f"(SGLANG_EXL3_CPU_ACT_RESIDUAL=1, SGLANG_EXL3_CPU_ACT_BLOCK=128), but residual={residual.get()} "
            f"block={block.get()} are set: unset them or set those values"
        )


def worker_trace_defines(config: ExpertStreamConfig) -> list[str]:
    """A separate diagnostic CPU build; normal forwards compile no worker trace state or clock reads.

    On with SGLANG_EXPERT_STREAM_TRACE's cpu_worker.prefix."""
    if not config.trace.cpu_worker.prefix:
        return []
    if not config.cpu_experts.enabled:
        raise ValueError("EXL3 worker tracing requires the optimized CPU expert kernel")
    if not config.trace.job.prefix:
        raise ValueError("EXL3 worker tracing requires instrumented expert job tracing for attribution")
    return ["-DEXL3_MOE_CPU_WORKER_TRACE=1"]


@cache_once
def _compiler_identity(compiler: str) -> tuple[str, bool]:
    """``compiler``'s version and whether it is Clang."""
    version = subprocess.check_output([compiler, "-dumpfullversion", "-dumpversion"], text=True).strip()
    return version, "clang" in subprocess.check_output([compiler, "--version"], text=True).lower()


def check_kernel_compiler() -> None:
    """The recorded winning arithmetic and OpenMP unrolling were validated with GCC 15, and the kernel now builds with
    the host module's compiler (``CXX``): refuse any other, and a SGLANG_EXL3_CPU_CXX that names a different one."""
    compiler = toolchain.host_compiler_path()
    version, clang = _compiler_identity(compiler)
    if version.split(".", 1)[0] != "15":
        raise RuntimeError(
            f"optimized EXL3 CPU experts require GCC 15; {compiler} reports {version}. "
            "Set CXX to your GCC 15 g++: the kernel is built into the expert-stream host module with it."
        )
    if clang:
        raise RuntimeError("optimized EXL3 CPU experts require GCC 15, rather than Clang")
    named = envs.SGLANG_EXL3_CPU_CXX.get()
    if named and named != compiler:
        raise ValueError(
            f"SGLANG_EXL3_CPU_CXX={named} but the host module builds with CXX={compiler}: the EXL3 CPU kernel is "
            "compiled into it now, so set both to the same GCC 15 g++ (or unset SGLANG_EXL3_CPU_CXX)"
        )


def _exl3(config: ExpertStreamConfig) -> CpuKernelBuild:
    if config.cpu_experts.enabled or config.draft.cpu_experts:
        check_optimized_act_options()
        check_kernel_compiler()
    defines = (*max_m_defines(), *worker_trace_defines(config))
    return CpuKernelBuild(
        cpp_files=("exl3/optimized/kernel.cpp",),
        source_cflags=(("exl3/optimized/kernel.cpp", (*_KERNEL_CXX, *defines)),),
        ldflags=("-fopenmp",),
    )


def _nvfp4(config: ExpertStreamConfig) -> CpuKernelBuild:
    return CpuKernelBuild(
        cpp_files=("nvfp4/optimized/kernel.cpp", "nvfp4/upstream/nvfp4.c"),
        source_cflags=(
            ("nvfp4/optimized/kernel.cpp", _KERNEL_CXX),
            ("nvfp4/upstream/nvfp4.c", _KERNEL_C),
        ),
        ldflags=("-fopenmp",),
    )


def _fake(config: ExpertStreamConfig) -> CpuKernelBuild:
    """The test format's fake kernel: an ordinary optimized host-flag file, no fast-math."""
    return CpuKernelBuild(
        cpp_files=("moe/formats/fake/fake_kernel.cpp",),
        source_cflags=(("moe/formats/fake/fake_kernel.cpp", ("-std=c++20", "-O2", "-pthread", *_HIDDEN)),),
    )


# The formats that have a CPU kernel; one with none (mxfp4) compiles none.
_KERNELS = {"exl3": _exl3, "nvfp4": _nvfp4, "fake": _fake}


def cpu_kernel_build(fmt: str, config: ExpertStreamConfig) -> Optional[CpuKernelBuild]:
    """The CPU kernel to compile into ``fmt``'s host module under ``config``, or None when the format has none."""
    build = _KERNELS.get(fmt)
    return None if build is None else build(config)
