"""JIT build of exllamav3's CUDA extension for the EXL3 quant method.

Built from a pinned exllamav3 checkout rather than a vendored subset: the link
closure of its template instantiations cannot be read off its #includes, and
the first question is whether the kernels work on sm_120 at all.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_engine.config import ResolvedConfig, resolved_config
from sglang.srt.layers.moe.expert_engine.removed_env import VENDORED_ENV_MARKER

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


def _max_m_defines() -> list[str]:
    """-DEXL3_MOE_CPU_MAX_M from SGLANG_EXL3_CPU_MAX_M; none at 0, the kernel's default."""
    max_m = envs.SGLANG_EXL3_CPU_MAX_M.get()
    if not max_m:
        return []
    if max_m < 2 or max_m > 16 or max_m % 2:
        raise ValueError(f"SGLANG_EXL3_CPU_MAX_M must be 0 or even in [2, 16], got {max_m}")
    return [f"-DEXL3_MOE_CPU_MAX_M={max_m}"]


def cpu_act_defines(config: Optional[ResolvedConfig] = None) -> list[str]:
    """Compiler defines for the CPU kernel's activation quantization options; empty when both are off.

    With SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS the CPU experts run only the optimized kernel (csrc/exl3/optimized),
    so the defines are always its residual/128 pair; a flag explicitly set to anything else is refused. ``config``
    defaults to ``resolved_config()``."""
    config = config or resolved_config()
    if config.stream.cpu_experts.enabled:
        residual, block = envs.SGLANG_EXL3_CPU_ACT_RESIDUAL, envs.SGLANG_EXL3_CPU_ACT_BLOCK
        if (residual.is_set() and not residual.get()) or (block.is_set() and block.get() != 128):
            raise ValueError(
                "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS builds the optimized EXL3 CPU kernel "
                f"(SGLANG_EXL3_CPU_ACT_RESIDUAL=1, SGLANG_EXL3_CPU_ACT_BLOCK=128), but residual={residual.get()} "
                f"block={block.get()} are set: unset them or set those values"
            )
        return list(OPTIMIZED_CPU_DEFINES) + _max_m_defines()
    if envs.SGLANG_EXL3_CPU_MAX_M.is_set() and envs.SGLANG_EXL3_CPU_MAX_M.get():
        raise ValueError(
            "SGLANG_EXL3_CPU_MAX_M sizes the optimized EXL3 CPU kernel, which SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 "
            "builds; the vendored kernel keeps its own"
        )
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
    """The empty string for upstream's kernel, else a suffix naming the options, e.g. "_resid_b128" or "_resid_b128_m16_cpu_v1"."""
    flavor = ""
    if "-DEXL3_MOE_CPU_ACT_RESIDUAL=1" in defines:
        flavor += "_resid"
    for d in defines:
        if d.startswith("-DEXL3_MOE_CPU_ACT_BLOCK="):
            flavor += "_b" + d.split("=", 1)[1]
        elif d.startswith("-DEXL3_MOE_CPU_MAX_M="):
            flavor += "_m" + d.split("=", 1)[1]
    if optimized_cpu(defines):
        flavor += "_cpu_v1"
    return flavor


def march_suffix(march_flags: list[str]) -> str:
    """A name suffix for the build's ``-march`` flags, so two marches never share a module name or build directory."""
    name = march_flags[0].split("=", 1)[1] if march_flags else "default"
    return "_march_" + re.sub(r"[^A-Za-z0-9]", "_", name)


def worker_trace_defines(config: Optional[ResolvedConfig] = None) -> list[str]:
    """A separate diagnostic CPU build; normal forwards compile no worker trace state or clock reads.

    On with SGLANG_EXPERT_STREAM_TRACE's cpu_worker.prefix; ``config`` defaults to ``resolved_config()``."""
    stream = (config or resolved_config()).stream
    if not stream.trace.cpu_worker.prefix:
        return []
    if not stream.cpu_experts.enabled:
        raise ValueError("EXL3 worker tracing requires the optimized CPU expert kernel")
    if not stream.trace.job.prefix:
        raise ValueError("EXL3 worker tracing requires instrumented expert job tracing for attribution")
    return ["-DEXL3_MOE_CPU_WORKER_TRACE=1"]


def set_vendored_kernel_env(config: ResolvedConfig) -> None:
    """Set the environment variables the vendored and upstream CPU kernels (moe_mul1.cpp) read for themselves.

    Those kernels are exllamav3's code, whose thread pinning and small-job worker cap are read with getenv, so they
    cannot take a parameter. This is the one place the engine config reaches them: EXL3_MOE_CPU_PIN from
    SGLANG_EXL3_MOE_CPU_PIN and EXL3_MOE_CPU_SMALL_WORKERS from SGLANG_EXL3_MOE_CPU_SMALL_WORKERS, each only when
    configured, so an unconfigured launch leaves the kernels' defaults. The optimized kernel reads none of these.
    """
    cpu = config.exl3.cpu
    vendored = {
        "EXL3_MOE_CPU_PIN": cpu.pin,
        "EXL3_MOE_CPU_SMALL_WORKERS": str(cpu.small_workers) if cpu.small_workers else None,
    }
    set_names = []
    for name, value in vendored.items():
        if value is not None:
            os.environ[name] = value
            set_names.append(name)
    if set_names:
        # check_removed_env in a child process would refuse these names; the marker says the engine set them.
        known = [n for n in os.environ.get(VENDORED_ENV_MARKER, "").split(",") if n]
        os.environ[VENDORED_ENV_MARKER] = ",".join(dict.fromkeys(known + set_names))


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
    config = resolved_config()
    defines = cpu_act_defines(config)
    flavor = build_flavor(defines)
    optimized = optimized_cpu(defines)
    march_flags: list[str] = []
    if optimized:
        from sglang.kernels.jit.utils.compile.toolchain import host_arch_flags

        check_cpu_compiler()
        march_flags = host_arch_flags()
        flavor += march_suffix(march_flags)
    else:
        set_vendored_kernel_env(config)
    # One build directory per flavor: extensions sharing a directory overwrite each other's build.ninja.
    build_dir = os.path.expanduser(envs.SGLANG_EXL3_BUILD_DIR.get())
    if flavor:
        build_dir = os.path.join(build_dir, flavor.lstrip("_"))
    trace_defines = worker_trace_defines(config)
    if trace_defines:
        # Same extension name/CUDA flags, separate build directory: CUDA objects may be reused, CPU flags differ.
        build_dir = os.path.join(build_dir, "worker_trace")
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
            + trace_defines
            + ([*march_flags,"-std=c++20", "-fopenmp", "-pthread"] if optimized else []),
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
