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


def extension_sources(ext_dir: str) -> list[str]:
    sources = []
    for root, _, files in os.walk(ext_dir):
        for name in files:
            if name.endswith((".c", ".cpp", ".cu")):
                sources.append(os.path.join(root, name))
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
    build_dir = os.path.expanduser(envs.SGLANG_EXL3_BUILD_DIR.get())
    os.makedirs(build_dir, exist_ok=True)
    # sm_120 only: the RTX 5090 is the one target, and an unset list makes torch
    # probe the GPU, which a CPU-only build must not touch.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_exl3_ext",
        sources=extension_sources(ext_dir),
        extra_include_paths=[ext_dir],
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        build_directory=build_dir,
        verbose=False,
    )
