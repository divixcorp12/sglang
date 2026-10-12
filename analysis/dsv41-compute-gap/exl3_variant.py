"""Private experimental builds of the exllamav3 extension, never the production one.

A variant is the pinned exllamav3 source with `exl3-patches/0001-experiment-knobs.patch` applied (a
separate checkout, `EXL3_VARIANT_SRC`), compiled with extra `-D` defines into its own build dir under
its own module name. With no defines the patched source compiles to the pinned kernels; the patch only
adds `#ifndef` guards and an opt-in env knob.

    python exl3_variant.py build <name>      # compile (CPU only)
    load_variant(<name>)                      # import in a benchmark
    use_variant(<name>)                       # also route sglang's exl3_ext() calls to it
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

from sglang.srt.layers.quantization import exl3_ext as _exl3_ext_mod
from sglang.srt.layers.quantization.exl3_ext import (
    _EXTRA_CFLAGS,
    _EXTRA_CUDA_CFLAGS,
    EXLLAMAV3_COMMIT,
    extension_sources,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "exl3-patches", "0001-experiment-knobs.patch")
BUILD_ROOT = os.environ.get("EXL3_VARIANT_BUILD_ROOT", "/mnt/nvme1/compute-transfer-gap/exl3-builds")

VARIANTS: dict[str, list[str]] = {
    "knobs": [],
    "gemv_d2": ["-DGEMV_STAGE_D=2"],
    "gemv_d3": ["-DGEMV_STAGE_D=3"],
    "gemv_d6": ["-DGEMV_STAGE_D=6"],
    "gemv_nostage": ["-DGEMV_STAGE_BITS_MASK=0"],
    "moe_sh2": ["-DMOE_SH_STAGES=2"],
    "moe_sh4": ["-DMOE_SH_STAGES=4"],
    "moe_fs2": ["-DMOE_FRAG_STAGES=2"],
    "moe_sh5": ["-DMOE_SH_STAGES=5"],
    "moe_sh6": ["-DMOE_SH_STAGES=6"],
    "moe_sh4_fs4": ["-DMOE_SH_STAGES=4", "-DMOE_FRAG_STAGES=4"],
    "moe_sh4_fs2": ["-DMOE_SH_STAGES=4", "-DMOE_FRAG_STAGES=2"],
}


def variant_src() -> str:
    src = os.environ["EXL3_VARIANT_SRC"]
    head = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    if head != EXLLAMAV3_COMMIT:
        raise RuntimeError(f"variant source {src} is at {head}, not the pinned {EXLLAMAV3_COMMIT}")
    diff = subprocess.check_output(["git", "-C", src, "diff"], text=True)
    if diff != open(PATCH).read():
        raise RuntimeError(f"variant source {src} does not carry exactly {PATCH}")
    return src


def provenance(name: str) -> dict:
    return {
        "variant": name,
        "defines": VARIANTS[name],
        "exllamav3_commit": EXLLAMAV3_COMMIT,
        "patch_sha256": hashlib.sha256(open(PATCH, "rb").read()).hexdigest(),
    }


def load_variant(name: str, verbose: bool = False):
    src = variant_src()
    ext_dir = os.path.join(src, "exllamav3", "exllamav3_ext")
    build_dir = os.path.join(BUILD_ROOT, name)
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    return load(
        name=f"exl3_exp_{name}",
        sources=extension_sources(ext_dir),
        extra_include_paths=[ext_dir],
        extra_cflags=_EXTRA_CFLAGS + VARIANTS[name],
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS + VARIANTS[name],
        build_directory=build_dir,
        verbose=verbose,
    )


def use_variant(name: str):
    """Make every already-imported sglang module that bound ``exl3_ext`` return this variant."""
    ext = load_variant(name)

    def _variant():
        return ext

    _exl3_ext_mod.exl3_ext = _variant
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("sglang") and getattr(mod, "exl3_ext", None) is not None:
            if callable(mod.exl3_ext) and mod is not _exl3_ext_mod:
                mod.exl3_ext = _variant
    return ext


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "build":
        sys.exit(__doc__)
    load_variant(sys.argv[2], verbose=True)
    print("BUILT", provenance(sys.argv[2]))
