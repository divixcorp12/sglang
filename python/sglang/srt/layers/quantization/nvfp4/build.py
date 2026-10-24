"""Build a native harness linked with the NVFP4 CPU expert kernel, without CMake or torch.

The server builds the kernel into the expert-stream host module (``kernels/ops/moe/cpu_kernel_build.py``); this builds
the same sources for the harnesses that call the kernel accessor themselves (test/registered/unit/kernels/*.cpp,
test/manual/dsv41/nvfp4_cpu_forward_ab.cpp, nvfp4/bench).

The kernel is optimized/kernel.cpp plus the vendored GGML C subset (../upstream/nvfp4.c, compiled as C: it relies on
C's implicit void* conversions). Both build with ``-Ofast -march=<resolved>``, the kernel flag set of every CPU
expert kernel. The ISA tier (scalar or AVX2) is compile-time, from the resolved ``-march``
(``SGLANG_JIT_HOST_MARCH``, else this machine's): an executable runs only on CPUs that have its -march. A lower tier is
built at a lower -march (``x86-64-v3`` AVX2, ``x86-64`` scalar). The kernel's results are not bit-identical to GGML's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

SRC = Path(__file__).resolve().parents[4] / "kernels/jit/csrc/nvfp4/optimized"
UPSTREAM = SRC.parent / "upstream"
CXX_FLAGS = ["-std=c++20", "-Ofast", "-fPIC", "-pthread", "-fopenmp"]
C_FLAGS = ["-x", "c", "-std=c11", "-Ofast", "-fPIC"]


def arch_flags(cxx: str, march: str | None = None) -> list[str]:
    """``-march=<march>``, else the JIT's resolved host -march (toolchain.host_arch_flags, which ``cxx`` resolves)."""
    if march:
        return [f"-march={march}"]
    from sglang.kernels.jit.utils.compile.toolchain import host_arch_flags

    saved = os.environ.get("CXX")
    os.environ["CXX"] = cxx
    try:
        return host_arch_flags()
    finally:
        if saved is None:
            os.environ.pop("CXX")
        else:
            os.environ["CXX"] = saved


def build(
    output: Path,
    *,
    cxx: str,
    main: Path,
    extra_flags: Sequence[str] = (),
    march: str | None = None,
) -> Path:
    """Compile the kernel and the harness source ``main`` into the executable ``output``. ``march`` overrides the
    resolved -march (e.g. ``x86-64-v3``)."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common = arch_flags(cxx, march) + list(extra_flags)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        c_object = Path(tmp) / "nvfp4.o"
        subprocess.run([cxx, *C_FLAGS, *common, "-c", str(UPSTREAM / "nvfp4.c"), "-o", str(c_object)], check=True)
        sources = [str(SRC / "kernel.cpp"), str(Path(main).resolve())]
        subprocess.run(
            [cxx, *CXX_FLAGS, *common, "-I", str(SRC), *sources, str(c_object), "-o", str(output)], check=True
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination; keep it outside the source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--main", type=Path, required=True, help="the harness source to link with the kernel")
    parser.add_argument("--march", help="build for this -march instead of the resolved one (e.g. x86-64-v3)")
    args = parser.parse_args()
    print(build(args.output, cxx=args.cxx, main=args.main, march=args.march))


if __name__ == "__main__":
    main()
