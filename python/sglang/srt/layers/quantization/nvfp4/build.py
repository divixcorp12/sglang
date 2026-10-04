"""Build the NVFP4 CPU expert library (or a native harness linked with it) without CMake.

The kernel is optimized/kernel.cpp plus the vendored GGML C subset (../upstream/nvfp4.c, compiled as C: it relies on
C's implicit void* conversions). -ffp-contract=off is part of the arithmetic contract: never build it with -Ofast.
One build holds every ISA tier (scalar, and AVX2 by function attribute) and picks one at run time, so it runs on any
x86-64 host; NVFP4_CPU_MAX_ISA=scalar caps the tier and NVFP4_CPU_REPORT_ISA=1 prints the one chosen.
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
CXX_FLAGS = ["-std=c++20", "-O3", "-ffp-contract=off", "-fPIC", "-pthread", "-fopenmp"]
C_FLAGS = ["-x", "c", "-std=c11", "-O3", "-ffp-contract=off", "-fPIC"]
# Baseline x86-64, pinned: a toolchain's default -march may include AVX2 (RHEL 10's GCC defaults to x86-64-v3), which
# would compile the scalar tier and the rest of the library for AVX2 too.
ARCH_FLAGS = ["-march=x86-64", "-mtune=generic"]


def build(
    output: Path,
    *,
    cxx: str,
    main: Path | None = None,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Compile the kernel into ``output``: a shared library, or with ``main``, that harness's executable."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common = ARCH_FLAGS + list(extra_flags)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        c_object = Path(tmp) / "nvfp4.o"
        subprocess.run([cxx, *C_FLAGS, *common, "-c", str(UPSTREAM / "nvfp4.c"), "-o", str(c_object)], check=True)
        sources = [str(SRC / "kernel.cpp")] + ([str(Path(main).resolve())] if main else [])
        link = [] if main else ["-shared"]
        subprocess.run(
            [cxx, *CXX_FLAGS, *common, "-I", str(SRC), *sources, str(c_object), *link, "-o", str(output)], check=True
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination; keep it outside the source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--main", type=Path, help="link this harness source into an executable instead")
    args = parser.parse_args()
    print(
        build(
            args.output,
            cxx=args.cxx,
            main=args.main,
        )
    )


if __name__ == "__main__":
    main()
