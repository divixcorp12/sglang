"""Build the NVFP4 CPU expert library (or a native harness linked with it) without CMake.

The kernel is optimized/moe_mul1.cpp plus the vendored GGML C subset (../upstream/nvfp4.c, compiled as C: it relies on
C's implicit void* conversions). -ffp-contract=off is part of the arithmetic contract: never build it with -Ofast.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

SRC = Path(__file__).resolve().parent
UPSTREAM = SRC.parent / "upstream"
CXX_FLAGS = ["-std=c++20", "-O3", "-ffp-contract=off", "-fPIC", "-pthread", "-fopenmp"]
C_FLAGS = ["-x", "c", "-std=c11", "-O3", "-ffp-contract=off", "-fPIC"]


def build(
    output: Path,
    *,
    cxx: str,
    native: bool = True,
    upstream_baseline: bool = False,
    main: Path | None = None,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Compile the kernel into ``output``: a shared library, or with ``main``, that harness's executable.

    ``native`` adds -march=native (the AVX2 dot product on an AVX2 host). Otherwise the build pins baseline x86-64
    (the scalar loop): a toolchain's default -march may include AVX2 (RHEL 10's GCC defaults to x86-64-v3).
    ``upstream_baseline`` builds the bench's baseline: each row converted to GGML blocks before GGML's own dot product.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common = (["-march=native"] if native else ["-march=x86-64", "-mtune=generic"]) + list(extra_flags)
    if upstream_baseline:
        common.append("-DNVFP4_CPU_UPSTREAM_BASELINE=1")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        c_object = Path(tmp) / "nvfp4.o"
        subprocess.run([cxx, *C_FLAGS, *common, "-c", str(UPSTREAM / "nvfp4.c"), "-o", str(c_object)], check=True)
        sources = [str(SRC / "moe_mul1.cpp")] + ([str(Path(main).resolve())] if main else [])
        link = [] if main else ["-shared"]
        subprocess.run(
            [cxx, *CXX_FLAGS, *common, "-I", str(SRC), *sources, str(c_object), *link, "-o", str(output)], check=True
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination; keep it outside the source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--portable", action="store_true", help="baseline x86-64 instead of -march=native (the scalar dot product)")
    parser.add_argument("--upstream-baseline", action="store_true", help="the bench's GGML-conversion baseline")
    parser.add_argument("--main", type=Path, help="link this harness source into an executable instead")
    args = parser.parse_args()
    print(
        build(
            args.output,
            cxx=args.cxx,
            native=not args.portable,
            upstream_baseline=args.upstream_baseline,
            main=args.main,
        )
    )


if __name__ == "__main__":
    main()
