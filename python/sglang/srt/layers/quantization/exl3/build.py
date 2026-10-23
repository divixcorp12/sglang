"""Build the EXL3 CPU kernel as a standalone shared library: no torch, no EXL3 CUDA extension, no tvm-ffi.

The server compiles the same sources into the expert-stream host module (``kernels/ops/moe/cpu_kernel_build.py``);
this is for harnesses that load the kernel themselves. The ISA tier is the build's -march
(``SGLANG_JIT_HOST_MARCH``, else this machine's)."""
from pathlib import Path
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Destination .so; kept outside the source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    version = subprocess.check_output([args.cxx, "-dumpfullversion", "-dumpversion"], text=True).strip()
    identity = subprocess.check_output([args.cxx, "--version"], text=True)
    if version.split(".", 1)[0] != "15" or "clang" in identity.lower():
        parser.error(f"Use the validated GCC 15 compiler, found {version}")

    src = Path(__file__).resolve().parents[4] / "kernels/jit/csrc/exl3/optimized"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from sglang.kernels.jit.utils.compile.toolchain import host_arch_flags

    os.environ["CXX"] = args.cxx  # host_arch_flags resolves -march with $CXX
    cmd = [args.cxx, "-Ofast", *host_arch_flags(), "-std=c++20", "-fPIC", "-shared", "-pthread", "-fopenmp",
           "-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128",
           "-I" + str(src), str(src / "kernel.cpp"), "-o", str(output)]
    subprocess.run(cmd, check=True)
    print(output)


if __name__ == "__main__":
    main()
