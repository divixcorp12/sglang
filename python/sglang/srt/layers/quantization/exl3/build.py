"""Build the CPU library without compiling EXL3's CUDA extension."""
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
    import torch

    src = Path(__file__).resolve().parents[4] / "kernels/jit/csrc/exl3/optimized"
    torch_root = Path(torch.__file__).resolve().parent
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.cxx, "-Ofast", "-march=native", "-std=c++20", "-fPIC", "-shared", "-pthread", "-fopenmp",
           "-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128",
           f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
           "-I" + str(src), "-isystem", str(torch_root / "include"),
           "-isystem", str(torch_root / "include/torch/csrc/api/include"), str(src / "kernel.cpp"),
           "-L" + str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
           "-ltorch_cpu", "-lc10", "-o", str(output)]
    subprocess.run(cmd, check=True)
    print(output)


if __name__ == "__main__":
    main()
