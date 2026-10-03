"""Build the standalone CpuExpertForward plugin; no CUDA or torch needed."""

import argparse
import os
from pathlib import Path
import subprocess


def build(output, cxx=None, *, native=False, sanitize=False):
    src = Path(__file__).resolve().parent
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        cxx or os.environ.get("CXX", "c++"),
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-pthread",
        "-ffp-contract=off",
    ]
    if native:
        command += ["-march=native"]
    if sanitize:
        command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    subprocess.run(command + [str(src / "moe_mul1.cpp"), "-o", str(output)], check=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument(
        "--native", action="store_true", help="Enable this CPU's ISA, including AVX2"
    )
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    print(build(args.output, args.cxx, native=args.native, sanitize=args.sanitize))
