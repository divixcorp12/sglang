"""The NVFP4 CPU expert library, built on first use by nvfp4_cpu/optimized/build.py.

The library is cached under ``build_dir`` by a hash of its sources, the build flags, the compiler's version and what
-march=native means on this host, so an edit, a compiler change or a different CPU builds a new one; a file lock keeps
concurrent processes from building the same one twice. The compiler is $CXX, else g++.
"""

import ctypes
import functools
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Optional

from filelock import FileLock

_KERNEL = Path(__file__).resolve().parent / "nvfp4_cpu"
# cpu_experts_cabi.h includes the engine's forward ABI header, so the library depends on it too.
_FORWARD_ABI = _KERNEL.parents[3] / "kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"
_DEFAULT_BUILD_DIR = "~/.cache/sglang/nvfp4_cpu"


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", _KERNEL / "optimized" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources() -> list[Path]:
    optimized, upstream = _KERNEL / "optimized", _KERNEL / "upstream"
    return sorted(
        [*optimized.glob("*.cpp"), *optimized.glob("*.hpp"), *optimized.glob("*.h"), *optimized.glob("*.py")]
        + [*upstream.glob("*.c"), *upstream.glob("*.h"), _FORWARD_ABI]
    )


def library_path(build_dir: Path, cxx: str) -> Path:
    """Where the library for the current sources, flags, compiler and host CPU lives."""
    build = _build_module()
    digest = hashlib.sha256()
    digest.update(subprocess.check_output([cxx, "--version"]))
    digest.update(subprocess.check_output([cxx, "-march=native", "-Q", "--help=target"]))
    digest.update(" ".join(build.CXX_FLAGS + build.C_FLAGS).encode())
    for path in _sources():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return build_dir / f"libsglang_nvfp4_cpu_{digest.hexdigest()[:16]}.so"


@functools.cache
def nvfp4_cpu_library(build_dir: Optional[str] = None) -> ctypes.CDLL:
    """The native library (built here first if needed), exporting the cpu_experts_cabi.h functions."""
    cxx = os.environ.get("CXX", "g++")
    root = Path(os.path.expanduser(build_dir or _DEFAULT_BUILD_DIR))
    root.mkdir(parents=True, exist_ok=True)
    path = library_path(root, cxx)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.so")
            _build_module().build(partial, cxx=cxx)
            os.replace(partial, path)
    return ctypes.CDLL(str(path))
