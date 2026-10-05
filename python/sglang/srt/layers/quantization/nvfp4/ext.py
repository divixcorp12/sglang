"""The NVFP4 CPU expert library, built on first use by quantization/nvfp4/build.py and loaded with tvm-ffi (its one
export hands out the kernel's address).

The library is cached under ``build_dir`` by a hash of its sources, the build flags and the compiler's version, so an
edit or a compiler change builds a new one; a file lock keeps concurrent processes from building the same one twice.
The build targets baseline x86-64 and picks its ISA tier at run time, so the host CPU is not part of the key. The
compiler is $CXX, else g++.
"""

import functools
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Optional

from filelock import FileLock

_BUILD = Path(__file__).resolve().parent / "build.py"
_KERNEL = Path(__file__).resolve().parents[4] / "kernels/jit/csrc/nvfp4"
# quant.hpp includes the shared CPU experts framework (kernel.hpp among it), so the library depends on it too.
_COMMON = _KERNEL.parent / "moe/expert_stream/host/cpu_experts"
_DEFAULT_BUILD_DIR = "~/.cache/sglang/nvfp4_cpu"


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", _BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources() -> list[Path]:
    optimized, upstream = _KERNEL / "optimized", _KERNEL / "upstream"
    return sorted(
        [*optimized.glob("*.cpp"), *optimized.glob("*.hpp"), *optimized.glob("*.h"), _BUILD]
        + [*upstream.glob("*.c"), *upstream.glob("*.h"), *_COMMON.glob("*.hpp")]
    )


def library_path(build_dir: Path, cxx: str) -> Path:
    """Where the library for the current sources, flags and compiler lives."""
    import tvm_ffi

    build = _build_module()
    digest = hashlib.sha256()
    digest.update(tvm_ffi.__version__.encode())
    digest.update(subprocess.check_output([cxx, "--version"]))
    digest.update(" ".join(build.CXX_FLAGS + build.C_FLAGS + build.ARCH_FLAGS).encode())
    for path in _sources():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return build_dir / f"libsglang_nvfp4_cpu_{digest.hexdigest()[:16]}.so"


def nvfp4_cpu_library_path(build_dir: Optional[str] = None) -> Path:
    """The built library's path (built here first if needed)."""
    cxx = os.environ.get("CXX", "g++")
    root = Path(os.path.expanduser(build_dir or _DEFAULT_BUILD_DIR))
    root.mkdir(parents=True, exist_ok=True)
    path = library_path(root, cxx)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.so")
            _build_module().build(partial, cxx=cxx)
            os.replace(partial, path)
    return path


@functools.cache
def nvfp4_cpu_module(build_dir: Optional[str] = None):
    """The library as a tvm-ffi module. Cached for the process: a host holds its kernel's address, so the module is
    never unloaded."""
    from tvm_ffi import load_module

    return load_module(str(nvfp4_cpu_library_path(build_dir)))


def nvfp4_cpu_kernel_address(build_dir: Optional[str] = None) -> int:
    """The address of the library's CpuExpertKernel, for ExpertStreamHost.enable_cpu_experts."""
    return int(nvfp4_cpu_module(build_dir).nvfp4_cpu_kernel_address())
