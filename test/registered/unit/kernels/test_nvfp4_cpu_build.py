"""The NVFP4 CPU expert library builds from Python and passes its native harnesses (Linux, GCC with OpenMP).

``nvfp4_cpu/optimized/build.py`` compiles the kernel into a shared library or, with a harness ``main``, into an
executable; ``nvfp4_cpu_ext.nvfp4_cpu_library`` builds the library once per content hash and loads it. The two native
harnesses are the ones the removed CMake build ran under CTest.
"""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
OPTIMIZED = REPO / "python/sglang/srt/layers/quantization/nvfp4_cpu/optimized"
CXX = os.environ.get("CXX") or shutil.which("g++")
C_ABI = (
    "sglang_nvfp4_cpu_experts_register_layer",
    "sglang_nvfp4_cpu_experts_free_layer",
    "sglang_nvfp4_cpu_experts_forward",
    "sglang_nvfp4_cpu_experts_keep_warm",
    "sglang_nvfp4_cpu_experts_set_cores",
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", OPTIMIZED / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_py_makes_a_library_exporting_the_c_abi(tmp_path):
    library = ctypes.CDLL(str(_build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)))
    for name in C_ABI:
        getattr(library, name)


def test_the_loader_builds_once_per_content_and_reuses_the_library(tmp_path):
    from sglang.srt.layers.quantization import nvfp4_cpu_ext

    nvfp4_cpu_ext.nvfp4_cpu_library.cache_clear()
    library = nvfp4_cpu_ext.nvfp4_cpu_library(str(tmp_path))
    built = list(tmp_path.glob("*.so"))
    assert len(built) == 1
    stamp = built[0].stat().st_mtime_ns
    nvfp4_cpu_ext.nvfp4_cpu_library.cache_clear()
    nvfp4_cpu_ext.nvfp4_cpu_library(str(tmp_path))
    assert [p.stat().st_mtime_ns for p in tmp_path.glob("*.so")] == [stamp]
    for name in C_ABI:
        getattr(library, name)


def test_the_scalar_cap_runs_the_scalar_tier(tmp_path):
    """NVFP4_CPU_MAX_ISA=scalar must select the scalar tier in a library that also holds the AVX2 one."""
    exe = _build_module().build(
        tmp_path / "ab", cxx=CXX, main=REPO / "test/manual/dsv41/nvfp4_cpu_forward_ab.cpp"
    )
    cores = sorted(os.sched_getaffinity(0))[:1]
    capped = subprocess.run(
        [str(exe), str(tmp_path / "scalar.bin"), str(cores[0])],
        env={**os.environ, "NVFP4_CPU_MAX_ISA": "scalar", "NVFP4_CPU_REPORT_ISA": "1"},
        capture_output=True, text=True, timeout=900,
    )
    assert capped.returncode == 0, capped.stdout + capped.stderr
    assert "nvfp4 isa scalar" in capped.stderr


SANITIZERS = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]


def _links_sanitizers(tmp_path) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    probe = subprocess.run([CXX, *SANITIZERS, str(source), "-o", str(tmp_path / "probe")], capture_output=True)
    return probe.returncode == 0


@pytest.mark.parametrize(
    "harness, flags",
    [
        ("nvfp4_cpu_sanitizer.cpp", []),
        ("nvfp4_cpu_sanitizer.cpp", SANITIZERS),
        ("nvfp4_cpu_ggml_check.cpp", []),
    ],
    ids=["sanitizer-harness", "sanitizer-harness-asan-ubsan", "ggml-check"],
)
def test_the_native_harness_passes(tmp_path, harness, flags):
    if flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = _build_module().build(
        tmp_path / Path(harness).stem, cxx=CXX, main=REPO / "test/registered/unit/kernels" / harness, extra_flags=flags
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
