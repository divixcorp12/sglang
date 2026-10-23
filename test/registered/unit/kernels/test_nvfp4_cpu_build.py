"""The NVFP4 CPU expert library builds from Python and passes its native harnesses (Linux, GCC with OpenMP).

``quantization/nvfp4/build.py`` compiles the kernel into a shared library or, with a harness ``main``, into an
executable; ``nvfp4_cpu_ext.nvfp4_cpu_library_path`` builds the library once per content hash (``nvfp4_cpu_module``
loads it). The two native harnesses are the ones the removed CMake build ran under CTest.
"""

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
BUILD = REPO / "python/sglang/srt/layers/quantization/nvfp4/build.py"
CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_py_makes_a_library_exporting_no_c_abi(tmp_path):
    library = _build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)
    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(library)], capture_output=True, text=True, check=True
    ).stdout
    assert "sglang_nvfp4_cpu_experts_" not in symbols
    assert "__tvm_ffi_nvfp4_cpu_kernel_address" in symbols  # tvm-ffi's TVM_FFI_DLL_EXPORT_TYPED_FUNC prefix
    assert "__tvm_ffi_nvfp4_cpu_params" in symbols


def test_the_loader_builds_once_per_content_and_reuses_the_library(tmp_path):
    from sglang.srt.layers.quantization.nvfp4 import ext as nvfp4_cpu_ext

    nvfp4_cpu_ext.nvfp4_cpu_module.cache_clear()
    path = nvfp4_cpu_ext.nvfp4_cpu_library_path(str(tmp_path))
    built = list(tmp_path.glob("*.so"))
    assert len(built) == 1 and Path(path) == built[0]
    stamp = built[0].stat().st_mtime_ns
    nvfp4_cpu_ext.nvfp4_cpu_module.cache_clear()
    nvfp4_cpu_ext.nvfp4_cpu_library_path(str(tmp_path))
    assert [p.stat().st_mtime_ns for p in tmp_path.glob("*.so")] == [stamp]


def test_the_library_hands_out_its_kernel_address(tmp_path):
    from tvm_ffi import load_module

    module = load_module(str(_build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)))
    address = int(module.nvfp4_cpu_kernel_address())
    assert address != 0 and int(module.nvfp4_cpu_kernel_address()) == address


def test_the_library_packs_the_params_make_layer_reads(tmp_path):
    """nvfp4_cpu_params is SglangNvfp4CpuParams {int32 w13_layout, float inv_input_scale13, float inv_input_scale2}."""
    import struct

    from tvm_ffi import load_module

    module = load_module(str(_build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)))
    assert bytes(module.nvfp4_cpu_params(2, 0.5, 1.0)) == struct.pack("<iff", 2, 0.5, 1.0)


def test_the_module_is_loaded_once_and_kept(tmp_path):
    """Review Focus 1: the host holds the kernel's address, so the module that owns it must outlive every host: the
    loader is cached for the process and hands out one module."""
    from sglang.srt.layers.quantization.nvfp4 import ext as nvfp4_cpu_ext

    nvfp4_cpu_ext.nvfp4_cpu_module.cache_clear()
    first = nvfp4_cpu_ext.nvfp4_cpu_module(str(tmp_path))
    assert nvfp4_cpu_ext.nvfp4_cpu_module(str(tmp_path)) is first


def test_the_march_picks_the_tier(tmp_path):
    """A build at x86-64 holds the scalar tier and one at x86-64-v3 the AVX2 tier: the AVX2 tier's FMA accumulation
    rounds differently from the scalar loop, so the two builds' forwards dump different bytes."""
    if not _host_has_avx2():
        pytest.skip("the host cannot run an x86-64-v3 build")
    harness = REPO / "test/manual/dsv41/nvfp4_cpu_forward_ab.cpp"
    core = str(sorted(os.sched_getaffinity(0))[0])
    dumps = {}
    for march in ("x86-64", "x86-64-v3"):
        exe = _build_module().build(tmp_path / f"ab-{march}", cxx=CXX, main=harness, march=march)
        result = subprocess.run(
            [str(exe), str(tmp_path / f"{march}.bin"), core], capture_output=True, text=True, timeout=900
        )
        assert result.returncode == 0, result.stdout + result.stderr
        dumps[march] = (tmp_path / f"{march}.bin").read_bytes()
    assert dumps["x86-64"] != dumps["x86-64-v3"]


def _host_has_avx2() -> bool:
    flags = next((l.split(":", 1)[1].split() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("flags")), [])
    return {"avx2", "fma", "f16c", "bmi1", "bmi2", "abm", "movbe"} <= set(flags)


def _disassembly(library: Path) -> str:
    return subprocess.run(
        ["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], capture_output=True, text=True, check=True
    ).stdout


def test_the_library_holds_only_its_march_tier(tmp_path):
    # The tier is compile-time: a build at x86-64 must run on a host without AVX, so it holds no vector register at
    # all; a build at x86-64-v3 holds the AVX2 tier and no AVX-512 register (NVFP4 has no AVX-512 tier).
    scalar = _disassembly(_build_module().build(tmp_path / "libscalar.so", cxx=CXX, march="x86-64"))
    assert "%ymm" not in scalar and "%zmm" not in scalar
    avx2 = _disassembly(_build_module().build(tmp_path / "libavx2.so", cxx=CXX, march="x86-64-v3"))
    assert "%ymm" in avx2, "the AVX2 tier is missing from the x86-64-v3 library"
    assert "%zmm" not in avx2


SANITIZERS = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]


def _links_sanitizers(tmp_path) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    probe = subprocess.run([CXX, *SANITIZERS, str(source), "-o", str(tmp_path / "probe")], capture_output=True)
    return probe.returncode == 0


# Each harness runs at the host's tier (march None: the resolved -march) and at the scalar tier (x86-64); the GGML
# check calls both tiers' dot products itself where its build has them.
@pytest.mark.parametrize(
    "harness, flags, march",
    [
        ("nvfp4_cpu_sanitizer.cpp", [], None),
        ("nvfp4_cpu_sanitizer.cpp", [], "x86-64"),
        ("nvfp4_cpu_sanitizer.cpp", SANITIZERS, None),
        ("nvfp4_cpu_sanitizer.cpp", SANITIZERS, "x86-64"),
        ("nvfp4_cpu_ggml_check.cpp", [], None),
        ("nvfp4_cpu_ggml_check.cpp", [], "x86-64"),
    ],
    ids=[
        "sanitizer-harness",
        "sanitizer-harness-scalar",
        "sanitizer-harness-asan-ubsan",
        "sanitizer-harness-asan-ubsan-scalar",
        "ggml-check",
        "ggml-check-scalar",
    ],
)
def test_the_native_harness_passes(tmp_path, harness, flags, march):
    if flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = _build_module().build(
        tmp_path / Path(harness).stem,
        cxx=CXX,
        main=REPO / "test/registered/unit/kernels" / harness,
        extra_flags=flags,
        march=march,
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
