"""The NVFP4 CPU expert library builds from Python and passes its native harnesses (Linux, GCC with OpenMP).

``nvfp4_cpu/optimized/build.py`` compiles the kernel into a shared library or, with a harness ``main``, into an
executable; ``nvfp4_cpu_ext.nvfp4_cpu_library`` builds the library once per content hash and loads it. The two native
harnesses are the ones the removed CMake build ran under CTest.
"""

import ctypes
import importlib.util
import os
import re
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

    # The report prints when the tier is detected, before dispatch reads it: the dumps prove dispatch ran that tier. The
    # AVX2 tier's FMA accumulation rounds differently from the scalar loop, so on an AVX2 host the uncapped dump differs.
    env = {k: v for k, v in os.environ.items() if k != "NVFP4_CPU_MAX_ISA"}
    uncapped = subprocess.run(
        [str(exe), str(tmp_path / "default.bin"), str(cores[0])],
        env={**env, "NVFP4_CPU_REPORT_ISA": "1"},
        capture_output=True, text=True, timeout=900,
    )
    assert uncapped.returncode == 0, uncapped.stdout + uncapped.stderr
    if "nvfp4 isa scalar" in uncapped.stderr:
        pytest.skip("this host runs the scalar tier uncapped, so there is no second tier to tell apart")
    assert "nvfp4 isa avx2" in uncapped.stderr, uncapped.stderr
    assert (tmp_path / "default.bin").read_bytes() != (tmp_path / "scalar.bin").read_bytes()


# Functions allowed to hold AVX registers: the AVX2 tier's dot product and its helpers, the plan's AVX2 entries (which
# inline them), and the framework's AVX2 keep-warm loop. Matched on demangled names, with any GCC clone suffix.
AVX2_TIER = [
    re.compile(r"DotRows<\(sglang::cpu_experts::\(anonymous namespace\)::Isa\)1>::rows<"),
    re.compile(r"::avx2_detail::"),
    re.compile(r"ForwardPlan<[^>]*Isa\)1>::(gate_up|down)_avx2\("),
    re.compile(r"::keep_warm_detail::avx2\("),
]


def _functions_using(disassembly: str, registers=("%ymm", "%zmm")) -> set[str]:
    found, current = set(), None
    for line in disassembly.splitlines():
        header = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if header:
            current = header.group(1)
        elif current and any(r in line for r in registers):
            found.add(current)
    return found


def test_only_the_avx2_tier_uses_avx_registers(tmp_path):
    # The library is built for baseline x86-64 and must run on a host without AVX: vector registers may appear only in
    # the AVX2 tier's functions, which run only after detection chose that tier.
    library = _build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)
    disassembly = subprocess.run(
        ["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], capture_output=True, text=True, check=True
    ).stdout
    used = _functions_using(disassembly)
    assert used, "the AVX2 tier is missing from the library"
    outside = sorted(f for f in used if not any(p.search(f) for p in AVX2_TIER))
    assert not outside, outside
    assert not _functions_using(disassembly, ("%zmm",)), "NVFP4 has no AVX-512 tier"


SANITIZERS = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]


def _links_sanitizers(tmp_path) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    probe = subprocess.run([CXX, *SANITIZERS, str(source), "-o", str(tmp_path / "probe")], capture_output=True)
    return probe.returncode == 0


# The sanitizer harness forwards through the library, so it runs at each tier (cap None: the host's); the GGML check
# calls both tiers' dot products itself.
@pytest.mark.parametrize(
    "harness, flags, cap",
    [
        ("nvfp4_cpu_sanitizer.cpp", [], None),
        ("nvfp4_cpu_sanitizer.cpp", [], "scalar"),
        ("nvfp4_cpu_sanitizer.cpp", SANITIZERS, None),
        ("nvfp4_cpu_sanitizer.cpp", SANITIZERS, "scalar"),
        ("nvfp4_cpu_ggml_check.cpp", [], None),
    ],
    ids=[
        "sanitizer-harness",
        "sanitizer-harness-scalar",
        "sanitizer-harness-asan-ubsan",
        "sanitizer-harness-asan-ubsan-scalar",
        "ggml-check",
    ],
)
def test_the_native_harness_passes(tmp_path, harness, flags, cap):
    if flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = _build_module().build(
        tmp_path / Path(harness).stem, cxx=CXX, main=REPO / "test/registered/unit/kernels" / harness, extra_flags=flags
    )
    env = {k: v for k, v in os.environ.items() if k != "NVFP4_CPU_MAX_ISA"}
    if cap:
        env["NVFP4_CPU_MAX_ISA"] = cap
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
