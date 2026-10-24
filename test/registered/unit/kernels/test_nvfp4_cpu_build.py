"""The NVFP4 CPU expert kernel builds into native harness executables and passes them (Linux, GCC with OpenMP).

``quantization/nvfp4/build.py`` compiles the kernel and a harness ``main`` into an executable (the server builds the
same sources into the expert-stream host module, ``kernels/ops/moe/cpu_kernel_build.py``). The two native harnesses are
the ones the removed CMake build ran under CTest.
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


def _disassembly(binary: Path) -> str:
    return subprocess.run(
        ["objdump", "-d", "-C", "--no-show-raw-insn", str(binary)], capture_output=True, text=True, check=True
    ).stdout


def test_the_executable_holds_only_its_march_tier(tmp_path):
    # The tier is compile-time: a build at x86-64 must run on a host without AVX, so it holds no vector register at
    # all; a build at x86-64-v3 holds the AVX2 tier and no AVX-512 register (NVFP4 has no AVX-512 tier).
    harness = REPO / "test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp"
    scalar = _disassembly(_build_module().build(tmp_path / "scalar", cxx=CXX, main=harness, march="x86-64"))
    assert "%ymm" not in scalar and "%zmm" not in scalar
    avx2 = _disassembly(_build_module().build(tmp_path / "avx2", cxx=CXX, main=harness, march="x86-64-v3"))
    assert "%ymm" in avx2, "the AVX2 tier is missing from the x86-64-v3 build"
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
