"""The header-only CPU experts framework (host/cpu_experts) passes its native harness (Linux, GCC with OpenMP).

``cpu_experts_common_check.cpp`` drives a toy quant through ``ExpertForward`` as a ``CpuExpertKernel`` and prints
``ok <check>`` per contract it holds, and ``cpu_experts_cross_library_check.cpp`` links two toy libraries and passes a
layer between them.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

HARNESS = Path(__file__).resolve().parent / "cpu_experts_common_check.cpp"
TOY_LIBRARY = Path(__file__).resolve().parent / "cpu_experts_common_toy_lib.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")
CXX_FLAGS = ["-std=c++20", "-O2", "-fopenmp", "-pthread"]
SANITIZERS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g"]
CHECKS = (
    "make_layer_then_forward_overwrites_and_accumulates",
    "params_and_slabs_are_validated",
    "a_layer_of_another_kernel_is_refused",
    "routes_are_validated",
    "forwards_run_at_once",
    "isa_cap_env_lowers_the_tier",
    "cores_are_validated_and_bound_the_team",
    "each_calls_team_runs_on_its_own_cores",
    "a_failed_pin_throws_runtime_error_and_leaves_out_untouched",
    "warm_returns_when_the_word_moves",
    "a_team_barrier_orders_its_phases",
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the framework needs Linux and a GCC with OpenMP"
)


def _links_sanitizers(tmp_path) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    probe = subprocess.run([CXX, *SANITIZERS, str(source), "-o", str(tmp_path / "probe")], capture_output=True)
    return probe.returncode == 0


@pytest.mark.parametrize("flags", [[], SANITIZERS], ids=["plain", "asan-ubsan"])
def test_the_native_harness_passes(tmp_path, flags):
    if flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = tmp_path / "cpu_experts_common_check"
    subprocess.run([CXX, *CXX_FLAGS, *flags, str(HARNESS), "-o", str(exe)], check=True)
    # The harness sets TOY_CPU_MAX_ISA and TOY_CPU_REPORT_ISA itself; a value inherited from the caller must not
    # pre-empt them.
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOY_CPU_")}
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = set(result.stdout.split("\n"))
    missing = [name for name in CHECKS if f"ok {name}" not in passed]
    assert not missing, f"checks that did not report ok: {missing}\n{result.stdout}{result.stderr}"
    # The harness sets TOY_CPU_REPORT_ISA=1: isa() reports the tier it settled on, once.
    assert result.stderr.count("toy isa ") == 1 and "toy isa scalar\n" in result.stderr, result.stderr


def _build_toy_library(output: Path, *flags: str) -> Path:
    subprocess.run([CXX, *CXX_FLAGS, "-fPIC", "-shared", *flags, str(TOY_LIBRARY), "-o", str(output)], check=True)
    return output


CROSS = Path(__file__).resolve().parent / "cpu_experts_cross_library_check.cpp"
# As the expert-stream host module builds (expert_stream_transport._host_module_cached): the interface must cross a
# hidden-visibility library.
HIDDEN = ["-fvisibility=hidden", "-fvisibility-inlines-hidden"]


def test_two_libraries_refuse_each_others_layers_across_the_so_boundary(tmp_path):
    for name in ("a", "b"):
        _build_toy_library(
            tmp_path / f"libtoy_{name}.so", *HIDDEN, f"-DTOY_KERNEL=toy_kernel_{name}", f'-DTOY_NAME="toy_{name}"'
        )
    exe = tmp_path / "cross_library"
    subprocess.run(
        [CXX, *CXX_FLAGS, str(CROSS), f"-L{tmp_path}", "-ltoy_a", "-ltoy_b", f"-Wl,-rpath,{tmp_path}", "-o", str(exe)],
        check=True,
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "ok cross_library" in result.stdout, result.stdout + result.stderr


def test_a_scalar_quant_library_uses_no_avx_registers(tmp_path):
    # A quant whose top tier is Scalar must compile no vector code from the framework, so a portable (baseline x86-64)
    # build stays free of AVX registers; the Avx2 build is the control that the scan finds them.
    portable = ["-march=x86-64", "-mtune=generic"]
    scalar = _build_toy_library(tmp_path / "libtoy_scalar.so", *portable, "-DTOY_TOP_ISA=Scalar")
    avx2 = _build_toy_library(tmp_path / "libtoy_avx2.so", *portable)

    def disassembly(library: Path) -> str:
        return subprocess.run(["objdump", "-d", str(library)], capture_output=True, text=True, check=True).stdout

    assert "%ymm" not in disassembly(scalar) and "%zmm" not in disassembly(scalar)
    assert "%ymm" in disassembly(avx2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
