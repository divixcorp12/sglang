"""The header-only CPU experts framework (host/cpu_experts) passes its native harness (Linux, GCC with OpenMP).

``cpu_experts_common_check.cpp`` drives a toy quant through ``ExpertForward`` (a kernel type, binding.hpp) and prints
``ok <check>`` per contract it holds.
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
    "routes_are_validated",
    "forwards_run_at_once",
    "isa_is_the_build_tier_lowered_to_the_quants_top",
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


@pytest.mark.parametrize(
    "flags",
    # -Ofast as the kernels build, where a std::isfinite weight check would fold away.
    [["-Ofast", "-march=x86-64"], ["-Ofast", "-march=native"], ["-march=x86-64", *SANITIZERS]],
    ids=["scalar-tier", "host-tier", "asan-ubsan"],
)
def test_the_native_harness_passes(tmp_path, flags):
    if SANITIZERS[0] in flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = tmp_path / "cpu_experts_common_check"
    subprocess.run([CXX, *CXX_FLAGS, *flags, str(HARNESS), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = set(result.stdout.split("\n"))
    missing = [name for name in CHECKS if f"ok {name}" not in passed]
    assert not missing, f"checks that did not report ok: {missing}\n{result.stdout}{result.stderr}"


def _build_toy_library(output: Path, *flags: str) -> Path:
    subprocess.run([CXX, *CXX_FLAGS, "-fPIC", "-shared", *flags, str(TOY_LIBRARY), "-o", str(output)], check=True)
    return output


def _disassembly(library: Path) -> str:
    return subprocess.run(["objdump", "-d", "-C", str(library)], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize(
    "march, loops",
    [("x86-64", set()), ("x86-64-v3", {"avx2"}), ("skylake-avx512", {"bw"})],
    ids=["scalar", "avx2", "bw"],
)
def test_the_keep_warm_loop_compiled_is_the_build_tiers(tmp_path, march, loops):
    # The tier is the build's -march: a library holds only its tier's loop, so a baseline x86-64 build carries no AVX.
    library = _build_toy_library(tmp_path / "libtoy.so", f"-march={march}", "-DTOY_TOP_ISA=Vbmi")
    text = _disassembly(library)
    assert {name for name in ("avx2", "bw") if f"keep_warm_detail::{name}(" in text} == loops
    if march == "x86-64":
        assert "%ymm" not in text and "%zmm" not in text


def test_a_quants_top_tier_lowers_the_keep_warm_loop(tmp_path):
    library = _build_toy_library(tmp_path / "libtoy.so", "-march=skylake-avx512", "-DTOY_TOP_ISA=Avx2")
    text = _disassembly(library)
    assert "keep_warm_detail::avx2(" in text and "keep_warm_detail::bw(" not in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
