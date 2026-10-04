"""The header-only CPU experts framework (cpu_experts_common) passes its native harness (Linux, GCC with OpenMP).

``cpu_experts_common_check.cpp`` drives a toy quant through ``ExpertForward`` and the C ABI macro and prints
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
CXX = os.environ.get("CXX") or shutil.which("g++")
CXX_FLAGS = ["-std=c++20", "-O2", "-fopenmp", "-pthread"]
SANITIZERS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g"]
CHECKS = (
    "register_then_forward_overwrites_and_accumulates",
    "abi_versions_are_checked",
    "slot_bytes_below_the_minimum_are_refused",
    "unknown_handle_is_refused",
    "routes_are_validated",
    "concurrent_forward_returns_3",
    "isa_cap_env_lowers_the_tier",
    "set_cores_after_the_first_forward_returns_2",
    "keep_warm_returns_when_the_word_moves",
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
    # The harness sets TOY_CPU_MAX_ISA itself; a value inherited from the caller must not pre-empt it.
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOY_CPU_")}
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = set(result.stdout.split("\n"))
    missing = [name for name in CHECKS if f"ok {name}" not in passed]
    assert not missing, f"checks that did not report ok: {missing}\n{result.stdout}{result.stderr}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
