"""The EXL3 CPU kernel's job interface (exl3/optimized/entry.h) gives the same bits however its tasks are split (Linux, GCC, x86).

``exl3_cpu_job_check.cpp`` runs random generic-plan calls (overwrite, accumulate and overwrite-by-store, staged and not)
through a one-worker JobRunner and a several-worker JobRunner (dynamic runs), and requires identical outputs. The kernel is built at the scalar tier (``-march=x86-64``) and, where the CPU
runs it, the AVX2 tier (``-march=x86-64-v3``). The DSV4.1 plan and the AVX-512 tiers are checked on divix01
(``test/manual/dsv41/run_exl3_cpu_forward_checks.sh``).
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "exl3_cpu_job_check.cpp"
KERNEL = HERE.parents[3] / "python/sglang/kernels/jit/csrc/exl3/optimized/kernel.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64") or CXX is None,
    reason="the EXL3 CPU kernel needs Linux on x86, OpenMP and a C++20 compiler",
)


def _cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _runs_avx2() -> bool:
    return {"avx2", "fma", "f16c", "bmi1", "bmi2"} <= _cpu_flags()


@pytest.mark.parametrize(
    "march",
    ["x86-64", pytest.param("x86-64-v3", marks=pytest.mark.skipif(not _runs_avx2(), reason="no AVX2 on this CPU"))],
    ids=["scalar", "avx2"],
)
def test_the_job_interface_gives_the_same_bits_however_the_tasks_are_split(tmp_path, march):
    kernel_object = tmp_path / "kernel.o"
    harness_object = tmp_path / "harness.o"
    exe = tmp_path / "exl3_cpu_job_check"
    subprocess.run([CXX, "-std=c++20", "-Ofast", f"-march={march}", "-fopenmp", "-c", str(KERNEL), "-o", str(kernel_object)], check=True)
    subprocess.run([CXX, "-std=c++20", "-O2", "-pthread", "-c", str(HARNESS), "-o", str(harness_object)], check=True)
    subprocess.run([CXX, "-fopenmp", "-pthread", str(kernel_object), str(harness_object), "-o", str(exe)], check=True)
    workers = min(4, os.cpu_count() or 1)
    result = subprocess.run([str(exe), str(workers)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok "), result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
