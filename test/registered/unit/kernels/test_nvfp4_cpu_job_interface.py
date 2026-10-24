"""The NVFP4 CPU kernel's job interface (nvfp4/optimized/entry.h) gives the same bits however its tasks are split (Linux, GCC, x86).

``nvfp4_cpu_job_check.cpp`` runs random calls (overwrite and accumulate, staged and not, generic and MiMo V2.6 Pro
plans, and inputs Q8_0 cannot represent) through a one-worker JobRunner and a several-worker JobRunner (dynamic runs),
and requires identical outputs. The kernel is built at the
scalar tier (``-march=x86-64``) and, where the CPU runs it, the AVX2 tier (``-march=x86-64-v3``). The AVX2 and scalar
dumps against another revision are checked on divix01 (``test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh``).
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
BUILD = REPO / "python/sglang/srt/layers/quantization/nvfp4/build.py"
HARNESS = REPO / "test/registered/unit/kernels/nvfp4_cpu_job_check.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs_avx2() -> bool:
    flags = next(
        (l.split(":", 1)[1].split() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("flags")), []
    )
    return {"avx2", "fma", "f16c", "bmi1", "bmi2", "abm", "movbe"} <= set(flags)


@pytest.mark.parametrize(
    "march",
    ["x86-64", pytest.param("x86-64-v3", marks=pytest.mark.skipif(not _runs_avx2(), reason="no AVX2 on this CPU"))],
    ids=["scalar", "avx2"],
)
def test_the_job_interface_gives_the_same_bits_however_the_tasks_are_split(tmp_path, march):
    exe = _build_module().build(tmp_path / "nvfp4_cpu_job_check", cxx=CXX, main=HARNESS, march=march)
    workers = min(4, os.cpu_count() or 1)
    result = subprocess.run([str(exe), str(workers)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok "), result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
