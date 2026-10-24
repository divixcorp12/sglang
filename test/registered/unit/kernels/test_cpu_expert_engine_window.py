"""The CPU expert engine on its job window (host/cpu_experts.h) passes its native harness (Linux, GCC, x86).

``cpu_expert_engine_check.cpp`` drives ``BasicCpuExpertEngine`` directly, with no tier, over the fake kernel (the scheduler
scenarios: a staged gate holds only its job, a full window stops admission, stop() drains, and the fail-stops) and over the
EXL3 kernel at the scalar and AVX2 tiers (several records in flight with mixed hit, miss and staged jobs, at one and six
tokens a row, and one record of 32 tokens: the outputs equal the same jobs run one at a time, bit for bit). The fake
scenarios also run under AddressSanitizer and ThreadSanitizer where the compiler links them.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="base-a-test-cpu")

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "cpu_expert_engine_check.cpp"
CSRC = HERE.parents[3] / "python/sglang/kernels/jit/csrc"
FAKE_KERNEL = CSRC / "moe/formats/fake/fake_kernel.cpp"
EXL3_KERNEL = CSRC / "exl3/optimized/kernel.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")
CXX_FLAGS = ["-std=c++20", "-pthread"]
ADDRESS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g"]
THREAD = ["-fsanitize=thread", "-fno-omit-frame-pointer", "-g"]
FAKE_SCENARIOS = ("mixed", "tokens32", "gate", "full", "stop", "stop_opens_gate")
FATAL = {
    "watchdog": "the CPU job of row 3 incomplete after",
    "refused": "CPU expert forward of row 5 failed: the job refused its input (status 2)",
    "rejected": "CPU expert forward of row 6 failed: fake CPU expert prepare rejected the call",
}

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64") or CXX is None,
    reason="the engine needs Linux on x86 and a C++20 compiler",
)


def _links(tmp_path, flags) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    return subprocess.run([CXX, *flags, str(source), "-o", str(tmp_path / "probe")], capture_output=True).returncode == 0


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


def _tvm_includes() -> list[str]:
    """The engine's headers name tvm-ffi's tensor type (reader_base.h); the harness uses none of it."""
    try:
        from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path
    except ImportError:
        pytest.skip("tvm_ffi is not installed")
    return [f"-I{path}" for path in dict.fromkeys([find_include_path(), find_dlpack_include_path()])]


def _build(tmp_path, name, defines, kernel, flags, kernel_flags=()):
    exe = tmp_path / name
    objects = []
    if kernel_flags:  # the EXL3 kernel at the tier under test, as the host module compiles it
        objects.append(str(tmp_path / "kernel.o"))
        subprocess.run(
            [CXX, "-std=c++20", "-Ofast", *kernel_flags, "-fopenmp", "-c", str(kernel), "-o", objects[0]], check=True
        )
    else:
        objects = [str(kernel)]
    subprocess.run(
        [CXX, *CXX_FLAGS, *defines, *flags, *_tvm_includes(), "-march=x86-64", str(HARNESS), *objects, "-fopenmp",
         "-o", str(exe)],
        check=True,
    )
    return exe


def _run(exe, scenario, timeout=600):
    return subprocess.run([str(exe), scenario], capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module", params=["o2", "asan-ubsan", "tsan"])
def fake_exe(request, tmp_path_factory):
    flags = {"o2": ["-O2"], "asan-ubsan": ["-O1", *ADDRESS], "tsan": ["-O1", *THREAD]}[request.param]
    tmp_path = tmp_path_factory.mktemp(request.param)
    sanitizers = [f for f in flags if f.startswith("-fsanitize")]
    if sanitizers and not _links(tmp_path, sanitizers):
        pytest.skip(f"{CXX} cannot link {sanitizers[0]} (its runtime is not installed)")
    return _build(tmp_path, "engine_check_fake", ["-DENGINE_KERNEL_FAKE"], FAKE_KERNEL, flags)


@pytest.mark.parametrize("scenario", FAKE_SCENARIOS)
def test_the_engine_window_scenarios_pass_over_the_fake_kernel(fake_exe, scenario):
    result = _run(fake_exe, scenario)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ok {scenario}" in result.stdout.split("\n"), result.stdout + result.stderr


@pytest.mark.parametrize("scenario", sorted(FATAL))
def test_a_stuck_refused_or_rejected_job_fails_stop_naming_its_row(fake_exe, scenario):
    result = _run(fake_exe, scenario, timeout=120)
    assert result.returncode == -signal.SIGABRT, (result.returncode, result.stdout, result.stderr[-2000:])
    assert FATAL[scenario] in result.stderr, result.stderr[-2000:]


@pytest.mark.parametrize(
    "march",
    ["x86-64", pytest.param("x86-64-v3", marks=pytest.mark.skipif(not _runs_avx2(), reason="no AVX2 on this CPU"))],
    ids=["scalar", "avx2"],
)
@pytest.mark.parametrize("scenario", ["mixed", "tokens32"])
def test_records_in_flight_match_the_same_jobs_one_at_a_time_over_the_exl3_kernel(tmp_path, march, scenario):
    exe = _build(tmp_path, "engine_check_exl3", ["-DENGINE_KERNEL_EXL3"], EXL3_KERNEL, ["-O2"], [f"-march={march}"])
    result = _run(exe, scenario)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ok {scenario}" in result.stdout.split("\n"), result.stdout + result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
