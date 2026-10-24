"""The CPU experts job window (host/cpu_experts/job_window.hpp) passes its native harness (Linux, GCC, x86).

``cpu_experts_job_window_check.cpp`` drives the fake kernel's job interface (formats/fake) through a ``JobWindow`` on a
real ``Team`` and prints ``ok <check>`` per contract it holds.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "cpu_experts_job_window_check.cpp"
FAKE_KERNEL = HERE.parents[3] / "python/sglang/kernels/jit/csrc/moe/formats/fake/fake_kernel.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")
CXX_FLAGS = ["-std=c++20", "-pthread", "-march=x86-64"]
ADDRESS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g"]
THREAD = ["-fsanitize=thread", "-fno-omit-frame-pointer", "-g"]
CHECKS = (
    "every_task_runs_once_and_phases_run_in_order",
    "same_output_out_phases_run_in_sequence_order_while_other_outputs_interleave",
    "a_job_with_no_out_tasks_keeps_the_chain_ordered",
    "done_never_passes_an_unfinished_sequence",
    "the_done_word_skips_sequences_nobody_admitted",
    "a_staged_gate_holds_only_its_job",
    "a_full_window_stops_admission",
    "stop_drains",
    "status_2_is_surfaced_at_completion",
    "a_throwing_prepare_surfaces_to_the_admitter",
    "phases_with_no_tasks_complete_at_once_or_wait_for_their_gate",
    "the_done_word_reaches_the_last_sequence_under_out_of_order_completions",
    "no_lost_wakeup_when_a_helper_completes_the_job",
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64") or CXX is None,
    reason="the window needs Linux on x86 and a C++20 compiler",
)


def _links(tmp_path, flags) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    return subprocess.run([CXX, *flags, str(source), "-o", str(tmp_path / "probe")], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "flags",
    [["-O2"], ["-O2", *ADDRESS], ["-O1", *THREAD]],
    ids=["o2", "asan-ubsan", "tsan"],
)
def test_the_job_window_harness_passes(tmp_path, flags):
    sanitizers = [f for f in flags if f.startswith("-fsanitize")]
    if sanitizers and not _links(tmp_path, sanitizers):
        pytest.skip(f"{CXX} cannot link {sanitizers[0]} (its runtime is not installed)")
    exe = tmp_path / "cpu_experts_job_window_check"
    subprocess.run([CXX, *CXX_FLAGS, *flags, str(HARNESS), str(FAKE_KERNEL), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = set(result.stdout.split("\n"))
    missing = [name for name in CHECKS if f"ok {name}" not in passed]
    assert not missing, f"checks that did not report ok: {missing}\n{result.stdout}{result.stderr}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
