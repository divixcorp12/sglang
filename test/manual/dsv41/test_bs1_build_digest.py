"""The one-token build's kernels compile to the machine code recorded before the wire was widened (plan
2026-10-06-dsv41-dspark-both-cpu-experts Task 2): the SASS of every BS1 kernel the widening touched (CW, CC, the DIRECT
destinations and commit kernels, the route tables) and of the unchanged ones (C1, S). Same machine code, same outputs
and timing. The post kernel and the host .text are compared in Task 2 only (bs1_build_digest.py --permanent): later
tasks change them on purpose, inert at one token. A toolchain (nvcc or GPU arch) other than the golden's FAILS with "re-record the golden": a silent skip would turn the pin
into nothing after an upgrade. SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED=1 turns it back into a skip. A missing nvcc or
cuobjdump skips. Run on divix01
under cc-gpu.lock (it runs the BS1 suites)."""

import os
import shutil
import subprocess
import sys

import pytest
import torch

from sglang.srt.environ import envs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "bs1_build_digest.json")


def digest_outcome(returncode: int, output: str, acknowledged: bool) -> tuple[str, str]:
    """("pass" | "skip" | "fail", message) for bs1_build_digest.py's exit status: 2 is a toolchain other than the
    golden's, which fails unless acknowledged (the golden then needs re-recording on the new toolchain)."""
    if returncode == 0:
        return "pass", ""
    if returncode == 2:
        if acknowledged:
            return "skip", output.strip()
        return "fail", (
            f"{output.strip()}\nthe toolchain changed, so the pin compared nothing: re-record the golden "
            f"(scripts/dsv41/bs1_build_digest.py --write {GOLDEN}), or set "
            "SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED=1 to skip"
        )
    return "fail", output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_one_token_build_is_unchanged():
    for tool in ("nvcc", "cuobjdump"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is not on PATH")
    run = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "dsv41", "bs1_build_digest.py"), "--compare", GOLDEN, "--permanent"],
        capture_output=True, text=True, cwd=REPO,
    )
    verdict, message = digest_outcome(run.returncode, run.stdout + run.stderr, envs.SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED.get())
    if verdict == "skip":
        pytest.skip(message)
    assert verdict == "pass", message


def test_a_toolchain_change_fails_with_the_re_record_message():
    verdict, message = digest_outcome(2, "toolchain differs: golden sm_120 nvcc 13.3, now sm_120 nvcc 13.4", False)
    assert verdict == "fail" and "re-record the golden" in message and "nvcc 13.4" in message


def test_the_acknowledge_env_turns_a_toolchain_change_into_a_skip():
    with envs.SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED.override(True):
        assert envs.SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED.get() is True
        assert digest_outcome(2, "toolchain differs", True)[0] == "skip"
    assert envs.SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED.get() is False


@pytest.mark.parametrize("acknowledged", [False, True])
def test_a_real_digest_difference_always_fails(acknowledged):
    assert digest_outcome(1, "device exl3_ram_miss_post_kernel::x", acknowledged)[0] == "fail"
    assert digest_outcome(0, "BS1 build unchanged", acknowledged)[0] == "pass"
