"""The one-token build's kernels compile to the machine code recorded before the wire was widened (plan
2026-10-06-dsv41-dspark-both-cpu-experts Task 2): the SASS of every BS1 kernel the widening touched (CW, CC, the DIRECT
destinations and commit kernels, the route tables) and of the unchanged ones (C1, S). Same machine code, same outputs
and timing. The post kernel and the host .text are compared in Task 2 only (bs1_build_digest.py --permanent): later
tasks change them on purpose, inert at one token. Skips on another GPU arch or nvcc than the golden's. Run on divix01
under cc-gpu.lock (it runs the BS1 suites)."""

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "bs1_build_digest.json")


def test_the_one_token_build_is_unchanged():
    run = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "dsv41", "bs1_build_digest.py"), "--compare", GOLDEN, "--permanent"],
        capture_output=True, text=True, cwd=REPO,
    )
    if run.returncode == 2:
        pytest.skip(run.stdout.strip())
    assert run.returncode == 0, run.stdout + run.stderr
