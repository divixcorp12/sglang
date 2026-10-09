"""A diagnostic served run of the RAM prefetch arm with the InstrBuild job trace, for spec_margin.py.

Starts the ``dspark-both-prefetch`` server of both_cpu_ab.py with ``SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX`` set (which
loads the instrumented host), serves the A/B's session set through logprob_probe.py, and shuts the launcher down
alone so every engine flushes its event file. Then bins the speculative reads by gate rank and margin. The
instrumented build's timings are not a throughput number.

Locks: rowimg-disk.lock, then cc-gpu.lock (the protocol's order); the server runs under taskset on SERVER_CORES.

Usage: spec_margin_capture.py OUT_DIR [--prompts 8] [--max-tokens 128] [--top-k-only] [--scorer cpu|gpu]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402
import both_cpu_ab  # noqa: E402

PORT = 30032
DISK_LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"
PROBE = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")


    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "spec_margin.py"), os.path.join(a.out_dir, "events.*.jsonl"),
         "--json", os.path.join(a.out_dir, "spec_margin.json")]
    ).returncode
    summary = counter_summary(os.path.join(a.out_dir, "server.log"))
    with open(os.path.join(a.out_dir, "counters.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("counters", json.dumps(summary))
    return rc


def stop_server(server: subprocess.Popen, timeout: float = 300) -> None:
    """SIGTERM the launcher alone so the scheduler shuts down and the engines write their event files; a group
    SIGTERM kills the scheduler first and loses them."""
    os.kill(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _healthy(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top-k-only", action="store_true")
    p.add_argument("--scorer", choices=("cpu", "gpu"), default="cpu")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    env = os.environ | server_env(a.out_dir, a.top_k_only, a.scorer) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PORT, dspark=True).argv()
    with open(DISK_LOCK, "w") as disk, open(arm_env.GPU_LOCK, "w") as gpu:
        fcntl.flock(disk, fcntl.LOCK_EX)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        with open(os.path.join(a.out_dir, "server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        rc = 1
        try:
            if not _healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
                return 1
            rc = subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, PROBE, "--port", str(PORT),
                 "--sessions", both_cpu_ab.SESSIONS, "--prompts", str(a.prompts), "--max-tokens", str(a.max_tokens),
                 "--top-logprobs", "1", "--out", os.path.join(a.out_dir, "probe.json")], cwd=REPO,
            ).returncode
        finally:
            stop_server(server)
    if rc != 0:
        return rc
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add analysis/dsv41-drive/dspark/both_cpu_ab.py analysis/dsv41-drive/dspark/spec_margin_capture.py
git commit -m "Add the GPU-scorer A/B arms, a paired summary and the capture's --scorer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/scripts/test_both_cpu_ab.py test/registered/unit/scripts/test_spec_margin_capture.py test/registered/unit/scripts/test_spec_margin.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE


if __name__ == "__main__":
    sys.exit(main())
