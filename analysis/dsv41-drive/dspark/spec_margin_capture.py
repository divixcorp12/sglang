"""A diagnostic served run of an A/B arm with the InstrBuild job trace, for spec_margin.py and layer_misses.py.

Starts an arm's server of both_cpu_ab.py (``--arm``, the ``dspark-both-prefetch`` arm by default) with ``SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX`` set (which
loads the instrumented host), serves the A/B's session set through logprob_probe.py, and shuts the launcher down
alone so every engine flushes its event file. Then breaks the layer records down (layer_misses.py) and, when the arm
prefetches, bins the speculative reads by gate rank and margin. The instrumented build's timings are not a throughput
number.

Locks: rowimg-disk.lock, then cc-gpu.lock (the protocol's order); the server runs under taskset on SERVER_CORES.

Usage: spec_margin_capture.py OUT_DIR [--arm ARM] [--no-verify-split] [--prompts 8] [--max-tokens 128] [--top-k-only] [--scorer cpu|gpu]
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


def server_env(out_dir: str, top_k_only: bool, scorer: str | None = None, arm: str = "dspark-both-prefetch",
               verify_split: bool = True) -> dict:
    """The arm's env with the job trace on; ``top_k_only`` and ``scorer`` override the arm's only when given. With
    ``verify_split`` it also writes the stage trace's route log, the router capture and the per-verify accept log,
    which verify_split.py joins; they add host work, so a timing comparison must use the same setting on both runs."""
    overrides, _ = both_cpu_ab.ARMS[arm]
    asked = {}
    if top_k_only:
        asked["SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"] = "1"
    if scorer is not None:
        asked["SGLANG_DSV41_RAM_PREFETCH_SCORER"] = scorer
    return arm_env.arm_env({
        **overrides,
        **asked,
        **({"SGLANG_DSV41_EXPERT_TRACE_PATH": os.path.join(out_dir, "stages.jsonl"),
            "SGLANG_DSV41_ROUTER_CAPTURE_PATH": os.path.join(out_dir, "router"),
            "SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH": os.path.join(out_dir, "verify-accept")} if verify_split else {}),
        "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": os.path.join(out_dir, "events"),
        "SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY": "524288",
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out_dir, "metrics.jsonl"),
    })


def counter_summary(log_path: str) -> dict:
    """From the server's last counters line: the speculative thread's time per record (the CPU scorer's scoring, the GPU
    scorer's wait for the slot) and the share of used reads still in flight at their demand."""
    try:
        with open(log_path, errors="replace") as f:
            lines = [line for line in f if both_cpu_ab.COUNTER_MARKER in line]
    except FileNotFoundError:
        return {}
    if not lines:
        return {}
    c = json.loads(lines[-1].split(both_cpu_ab.COUNTER_MARKER, 1)[1])
    keys = ("spec_scored", "spec_score_ns", "spec_issued", "spec_landed", "spec_used", "spec_promoted",
            "spec_dropped", "spec_late")
    out = {k: c.get(k) for k in keys}
    if c.get("spec_scored"):
        out["wait_or_score_us_per_record"] = c["spec_score_ns"] / c["spec_scored"] / 1000
    if c.get("spec_used"):
        out["in_flight_at_use"] = c.get("spec_promoted", 0) / c["spec_used"]
    return out


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
    p.add_argument("--scorer", choices=("cpu", "gpu"))
    p.add_argument("--arm", choices=tuple(both_cpu_ab.ARMS), default="dspark-both-prefetch")
    p.add_argument("--no-verify-split", dest="verify_split", action="store_false",
                   help="skip the route log, router capture and accept log (verify_split.py's inputs)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    env = os.environ | server_env(a.out_dir, a.top_k_only, a.scorer, a.arm, a.verify_split) | {"PYTHONPATH": os.path.join(REPO, "python")}
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
    events = os.path.join(a.out_dir, "events.*.jsonl")
    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "layer_misses.py"), events, "--json",
         os.path.join(a.out_dir, "layer_misses.json")]
    ).returncode
    if server_env(a.out_dir, a.top_k_only, a.scorer, a.arm, a.verify_split)["SGLANG_DSV41_RAM_PREFETCH"] == "1":
        rc = rc or subprocess.run(
            [sys.executable, os.path.join(HERE, "spec_margin.py"), events, "--json",
             os.path.join(a.out_dir, "spec_margin.json")]
        ).returncode
    if a.verify_split:
        rc = rc or subprocess.run(
            [sys.executable, os.path.join(HERE, "verify_split.py"), a.out_dir, "--json",
             os.path.join(a.out_dir, "verify_split.json")]
        ).returncode
    summary = counter_summary(os.path.join(a.out_dir, "server.log"))
    with open(os.path.join(a.out_dir, "counters.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("counters", json.dumps(summary))
    return rc


if __name__ == "__main__":
    sys.exit(main())
