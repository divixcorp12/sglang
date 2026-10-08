"""Capture a DSpark verify route log with router inputs, for verify_replay.py.

Starts the dspark-both server (the recipe's DSpark env and argv, both CPU-expert clients, the 104 GiB tier) with the
stage trace and router capture pointed at OUT_DIR, serves ``--prompts`` corpus sessions from ``--sessions-skip``
through logprob_probe.py, and shuts the server down normally so the route log's final read lands. Then checks the
capture: no forward lost, every verify forward with a router record. A diagnostic run: its timings are not a
throughput number.

Locks: rowimg-disk.lock, then cc-gpu.lock (the protocol's order); the server runs under taskset on SERVER_CORES.

Usage: capture_verify.py OUT_DIR [--sessions-skip 0|9] [--prompts 8] [--max-tokens 128]
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
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
import arm_env  # noqa: E402
import session_subset  # noqa: E402

PORT = 30031
DISK_LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"
PROBE = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")


def write_sessions(corpus: str, out: str, *, skip: int, n: int) -> str:
    """The first ``n`` corpus rows from ``skip``: 0 is the suite's eight, 9 skips the suite and its warm-up row."""
    rows = session_subset.load_raw_sessions(corpus, n=n, skip=skip)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def server_env(out_dir: str) -> dict:
    return arm_env.arm_env({
        **arm_env.dspark_env(),
        "SGLANG_DSV41_EXPERT_TRACE_PATH": os.path.join(out_dir, "trace.jsonl"),
        "SGLANG_DSV41_ROUTER_CAPTURE_PATH": os.path.join(out_dir, "router"),
        # The served RAM-miss counters the replay's tier model is validated against.
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out_dir, "metrics.jsonl"),
    })


def check_capture(out_dir: str) -> dict:
    from tier_sim import load_forwards

    loaded = load_forwards(os.path.join(out_dir, "trace.jsonl"), allow_dropped=True)
    if loaded["dropped"]:
        raise RuntimeError(f"the route reader lost {loaded['dropped']} graph forwards; the capture cannot be replayed")
    verifies = [f for f in loaded["forwards"] if f.get("verify")]
    without = [f["seq"] for f in verifies if f.get("router") is None]
    if without:
        raise RuntimeError(f"{len(without)} verify forwards have no router record (first seq {without[0]})")
    with open(os.path.join(out_dir, "router.json")) as f:
        header = json.load(f)
    return {"verify_forwards": len(verifies), "forwards": len(loaded["forwards"]), "tokens": header["tokens"],
            "layers": len(header["layer_ids"])}


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
    p.add_argument("--sessions-skip", type=int, default=0)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sessions = write_sessions(session_subset.CORPUS_PATH, os.path.join(a.out_dir, "sessions.jsonl"), skip=a.sessions_skip, n=a.prompts)
    env = os.environ | server_env(a.out_dir) | {"PYTHONPATH": os.path.join(REPO, "python")}
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
                 "--sessions", sessions, "--prompts", str(a.prompts), "--max-tokens", str(a.max_tokens),
                 "--top-logprobs", "1", "--out", os.path.join(a.out_dir, "probe.json")], cwd=REPO,
            ).returncode
        finally:
            # A normal shutdown runs the route log's final read; a SIGKILL would lose the last ring entries.
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    if rc != 0:
        return rc
    summary = check_capture(a.out_dir)
    summary.update({"sessions_skip": a.sessions_skip, "prompts": a.prompts, "max_tokens": a.max_tokens,
                    "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()})
    with open(os.path.join(a.out_dir, "capture.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
