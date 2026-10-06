"""Server A/B (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 17): today's production recipe against DSpark with
both CPU-expert clients, and DSpark with the draft's only (the target's CPU experts off, §33.9's configuration in a
server) to isolate them.

Per arm: run_arm.sh's timed set over the 8 corpus sessions (ms/token, and accept length from the driver's
spec_tokens_details), then a short-lived server on the same env for logprob_probe.py (8 prompts, 128 tokens, top-5).
The text bar is dspark_text_band.py against prod. Run on divix01 from the pushed worktree holding rowimg-disk.lock
only: run_arm.sh takes cc-gpu.lock itself, and the probe phase takes it here (lock order: disk, then GPU).

    flock /data/models/slang/nvfp4-work/rowimg-disk.lock python analysis/dsv41-drive/dspark/both_cpu_ab.py OUT [ARM ...]
"""

import fcntl
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

PORT = 30017
PROBE_PORT = 30018
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
DRAFT_ONLY = {
    "SGLANG_DSV41_CPU_EXPERTS": "0",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": "8",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "0",
}
# arm: (overrides on base_env, DSpark argv)
ARMS = {
    "prod": ({}, False),
    "dspark-draft-only": ({**arm_env.dspark_env(), **DRAFT_ONLY}, True),
    "dspark-both": (arm_env.dspark_env(), True),
}


def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    return {**overrides, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")}


def run_timed(arm: str, out: str) -> int:
    _, dspark = ARMS[arm]
    env = os.environ | {
        "DSV41_RUN_ROOT": out,
        "DSV41_SESSION_INDICES": ",".join(str(i) for i in range(8)),
        "DSV41_EXTRA_SERVER_ARGS": shlex.join(arm_env.DSPARK_ARGV) if dspark else "",
    }
    cmd = [os.path.join(REPO, "benchmarks", "dsv41_baseline", "run_arm.sh"), arm, str(PORT)]
    cmd += [f"{k}={v}" for k, v in _overrides(arm, out).items()]
    print("===", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=REPO).returncode


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


def run_probe(arm: str, out: str) -> int:
    """The arm's server on PROBE_PORT under cc-gpu.lock, for logprob_probe.py only."""
    _, dspark = ARMS[arm]
    env = os.environ | arm_env.arm_env(_overrides(arm, out)) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PROBE_PORT, dspark=dspark).argv()
    with open(arm_env.GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(os.path.join(out, f"{arm}.probe-server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        try:
            if not _healthy(PROBE_PORT, time.monotonic() + arm_env.HEALTH_TIMEOUT_S):
                return 1
            probe = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
            return subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, probe, "--port", str(PROBE_PORT),
                 "--sessions", SESSIONS, "--prompts", "8", "--max-tokens", "128", "--top-logprobs", "5",
                 "--out", os.path.join(out, f"{arm}.probe.json")], cwd=REPO,
            ).returncode
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def _results(out: str, arm: str) -> list[dict]:
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    with open(os.path.join(out, "servers", arm, runs[-1], "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(out: str) -> dict:
    sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
    import dspark_text_band

    summary = {}
    for arm in ARMS:
        try:
            rows = _results(out, arm)
        except FileNotFoundError:
            continue
        ms = [1000.0 / r["decode_tokens_per_sec"] for r in rows if r.get("decode_tokens_per_sec")]
        spec = [r.get("spec_tokens_details") or {} for r in rows]
        verifies = sum(s.get("spec_verify_ct", 0) for s in spec)
        entry = {
            "sessions": len(rows),
            "ms_per_token_median": statistics.median(ms) if ms else None,
            "ms_per_token": ms,
            "accept_length": (sum(r["completion_tokens"] for r in rows) / verifies) if verifies else None,
        }
        metrics = os.path.join(out, f"{arm}.metrics.jsonl")
        if os.path.exists(metrics):
            with open(metrics) as f:
                last = json.loads([line for line in f if line.strip()][-1])
            graphed = last.get("counters", {}).get("graphed_verify")
            if graphed and graphed.get("graphed_verify_ct"):
                entry["reverify_rate"] = graphed["verify_overflow_ct"] / graphed["graphed_verify_ct"]
                entry["reverify_ct"] = graphed["verify_overflow_ct"]
        probes = {a: os.path.join(out, f"{a}.probe.json") for a in ("prod", arm)}
        if arm != "prod" and all(os.path.exists(p) for p in probes.values()):
            with open(probes["prod"]) as f:
                base = json.load(f)
            with open(probes[arm]) as f:
                other = json.load(f)
            entry["text_vs_prod"] = dspark_text_band.compare(base, other)
        summary[arm] = entry
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    out = os.path.abspath(sys.argv[1])
    arms = sys.argv[2:] or list(ARMS)
    os.makedirs(out, exist_ok=True)
    for arm in arms:
        for step in (run_timed, run_probe):
            rc = step(arm, out)
            print(f"{arm} {step.__name__}: rc={rc}", flush=True)
            if rc:
                sys.exit(rc)
    print(json.dumps(summarize(out), indent=2), flush=True)


if __name__ == "__main__":
    main()
