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
# Every arm but the prefetch's states the RAM prefetch off, so a shell that exports it cannot carry it into an arm.
PREFETCH_OFF = {"SGLANG_DSV41_RAM_PREFETCH": "0"}
# arm: (overrides on base_env, DSpark argv)
ARMS = {
    "prod": ({**PREFETCH_OFF}, False),
    "dspark-draft-only": ({**arm_env.dspark_env(), **DRAFT_ONLY, **PREFETCH_OFF}, True),
    "dspark-both": ({**arm_env.dspark_env(), **PREFETCH_OFF}, True),
    # Experiment-only; retire with plan 2026-10-07-dsv41-row-weighted-serving-experiment. B arms, one source at a time.
    "dspark-both-rw-draft": (
        {**arm_env.dspark_env(), **PREFETCH_OFF, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "draft"},
        True,
    ),
    "dspark-both-rw-target": (
        {**arm_env.dspark_env(), **PREFETCH_OFF, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "target"},
        True,
    ),
    # NVMe-to-RAM prefetch (spec 2026-10-08-dsv41-ram-prefetch-design, The A/B): the replay's best arm, h=1, one
    # candidate per token, one row per layer, a pool of 2 per row and group; all pinned so a shell cannot leak.
    "dspark-both-prefetch": (
        {
            **arm_env.dspark_env(),
            "SGLANG_DSV41_RAM_PREFETCH": "1",
            "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": "1",
            "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "1",
            "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "2",
        },
        True,
    ),
}
# Build caches an experiment keeps private (run protocol): passed to every arm's server when set in the driver's env.
PASSTHROUGH = ("SGLANG_JIT_CACHE_DIR", "SGLANG_EXL3_BUILD_DIR")
# An arm whose outputs are also compared with its A's, not only with prod's.
REFERENCE = {"dspark-both-prefetch": "dspark-both"}
COUNTER_MARKER = "exl3 RAM miss thread counters "
RAM_KEYS = (
    "rows_read",
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
)


def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    passed = {k: os.environ[k] for k in PASSTHROUGH if os.environ.get(k)}
    return {**overrides, **passed, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")}


def _probe_overrides(arm: str, out: str) -> dict:
    """The arm's overrides for the probe server, with a metrics file of its own: the timed server's file is appended to
    and summarize() reads its last record, which a second server would replace."""
    return {**_overrides(arm, out), "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.probe.metrics.jsonl")}


def _health_timeout_s(dspark: bool) -> int:
    return arm_env.DSPARK_HEALTH_TIMEOUT_S if dspark else arm_env.HEALTH_TIMEOUT_S


def run_timed(arm: str, out: str) -> int:
    _, dspark = ARMS[arm]
    env = os.environ | {
        "DSV41_RUN_ROOT": out,
        "DSV41_SESSION_INDICES": ",".join(str(i) for i in range(8)),
        "DSV41_EXTRA_SERVER_ARGS": shlex.join(arm_env.DSPARK_ARGV) if dspark else "",
        "DSV41_HEALTH_TIMEOUT_S": str(_health_timeout_s(dspark)),
    }
    if dspark:
        # run_arm.sh builds its argv without dspark=True, so the recipe's fraction reaches it as the arm override.
        env["DSV41_MEM_FRACTION_STATIC"] = arm_env.DSPARK_MEM_FRACTION_STATIC
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
    env = os.environ | arm_env.arm_env(_probe_overrides(arm, out)) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PROBE_PORT, dspark=dspark).argv()
    with open(arm_env.GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(os.path.join(out, f"{arm}.probe-server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        try:
            if not _healthy(PROBE_PORT, time.monotonic() + _health_timeout_s(dspark)):
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


def _server_counters(out: str, arm: str):
    """The last counters line of the arm's timed server: the service's lifetime (warm-up and the timed set)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    path = os.path.join(out, "servers", arm, runs[-1], "server.log")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        lines = [line for line in f if COUNTER_MARKER in line]
    return json.loads(lines[-1].split(COUNTER_MARKER, 1)[1]) if lines else None


def _warmup_tokens(out: str, arm: str) -> list[int]:
    """Completion tokens of each warm-up round the arm's timed server ran (run_arm.sh's results-warmup-N.jsonl)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    run = os.path.join(out, "servers", arm, runs[-1])
    tokens = []
    for name in sorted(os.listdir(run)):
        if name.startswith("results-warmup-") and name.endswith(".jsonl"):
            with open(os.path.join(run, name)) as f:
                tokens.append(sum(json.loads(line).get("completion_tokens") or 0 for line in f if line.strip()))
    return tokens


def _compare(base_path: str, other_path: str):
    sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
    import dspark_text_band

    with open(base_path) as f:
        base = json.load(f)
    with open(other_path) as f:
        other = json.load(f)
    try:
        return dspark_text_band.compare(base, other)
    except ValueError as error:
        return {"error": str(error)}


def summarize(out: str) -> dict:
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
            entry["text_vs_prod"] = _compare(probes["prod"], probes[arm])
        counters = _server_counters(out, arm)
        if counters:
            entry["ram"] = {k: counters.get(k) for k in RAM_KEYS}
            entry["ram"]["scope"] = "server lifetime: warm-up and the timed set"
            # The counters are server-lifetime, so the denominator is too: warm-up rounds plus the timed set.
            warmups = _warmup_tokens(out, arm)
            lifetime = sum(warmups) + sum(r["completion_tokens"] for r in rows)
            entry["warmup_rounds"] = len(warmups)
            entry["lifetime_tokens"] = lifetime
            entry["ram_rows_per_token_lifetime"] = counters["rows_read"] / lifetime if lifetime else None
            # Drive load next to the saving: every NVMe row, and the speculative rows read but never used.
            if lifetime and "spec_issued" in counters:
                entry["nvme_rows_per_token_lifetime"] = (counters["rows_read"] + counters["spec_issued"]) / lifetime
            if lifetime and "spec_landed" in counters and "spec_used" in counters:
                entry["spec_wasted_per_token_lifetime"] = (counters["spec_landed"] - counters["spec_used"]) / lifetime
        reference = REFERENCE.get(arm)
        ref_probes = {a: os.path.join(out, f"{a}.probe.json") for a in (reference, arm)} if reference else {}
        if reference and not os.path.exists(ref_probes[reference]):
            entry["text_vs_reference"] = "missing reference probe"
        elif reference and os.path.exists(ref_probes[arm]):
            entry["text_vs_reference"] = _compare(ref_probes[reference], ref_probes[arm])
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
