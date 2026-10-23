"""A diagnostic served run of an A/B arm with the InstrBuild job trace, for spec_margin.py and layer_misses.py.

Starts an arm's server of both_cpu_ab.py (``--arm``, the ``dspark-both-prefetch`` arm by default) with the ``job.prefix`` of ``SGLANG_EXPERT_STREAM_TRACE`` set (which
loads the instrumented host), serves the A/B's session set through logprob_probe.py, and shuts the launcher down
alone so every engine flushes its event file. Then breaks the layer records down (layer_misses.py) and, when the arm
prefetches, bins the speculative reads by gate rank and margin. The instrumented build's timings are not a throughput
number.

Locks: rowimg-disk.lock, then cc-gpu.lock (the protocol's order), taken here through run_locks.take, which also accepts
them held by an outer `flock`; the server runs under taskset on SERVER_CORES.

With --long-prompt PATH, long_prompt.py replaces logprob_probe.py: one streaming request of that prompt, thinking on,
cancelled --decode-seconds after its first token (after --ttft-repeats short ones for TTFT samples), on the same
collectors; long_compare.py reads the result.

Usage: spec_margin_capture.py OUT_DIR [--arm ARM] [--no-verify-split] [--prompts 8] [--max-tokens 128] [--top-k-only] [--scorer cpu|gpu]
       spec_margin_capture.py OUT_DIR [--arm ARM] --long-prompt PATH [--decode-seconds 300] [--ttft-repeats 0]
"""

from __future__ import annotations

import argparse
import glob
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
import run_locks  # noqa: E402
import both_cpu_ab  # noqa: E402

PORT = 30032
DISK_LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"
PROBE = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
LONG_PROMPT = os.path.join(HERE, "long_prompt.py")
# The job trace keeps its first CAPACITY events per file and drops the rest. A verify costs the copy engine's file ~410
# (share28-ab-20261016) at ~270 ms per instrumented verify, so 300 s of decode needs ~460k: the default 524288 would
# cut the window's end, and this (the trace's maximum) covers about 600 s.
LONG_TRACE_CAPACITY = "1048576"


def server_env(out_dir: str, top_k_only: bool, scorer: str | None = None, arm: str = "dspark-both-prefetch",
               verify_split: bool = True) -> dict:
    """The arm's env with the job trace on; ``top_k_only`` and ``scorer`` override the arm's only when given. With
    ``verify_split`` it also writes the stage trace's route log, the router capture and the per-verify accept log,
    which verify_split.py joins; they add host work, so a timing comparison must use the same setting on both runs. An
    STS collection arm writes its shards under OUT_DIR (both_cpu_ab.sts_collect_stem)."""
    overrides, _ = both_cpu_ab.ARMS[arm]
    asked = {}
    if arm in both_cpu_ab.STS_COLLECT_ARMS:
        asked[both_cpu_ab.STS_COLLECT] = both_cpu_ab.sts_collect_stem(out_dir)
    if top_k_only:
        asked["SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_TOP_K_ONLY"] = "1"
    if scorer is not None:
        asked["SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"] = scorer
    trace = both_cpu_ab.member
    return arm_env.arm_env(both_cpu_ab.collapse({
        **overrides,
        **asked,
        **({trace(both_cpu_ab.TRACE, "expert_path"): os.path.join(out_dir, "stages.jsonl"),
            trace(both_cpu_ab.TRACE, "router_capture_path"): os.path.join(out_dir, "router"),
            trace(both_cpu_ab.TRACE, "verify_accept_path"): os.path.join(out_dir, "verify-accept")}
           if verify_split else {}),
        trace(both_cpu_ab.TRACE, "job.prefix"): os.path.join(out_dir, "events"),
        trace(both_cpu_ab.TRACE, "job.capacity"): 524288,
        trace(both_cpu_ab.TRACE, "hot_metrics_file"): os.path.join(out_dir, "metrics.jsonl"),
    }))


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
            "spec_dropped", "spec_late", "rows_read", "spec_promoted_early", "spec_promote_no_delta",
            "spec_promote_no_room", "spec_promote_no_victim", "spec_promote_victim_routed", "spec_promote_used",
            "spec_promote_reading", "spec_publish_backstop", "spec_promote_row_busy", "spec_pool_served",
            "spec_skip_landed_gpu",
            "spec_skip_landed_cpu", "spec_skip_reading_gpu", "spec_skip_reading_cpu",
            "drip_issued", "drip_landed", "drip_used", "drip_abandoned", "drip_no_slot",
            "drip_paused", "drip_skipped_bytes", "drip_late",
            # rows_read is the service's demand misses only; prefill fills show in the drive load line's fill_bytes
            # and their victims here.
            "evictions", "eager_evictions")
    out = {k: c.get(k) for k in keys}
    if c.get("spec_scored"):
        out["wait_or_score_us_per_record"] = c["spec_score_ns"] / c["spec_scored"] / 1000
    if c.get("spec_used"):
        out["in_flight_at_use"] = c.get("spec_promoted", 0) / c["spec_used"]
    return out


class Interrupted(SystemExit):
    """SIGTERM or SIGHUP, raised in the main thread so the finally blocks run: the default action ends the process at
    once, and a server started under it would outlive it (reparented to PID 1, holding the GPU)."""


def raise_on_signal() -> None:
    def handler(signum, frame):
        raise Interrupted(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


def launch_server(argv, env, out_dir: str, log, cwd: str = REPO) -> subprocess.Popen:
    """Start the server as the leader of its own session and record its process group in OUT/server.pgid, so a wrapper
    can clear the group if this process dies before it can."""
    server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)
    with open(os.path.join(out_dir, "server.pgid"), "w") as f:
        f.write(f"{server.pid}\n")
    return server


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_server(server: subprocess.Popen, timeout: float = 300) -> None:
    """SIGTERM the launcher alone so the scheduler shuts down and the engines write their event files; a group
    SIGTERM kills the scheduler first and loses them. Then SIGKILL the whole group (children outlive the launcher) and
    wait until no member is left."""
    try:
        os.kill(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 30
    while group_alive(server.pid) and time.monotonic() < deadline:
        time.sleep(0.1)


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


def client_argv(a: argparse.Namespace) -> list[str]:
    """The serving client: logprob_probe.py over the A/B's sessions, or long_prompt.py with --long-prompt."""
    if a.long_prompt:
        sampling = ([] if not a.temperature else ["--temperature", str(a.temperature)]) + (
            [] if a.top_p is None else ["--top-p", str(a.top_p)])
        return [arm_env.PYTHON, LONG_PROMPT, "--port", str(PORT), "--prompt", os.path.abspath(a.long_prompt),
                "--decode-seconds", str(a.decode_seconds), "--ttft-repeats", str(a.ttft_repeats),
                "--out-dir", a.out_dir, *sampling]
    if a.prompts_file:
        # Every prompt of every file, sampled as asked and with thinking on; no logprobs (nothing here reads them).
        sampling = ["--temperature", str(a.temperature)] + ([] if a.top_p is None else ["--top-p", str(a.top_p)])
        files = [x for path in a.prompts_file for x in ("--prompts-file", os.path.abspath(path))]
        return [arm_env.PYTHON, PROBE, "--port", str(PORT), "--prompts", str(10**6), "--max-tokens", str(a.max_tokens),
                "--top-logprobs", "0", *sampling, *files, "--out", os.path.join(a.out_dir, "probe.json")]
    return [arm_env.PYTHON, PROBE, "--port", str(PORT), "--sessions", both_cpu_ab.SESSIONS, "--prompts", str(a.prompts),
            "--max-tokens", str(a.max_tokens), "--top-logprobs", "1", "--out", os.path.join(a.out_dir, "probe.json")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top-k-only", action="store_true")
    p.add_argument("--scorer", choices=("cpu", "gpu"))
    p.add_argument("--arm", choices=tuple(both_cpu_ab.ARMS), default="dspark-both-prefetch")
    p.add_argument("--analysis-warn-only", action="store_true",
                   help="a refusal by layer_misses.py or spec_margin.py after a complete run (a job-trace buffer overflow) "
                        "is printed as a warning and does not fail the capture; the server is unaffected")
    p.add_argument("--no-verify-split", dest="verify_split", action="store_false",
                   help="skip the route log, router capture and accept log (verify_split.py's inputs)")
    p.add_argument("--draft-top-prob", action="store_true",
                   help="also log the draft's per-position top probability each verify (draft-top-prob.*.jsonl); "
                        "one extra draft LM-head pass per verify, so compare timings only against runs with it on")
    p.add_argument("--long-prompt", metavar="PATH",
                   help="serve this prompt once, streaming with thinking on, instead of the probe's sessions")
    p.add_argument("--prompts-file", metavar="JSON", action="append", default=[],
                   help="serve every prompt of this JSON list of strings (repeatable) through logprob_probe.py, one "
                        "request each, thinking on, at --temperature/--top-p and capped at --max-tokens new tokens; "
                        "probe.json records each request's CLOCK_MONOTONIC span")
    p.add_argument("--decode-seconds", type=float, default=300.0,
                   help="with --long-prompt: cancel the request this long after its first streamed token")
    p.add_argument("--ttft-repeats", type=int, default=0,
                   help="with --long-prompt: first send it this many times with 20 s of decode, for TTFT samples")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="with --long-prompt: sample at this temperature (0, the default, is greedy)")
    p.add_argument("--top-p", type=float, default=None, help="with --long-prompt and --temperature: nucleus top-p")
    p.add_argument("--then-generate", metavar="JSON",
                   help="after the client succeeds and before the shutdown, POST this /generate body once and save the "
                        "reply as OUT_DIR/then_generate.json (e.g. a sampled request outside the timed windows)")
    p.add_argument("--resolve-env", metavar="KEYS",
                   help="launch nothing: print the server's env (the comma-separated KEYS from the caller's env under "
                        "the capture's own), its argv, and which of KEYS the capture overrides, as JSON")
    a = p.parse_args(argv)
    if a.long_prompt and a.prompts_file:
        p.error("--long-prompt and --prompts-file are exclusive")
    if not a.long_prompt and (a.decode_seconds != 300.0 or a.ttft_repeats):
        p.error("--decode-seconds and --ttft-repeats need --long-prompt")
    if not (a.long_prompt or a.prompts_file) and (a.temperature or a.top_p is not None):
        p.error("--temperature and --top-p need --long-prompt or --prompts-file")
    for path in a.prompts_file:
        if not os.path.isfile(path):
            p.error(f"--prompts-file {path}: no such file")
    if a.temperature < 0 or (a.top_p is not None and (not 0 < a.top_p <= 1 or not a.temperature)):
        p.error("--temperature must be non-negative, and --top-p in (0, 1] with a positive --temperature")
    if a.long_prompt and not os.path.isfile(a.long_prompt):
        p.error(f"--long-prompt {a.long_prompt}: no such file")
    if a.decode_seconds <= 0 or a.ttft_repeats < 0:
        p.error("--decode-seconds must be positive and --ttft-repeats non-negative")
    return a


def capture_env(a: argparse.Namespace) -> dict:
    """What the capture sets over the caller's env for its server; a caller's value for one of these keys is lost."""
    env = server_env(a.out_dir, a.top_k_only, a.scorer, a.arm, a.verify_split)
    if a.long_prompt or a.prompts_file:
        trace = json.loads(env[both_cpu_ab.TRACE])
        trace["job"]["capacity"] = int(LONG_TRACE_CAPACITY)
        env[both_cpu_ab.TRACE] = json.dumps(trace)
    if a.draft_top_prob:
        env["SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH"] = os.path.join(a.out_dir, "draft-top-prob")
    return env


def server_argv(arm: str) -> list[str]:
    # The production arms warm up as production does (ServerArgs.prod(): dsv41_decode_paths too).
    decode_warmup = arm in both_cpu_ab.PROD_ARMS
    return [*arm_env.ServerArgs(port=PORT, dspark=True, decode_warmup=decode_warmup).argv(),
            *both_cpu_ab.server_flags(arm)]


def resolve_env(a: argparse.Namespace, keys: list[str]) -> dict:
    """The server's env as resolved-env.json records it: the caller's ``keys`` (an arm's assignments) under the
    capture's env, and the keys the capture would override with another value, which the arm would not get."""
    own = capture_env(a)
    asked = {k: os.environ[k] for k in keys if k in os.environ}
    return {"env": asked | own, "argv": server_argv(a.arm),
            "overridden": sorted(k for k, v in asked.items() if k in own and own[k] != v)}


def then_generate(body_path: str, out_path: str, timeout_s: float = 600.0) -> int:
    """POST ``body_path`` to the server's /generate once; the reply goes to ``out_path``. 0 on HTTP 200."""
    with open(body_path, "rb") as f:
        body = f.read()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as r:
            reply = r.read()
    except OSError as error:
        print(f"then-generate failed: {error!r}")
        return 1
    with open(out_path, "wb") as f:
        f.write(reply)
    print(f"then-generate: {len(reply)} bytes -> {out_path}")
    return 0


def dropped_events(out_dir: str) -> dict[str, int]:
    """Per job-trace file that overflowed its buffer, the events dropped (the file's last line is {"dropped": N})."""
    out = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "events.*.jsonl"))):
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [x for x in f.read().decode(errors="replace").splitlines() if x.strip()]
        try:
            n = json.loads(lines[-1]).get("dropped", 0) if lines else 0
        except (ValueError, AttributeError):
            n = 0
        if n:
            out[os.path.basename(path)] = n
    return out


def post_run(a: argparse.Namespace) -> int:
    """The analyses over a completed run's trace. With --analysis-warn-only a refusal by layer_misses.py or spec_margin.py
    (a job-trace buffer overflow) is a warning with the dropped-event count, and the run still counts as completed."""
    events = os.path.join(a.out_dir, "events.*.jsonl")
    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "layer_misses.py"), events, "--json",
         os.path.join(a.out_dir, "layer_misses.json")]
    ).returncode
    if server_env(a.out_dir, a.top_k_only, a.scorer, a.arm, a.verify_split)["SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH"] == "1":
        rc = rc or subprocess.run(
            [sys.executable, os.path.join(HERE, "spec_margin.py"), events, "--json",
             os.path.join(a.out_dir, "spec_margin.json")]
        ).returncode
    if rc and a.analysis_warn_only:
        print(f"WARNING: the post-run analysis exited {rc}; the run itself completed (--analysis-warn-only). "
              f"Job-trace events dropped: {dropped_events(a.out_dir) or 'none'}", flush=True)
        rc = 0
    if a.verify_split:
        rc = rc or subprocess.run(
            [sys.executable, os.path.join(HERE, "verify_split.py"), a.out_dir, "--json",
             os.path.join(a.out_dir, "verify_split.json")]
        ).returncode
    return rc


def main() -> int:
    a = parse_args()
    if a.resolve_env is not None:
        print(json.dumps(resolve_env(a, [k for k in a.resolve_env.split(",") if k])))
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    env = os.environ | capture_env(a) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = server_argv(a.arm)
    raise_on_signal()
    with run_locks.take(DISK_LOCK), run_locks.take(arm_env.GPU_LOCK):
        server = None
        interrupted = False
        rc = 1
        try:
            with open(os.path.join(a.out_dir, "server.log"), "w") as log:
                server = launch_server(["taskset", "-c", arm_env.SERVER_CORES, *argv], env, a.out_dir, log)
            if not _healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
                return 1
            rc = subprocess.run(["taskset", "-c", arm_env.DRIVER_CORES, *client_argv(a)], cwd=REPO).returncode
            if rc == 0 and a.then_generate:
                rc = then_generate(a.then_generate, os.path.join(a.out_dir, "then_generate.json"))
        except BaseException:
            interrupted = True
            raise
        finally:
            if server is not None:
                # Aborted (the wrapper's load watchdog): give the launcher 30 s to shut down, then kill the group.
                stop_server(server, 30 if interrupted else 300)
    if rc != 0:
        return rc
    rc = post_run(a)
    summary = counter_summary(os.path.join(a.out_dir, "server.log"))
    with open(os.path.join(a.out_dir, "counters.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("counters", json.dumps(summary))
    return rc


if __name__ == "__main__":
    sys.exit(main())
