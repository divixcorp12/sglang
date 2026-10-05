"""DSpark graphed verify, D2-3 (plan 2026-10-05-dsv41-dspark-graph-d2-3-graphed-verify Task 6).

Three arms, one Engine each through trace_corpus.py, the same sessions, the hybrid draft (§33.4), CPU experts off:
  eager     the verify eager (decode disabled), as §33.4's hybrid arm;
  graphed   the verify in the breakable decode graph at W miss lanes; an overflowed verify is re-run eagerly;
  reverify  graphed, with every verify re-run eagerly (SGLANG_TEST_DSPARK_FORCE_REVERIFY): its text must equal eager's.
Run on divix01 from a worktree at the pushed branch, holding rowimg-disk.lock then cc-gpu.lock:
  python analysis/dsv41-drive/dspark/graphed_verify.py OUTDIR [ARM ...]
"""

import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ab_cpu_draft import ARMS, COMMON, DRAFT, PYTHON, REPO, SESSIONS  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

MISS_LANES = os.environ.get("D23_MISS_LANES", "8")
# COMMON turns the recipe's in-graph expert path off for an eager verify; a graphed verify keeps the recipe's.
EAGER_ONLY = (
    "SGLANG_MOE_EXPERT_GRAPH_GATHER",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
    "SGLANG_MOE_EXPERT_FUSED_PLAN",
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS",
)
GRAPHED = {
    **{k: v for k, v in COMMON.items() if k not in EAGER_ONLY},
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": MISS_LANES,
    "SGLANG_RAGGED_VERIFY_MODE": "static",
}
D23_ARMS = {
    "eager": ({**COMMON, **ARMS["hybrid"]}, False),
    "graphed": ({**GRAPHED, **ARMS["hybrid"]}, True),
    "reverify": ({**GRAPHED, **ARMS["hybrid"], "SGLANG_TEST_DSPARK_FORCE_REVERIFY": "1"}, True),
}


def run(arm: str, outdir: str, n: int, new_tokens: int) -> int:
    overrides, graphs = D23_ARMS[arm]
    overrides = {
        **overrides,
        "SGLANG_DSPARK_DEBUG_DUMP": "target_verify_gpu_time",
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(outdir, f"{arm}.metrics.jsonl"),
    }
    env = os.environ | arm_env.arm_env(overrides)
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    env.setdefault("OMP_NUM_THREADS", "16")
    cmd = [
        PYTHON, os.path.join(REPO, "scripts", "dsv41", "trace_corpus.py"),
        "--model", arm_env.MODEL_PATH,
        "--sessions", SESSIONS,
        "--n", str(n),
        "--skip", os.environ.get("AB_SKIP", "8"),
        "--prompt-tokens", "256",
        "--new-tokens", str(new_tokens),
        "--stop-at-eos",
        "--log-level", "info",
        "--dspark", DRAFT,
        "--out", os.path.join(outdir, f"{arm}.json"),
    ] + (["--graphs"] if graphs else [])
    print(f"=== {arm}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(outdir, f"{arm}.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode


def _last_metrics(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def summarize(outdir: str) -> dict:
    """Per arm: tok/s, accept length, verify GPU ms; for graphed arms the overflow, re-verify rate and text parity."""
    reports = {}
    for arm in D23_ARMS:
        path = os.path.join(outdir, f"{arm}.json")
        if os.path.exists(path):
            with open(path) as f:
                reports[arm] = json.load(f)
    eager_texts = [s.get("output_text") for s in reports["eager"]["per_session"]] if "eager" in reports else None
    summary = {}
    for arm, report in reports.items():
        sessions = report["per_session"]
        records = (report.get("dspark_info_record") or {}).get("records", [])
        verify_ms = sorted(r["target_verify_gpu_ms"] for r in records if r.get("target_verify_gpu_ms") is not None)
        entry = {
            "mean_decode_tok_s": report["mean_decode_tok_s"],
            "accept_length": sum(s["completion_tokens"] for s in sessions) / sum(s["spec_verify_ct"] for s in sessions),
            "verify_ms": {
                "n": len(verify_ms),
                "mean": statistics.fmean(verify_ms) if verify_ms else None,
                "p50": verify_ms[len(verify_ms) // 2] if verify_ms else None,
                "p95": verify_ms[min(len(verify_ms) - 1, int(0.95 * len(verify_ms)))] if verify_ms else None,
            },
        }
        if eager_texts is not None and arm != "eager":
            entry["text_matches_eager"] = sum(
                s.get("output_text") == text for s, text in zip(sessions, eager_texts)
            )
        metrics = _last_metrics(os.path.join(outdir, f"{arm}.metrics.jsonl"))
        counters = (metrics or {}).get("counters", {})
        graphed = counters.get("graphed_verify")
        if graphed:
            verifies = graphed["graphed_verify_ct"]
            overflow = counters["residency_gpu"]["gather_overflow"]
            rates = [layer / verifies for layer in overflow]
            entry["graphed_verify_ct"] = verifies
            entry["reverify_rate"] = graphed["verify_overflow_ct"] / verifies
            entry["layer_overflow_rate"] = {"mean": sum(rates) / len(rates), "max": max(rates)}
            entry["insertion_truncated"] = sum(counters["residency_gpu"]["insertion_truncated"])
        summary[arm] = entry
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    outdir = sys.argv[1]
    arms = sys.argv[2:] or list(D23_ARMS)
    n = int(os.environ.get("AB_SESSIONS", "8"))
    new_tokens = int(os.environ.get("AB_NEW_TOKENS", "128"))
    os.makedirs(outdir, exist_ok=True)
    for arm in arms:
        rc = run(arm, outdir, n, new_tokens)
        print(f"{arm}: rc={rc}", flush=True)
        if rc:
            sys.exit(rc)
    print(json.dumps(summarize(outdir), indent=2), flush=True)


if __name__ == "__main__":
    main()
