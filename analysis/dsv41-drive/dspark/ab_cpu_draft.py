"""DSpark A/B (plans 2026-10-02-dsv41-dspark-cpu-draft Task 7, 2026-10-05-dsv41-dspark-port Task 7): the draft's
experts resident in VRAM vs on the CPU.

Eager (the EXL3 gate refuses speculation under a decode graph), one Engine per arm through trace_corpus.py, the same
sessions. Run on divix01 from a worktree at the pushed branch, holding rowimg-disk.lock then cc-gpu.lock:
  python analysis/dsv41-drive/dspark/ab_cpu_draft.py OUTDIR [ARM ...]
ARM is resident, hybrid or routes (resident plus the draft route probe).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

PYTHON = "/data/models/slang/.venv/bin/python"
DRAFT = "/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-dspark-draft"
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
RESIDENT = "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json"
# The 2026-09-24 DSpark launch's eager overrides, less the variables §32.7 retired.
COMMON = {
    # The base recipe runs the target's CPU experts in the decode graph's copy wait; the gate refuses them with
    # speculation, which is eager here.
    "SGLANG_DSV41_CPU_EXPERTS": "0",
    # The optimized EXL3 CPU build, the only one exporting a CpuExpertKernel (the hybrid arm's draft runs on it), in
    # both arms so they load the same extension.
    "SGLANG_EXL3_CPU_ACT_RESIDUAL": "1",
    "SGLANG_EXL3_CPU_ACT_BLOCK": "128",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "0",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "0",
    "SGLANG_MOE_EXPERT_FUSED_PLAN": "0",
    "SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING": "0",
    # Device wait serves from the uring store, which the 2026-09-24 DSpark launch kept off.
    "SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT": "0",
    "SGLANG_SM120_FLASHMLA_BACKEND": "triton",
    # Layer-major prefill needs --max-running-requests 1; trace_corpus launches with 4.
    "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0",
    # Prefill fills are the RAM-miss service's reads, which only run with graph gather (option C).
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS": "0",
}
ARMS = {
    "resident": {"SGLANG_MOE_HOT_GPU_MB": "7168"},
    # Each stage's top-32 experts stay on the GPU (resident-top32.json, from the routes arm's sessions 0-7); the
    # other 288 free 4,872 MiB for the target's hot cache.
    "hybrid": {
        "SGLANG_MOE_HOT_GPU_MB": "12040",
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        # NUMA node 0's physical cores 6-17 (node 0 is 0-17,36-53): the target's CPU experts are off, so the
        # recipe's CPU-expert cores are free.
        "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "6-17",
        "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": RESIDENT,
        "EXL3_MOE_CPU_PIN": "0",
    },
    # The resident arm with the draft's topk ids logged to OUTDIR/routes.jsonl (draft_routes_report.py reads it).
    "routes": {"SGLANG_MOE_HOT_GPU_MB": "7168"},
}


def run(arm: str, outdir: str, n: int, new_tokens: int) -> int:
    overrides = {**COMMON, **ARMS[arm]}
    if arm == "routes":
        overrides["SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH"] = os.path.join(outdir, "routes.jsonl")
    # arm_env holds only the recipe; without the inherited PATH the extension build cannot find cc1plus.
    env = os.environ | arm_env.arm_env(overrides)
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    env.setdefault("OMP_NUM_THREADS", "16")
    cmd = [
        PYTHON, os.path.join(REPO, "scripts", "dsv41", "trace_corpus.py"),
        "--model", arm_env.MODEL_PATH,
        "--sessions", SESSIONS,
        "--n", str(n),
        "--skip", os.environ.get("AB_SKIP", "0"),
        "--prompt-tokens", "256",
        "--new-tokens", str(new_tokens),
        "--stop-at-eos",
        "--log-level", "info",
        "--dspark", DRAFT,
        "--out", os.path.join(outdir, f"{arm}.json"),
    ]
    print(f"=== {arm}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(outdir, f"{arm}.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode


def main():
    outdir = sys.argv[1]
    arms = sys.argv[2:] or list(ARMS)
    n = int(os.environ.get("AB_SESSIONS", "8"))
    new_tokens = int(os.environ.get("AB_NEW_TOKENS", "128"))
    os.makedirs(outdir, exist_ok=True)
    for arm in arms:
        rc = run(arm, outdir, n, new_tokens)
        print(f"{arm}: rc={rc}", flush=True)
        if rc:
            sys.exit(rc)


if __name__ == "__main__":
    main()
