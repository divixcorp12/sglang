"""Repeat a saved diagnostic arm with job resources, /proc samples and root CPU tracing.

Run on divix01 from a pushed private worktree. The reference is a previous capture-command.json.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import generations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30029)
    parser.add_argument("--no-nsys", action="store_true", help="collect job resources and /proc only")
    parser.add_argument("--worker-phases", action="store_true", help="compile EXL3 per-worker phase diagnostics")
    parser.add_argument("--worker-min-us", type=int, default=6000)
    parser.add_argument("--worker-capacity", type=int, default=131072)
    parser.add_argument("--job-capacity", type=int, default=524288)
    parser.add_argument("--draft-pending-trigger-us", type=int, default=0)
    parser.add_argument("--draft-arrival-trigger-us", type=int, default=0)
    parser.add_argument("--draft-forward-trigger-us", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.worker_min_us <= 1000000000:
        parser.error("worker-min-us must be in [0, 1000000000]")
    if any(not 1 <= c <= 1048576 for c in (args.worker_capacity, args.job_capacity)):
        parser.error("trace capacities must be in [1, 1048576]")
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "sampler.stop"
    if marker.exists() or (args.output / "capture-command.json").exists():
        raise SystemExit("refusing to overwrite an earlier capture")
    reference = json.loads(args.reference.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD:python"], cwd=root, text=True).strip()
    generations.register(tree, "expert-stall-resources-" + head[:10])
    env = {**os.environ, **reference["harness_env"], "PYTHONPATH": str(root / "python"),
           "DSV41_WORKTREE": str(root), "DSV41_RUN_ROOT": str(args.output), "EXPECT_SHA": head,
           "NSYS_SAMPLE": "none", "NSYS_CPUCTXSW": "none", "NSYS_SYSTEM_CPU": "1"}
    if args.no_nsys:
        env.update(NSYS_TRACE="0", NSYS_GPU_METRICS="0", NSYS_SYSTEM_CPU="0")
    prefix = str(args.output / "events")
    overrides = [arg for arg in reference["command"][4:]
                 if not arg.startswith(("SGLANG_MOE_HOT_METRICS_FILE=", "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=",
                                        "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE=",
                                        "SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY=",
                                        "SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX=",
                                        "SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US=",
                                        "SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY="))]
    overrides += ["SGLANG_MOE_HOT_METRICS_FILE=" + str(args.output / "metrics.jsonl"),
                  "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=" + prefix,
                  "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE=1",
                  "SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY=" + str(args.job_capacity)]
    overrides += ["SGLANG_DRAFT_ARRIVAL_TRIGGER_US=" + str(args.draft_arrival_trigger_us),
                  "SGLANG_DRAFT_PENDING_TRIGGER_US=" + str(args.draft_pending_trigger_us),
                  "SGLANG_DRAFT_FORWARD_TRIGGER_US=" + str(args.draft_forward_trigger_us)]
    if args.worker_phases:
        overrides += ["SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX=" + str(args.output / "worker-phases"),
                      "SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US=" + str(args.worker_min_us),
                      "SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY=" + str(args.worker_capacity),
                      "SGLANG_EXL3_BUILD_DIR=" + str(args.output / "exl3-build")]
    command = ["bash", str(root / "benchmarks/dsv41_baseline/run_arm.sh"), "stall-cpu-s2",
               str(args.port), *overrides]
    imported = subprocess.check_output([sys.executable, "-c", "import sglang; print(sglang.__file__)"],
                                       cwd=root, env=env, text=True).strip()
    if Path(imported).resolve() != root / "python/sglang/__init__.py":
        raise SystemExit("unexpected sglang import: " + imported)
    metadata = {"head": head, "sglang_file": imported, "command": command,
                "harness_env": {k: env[k] for k in reference["harness_env"]},
                "extra_harness_env": {k: env[k] for k in ("NSYS_TRACE", "NSYS_GPU_METRICS", "NSYS_SYSTEM_CPU")},
                "reference": str(args.reference),
                "start_ns": time.monotonic_ns(), "epoch_ns": time.time_ns()}
    (args.output / "capture-command.json").write_text(json.dumps(metadata, indent=2))
    with open("/data/models/slang/nvfp4-work/rowimg-disk.lock", "w") as disk:
        print("waiting for rowimg-disk.lock", flush=True)
        fcntl.flock(disk, fcntl.LOCK_EX)
        with open("/data/models/slang/nvfp4-work/cc-gpu.lock", "w") as gpu:
            print("waiting for cc-gpu.lock availability", flush=True)
            while True:
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(gpu, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    time.sleep(5)
        sampler = subprocess.Popen(["taskset", "-c", "30", sys.executable,
                                    str(root / "benchmarks/dsv41_baseline/stall_sampler.py"),
                                    "--prefix", prefix, "--output", str(args.output / "system-samples.jsonl"),
                                    "--stop-file", str(marker)], env=env, cwd=root)
        try:
            status = subprocess.call(command, env=env, cwd=root)
        finally:
            marker.touch()
            sampler.wait(timeout=10)
        (args.output / "exit-status.json").write_text(json.dumps({"status": status,
                "end_ns": time.monotonic_ns(), "sampler_status": sampler.returncode}) + "\n")
        raise SystemExit(status)


if __name__ == "__main__":
    main()
