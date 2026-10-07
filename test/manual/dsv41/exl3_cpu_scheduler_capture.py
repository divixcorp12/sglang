"""Gate a user CPU replay inside a bounded, privileged Nsight scheduler capture.

Uses the existing NOPASSWD nsys-profile collector. Does not alter host permissions,
run benchmark code as root, inject CUDA tracing, or change production settings.
"""
import argparse
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **json.loads(args.reference.read_text())}
    for k in tuple(env):
        if k.startswith(("SGLANG_DSV41_EXPERT_JOB_TRACE", "SGLANG_EXL3_CPU_WORKER_TRACE")) or k == "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE":
            env.pop(k)
    root = Path(__file__).resolve().parents[3]
    env.update(PYTHONPATH=str(root / "python"), OMP_NUM_THREADS="10", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MAX_JOBS="4", EXL3_MOE_CPU_PIN="0", SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=str(args.output / "unused-events"),
               SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX=str(args.output / "worker-phases"),
               SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US="0", SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY="131072")
    ready, start = args.output / "ready.json", args.output / "start"
    command = [sys.executable, str(Path(__file__).with_name("exl3_cpu_stall_measure.py")), "arm", "--measurement", "wake",
               "--group", "1", "--reps", "100", "--warmups", "5", "--library", str(args.library),
               "--ready-file", str(ready), "--start-file", str(start), "--output", str(args.output / "arm.json")]
    capture = ["sudo", "-n", "/usr/local/sbin/nsys-profile", "profile", "--trace=none", "--sample=none",
               "--cpuctxsw=system-wide", "--ftrace=sched/sched_switch,sched/sched_wakeup", "--duration=10",
               "--force-overwrite=true", "--output=" + str(args.output / "scheduler"), "/usr/bin/sleep", "10"]
    metadata = {"arm_command": command, "capture_command": capture, "start_ns": time.monotonic_ns(),
                "epoch_ns": time.time_ns(), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    collector = None
    with (args.output / "arm.log").open("w") as arm_log, (args.output / "collector.log").open("w") as collector_log:
        arm = subprocess.Popen(command, env=env, stdout=arm_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while not ready.exists():
                if arm.poll() is not None:
                    raise RuntimeError("arm exited before readiness; inspect arm.log")
                if time.monotonic() > deadline:
                    raise RuntimeError("arm readiness timed out")
                time.sleep(.1)
            # A pipe/file buffers Nsight's readiness line until profiling has stopped.
            # A PTY makes it line-buffered; stream the bytes into the persistent log.
            master, slave = pty.openpty()
            collector = subprocess.Popen(capture, stdout=slave, stderr=subprocess.STDOUT)
            os.close(slave)
            collected = ""
            deadline = time.monotonic() + 15
            while "Collecting data" not in collected:
                if collector.poll() is not None:
                    raise RuntimeError("collector exited before readiness; inspect collector.log")
                if time.monotonic() > deadline:
                    raise RuntimeError("collector readiness timed out")
                if select.select([master], [], [], .1)[0]:
                    chunk = os.read(master, 65536).decode(errors="replace")
                    collected += chunk
                    collector_log.write(chunk)
                    collector_log.flush()
            metadata["release_ns"] = time.monotonic_ns()
            start.touch()
            metadata["arm_status"] = arm.wait(timeout=30)
            metadata["collector_status"] = collector.wait(timeout=45)
            while select.select([master], [], [], 0)[0]:
                try:
                    chunk = os.read(master, 65536).decode(errors="replace")
                except OSError:
                    break
                if not chunk:
                    break
                collector_log.write(chunk)
            os.close(master)
            if metadata["arm_status"] or metadata["collector_status"]:
                raise RuntimeError("capture/arm failed; inspect logs")
        finally:
            if arm.poll() is None:
                arm.terminate()
                arm.wait(timeout=5)
            if collector is not None and collector.poll() is None:
                collector.terminate()
                collector.wait(timeout=10)
            metadata["end_ns"] = time.monotonic_ns()
            (args.output / "capture-command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
