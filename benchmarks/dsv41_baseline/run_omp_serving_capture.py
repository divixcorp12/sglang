"""One opt-in serving arm: broad OpenMP initialization, restored launch affinity.

Retains the saved B split/NUMA/thread configuration. Reads audited runtime counters
from /proc/PID/mem outside the server and gates a root scheduler-only Nsight capture
at the timed request. CUDA node tracing is deliberately absent from this arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/manual/dsv41/omp_bootstrap"))
from sitecustomize import OFFSETS, RUNTIME_SHA


def sample_runtime(directory, out):
    """Only processes proven to belong to this unique bootstrap directory."""
    manifests = {}
    for path in directory.glob("*.json"):
        data = json.loads(path.read_text())
        manifests[data["pid"]] = data
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if not (proc / "comm").read_text().startswith("sglang::sched"):
                continue
            ancestor = proc
            for _ in range(8):
                data = manifests.get(int(ancestor.name))
                if data:
                    break
                parent = (ancestor / "stat").read_text().rsplit(")", 1)[1].split()[1]
                if parent == "0":
                    break
                ancestor = Path("/proc") / parent
            else:
                data = None
            if not data:
                continue
            maps = (proc / "maps").read_text().splitlines()
            bases = [int(line.split("-", 1)[0], 16) for line in maps
                     if line.split()[2] == "00000000" and line.endswith(data["runtime"])]
            if len(bases) != 1:
                raise RuntimeError("runtime mapping differs in scheduler")
            fd = os.open(proc / "mem", os.O_RDONLY)
            try:
                values = {name: int.from_bytes(os.pread(fd, 8, bases[0] + offset), "little")
                          for name, offset in OFFSETS.items()}
            finally:
                os.close(fd)
            threads = [dict(tid=int(p.name), comm=(p / "comm").read_text().strip(),
                            affinity=sorted(os.sched_getaffinity(int(p.name))))
                       for p in (proc / "task").iterdir()
                       if (p / "comm").read_text().startswith("exl3-")]
            out.write(json.dumps(dict(ns=time.monotonic_ns(), pid=int(proc.name),
                                      counters=values, expert_threads=threads)) + "\n")
            out.flush()
        except (FileNotFoundError, ProcessLookupError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-build", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30034)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 120:
        parser.error("seconds must be in [10, 120]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = Path(sys.prefix) / "lib/python3.13/site-packages/torch/lib/libgomp.so.1"
    if hashlib.sha256(library.read_bytes()).hexdigest() != RUNTIME_SHA:
        raise SystemExit("runtime hash differs: re-audit before running")
    shutil.copytree(args.seed_build, output / "exl3-build")
    gate = output / "timed.start"
    manifest_dir = output / "runtime-init"
    env = {**os.environ, "DSV41_OMP_BOOTSTRAP": "1",
           "DSV41_OMP_INIT_CPUS": ",".join(map(str, range(64))),
           "DSV41_OMP_LIBRARY": str(library), "DSV41_OMP_MANIFEST_DIR": str(manifest_dir),
           "DSV41_TIMED_START_FILE": str(gate), "DSV41_MAX_SESSIONS": "1"}
    # These are inherited by the server; existing capture adds its own prefixes.
    env.update(SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX=str(output / "hold"),
               SGLANG_CPU_EXPERT_HOLD_TRACE_CAPACITY="1048576")
    command = [sys.executable, str(Path(__file__).with_name("run_stall_capture.py")),
               "--reference", str(args.reference), "--output", str(output),
               "--port", str(args.port), "--no-nsys", "--worker-phases"]
    report = Path("/mnt/nvme1/dsv41-nsys") / (output.name + "-scheduler")
    profile_command = ["sudo", "-n", "/usr/local/sbin/nsys-profile", "profile",
                       "--trace=none", "--sample=none", "--cpuctxsw=system-wide",
                       "--ftrace=sched/sched_switch,sched/sched_wakeup",
                       "--duration=" + str(args.seconds), "--force-overwrite=true",
                       "--output=" + str(report), "/usr/bin/sleep", str(args.seconds)]
    (output / "omp-command.json").write_text(json.dumps(dict(command=command,
        diagnostic_env={k: v for k, v in env.items() if k.startswith("DSV41_OMP") or
                        k in ("DSV41_TIMED_START_FILE", "SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX",
                              "SGLANG_CPU_EXPERT_HOLD_TRACE_CAPACITY")},
        scheduler_command=profile_command), indent=2) + "\n")
    profiler = None
    master = slave = None
    gate_opened = False
    profiler_ready = False
    profile_text = ""
    with (output / "driver.log").open("w") as log, \
         (output / "runtime-samples.jsonl").open("w") as samples, \
         (output / "scheduler.log").open("w") as profile_log:
        arm = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            while arm.poll() is None or (profiler is not None and profiler.poll() is None):
                sample_runtime(manifest_dir, samples)
                ready = any('"label": "server_ready"' in p.read_text()
                            for p in output.glob("servers/*/*/boundary-samples.jsonl"))
                if ready and profiler is None:
                    master, slave = pty.openpty()
                    profiler = subprocess.Popen(profile_command, stdout=slave, stderr=slave)
                    os.close(slave)
                    slave = None
                    os.set_blocking(master, False)
                    profile_started = time.monotonic()
                    print("server ready; starting scheduler capture", flush=True)
                if master is not None and select.select([master], [], [], 0)[0]:
                    try:
                        chunk = os.read(master, 65536).decode(errors="replace")
                    except OSError:
                        chunk = ""
                    if chunk:
                        profile_text += chunk
                        profile_log.write(chunk); profile_log.flush()
                        profiler_ready |= "Collecting data" in profile_text
                if profiler_ready and not gate_opened:
                    gate.write_text(json.dumps(dict(ns=time.monotonic_ns())) + "\n")
                    gate_opened = True
                    print("scheduler collecting; timed gate open", flush=True)
                if profiler is not None and not gate_opened and (
                        profiler.poll() is not None or time.monotonic() - profile_started > 45):
                    raise RuntimeError("scheduler profiler did not become ready; gate remains closed")
                time.sleep(.25)
        finally:
            if arm.poll() is None:
                os.killpg(arm.pid, signal.SIGTERM)
                arm.wait(timeout=180)
            if profiler is not None and profiler.poll() is None:
                profiler.wait(timeout=150)
            if master is not None:
                os.close(master)
        status = dict(arm_status=arm.returncode, profiler_status=profiler.returncode if profiler else None,
                      gate_opened=gate_opened, scheduler_report=str(report) + ".nsys-rep")
        (output / "omp-status.json").write_text(json.dumps(status) + "\n")
        print(json.dumps(status), flush=True)
        if arm.returncode or profiler is None or profiler.returncode or not gate_opened:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
