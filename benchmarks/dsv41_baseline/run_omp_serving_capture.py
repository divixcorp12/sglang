"""One opt-in serving arm: broad OpenMP initialization, restored launch affinity.

Retains the saved B split/NUMA/thread configuration. Reads audited runtime counters
from /proc/PID/mem outside the server and gates a root scheduler-only Nsight capture
at the timed request. CUDA node tracing is deliberately absent from this arm.
"""
import argparse
import ctypes
import mmap
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "diagnostic_omp_bootstrap", ROOT / "test/manual/dsv41/omp_bootstrap/sitecustomize.py")
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
OFFSETS, RUNTIME_SHA = _bootstrap.OFFSETS, _bootstrap.RUNTIME_SHA


def select_engine_leader(rows):
    candidates = {(r["pid"], t["tid"]) for r in rows[-4:] for t in r["expert_threads"]
                  if t["comm"] == "exl3-cpu-exp0"}
    if len({pid for pid, _ in candidates}) != 1:
        raise RuntimeError("expected one owned node 0 CPU engine")
    # OpenMP workers inherit the caller's comm. The engine starts before it
    # creates its team, so the oldest matching TID is the engine leader.
    return min(candidates)


def sample_runtime(directory, out, counters=True):
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
            # A zombie retains its comm/stat while Linux has already removed its mappings.
            # It is a normal shutdown observation, not a changed runtime binary.
            if not maps:
                continue
            bases = [int(line.split("-", 1)[0], 16) for line in maps
                     if line.split()[2] == "00000000" and line.endswith(data["runtime"])]
            if len(bases) != 1:
                raise RuntimeError("runtime mapping differs in scheduler")
            values = {}
            if counters:
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
    parser.add_argument("--magic-trace", type=Path, help="Use Intel PT on node 0 engine leader instead of Nsight")
    parser.add_argument("--magic-host-module", help="Exact loaded JIT DSO basename owning the CPU engine; required if ambiguous")
    parser.add_argument("--no-instrumentation", action="store_true", help="compile metrics out; retain only independent forward trigger")
    parser.add_argument("--draft-pending-trigger-us", type=int, default=500)
    parser.add_argument("--draft-arrival-trigger-us", type=int, default=500)
    parser.add_argument("--draft-forward-trigger-us", type=int, default=5000)
    parser.add_argument("--magic-health-diagnostic", action="store_true",
                        help="Capture a warm-up after HTTP health, stop after snapshot; never a benchmark")
    parser.add_argument("--magic-steady-decode", action="store_true",
                        help="Retain formal warm-up gate; arm after 30s, 256 tokens and 100 stream updates in a long decode")
    args = parser.parse_args()
    if args.magic_health_diagnostic and not args.magic_trace:
        parser.error("magic-health-diagnostic requires magic-trace")
    if args.magic_steady_decode and (not args.magic_trace or args.magic_health_diagnostic):
        parser.error("magic-steady-decode requires magic-trace and excludes magic-health-diagnostic")
    if args.no_instrumentation and (not args.magic_trace or args.draft_pending_trigger_us or args.draft_arrival_trigger_us):
        parser.error("no-instrumentation requires magic-trace and zero pending/arrival thresholds")
    if not 10 <= args.seconds <= 120:
        parser.error("seconds must be in [10, 120]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = Path(sys.prefix) / "lib/python3.13/site-packages/torch/lib/libgomp.so.1"
    if hashlib.sha256(library.read_bytes()).hexdigest() != RUNTIME_SHA:
        raise SystemExit("runtime hash differs: re-audit before running")
    shutil.copytree(args.seed_build, output / "exl3-build")
    gate = output / ("diagnostic.start" if args.magic_health_diagnostic else "timed.start")
    # Keep this inode and size unchanged while the server maps the flag.
    trace_gate_path = output / "trace.gate"
    trace_gate_path.write_bytes(bytes(4))
    with trace_gate_path.open("r+b") as trace_gate_file:
        trace_gate = mmap.mmap(trace_gate_file.fileno(), 4)
    manifest_dir = output / "runtime-init"
    env = {**os.environ, "DSV41_OMP_BOOTSTRAP": "1",
           "DSV41_OMP_INIT_CPUS": ",".join(map(str, range(64))),
           "DSV41_OMP_LIBRARY": str(library), "DSV41_OMP_MANIFEST_DIR": str(manifest_dir),
           "DSV41_TIMED_START_FILE": str(gate), "DSV41_MAX_SESSIONS": "1"}
    if args.magic_health_diagnostic:
        env["DSV41_DIAGNOSTIC_STOP_FILE"] = str(output / "diagnostic.stop")
    if args.magic_steady_decode:
        env.pop("DSV41_DIAGNOSTIC_STOP_FILE", None)
        env["DSV41_STEADY_TRACE_READY_FILE"] = str(output / "steady-decode.ready.json")
        env["DSV41_STEADY_TRACE_STOP_FILE"] = str(output / "steady-decode.stop")
    # These are inherited by the server; existing capture adds its own prefixes.
    env["SGLANG_CPU_EXPERT_TRACE_GATE"] = str(trace_gate_path)
    if not args.no_instrumentation:
        env.update(SGLANG_CPU_EXPERT_TRACE_GATE=str(trace_gate_path),
               SGLANG_EXL3_CPU_SCRATCH_TRACE_PREFIX=str(output / "scratch"))
    command = [sys.executable, str(Path(__file__).with_name("run_stall_capture.py")),
               "--reference", str(args.reference), "--output", str(output),
               "--port", str(args.port), "--no-nsys"]
    command += (["--no-instrumentation"] if args.no_instrumentation else
                ["--worker-phases", "--worker-min-us", "0", "--worker-capacity", "262144", "--job-capacity", "1048576"])
    if args.magic_trace:
        command += ["--draft-arrival-trigger-us", str(args.draft_arrival_trigger_us),
                    "--draft-pending-trigger-us", str(args.draft_pending_trigger_us),
                    "--draft-forward-trigger-us", str(args.draft_forward_trigger_us)]
    report = Path("/mnt/nvme1/dsv41-nsys") / (output.name + "-scheduler")
    profile_command = ["sudo", "-n", "/usr/local/sbin/nsys-profile", "profile",
                       "--trace=none", "--sample=none", "--cpuctxsw=system-wide",
                       "--ftrace=sched/sched_switch,sched/sched_wakeup",
                       "--duration=" + str(args.seconds), "--force-overwrite=true",
                       "--output=" + str(report), "/usr/bin/sleep", str(args.seconds)]
    (output / "omp-command.json").write_text(json.dumps(dict(command=command,
        diagnostic_env={k: v for k, v in env.items() if k.startswith(("DSV41_OMP", "DSV41_STEADY")) or
                        k in ("DSV41_TIMED_START_FILE", "SGLANG_CPU_EXPERT_TRACE_GATE",
                              "SGLANG_EXL3_CPU_SCRATCH_TRACE_PREFIX")},
        scheduler_command=profile_command), indent=2) + "\n")
    profiler = None
    master = slave = None
    gate_opened = False
    profiler_ready = False
    profiler_stop_requested = False
    workload_started = False
    workload_started_at = capture_started_at = None
    profile_text = ""
    with (output / "driver.log").open("w") as log, \
         (output / "runtime-samples.jsonl").open("w") as samples, \
         (output / "scheduler.log").open("w") as profile_log:
        arm = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            while arm.poll() is None or (profiler is not None and profiler.poll() is None):
                if not args.no_instrumentation or profiler is None:
                    sample_runtime(manifest_dir, samples, counters=not args.no_instrumentation)
                ready = any('"label": "server_ready"' in p.read_text()
                            for p in output.glob("servers/*/*/boundary-samples.jsonl"))
                if args.magic_health_diagnostic:
                    ready = "stall-cpu-s2 healthy" in (output / "driver.log").read_text()
                if ready and profiler is None:
                    if args.magic_trace:
                        # The bootstrap manifests prove ownership before selecting a TID.
                        rows = [json.loads(line) for line in (output / "runtime-samples.jsonl").read_text().splitlines()]
                        pid, tid = select_engine_leader(rows)
                        # Optimized JIT code is a dlopen DSO; resolve the uprobe address from its own mapping.
                        mappings = (Path("/proc") / str(pid) / "maps").read_text().splitlines()
                        addresses = {}
                        for line in mappings:
                            fields = line.split()
                            if len(fields) < 6 or fields[2] != "00000000" or "expert_stream_host_exl3_" not in fields[-1]:
                                continue
                            if args.magic_host_module and Path(fields[-1]).name != args.magic_host_module:
                                continue
                            symbols = subprocess.check_output(["nm", "-D", "--defined-only", fields[-1]], text=True)
                            for symbol in symbols.splitlines():
                                if symbol.endswith(" sglang_draft_delay_trigger"):
                                    addresses[fields[-1]] = int(fields[0].split("-")[0], 16) + int(symbol.split()[0], 16)
                        if len(addresses) != 1:
                            raise RuntimeError(f"expected one trigger DSO, select --magic-host-module: {list(addresses)}")
                        trigger_module, address = next(iter(addresses.items()))
                        # magic-trace treats addr: as an ELF address in /proc/TID/exe,
                        # and adds that executable's PIE load bias. Undo it for this DSO address.
                        exe = (Path("/proc") / str(pid) / "exe").resolve()
                        with exe.open("rb") as binary:
                            header = binary.read(64)
                            selected_address = address
                            if struct.unpack_from("<H", header, 16)[0] == 3:
                                phoff = struct.unpack_from("<Q", header, 32)[0]
                                size,count = struct.unpack_from("<HH", header, 54)
                                binary.seek(phoff)
                                entries = [binary.read(size) for _ in range(count)]
                                base_vaddr = min(struct.unpack_from("<Q", e, 16)[0] for e in entries
                                                 if struct.unpack_from("<I",e,0)[0] == 1)
                                exe_base = min(int(line.split("-",1)[0],16) for line in mappings
                                               if line.split()[-1] == str(exe))
                                selected_address -= exe_base - base_vaddr
                        profile_command = [str(args.magic_trace.resolve()), "attach", "-pid", str(tid),
                            "-trigger", "addr:" + hex(selected_address), "-snapshot-size", "4M",
                            "-working-directory", str(output / "magic-work"),
                            "-output", str(output / "draft-delay.fxt.gz")]
                        (output / "magic-command.json").write_text(json.dumps(dict(command=profile_command,
                            pid=pid, tid=tid, trigger_address=address, trigger_module=trigger_module), indent=2) + "\n")
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
                        profiler_ready |= ("[ Attached." in profile_text if args.magic_trace else "Collecting data" in profile_text)
                if profiler_ready and not workload_started:
                    stamp = time.monotonic_ns()
                    gate.write_text(json.dumps(dict(ns=stamp, trace_gate=str(trace_gate_path),
                                                   trigger_closed_for_decode_burnin=args.magic_steady_decode)) + "\n")
                    workload_started = True
                    workload_started_at = time.monotonic()
                    print("profiler attached; workload gate open", flush=True)
                steady_marker = output / "steady-decode.ready.json"
                if workload_started and not gate_opened and (not args.magic_steady_decode or steady_marker.exists()):
                    admission = json.loads(steady_marker.read_text()) if args.magic_steady_decode else {}
                    stamp = time.monotonic_ns()
                    if args.magic_steady_decode:
                        if (not admission.get("formal_warmup_passed") or admission["decode_seconds"] < 30 or
                                admission["completion_tokens"] < 256 or admission["progress_updates"] < 100):
                            raise RuntimeError("invalid steady decode admission evidence")
                        (output / "steady-trace-armed.json").write_text(json.dumps(dict(
                            ns=stamp, admission=admission, trace_gate=str(trace_gate_path)), indent=2) + "\n")
                    ctypes.c_uint32.from_buffer(trace_gate).value = 1
                    gate_opened = True
                    capture_started_at = time.monotonic()
                    print("trigger gate open" + (" after steady decode burn-in" if args.magic_steady_decode else ""), flush=True)
                if profiler is not None and not profiler_ready and (
                        profiler.poll() is not None or time.monotonic() - profile_started > 45):
                    raise RuntimeError("scheduler profiler did not become ready; gate remains closed")
                if (args.magic_steady_decode and workload_started and not gate_opened and
                        (arm.poll() is not None or profiler.poll() is not None or time.monotonic() - workload_started_at > 600)):
                    raise RuntimeError("steady decode admission failed; trigger gate remains closed")
                if (args.magic_trace and gate_opened and profiler.poll() is None and not profiler_stop_requested
                        and "Snapshot taken" not in profile_text and time.monotonic() - capture_started_at > args.seconds):
                    profiler_stop_requested = True
                    # v1.2.4 SIGINT alone can yield AUX bookkeeping without PT
                    # bytes. Snapshot only this collector's recorder first.
                    helper_spec = importlib.util.spec_from_file_location("magic_capture",
                        ROOT / "tools/skills/magic-trace/scripts/capture.py")
                    sys.path.insert(0, str(ROOT / "tools/skills/magic-trace/scripts"))
                    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
                    recorder = helper.owned_perf(profiler.pid, output / "magic-work/perf.data")
                    os.kill(recorder, signal.SIGUSR2)
                    time.sleep(.5)
                    profiler.send_signal(signal.SIGINT)  # bounded fallback snapshot if no threshold fired
                if args.magic_health_diagnostic and gate_opened and profiler.poll() is not None:
                    (output / "diagnostic.stop").touch()
                if args.magic_steady_decode and gate_opened and profiler.poll() is not None:
                    (output / "steady-decode.stop").touch()
                time.sleep(.25)
        finally:
            if arm.poll() is None:
                os.killpg(arm.pid, signal.SIGTERM)
                arm.wait(timeout=180)
            if profiler is not None and profiler.poll() is None:
                if not gate_opened:
                    profiler.send_signal(signal.SIGINT)
                profiler.wait(timeout=150)
            if master is not None:
                os.close(master)
            trace_gate.close()
        if args.magic_trace and gate_opened and arm.returncode == 0 and not args.no_instrumentation:
            info = json.loads((output / "magic-command.json").read_text())
            clock_path = output / f"events.{info['pid']}.draft-clock-end.json"
            clock_env = {**env, "PYTHONPATH": str(ROOT/"python"),
                         "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(output/"events")}
            with (output/"clock-end.log").open("w") as clock_log:
                subprocess.run(["flock", "/data/models/slang/nvfp4-work/cc-gpu.lock", "taskset", "-c", "32-63",
                    sys.executable, str(ROOT/"benchmarks/dsv41_baseline/draft_clock_anchor.py"), str(clock_path)],
                    env=clock_env, stdout=clock_log, stderr=subprocess.STDOUT, check=True, timeout=180)
        status = dict(arm_status=arm.returncode, profiler_status=profiler.returncode if profiler else None,
                      diagnostic_only=args.magic_health_diagnostic or args.magic_steady_decode,
                      steady_decode=args.magic_steady_decode,
                      instrumentation=not args.no_instrumentation,
                      gate_opened=gate_opened, scheduler_report=str(report) + ".nsys-rep")
        if args.magic_trace:
            trace = output / "draft-delay.fxt.gz"
            status["trace_bytes"] = trace.stat().st_size if trace.exists() else 0
            status["fallback_snapshot"] = profiler_stop_requested
        (output / "omp-status.json").write_text(json.dumps(status) + "\n")
        print(json.dumps(status), flush=True)
        if arm.returncode or profiler is None or profiler.returncode or not gate_opened:
            raise SystemExit(1)
        if args.magic_trace and status["trace_bytes"] <= 100:
            raise SystemExit("magic-trace decoded an empty timeline")
        if args.magic_trace:
            subprocess.run([sys.executable, str(ROOT / "tools/skills/magic-trace/scripts/verify_trace.py"),
                            str(output / "draft-delay.fxt.gz"), "--perf-data", str(output / "magic-work/perf.data"),
                            "--log", str(output / "scheduler.log"), "--output", str(output / "trace-verification.json")], check=True)


if __name__ == "__main__":
    main()
