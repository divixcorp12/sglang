#!/usr/bin/env python3
"""Bounded attach with readiness, explicit perf fallback snapshot, and validation."""
import argparse
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import subprocess
import time

from resolve_trigger import resolve, target_identity
from verify_trace import sha256, verify


def owned_perf(collector_pid, raw_path):
    parents, recorders = {}, []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
            pid = int(proc.name)
            parents[pid] = int(status["PPid"])
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if status["Name"].strip() == "perf" and b"record" in argv:
                matches = False
                for fd in (proc / "fd").iterdir():
                    try:
                        matches |= os.readlink(fd) == str(raw_path)
                    except (FileNotFoundError, PermissionError):
                        pass
                if matches:
                    recorders.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = []
    for pid in recorders:
        parent = pid
        for _ in range(16):
            parent = parents.get(parent, 0)
            if parent == collector_pid:
                owned.append(pid)
                break
            if parent == 0:
                break
    if len(owned) != 1:
        raise ValueError(f"expected one owned perf recorder for {raw_path}, found {owned}")
    return owned[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--tid", type=int)
    parser.add_argument("--trigger")
    parser.add_argument("--module")
    parser.add_argument("--symbol")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--snapshot-size", default="4M")
    parser.add_argument("--decode-timeout", type=float, default=150)
    parser.add_argument("--multi-thread", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120 or not 1 <= args.decode_timeout <= 600:
        parser.error("seconds must be 1..120; decode-timeout 1..600")
    if args.trigger and args.symbol or args.module and not args.symbol:
        parser.error("use --trigger OR --symbol [--module]")
    tid = args.tid or args.pid
    if args.multi_thread and tid != args.pid:
        parser.error("multi-thread requires process PID as TID")
    tool = shutil.which(args.tool)
    if not tool:
        parser.error("magic-trace executable not found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw, trace = output / "raw", output / "trace.fxt.gz"
    raw.mkdir()
    info = dict(target=target_identity(args.pid, tid), tool=str(Path(tool).resolve()),
                tool_sha256=sha256(tool), started_monotonic_ns=time.monotonic_ns(),
                requested_mode="sampling" if args.sampling else "intel-pt", fallback_requested=False)
    trigger = args.trigger
    if args.symbol:
        info["trigger_resolution"] = resolve(args.pid, tid, args.module, args.symbol)
        trigger = info["trigger_resolution"]["selection"]
    command = [tool, "attach", "-pid", str(tid), "-working-directory", str(raw), "-output", str(trace)]
    command += ["-sampling"] if args.sampling else ["-snapshot-size", args.snapshot_size]
    if trigger:
        command += ["-trigger", trigger]
    if args.multi_thread:
        command += ["-multi-thread"]
    info["command"] = command
    info["tool_version"] = subprocess.check_output([tool, "version"], text=True, stderr=subprocess.STDOUT).strip()
    real_perf = os.environ.get("MAGIC_TRACE_REAL_PERF", "/usr/bin/perf")
    info["perf_version"] = subprocess.check_output([real_perf, "version"], text=True).strip()
    info["perf_adapter"] = os.environ.get("MAGIC_TRACE_PERF_PATH")
    master, slave = pty.openpty()
    process = None
    failure = None
    try:
        with (output / "capture.log").open("w") as log:
            process = subprocess.Popen(command, stdout=slave, stderr=slave, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            os.close(slave)
            slave = None
            os.set_blocking(master, False)
            began = time.monotonic()
            attached = snapshot_at = fallback_at = stopped_at = None
            tail = ""
            while True:
                if select.select([master], [], [], .05)[0]:
                    try:
                        chunk = os.read(master, 65536).decode(errors="replace")
                    except OSError:
                        chunk = ""
                    if chunk:
                        log.write(chunk); log.flush()
                        tail = (tail + chunk)[-1024 * 1024:]
                now = time.monotonic()
                if attached is None and "[ Attached." in tail:
                    attached = now
                    # Ensure the target process did not change between selection and attach.
                    current = target_identity(args.pid, tid)
                    if any(current[key] != info["target"][key] for key in
                           ("start_ticks", "thread_start_ticks", "executable_sha256")):
                        raise ValueError("target process identity changed")
                    if "trigger_resolution" in info:
                        match = re.search(r"@\s+(0x[0-9a-fA-F]+)", tail)
                        expected = info["trigger_resolution"]["runtime_trigger"]
                        if not match or int(match[1], 16) != expected:
                            raise ValueError("collector attach address differs from resolved trigger")
                    info["attached_monotonic_ns"] = time.monotonic_ns()
                    if args.ready_file:
                        with args.ready_file.open("x") as ready:
                            json.dump(dict(pid=args.pid, tid=tid, ns=time.monotonic_ns()), ready)
                    print(f"attached to {args.pid}/{tid}", flush=True)
                if snapshot_at is None and "Snapshot taken" in tail:
                    snapshot_at = now
                if process.poll() is not None:
                    # Drain remaining PTY data before closing the log.
                    while select.select([master], [], [], 0)[0]:
                        try:
                            chunk = os.read(master, 65536).decode(errors="replace")
                        except OSError:
                            break
                        if not chunk:
                            break
                        log.write(chunk)
                    break
                if attached is None and now - began > 30:
                    raise TimeoutError("collector attachment timeout")
                if attached is not None and snapshot_at is None and fallback_at is None and now - attached >= args.seconds:
                    info["fallback_requested"] = True
                    if not args.sampling:
                        recorder = owned_perf(process.pid, raw / "perf.data")
                        os.kill(recorder, signal.SIGUSR2)
                        info["fallback_perf_pid"] = recorder
                    fallback_at = now
                if fallback_at is not None and stopped_at is None and now - fallback_at >= .5:
                    process.send_signal(signal.SIGINT)
                    stopped_at = now
                decoding_since = snapshot_at if snapshot_at is not None else stopped_at
                if decoding_since is not None and now - decoding_since > args.decode_timeout:
                    raise TimeoutError("decoding timeout; raw recording preserved")
            if attached is None or process.returncode:
                raise ValueError(f"collector failed or never attached; exit={process.returncode}; see capture.log")
        result = verify(trace, None if args.sampling else raw / "perf.data", output / "capture.log")
        (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        hits = raw / "hits.sexp"
        info["trigger_hit_present"] = hits.exists() and hits.read_text().strip() not in ("", "()")
        info["timeline_events"] = result["fxt"]["timeline_events"]
        info["trace_sha256"] = result["fxt"]["sha256"]
        info["validated"] = True
    except (OSError, ValueError, TimeoutError, subprocess.CalledProcessError) as error:
        failure = str(error)
        info.update(validated=False, error=failure)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if slave is not None:
            os.close(slave)
        os.close(master)
        info["collector_exit"] = process.returncode if process is not None else None
        (output / "capture.json").write_text(json.dumps(info, indent=2) + "\n")
    if failure:
        raise SystemExit(failure)
    print(json.dumps({k: info[k] for k in ("validated", "timeline_events", "trace_sha256", "fallback_requested", "trigger_hit_present")}))


if __name__ == "__main__":
    main()
