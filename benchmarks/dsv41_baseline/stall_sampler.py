"""Bounded /proc diagnostics for one server selected by its unique job-trace prefix.

CPU experts are sampled every 50 ms; global and per-node memory counters every 250 ms.
This is supporting evidence: short waits can fall between samples, and inaccessible wchan is not a wait verdict.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time


def stat_record(text):
    fields = text.rsplit(")", 1)[1].split()
    return {"comm": text.split("(", 1)[1].rsplit(")", 1)[0], "state": fields[0],
            "minflt": int(fields[7]), "majflt": int(fields[9]),
            "utime_ticks": int(fields[11]), "stime_ticks": int(fields[12]),
            "starttime_ticks": int(fields[19]), "cpu": int(fields[36])}


def counters(text):
    result = {}
    for line in text.splitlines():
        fields = line.replace(":", "").split()
        if fields[:1] == ["Node"]:
            fields = fields[2:]
        if len(fields) >= 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--max-seconds", type=float, default=2400)
    parser.add_argument("--max-bytes", type=int, default=128 * 1024**2)
    args = parser.parse_args()
    if args.interval < 0.02 or args.max_seconds <= 0 or args.max_bytes < 1024:
        parser.error("interval >= 0.02, positive max-seconds and max-bytes >= 1024 required")
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    errors = Counter()

    def read(path):
        try:
            return path.read_text()
        except (OSError, UnicodeError) as error:
            errors[type(error).__name__] += 1
            return ""

    wanted = ("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=" + args.prefix).encode()
    selected = None
    started = time.monotonic()
    discovery_at = memory_at = 0
    samples = written = 0
    with args.output.open("w") as out:
        def write(row):
            nonlocal written
            data = json.dumps(row, separators=(",", ":")) + "\n"
            out.write(data)
            written += len(data.encode())

        write({"schema": 1, "clock": "CLOCK_MONOTONIC", "interval_s": args.interval,
               "epoch_ns": time.time_ns(), "ns": time.monotonic_ns()})
        while not stopped and not args.stop_file.exists() and time.monotonic() - started < args.max_seconds:
            tick = time.monotonic()
            if selected is None and tick >= discovery_at:
                discovery_at = tick + 2
                for directory in Path("/proc").iterdir():
                    if not directory.name.isdecimal():
                        continue
                    try:
                        if directory.joinpath("comm").read_text().strip() != "sglang::scheduler":
                            continue
                        if wanted in directory.joinpath("environ").read_bytes().split(b"\0"):
                            selected = directory
                            break
                    except OSError:
                        continue
            row = {"ns": time.monotonic_ns(), "pid": int(selected.name) if selected else None}
            if selected is not None:
                threads = []
                try:
                    directories = list(selected.joinpath("task").iterdir())
                except OSError:
                    directories = []
                for directory in directories:
                    stat = read(directory / "stat")
                    if not stat:
                        continue
                    record = stat_record(stat)
                    if not record["comm"].startswith("exl3-"):
                        continue
                    record["tid"] = int(directory.name)
                    status = counters(read(directory / "status"))
                    record["nvcsw"] = status.get("voluntary_ctxt_switches")
                    record["nivcsw"] = status.get("nonvoluntary_ctxt_switches")
                    record["wchan"] = read(directory / "wchan").strip()
                    sched = read(directory / "schedstat").split()
                    if len(sched) == 3:
                        record["schedstat"] = list(map(int, sched))  # runtime ns, runqueue wait ns, timeslices
                    threads.append(record)
                row["threads"] = threads
                status = counters(read(selected / "status"))
                row["process_memory_kb"] = {k: status.get(k) for k in ("VmRSS", "VmSwap", "VmLck", "VmPin")}
            if tick >= memory_at:
                memory_at = tick + 0.25
                row["meminfo_kb"] = counters(read(Path("/proc/meminfo")))
                row["vmstat"] = counters(read(Path("/proc/vmstat")))
                row["nodes"] = {p.name: {"meminfo_kb": counters(read(p / "meminfo")),
                                        "vmstat": counters(read(p / "vmstat"))}
                                for p in Path("/sys/devices/system/node").glob("node[0-9]*")}
            write(row)
            samples += 1
            if written >= args.max_bytes:
                break
            out.flush()
            time.sleep(max(0, args.interval - (time.monotonic() - tick)))
        write({"footer": True, "samples": samples, "bytes_before_footer": written,
               "limit_reached": written >= args.max_bytes or time.monotonic() - started >= args.max_seconds,
               "errors": dict(errors)})


if __name__ == "__main__":
    main()
