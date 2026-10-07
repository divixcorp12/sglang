"""Temporarily exclude two identified competitors from CPU expert cores.

Run as the privileged Nsight target, while the CPU benchmark remains unprivileged.
A separate guardian restores the durable snapshot if this helper dies or hangs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(tid):
    stat = Path(f"/proc/{tid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2:].split()[19])


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def restore(directory):
    snapshot = json.loads((directory / "affinity-snapshot.json").read_text())
    results = []
    for process in snapshot["processes"]:
        pid = process["pid"]
        try:
            if identity(pid) != process["starttime"]:
                continue
            for task in Path(f"/proc/{pid}/task").iterdir():
                tid = int(task.name)
                try:
                    birth = identity(tid)
                    original = process["threads"].get(str(tid))
                    mask = original["cpus"] if original and original["starttime"] == birth else process["leader_cpus"]
                    os.sched_setaffinity(tid, mask)
                    actual = sorted(os.sched_getaffinity(tid))
                    results.append(dict(pid=pid, tid=tid, starttime=birth, expected=mask, actual=actual, match=mask == actual))
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    valid = all(r["match"] for r in results)
    write_json(directory / "affinity-restoration.json", dict(restored=valid, monotonic_ns=time.monotonic_ns(), threads=results))
    if not valid:
        raise RuntimeError("affinity restoration mismatch")


def guard(directory, parent, parent_birth):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        marker = directory / "affinity-restoration.json"
        if marker.exists() and json.loads(marker.read_text())["restored"]:
            return
        try:
            if identity(parent) != parent_birth:
                break
        except FileNotFoundError:
            break
        time.sleep(.1)
    try:
        if identity(parent) == parent_birth:
            os.kill(parent, signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
        pass
    restore(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=9.)
    args = parser.parse_args()
    assert os.geteuid() == 0 and 0 < args.seconds <= 10
    os.umask(0o022)
    manifest = json.loads(args.manifest.read_text())
    assert sorted(p["comm"] for p in manifest) == ["cadvisor", "ray::DashboardA"]
    excluded = set()
    for cpu in (*range(6, 16), *range(18, 28)):
        siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
        for part in siblings.split(","):
            limits = list(map(int, part.split("-")))
            excluded.update(range(limits[0], limits[-1] + 1))
    allowed = set(range(64)) - excluded
    os.sched_setaffinity(0, allowed)
    processes = []
    for expected in manifest:
        pid = expected["pid"]
        assert identity(pid) == expected["starttime"]
        assert Path(f"/proc/{pid}/comm").read_text().strip() == expected["comm"]
        process = {**expected, "leader_cpus": sorted(os.sched_getaffinity(pid)), "threads": {}}
        for task in Path(f"/proc/{pid}/task").iterdir():
            tid = int(task.name)
            try:
                process["threads"][str(tid)] = dict(starttime=identity(tid), cpus=sorted(os.sched_getaffinity(tid)))
            except (FileNotFoundError, ProcessLookupError):
                pass
        processes.append(process)
    snapshot = dict(processes=processes, excluded_cpus=sorted(excluded), temporary_cpus=sorted(allowed))
    write_json(args.directory / "affinity-snapshot.json", snapshot)
    parent, parent_birth = os.getpid(), identity(os.getpid())
    # Exec a fresh interpreter: Nsight can inject threads even with --trace=none,
    # making Python work in a raw fork child unsafe. Detach the recovery process
    # from the collector's process group, with its own hard restoration deadline.
    guardian = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--guardian",
                                 str(args.directory), str(parent), str(parent_birth)], start_new_session=True)

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    checks = 0
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            for process in processes:
                pid = process["pid"]
                assert identity(pid) == process["starttime"], "target restarted during capture"
                for task in Path(f"/proc/{pid}/task").iterdir():
                    tid = int(task.name)
                    try:
                        mask = set(os.sched_getaffinity(tid))
                        if not mask <= allowed:
                            original = process["threads"].get(str(tid))
                            if original and identity(tid) == original["starttime"]:
                                temporary = set(original["cpus"]) & allowed
                            else:
                                temporary = set(process["leader_cpus"]) & allowed
                            assert temporary, "target has no allowed housekeeping CPU"
                            os.sched_setaffinity(tid, temporary)
                        assert set(os.sched_getaffinity(tid)) <= allowed
                        checks += 1
                    except (FileNotFoundError, ProcessLookupError):
                        pass
            if not (args.directory / "affinity-ready.json").exists():
                write_json(args.directory / "affinity-ready.json", dict(monotonic_ns=time.monotonic_ns(), temporary_cpus=sorted(allowed)))
            time.sleep(.1)
    finally:
        restore(args.directory)
        write_json(args.directory / "affinity-checks.json", dict(thread_checks=checks, end_ns=time.monotonic_ns()))
        guardian.wait(timeout=25)
        assert guardian.returncode == 0, "affinity recovery guardian failed"


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--guardian":
        guard(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
