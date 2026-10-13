#!/usr/bin/env python3
"""Quiet-box preflight for the T3 quick run: refuse while any GPU compute process or drive user exists.

Exit 0 prints "preflight ok"; exit 1 lists the offending GPU pids and the pids holding files on the
expert drives outside their `dsv41_flash` mirror roots.
"""

import subprocess
import sys

MOUNTS = ("/mnt/nvme0", "/mnt/nvme2", "/mnt/nvme4")
MIRROR_DIR = "dsv41_flash"


def gpu_compute_pids(nvidia_smi_csv):
    """Pids of `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` output."""
    pids = []
    for line in nvidia_smi_csv.splitlines():
        first = line.split(",")[0].strip()
        if first.isdigit():
            pids.append(int(first))
    return pids


def _under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def drive_users(lsof_text, *, mounts=MOUNTS, allowed_roots):
    """{pid: [path]} of open files under `mounts` that lie outside every root in `allowed_roots`.

    Parses default `lsof -nP` columns: COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME, where NAME
    may hold spaces and is taken from the first token after TYPE that starts with "/".
    """
    users = {}
    for line in lsof_text.splitlines():
        fields = line.split()
        if len(fields) < 9 or not fields[1].isdigit():
            continue
        start = next((i for i in range(5, len(fields)) if fields[i].startswith("/")), None)
        if start is None:
            continue
        path = " ".join(fields[start:])
        if not any(_under(path, mount) for mount in mounts):
            continue
        if any(_under(path, root) for root in allowed_roots):
            continue
        users.setdefault(int(fields[1]), []).append(path)
    return users


def _run(argv, *, ok_codes):
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode not in ok_codes:
        raise SystemExit(f"preflight: {argv[0]} exited {done.returncode}: {done.stderr.strip()}")
    return done.stdout


def main():
    smi = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], ok_codes=(0,)
    )
    # lsof exits 1 when nothing matches
    lsof = _run(["lsof", "-nP", "+f", "--", *MOUNTS], ok_codes=(0, 1))
    roots = tuple(f"{mount}/{MIRROR_DIR}" for mount in MOUNTS)
    gpu = gpu_compute_pids(smi)
    users = drive_users(lsof, mounts=MOUNTS, allowed_roots=roots)
    if not gpu and not users:
        print("preflight ok")
        return 0
    for pid in gpu:
        print(f"preflight refused: GPU compute process pid {pid}")
    for pid, paths in sorted(users.items()):
        print(f"preflight refused: pid {pid} uses {', '.join(sorted(set(paths)))}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
