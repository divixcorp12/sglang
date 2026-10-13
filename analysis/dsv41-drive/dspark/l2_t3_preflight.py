#!/usr/bin/env python3
"""Quiet-box preflight for the T3 quick run: refuse while any GPU compute process exists or an expert drive is busy.

Exit 0 prints "preflight ok"; exit 1 lists the offending GPU pids or the busy drives. Pids holding files on the expert
drives outside their `dsv41_flash` mirror roots are listed as notes only: idle long-lived holders (a Ray cluster's
logs, a virtualenv) were present during the F108 baseline too, and what disturbs a run is traffic, not an open file.
"""

import subprocess
import sys
import time

MOUNTS = ("/mnt/nvme0", "/mnt/nvme2", "/mnt/nvme4")
MIRROR_DIR = "dsv41_flash"
SAMPLE_S = 5.0
# Arbitrary: well above idle log writes (tens of kB/s), far below any real reader.
BUSY_MB_PER_S = 1.0
SECTOR_BYTES = 512


def diskstat_bytes(diskstats_text, *, devices):
    """{device: bytes read + written} from /proc/diskstats for the named devices (e.g. nvme0n1p1)."""
    out = {}
    for line in diskstats_text.splitlines():
        f = line.split()
        if len(f) >= 10 and f[2] in devices:
            out[f[2]] = (int(f[5]) + int(f[9])) * SECTOR_BYTES
    return out


def busy_devices(before, after, *, seconds, limit_mb_per_s=BUSY_MB_PER_S):
    """{device: MB/s} of devices whose read+write rate between two diskstat_bytes samples exceeds the limit."""
    rates = {d: (after[d] - before[d]) / seconds / 1e6 for d in after if d in before}
    return {d: r for d, r in rates.items() if r > limit_mb_per_s}


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
    # lsof exits 1 when nothing matches. Run unprivileged, it sees only this user's processes; jobs of other users on
    # the mirror drives are not detected.
    lsof = _run(["lsof", "-nP", "+f", "--", *MOUNTS], ok_codes=(0, 1))
    roots = tuple(f"{mount}/{MIRROR_DIR}" for mount in MOUNTS)
    gpu = gpu_compute_pids(smi)
    users = drive_users(lsof, mounts=MOUNTS, allowed_roots=roots)
    devices = {_run(["findmnt", "-n", "-o", "SOURCE", m], ok_codes=(0,)).strip().removeprefix("/dev/") for m in MOUNTS}
    with open("/proc/diskstats") as f:
        before = diskstat_bytes(f.read(), devices=devices)
    time.sleep(SAMPLE_S)
    with open("/proc/diskstats") as f:
        after = diskstat_bytes(f.read(), devices=devices)
    busy = busy_devices(before, after, seconds=SAMPLE_S)
    for pid, paths in sorted(users.items()):
        print(f"preflight note: pid {pid} holds {len(set(paths))} file(s) on the expert drives, e.g. {sorted(paths)[0]}")
    if not gpu and not busy:
        print(f"preflight ok (expert drives under {BUSY_MB_PER_S} MB/s over {SAMPLE_S:.0f} s)")
        return 0
    for pid in gpu:
        print(f"preflight refused: GPU compute process pid {pid}")
    for device, rate in sorted(busy.items()):
        print(f"preflight refused: {device} moved {rate:.1f} MB/s over {SAMPLE_S:.0f} s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
