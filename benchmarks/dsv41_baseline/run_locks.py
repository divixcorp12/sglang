"""Take a divix01 run lock (rowimg-disk.lock, cc-gpu.lock) without deadlocking on our own caller.

A script launched as `flock rowimg-disk.lock bash run.sh` that then flocks the same file itself opens a second file
description, which flock(2) treats as another owner: the script waits forever on its own ancestor, silently.
`take` treats a lock an ancestor holds as already ours, and names any other holder before waiting on it.
"""

from __future__ import annotations

import fcntl
import os
from typing import IO


def _ancestors() -> set[int]:
    pids, pid = set(), os.getppid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                # comm may contain spaces and parens; ppid is the second field after the last ')'.
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _holders(path: str) -> list[int]:
    st = os.stat(path)
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    pids = []
    with open("/proc/locks") as f:
        for line in f:
            fields = line.split()
            if "->" in fields or "FLOCK" not in fields:
                continue
            i = fields.index("FLOCK")
            if len(fields) > i + 4 and fields[i + 4] == key:
                pids.append(int(fields[i + 3]))
    return pids


def _cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:160]
    except OSError:
        return "?"


def take(path: str) -> IO:
    """Open ``path`` and hold its exclusive flock, or rely on an ancestor's; the caller keeps the file open."""
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        pass
    holders = _holders(path)
    ours = sorted(set(holders) & _ancestors())
    if ours:
        print(f"{path} is held by our parent pid {ours[0]} ({_cmd(ours[0])}); using it", flush=True)
        return f
    who = ", ".join(f"pid {p} ({_cmd(p)})" for p in holders) or "an unknown pid"
    print(f"waiting for {path} held by {who}", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f
