"""Server-only diagnostic startup hook, never on the normal SGLang import path.

Initialize the audited OpenMP binary with a chosen CPU budget, then restore the
launch affinity before Torch/model imports. PYTHONPATH propagates this to spawn
children; fork children inherit the already initialized runtime. No threads start.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

RUNTIME_SHA = "e28fb2896a3d612b46a27d7f8ef840d34c9efbc29b70a044c42095fb94856890"
OFFSETS = dict(managed_threads=0x237040, available_cpus=0x237048,
               normal_spin_count=0x2373c8, throttled_spin_count=0x2373d8)


def bootstrap():
    original = os.sched_getaffinity(0)
    requested = {int(cpu) for cpu in os.environ["DSV41_OMP_INIT_CPUS"].split(",")}
    if not requested or not requested <= set(range(64)):
        raise RuntimeError("diagnostic CPUs must be in 0..63")
    path = Path(os.environ["DSV41_OMP_LIBRARY"]).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RUNTIME_SHA:
        raise RuntimeError("OpenMP hash differs; re-audit private offsets before measuring")
    if any("libgomp" in line for line in Path("/proc/self/maps").read_text().splitlines()):
        raise RuntimeError("OpenMP was initialized before the diagnostic hook")
    try:
        os.sched_setaffinity(0, requested)
        library = ctypes.CDLL(str(path))
    finally:
        os.sched_setaffinity(0, original)
    bases = [int(line.split("-", 1)[0], 16) for line in
             Path("/proc/self/maps").read_text().splitlines()
             if line.split()[2] == "00000000" and line.endswith(str(path))]
    if len(bases) != 1:
        raise RuntimeError("cannot identify audited OpenMP mapping")
    base = bases[0]
    counters = {name: ctypes.c_uint64.from_address(base + offset).value
                for name, offset in OFFSETS.items()}
    if counters["available_cpus"] != len(requested):
        raise RuntimeError("OpenMP cached CPU count did not match requested budget")
    directory = Path(os.environ["DSV41_OMP_MANIFEST_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(pid=os.getpid(), ppid=os.getppid(), ns=time.monotonic_ns(),
                runtime=str(path), runtime_sha256=RUNTIME_SHA, base=base, offsets=OFFSETS,
                init_affinity=sorted(requested), restored_affinity=sorted(os.sched_getaffinity(0)),
                counters=counters)
    temporary = directory / f".{os.getpid()}.tmp"
    temporary.write_text(json.dumps(data) + "\n")
    temporary.replace(directory / f"{os.getpid()}.json")
    # Keep the dlopen handle alive for the process lifetime.
    return library


if os.environ.get("DSV41_OMP_INIT_CPUS"):
    try:
        _runtime = bootstrap()
    except Exception as error:
        print(f"REFUSE OpenMP bootstrap: {error}", file=sys.stderr, flush=True)
        # sitecustomize exceptions normally only warn and allow startup to continue.
        os._exit(78)
