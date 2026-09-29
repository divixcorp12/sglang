"""Builds the LD_PRELOAD counting shim and runs the threaded production-configuration scenario under it in a child
process (plan 2026-09-29-hotpath-zero-overhead Task 3). ``run_child`` returns per-thread counts over ``requests``
requests posted after ``warmup`` requests (the window is armed only around the measured requests), and how many
threads the shim recognized per role (``threads``), so that a count of zero can be told apart from a shim that never
saw the thread."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

KINDS = ("malloc", "free", "mutex", "cond", "clock", "sleep")
THREADS = ("service", "copy")
SOURCE = Path(__file__).with_name("hotpath_shim.c")


def build(tmp_dir) -> Path:
    out = Path(tmp_dir) / "hotpath_shim.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", str(out), str(SOURCE), "-ldl", "-lpthread"], check=True)
    return out


CHILD = textwrap.dedent(
    """
    import ctypes, json, sys, threading, time
    from pathlib import Path

    # Resolved from the global scope, not dlopen'd by path: a path load would succeed without LD_PRELOAD and count
    # nothing, because a library loaded after the host module does not interpose its calls.
    shim = ctypes.CDLL(None)
    if not hasattr(shim, "hotpath_shim_arm"):
        sys.exit("hotpath shim is not preloaded (LD_PRELOAD did not reach the child)")
    shim.hotpath_shim_count.restype = ctypes.c_long
    shim.hotpath_shim_threads.restype = ctypes.c_long

    from sglang.kernels.ops.moe.expert_stream_transport import sim_wait
    from sglang.test import hotpath_script as hp

    variant, requests, warmup, tmp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    s, page, host, sim, dst = hp.build_host(Path(tmp), variant=None if variant == "default" else variant)
    host.start_thread(fatal_wait_s=60.0, spin_us=50_000)  # 50 ms of spin: the measured window never idles into sleep
    stop = threading.Event()

    def releaser():
        while not stop.is_set():
            host.copy_engine_release(-1)
            time.sleep(0.0005)

    t = threading.Thread(target=releaser, daemon=True)
    t.start()

    def one(i):
        lanes = [i %% hp.EXPERTS, (i + 3) %% hp.EXPERTS]
        seq = hp.next_seq(page)
        hp.write_hot_record(page, host, seq, [])
        req = sim.post(0, lanes, dst=[0, 1], copy_engine=True)
        assert sim_wait(page, req.seq, 10.0) == 1, "request not served"
        waited, ack_lanes = hp.accept(sim, req)
        sim.ack(req, waited, lanes=ack_lanes)
        sim.deliver()

    for i in range(warmup):
        one(i)
    shim.hotpath_shim_reset()
    shim.hotpath_shim_arm(1)
    for i in range(warmup, warmup + requests):
        one(i)
    shim.hotpath_shim_arm(0)
    stop.set()
    t.join()
    counts = {name: {kind: shim.hotpath_shim_count(th, k) for k, kind in enumerate(%r)}
              for th, name in enumerate(%r)}
    threads = {name: shim.hotpath_shim_threads(th) for th, name in enumerate(%r)}
    host.stop()
    print("HOTPATH-COUNTS " + json.dumps({**counts, "threads": threads, "requests": requests}))
    """
    % (KINDS, THREADS, THREADS)
)


def run_child(shim: Path, *, variant: str = "default", requests: int = 200, warmup: int = 50, tmp) -> dict:
    env = dict(os.environ, LD_PRELOAD=str(shim))
    proc = subprocess.run([sys.executable, "-c", CHILD, variant, str(requests), str(warmup), str(tmp)],
                          env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("HOTPATH-COUNTS ")), None)
    assert proc.returncode == 0 and line, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(line.split(" ", 1)[1])
