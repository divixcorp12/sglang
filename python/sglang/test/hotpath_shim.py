"""Builds the LD_PRELOAD counting shim and runs the threaded production-configuration scenario under it in a child
process (plan 2026-09-29-hotpath-zero-overhead Tasks 3 and 12). ``run_child`` returns per-thread counts over
``requests`` steps posted after ``warmup`` steps (the window is armed only around the measured steps), and how many
threads the shim recognized per role (``threads``), so that a count of zero can be told apart from a shim that never
saw the thread.

Each step posts one request on row 0 whose first lane is the previous step's miss, now resident (a hit the copy engine
takes: a COPYING lane and a copy job) and whose second lane is a new miss (read, LOADING). The copy job's mark is
released by that step, one mark at a time, after the request was served; the step then waits until the copy thread
retired it. Every ``DEFER_EVERY``-th step also holds its copy while it posts a request for all four non-resident
experts: with one slot under the copy lease that request must defer, and it is served only once the step releases the
mark. The window therefore covers hits, copy jobs, the copy thread's completions and lease releases, and deferrals;
the child reports how many of each it saw (``copy_jobs`` = marks the copy thread recorded, ``copies_done`` = requests
whose CopyDone the service published, ``deferrals`` = the core counter's delta, ``posts``).

The service thread's ``sleep`` count is 0 only because the child starts the thread with ``spin_us=50_000``: the
measured window never idles for 50 ms, so ``RamThread`` never reaches its idle ``nanosleep``. A smaller ``spin_us``
(production's default) lets idle gaps between posts sleep and count, which is the idle path (spec L12), not the
request path. ``copy_spin_us`` is the copy thread's spin budget before its futex sleep (the default is the tests'
200 us, which the Python-paced gaps between steps exceed, so the copy thread sleeps between steps and each step's
submit wakes it; 50_000 keeps it spinning through the window). What a zero cannot rule out at all is listed in
``hotpath_shim.c``'s header comment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

KINDS = ("malloc", "free", "mutex", "cond", "clock", "sleep", "futex")
THREADS = ("service", "copy")
SOURCE = Path(__file__).with_name("hotpath_shim.c")


def build(tmp_dir) -> Path:
    out = Path(tmp_dir) / "hotpath_shim.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", str(out), str(SOURCE), "-ldl", "-lpthread"], check=True)
    return out


CHILD = textwrap.dedent(
    """
    import ctypes, json, sys, time
    from pathlib import Path

    # Resolved from the global scope, not dlopen'd by path: a path load would succeed without LD_PRELOAD and count
    # nothing, because a library loaded after the host module does not interpose its calls.
    shim = ctypes.CDLL(None)
    if not hasattr(shim, "hotpath_shim_arm"):
        sys.exit("hotpath shim is not preloaded (LD_PRELOAD did not reach the child)")
    shim.hotpath_shim_count.restype = ctypes.c_long
    shim.hotpath_shim_threads.restype = ctypes.c_long

    from sglang.kernels.ops.moe import expert_lease_block as lease
    from sglang.kernels.ops.moe.expert_stream_transport import page_word, sim_wait
    from sglang.test import hotpath_script as hp

    variant, requests, warmup, tmp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    copy_spin_us = int(sys.argv[5])
    DEFER_EVERY = 25
    s, page, host, sim, dst = hp.build_host(Path(tmp), variant=None if variant == "default" else variant,
                                            copy_spin_us=copy_spin_us)
    host.start_thread(fatal_wait_s=60.0, spin_us=50_000)  # 50 ms of spin: the measured window never idles into sleep
    seen = {"posts": 0, "copies_done": 0, "deferrals": 0}
    last = [None]

    def cold():
        mapping = host.mapping(0)
        return [e for e in range(hp.EXPERTS) if mapping[e] < 0]

    def post(lanes, **kw):
        hp.write_hot_record(page, host, hp.next_seq(page), [])
        seen["posts"] += 1
        return sim.post(0, lanes, **kw)

    def serve(req):
        assert sim_wait(page, req.seq, 10.0) == 1, "request not served"
        waited, ack_lanes = hp.accept(sim, req)
        sim.ack(req, waited, lanes=ack_lanes)
        sim.deliver()
        return any(sim.row_result(req, lane)["tag"] == lease.COPYING for lane in range(len(req.lanes)))

    def step(i):
        free = cold()
        miss = min(free, key=lambda e: (e - (last[0] or 0)) %% hp.EXPERTS)
        lanes = [miss] if last[0] is None else [last[0], miss]
        req = post(lanes, dst=[0, 1][: len(lanes)], copy_engine=True)
        copying = serve(req)
        defer = i %% DEFER_EVERY == DEFER_EVERY - 1
        if defer:
            assert copying, "the deferral step needs a copy in flight"
            wanted = cold()
            before = host.counters()["deferred"]
            held = post(wanted)  # every non-resident expert, while the copy holds one of the four slots
            deadline = time.monotonic() + 10.0
            while host.counters()["deferred"] == before:
                assert time.monotonic() < deadline, "the request did not defer"
            assert page_word(page, "demand_done") == req.seq, "a slot under a copy lease was evicted"
            seen["deferrals"] += host.counters()["deferred"] - before
        if copying:
            host.copy_engine_release(1)  # this step's mark, and only it
            assert host.copy_engine_idle(5.0), "the copy thread did not retire the job"
            tag, gen, _mask = sim.copy_done(req)
            seen["copies_done"] += tag == lease.COPIED and gen == req.gen
        if defer:
            serve(held)
            last[0] = wanted[-1]
        else:
            last[0] = miss

    for i in range(warmup):
        step(i)
    for key in seen:
        seen[key] = 0
    marked = host.copy_engine_marked()
    shim.hotpath_shim_reset()
    shim.hotpath_shim_arm(1)
    for i in range(warmup, warmup + requests):
        step(i)
    shim.hotpath_shim_arm(0)
    seen["copy_jobs"] = host.copy_engine_marked() - marked
    counts = {name: {kind: shim.hotpath_shim_count(th, k) for k, kind in enumerate(%r)}
              for th, name in enumerate(%r)}
    threads = {name: shim.hotpath_shim_threads(th) for th, name in enumerate(%r)}
    host.stop()
    print("HOTPATH-COUNTS " + json.dumps({**counts, "threads": threads, "requests": requests, **seen}))
    """
    % (KINDS, THREADS, THREADS)
)


def run_child(shim: Path, *, variant: str = "default", requests: int = 200, warmup: int = 50, tmp,
              copy_spin_us: int = 200) -> dict:
    env = dict(os.environ, LD_PRELOAD=str(shim))
    proc = subprocess.run([sys.executable, "-c", CHILD, variant, str(requests), str(warmup), str(tmp),
                           str(copy_spin_us)],
                          env=env, capture_output=True, text=True, timeout=600)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("HOTPATH-COUNTS ")), None)
    assert proc.returncode == 0 and line, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(line.split(" ", 1)[1])
