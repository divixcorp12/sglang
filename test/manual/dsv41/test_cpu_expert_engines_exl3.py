"""Two CPU expert engines of the EXL3 kernel on disjoint cores (spec 2026-10-03-numa-node-distributor-design,
Testing 4). Each engine runs its own OpenMP team on its own cores; two running at once give, bit for bit, what one
gives alone. Needs the optimized ext build (RUN_EXT) and 4 cores in the affinity mask.

Every pinned forward runs on a helper thread: a forward pins its calling thread (worker 0), and a pinned pytest thread
would leave the next test one core."""

import ctypes
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

sys.path.insert(0, os.path.dirname(__file__))
from test_cpu_expert_pool_exl3 import CAP, LIMIT, _random_slabs  # noqa: E402

HIDDEN, INTER = 5120, 2304  # DeepSeek V4.1's shape: the DSV4.1 plan on AVX-512BW
REPEATS = 40
CORES = sorted(os.sched_getaffinity(0))  # read at import, before any forward could pin this thread
KEEP_WARM = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64)


def _kernel():
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the engine ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    if len(CORES) < 4:
        pytest.skip("needs 4 cores in the affinity mask")
    return Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT), CORES[:4]


def _forward(trait, layer, x, slots, weights, engine, threads):
    """One-row call of the C ABI's forward on ``engine``; returns (status, out)."""
    from sglang.srt.layers.moe.cpu_experts.pool import (
        CPU_EXPERTS_FORWARD_ABI_VERSION,
        CpuExpertForward,
        CpuExpertsForwardCall,
    )

    s = torch.tensor(slots, dtype=torch.int32)
    w = torch.tensor(weights, dtype=torch.float32)
    out = torch.full((HIDDEN,), float("nan"))
    call = CpuExpertsForwardCall(
        abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=layer, x=x.data_ptr(),
        slots=ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
        weights=ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), k=len(slots), threads=threads, accumulate=0,
        engine=engine,
    )
    return CpuExpertForward(trait.native_forward())(ctypes.byref(call)), out


def _inputs():
    g = torch.Generator().manual_seed(20261004)
    return [
        ((torch.randn(HIDDEN, generator=g) * scale).half(), [i % CAP, (i + 2) % CAP, (i + 5) % CAP], [0.5, 0.3, 0.2])
        for i, scale in enumerate([1.0, 4.0, 8.0, 0.5] * 3)
    ]


def _on_threads(fn, args):
    """fn(arg) for each arg, each on its own thread, all at once; returns their results in order."""
    results = [None] * len(args)

    def run(i):
        results[i] = fn(args[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(args))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_two_engines_at_once_match_one_engine_bit_for_bit(monkeypatch):
    """Review Focus 3. Mutants: one process-wide core list (engine B's team pinned onto A's cores) -- red on the
    affinity test below; a shared static scratch, or a forward lock returning 3 -- red here."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
    layer = trait.register_layer(_random_slabs(20261003, HIDDEN, INTER), CAP)
    inputs = _inputs()
    try:
        (want,) = _on_threads(
            lambda engine: [_forward(trait, layer, x, s, w, engine, 2) for x, s, w in inputs], [a]
        )
        assert all(rc == 0 and torch.isfinite(out).all() for rc, out in want)

        def run(engine):
            bad = []
            for _ in range(REPEATS):
                for i, (x, slots, weights) in enumerate(inputs):
                    rc, out = _forward(trait, layer, x, slots, weights, engine, 2)
                    if rc != 0 or not torch.equal(out, want[i][1]):
                        bad.append((engine, i, rc))
            return bad

        assert _on_threads(run, [a, b]) == [[], []]
    finally:
        trait.free_layer(layer)
        trait.native_free_engine(a)
        trait.native_free_engine(b)


def test_each_engines_workers_run_on_its_own_cores(monkeypatch):
    """Two keep-warms at once, one per engine: while they run, the process has a thread pinned to each of the four
    engine cores. Mutant: pin every worker from one core list -- red."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
    keep_warm = KEEP_WARM(trait.native_keep_warm())
    word = torch.zeros(1, dtype=torch.int32)
    deadline = time.monotonic_ns() + 10_000_000_000
    rc = {}
    runners = [
        threading.Thread(target=lambda e=e: rc.setdefault(e, keep_warm(e, 2, word.data_ptr(), 0, deadline)))
        for e in (a, b)
    ]
    try:
        for runner in runners:
            runner.start()
        time.sleep(0.3)
        pinned = set()
        for tid in os.listdir("/proc/self/task"):
            mask = os.sched_getaffinity(int(tid))
            if len(mask) == 1:
                pinned |= mask
        word[0] = 1
        for runner in runners:
            runner.join(5)
        assert set(cores) <= pinned, (cores, sorted(pinned))
        assert rc == {a: 0, b: 0}
    finally:
        trait.native_free_engine(a)
        trait.native_free_engine(b)


def test_the_engine_abi_refuses_what_it_cannot_run(monkeypatch):
    """A repeated core, more workers than the engine's cores, an engine never created and a freed engine are refused
    (2) before any work; engine 0 runs unpinned workers (this thread stays unpinned, so it runs here)."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    with pytest.raises(RuntimeError, match="refused engine cores"):
        trait.native_create_engine([cores[0], cores[0]])
    a = trait.native_create_engine(cores[:2])
    layer = trait.register_layer(_random_slabs(1, HIDDEN, INTER), CAP)
    x, slots, weights = _inputs()[0]
    keep_warm = KEEP_WARM(trait.native_keep_warm())
    word = torch.zeros(1, dtype=torch.int32)
    try:
        rc, out = _forward(trait, layer, x, slots, weights, a, 3)
        assert rc == 2 and out.isnan().all(), "more workers than the engine's cores"
        assert _forward(trait, layer, x, slots, weights, a + 1000, 2)[0] == 2, "an engine never created"
        assert _forward(trait, layer, x, slots, weights, 0, 2)[0] == 0, "engine 0: unpinned workers"
        assert keep_warm(a, 3, word.data_ptr(), 0, time.monotonic_ns()) == 2
        assert keep_warm(a + 1000, 1, word.data_ptr(), 0, time.monotonic_ns()) == 2
        trait.native_free_engine(a)
        assert _forward(trait, layer, x, slots, weights, a, 2)[0] == 2, "a freed engine"
        with pytest.raises(RuntimeError, match="no engine"):
            trait.native_free_engine(a)
    finally:
        trait.free_layer(layer)


_LIMIT_SCRIPT = """
import os, sys, threading
sys.path.insert(0, sys.argv[1])
from test_cpu_expert_engines_exl3 import HIDDEN, INTER, _forward, _inputs, _kernel
from test_cpu_expert_pool_exl3 import CAP, _random_slabs
trait, cores = _kernel()
a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
layer = trait.register_layer(_random_slabs(5, HIDDEN, INTER), CAP)
rcs = []
def run(engine):
    for x, slots, weights in _inputs() * 10:
        rcs.append(_forward(trait, layer, x, slots, weights, engine, 2)[0])
workers = [threading.Thread(target=run, args=(e,)) for e in (a, b)]
[w.start() for w in workers]
[w.join() for w in workers]
print("rcs", sorted(set(rcs)))
"""


def test_two_full_teams_run_at_once_at_a_thread_limit_of_their_sum(monkeypatch):
    """Review Focus 3: OMP_THREAD_LIMIT equal to the two engines' workers (2 + 2) leaves neither team short (a short
    team is status 1, which run_team returns rather than run), the bound ThreadingConfig enforces. libgomp reads the
    limit at load, so this runs in a child."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    _kernel()  # skips here, not in the child
    env = dict(os.environ, OMP_THREAD_LIMIT="4", EXL3_MOE_CPU_PIN="0")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_LIMIT_SCRIPT), os.path.dirname(__file__)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "rcs [0]" in result.stdout, result.stdout[-2000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
