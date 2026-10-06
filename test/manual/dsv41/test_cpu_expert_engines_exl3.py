"""Two CPU expert core groups of the EXL3 kernel on disjoint cores (spec 2026-10-03-numa-node-distributor-design,
Testing 4). Each group runs its own OpenMP team on its own cores; two running at once give, bit for bit, what one
gives alone. Needs the optimized ext build (RUN_EXT) and 4 cores in the affinity mask.

Every pinned forward runs on a helper thread: a forward pins its calling thread (worker 0), and a pinned pytest thread
would leave the next test one core."""

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

CAP = 6
LIMIT = 10.0


def _random_slabs(seed, hidden, inter):
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, hidden // 16, inter // 16, 48),
        "w13_suh": signs(CAP, 2, hidden),
        "w13_svh": signs(CAP, 2, inter),
        "w2_trellis": codes(CAP, inter // 16, hidden // 16, 48),
        "w2_suh": signs(CAP, inter),
        "w2_svh": signs(CAP, hidden),
    }


HIDDEN, INTER = 5120, 2304  # DeepSeek V4.1's shape: the DSV4.1 plan on AVX-512BW
REPEATS = 40
CORES = sorted(os.sched_getaffinity(0))  # read at import, before any forward could pin this thread


def _kernel():
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the kernel under test is the optimized one: set SGLANG_DSV41_CPU_EXPERTS=1")
    if len(CORES) < 4:
        pytest.skip("needs 4 cores in the affinity mask")
    return Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT), CORES[:4]


def _forward(trait, layer, x, slots, weights, cores, threads):
    """One-row ``kernel_forward`` on ``cores``; returns (status, out)."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    out = torch.full((1, HIDDEN), float("nan"))
    status, _ = es.kernel_forward(
        layer, x.unsqueeze(0), torch.tensor([slots], dtype=torch.int32), torch.tensor([weights], dtype=torch.float32),
        out, threads=threads, cores=cores, variant="instr",
    )
    return status, out


def _layer(trait, seed):
    from sglang.kernels.ops.moe import expert_stream_transport as es

    return es.kernel_layer(trait.kernel_address(), trait.layer_spec(_random_slabs(seed, HIDDEN, INTER), CAP), variant="instr")


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


def test_the_extension_hands_out_one_kernel_address():
    trait, _ = _kernel()
    address = trait.kernel_address()
    assert address != 0 and trait.kernel_address() == address


def test_the_extension_packs_the_params_make_layer_reads():
    """sglang_exl3_cpu::params is SglangExl3CpuParams {int32 bits, int32 swizzled}, little-endian."""
    import struct

    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait, _ = _kernel()
    swizzled = Exl3CpuQuantTrait(trait.ext, act_limit=10.0, swizzled=True)
    assert trait._params(3) == struct.pack("<ii", 3, 0) and swizzled._params(4) == struct.pack("<ii", 4, 1)


def test_two_core_groups_at_once_match_one_group_bit_for_bit(monkeypatch):
    """Review Focus 3. Mutants: one process-wide core list (group B's team pinned onto A's cores) -- red on the
    affinity test below; a shared static scratch, or a forward lock returning 3 -- red here."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    a, b = cores[:2], cores[2:4]
    layer = _layer(trait, 20261003)
    inputs = _inputs()
    try:
        (want,) = _on_threads(lambda on: [_forward(trait, layer, x, s, w, on, 2) for x, s, w in inputs], [a])
        assert all(rc == 0 and torch.isfinite(out).all() for rc, out in want)

        def run(on):
            bad = []
            for _ in range(REPEATS):
                for i, (x, slots, weights) in enumerate(inputs):
                    rc, out = _forward(trait, layer, x, slots, weights, on, 2)
                    if rc != 0 or not torch.equal(out, want[i][1]):
                        bad.append((on, i, rc))
            return bad

        assert _on_threads(run, [a, b]) == [[], []]
    finally:
        es.kernel_drop(layer, variant="instr")


def test_each_groups_workers_run_on_its_own_cores(monkeypatch):
    """Two core groups forwarding at once: while they run, the process has a thread pinned to each of the four
    cores. Mutant: pin every worker from one core list -- red."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = _layer(trait, 20261005)
    x, slots, weights = _inputs()[0]
    stop = threading.Event()
    rcs = []

    def run(on):
        while not stop.is_set():
            rcs.append(_forward(trait, layer, x, slots, weights, on, 2)[0])

    runners = [threading.Thread(target=run, args=(on,)) for on in (cores[:2], cores[2:4])]
    try:
        for runner in runners:
            runner.start()
        time.sleep(0.3)
        pinned = set()
        for tid in os.listdir("/proc/self/task"):
            try:
                mask = os.sched_getaffinity(int(tid))
            except OSError:  # a worker that exited between the listing and the read
                continue
            if len(mask) == 1:
                pinned |= mask
        stop.set()
        for runner in runners:
            runner.join(30)
        assert set(cores) <= pinned, (cores, sorted(pinned))
        assert set(rcs) == {0}
    finally:
        stop.set()
        es.kernel_drop(layer, variant="instr")


def test_the_kernel_refuses_what_it_cannot_run(monkeypatch):
    """More workers than the call's cores is refused (2) before any work; no cores at all runs unpinned workers (this
    thread stays unpinned, so it runs here)."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = _layer(trait, 1)
    x, slots, weights = _inputs()[0]
    try:
        rc, out = _forward(trait, layer, x, slots, weights, cores[:2], 3)
        assert rc == 2 and out.isnan().all(), "more workers than the call's cores"
        assert _forward(trait, layer, x, slots, weights, (), 2)[0] == 0, "no cores: unpinned workers"
    finally:
        es.kernel_drop(layer, variant="instr")


_LIMIT_SCRIPT = """
import os, sys, threading
sys.path.insert(0, sys.argv[1])
from test_cpu_expert_engines_exl3 import _forward, _inputs, _kernel, _layer
trait, cores = _kernel()
layer = _layer(trait, 5)
rcs = []
def run(on):
    for x, slots, weights in _inputs() * 10:
        rcs.append(_forward(trait, layer, x, slots, weights, on, 2)[0])
workers = [threading.Thread(target=run, args=(on,)) for on in (cores[:2], cores[2:4])]
[w.start() for w in workers]
[w.join() for w in workers]
print("rcs", sorted(set(rcs)))
"""


def test_two_full_teams_run_at_once_at_a_thread_limit_of_their_sum(monkeypatch):
    """Review Focus 3: OMP_THREAD_LIMIT equal to the two core groups' workers (2 + 2) leaves neither team short (a short
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



def test_a_multi_row_forward_on_prod_matches_one_row_forwards(monkeypatch):
    """The DSpark draft (cpu_experts/draft.py) runs m rows per kernel_forward on the production build; the kernel groups
    the rows' routes by expert, so this pins that grouping to the one-row results bit for bit, at m = 1..6."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = es.kernel_layer(trait.kernel_address(), trait.layer_spec(_random_slabs(20261005, HIDDEN, INTER), CAP),
                            variant="prod")
    inputs = _inputs()[:6]

    def forward(rows):
        out = torch.full((len(rows), HIDDEN), float("nan"))
        status, why = es.kernel_forward(
            layer, torch.stack([x for x, _, _ in rows]), torch.tensor([s for _, s, _ in rows], dtype=torch.int32),
            torch.tensor([w for _, _, w in rows], dtype=torch.float32), out, threads=2, cores=cores[:2], variant="prod",
        )
        assert (status, why) == (0, "")
        return out

    def run(_):
        singles = torch.cat([forward([row]) for row in inputs])
        return [(m, torch.equal(forward(inputs[:m]), singles[:m])) for m in range(1, 7)]

    try:
        (results,) = _on_threads(run, [None])
        assert results == [(m, True) for m in range(1, 7)]
    finally:
        es.kernel_drop(layer, variant="prod")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
