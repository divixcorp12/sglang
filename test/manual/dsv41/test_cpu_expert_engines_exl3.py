"""Two CPU expert core groups of the EXL3 kernel on disjoint cores (spec 2026-10-03-numa-node-distributor-design,
Testing 4). Each group runs its own OpenMP team on its own cores; two running at once give, bit for bit, what one
gives alone. Needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 (the kernel is compiled into the host module) and 4
cores in the affinity mask.

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

sys.path.insert(0, os.path.dirname(__file__))

CAP = 6
LIMIT = 10.0


def _random_slabs(seed, hidden, inter, bits_w13=3, bits_w2=3):
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, hidden // 16, inter // 16, 16 * bits_w13),
        "w13_suh": signs(CAP, 2, hidden),
        "w13_svh": signs(CAP, 2, inter),
        "w2_trellis": codes(CAP, inter // 16, hidden // 16, 16 * bits_w2),
        "w2_suh": signs(CAP, inter),
        "w2_svh": signs(CAP, hidden),
    }


HIDDEN, INTER = 5120, 2304  # DeepSeek V4.1's shape: the DSV4.1 plan on AVX-512BW
REPEATS = 40
CORES = sorted(os.sched_getaffinity(0))  # read at import, before any forward could pin this thread


def _trait(**kwargs):
    """An EXL3 trait over the host module's plan list."""
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait = Exl3CpuQuantTrait(**{"act_limit": LIMIT, **kwargs})
    trait.attach_host(es.cpu_kernel_options(variant="instr"))
    return trait


def _has_bw():
    from sglang.kernels.ops.moe import expert_stream_transport as es

    return es.cpu_kernel_options(variant="instr").cpu_kernel_isa() in ("bw", "vnni", "vbmi")


def _kernel():
    from sglang.srt.layers.moe.expert_engine.config import resolved_config

    if not resolved_config().stream.cpu_experts.enabled:
        pytest.skip("the kernel under test is the optimized one: set SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
    if len(CORES) < 4:
        pytest.skip("needs 4 cores in the affinity mask")
    return _trait(), CORES[:4]


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

    return es.kernel_layer(trait.layer_spec(_random_slabs(seed, HIDDEN, INTER), CAP), variant="instr")


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


def _facts(bits_w13, bits_w2, swizzled=0):
    from sglang.srt.layers.quantization.exl3.schemes.exl3_cpu_experts import Exl3LayerFacts

    return Exl3LayerFacts(
        hidden=HIDDEN, intermediate=INTER, bits_w13=bits_w13, bits_w2=bits_w2, act_limit=LIMIT, swizzled=swizzled
    )


# The kernel's parameter list (Exl3CpuKernel::params): bits_w13, bits_w2, swizzled, plan.
GENERIC, DSV41 = 0, 1  # entry.h's SglangExl3CpuPlan


def test_the_trait_lists_the_params_make_layer_reads():
    trait, _ = _kernel()
    assert trait._params(_facts(3, 3), DSV41) == (3, 3, 0, DSV41)
    assert trait._params(_facts(4, 2, swizzled=1), GENERIC) == (4, 2, 1, GENERIC)


def test_the_host_lists_the_kernels_plans_generic_first():
    """cpu_kernel_plans: the generic plan first, which any layer may take, then the DSV4.1 plan with exactly the
    facts its Shape fixes; on an AVX-512 build of the residual block-128 activations, it is available."""
    trait, _ = _kernel()
    plans = trait._plans()
    assert [(p.id, p.name, p.fixed) for p in plans] == [(GENERIC, "generic", False), (DSV41, "dsv41", True)]
    dsv41 = plans[1]
    assert (dsv41.hidden, dsv41.intermediate, dsv41.bits_w13, dsv41.bits_w2, dsv41.swizzled, dsv41.act_limit) == (
        5120, 2304, 3, 3, 0, 10.0
    )
    assert (dsv41.min_isa, plans[0].available) == ("bw", True)
    assert dsv41.available == _has_bw()


def _refused(trait, slabs, params):
    import dataclasses

    from sglang.kernels.ops.moe import expert_stream_transport as es

    spec = dataclasses.replace(trait.layer_spec(slabs, CAP), params=params)
    with pytest.raises(Exception) as refused:
        es.kernel_layer(spec, variant="prod")
    return str(refused.value)


@pytest.mark.parametrize(
    "bits_w13, bits_w2, swizzled, limit, plan, why",
    [
        (2, 2, 0, LIMIT, DSV41, "plan dsv41 does not accept the layer"),  # a K2 layer forced onto the K3 plan
        (3, 2, 0, LIMIT, DSV41, "plan dsv41 does not accept the layer"),  # w2 differs
        (3, 3, 1, LIMIT, DSV41, "plan dsv41 does not accept the layer"),  # swizzled
        (3, 3, 0, 7.0, DSV41, "plan dsv41 does not accept the layer"),  # another clamp
        (3, 3, 0, LIMIT, 5, "plan is not a compiled plan id"),
    ],
    ids=["k2", "w2-differs", "swizzled", "act-limit", "unknown-id"],
)
def test_registration_refuses_a_plan_that_does_not_accept_the_layer(bits_w13, bits_w2, swizzled, limit, plan, why):
    """A layer's params name its plan; make_layer refuses one whose facts the layer does not have, so a forward never
    runs a plan whose compiled shape, K or layout differs from the slabs'."""
    _kernel()
    if not _has_bw():
        pytest.skip("the DSV4.1 plan needs AVX-512BW")
    trait = _trait(act_limit=limit)
    slabs = _random_slabs(20261008, HIDDEN, INTER, bits_w13=bits_w13, bits_w2=bits_w2)
    assert why in _refused(trait, slabs, (bits_w13, bits_w2, swizzled, plan))


def test_two_core_groups_at_once_match_one_group_bit_for_bit(monkeypatch):
    """Review Focus 3. Mutants: one process-wide core list (group B's team pinned onto A's cores) -- red on the
    affinity test below; a shared static scratch, or a forward lock returning 3 -- red here."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
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

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
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

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = _layer(trait, 1)
    x, slots, weights = _inputs()[0]
    try:
        rc, out = _forward(trait, layer, x, slots, weights, cores[:2], 3)
        assert rc == 2 and out.isnan().all(), "more workers than the call's cores"
        assert _forward(trait, layer, x, slots, weights, (), 2)[0] == 0, "no cores: unpinned workers"
    finally:
        es.kernel_drop(layer, variant="instr")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_a_non_finite_weight_is_refused_in_the_finite_math_build(monkeypatch, bad):
    """The kernel is built -Ofast, where std::isfinite may fold to true; the call check tests the weight's exponent bits."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = _layer(trait, 1)
    x, slots, weights = _inputs()[0]
    try:
        rc, out = _forward(trait, layer, x, slots, [bad] + list(weights[1:]), cores[:2], 2)
        assert rc == 2 and out.isnan().all()
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
    """Review Focus 3: OMP_THREAD_LIMIT equal to the two core groups' workers (2 + 2) leaves neither team short: the
    teams are the engine's own threads (team.hpp), which no OpenMP limit counts. libgomp reads the limit at load, so
    this runs in a child."""
    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
    _kernel()  # skips here, not in the child
    env = dict(os.environ, OMP_THREAD_LIMIT="4", SGLANG_EXL3_MOE_CPU_PIN="0")
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

    monkeypatch.setenv("SGLANG_EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    layer = es.kernel_layer(trait.layer_spec(_random_slabs(20261005, HIDDEN, INTER), CAP), variant="prod")
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
