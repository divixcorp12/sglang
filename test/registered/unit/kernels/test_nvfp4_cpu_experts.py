"""The NVFP4 CPU expert kernel under its team (cpu_experts/team.hpp), through the expert-stream host's kernel test
exports (Linux, GCC).

Each case runs in its own process: a preloaded pin shim is process-wide. The child makes the sanitizer
harness's 80 x 80 layer (every weight nibble 1.0, every scale 1.0) on the NVFP4 host module and prints one line per call.
A tier is the host module's -march (SGLANG_JIT_HOST_MARCH), so each tier is its own module build.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)

CHILD = r"""
import struct, sys, threading
import numpy as np
import torch
from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

calls = [int(t) for t in sys.argv[1].split(",")]
cores = [int(c) for c in sys.argv[2].split(",")] if sys.argv[2] else []
alpha_value = float(sys.argv[3])
groups = int(sys.argv[4])

H = N = 80
CAP = 2
w13 = np.full(CAP * N * H, 0x22, np.uint8)
w2 = np.full(CAP * H * N // 2, 0x22, np.uint8)
sf13 = np.full(CAP * 256 * 8, 56, np.uint8)
sf2 = np.full(CAP * 128 * 8, 56, np.uint8)
alpha = np.full(CAP, alpha_value, np.float32)
slabs = [(w13, N * H), (w2, H * N // 2), (sf13, 256 * 8), (sf2, 128 * 8), (alpha, 4), (alpha, 4)]
one = struct.unpack("<I", struct.pack("<f", 1.0))[0]
spec = CpuExpertLayerSpec(capacity=CAP, hidden=H, intermediate=N, act_limit=0.0,
                          slabs=tuple((a.ctypes.data, s) for a, s in slabs) + ((0, 0),),
                          params=(0, one, one))
layer = es.kernel_layer(spec, layout=es.NVFP4_BUILD, variant="instr")
x = torch.full((1, H), 1.0, dtype=torch.float16)
slots = torch.zeros((1, 1), dtype=torch.int32)
weights = torch.ones((1, 1), dtype=torch.float32)


def forward(threads, on):
    out = torch.full((1, H), 123.0)
    status, _ = es.kernel_forward(layer, x, slots, weights, out, threads=threads, cores=on,
                                  layout=es.NVFP4_BUILD, variant="instr")
    return status, "untouched" if bool((out == 123.0).all()) else "written", out


if groups < 2:
    for threads in calls:
        status, state, _ = forward(threads, cores)
        print("forward", threads, status, state)
else:
    width = len(cores) // groups
    parts = [cores[i * width:(i + 1) * width] for i in range(groups)]
    results = {i: [] for i in range(groups)}

    def run(i):
        for _ in range(20):
            for threads in calls:
                results[i].append((threads,) + forward(threads, parts[i]))

    runners = [threading.Thread(target=run, args=(i,)) for i in range(groups)]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()
    for i in range(groups):
        for threads, status, state, _ in results[i]:
            print("forward", i, threads, status, state)
    first, second = results[0], results[1]
    print("same" if all(torch.equal(p[3], q[3]) for p, q in zip(first, second)) else "different")
"""

_WARM = (
    "from sglang.kernels.ops.moe import expert_stream_transport as es; "
    "es._host_module(es.NVFP4_BUILD, 'instr')"
)


def _host_has_avx2() -> bool:
    # What an x86-64-v3 build needs: the AVX2 tier of isa.hpp plus the rest of the v3 level.
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    flags = next((line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")), [])
    return {"avx2", "fma", "f16c", "bmi1", "bmi2", "abm", "movbe"} <= set(flags)


# A tier is a host module built at its -march; each case runs against the AVX2 module and the scalar one. The module
# is built here, once per tier, so the timed children find it cached.
@pytest.fixture(scope="module", params=["x86-64-v3", "x86-64"], ids=["avx2", "scalar"])
def tier_env(request):
    if request.param == "x86-64-v3" and not _host_has_avx2():
        pytest.skip("the host has no AVX2/FMA/F16C, so it cannot run an x86-64-v3 build")
    env = {**os.environ, "SGLANG_JIT_HOST_MARCH": request.param}
    subprocess.run([sys.executable, "-c", _WARM], check=True, env=env)
    return env


def _run(env, calls, cores=(), alpha=1.0, groups=1, **extra_env):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, calls, ",".join(map(str, cores)), str(alpha), str(groups)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**env, **extra_env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.split("\n")[:-1]


def _allowed(count):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < count:
        pytest.skip(f"needs {count} allowed CPUs")
    return allowed[:count]


def test_a_later_forward_may_use_more_workers_than_the_first(tier_env):
    assert _run(tier_env, "1,3") == ["forward 1 0 written", "forward 3 0 written"]


def test_more_workers_than_cores_is_refused_and_leaves_out_untouched(tier_env):
    assert _run(tier_env, "2", cores=_allowed(1)) == ["forward 2 2 untouched"]


def test_a_worker_that_cannot_be_pinned_fails_the_forward_and_leaves_out_untouched(tier_env, tmp_path):
    # The kernel cannot pin the cores it is given, so the forward fails. A preloaded
    # pthread_setaffinity_np that always fails stands in for that.
    shim = tmp_path / "unpinnable.c"
    shim.write_text("int pthread_setaffinity_np(unsigned long t, unsigned long n, const void* s) { return 22; }\n")
    so = tmp_path / "libunpinnable.so"
    subprocess.run([CXX, "-x", "c", "-shared", "-fPIC", str(shim), "-o", str(so)], check=True)
    lines = _run(tier_env, "2", cores=_allowed(2), LD_PRELOAD=str(so))
    assert lines[:1] == ["forward 2 1 untouched"]


def test_two_core_groups_forward_at_once(tier_env):
    """Part 3: a CPU expert thread per NUMA group, both forwarding at once; with per-call cores both run and agree
    byte for byte."""
    lines = _run(tier_env, "2,2,2", cores=_allowed(4), groups=2)
    forwards = [line.split() for line in lines if line.startswith("forward")]
    assert len(forwards) == 2 * 20 * 3 and all(f[3] == "0" and f[4] == "written" for f in forwards), lines
    assert lines[-1] == "same"


def test_an_intermediate_q8_cannot_represent_returns_2_and_leaves_out_untouched(tier_env):
    # gate = up = 80 * 1e30, so SiLU(gate) * up overflows to inf before the down projection's quantization.
    assert _run(tier_env, "2", alpha=1e30) == ["forward 2 2 untouched"]


def _slabs(torch):
    """The CHILD's layer: H = N = 80, two slots."""
    return {
        "w13": torch.full((2, 80 * 80), 0x22, dtype=torch.uint8),
        "w2": torch.full((2, 80 * 80 // 2), 0x22, dtype=torch.uint8),
        "sf13": torch.full((2, 256 * 8), 56, dtype=torch.uint8),
        "sf2": torch.full((2, 128 * 8), 56, dtype=torch.uint8),
        "gate_alpha": torch.ones(2, 1),
        "down_alpha": torch.ones(2, 1),
    }


def test_the_scheme_layer_runs_through_the_kernel():
    import torch

    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0)
    layer = es.kernel_layer(trait.layer_spec(_slabs(torch), capacity=2), layout=es.NVFP4_BUILD, variant="instr")
    saved = os.sched_getaffinity(0)
    try:
        x = torch.ones(1, 80, dtype=torch.float16)
        out = torch.full((1, 80), 123.0)
        status = es.kernel_forward(
            layer, x, torch.zeros(1, 1), torch.ones(1, 1), out, threads=1, layout=es.NVFP4_BUILD, variant="instr"
        )[0]
        assert status == 0
        assert torch.isfinite(out).all() and not (out == 123.0).any()
        refused = torch.full((1, 80), 123.0)
        status, why = es.kernel_forward(
            layer, x, torch.full((1, 1), 2), torch.ones(1, 1), refused, threads=1, layout=es.NVFP4_BUILD,
            variant="instr",
        )
        assert status == 2 and "nvfp4" in why and (refused == 123.0).all()  # slot 2 is past the capacity
    finally:
        es.kernel_drop(layer, layout=es.NVFP4_BUILD, variant="instr")
        os.sched_setaffinity(0, saved)


def test_the_kernel_refuses_params_of_the_wrong_size():
    """Review Focus 4: the kernel's params() checks the list's length and names the layout."""
    import dataclasses

    import torch

    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0)
    spec = dataclasses.replace(trait.layer_spec(_slabs(torch), capacity=2), params=(0, 0))
    with pytest.raises(Exception, match="NVFP4 CPU params are .*got 2 values"):
        es.kernel_layer(spec, layout=es.NVFP4_BUILD, variant="instr")


def test_the_scheme_describes_a_layer_for_make_layer():
    import torch

    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, w13_layout=2, inv_input_scale13=0.5)
    slabs = {
        "w13": torch.zeros((2, 80 * 80), dtype=torch.uint8),
        "w2": torch.zeros((2, 80 * 80 // 2), dtype=torch.uint8),
        "sf13": torch.zeros((2, 256 * 8), dtype=torch.uint8),
        "sf2": torch.zeros((2, 128 * 8), dtype=torch.uint8),
        "gate_alpha": torch.ones(2, 1),
        "down_alpha": torch.ones(2, 1),
    }
    spec = trait.layer_spec(slabs, capacity=2)
    assert (spec.capacity, spec.hidden, spec.intermediate, spec.act_limit) == (2, 80, 80, 0.0)
    names = ("w13", "w2", "sf13", "sf2", "gate_alpha", "down_alpha")
    assert spec.slabs == tuple((slabs[n].data_ptr(), slabs[n][0].numel() * slabs[n].element_size()) for n in names) + ((0, 0),)
    assert spec.params == (2, 0x3F000000, 0x3F800000)  # w13_layout, then 0.5 and 1.0 as binary32 bits
    assert all(any(k is slabs[n] for k in spec.keep) for n in names)
    with pytest.raises(ValueError, match="w2"):
        trait.layer_spec({**slabs, "w2": slabs["w2"][:1]}, capacity=2)
