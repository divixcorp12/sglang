"""The NVFP4 CPU expert C ABI under its OpenMP team (Linux, GCC with OpenMP).

Each case runs in its own process: core configuration freezes at a process's first forward, and the OpenMP
environment is read when the team first forms. The child registers the sanitizer harness's 80 x 80 layer (every
weight nibble 1.0, every scale 1.0) and prints one line per call.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)

CHILD = r"""
import ctypes, sys
import numpy as np
from sglang.srt.layers.moe.cpu_experts.pool import (
    CPU_EXPERTS_FORWARD_ABI_VERSION, CPU_EXPERTS_LAYER_ABI_VERSION, CpuExpertsForwardCall, CpuExpertsLayer,
)

lib = ctypes.CDLL(sys.argv[1])
calls = [int(t) for t in sys.argv[2].split(",")]
cores = [int(c) for c in sys.argv[3].split(",")] if sys.argv[3] else []
alpha_value = float(sys.argv[4])


# SglangNvfp4CpuParams (nvfp4_cpu/optimized/cpu_experts_cabi.h).
class Params(ctypes.Structure):
    _fields_ = [
        ("w13_layout", ctypes.c_int32), ("inv_input_scale13", ctypes.c_float), ("inv_input_scale2", ctypes.c_float),
    ]


H = N = 80
CAP = 2
w13 = np.full(CAP * N * H, 0x22, np.uint8)
w2 = np.full(CAP * H * N // 2, 0x22, np.uint8)
sf13 = np.full(CAP * 256 * 8, 56, np.uint8)
sf2 = np.full(CAP * 128 * 8, 56, np.uint8)
alpha = np.full(CAP, alpha_value, np.float32)
params = Params(w13_layout=0, inv_input_scale13=1.0, inv_input_scale2=1.0)
d = CpuExpertsLayer(
    abi_version=CPU_EXPERTS_LAYER_ABI_VERSION, capacity=CAP, hidden=H, intermediate=N, activation=0, act_limit=0.0,
    slab_count=7, params=ctypes.cast(ctypes.pointer(params), ctypes.c_void_p),
)
for i, (slab, stride) in enumerate([(w13, N * H), (w2, H * N // 2), (sf13, 256 * 8), (sf2, 128 * 8), (alpha, 4), (alpha, 4)]):
    d.slabs[i] = slab.ctypes.data
    d.slot_bytes[i] = stride
handle = ctypes.c_int64(-1)
assert lib.sglang_nvfp4_cpu_experts_register_layer(ctypes.byref(d), ctypes.byref(handle)) == 0
core_array = (ctypes.c_int32 * max(len(cores), 1))(*cores)
if cores:
    print("cores", lib.sglang_nvfp4_cpu_experts_set_cores(core_array, len(cores)))
x = np.full(H, 0x3C00, np.uint16)  # fp16 1.0
slots = np.zeros(1, np.int32)
weights = np.ones(1, np.float32)
for threads in calls:
    out = np.full(H, 123.0, np.float32)
    call = CpuExpertsForwardCall(
        abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=handle.value, x=x.ctypes.data,
        slots=slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        weights=weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k=1, threads=threads, accumulate=0)
    status = lib.sglang_nvfp4_cpu_experts_forward(ctypes.byref(call))
    print("forward", threads, status, "untouched" if (out == 123.0).all() else "written")
if cores:
    print("cores-after", lib.sglang_nvfp4_cpu_experts_set_cores(core_array, len(cores)))
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "nvfp4_cpu_build", REPO / "python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build.build(tmp_path_factory.mktemp("nvfp4") / "libnvfp4.so", cxx=CXX)


def _run(library, calls, cores=(), alpha=1.0, **env):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(library), calls, ",".join(map(str, cores)), str(alpha)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.split("\n")[:-1]


def _allowed(count):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < count:
        pytest.skip(f"needs {count} allowed CPUs")
    return allowed[:count]


def test_a_later_forward_may_use_more_workers_than_the_first(library):
    assert _run(library, "1,3") == ["forward 1 0 written", "forward 3 0 written"]


def test_a_team_smaller_than_requested_fails_the_forward_and_leaves_out_untouched(library):
    assert _run(library, "4", OMP_THREAD_LIMIT="2") == ["forward 4 1 untouched"]


def test_more_workers_than_configured_cores_fails_the_forward(library):
    # Only the first two lines: whether cores can still change after this refused forward differs between the old
    # pool (it refused before starting, so yes) and the OpenMP team (the cores froze at the attempt), and neither is
    # a contract.
    assert _run(library, "2", cores=_allowed(1))[:2] == ["cores 0", "forward 2 1 untouched"]


def test_cores_cannot_change_after_the_first_forward(library):
    assert _run(library, "2", cores=_allowed(2)) == ["cores 0", "forward 2 0 written", "cores-after 2"]


def test_a_worker_that_cannot_be_pinned_fails_the_forward_and_leaves_out_untouched(library, tmp_path):
    # set_cores checked the cores against the affinity mask; pinning can still fail later (a cgroup change). A preloaded
    # pthread_setaffinity_np that always fails stands in for that.
    shim = tmp_path / "unpinnable.c"
    shim.write_text("int pthread_setaffinity_np(unsigned long t, unsigned long n, const void* s) { return 22; }\n")
    so = tmp_path / "libunpinnable.so"
    subprocess.run([CXX, "-x", "c", "-shared", "-fPIC", str(shim), "-o", str(so)], check=True)
    lines = _run(library, "2", cores=_allowed(2), LD_PRELOAD=str(so))
    assert lines[:2] == ["cores 0", "forward 2 1 untouched"]


def test_an_intermediate_q8_cannot_represent_returns_2_and_leaves_out_untouched(library):
    # gate = up = 80 * 1e30, so SiLU(gate) * up overflows to inf before the down projection's quantization.
    assert _run(library, "2", alpha=1e30) == ["forward 2 2 untouched"]
