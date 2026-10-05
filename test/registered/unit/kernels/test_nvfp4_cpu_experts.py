"""The NVFP4 CPU expert C ABI under its OpenMP team (Linux, GCC with OpenMP).

Each case runs in its own process: the OpenMP environment is read when the team first forms. The child registers
the sanitizer harness's 80 x 80 layer (every weight nibble 1.0, every scale 1.0) and prints one line per call.
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
import ctypes, sys, threading
import numpy as np
from sglang.srt.layers.moe.cpu_experts.pool import (
    CPU_EXPERTS_FORWARD_ABI_VERSION, CPU_EXPERTS_LAYER_ABI_VERSION, CpuExpertsForwardCall, CpuExpertsLayer,
)


# SglangNvfp4CpuParams (csrc/nvfp4/optimized/cpu_experts_cabi.h); the scheme's Nvfp4CpuParams, kept local so each child
# process imports only the pool and not every quantization config.
class Params(ctypes.Structure):
    _fields_ = [
        ("w13_layout", ctypes.c_int32), ("inv_input_scale13", ctypes.c_float), ("inv_input_scale2", ctypes.c_float),
    ]

lib = ctypes.CDLL(sys.argv[1])
calls = [int(t) for t in sys.argv[2].split(",")]
cores = [int(c) for c in sys.argv[3].split(",")] if sys.argv[3] else []
alpha_value = float(sys.argv[4])


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
engine_count = int(sys.argv[5])
engines = []
if cores:
    lib.sglang_nvfp4_cpu_experts_engine_create.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
    ]
    lib.sglang_nvfp4_cpu_experts_engine_free.argtypes = [ctypes.c_int64]
    width = len(cores) // engine_count
    for i in range(engine_count):
        part = cores[i * width:(i + 1) * width]
        created = ctypes.c_int64(0)
        print("engine", lib.sglang_nvfp4_cpu_experts_engine_create(
            (ctypes.c_int32 * len(part))(*part), len(part), ctypes.byref(created)))
        engines.append(created.value)
x = np.full(H, 0x3C00, np.uint16)  # fp16 1.0
slots = np.zeros(1, np.int32)
weights = np.ones(1, np.float32)


def forward(threads, engine):
    out = np.full(H, 123.0, np.float32)
    call = CpuExpertsForwardCall(
        abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=handle.value, x=x.ctypes.data,
        slots=slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        weights=weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k=1, threads=threads, accumulate=0, engine=engine)
    status = lib.sglang_nvfp4_cpu_experts_forward(ctypes.byref(call))
    return status, "untouched" if (out == 123.0).all() else "written", out


if len(engines) < 2:
    engine = engines[0] if engines else 0
    for threads in calls:
        status, state, _ = forward(threads, engine)
        print("forward", threads, status, state)
    if engines:
        print("free", lib.sglang_nvfp4_cpu_experts_engine_free(engine))
        status, state, _ = forward(calls[0], engine)
        print("freed", status, state)
else:
    results = {engine: [] for engine in engines}

    def run(engine):
        for _ in range(20):
            for threads in calls:
                results[engine].append((threads,) + forward(threads, engine))

    runners = [threading.Thread(target=run, args=(engine,)) for engine in engines]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()
    for engine in engines:
        for threads, status, state, _ in results[engine]:
            print("forward", engine, threads, status, state)
    first, second = (results[engine] for engine in engines)
    print("same" if all(np.array_equal(p[3], q[3]) for p, q in zip(first, second)) else "different")
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "nvfp4_cpu_build", REPO / "python/sglang/srt/layers/quantization/nvfp4/build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build.build(tmp_path_factory.mktemp("nvfp4") / "libnvfp4.so", cxx=CXX)


def _host_has_avx2() -> bool:
    # What detect_isa requires for the AVX2 tier (expert_stream/host/cpu_experts/isa.hpp).
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    flags = next((line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")), [])
    return {"avx2", "fma", "f16c"} <= set(flags)


# One library holds every tier; each case runs at each, capped by NVFP4_CPU_MAX_ISA. A cap never raises the tier, so
# the avx2 case is skipped where it would silently run the scalar tier again.
@pytest.fixture(params=["avx2", "scalar"])
def library(built, request):
    if request.param == "avx2" and not _host_has_avx2():
        pytest.skip("the host has no AVX2/FMA/F16C, so the avx2 cap would run the scalar tier")
    return built, request.param


def _run(library, calls, cores=(), alpha=1.0, engines=1, **env):
    path, isa = library
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(path), calls, ",".join(map(str, cores)), str(alpha), str(engines)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "NVFP4_CPU_MAX_ISA": isa, **env},
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


def test_more_workers_than_the_engines_cores_is_refused_and_leaves_out_untouched(library):
    assert _run(library, "2", cores=_allowed(1)) == ["engine 0", "forward 2 2 untouched", "free 0", "freed 2 untouched"]


def test_a_freed_engine_is_refused(library):
    assert _run(library, "2", cores=_allowed(2)) == ["engine 0", "forward 2 0 written", "free 0", "freed 2 untouched"]


def test_a_worker_that_cannot_be_pinned_fails_the_forward_and_leaves_out_untouched(library, tmp_path):
    # engine_create checks only range and uniqueness; a core the kernel cannot pin fails the forward. A preloaded
    # pthread_setaffinity_np that always fails stands in for that.
    shim = tmp_path / "unpinnable.c"
    shim.write_text("int pthread_setaffinity_np(unsigned long t, unsigned long n, const void* s) { return 22; }\n")
    so = tmp_path / "libunpinnable.so"
    subprocess.run([CXX, "-x", "c", "-shared", "-fPIC", str(shim), "-o", str(so)], check=True)
    lines = _run(library, "2", cores=_allowed(2), LD_PRELOAD=str(so))
    assert lines[:2] == ["engine 0", "forward 2 1 untouched"]


def test_two_engines_forward_at_once_without_a_busy_status(library):
    """Part 3: a CPU expert engine per NUMA group, both forwarding at once. The process-wide forward lock returned 3
    to the second; with per-engine cores both run and agree byte for byte."""
    lines = _run(library, "2,2,2", cores=_allowed(4), engines=2)
    assert lines[:2] == ["engine 0", "engine 0"]
    forwards = [line.split() for line in lines if line.startswith("forward")]
    assert len(forwards) == 2 * 20 * 3 and all(f[3] == "0" and f[4] == "written" for f in forwards), lines
    assert lines[-1] == "same"


def test_an_intermediate_q8_cannot_represent_returns_2_and_leaves_out_untouched(library):
    # gate = up = 80 * 1e30, so SiLU(gate) * up overflows to inf before the down projection's quantization.
    assert _run(library, "2", alpha=1e30) == ["forward 2 2 untouched"]


def test_the_scheme_registers_a_layer_and_runs_it(built):
    import ctypes

    import torch

    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, library=ctypes.CDLL(str(built)))
    slabs = {  # the CHILD's layer: H = N = 80, two slots
        "w13": torch.full((2, 80 * 80), 0x22, dtype=torch.uint8),
        "w2": torch.full((2, 80 * 80 // 2), 0x22, dtype=torch.uint8),
        "sf13": torch.full((2, 256 * 8), 56, dtype=torch.uint8),
        "sf2": torch.full((2, 128 * 8), 56, dtype=torch.uint8),
        "gate_alpha": torch.ones(2, 1),
        "down_alpha": torch.ones(2, 1),
    }
    handle = trait.register_layer(slabs, capacity=2)
    try:
        x = torch.ones(1, 80, dtype=torch.float16)
        out = torch.full((1, 80), 123.0)
        trait.forward(handle, x, torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1), out, threads=1)
        assert torch.isfinite(out).all() and not (out == 123.0).any()

        refused = torch.full((1, 80), 123.0)
        with pytest.raises(RuntimeError, match="status 2"):  # slot 2 is past the capacity
            trait.forward(handle, x, torch.full((1, 1), 2, dtype=torch.int64), torch.ones(1, 1), refused, threads=1)
        assert (refused == 123.0).all()
        with pytest.raises(ValueError, match="NVFP4 CPU forward takes"):
            trait.forward(handle, x, torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1), out[:, :40], threads=1)
    finally:
        trait.free_layer(handle)
    with pytest.raises(RuntimeError, match="status 2"):
        trait.free_layer(handle)


def test_the_scheme_creates_and_frees_engines(built):
    """The pool's engine thread creates its engine through the trait (CpuExpertQuantTrait.native_create_engine): a
    handle, never 0, for distinct cores; a repeated core and a second free are refused, naming why. Nothing forwards
    here, so this thread stays unpinned."""
    import ctypes

    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, library=ctypes.CDLL(str(built)))
    cores = _allowed(2)
    engine = trait.native_create_engine(cores)
    assert engine != 0
    with pytest.raises(RuntimeError, match="refused engine cores"):
        trait.native_create_engine([cores[0], cores[0]])
    trait.native_free_engine(engine)
    with pytest.raises(RuntimeError, match="no engine"):
        trait.native_free_engine(engine)


def test_the_scheme_describes_a_layer_for_make_layer(built):
    import struct

    import torch
    from tvm_ffi import load_module

    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, w13_layout=2, inv_input_scale13=0.5,
                               module=load_module(str(built)))
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
    assert spec.params == struct.pack("<iff", 2, 0.5, 1.0)
    assert all(any(k is slabs[n] for k in spec.keep) for n in names)
    assert trait.kernel_address() != 0
    with pytest.raises(ValueError, match="w2"):
        trait.layer_spec({**slabs, "w2": slabs["w2"][:1]}, capacity=2)
