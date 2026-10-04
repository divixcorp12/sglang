"""The header-only CPU experts framework (cpu_experts_common) passes its native harness (Linux, GCC with OpenMP).

``cpu_experts_common_check.cpp`` drives a toy quant through ``ExpertForward`` and the C ABI macro and prints
``ok <check>`` per contract it holds.
"""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

HARNESS = Path(__file__).resolve().parent / "cpu_experts_common_check.cpp"
TOY_LIBRARY = Path(__file__).resolve().parent / "cpu_experts_common_toy_lib.cpp"
CXX = os.environ.get("CXX") or shutil.which("g++")
CXX_FLAGS = ["-std=c++20", "-O2", "-fopenmp", "-pthread"]
SANITIZERS = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g"]
CHECKS = (
    "register_then_forward_overwrites_and_accumulates",
    "abi_versions_are_checked",
    "slot_bytes_below_the_minimum_are_refused",
    "unknown_handle_is_refused",
    "routes_are_validated",
    "concurrent_forward_returns_3",
    "isa_cap_env_lowers_the_tier",
    "set_cores_after_the_first_forward_returns_2",
    "keep_warm_returns_when_the_word_moves",
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the framework needs Linux and a GCC with OpenMP"
)


def _links_sanitizers(tmp_path) -> bool:
    source = tmp_path / "probe.cpp"
    source.write_text("int main() { return 0; }\n")
    probe = subprocess.run([CXX, *SANITIZERS, str(source), "-o", str(tmp_path / "probe")], capture_output=True)
    return probe.returncode == 0


@pytest.mark.parametrize("flags", [[], SANITIZERS], ids=["plain", "asan-ubsan"])
def test_the_native_harness_passes(tmp_path, flags):
    if flags and not _links_sanitizers(tmp_path):
        pytest.skip(f"{CXX} cannot link ASan/UBSan (its sanitizer runtimes are not installed)")
    exe = tmp_path / "cpu_experts_common_check"
    subprocess.run([CXX, *CXX_FLAGS, *flags, str(HARNESS), "-o", str(exe)], check=True)
    # The harness sets TOY_CPU_MAX_ISA itself; a value inherited from the caller must not pre-empt it.
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOY_CPU_")}
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = set(result.stdout.split("\n"))
    missing = [name for name in CHECKS if f"ok {name}" not in passed]
    assert not missing, f"checks that did not report ok: {missing}\n{result.stdout}{result.stderr}"


class _Layer(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capacity", ctypes.c_int32),
        ("hidden", ctypes.c_int32),
        ("intermediate", ctypes.c_int32),
        ("activation", ctypes.c_int32),
        ("act_limit", ctypes.c_float),
        ("slab_count", ctypes.c_int32),
        ("slabs", ctypes.c_void_p * 8),
        ("slot_bytes", ctypes.c_uint64 * 8),
        ("params", ctypes.c_void_p),
    ]


class _Forward(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("rows", ctypes.c_int32),
        ("layer", ctypes.c_int64),
        ("x", ctypes.c_void_p),
        ("slots", ctypes.POINTER(ctypes.c_int32)),
        ("weights", ctypes.POINTER(ctypes.c_float)),
        ("out", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("accumulate", ctypes.c_int32),
    ]


def _build_toy_library(output: Path, *flags: str) -> Path:
    subprocess.run([CXX, *CXX_FLAGS, "-fPIC", "-shared", *flags, str(TOY_LIBRARY), "-o", str(output)], check=True)
    return output


def test_each_library_keeps_its_own_registry_and_cores(tmp_path):
    # Two quant libraries in one process must not share the framework's state: a forward in one may not freeze the
    # other's cores, and a handle registered in one is unknown to the other.
    a = ctypes.CDLL(str(_build_toy_library(tmp_path / "libtoy_a.so")))
    b = ctypes.CDLL(str(_build_toy_library(tmp_path / "libtoy_b.so")))
    hidden, capacity = 16, 2
    slab = (ctypes.c_float * (hidden * capacity))(*range(hidden * capacity))
    scale = ctypes.c_float(1.0)
    layer = _Layer(abi_version=1, capacity=capacity, hidden=hidden, intermediate=hidden, slab_count=1)
    layer.slabs[0] = ctypes.cast(slab, ctypes.c_void_p)
    layer.slot_bytes[0] = hidden * 4
    layer.params = ctypes.cast(ctypes.byref(scale), ctypes.c_void_p)
    handle = ctypes.c_int64(-1)
    assert a.sglang_toy_cpu_experts_register_layer(ctypes.byref(layer), ctypes.byref(handle)) == 0

    x = (ctypes.c_uint16 * hidden)()
    slots = (ctypes.c_int32 * 1)(1)
    weights = (ctypes.c_float * 1)(1.0)
    out = (ctypes.c_float * hidden)()
    call = _Forward(abi_version=1, rows=1, layer=handle.value, k=1, threads=1, accumulate=0)
    call.x = ctypes.cast(x, ctypes.c_void_p)
    call.slots, call.weights, call.out = slots, weights, out
    assert a.sglang_toy_cpu_experts_forward(ctypes.byref(call)) == 0
    assert list(out) == [float(hidden + h) for h in range(hidden)]

    core = (ctypes.c_int32 * 1)(min(os.sched_getaffinity(0)))
    assert a.sglang_toy_cpu_experts_set_cores(core, 1) == 2
    assert b.sglang_toy_cpu_experts_set_cores(core, 1) == 0
    assert b.sglang_toy_cpu_experts_forward(ctypes.byref(call)) == 2


def test_a_scalar_quant_library_uses_no_avx_registers(tmp_path):
    # A quant whose top tier is Scalar must compile no vector code from the framework, so a portable (baseline x86-64)
    # build stays free of AVX registers; the Avx2 build is the control that the scan finds them.
    portable = ["-march=x86-64", "-mtune=generic"]
    scalar = _build_toy_library(tmp_path / "libtoy_scalar.so", *portable, "-DTOY_TOP_ISA=Scalar")
    avx2 = _build_toy_library(tmp_path / "libtoy_avx2.so", *portable)

    def disassembly(library: Path) -> str:
        return subprocess.run(["objdump", "-d", str(library)], capture_output=True, text=True, check=True).stdout

    assert "%ymm" not in disassembly(scalar) and "%zmm" not in disassembly(scalar)
    assert "%ymm" in disassembly(avx2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
