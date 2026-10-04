"""CpuExpertPool with Exl3CpuQuantTrait runs the real CPU kernel bit-identically to direct calls over the same slabs (needs the ext build).

Run with EXL3_MOE_CPU_PIN=0 under taskset on at least 2 cores.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC"
)

CAP, H, INTER = 6, 512, 256
LIMIT = 10.0


def _random_slabs(seed, hidden=H, inter=INTER):
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    return {
        "w13_trellis": torch.randint(
            -32768,
            32767,
            (CAP, 2, hidden // 16, inter // 16, 48),
            generator=g,
            dtype=torch.int16,
        ),
        "w13_suh": signs(CAP, 2, hidden),
        "w13_svh": signs(CAP, 2, inter),
        "w2_trellis": torch.randint(
            -32768,
            32767,
            (CAP, inter // 16, hidden // 16, 48),
            generator=g,
            dtype=torch.int16,
        ),
        "w2_suh": signs(CAP, inter),
        "w2_svh": signs(CAP, hidden),
    }


def _direct_layer(ext, s):
    """p0_real.make_layer's call: one view per expert, gate = w13 part 0, up = part 1."""
    rows = range(CAP)
    return ext.exl3_moe_cpu_make_layer(
        [s["w13_trellis"][i, 0] for i in rows],
        [s["w13_suh"][i, 0] for i in rows],
        [s["w13_svh"][i, 0] for i in rows],
        [s["w13_trellis"][i, 1] for i in rows],
        [s["w13_suh"][i, 1] for i in rows],
        [s["w13_svh"][i, 1] for i in rows],
        [s["w2_trellis"][i] for i in rows],
        [s["w2_suh"][i] for i in rows],
        [s["w2_svh"][i] for i in rows],
        [],
        [],
        [],
        0,
        LIMIT,
        0,
    )


@pytest.mark.parametrize("hidden, inter", [(H, INTER), (5120, 2304)], ids=["generic", "dsv41"])
def test_pool_matches_direct_kernel_calls_bit_for_bit(monkeypatch, hidden, inter):
    """The pool's slab registration runs bit-identically to make_layer's per-expert registration over the same slabs,
    at a generic shape and at DeepSeek V4.1's, where both take the DSV4.1 plan on AVX-512BW."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
    from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the slab ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    ext = exl3_ext()
    slabs = _random_slabs(20260929, hidden, inter)
    threads = min(4, len(cores))
    direct = _direct_layer(ext, slabs)
    pool = CpuExpertPool(
        Exl3CpuQuantTrait(ext, act_limit=LIMIT),
        {3: slabs},
        cores=cores,
        threads=threads,
    )
    pool.bind_current_thread()
    g = torch.Generator().manual_seed(1)
    try:
        for scale in (1.0, 8.0):
            x = (torch.randn(1, hidden, generator=g) * scale).half()
            slots = torch.tensor([[4, 0, 5]])
            w = torch.tensor([[0.5, 0.3, 0.2]]).half()
            want = torch.zeros(1, hidden)
            ext.exl3_moe_cpu_forward(direct, x, slots, w, want, threads)
            assert want.abs().max() > 0
            got = torch.zeros(1, hidden)
            pool.compute(
                3,
                torch.cat([slots, torch.tensor([[-1]])], 1),
                torch.cat([w, torch.zeros(1, 1).half()], 1),
                x,
                got,
            )
            assert torch.equal(got, want)
        # Row contents change under the registered views: the tier's refill must reach the kernel with no re-register.
        slabs["w2_suh"][4].neg_()
        changed = torch.zeros(1, hidden)
        pool.compute(3, slots, w, x, changed)
        assert not torch.equal(changed, want)
        ext.exl3_moe_cpu_forward(direct, x, slots, w, want, threads)
        assert torch.equal(changed, want)
    finally:
        pool.close()
        ext.exl3_moe_cpu_free_layer(direct)



C_NAMES = tuple(
    f"sglang_exl3_cpu_experts_{name}" for name in ("register_layer", "free_layer", "forward", "keep_warm", "set_cores")
)


def _optimized_ext():
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the C ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    return exl3_ext()


def _c_forward(trait, handle, x, slots, weights, out, accumulate=0):
    """One-row call of the C ABI's forward; returns its status."""
    import ctypes

    from sglang.srt.layers.moe.cpu_experts.pool import (
        CPU_EXPERTS_FORWARD_ABI_VERSION,
        CpuExpertForward,
        CpuExpertsForwardCall,
    )

    s = torch.tensor(slots, dtype=torch.int32)
    w = torch.tensor(weights, dtype=torch.float32)
    call = CpuExpertsForwardCall(
        abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=handle, x=x.data_ptr(),
        slots=ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
        weights=ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), k=len(slots), threads=1, accumulate=accumulate,
    )
    return CpuExpertForward(trait.native_forward())(ctypes.byref(call))


def test_the_exl3_library_exports_the_five_c_names(monkeypatch):
    """Every CPU expert quant exports the same five C functions (expert_stream/host/cpu_experts/cabi.hpp)."""
    import ctypes

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    library = ctypes.CDLL(_optimized_ext().__file__)
    for name in C_NAMES:
        assert getattr(library, name), name


def test_the_c_abi_forward_overwrites_or_accumulates(monkeypatch):
    """The CpuExpertForward the service calls, over a layer registered through register_layer: accumulate=0
    overwrites whatever out held; accumulate=1 adds, so a record's CPU misses sent in two jobs sum to the two experts'
    outputs. Not bitwise: -Ofast may fuse the add."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait = Exl3CpuQuantTrait(_optimized_ext(), act_limit=LIMIT)
    layer = trait.register_layer(_random_slabs(20261001), CAP)
    x = (torch.randn(H, generator=torch.Generator().manual_seed(2))).half()

    def run(slots, weights, out, accumulate):
        assert _c_forward(trait, layer, x, slots, weights, out, accumulate) == 0

    try:
        first, second = torch.empty(H), torch.empty(H)
        run([0], [0.5], first, 0)
        run([4], [0.25], second, 0)
        assert first.abs().max() > 0 and second.abs().max() > 0
        overwritten = torch.full((H,), 1e6)
        run([0], [0.5], overwritten, 0)
        assert torch.equal(overwritten, first), "accumulate=0 kept what out held"
        summed = torch.full((H,), 1e6)
        run([0], [0.5], summed, 0)
        run([4], [0.25], summed, 1)
        torch.testing.assert_close(summed, first + second, rtol=1e-6, atol=1e-6)
    finally:
        trait.free_layer(layer)


def test_the_c_abi_refuses_a_slot_past_capacity(monkeypatch):
    """A routed slot at the layer's capacity is outside its slabs: the forward refuses the call (2) before writing,
    so out keeps what it held, for accumulate=0 as for 1."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait = Exl3CpuQuantTrait(_optimized_ext(), act_limit=LIMIT)
    layer = trait.register_layer(_random_slabs(20261003), CAP)
    x = (torch.randn(H, generator=torch.Generator().manual_seed(3))).half()
    try:
        for accumulate in (0, 1):
            out = torch.full((H,), 1e6)
            assert _c_forward(trait, layer, x, [0, CAP], [0.5, 0.5], out, accumulate) == 2
            assert torch.equal(out, torch.full((H,), 1e6)), "a refused forward wrote out"
        out = torch.full((H,), 1e6)
        assert _c_forward(trait, layer, x, [CAP - 1], [0.5], out) == 0, "the last slot is inside the layer"
    finally:
        trait.free_layer(layer)


_SET_CORES_CHILD = """
import ctypes, os, sys
import torch
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import test_cpu_expert_pool_exl3 as t
from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
ext = t._optimized_ext()
set_cores = ctypes.CDLL(ext.__file__).sglang_exl3_cpu_experts_set_cores
set_cores.argtypes, set_cores.restype = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32], ctypes.c_int
cores = (ctypes.c_int32 * 1)(sorted(os.sched_getaffinity(0))[0])
print("before", set_cores(cores, 1))
trait = Exl3CpuQuantTrait(ext, act_limit=t.LIMIT)
layer = trait.register_layer(t._random_slabs(20261004), t.CAP)
x = torch.randn(t.H, generator=torch.Generator().manual_seed(4)).half()
print("forward", t._c_forward(trait, layer, x, [1], [1.0], torch.empty(t.H)))
trait.free_layer(layer)
print("after", set_cores(cores, 1))
try:
    trait.native_set_cores([cores[0]])
except RuntimeError as error:
    print("trait", "status 2" in str(error))
"""


def test_set_cores_after_the_first_forward_returns_2(monkeypatch):
    """In a fresh process: set_cores is accepted (0) before the first forward, which freezes the worker cores; a later
    set_cores, even with the same valid core, is refused (2), and the trait raises naming the status."""
    import subprocess
    import sys

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    _optimized_ext()  # skips here, not in the child, when the optimized kernel is not selected
    result = subprocess.run(
        [sys.executable, "-c", _SET_CORES_CHILD, __file__], capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.split("\n") if line.split(" ")[0] in ("before", "forward", "after", "trait")]
    assert lines == ["before 0", "forward 0", "after 2", "trait True"], result.stdout + result.stderr


_PIN_FAILURE_CHILD = """
import ctypes, os, sys
import torch
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import test_cpu_expert_pool_exl3 as t
from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
ext = t._optimized_ext()
set_cores = ctypes.CDLL(ext.__file__).sglang_exl3_cpu_experts_set_cores
set_cores.argtypes, set_cores.restype = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32], ctypes.c_int
print("set", set_cores((ctypes.c_int32 * 1)(1023), 1))
trait = Exl3CpuQuantTrait(ext, act_limit=t.LIMIT)
layer = trait.register_layer(t._random_slabs(20261005), t.CAP)
x = torch.randn(t.H, generator=torch.Generator().manual_seed(5)).half()
out = torch.full((t.H,), 1e6)
print("forward", t._c_forward(trait, layer, x, [1], [1.0], out))
print("untouched", torch.equal(out, torch.full((t.H,), 1e6)))
"""


def test_a_forward_whose_worker_cannot_be_pinned_leaves_out_untouched(monkeypatch):
    """In a fresh process whose only worker core (1023) does not exist: set_cores accepts it, the first forward fails
    its pin (1) and names the reason on stderr, and an overwrite call (accumulate=0) has not zeroed out."""
    import subprocess
    import sys

    if os.cpu_count() > 1023:
        pytest.skip("core 1023 exists on this host")
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    _optimized_ext()
    result = subprocess.run(
        [sys.executable, "-c", _PIN_FAILURE_CHILD, __file__], capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.split("\n") if line.split(" ")[0] in ("set", "forward", "untouched")]
    assert lines == ["set 0", "forward 1", "untouched True"], result.stdout + result.stderr
    assert "cpu_experts: cannot pin" in result.stderr, result.stderr


def test_the_c_abi_keep_warm_runs_until_its_word_moves_or_its_deadline(monkeypatch):
    """The CpuExpertKeepWarm the idle CPU expert thread calls: it holds its workers until the word differs from
    `seen` (a submit bumped it) or the CLOCK_MONOTONIC deadline passes, and refuses bad arguments with 2."""
    import ctypes
    import threading
    import time

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the keep-warm ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    keep_warm = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64)(
        Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT).native_keep_warm()
    )
    word = torch.zeros(1, dtype=torch.int32)
    far = time.monotonic_ns() + 60_000_000_000
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("rc", keep_warm(2, word.data_ptr(), 0, far)))
    worker.start()
    time.sleep(0.05)
    assert worker.is_alive(), "keep-warm returned before its word moved or its deadline"
    word[0] = 1
    worker.join(2)
    assert not worker.is_alive() and result["rc"] == 0

    start = time.monotonic()
    assert keep_warm(2, word.data_ptr(), 0, far) == 0, "a word already past `seen`"
    assert keep_warm(2, word.data_ptr(), 1, time.monotonic_ns()) == 0, "a deadline already passed"
    assert time.monotonic() - start < 0.5
    assert keep_warm(0, word.data_ptr(), 1, far) == 2
    assert keep_warm(2, None, 1, far) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
