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
    from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
    from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
    from sglang.srt.layers.quantization.exl3_ext import cpu_act_defines, exl3_ext, optimized_cpu

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



def test_the_c_abi_forward_overwrites_or_accumulates(monkeypatch):
    """The CpuExpertForward the service calls: accumulate=0 overwrites whatever out held; accumulate=1 adds, so a
    record's CPU misses sent in two jobs sum to the two experts' outputs. Not bitwise: -Ofast may fuse the add."""
    import ctypes

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
    from sglang.srt.layers.moe.cpu_experts.pool import (
        CPU_EXPERTS_FORWARD_ABI_VERSION,
        CpuExpertForward,
        CpuExpertsForwardCall,
    )
    from sglang.srt.layers.quantization.exl3_ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the slab ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    ext = exl3_ext()
    slabs = _random_slabs(20261001)
    direct = _direct_layer(ext, slabs)
    forward = CpuExpertForward(Exl3CpuQuantTrait(ext, act_limit=LIMIT).native_forward())
    x = (torch.randn(H, generator=torch.Generator().manual_seed(2))).half()

    def run(slots, weights, out, accumulate):
        s = torch.tensor(slots, dtype=torch.int32)
        w = torch.tensor(weights, dtype=torch.float32)
        call = CpuExpertsForwardCall(
            abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=direct, x=x.data_ptr(),
            slots=ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            weights=ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), k=len(slots), threads=1, accumulate=accumulate,
        )
        assert forward(ctypes.byref(call)) == 0

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
        ext.exl3_moe_cpu_free_layer(direct)


def test_the_c_abi_keep_warm_runs_until_its_word_moves_or_its_deadline(monkeypatch):
    """The CpuExpertKeepWarm the idle CPU expert thread calls: it holds its workers until the word differs from
    `seen` (a submit bumped it) or the CLOCK_MONOTONIC deadline passes, and refuses bad arguments with 2."""
    import ctypes
    import threading
    import time

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
    from sglang.srt.layers.quantization.exl3_ext import cpu_act_defines, exl3_ext, optimized_cpu

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
