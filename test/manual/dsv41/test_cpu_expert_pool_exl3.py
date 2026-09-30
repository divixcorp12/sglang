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


def _random_slabs(seed):
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    return {
        "w13_trellis": torch.randint(
            -32768,
            32767,
            (CAP, 2, H // 16, INTER // 16, 48),
            generator=g,
            dtype=torch.int16,
        ),
        "w13_suh": signs(CAP, 2, H),
        "w13_svh": signs(CAP, 2, INTER),
        "w2_trellis": torch.randint(
            -32768,
            32767,
            (CAP, INTER // 16, H // 16, 48),
            generator=g,
            dtype=torch.int16,
        ),
        "w2_suh": signs(CAP, INTER),
        "w2_svh": signs(CAP, H),
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


def test_pool_matches_direct_kernel_calls_bit_for_bit(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
    from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    ext = exl3_ext()
    slabs = _random_slabs(20260929)
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
            x = (torch.randn(1, H, generator=g) * scale).half()
            slots = torch.tensor([[4, 0, 5]])
            w = torch.tensor([[0.5, 0.3, 0.2]]).half()
            want = torch.zeros(1, H)
            ext.exl3_moe_cpu_forward(direct, x, slots, w, want, threads)
            assert want.abs().max() > 0
            got = torch.zeros(1, H)
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
        changed = torch.zeros(1, H)
        pool.compute(3, slots, w, x, changed)
        assert not torch.equal(changed, want)
        ext.exl3_moe_cpu_forward(direct, x, slots, w, want, threads)
        assert torch.equal(changed, want)
    finally:
        pool.close()
        ext.exl3_moe_cpu_free_layer(direct)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
