"""DeepSeek-V4.1 expert streaming has one set of tier budgets: the framework's."""

import pytest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_duplicate_dsv41_budget_knobs_are_gone():
    for name in (
        "SGLANG_DSV41_EXPERT_RAM_GIB",
        "SGLANG_DSV41_EXPERT_VRAM_GIB",
        "SGLANG_DSV41_EXPERT_SEED_PATH",
    ):
        assert not hasattr(envs, name), name


def test_streaming_knobs_that_remain():
    assert envs.SGLANG_DSV41_EXPERT_STREAM.get() is False
    assert envs.SGLANG_DSV41_EXPERT_DIR.get() == ""
    assert envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get() == ""
    assert envs.SGLANG_DSV41_ENGRAM_RAM_GIB.get() == 0.0
    # The framework budgets EXL3 streaming uses instead.
    assert envs.SGLANG_MOE_PINNED_HOST_MB.get() == 0
    assert envs.SGLANG_MOE_HOT_GPU_MB.get() == 0
    assert envs.SGLANG_MOE_HOT_SEED.get() == ""


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
