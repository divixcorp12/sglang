"""Which breakable-graph break-points belong to the prefill graph (CPU).

The breakable decode graph captures DSpark's target verify, and ForwardMode.is_extend() counts TARGET_VERIFY as an
extend. A break-point that reads the prefill runner's piecewise forward context must not fire there: the decode runner
never sets that context (D2-3's graphed arm died capturing a verify, in deepseek_v4_engram_hash_ids)."""

import sys

import pytest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
    is_in_breakable_prefill_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "mode, expected",
    [
        (ForwardMode.EXTEND, True),
        (ForwardMode.MIXED, True),
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DECODE, False),
    ],
)
def test_inside_a_breakable_graph_only_a_non_verify_extend_is_the_prefill_graph(mode, expected):
    with enable_breakable_cuda_graph():
        assert is_in_breakable_prefill_graph(mode) is expected


def test_outside_a_breakable_graph_no_mode_is():
    assert not is_in_breakable_prefill_graph(ForwardMode.EXTEND)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
