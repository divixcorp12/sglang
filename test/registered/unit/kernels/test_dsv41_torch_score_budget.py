"""DSV4.1 torch prefill indexer: the score byte budget only sets how many query rows one chunk scores."""

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    _torch_score_rows_per_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BUDGET = envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB


def test_default_is_the_builtin_one_gib():
    with BUDGET.override(0):
        assert _torch_score_rows_per_chunk(32, 30000) == (1 << 30) // (32 * 30000 * 2)
        # a context too wide for even one row still makes progress
        assert _torch_score_rows_per_chunk(32, 1 << 30) == 1


def test_the_env_overrides_the_builtin_cap():
    with BUDGET.override(256):
        assert _torch_score_rows_per_chunk(32, 30000) == (256 << 20) // (32 * 30000 * 2)
