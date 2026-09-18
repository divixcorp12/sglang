"""Routing-skew summaries used as the first G/f proxy."""

import importlib.util
import os

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41", "router_stats.py")
_spec = importlib.util.spec_from_file_location("router_stats", _PATH)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def test_uniform_router_has_zero_gini_and_proportional_hits():
    counts = np.full((2, 100), 7)
    s = rs.skew_summary(counts)
    assert s[0]["gini"] == pytest.approx(0.0, abs=1e-9)
    assert s[1]["top_frac_mass"]["10%"] == pytest.approx(0.10)
    assert rs.cache_hit_rate(counts, 0.25) == pytest.approx([0.25, 0.25])


def test_concentrated_router():
    counts = np.zeros((1, 10))
    counts[0, 3] = 90
    counts[0, 7] = 10
    s = rs.skew_summary(counts)[0]
    assert s["unused_experts"] == 8
    assert s["top_frac_mass"]["10%"] == pytest.approx(0.9)
    assert rs.cache_hit_rate(counts, 0.2) == pytest.approx([1.0])
    assert s["gini"] > 0.8


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
