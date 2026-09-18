"""Top-20 logprob comparison metrics."""

import importlib.util
import os

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41", "compare_oracle.py")
_spec = importlib.util.spec_from_file_location("compare_oracle", _PATH)
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)


def test_identical():
    ids = np.array([[5, 3, 9], [1, 2, 3]])
    lp = np.array([[-0.1, -2.0, -3.0], [-0.5, -1.0, -4.0]])
    m = co.compare_topk(ids, lp, ids, lp)
    assert m == {"top1_agree": 1.0, "mean_overlap": 1.0, "mean_abs_dlp": 0.0, "max_abs_dlp": 0.0}


def test_disagreement_and_shift():
    ref_ids = np.array([[5, 3, 9]])
    ref_lp = np.array([[-0.1, -2.0, -3.0]])
    got_ids = np.array([[3, 5, 7]])
    got_lp = np.array([[-0.2, -0.3, -3.0]])
    m = co.compare_topk(ref_ids, ref_lp, got_ids, got_lp)
    assert m["top1_agree"] == 0.0
    assert m["mean_overlap"] == pytest.approx(2 / 3)
    assert m["max_abs_dlp"] == pytest.approx(1.8)
    assert m["mean_abs_dlp"] == pytest.approx((0.2 + 1.8) / 2)


def test_accept():
    assert co.accepts({"top1_agree": 0.99, "mean_abs_dlp": 0.01, "mean_overlap": 1, "max_abs_dlp": 0.3})
    assert not co.accepts({"top1_agree": 0.9, "mean_abs_dlp": 0.01, "mean_overlap": 1, "max_abs_dlp": 0.3})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
